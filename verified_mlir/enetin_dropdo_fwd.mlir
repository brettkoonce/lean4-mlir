module @m {
  func.func @enetin_dropdo_fwd(%x: tensor<64x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x1000xf32>, %bd: tensor<1000xf32>, %dp2: tensor<64xf32>, %dp4: tensor<64xf32>, %dp6: tensor<64xf32>, %dp7: tensor<64xf32>, %dp9: tensor<64xf32>, %dp10: tensor<64xf32>, %dp12: tensor<64xf32>, %dp13: tensor<64xf32>, %dp14: tensor<64xf32>, %do: tensor<64x1280xf32>) -> tensor<64x1000xf32> {
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
    %v0 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<64x32x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<64x32x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<802816.0> : tensor<64x32x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-5> : tensor<64x32x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<64x32x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<64x32x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<64x32x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<64x32x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<64x32x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<64x32x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<64x32x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<64x32x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<64x32x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v25 = stablehlo.logistic %v24 : tensor<64x401408xf32>
    %v26 = stablehlo.multiply %v24, %v25 : tensor<64x401408xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v28 = stablehlo.convolution(%v27, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<64x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<64x32x112x112xf32>
    %v29 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<64x32x112x112xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v33 = stablehlo.constant dense<0.0> : tensor<f32>
    %v34 = stablehlo.constant dense<802816.0> : tensor<64x32x112x112xf32>
    %v35 = stablehlo.constant dense<1.0e-5> : tensor<64x32x112x112xf32>
    %v36 = stablehlo.reduce(%v32 init: %v33) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v37 = stablehlo.broadcast_in_dim %v36, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v38 = stablehlo.divide %v37, %v34 : tensor<64x32x112x112xf32>
    %v39 = stablehlo.subtract %v32, %v38 : tensor<64x32x112x112xf32>
    %v40 = stablehlo.multiply %v39, %v39 : tensor<64x32x112x112xf32>
    %v41 = stablehlo.reduce(%v40 init: %v33) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v42 = stablehlo.broadcast_in_dim %v41, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v43 = stablehlo.divide %v42, %v34 : tensor<64x32x112x112xf32>
    %v44 = stablehlo.add %v43, %v35 : tensor<64x32x112x112xf32>
    %v45 = stablehlo.rsqrt %v44 : tensor<64x32x112x112xf32>
    %v46 = stablehlo.multiply %v39, %v45 : tensor<64x32x112x112xf32>
    %v47 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v48 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v49 = stablehlo.multiply %v46, %v47 : tensor<64x32x112x112xf32>
    %v50 = stablehlo.add %v49, %v48 : tensor<64x32x112x112xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v52 = stablehlo.logistic %v51 : tensor<64x401408xf32>
    %v53 = stablehlo.multiply %v51, %v52 : tensor<64x401408xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v55 = stablehlo.constant dense<0.0> : tensor<f32>
    %v56 = stablehlo.reduce(%v54 init: %v55) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v57 = stablehlo.constant dense<12544.0> : tensor<64x32xf32>
    %v58 = stablehlo.divide %v56, %v57 : tensor<64x32xf32>
    %v59 = stablehlo.dot_general %v58, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<32x8xf32>) -> tensor<64x8xf32>
    %v60 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<64x8xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<64x8xf32>
    %v62 = stablehlo.logistic %v61 : tensor<64x8xf32>
    %v63 = stablehlo.multiply %v61, %v62 : tensor<64x8xf32>
    %v64 = stablehlo.dot_general %v63, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x8xf32>, tensor<8x32xf32>) -> tensor<64x32xf32>
    %v65 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<64x32xf32>
    %v66 = stablehlo.add %v64, %v65 : tensor<64x32xf32>
    %v67 = stablehlo.reshape %v53 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v70 = stablehlo.constant dense<12544.0> : tensor<64x32xf32>
    %v71 = stablehlo.divide %v69, %v70 : tensor<64x32xf32>
    %v72 = stablehlo.dot_general %v71, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x32xf32>, tensor<32x8xf32>) -> tensor<64x8xf32>
    %v73 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<64x8xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<64x8xf32>
    %v75 = stablehlo.logistic %v74 : tensor<64x8xf32>
    %v76 = stablehlo.multiply %v74, %v75 : tensor<64x8xf32>
    %v77 = stablehlo.dot_general %v76, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x8xf32>, tensor<8x32xf32>) -> tensor<64x32xf32>
    %v78 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<64x32xf32>
    %v79 = stablehlo.add %v77, %v78 : tensor<64x32xf32>
    %v80 = stablehlo.logistic %v79 : tensor<64x32xf32>
    %v81 = stablehlo.broadcast_in_dim %v80, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x112x112xf32>
    %v82 = stablehlo.multiply %v67, %v81 : tensor<64x32x112x112xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v85 = stablehlo.convolution(%v84, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<64x16x112x112xf32>
    %v86 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v87 = stablehlo.add %v85, %v86 : tensor<64x16x112x112xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<64x16x112x112xf32>) -> tensor<64x200704xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<64x200704xf32>) -> tensor<64x16x112x112xf32>
    %v90 = stablehlo.constant dense<0.0> : tensor<f32>
    %v91 = stablehlo.constant dense<802816.0> : tensor<64x16x112x112xf32>
    %v92 = stablehlo.constant dense<1.0e-5> : tensor<64x16x112x112xf32>
    %v93 = stablehlo.reduce(%v89 init: %v90) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v94 = stablehlo.broadcast_in_dim %v93, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v95 = stablehlo.divide %v94, %v91 : tensor<64x16x112x112xf32>
    %v96 = stablehlo.subtract %v89, %v95 : tensor<64x16x112x112xf32>
    %v97 = stablehlo.multiply %v96, %v96 : tensor<64x16x112x112xf32>
    %v98 = stablehlo.reduce(%v97 init: %v90) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v99 = stablehlo.broadcast_in_dim %v98, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v100 = stablehlo.divide %v99, %v91 : tensor<64x16x112x112xf32>
    %v101 = stablehlo.add %v100, %v92 : tensor<64x16x112x112xf32>
    %v102 = stablehlo.rsqrt %v101 : tensor<64x16x112x112xf32>
    %v103 = stablehlo.multiply %v96, %v102 : tensor<64x16x112x112xf32>
    %v104 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v105 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v106 = stablehlo.multiply %v103, %v104 : tensor<64x16x112x112xf32>
    %v107 = stablehlo.add %v106, %v105 : tensor<64x16x112x112xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<64x16x112x112xf32>) -> tensor<64x200704xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<64x200704xf32>) -> tensor<64x16x112x112xf32>
    %v110 = stablehlo.convolution(%v109, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<64x96x112x112xf32>
    %v111 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v112 = stablehlo.add %v110, %v111 : tensor<64x96x112x112xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<64x96x112x112xf32>) -> tensor<64x1204224xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<64x1204224xf32>) -> tensor<64x96x112x112xf32>
    %v115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v116 = stablehlo.constant dense<802816.0> : tensor<64x96x112x112xf32>
    %v117 = stablehlo.constant dense<1.0e-5> : tensor<64x96x112x112xf32>
    %v118 = stablehlo.reduce(%v114 init: %v115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v119 = stablehlo.broadcast_in_dim %v118, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v120 = stablehlo.divide %v119, %v116 : tensor<64x96x112x112xf32>
    %v121 = stablehlo.subtract %v114, %v120 : tensor<64x96x112x112xf32>
    %v122 = stablehlo.multiply %v121, %v121 : tensor<64x96x112x112xf32>
    %v123 = stablehlo.reduce(%v122 init: %v115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v124 = stablehlo.broadcast_in_dim %v123, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v125 = stablehlo.divide %v124, %v116 : tensor<64x96x112x112xf32>
    %v126 = stablehlo.add %v125, %v117 : tensor<64x96x112x112xf32>
    %v127 = stablehlo.rsqrt %v126 : tensor<64x96x112x112xf32>
    %v128 = stablehlo.multiply %v121, %v127 : tensor<64x96x112x112xf32>
    %v129 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v130 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v131 = stablehlo.multiply %v128, %v129 : tensor<64x96x112x112xf32>
    %v132 = stablehlo.add %v131, %v130 : tensor<64x96x112x112xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<64x96x112x112xf32>) -> tensor<64x1204224xf32>
    %v134 = stablehlo.logistic %v133 : tensor<64x1204224xf32>
    %v135 = stablehlo.multiply %v133, %v134 : tensor<64x1204224xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<64x1204224xf32>) -> tensor<64x96x112x112xf32>
    %v137 = stablehlo.convolution(%v136, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<64x96x56x56xf32>
    %v138 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v139 = stablehlo.add %v137, %v138 : tensor<64x96x56x56xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v143 = stablehlo.constant dense<200704.0> : tensor<64x96x56x56xf32>
    %v144 = stablehlo.constant dense<1.0e-5> : tensor<64x96x56x56xf32>
    %v145 = stablehlo.reduce(%v141 init: %v142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v146 = stablehlo.broadcast_in_dim %v145, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v147 = stablehlo.divide %v146, %v143 : tensor<64x96x56x56xf32>
    %v148 = stablehlo.subtract %v141, %v147 : tensor<64x96x56x56xf32>
    %v149 = stablehlo.multiply %v148, %v148 : tensor<64x96x56x56xf32>
    %v150 = stablehlo.reduce(%v149 init: %v142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v152 = stablehlo.divide %v151, %v143 : tensor<64x96x56x56xf32>
    %v153 = stablehlo.add %v152, %v144 : tensor<64x96x56x56xf32>
    %v154 = stablehlo.rsqrt %v153 : tensor<64x96x56x56xf32>
    %v155 = stablehlo.multiply %v148, %v154 : tensor<64x96x56x56xf32>
    %v156 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v157 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v158 = stablehlo.multiply %v155, %v156 : tensor<64x96x56x56xf32>
    %v159 = stablehlo.add %v158, %v157 : tensor<64x96x56x56xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v161 = stablehlo.logistic %v160 : tensor<64x301056xf32>
    %v162 = stablehlo.multiply %v160, %v161 : tensor<64x301056xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v165 = stablehlo.reduce(%v163 init: %v164) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v166 = stablehlo.constant dense<3136.0> : tensor<64x96xf32>
    %v167 = stablehlo.divide %v165, %v166 : tensor<64x96xf32>
    %v168 = stablehlo.dot_general %v167, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x96xf32>, tensor<96x4xf32>) -> tensor<64x4xf32>
    %v169 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<64x4xf32>
    %v170 = stablehlo.add %v168, %v169 : tensor<64x4xf32>
    %v171 = stablehlo.logistic %v170 : tensor<64x4xf32>
    %v172 = stablehlo.multiply %v170, %v171 : tensor<64x4xf32>
    %v173 = stablehlo.dot_general %v172, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x4xf32>, tensor<4x96xf32>) -> tensor<64x96xf32>
    %v174 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<64x96xf32>
    %v175 = stablehlo.add %v173, %v174 : tensor<64x96xf32>
    %v176 = stablehlo.reshape %v162 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v178 = stablehlo.reduce(%v176 init: %v177) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v179 = stablehlo.constant dense<3136.0> : tensor<64x96xf32>
    %v180 = stablehlo.divide %v178, %v179 : tensor<64x96xf32>
    %v181 = stablehlo.dot_general %v180, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x96xf32>, tensor<96x4xf32>) -> tensor<64x4xf32>
    %v182 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<64x4xf32>
    %v183 = stablehlo.add %v181, %v182 : tensor<64x4xf32>
    %v184 = stablehlo.logistic %v183 : tensor<64x4xf32>
    %v185 = stablehlo.multiply %v183, %v184 : tensor<64x4xf32>
    %v186 = stablehlo.dot_general %v185, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x4xf32>, tensor<4x96xf32>) -> tensor<64x96xf32>
    %v187 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<64x96xf32>
    %v188 = stablehlo.add %v186, %v187 : tensor<64x96xf32>
    %v189 = stablehlo.logistic %v188 : tensor<64x96xf32>
    %v190 = stablehlo.broadcast_in_dim %v189, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x56x56xf32>
    %v191 = stablehlo.multiply %v176, %v190 : tensor<64x96x56x56xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v194 = stablehlo.convolution(%v193, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<64x24x56x56xf32>
    %v195 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v196 = stablehlo.add %v194, %v195 : tensor<64x24x56x56xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v200 = stablehlo.constant dense<200704.0> : tensor<64x24x56x56xf32>
    %v201 = stablehlo.constant dense<1.0e-5> : tensor<64x24x56x56xf32>
    %v202 = stablehlo.reduce(%v198 init: %v199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v203 = stablehlo.broadcast_in_dim %v202, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v204 = stablehlo.divide %v203, %v200 : tensor<64x24x56x56xf32>
    %v205 = stablehlo.subtract %v198, %v204 : tensor<64x24x56x56xf32>
    %v206 = stablehlo.multiply %v205, %v205 : tensor<64x24x56x56xf32>
    %v207 = stablehlo.reduce(%v206 init: %v199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v208 = stablehlo.broadcast_in_dim %v207, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v209 = stablehlo.divide %v208, %v200 : tensor<64x24x56x56xf32>
    %v210 = stablehlo.add %v209, %v201 : tensor<64x24x56x56xf32>
    %v211 = stablehlo.rsqrt %v210 : tensor<64x24x56x56xf32>
    %v212 = stablehlo.multiply %v205, %v211 : tensor<64x24x56x56xf32>
    %v213 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v214 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v215 = stablehlo.multiply %v212, %v213 : tensor<64x24x56x56xf32>
    %v216 = stablehlo.add %v215, %v214 : tensor<64x24x56x56xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v219 = stablehlo.convolution(%v218, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<64x144x56x56xf32>
    %v220 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v221 = stablehlo.add %v219, %v220 : tensor<64x144x56x56xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v225 = stablehlo.constant dense<200704.0> : tensor<64x144x56x56xf32>
    %v226 = stablehlo.constant dense<1.0e-5> : tensor<64x144x56x56xf32>
    %v227 = stablehlo.reduce(%v223 init: %v224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v228 = stablehlo.broadcast_in_dim %v227, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v229 = stablehlo.divide %v228, %v225 : tensor<64x144x56x56xf32>
    %v230 = stablehlo.subtract %v223, %v229 : tensor<64x144x56x56xf32>
    %v231 = stablehlo.multiply %v230, %v230 : tensor<64x144x56x56xf32>
    %v232 = stablehlo.reduce(%v231 init: %v224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v233 = stablehlo.broadcast_in_dim %v232, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v234 = stablehlo.divide %v233, %v225 : tensor<64x144x56x56xf32>
    %v235 = stablehlo.add %v234, %v226 : tensor<64x144x56x56xf32>
    %v236 = stablehlo.rsqrt %v235 : tensor<64x144x56x56xf32>
    %v237 = stablehlo.multiply %v230, %v236 : tensor<64x144x56x56xf32>
    %v238 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v239 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v240 = stablehlo.multiply %v237, %v238 : tensor<64x144x56x56xf32>
    %v241 = stablehlo.add %v240, %v239 : tensor<64x144x56x56xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v243 = stablehlo.logistic %v242 : tensor<64x451584xf32>
    %v244 = stablehlo.multiply %v242, %v243 : tensor<64x451584xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v246 = stablehlo.convolution(%v245, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<64x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<64x144x56x56xf32>
    %v247 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v248 = stablehlo.add %v246, %v247 : tensor<64x144x56x56xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v252 = stablehlo.constant dense<200704.0> : tensor<64x144x56x56xf32>
    %v253 = stablehlo.constant dense<1.0e-5> : tensor<64x144x56x56xf32>
    %v254 = stablehlo.reduce(%v250 init: %v251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v255 = stablehlo.broadcast_in_dim %v254, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v256 = stablehlo.divide %v255, %v252 : tensor<64x144x56x56xf32>
    %v257 = stablehlo.subtract %v250, %v256 : tensor<64x144x56x56xf32>
    %v258 = stablehlo.multiply %v257, %v257 : tensor<64x144x56x56xf32>
    %v259 = stablehlo.reduce(%v258 init: %v251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v260 = stablehlo.broadcast_in_dim %v259, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v261 = stablehlo.divide %v260, %v252 : tensor<64x144x56x56xf32>
    %v262 = stablehlo.add %v261, %v253 : tensor<64x144x56x56xf32>
    %v263 = stablehlo.rsqrt %v262 : tensor<64x144x56x56xf32>
    %v264 = stablehlo.multiply %v257, %v263 : tensor<64x144x56x56xf32>
    %v265 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v266 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v267 = stablehlo.multiply %v264, %v265 : tensor<64x144x56x56xf32>
    %v268 = stablehlo.add %v267, %v266 : tensor<64x144x56x56xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v270 = stablehlo.logistic %v269 : tensor<64x451584xf32>
    %v271 = stablehlo.multiply %v269, %v270 : tensor<64x451584xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v274 = stablehlo.reduce(%v272 init: %v273) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v275 = stablehlo.constant dense<3136.0> : tensor<64x144xf32>
    %v276 = stablehlo.divide %v274, %v275 : tensor<64x144xf32>
    %v277 = stablehlo.dot_general %v276, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x144xf32>, tensor<144x6xf32>) -> tensor<64x6xf32>
    %v278 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<64x6xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<64x6xf32>
    %v280 = stablehlo.logistic %v279 : tensor<64x6xf32>
    %v281 = stablehlo.multiply %v279, %v280 : tensor<64x6xf32>
    %v282 = stablehlo.dot_general %v281, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x6xf32>, tensor<6x144xf32>) -> tensor<64x144xf32>
    %v283 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<64x144xf32>
    %v284 = stablehlo.add %v282, %v283 : tensor<64x144xf32>
    %v285 = stablehlo.reshape %v271 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v287 = stablehlo.reduce(%v285 init: %v286) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v288 = stablehlo.constant dense<3136.0> : tensor<64x144xf32>
    %v289 = stablehlo.divide %v287, %v288 : tensor<64x144xf32>
    %v290 = stablehlo.dot_general %v289, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x144xf32>, tensor<144x6xf32>) -> tensor<64x6xf32>
    %v291 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<64x6xf32>
    %v292 = stablehlo.add %v290, %v291 : tensor<64x6xf32>
    %v293 = stablehlo.logistic %v292 : tensor<64x6xf32>
    %v294 = stablehlo.multiply %v292, %v293 : tensor<64x6xf32>
    %v295 = stablehlo.dot_general %v294, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x6xf32>, tensor<6x144xf32>) -> tensor<64x144xf32>
    %v296 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<64x144xf32>
    %v297 = stablehlo.add %v295, %v296 : tensor<64x144xf32>
    %v298 = stablehlo.logistic %v297 : tensor<64x144xf32>
    %v299 = stablehlo.broadcast_in_dim %v298, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x56x56xf32>
    %v300 = stablehlo.multiply %v285, %v299 : tensor<64x144x56x56xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v303 = stablehlo.convolution(%v302, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<64x24x56x56xf32>
    %v304 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v305 = stablehlo.add %v303, %v304 : tensor<64x24x56x56xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v309 = stablehlo.constant dense<200704.0> : tensor<64x24x56x56xf32>
    %v310 = stablehlo.constant dense<1.0e-5> : tensor<64x24x56x56xf32>
    %v311 = stablehlo.reduce(%v307 init: %v308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v312 = stablehlo.broadcast_in_dim %v311, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v313 = stablehlo.divide %v312, %v309 : tensor<64x24x56x56xf32>
    %v314 = stablehlo.subtract %v307, %v313 : tensor<64x24x56x56xf32>
    %v315 = stablehlo.multiply %v314, %v314 : tensor<64x24x56x56xf32>
    %v316 = stablehlo.reduce(%v315 init: %v308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v317 = stablehlo.broadcast_in_dim %v316, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v318 = stablehlo.divide %v317, %v309 : tensor<64x24x56x56xf32>
    %v319 = stablehlo.add %v318, %v310 : tensor<64x24x56x56xf32>
    %v320 = stablehlo.rsqrt %v319 : tensor<64x24x56x56xf32>
    %v321 = stablehlo.multiply %v314, %v320 : tensor<64x24x56x56xf32>
    %v322 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v323 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v324 = stablehlo.multiply %v321, %v322 : tensor<64x24x56x56xf32>
    %v325 = stablehlo.add %v324, %v323 : tensor<64x24x56x56xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v327 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<64xf32>) -> tensor<64x75264xf32>
    %v328 = stablehlo.multiply %v327, %v326 : tensor<64x75264xf32>
    %v329 = stablehlo.add %v328, %v217 : tensor<64x75264xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v331 = stablehlo.convolution(%v330, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<64x144x56x56xf32>
    %v332 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v333 = stablehlo.add %v331, %v332 : tensor<64x144x56x56xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v337 = stablehlo.constant dense<200704.0> : tensor<64x144x56x56xf32>
    %v338 = stablehlo.constant dense<1.0e-5> : tensor<64x144x56x56xf32>
    %v339 = stablehlo.reduce(%v335 init: %v336) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v340 = stablehlo.broadcast_in_dim %v339, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v341 = stablehlo.divide %v340, %v337 : tensor<64x144x56x56xf32>
    %v342 = stablehlo.subtract %v335, %v341 : tensor<64x144x56x56xf32>
    %v343 = stablehlo.multiply %v342, %v342 : tensor<64x144x56x56xf32>
    %v344 = stablehlo.reduce(%v343 init: %v336) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v345 = stablehlo.broadcast_in_dim %v344, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v346 = stablehlo.divide %v345, %v337 : tensor<64x144x56x56xf32>
    %v347 = stablehlo.add %v346, %v338 : tensor<64x144x56x56xf32>
    %v348 = stablehlo.rsqrt %v347 : tensor<64x144x56x56xf32>
    %v349 = stablehlo.multiply %v342, %v348 : tensor<64x144x56x56xf32>
    %v350 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v351 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v352 = stablehlo.multiply %v349, %v350 : tensor<64x144x56x56xf32>
    %v353 = stablehlo.add %v352, %v351 : tensor<64x144x56x56xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v355 = stablehlo.logistic %v354 : tensor<64x451584xf32>
    %v356 = stablehlo.multiply %v354, %v355 : tensor<64x451584xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v358 = stablehlo.convolution(%v357, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<64x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<64x144x28x28xf32>
    %v359 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v360 = stablehlo.add %v358, %v359 : tensor<64x144x28x28xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v364 = stablehlo.constant dense<50176.0> : tensor<64x144x28x28xf32>
    %v365 = stablehlo.constant dense<1.0e-5> : tensor<64x144x28x28xf32>
    %v366 = stablehlo.reduce(%v362 init: %v363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v367 = stablehlo.broadcast_in_dim %v366, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v368 = stablehlo.divide %v367, %v364 : tensor<64x144x28x28xf32>
    %v369 = stablehlo.subtract %v362, %v368 : tensor<64x144x28x28xf32>
    %v370 = stablehlo.multiply %v369, %v369 : tensor<64x144x28x28xf32>
    %v371 = stablehlo.reduce(%v370 init: %v363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v372 = stablehlo.broadcast_in_dim %v371, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v373 = stablehlo.divide %v372, %v364 : tensor<64x144x28x28xf32>
    %v374 = stablehlo.add %v373, %v365 : tensor<64x144x28x28xf32>
    %v375 = stablehlo.rsqrt %v374 : tensor<64x144x28x28xf32>
    %v376 = stablehlo.multiply %v369, %v375 : tensor<64x144x28x28xf32>
    %v377 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v379 = stablehlo.multiply %v376, %v377 : tensor<64x144x28x28xf32>
    %v380 = stablehlo.add %v379, %v378 : tensor<64x144x28x28xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v382 = stablehlo.logistic %v381 : tensor<64x112896xf32>
    %v383 = stablehlo.multiply %v381, %v382 : tensor<64x112896xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v386 = stablehlo.reduce(%v384 init: %v385) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v387 = stablehlo.constant dense<784.0> : tensor<64x144xf32>
    %v388 = stablehlo.divide %v386, %v387 : tensor<64x144xf32>
    %v389 = stablehlo.dot_general %v388, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x144xf32>, tensor<144x6xf32>) -> tensor<64x6xf32>
    %v390 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<64x6xf32>
    %v391 = stablehlo.add %v389, %v390 : tensor<64x6xf32>
    %v392 = stablehlo.logistic %v391 : tensor<64x6xf32>
    %v393 = stablehlo.multiply %v391, %v392 : tensor<64x6xf32>
    %v394 = stablehlo.dot_general %v393, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x6xf32>, tensor<6x144xf32>) -> tensor<64x144xf32>
    %v395 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<64x144xf32>
    %v396 = stablehlo.add %v394, %v395 : tensor<64x144xf32>
    %v397 = stablehlo.reshape %v383 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v399 = stablehlo.reduce(%v397 init: %v398) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v400 = stablehlo.constant dense<784.0> : tensor<64x144xf32>
    %v401 = stablehlo.divide %v399, %v400 : tensor<64x144xf32>
    %v402 = stablehlo.dot_general %v401, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x144xf32>, tensor<144x6xf32>) -> tensor<64x6xf32>
    %v403 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<64x6xf32>
    %v404 = stablehlo.add %v402, %v403 : tensor<64x6xf32>
    %v405 = stablehlo.logistic %v404 : tensor<64x6xf32>
    %v406 = stablehlo.multiply %v404, %v405 : tensor<64x6xf32>
    %v407 = stablehlo.dot_general %v406, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x6xf32>, tensor<6x144xf32>) -> tensor<64x144xf32>
    %v408 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<64x144xf32>
    %v409 = stablehlo.add %v407, %v408 : tensor<64x144xf32>
    %v410 = stablehlo.logistic %v409 : tensor<64x144xf32>
    %v411 = stablehlo.broadcast_in_dim %v410, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x28x28xf32>
    %v412 = stablehlo.multiply %v397, %v411 : tensor<64x144x28x28xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v415 = stablehlo.convolution(%v414, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<64x40x28x28xf32>
    %v416 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v417 = stablehlo.add %v415, %v416 : tensor<64x40x28x28xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v421 = stablehlo.constant dense<50176.0> : tensor<64x40x28x28xf32>
    %v422 = stablehlo.constant dense<1.0e-5> : tensor<64x40x28x28xf32>
    %v423 = stablehlo.reduce(%v419 init: %v420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v424 = stablehlo.broadcast_in_dim %v423, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v425 = stablehlo.divide %v424, %v421 : tensor<64x40x28x28xf32>
    %v426 = stablehlo.subtract %v419, %v425 : tensor<64x40x28x28xf32>
    %v427 = stablehlo.multiply %v426, %v426 : tensor<64x40x28x28xf32>
    %v428 = stablehlo.reduce(%v427 init: %v420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v429 = stablehlo.broadcast_in_dim %v428, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v430 = stablehlo.divide %v429, %v421 : tensor<64x40x28x28xf32>
    %v431 = stablehlo.add %v430, %v422 : tensor<64x40x28x28xf32>
    %v432 = stablehlo.rsqrt %v431 : tensor<64x40x28x28xf32>
    %v433 = stablehlo.multiply %v426, %v432 : tensor<64x40x28x28xf32>
    %v434 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v435 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v436 = stablehlo.multiply %v433, %v434 : tensor<64x40x28x28xf32>
    %v437 = stablehlo.add %v436, %v435 : tensor<64x40x28x28xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v439 = stablehlo.reshape %v438 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v440 = stablehlo.convolution(%v439, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<64x240x28x28xf32>
    %v441 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v442 = stablehlo.add %v440, %v441 : tensor<64x240x28x28xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v444 = stablehlo.reshape %v443 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v445 = stablehlo.constant dense<0.0> : tensor<f32>
    %v446 = stablehlo.constant dense<50176.0> : tensor<64x240x28x28xf32>
    %v447 = stablehlo.constant dense<1.0e-5> : tensor<64x240x28x28xf32>
    %v448 = stablehlo.reduce(%v444 init: %v445) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v449 = stablehlo.broadcast_in_dim %v448, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v450 = stablehlo.divide %v449, %v446 : tensor<64x240x28x28xf32>
    %v451 = stablehlo.subtract %v444, %v450 : tensor<64x240x28x28xf32>
    %v452 = stablehlo.multiply %v451, %v451 : tensor<64x240x28x28xf32>
    %v453 = stablehlo.reduce(%v452 init: %v445) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v454 = stablehlo.broadcast_in_dim %v453, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v455 = stablehlo.divide %v454, %v446 : tensor<64x240x28x28xf32>
    %v456 = stablehlo.add %v455, %v447 : tensor<64x240x28x28xf32>
    %v457 = stablehlo.rsqrt %v456 : tensor<64x240x28x28xf32>
    %v458 = stablehlo.multiply %v451, %v457 : tensor<64x240x28x28xf32>
    %v459 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v460 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v461 = stablehlo.multiply %v458, %v459 : tensor<64x240x28x28xf32>
    %v462 = stablehlo.add %v461, %v460 : tensor<64x240x28x28xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v464 = stablehlo.logistic %v463 : tensor<64x188160xf32>
    %v465 = stablehlo.multiply %v463, %v464 : tensor<64x188160xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v467 = stablehlo.convolution(%v466, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<64x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<64x240x28x28xf32>
    %v468 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v469 = stablehlo.add %v467, %v468 : tensor<64x240x28x28xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v472 = stablehlo.constant dense<0.0> : tensor<f32>
    %v473 = stablehlo.constant dense<50176.0> : tensor<64x240x28x28xf32>
    %v474 = stablehlo.constant dense<1.0e-5> : tensor<64x240x28x28xf32>
    %v475 = stablehlo.reduce(%v471 init: %v472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v476 = stablehlo.broadcast_in_dim %v475, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v477 = stablehlo.divide %v476, %v473 : tensor<64x240x28x28xf32>
    %v478 = stablehlo.subtract %v471, %v477 : tensor<64x240x28x28xf32>
    %v479 = stablehlo.multiply %v478, %v478 : tensor<64x240x28x28xf32>
    %v480 = stablehlo.reduce(%v479 init: %v472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v481 = stablehlo.broadcast_in_dim %v480, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v482 = stablehlo.divide %v481, %v473 : tensor<64x240x28x28xf32>
    %v483 = stablehlo.add %v482, %v474 : tensor<64x240x28x28xf32>
    %v484 = stablehlo.rsqrt %v483 : tensor<64x240x28x28xf32>
    %v485 = stablehlo.multiply %v478, %v484 : tensor<64x240x28x28xf32>
    %v486 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v487 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v488 = stablehlo.multiply %v485, %v486 : tensor<64x240x28x28xf32>
    %v489 = stablehlo.add %v488, %v487 : tensor<64x240x28x28xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v491 = stablehlo.logistic %v490 : tensor<64x188160xf32>
    %v492 = stablehlo.multiply %v490, %v491 : tensor<64x188160xf32>
    %v493 = stablehlo.reshape %v492 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v495 = stablehlo.reduce(%v493 init: %v494) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<64x240xf32>
    %v496 = stablehlo.constant dense<784.0> : tensor<64x240xf32>
    %v497 = stablehlo.divide %v495, %v496 : tensor<64x240xf32>
    %v498 = stablehlo.dot_general %v497, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x240xf32>, tensor<240x10xf32>) -> tensor<64x10xf32>
    %v499 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
    %v500 = stablehlo.add %v498, %v499 : tensor<64x10xf32>
    %v501 = stablehlo.logistic %v500 : tensor<64x10xf32>
    %v502 = stablehlo.multiply %v500, %v501 : tensor<64x10xf32>
    %v503 = stablehlo.dot_general %v502, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x10xf32>, tensor<10x240xf32>) -> tensor<64x240xf32>
    %v504 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<64x240xf32>
    %v505 = stablehlo.add %v503, %v504 : tensor<64x240xf32>
    %v506 = stablehlo.reshape %v492 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v508 = stablehlo.reduce(%v506 init: %v507) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<64x240xf32>
    %v509 = stablehlo.constant dense<784.0> : tensor<64x240xf32>
    %v510 = stablehlo.divide %v508, %v509 : tensor<64x240xf32>
    %v511 = stablehlo.dot_general %v510, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x240xf32>, tensor<240x10xf32>) -> tensor<64x10xf32>
    %v512 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
    %v513 = stablehlo.add %v511, %v512 : tensor<64x10xf32>
    %v514 = stablehlo.logistic %v513 : tensor<64x10xf32>
    %v515 = stablehlo.multiply %v513, %v514 : tensor<64x10xf32>
    %v516 = stablehlo.dot_general %v515, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x10xf32>, tensor<10x240xf32>) -> tensor<64x240xf32>
    %v517 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<64x240xf32>
    %v518 = stablehlo.add %v516, %v517 : tensor<64x240xf32>
    %v519 = stablehlo.logistic %v518 : tensor<64x240xf32>
    %v520 = stablehlo.broadcast_in_dim %v519, dims = [0, 1] : (tensor<64x240xf32>) -> tensor<64x240x28x28xf32>
    %v521 = stablehlo.multiply %v506, %v520 : tensor<64x240x28x28xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v524 = stablehlo.convolution(%v523, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<64x40x28x28xf32>
    %v525 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v526 = stablehlo.add %v524, %v525 : tensor<64x40x28x28xf32>
    %v527 = stablehlo.reshape %v526 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v528 = stablehlo.reshape %v527 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v530 = stablehlo.constant dense<50176.0> : tensor<64x40x28x28xf32>
    %v531 = stablehlo.constant dense<1.0e-5> : tensor<64x40x28x28xf32>
    %v532 = stablehlo.reduce(%v528 init: %v529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v533 = stablehlo.broadcast_in_dim %v532, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v534 = stablehlo.divide %v533, %v530 : tensor<64x40x28x28xf32>
    %v535 = stablehlo.subtract %v528, %v534 : tensor<64x40x28x28xf32>
    %v536 = stablehlo.multiply %v535, %v535 : tensor<64x40x28x28xf32>
    %v537 = stablehlo.reduce(%v536 init: %v529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v538 = stablehlo.broadcast_in_dim %v537, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v539 = stablehlo.divide %v538, %v530 : tensor<64x40x28x28xf32>
    %v540 = stablehlo.add %v539, %v531 : tensor<64x40x28x28xf32>
    %v541 = stablehlo.rsqrt %v540 : tensor<64x40x28x28xf32>
    %v542 = stablehlo.multiply %v535, %v541 : tensor<64x40x28x28xf32>
    %v543 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v544 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<64x40x28x28xf32>
    %v545 = stablehlo.multiply %v542, %v543 : tensor<64x40x28x28xf32>
    %v546 = stablehlo.add %v545, %v544 : tensor<64x40x28x28xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<64x40x28x28xf32>) -> tensor<64x31360xf32>
    %v548 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<64xf32>) -> tensor<64x31360xf32>
    %v549 = stablehlo.multiply %v548, %v547 : tensor<64x31360xf32>
    %v550 = stablehlo.add %v549, %v438 : tensor<64x31360xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<64x31360xf32>) -> tensor<64x40x28x28xf32>
    %v552 = stablehlo.convolution(%v551, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<64x240x28x28xf32>
    %v553 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v554 = stablehlo.add %v552, %v553 : tensor<64x240x28x28xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v557 = stablehlo.constant dense<0.0> : tensor<f32>
    %v558 = stablehlo.constant dense<50176.0> : tensor<64x240x28x28xf32>
    %v559 = stablehlo.constant dense<1.0e-5> : tensor<64x240x28x28xf32>
    %v560 = stablehlo.reduce(%v556 init: %v557) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v561 = stablehlo.broadcast_in_dim %v560, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v562 = stablehlo.divide %v561, %v558 : tensor<64x240x28x28xf32>
    %v563 = stablehlo.subtract %v556, %v562 : tensor<64x240x28x28xf32>
    %v564 = stablehlo.multiply %v563, %v563 : tensor<64x240x28x28xf32>
    %v565 = stablehlo.reduce(%v564 init: %v557) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v566 = stablehlo.broadcast_in_dim %v565, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v567 = stablehlo.divide %v566, %v558 : tensor<64x240x28x28xf32>
    %v568 = stablehlo.add %v567, %v559 : tensor<64x240x28x28xf32>
    %v569 = stablehlo.rsqrt %v568 : tensor<64x240x28x28xf32>
    %v570 = stablehlo.multiply %v563, %v569 : tensor<64x240x28x28xf32>
    %v571 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v572 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<64x240x28x28xf32>
    %v573 = stablehlo.multiply %v570, %v571 : tensor<64x240x28x28xf32>
    %v574 = stablehlo.add %v573, %v572 : tensor<64x240x28x28xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<64x240x28x28xf32>) -> tensor<64x188160xf32>
    %v576 = stablehlo.logistic %v575 : tensor<64x188160xf32>
    %v577 = stablehlo.multiply %v575, %v576 : tensor<64x188160xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<64x188160xf32>) -> tensor<64x240x28x28xf32>
    %v579 = stablehlo.convolution(%v578, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<64x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<64x240x14x14xf32>
    %v580 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v581 = stablehlo.add %v579, %v580 : tensor<64x240x14x14xf32>
    %v582 = stablehlo.reshape %v581 : (tensor<64x240x14x14xf32>) -> tensor<64x47040xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v585 = stablehlo.constant dense<12544.0> : tensor<64x240x14x14xf32>
    %v586 = stablehlo.constant dense<1.0e-5> : tensor<64x240x14x14xf32>
    %v587 = stablehlo.reduce(%v583 init: %v584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v588 = stablehlo.broadcast_in_dim %v587, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v589 = stablehlo.divide %v588, %v585 : tensor<64x240x14x14xf32>
    %v590 = stablehlo.subtract %v583, %v589 : tensor<64x240x14x14xf32>
    %v591 = stablehlo.multiply %v590, %v590 : tensor<64x240x14x14xf32>
    %v592 = stablehlo.reduce(%v591 init: %v584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v593 = stablehlo.broadcast_in_dim %v592, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v594 = stablehlo.divide %v593, %v585 : tensor<64x240x14x14xf32>
    %v595 = stablehlo.add %v594, %v586 : tensor<64x240x14x14xf32>
    %v596 = stablehlo.rsqrt %v595 : tensor<64x240x14x14xf32>
    %v597 = stablehlo.multiply %v590, %v596 : tensor<64x240x14x14xf32>
    %v598 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v599 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<64x240x14x14xf32>
    %v600 = stablehlo.multiply %v597, %v598 : tensor<64x240x14x14xf32>
    %v601 = stablehlo.add %v600, %v599 : tensor<64x240x14x14xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<64x240x14x14xf32>) -> tensor<64x47040xf32>
    %v603 = stablehlo.logistic %v602 : tensor<64x47040xf32>
    %v604 = stablehlo.multiply %v602, %v603 : tensor<64x47040xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v607 = stablehlo.reduce(%v605 init: %v606) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x240x14x14xf32>, tensor<f32>) -> tensor<64x240xf32>
    %v608 = stablehlo.constant dense<196.0> : tensor<64x240xf32>
    %v609 = stablehlo.divide %v607, %v608 : tensor<64x240xf32>
    %v610 = stablehlo.dot_general %v609, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x240xf32>, tensor<240x10xf32>) -> tensor<64x10xf32>
    %v611 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
    %v612 = stablehlo.add %v610, %v611 : tensor<64x10xf32>
    %v613 = stablehlo.logistic %v612 : tensor<64x10xf32>
    %v614 = stablehlo.multiply %v612, %v613 : tensor<64x10xf32>
    %v615 = stablehlo.dot_general %v614, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x10xf32>, tensor<10x240xf32>) -> tensor<64x240xf32>
    %v616 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<64x240xf32>
    %v617 = stablehlo.add %v615, %v616 : tensor<64x240xf32>
    %v618 = stablehlo.reshape %v604 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v619 = stablehlo.constant dense<0.0> : tensor<f32>
    %v620 = stablehlo.reduce(%v618 init: %v619) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x240x14x14xf32>, tensor<f32>) -> tensor<64x240xf32>
    %v621 = stablehlo.constant dense<196.0> : tensor<64x240xf32>
    %v622 = stablehlo.divide %v620, %v621 : tensor<64x240xf32>
    %v623 = stablehlo.dot_general %v622, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x240xf32>, tensor<240x10xf32>) -> tensor<64x10xf32>
    %v624 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
    %v625 = stablehlo.add %v623, %v624 : tensor<64x10xf32>
    %v626 = stablehlo.logistic %v625 : tensor<64x10xf32>
    %v627 = stablehlo.multiply %v625, %v626 : tensor<64x10xf32>
    %v628 = stablehlo.dot_general %v627, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x10xf32>, tensor<10x240xf32>) -> tensor<64x240xf32>
    %v629 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<64x240xf32>
    %v630 = stablehlo.add %v628, %v629 : tensor<64x240xf32>
    %v631 = stablehlo.logistic %v630 : tensor<64x240xf32>
    %v632 = stablehlo.broadcast_in_dim %v631, dims = [0, 1] : (tensor<64x240xf32>) -> tensor<64x240x14x14xf32>
    %v633 = stablehlo.multiply %v618, %v632 : tensor<64x240x14x14xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<64x240x14x14xf32>) -> tensor<64x47040xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<64x47040xf32>) -> tensor<64x240x14x14xf32>
    %v636 = stablehlo.convolution(%v635, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<64x80x14x14xf32>
    %v637 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v638 = stablehlo.add %v636, %v637 : tensor<64x80x14x14xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v642 = stablehlo.constant dense<12544.0> : tensor<64x80x14x14xf32>
    %v643 = stablehlo.constant dense<1.0e-5> : tensor<64x80x14x14xf32>
    %v644 = stablehlo.reduce(%v640 init: %v641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v645 = stablehlo.broadcast_in_dim %v644, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v646 = stablehlo.divide %v645, %v642 : tensor<64x80x14x14xf32>
    %v647 = stablehlo.subtract %v640, %v646 : tensor<64x80x14x14xf32>
    %v648 = stablehlo.multiply %v647, %v647 : tensor<64x80x14x14xf32>
    %v649 = stablehlo.reduce(%v648 init: %v641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v650 = stablehlo.broadcast_in_dim %v649, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v651 = stablehlo.divide %v650, %v642 : tensor<64x80x14x14xf32>
    %v652 = stablehlo.add %v651, %v643 : tensor<64x80x14x14xf32>
    %v653 = stablehlo.rsqrt %v652 : tensor<64x80x14x14xf32>
    %v654 = stablehlo.multiply %v647, %v653 : tensor<64x80x14x14xf32>
    %v655 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v656 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v657 = stablehlo.multiply %v654, %v655 : tensor<64x80x14x14xf32>
    %v658 = stablehlo.add %v657, %v656 : tensor<64x80x14x14xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v661 = stablehlo.convolution(%v660, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<64x480x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v663 = stablehlo.add %v661, %v662 : tensor<64x480x14x14xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v667 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v668 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v669 = stablehlo.reduce(%v665 init: %v666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v670 = stablehlo.broadcast_in_dim %v669, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v671 = stablehlo.divide %v670, %v667 : tensor<64x480x14x14xf32>
    %v672 = stablehlo.subtract %v665, %v671 : tensor<64x480x14x14xf32>
    %v673 = stablehlo.multiply %v672, %v672 : tensor<64x480x14x14xf32>
    %v674 = stablehlo.reduce(%v673 init: %v666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v675 = stablehlo.broadcast_in_dim %v674, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v676 = stablehlo.divide %v675, %v667 : tensor<64x480x14x14xf32>
    %v677 = stablehlo.add %v676, %v668 : tensor<64x480x14x14xf32>
    %v678 = stablehlo.rsqrt %v677 : tensor<64x480x14x14xf32>
    %v679 = stablehlo.multiply %v672, %v678 : tensor<64x480x14x14xf32>
    %v680 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v681 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v682 = stablehlo.multiply %v679, %v680 : tensor<64x480x14x14xf32>
    %v683 = stablehlo.add %v682, %v681 : tensor<64x480x14x14xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v685 = stablehlo.logistic %v684 : tensor<64x94080xf32>
    %v686 = stablehlo.multiply %v684, %v685 : tensor<64x94080xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v688 = stablehlo.convolution(%v687, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<64x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<64x480x14x14xf32>
    %v689 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v690 = stablehlo.add %v688, %v689 : tensor<64x480x14x14xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v694 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v695 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v696 = stablehlo.reduce(%v692 init: %v693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v697 = stablehlo.broadcast_in_dim %v696, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v698 = stablehlo.divide %v697, %v694 : tensor<64x480x14x14xf32>
    %v699 = stablehlo.subtract %v692, %v698 : tensor<64x480x14x14xf32>
    %v700 = stablehlo.multiply %v699, %v699 : tensor<64x480x14x14xf32>
    %v701 = stablehlo.reduce(%v700 init: %v693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v702 = stablehlo.broadcast_in_dim %v701, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v703 = stablehlo.divide %v702, %v694 : tensor<64x480x14x14xf32>
    %v704 = stablehlo.add %v703, %v695 : tensor<64x480x14x14xf32>
    %v705 = stablehlo.rsqrt %v704 : tensor<64x480x14x14xf32>
    %v706 = stablehlo.multiply %v699, %v705 : tensor<64x480x14x14xf32>
    %v707 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v708 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v709 = stablehlo.multiply %v706, %v707 : tensor<64x480x14x14xf32>
    %v710 = stablehlo.add %v709, %v708 : tensor<64x480x14x14xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v712 = stablehlo.logistic %v711 : tensor<64x94080xf32>
    %v713 = stablehlo.multiply %v711, %v712 : tensor<64x94080xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v716 = stablehlo.reduce(%v714 init: %v715) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v717 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v718 = stablehlo.divide %v716, %v717 : tensor<64x480xf32>
    %v719 = stablehlo.dot_general %v718, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v720 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v721 = stablehlo.add %v719, %v720 : tensor<64x20xf32>
    %v722 = stablehlo.logistic %v721 : tensor<64x20xf32>
    %v723 = stablehlo.multiply %v721, %v722 : tensor<64x20xf32>
    %v724 = stablehlo.dot_general %v723, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v725 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v726 = stablehlo.add %v724, %v725 : tensor<64x480xf32>
    %v727 = stablehlo.reshape %v713 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v729 = stablehlo.reduce(%v727 init: %v728) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v730 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v731 = stablehlo.divide %v729, %v730 : tensor<64x480xf32>
    %v732 = stablehlo.dot_general %v731, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v733 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v734 = stablehlo.add %v732, %v733 : tensor<64x20xf32>
    %v735 = stablehlo.logistic %v734 : tensor<64x20xf32>
    %v736 = stablehlo.multiply %v734, %v735 : tensor<64x20xf32>
    %v737 = stablehlo.dot_general %v736, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v738 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v739 = stablehlo.add %v737, %v738 : tensor<64x480xf32>
    %v740 = stablehlo.logistic %v739 : tensor<64x480xf32>
    %v741 = stablehlo.broadcast_in_dim %v740, dims = [0, 1] : (tensor<64x480xf32>) -> tensor<64x480x14x14xf32>
    %v742 = stablehlo.multiply %v727, %v741 : tensor<64x480x14x14xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v745 = stablehlo.convolution(%v744, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<64x80x14x14xf32>
    %v746 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v747 = stablehlo.add %v745, %v746 : tensor<64x80x14x14xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v751 = stablehlo.constant dense<12544.0> : tensor<64x80x14x14xf32>
    %v752 = stablehlo.constant dense<1.0e-5> : tensor<64x80x14x14xf32>
    %v753 = stablehlo.reduce(%v749 init: %v750) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v754 = stablehlo.broadcast_in_dim %v753, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v755 = stablehlo.divide %v754, %v751 : tensor<64x80x14x14xf32>
    %v756 = stablehlo.subtract %v749, %v755 : tensor<64x80x14x14xf32>
    %v757 = stablehlo.multiply %v756, %v756 : tensor<64x80x14x14xf32>
    %v758 = stablehlo.reduce(%v757 init: %v750) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v759 = stablehlo.broadcast_in_dim %v758, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v760 = stablehlo.divide %v759, %v751 : tensor<64x80x14x14xf32>
    %v761 = stablehlo.add %v760, %v752 : tensor<64x80x14x14xf32>
    %v762 = stablehlo.rsqrt %v761 : tensor<64x80x14x14xf32>
    %v763 = stablehlo.multiply %v756, %v762 : tensor<64x80x14x14xf32>
    %v764 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v765 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v766 = stablehlo.multiply %v763, %v764 : tensor<64x80x14x14xf32>
    %v767 = stablehlo.add %v766, %v765 : tensor<64x80x14x14xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v769 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<64xf32>) -> tensor<64x15680xf32>
    %v770 = stablehlo.multiply %v769, %v768 : tensor<64x15680xf32>
    %v771 = stablehlo.add %v770, %v659 : tensor<64x15680xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v773 = stablehlo.convolution(%v772, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<64x480x14x14xf32>
    %v774 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v775 = stablehlo.add %v773, %v774 : tensor<64x480x14x14xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v779 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v780 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v781 = stablehlo.reduce(%v777 init: %v778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v782 = stablehlo.broadcast_in_dim %v781, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v783 = stablehlo.divide %v782, %v779 : tensor<64x480x14x14xf32>
    %v784 = stablehlo.subtract %v777, %v783 : tensor<64x480x14x14xf32>
    %v785 = stablehlo.multiply %v784, %v784 : tensor<64x480x14x14xf32>
    %v786 = stablehlo.reduce(%v785 init: %v778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v787 = stablehlo.broadcast_in_dim %v786, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v788 = stablehlo.divide %v787, %v779 : tensor<64x480x14x14xf32>
    %v789 = stablehlo.add %v788, %v780 : tensor<64x480x14x14xf32>
    %v790 = stablehlo.rsqrt %v789 : tensor<64x480x14x14xf32>
    %v791 = stablehlo.multiply %v784, %v790 : tensor<64x480x14x14xf32>
    %v792 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v793 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v794 = stablehlo.multiply %v791, %v792 : tensor<64x480x14x14xf32>
    %v795 = stablehlo.add %v794, %v793 : tensor<64x480x14x14xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v797 = stablehlo.logistic %v796 : tensor<64x94080xf32>
    %v798 = stablehlo.multiply %v796, %v797 : tensor<64x94080xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v800 = stablehlo.convolution(%v799, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<64x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<64x480x14x14xf32>
    %v801 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v802 = stablehlo.add %v800, %v801 : tensor<64x480x14x14xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v806 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v807 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v808 = stablehlo.reduce(%v804 init: %v805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v809 = stablehlo.broadcast_in_dim %v808, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v810 = stablehlo.divide %v809, %v806 : tensor<64x480x14x14xf32>
    %v811 = stablehlo.subtract %v804, %v810 : tensor<64x480x14x14xf32>
    %v812 = stablehlo.multiply %v811, %v811 : tensor<64x480x14x14xf32>
    %v813 = stablehlo.reduce(%v812 init: %v805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v814 = stablehlo.broadcast_in_dim %v813, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v815 = stablehlo.divide %v814, %v806 : tensor<64x480x14x14xf32>
    %v816 = stablehlo.add %v815, %v807 : tensor<64x480x14x14xf32>
    %v817 = stablehlo.rsqrt %v816 : tensor<64x480x14x14xf32>
    %v818 = stablehlo.multiply %v811, %v817 : tensor<64x480x14x14xf32>
    %v819 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v820 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v821 = stablehlo.multiply %v818, %v819 : tensor<64x480x14x14xf32>
    %v822 = stablehlo.add %v821, %v820 : tensor<64x480x14x14xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v824 = stablehlo.logistic %v823 : tensor<64x94080xf32>
    %v825 = stablehlo.multiply %v823, %v824 : tensor<64x94080xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v827 = stablehlo.constant dense<0.0> : tensor<f32>
    %v828 = stablehlo.reduce(%v826 init: %v827) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v829 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v830 = stablehlo.divide %v828, %v829 : tensor<64x480xf32>
    %v831 = stablehlo.dot_general %v830, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v832 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v833 = stablehlo.add %v831, %v832 : tensor<64x20xf32>
    %v834 = stablehlo.logistic %v833 : tensor<64x20xf32>
    %v835 = stablehlo.multiply %v833, %v834 : tensor<64x20xf32>
    %v836 = stablehlo.dot_general %v835, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v837 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v838 = stablehlo.add %v836, %v837 : tensor<64x480xf32>
    %v839 = stablehlo.reshape %v825 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v841 = stablehlo.reduce(%v839 init: %v840) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v842 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v843 = stablehlo.divide %v841, %v842 : tensor<64x480xf32>
    %v844 = stablehlo.dot_general %v843, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v845 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v846 = stablehlo.add %v844, %v845 : tensor<64x20xf32>
    %v847 = stablehlo.logistic %v846 : tensor<64x20xf32>
    %v848 = stablehlo.multiply %v846, %v847 : tensor<64x20xf32>
    %v849 = stablehlo.dot_general %v848, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v850 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v851 = stablehlo.add %v849, %v850 : tensor<64x480xf32>
    %v852 = stablehlo.logistic %v851 : tensor<64x480xf32>
    %v853 = stablehlo.broadcast_in_dim %v852, dims = [0, 1] : (tensor<64x480xf32>) -> tensor<64x480x14x14xf32>
    %v854 = stablehlo.multiply %v839, %v853 : tensor<64x480x14x14xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v857 = stablehlo.convolution(%v856, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<64x80x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v859 = stablehlo.add %v857, %v858 : tensor<64x80x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v863 = stablehlo.constant dense<12544.0> : tensor<64x80x14x14xf32>
    %v864 = stablehlo.constant dense<1.0e-5> : tensor<64x80x14x14xf32>
    %v865 = stablehlo.reduce(%v861 init: %v862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v866 = stablehlo.broadcast_in_dim %v865, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v867 = stablehlo.divide %v866, %v863 : tensor<64x80x14x14xf32>
    %v868 = stablehlo.subtract %v861, %v867 : tensor<64x80x14x14xf32>
    %v869 = stablehlo.multiply %v868, %v868 : tensor<64x80x14x14xf32>
    %v870 = stablehlo.reduce(%v869 init: %v862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v871 = stablehlo.broadcast_in_dim %v870, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v872 = stablehlo.divide %v871, %v863 : tensor<64x80x14x14xf32>
    %v873 = stablehlo.add %v872, %v864 : tensor<64x80x14x14xf32>
    %v874 = stablehlo.rsqrt %v873 : tensor<64x80x14x14xf32>
    %v875 = stablehlo.multiply %v868, %v874 : tensor<64x80x14x14xf32>
    %v876 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v877 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v878 = stablehlo.multiply %v875, %v876 : tensor<64x80x14x14xf32>
    %v879 = stablehlo.add %v878, %v877 : tensor<64x80x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v881 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<64xf32>) -> tensor<64x15680xf32>
    %v882 = stablehlo.multiply %v881, %v880 : tensor<64x15680xf32>
    %v883 = stablehlo.add %v882, %v771 : tensor<64x15680xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v885 = stablehlo.convolution(%v884, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<64x480x14x14xf32>
    %v886 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v887 = stablehlo.add %v885, %v886 : tensor<64x480x14x14xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v891 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v892 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v893 = stablehlo.reduce(%v889 init: %v890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v894 = stablehlo.broadcast_in_dim %v893, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v895 = stablehlo.divide %v894, %v891 : tensor<64x480x14x14xf32>
    %v896 = stablehlo.subtract %v889, %v895 : tensor<64x480x14x14xf32>
    %v897 = stablehlo.multiply %v896, %v896 : tensor<64x480x14x14xf32>
    %v898 = stablehlo.reduce(%v897 init: %v890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v899 = stablehlo.broadcast_in_dim %v898, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v900 = stablehlo.divide %v899, %v891 : tensor<64x480x14x14xf32>
    %v901 = stablehlo.add %v900, %v892 : tensor<64x480x14x14xf32>
    %v902 = stablehlo.rsqrt %v901 : tensor<64x480x14x14xf32>
    %v903 = stablehlo.multiply %v896, %v902 : tensor<64x480x14x14xf32>
    %v904 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v905 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v906 = stablehlo.multiply %v903, %v904 : tensor<64x480x14x14xf32>
    %v907 = stablehlo.add %v906, %v905 : tensor<64x480x14x14xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v909 = stablehlo.logistic %v908 : tensor<64x94080xf32>
    %v910 = stablehlo.multiply %v908, %v909 : tensor<64x94080xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v912 = stablehlo.convolution(%v911, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<64x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<64x480x14x14xf32>
    %v913 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v914 = stablehlo.add %v912, %v913 : tensor<64x480x14x14xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v918 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v919 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v920 = stablehlo.reduce(%v916 init: %v917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v921 = stablehlo.broadcast_in_dim %v920, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v922 = stablehlo.divide %v921, %v918 : tensor<64x480x14x14xf32>
    %v923 = stablehlo.subtract %v916, %v922 : tensor<64x480x14x14xf32>
    %v924 = stablehlo.multiply %v923, %v923 : tensor<64x480x14x14xf32>
    %v925 = stablehlo.reduce(%v924 init: %v917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v926 = stablehlo.broadcast_in_dim %v925, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v927 = stablehlo.divide %v926, %v918 : tensor<64x480x14x14xf32>
    %v928 = stablehlo.add %v927, %v919 : tensor<64x480x14x14xf32>
    %v929 = stablehlo.rsqrt %v928 : tensor<64x480x14x14xf32>
    %v930 = stablehlo.multiply %v923, %v929 : tensor<64x480x14x14xf32>
    %v931 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v932 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v933 = stablehlo.multiply %v930, %v931 : tensor<64x480x14x14xf32>
    %v934 = stablehlo.add %v933, %v932 : tensor<64x480x14x14xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v936 = stablehlo.logistic %v935 : tensor<64x94080xf32>
    %v937 = stablehlo.multiply %v935, %v936 : tensor<64x94080xf32>
    %v938 = stablehlo.reshape %v937 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v940 = stablehlo.reduce(%v938 init: %v939) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v941 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v942 = stablehlo.divide %v940, %v941 : tensor<64x480xf32>
    %v943 = stablehlo.dot_general %v942, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v944 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v945 = stablehlo.add %v943, %v944 : tensor<64x20xf32>
    %v946 = stablehlo.logistic %v945 : tensor<64x20xf32>
    %v947 = stablehlo.multiply %v945, %v946 : tensor<64x20xf32>
    %v948 = stablehlo.dot_general %v947, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v949 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v950 = stablehlo.add %v948, %v949 : tensor<64x480xf32>
    %v951 = stablehlo.reshape %v937 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v952 = stablehlo.constant dense<0.0> : tensor<f32>
    %v953 = stablehlo.reduce(%v951 init: %v952) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<64x480xf32>
    %v954 = stablehlo.constant dense<196.0> : tensor<64x480xf32>
    %v955 = stablehlo.divide %v953, %v954 : tensor<64x480xf32>
    %v956 = stablehlo.dot_general %v955, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x480xf32>, tensor<480x20xf32>) -> tensor<64x20xf32>
    %v957 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<64x20xf32>
    %v958 = stablehlo.add %v956, %v957 : tensor<64x20xf32>
    %v959 = stablehlo.logistic %v958 : tensor<64x20xf32>
    %v960 = stablehlo.multiply %v958, %v959 : tensor<64x20xf32>
    %v961 = stablehlo.dot_general %v960, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x20xf32>, tensor<20x480xf32>) -> tensor<64x480xf32>
    %v962 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<64x480xf32>
    %v963 = stablehlo.add %v961, %v962 : tensor<64x480xf32>
    %v964 = stablehlo.logistic %v963 : tensor<64x480xf32>
    %v965 = stablehlo.broadcast_in_dim %v964, dims = [0, 1] : (tensor<64x480xf32>) -> tensor<64x480x14x14xf32>
    %v966 = stablehlo.multiply %v951, %v965 : tensor<64x480x14x14xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v969 = stablehlo.convolution(%v968, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<64x112x14x14xf32>
    %v970 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v971 = stablehlo.add %v969, %v970 : tensor<64x112x14x14xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v974 = stablehlo.constant dense<0.0> : tensor<f32>
    %v975 = stablehlo.constant dense<12544.0> : tensor<64x112x14x14xf32>
    %v976 = stablehlo.constant dense<1.0e-5> : tensor<64x112x14x14xf32>
    %v977 = stablehlo.reduce(%v973 init: %v974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v978 = stablehlo.broadcast_in_dim %v977, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v979 = stablehlo.divide %v978, %v975 : tensor<64x112x14x14xf32>
    %v980 = stablehlo.subtract %v973, %v979 : tensor<64x112x14x14xf32>
    %v981 = stablehlo.multiply %v980, %v980 : tensor<64x112x14x14xf32>
    %v982 = stablehlo.reduce(%v981 init: %v974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v983 = stablehlo.broadcast_in_dim %v982, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v984 = stablehlo.divide %v983, %v975 : tensor<64x112x14x14xf32>
    %v985 = stablehlo.add %v984, %v976 : tensor<64x112x14x14xf32>
    %v986 = stablehlo.rsqrt %v985 : tensor<64x112x14x14xf32>
    %v987 = stablehlo.multiply %v980, %v986 : tensor<64x112x14x14xf32>
    %v988 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v989 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v990 = stablehlo.multiply %v987, %v988 : tensor<64x112x14x14xf32>
    %v991 = stablehlo.add %v990, %v989 : tensor<64x112x14x14xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v994 = stablehlo.convolution(%v993, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<64x672x14x14xf32>
    %v995 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v996 = stablehlo.add %v994, %v995 : tensor<64x672x14x14xf32>
    %v997 = stablehlo.reshape %v996 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v998 = stablehlo.reshape %v997 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v999 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1000 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1001 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1002 = stablehlo.reduce(%v998 init: %v999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1003 = stablehlo.broadcast_in_dim %v1002, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1004 = stablehlo.divide %v1003, %v1000 : tensor<64x672x14x14xf32>
    %v1005 = stablehlo.subtract %v998, %v1004 : tensor<64x672x14x14xf32>
    %v1006 = stablehlo.multiply %v1005, %v1005 : tensor<64x672x14x14xf32>
    %v1007 = stablehlo.reduce(%v1006 init: %v999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1008 = stablehlo.broadcast_in_dim %v1007, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1009 = stablehlo.divide %v1008, %v1000 : tensor<64x672x14x14xf32>
    %v1010 = stablehlo.add %v1009, %v1001 : tensor<64x672x14x14xf32>
    %v1011 = stablehlo.rsqrt %v1010 : tensor<64x672x14x14xf32>
    %v1012 = stablehlo.multiply %v1005, %v1011 : tensor<64x672x14x14xf32>
    %v1013 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1014 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1015 = stablehlo.multiply %v1012, %v1013 : tensor<64x672x14x14xf32>
    %v1016 = stablehlo.add %v1015, %v1014 : tensor<64x672x14x14xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1018 = stablehlo.logistic %v1017 : tensor<64x131712xf32>
    %v1019 = stablehlo.multiply %v1017, %v1018 : tensor<64x131712xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1021 = stablehlo.convolution(%v1020, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<64x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<64x672x14x14xf32>
    %v1022 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1023 = stablehlo.add %v1021, %v1022 : tensor<64x672x14x14xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1025 = stablehlo.reshape %v1024 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1026 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1027 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1028 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1029 = stablehlo.reduce(%v1025 init: %v1026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1030 = stablehlo.broadcast_in_dim %v1029, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1031 = stablehlo.divide %v1030, %v1027 : tensor<64x672x14x14xf32>
    %v1032 = stablehlo.subtract %v1025, %v1031 : tensor<64x672x14x14xf32>
    %v1033 = stablehlo.multiply %v1032, %v1032 : tensor<64x672x14x14xf32>
    %v1034 = stablehlo.reduce(%v1033 init: %v1026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1035 = stablehlo.broadcast_in_dim %v1034, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1036 = stablehlo.divide %v1035, %v1027 : tensor<64x672x14x14xf32>
    %v1037 = stablehlo.add %v1036, %v1028 : tensor<64x672x14x14xf32>
    %v1038 = stablehlo.rsqrt %v1037 : tensor<64x672x14x14xf32>
    %v1039 = stablehlo.multiply %v1032, %v1038 : tensor<64x672x14x14xf32>
    %v1040 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1041 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1042 = stablehlo.multiply %v1039, %v1040 : tensor<64x672x14x14xf32>
    %v1043 = stablehlo.add %v1042, %v1041 : tensor<64x672x14x14xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1045 = stablehlo.logistic %v1044 : tensor<64x131712xf32>
    %v1046 = stablehlo.multiply %v1044, %v1045 : tensor<64x131712xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1049 = stablehlo.reduce(%v1047 init: %v1048) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1050 = stablehlo.constant dense<196.0> : tensor<64x672xf32>
    %v1051 = stablehlo.divide %v1049, %v1050 : tensor<64x672xf32>
    %v1052 = stablehlo.dot_general %v1051, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1053 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1054 = stablehlo.add %v1052, %v1053 : tensor<64x28xf32>
    %v1055 = stablehlo.logistic %v1054 : tensor<64x28xf32>
    %v1056 = stablehlo.multiply %v1054, %v1055 : tensor<64x28xf32>
    %v1057 = stablehlo.dot_general %v1056, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1058 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1059 = stablehlo.add %v1057, %v1058 : tensor<64x672xf32>
    %v1060 = stablehlo.reshape %v1046 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1062 = stablehlo.reduce(%v1060 init: %v1061) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1063 = stablehlo.constant dense<196.0> : tensor<64x672xf32>
    %v1064 = stablehlo.divide %v1062, %v1063 : tensor<64x672xf32>
    %v1065 = stablehlo.dot_general %v1064, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1066 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1067 = stablehlo.add %v1065, %v1066 : tensor<64x28xf32>
    %v1068 = stablehlo.logistic %v1067 : tensor<64x28xf32>
    %v1069 = stablehlo.multiply %v1067, %v1068 : tensor<64x28xf32>
    %v1070 = stablehlo.dot_general %v1069, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1071 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1072 = stablehlo.add %v1070, %v1071 : tensor<64x672xf32>
    %v1073 = stablehlo.logistic %v1072 : tensor<64x672xf32>
    %v1074 = stablehlo.broadcast_in_dim %v1073, dims = [0, 1] : (tensor<64x672xf32>) -> tensor<64x672x14x14xf32>
    %v1075 = stablehlo.multiply %v1060, %v1074 : tensor<64x672x14x14xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1078 = stablehlo.convolution(%v1077, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<64x112x14x14xf32>
    %v1079 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1080 = stablehlo.add %v1078, %v1079 : tensor<64x112x14x14xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1083 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1084 = stablehlo.constant dense<12544.0> : tensor<64x112x14x14xf32>
    %v1085 = stablehlo.constant dense<1.0e-5> : tensor<64x112x14x14xf32>
    %v1086 = stablehlo.reduce(%v1082 init: %v1083) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1087 = stablehlo.broadcast_in_dim %v1086, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1088 = stablehlo.divide %v1087, %v1084 : tensor<64x112x14x14xf32>
    %v1089 = stablehlo.subtract %v1082, %v1088 : tensor<64x112x14x14xf32>
    %v1090 = stablehlo.multiply %v1089, %v1089 : tensor<64x112x14x14xf32>
    %v1091 = stablehlo.reduce(%v1090 init: %v1083) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1092 = stablehlo.broadcast_in_dim %v1091, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1093 = stablehlo.divide %v1092, %v1084 : tensor<64x112x14x14xf32>
    %v1094 = stablehlo.add %v1093, %v1085 : tensor<64x112x14x14xf32>
    %v1095 = stablehlo.rsqrt %v1094 : tensor<64x112x14x14xf32>
    %v1096 = stablehlo.multiply %v1089, %v1095 : tensor<64x112x14x14xf32>
    %v1097 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1098 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1099 = stablehlo.multiply %v1096, %v1097 : tensor<64x112x14x14xf32>
    %v1100 = stablehlo.add %v1099, %v1098 : tensor<64x112x14x14xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1102 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<64xf32>) -> tensor<64x21952xf32>
    %v1103 = stablehlo.multiply %v1102, %v1101 : tensor<64x21952xf32>
    %v1104 = stablehlo.add %v1103, %v992 : tensor<64x21952xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1106 = stablehlo.convolution(%v1105, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<64x672x14x14xf32>
    %v1107 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1108 = stablehlo.add %v1106, %v1107 : tensor<64x672x14x14xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1110 = stablehlo.reshape %v1109 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1111 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1112 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1113 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1114 = stablehlo.reduce(%v1110 init: %v1111) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1115 = stablehlo.broadcast_in_dim %v1114, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1116 = stablehlo.divide %v1115, %v1112 : tensor<64x672x14x14xf32>
    %v1117 = stablehlo.subtract %v1110, %v1116 : tensor<64x672x14x14xf32>
    %v1118 = stablehlo.multiply %v1117, %v1117 : tensor<64x672x14x14xf32>
    %v1119 = stablehlo.reduce(%v1118 init: %v1111) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1120 = stablehlo.broadcast_in_dim %v1119, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1121 = stablehlo.divide %v1120, %v1112 : tensor<64x672x14x14xf32>
    %v1122 = stablehlo.add %v1121, %v1113 : tensor<64x672x14x14xf32>
    %v1123 = stablehlo.rsqrt %v1122 : tensor<64x672x14x14xf32>
    %v1124 = stablehlo.multiply %v1117, %v1123 : tensor<64x672x14x14xf32>
    %v1125 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1126 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1127 = stablehlo.multiply %v1124, %v1125 : tensor<64x672x14x14xf32>
    %v1128 = stablehlo.add %v1127, %v1126 : tensor<64x672x14x14xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1130 = stablehlo.logistic %v1129 : tensor<64x131712xf32>
    %v1131 = stablehlo.multiply %v1129, %v1130 : tensor<64x131712xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1133 = stablehlo.convolution(%v1132, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<64x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<64x672x14x14xf32>
    %v1134 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1135 = stablehlo.add %v1133, %v1134 : tensor<64x672x14x14xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1139 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1140 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1141 = stablehlo.reduce(%v1137 init: %v1138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1142 = stablehlo.broadcast_in_dim %v1141, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1143 = stablehlo.divide %v1142, %v1139 : tensor<64x672x14x14xf32>
    %v1144 = stablehlo.subtract %v1137, %v1143 : tensor<64x672x14x14xf32>
    %v1145 = stablehlo.multiply %v1144, %v1144 : tensor<64x672x14x14xf32>
    %v1146 = stablehlo.reduce(%v1145 init: %v1138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1147 = stablehlo.broadcast_in_dim %v1146, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1148 = stablehlo.divide %v1147, %v1139 : tensor<64x672x14x14xf32>
    %v1149 = stablehlo.add %v1148, %v1140 : tensor<64x672x14x14xf32>
    %v1150 = stablehlo.rsqrt %v1149 : tensor<64x672x14x14xf32>
    %v1151 = stablehlo.multiply %v1144, %v1150 : tensor<64x672x14x14xf32>
    %v1152 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1153 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1154 = stablehlo.multiply %v1151, %v1152 : tensor<64x672x14x14xf32>
    %v1155 = stablehlo.add %v1154, %v1153 : tensor<64x672x14x14xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1157 = stablehlo.logistic %v1156 : tensor<64x131712xf32>
    %v1158 = stablehlo.multiply %v1156, %v1157 : tensor<64x131712xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1161 = stablehlo.reduce(%v1159 init: %v1160) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1162 = stablehlo.constant dense<196.0> : tensor<64x672xf32>
    %v1163 = stablehlo.divide %v1161, %v1162 : tensor<64x672xf32>
    %v1164 = stablehlo.dot_general %v1163, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1165 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1166 = stablehlo.add %v1164, %v1165 : tensor<64x28xf32>
    %v1167 = stablehlo.logistic %v1166 : tensor<64x28xf32>
    %v1168 = stablehlo.multiply %v1166, %v1167 : tensor<64x28xf32>
    %v1169 = stablehlo.dot_general %v1168, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1170 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1171 = stablehlo.add %v1169, %v1170 : tensor<64x672xf32>
    %v1172 = stablehlo.reshape %v1158 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1174 = stablehlo.reduce(%v1172 init: %v1173) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1175 = stablehlo.constant dense<196.0> : tensor<64x672xf32>
    %v1176 = stablehlo.divide %v1174, %v1175 : tensor<64x672xf32>
    %v1177 = stablehlo.dot_general %v1176, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1178 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1179 = stablehlo.add %v1177, %v1178 : tensor<64x28xf32>
    %v1180 = stablehlo.logistic %v1179 : tensor<64x28xf32>
    %v1181 = stablehlo.multiply %v1179, %v1180 : tensor<64x28xf32>
    %v1182 = stablehlo.dot_general %v1181, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1183 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1184 = stablehlo.add %v1182, %v1183 : tensor<64x672xf32>
    %v1185 = stablehlo.logistic %v1184 : tensor<64x672xf32>
    %v1186 = stablehlo.broadcast_in_dim %v1185, dims = [0, 1] : (tensor<64x672xf32>) -> tensor<64x672x14x14xf32>
    %v1187 = stablehlo.multiply %v1172, %v1186 : tensor<64x672x14x14xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1190 = stablehlo.convolution(%v1189, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<64x112x14x14xf32>
    %v1191 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1192 = stablehlo.add %v1190, %v1191 : tensor<64x112x14x14xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1195 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1196 = stablehlo.constant dense<12544.0> : tensor<64x112x14x14xf32>
    %v1197 = stablehlo.constant dense<1.0e-5> : tensor<64x112x14x14xf32>
    %v1198 = stablehlo.reduce(%v1194 init: %v1195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1199 = stablehlo.broadcast_in_dim %v1198, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1200 = stablehlo.divide %v1199, %v1196 : tensor<64x112x14x14xf32>
    %v1201 = stablehlo.subtract %v1194, %v1200 : tensor<64x112x14x14xf32>
    %v1202 = stablehlo.multiply %v1201, %v1201 : tensor<64x112x14x14xf32>
    %v1203 = stablehlo.reduce(%v1202 init: %v1195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1204 = stablehlo.broadcast_in_dim %v1203, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1205 = stablehlo.divide %v1204, %v1196 : tensor<64x112x14x14xf32>
    %v1206 = stablehlo.add %v1205, %v1197 : tensor<64x112x14x14xf32>
    %v1207 = stablehlo.rsqrt %v1206 : tensor<64x112x14x14xf32>
    %v1208 = stablehlo.multiply %v1201, %v1207 : tensor<64x112x14x14xf32>
    %v1209 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1210 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<64x112x14x14xf32>
    %v1211 = stablehlo.multiply %v1208, %v1209 : tensor<64x112x14x14xf32>
    %v1212 = stablehlo.add %v1211, %v1210 : tensor<64x112x14x14xf32>
    %v1213 = stablehlo.reshape %v1212 : (tensor<64x112x14x14xf32>) -> tensor<64x21952xf32>
    %v1214 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<64xf32>) -> tensor<64x21952xf32>
    %v1215 = stablehlo.multiply %v1214, %v1213 : tensor<64x21952xf32>
    %v1216 = stablehlo.add %v1215, %v1104 : tensor<64x21952xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<64x21952xf32>) -> tensor<64x112x14x14xf32>
    %v1218 = stablehlo.convolution(%v1217, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<64x672x14x14xf32>
    %v1219 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1220 = stablehlo.add %v1218, %v1219 : tensor<64x672x14x14xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1222 = stablehlo.reshape %v1221 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1224 = stablehlo.constant dense<12544.0> : tensor<64x672x14x14xf32>
    %v1225 = stablehlo.constant dense<1.0e-5> : tensor<64x672x14x14xf32>
    %v1226 = stablehlo.reduce(%v1222 init: %v1223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1227 = stablehlo.broadcast_in_dim %v1226, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1228 = stablehlo.divide %v1227, %v1224 : tensor<64x672x14x14xf32>
    %v1229 = stablehlo.subtract %v1222, %v1228 : tensor<64x672x14x14xf32>
    %v1230 = stablehlo.multiply %v1229, %v1229 : tensor<64x672x14x14xf32>
    %v1231 = stablehlo.reduce(%v1230 init: %v1223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1232 = stablehlo.broadcast_in_dim %v1231, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1233 = stablehlo.divide %v1232, %v1224 : tensor<64x672x14x14xf32>
    %v1234 = stablehlo.add %v1233, %v1225 : tensor<64x672x14x14xf32>
    %v1235 = stablehlo.rsqrt %v1234 : tensor<64x672x14x14xf32>
    %v1236 = stablehlo.multiply %v1229, %v1235 : tensor<64x672x14x14xf32>
    %v1237 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1238 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x14x14xf32>
    %v1239 = stablehlo.multiply %v1236, %v1237 : tensor<64x672x14x14xf32>
    %v1240 = stablehlo.add %v1239, %v1238 : tensor<64x672x14x14xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<64x672x14x14xf32>) -> tensor<64x131712xf32>
    %v1242 = stablehlo.logistic %v1241 : tensor<64x131712xf32>
    %v1243 = stablehlo.multiply %v1241, %v1242 : tensor<64x131712xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<64x131712xf32>) -> tensor<64x672x14x14xf32>
    %v1245 = stablehlo.convolution(%v1244, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<64x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<64x672x7x7xf32>
    %v1246 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1247 = stablehlo.add %v1245, %v1246 : tensor<64x672x7x7xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<64x672x7x7xf32>) -> tensor<64x32928xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1250 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1251 = stablehlo.constant dense<3136.0> : tensor<64x672x7x7xf32>
    %v1252 = stablehlo.constant dense<1.0e-5> : tensor<64x672x7x7xf32>
    %v1253 = stablehlo.reduce(%v1249 init: %v1250) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1254 = stablehlo.broadcast_in_dim %v1253, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1255 = stablehlo.divide %v1254, %v1251 : tensor<64x672x7x7xf32>
    %v1256 = stablehlo.subtract %v1249, %v1255 : tensor<64x672x7x7xf32>
    %v1257 = stablehlo.multiply %v1256, %v1256 : tensor<64x672x7x7xf32>
    %v1258 = stablehlo.reduce(%v1257 init: %v1250) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1259 = stablehlo.broadcast_in_dim %v1258, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1260 = stablehlo.divide %v1259, %v1251 : tensor<64x672x7x7xf32>
    %v1261 = stablehlo.add %v1260, %v1252 : tensor<64x672x7x7xf32>
    %v1262 = stablehlo.rsqrt %v1261 : tensor<64x672x7x7xf32>
    %v1263 = stablehlo.multiply %v1256, %v1262 : tensor<64x672x7x7xf32>
    %v1264 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1265 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<64x672x7x7xf32>
    %v1266 = stablehlo.multiply %v1263, %v1264 : tensor<64x672x7x7xf32>
    %v1267 = stablehlo.add %v1266, %v1265 : tensor<64x672x7x7xf32>
    %v1268 = stablehlo.reshape %v1267 : (tensor<64x672x7x7xf32>) -> tensor<64x32928xf32>
    %v1269 = stablehlo.logistic %v1268 : tensor<64x32928xf32>
    %v1270 = stablehlo.multiply %v1268, %v1269 : tensor<64x32928xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1273 = stablehlo.reduce(%v1271 init: %v1272) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x7x7xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1274 = stablehlo.constant dense<49.0> : tensor<64x672xf32>
    %v1275 = stablehlo.divide %v1273, %v1274 : tensor<64x672xf32>
    %v1276 = stablehlo.dot_general %v1275, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1277 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1278 = stablehlo.add %v1276, %v1277 : tensor<64x28xf32>
    %v1279 = stablehlo.logistic %v1278 : tensor<64x28xf32>
    %v1280 = stablehlo.multiply %v1278, %v1279 : tensor<64x28xf32>
    %v1281 = stablehlo.dot_general %v1280, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1282 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1283 = stablehlo.add %v1281, %v1282 : tensor<64x672xf32>
    %v1284 = stablehlo.reshape %v1270 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1286 = stablehlo.reduce(%v1284 init: %v1285) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x672x7x7xf32>, tensor<f32>) -> tensor<64x672xf32>
    %v1287 = stablehlo.constant dense<49.0> : tensor<64x672xf32>
    %v1288 = stablehlo.divide %v1286, %v1287 : tensor<64x672xf32>
    %v1289 = stablehlo.dot_general %v1288, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x672xf32>, tensor<672x28xf32>) -> tensor<64x28xf32>
    %v1290 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<64x28xf32>
    %v1291 = stablehlo.add %v1289, %v1290 : tensor<64x28xf32>
    %v1292 = stablehlo.logistic %v1291 : tensor<64x28xf32>
    %v1293 = stablehlo.multiply %v1291, %v1292 : tensor<64x28xf32>
    %v1294 = stablehlo.dot_general %v1293, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x28xf32>, tensor<28x672xf32>) -> tensor<64x672xf32>
    %v1295 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<64x672xf32>
    %v1296 = stablehlo.add %v1294, %v1295 : tensor<64x672xf32>
    %v1297 = stablehlo.logistic %v1296 : tensor<64x672xf32>
    %v1298 = stablehlo.broadcast_in_dim %v1297, dims = [0, 1] : (tensor<64x672xf32>) -> tensor<64x672x7x7xf32>
    %v1299 = stablehlo.multiply %v1284, %v1298 : tensor<64x672x7x7xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<64x672x7x7xf32>) -> tensor<64x32928xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<64x32928xf32>) -> tensor<64x672x7x7xf32>
    %v1302 = stablehlo.convolution(%v1301, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<64x192x7x7xf32>
    %v1303 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1304 = stablehlo.add %v1302, %v1303 : tensor<64x192x7x7xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1306 = stablehlo.reshape %v1305 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1307 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1308 = stablehlo.constant dense<3136.0> : tensor<64x192x7x7xf32>
    %v1309 = stablehlo.constant dense<1.0e-5> : tensor<64x192x7x7xf32>
    %v1310 = stablehlo.reduce(%v1306 init: %v1307) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1311 = stablehlo.broadcast_in_dim %v1310, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1312 = stablehlo.divide %v1311, %v1308 : tensor<64x192x7x7xf32>
    %v1313 = stablehlo.subtract %v1306, %v1312 : tensor<64x192x7x7xf32>
    %v1314 = stablehlo.multiply %v1313, %v1313 : tensor<64x192x7x7xf32>
    %v1315 = stablehlo.reduce(%v1314 init: %v1307) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1316 = stablehlo.broadcast_in_dim %v1315, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1317 = stablehlo.divide %v1316, %v1308 : tensor<64x192x7x7xf32>
    %v1318 = stablehlo.add %v1317, %v1309 : tensor<64x192x7x7xf32>
    %v1319 = stablehlo.rsqrt %v1318 : tensor<64x192x7x7xf32>
    %v1320 = stablehlo.multiply %v1313, %v1319 : tensor<64x192x7x7xf32>
    %v1321 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1322 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1323 = stablehlo.multiply %v1320, %v1321 : tensor<64x192x7x7xf32>
    %v1324 = stablehlo.add %v1323, %v1322 : tensor<64x192x7x7xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1327 = stablehlo.convolution(%v1326, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<64x1152x7x7xf32>
    %v1328 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1329 = stablehlo.add %v1327, %v1328 : tensor<64x1152x7x7xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1333 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1334 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1335 = stablehlo.reduce(%v1331 init: %v1332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1336 = stablehlo.broadcast_in_dim %v1335, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1337 = stablehlo.divide %v1336, %v1333 : tensor<64x1152x7x7xf32>
    %v1338 = stablehlo.subtract %v1331, %v1337 : tensor<64x1152x7x7xf32>
    %v1339 = stablehlo.multiply %v1338, %v1338 : tensor<64x1152x7x7xf32>
    %v1340 = stablehlo.reduce(%v1339 init: %v1332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1341 = stablehlo.broadcast_in_dim %v1340, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1342 = stablehlo.divide %v1341, %v1333 : tensor<64x1152x7x7xf32>
    %v1343 = stablehlo.add %v1342, %v1334 : tensor<64x1152x7x7xf32>
    %v1344 = stablehlo.rsqrt %v1343 : tensor<64x1152x7x7xf32>
    %v1345 = stablehlo.multiply %v1338, %v1344 : tensor<64x1152x7x7xf32>
    %v1346 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1347 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1348 = stablehlo.multiply %v1345, %v1346 : tensor<64x1152x7x7xf32>
    %v1349 = stablehlo.add %v1348, %v1347 : tensor<64x1152x7x7xf32>
    %v1350 = stablehlo.reshape %v1349 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1351 = stablehlo.logistic %v1350 : tensor<64x56448xf32>
    %v1352 = stablehlo.multiply %v1350, %v1351 : tensor<64x56448xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1354 = stablehlo.convolution(%v1353, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<64x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<64x1152x7x7xf32>
    %v1355 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1356 = stablehlo.add %v1354, %v1355 : tensor<64x1152x7x7xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1360 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1361 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1362 = stablehlo.reduce(%v1358 init: %v1359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1363 = stablehlo.broadcast_in_dim %v1362, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1364 = stablehlo.divide %v1363, %v1360 : tensor<64x1152x7x7xf32>
    %v1365 = stablehlo.subtract %v1358, %v1364 : tensor<64x1152x7x7xf32>
    %v1366 = stablehlo.multiply %v1365, %v1365 : tensor<64x1152x7x7xf32>
    %v1367 = stablehlo.reduce(%v1366 init: %v1359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1368 = stablehlo.broadcast_in_dim %v1367, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1369 = stablehlo.divide %v1368, %v1360 : tensor<64x1152x7x7xf32>
    %v1370 = stablehlo.add %v1369, %v1361 : tensor<64x1152x7x7xf32>
    %v1371 = stablehlo.rsqrt %v1370 : tensor<64x1152x7x7xf32>
    %v1372 = stablehlo.multiply %v1365, %v1371 : tensor<64x1152x7x7xf32>
    %v1373 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1374 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1375 = stablehlo.multiply %v1372, %v1373 : tensor<64x1152x7x7xf32>
    %v1376 = stablehlo.add %v1375, %v1374 : tensor<64x1152x7x7xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1378 = stablehlo.logistic %v1377 : tensor<64x56448xf32>
    %v1379 = stablehlo.multiply %v1377, %v1378 : tensor<64x56448xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1382 = stablehlo.reduce(%v1380 init: %v1381) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1383 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1384 = stablehlo.divide %v1382, %v1383 : tensor<64x1152xf32>
    %v1385 = stablehlo.dot_general %v1384, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1386 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1387 = stablehlo.add %v1385, %v1386 : tensor<64x48xf32>
    %v1388 = stablehlo.logistic %v1387 : tensor<64x48xf32>
    %v1389 = stablehlo.multiply %v1387, %v1388 : tensor<64x48xf32>
    %v1390 = stablehlo.dot_general %v1389, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1391 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1392 = stablehlo.add %v1390, %v1391 : tensor<64x1152xf32>
    %v1393 = stablehlo.reshape %v1379 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1395 = stablehlo.reduce(%v1393 init: %v1394) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1396 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1397 = stablehlo.divide %v1395, %v1396 : tensor<64x1152xf32>
    %v1398 = stablehlo.dot_general %v1397, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1399 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1400 = stablehlo.add %v1398, %v1399 : tensor<64x48xf32>
    %v1401 = stablehlo.logistic %v1400 : tensor<64x48xf32>
    %v1402 = stablehlo.multiply %v1400, %v1401 : tensor<64x48xf32>
    %v1403 = stablehlo.dot_general %v1402, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1404 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1405 = stablehlo.add %v1403, %v1404 : tensor<64x1152xf32>
    %v1406 = stablehlo.logistic %v1405 : tensor<64x1152xf32>
    %v1407 = stablehlo.broadcast_in_dim %v1406, dims = [0, 1] : (tensor<64x1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1408 = stablehlo.multiply %v1393, %v1407 : tensor<64x1152x7x7xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1410 = stablehlo.reshape %v1409 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1411 = stablehlo.convolution(%v1410, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<64x192x7x7xf32>
    %v1412 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1413 = stablehlo.add %v1411, %v1412 : tensor<64x192x7x7xf32>
    %v1414 = stablehlo.reshape %v1413 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1417 = stablehlo.constant dense<3136.0> : tensor<64x192x7x7xf32>
    %v1418 = stablehlo.constant dense<1.0e-5> : tensor<64x192x7x7xf32>
    %v1419 = stablehlo.reduce(%v1415 init: %v1416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1420 = stablehlo.broadcast_in_dim %v1419, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1421 = stablehlo.divide %v1420, %v1417 : tensor<64x192x7x7xf32>
    %v1422 = stablehlo.subtract %v1415, %v1421 : tensor<64x192x7x7xf32>
    %v1423 = stablehlo.multiply %v1422, %v1422 : tensor<64x192x7x7xf32>
    %v1424 = stablehlo.reduce(%v1423 init: %v1416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1425 = stablehlo.broadcast_in_dim %v1424, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1426 = stablehlo.divide %v1425, %v1417 : tensor<64x192x7x7xf32>
    %v1427 = stablehlo.add %v1426, %v1418 : tensor<64x192x7x7xf32>
    %v1428 = stablehlo.rsqrt %v1427 : tensor<64x192x7x7xf32>
    %v1429 = stablehlo.multiply %v1422, %v1428 : tensor<64x192x7x7xf32>
    %v1430 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1431 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1432 = stablehlo.multiply %v1429, %v1430 : tensor<64x192x7x7xf32>
    %v1433 = stablehlo.add %v1432, %v1431 : tensor<64x192x7x7xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1435 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<64xf32>) -> tensor<64x9408xf32>
    %v1436 = stablehlo.multiply %v1435, %v1434 : tensor<64x9408xf32>
    %v1437 = stablehlo.add %v1436, %v1325 : tensor<64x9408xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1439 = stablehlo.convolution(%v1438, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<64x1152x7x7xf32>
    %v1440 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1441 = stablehlo.add %v1439, %v1440 : tensor<64x1152x7x7xf32>
    %v1442 = stablehlo.reshape %v1441 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1445 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1446 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1447 = stablehlo.reduce(%v1443 init: %v1444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1448 = stablehlo.broadcast_in_dim %v1447, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1449 = stablehlo.divide %v1448, %v1445 : tensor<64x1152x7x7xf32>
    %v1450 = stablehlo.subtract %v1443, %v1449 : tensor<64x1152x7x7xf32>
    %v1451 = stablehlo.multiply %v1450, %v1450 : tensor<64x1152x7x7xf32>
    %v1452 = stablehlo.reduce(%v1451 init: %v1444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1453 = stablehlo.broadcast_in_dim %v1452, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1454 = stablehlo.divide %v1453, %v1445 : tensor<64x1152x7x7xf32>
    %v1455 = stablehlo.add %v1454, %v1446 : tensor<64x1152x7x7xf32>
    %v1456 = stablehlo.rsqrt %v1455 : tensor<64x1152x7x7xf32>
    %v1457 = stablehlo.multiply %v1450, %v1456 : tensor<64x1152x7x7xf32>
    %v1458 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1459 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1460 = stablehlo.multiply %v1457, %v1458 : tensor<64x1152x7x7xf32>
    %v1461 = stablehlo.add %v1460, %v1459 : tensor<64x1152x7x7xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1463 = stablehlo.logistic %v1462 : tensor<64x56448xf32>
    %v1464 = stablehlo.multiply %v1462, %v1463 : tensor<64x56448xf32>
    %v1465 = stablehlo.reshape %v1464 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1466 = stablehlo.convolution(%v1465, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<64x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<64x1152x7x7xf32>
    %v1467 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1468 = stablehlo.add %v1466, %v1467 : tensor<64x1152x7x7xf32>
    %v1469 = stablehlo.reshape %v1468 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1470 = stablehlo.reshape %v1469 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1472 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1473 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1474 = stablehlo.reduce(%v1470 init: %v1471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1475 = stablehlo.broadcast_in_dim %v1474, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1476 = stablehlo.divide %v1475, %v1472 : tensor<64x1152x7x7xf32>
    %v1477 = stablehlo.subtract %v1470, %v1476 : tensor<64x1152x7x7xf32>
    %v1478 = stablehlo.multiply %v1477, %v1477 : tensor<64x1152x7x7xf32>
    %v1479 = stablehlo.reduce(%v1478 init: %v1471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1480 = stablehlo.broadcast_in_dim %v1479, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1481 = stablehlo.divide %v1480, %v1472 : tensor<64x1152x7x7xf32>
    %v1482 = stablehlo.add %v1481, %v1473 : tensor<64x1152x7x7xf32>
    %v1483 = stablehlo.rsqrt %v1482 : tensor<64x1152x7x7xf32>
    %v1484 = stablehlo.multiply %v1477, %v1483 : tensor<64x1152x7x7xf32>
    %v1485 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1486 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1487 = stablehlo.multiply %v1484, %v1485 : tensor<64x1152x7x7xf32>
    %v1488 = stablehlo.add %v1487, %v1486 : tensor<64x1152x7x7xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1490 = stablehlo.logistic %v1489 : tensor<64x56448xf32>
    %v1491 = stablehlo.multiply %v1489, %v1490 : tensor<64x56448xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1494 = stablehlo.reduce(%v1492 init: %v1493) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1495 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1496 = stablehlo.divide %v1494, %v1495 : tensor<64x1152xf32>
    %v1497 = stablehlo.dot_general %v1496, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1498 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1499 = stablehlo.add %v1497, %v1498 : tensor<64x48xf32>
    %v1500 = stablehlo.logistic %v1499 : tensor<64x48xf32>
    %v1501 = stablehlo.multiply %v1499, %v1500 : tensor<64x48xf32>
    %v1502 = stablehlo.dot_general %v1501, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1503 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1504 = stablehlo.add %v1502, %v1503 : tensor<64x1152xf32>
    %v1505 = stablehlo.reshape %v1491 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1507 = stablehlo.reduce(%v1505 init: %v1506) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1508 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1509 = stablehlo.divide %v1507, %v1508 : tensor<64x1152xf32>
    %v1510 = stablehlo.dot_general %v1509, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1511 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1512 = stablehlo.add %v1510, %v1511 : tensor<64x48xf32>
    %v1513 = stablehlo.logistic %v1512 : tensor<64x48xf32>
    %v1514 = stablehlo.multiply %v1512, %v1513 : tensor<64x48xf32>
    %v1515 = stablehlo.dot_general %v1514, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1516 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1517 = stablehlo.add %v1515, %v1516 : tensor<64x1152xf32>
    %v1518 = stablehlo.logistic %v1517 : tensor<64x1152xf32>
    %v1519 = stablehlo.broadcast_in_dim %v1518, dims = [0, 1] : (tensor<64x1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1520 = stablehlo.multiply %v1505, %v1519 : tensor<64x1152x7x7xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1523 = stablehlo.convolution(%v1522, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<64x192x7x7xf32>
    %v1524 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1525 = stablehlo.add %v1523, %v1524 : tensor<64x192x7x7xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1527 = stablehlo.reshape %v1526 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1529 = stablehlo.constant dense<3136.0> : tensor<64x192x7x7xf32>
    %v1530 = stablehlo.constant dense<1.0e-5> : tensor<64x192x7x7xf32>
    %v1531 = stablehlo.reduce(%v1527 init: %v1528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1532 = stablehlo.broadcast_in_dim %v1531, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1533 = stablehlo.divide %v1532, %v1529 : tensor<64x192x7x7xf32>
    %v1534 = stablehlo.subtract %v1527, %v1533 : tensor<64x192x7x7xf32>
    %v1535 = stablehlo.multiply %v1534, %v1534 : tensor<64x192x7x7xf32>
    %v1536 = stablehlo.reduce(%v1535 init: %v1528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1537 = stablehlo.broadcast_in_dim %v1536, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1538 = stablehlo.divide %v1537, %v1529 : tensor<64x192x7x7xf32>
    %v1539 = stablehlo.add %v1538, %v1530 : tensor<64x192x7x7xf32>
    %v1540 = stablehlo.rsqrt %v1539 : tensor<64x192x7x7xf32>
    %v1541 = stablehlo.multiply %v1534, %v1540 : tensor<64x192x7x7xf32>
    %v1542 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1543 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1544 = stablehlo.multiply %v1541, %v1542 : tensor<64x192x7x7xf32>
    %v1545 = stablehlo.add %v1544, %v1543 : tensor<64x192x7x7xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1547 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<64xf32>) -> tensor<64x9408xf32>
    %v1548 = stablehlo.multiply %v1547, %v1546 : tensor<64x9408xf32>
    %v1549 = stablehlo.add %v1548, %v1437 : tensor<64x9408xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1551 = stablehlo.convolution(%v1550, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<64x1152x7x7xf32>
    %v1552 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1553 = stablehlo.add %v1551, %v1552 : tensor<64x1152x7x7xf32>
    %v1554 = stablehlo.reshape %v1553 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1557 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1558 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1559 = stablehlo.reduce(%v1555 init: %v1556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1560 = stablehlo.broadcast_in_dim %v1559, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1561 = stablehlo.divide %v1560, %v1557 : tensor<64x1152x7x7xf32>
    %v1562 = stablehlo.subtract %v1555, %v1561 : tensor<64x1152x7x7xf32>
    %v1563 = stablehlo.multiply %v1562, %v1562 : tensor<64x1152x7x7xf32>
    %v1564 = stablehlo.reduce(%v1563 init: %v1556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1565 = stablehlo.broadcast_in_dim %v1564, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1566 = stablehlo.divide %v1565, %v1557 : tensor<64x1152x7x7xf32>
    %v1567 = stablehlo.add %v1566, %v1558 : tensor<64x1152x7x7xf32>
    %v1568 = stablehlo.rsqrt %v1567 : tensor<64x1152x7x7xf32>
    %v1569 = stablehlo.multiply %v1562, %v1568 : tensor<64x1152x7x7xf32>
    %v1570 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1571 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1572 = stablehlo.multiply %v1569, %v1570 : tensor<64x1152x7x7xf32>
    %v1573 = stablehlo.add %v1572, %v1571 : tensor<64x1152x7x7xf32>
    %v1574 = stablehlo.reshape %v1573 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1575 = stablehlo.logistic %v1574 : tensor<64x56448xf32>
    %v1576 = stablehlo.multiply %v1574, %v1575 : tensor<64x56448xf32>
    %v1577 = stablehlo.reshape %v1576 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1578 = stablehlo.convolution(%v1577, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<64x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<64x1152x7x7xf32>
    %v1579 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1580 = stablehlo.add %v1578, %v1579 : tensor<64x1152x7x7xf32>
    %v1581 = stablehlo.reshape %v1580 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1582 = stablehlo.reshape %v1581 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1584 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1585 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1586 = stablehlo.reduce(%v1582 init: %v1583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1587 = stablehlo.broadcast_in_dim %v1586, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1588 = stablehlo.divide %v1587, %v1584 : tensor<64x1152x7x7xf32>
    %v1589 = stablehlo.subtract %v1582, %v1588 : tensor<64x1152x7x7xf32>
    %v1590 = stablehlo.multiply %v1589, %v1589 : tensor<64x1152x7x7xf32>
    %v1591 = stablehlo.reduce(%v1590 init: %v1583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1592 = stablehlo.broadcast_in_dim %v1591, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1593 = stablehlo.divide %v1592, %v1584 : tensor<64x1152x7x7xf32>
    %v1594 = stablehlo.add %v1593, %v1585 : tensor<64x1152x7x7xf32>
    %v1595 = stablehlo.rsqrt %v1594 : tensor<64x1152x7x7xf32>
    %v1596 = stablehlo.multiply %v1589, %v1595 : tensor<64x1152x7x7xf32>
    %v1597 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1598 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1599 = stablehlo.multiply %v1596, %v1597 : tensor<64x1152x7x7xf32>
    %v1600 = stablehlo.add %v1599, %v1598 : tensor<64x1152x7x7xf32>
    %v1601 = stablehlo.reshape %v1600 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1602 = stablehlo.logistic %v1601 : tensor<64x56448xf32>
    %v1603 = stablehlo.multiply %v1601, %v1602 : tensor<64x56448xf32>
    %v1604 = stablehlo.reshape %v1603 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1605 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1606 = stablehlo.reduce(%v1604 init: %v1605) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1607 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1608 = stablehlo.divide %v1606, %v1607 : tensor<64x1152xf32>
    %v1609 = stablehlo.dot_general %v1608, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1610 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1611 = stablehlo.add %v1609, %v1610 : tensor<64x48xf32>
    %v1612 = stablehlo.logistic %v1611 : tensor<64x48xf32>
    %v1613 = stablehlo.multiply %v1611, %v1612 : tensor<64x48xf32>
    %v1614 = stablehlo.dot_general %v1613, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1615 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1616 = stablehlo.add %v1614, %v1615 : tensor<64x1152xf32>
    %v1617 = stablehlo.reshape %v1603 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1619 = stablehlo.reduce(%v1617 init: %v1618) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1620 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1621 = stablehlo.divide %v1619, %v1620 : tensor<64x1152xf32>
    %v1622 = stablehlo.dot_general %v1621, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1623 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1624 = stablehlo.add %v1622, %v1623 : tensor<64x48xf32>
    %v1625 = stablehlo.logistic %v1624 : tensor<64x48xf32>
    %v1626 = stablehlo.multiply %v1624, %v1625 : tensor<64x48xf32>
    %v1627 = stablehlo.dot_general %v1626, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1628 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1629 = stablehlo.add %v1627, %v1628 : tensor<64x1152xf32>
    %v1630 = stablehlo.logistic %v1629 : tensor<64x1152xf32>
    %v1631 = stablehlo.broadcast_in_dim %v1630, dims = [0, 1] : (tensor<64x1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1632 = stablehlo.multiply %v1617, %v1631 : tensor<64x1152x7x7xf32>
    %v1633 = stablehlo.reshape %v1632 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1635 = stablehlo.convolution(%v1634, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<64x192x7x7xf32>
    %v1636 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1637 = stablehlo.add %v1635, %v1636 : tensor<64x192x7x7xf32>
    %v1638 = stablehlo.reshape %v1637 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1641 = stablehlo.constant dense<3136.0> : tensor<64x192x7x7xf32>
    %v1642 = stablehlo.constant dense<1.0e-5> : tensor<64x192x7x7xf32>
    %v1643 = stablehlo.reduce(%v1639 init: %v1640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1644 = stablehlo.broadcast_in_dim %v1643, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1645 = stablehlo.divide %v1644, %v1641 : tensor<64x192x7x7xf32>
    %v1646 = stablehlo.subtract %v1639, %v1645 : tensor<64x192x7x7xf32>
    %v1647 = stablehlo.multiply %v1646, %v1646 : tensor<64x192x7x7xf32>
    %v1648 = stablehlo.reduce(%v1647 init: %v1640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1649 = stablehlo.broadcast_in_dim %v1648, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1650 = stablehlo.divide %v1649, %v1641 : tensor<64x192x7x7xf32>
    %v1651 = stablehlo.add %v1650, %v1642 : tensor<64x192x7x7xf32>
    %v1652 = stablehlo.rsqrt %v1651 : tensor<64x192x7x7xf32>
    %v1653 = stablehlo.multiply %v1646, %v1652 : tensor<64x192x7x7xf32>
    %v1654 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1655 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x7x7xf32>
    %v1656 = stablehlo.multiply %v1653, %v1654 : tensor<64x192x7x7xf32>
    %v1657 = stablehlo.add %v1656, %v1655 : tensor<64x192x7x7xf32>
    %v1658 = stablehlo.reshape %v1657 : (tensor<64x192x7x7xf32>) -> tensor<64x9408xf32>
    %v1659 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<64xf32>) -> tensor<64x9408xf32>
    %v1660 = stablehlo.multiply %v1659, %v1658 : tensor<64x9408xf32>
    %v1661 = stablehlo.add %v1660, %v1549 : tensor<64x9408xf32>
    %v1662 = stablehlo.reshape %v1661 : (tensor<64x9408xf32>) -> tensor<64x192x7x7xf32>
    %v1663 = stablehlo.convolution(%v1662, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<64x1152x7x7xf32>
    %v1664 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1665 = stablehlo.add %v1663, %v1664 : tensor<64x1152x7x7xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1667 = stablehlo.reshape %v1666 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1668 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1669 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1670 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1671 = stablehlo.reduce(%v1667 init: %v1668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1672 = stablehlo.broadcast_in_dim %v1671, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1673 = stablehlo.divide %v1672, %v1669 : tensor<64x1152x7x7xf32>
    %v1674 = stablehlo.subtract %v1667, %v1673 : tensor<64x1152x7x7xf32>
    %v1675 = stablehlo.multiply %v1674, %v1674 : tensor<64x1152x7x7xf32>
    %v1676 = stablehlo.reduce(%v1675 init: %v1668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1677 = stablehlo.broadcast_in_dim %v1676, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1678 = stablehlo.divide %v1677, %v1669 : tensor<64x1152x7x7xf32>
    %v1679 = stablehlo.add %v1678, %v1670 : tensor<64x1152x7x7xf32>
    %v1680 = stablehlo.rsqrt %v1679 : tensor<64x1152x7x7xf32>
    %v1681 = stablehlo.multiply %v1674, %v1680 : tensor<64x1152x7x7xf32>
    %v1682 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1683 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1684 = stablehlo.multiply %v1681, %v1682 : tensor<64x1152x7x7xf32>
    %v1685 = stablehlo.add %v1684, %v1683 : tensor<64x1152x7x7xf32>
    %v1686 = stablehlo.reshape %v1685 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1687 = stablehlo.logistic %v1686 : tensor<64x56448xf32>
    %v1688 = stablehlo.multiply %v1686, %v1687 : tensor<64x56448xf32>
    %v1689 = stablehlo.reshape %v1688 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1690 = stablehlo.convolution(%v1689, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<64x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<64x1152x7x7xf32>
    %v1691 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1692 = stablehlo.add %v1690, %v1691 : tensor<64x1152x7x7xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1694 = stablehlo.reshape %v1693 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1695 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1696 = stablehlo.constant dense<3136.0> : tensor<64x1152x7x7xf32>
    %v1697 = stablehlo.constant dense<1.0e-5> : tensor<64x1152x7x7xf32>
    %v1698 = stablehlo.reduce(%v1694 init: %v1695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1699 = stablehlo.broadcast_in_dim %v1698, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1700 = stablehlo.divide %v1699, %v1696 : tensor<64x1152x7x7xf32>
    %v1701 = stablehlo.subtract %v1694, %v1700 : tensor<64x1152x7x7xf32>
    %v1702 = stablehlo.multiply %v1701, %v1701 : tensor<64x1152x7x7xf32>
    %v1703 = stablehlo.reduce(%v1702 init: %v1695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1704 = stablehlo.broadcast_in_dim %v1703, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1705 = stablehlo.divide %v1704, %v1696 : tensor<64x1152x7x7xf32>
    %v1706 = stablehlo.add %v1705, %v1697 : tensor<64x1152x7x7xf32>
    %v1707 = stablehlo.rsqrt %v1706 : tensor<64x1152x7x7xf32>
    %v1708 = stablehlo.multiply %v1701, %v1707 : tensor<64x1152x7x7xf32>
    %v1709 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1710 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1711 = stablehlo.multiply %v1708, %v1709 : tensor<64x1152x7x7xf32>
    %v1712 = stablehlo.add %v1711, %v1710 : tensor<64x1152x7x7xf32>
    %v1713 = stablehlo.reshape %v1712 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1714 = stablehlo.logistic %v1713 : tensor<64x56448xf32>
    %v1715 = stablehlo.multiply %v1713, %v1714 : tensor<64x56448xf32>
    %v1716 = stablehlo.reshape %v1715 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1717 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1718 = stablehlo.reduce(%v1716 init: %v1717) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1719 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1720 = stablehlo.divide %v1718, %v1719 : tensor<64x1152xf32>
    %v1721 = stablehlo.dot_general %v1720, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1722 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1723 = stablehlo.add %v1721, %v1722 : tensor<64x48xf32>
    %v1724 = stablehlo.logistic %v1723 : tensor<64x48xf32>
    %v1725 = stablehlo.multiply %v1723, %v1724 : tensor<64x48xf32>
    %v1726 = stablehlo.dot_general %v1725, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1727 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1728 = stablehlo.add %v1726, %v1727 : tensor<64x1152xf32>
    %v1729 = stablehlo.reshape %v1715 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1731 = stablehlo.reduce(%v1729 init: %v1730) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1152x7x7xf32>, tensor<f32>) -> tensor<64x1152xf32>
    %v1732 = stablehlo.constant dense<49.0> : tensor<64x1152xf32>
    %v1733 = stablehlo.divide %v1731, %v1732 : tensor<64x1152xf32>
    %v1734 = stablehlo.dot_general %v1733, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1152xf32>, tensor<1152x48xf32>) -> tensor<64x48xf32>
    %v1735 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<64x48xf32>
    %v1736 = stablehlo.add %v1734, %v1735 : tensor<64x48xf32>
    %v1737 = stablehlo.logistic %v1736 : tensor<64x48xf32>
    %v1738 = stablehlo.multiply %v1736, %v1737 : tensor<64x48xf32>
    %v1739 = stablehlo.dot_general %v1738, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x48xf32>, tensor<48x1152xf32>) -> tensor<64x1152xf32>
    %v1740 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<64x1152xf32>
    %v1741 = stablehlo.add %v1739, %v1740 : tensor<64x1152xf32>
    %v1742 = stablehlo.logistic %v1741 : tensor<64x1152xf32>
    %v1743 = stablehlo.broadcast_in_dim %v1742, dims = [0, 1] : (tensor<64x1152xf32>) -> tensor<64x1152x7x7xf32>
    %v1744 = stablehlo.multiply %v1729, %v1743 : tensor<64x1152x7x7xf32>
    %v1745 = stablehlo.reshape %v1744 : (tensor<64x1152x7x7xf32>) -> tensor<64x56448xf32>
    %v1746 = stablehlo.reshape %v1745 : (tensor<64x56448xf32>) -> tensor<64x1152x7x7xf32>
    %v1747 = stablehlo.convolution(%v1746, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<64x320x7x7xf32>
    %v1748 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1749 = stablehlo.add %v1747, %v1748 : tensor<64x320x7x7xf32>
    %v1750 = stablehlo.reshape %v1749 : (tensor<64x320x7x7xf32>) -> tensor<64x15680xf32>
    %v1751 = stablehlo.reshape %v1750 : (tensor<64x15680xf32>) -> tensor<64x320x7x7xf32>
    %v1752 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1753 = stablehlo.constant dense<3136.0> : tensor<64x320x7x7xf32>
    %v1754 = stablehlo.constant dense<1.0e-5> : tensor<64x320x7x7xf32>
    %v1755 = stablehlo.reduce(%v1751 init: %v1752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1756 = stablehlo.broadcast_in_dim %v1755, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1757 = stablehlo.divide %v1756, %v1753 : tensor<64x320x7x7xf32>
    %v1758 = stablehlo.subtract %v1751, %v1757 : tensor<64x320x7x7xf32>
    %v1759 = stablehlo.multiply %v1758, %v1758 : tensor<64x320x7x7xf32>
    %v1760 = stablehlo.reduce(%v1759 init: %v1752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1761 = stablehlo.broadcast_in_dim %v1760, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1762 = stablehlo.divide %v1761, %v1753 : tensor<64x320x7x7xf32>
    %v1763 = stablehlo.add %v1762, %v1754 : tensor<64x320x7x7xf32>
    %v1764 = stablehlo.rsqrt %v1763 : tensor<64x320x7x7xf32>
    %v1765 = stablehlo.multiply %v1758, %v1764 : tensor<64x320x7x7xf32>
    %v1766 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1767 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1768 = stablehlo.multiply %v1765, %v1766 : tensor<64x320x7x7xf32>
    %v1769 = stablehlo.add %v1768, %v1767 : tensor<64x320x7x7xf32>
    %v1770 = stablehlo.reshape %v1769 : (tensor<64x320x7x7xf32>) -> tensor<64x15680xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<64x15680xf32>) -> tensor<64x320x7x7xf32>
    %v1772 = stablehlo.convolution(%v1771, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<64x1280x7x7xf32>
    %v1773 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1774 = stablehlo.add %v1772, %v1773 : tensor<64x1280x7x7xf32>
    %v1775 = stablehlo.reshape %v1774 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1776 = stablehlo.reshape %v1775 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1777 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1778 = stablehlo.constant dense<3136.0> : tensor<64x1280x7x7xf32>
    %v1779 = stablehlo.constant dense<1.0e-5> : tensor<64x1280x7x7xf32>
    %v1780 = stablehlo.reduce(%v1776 init: %v1777) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1781 = stablehlo.broadcast_in_dim %v1780, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1782 = stablehlo.divide %v1781, %v1778 : tensor<64x1280x7x7xf32>
    %v1783 = stablehlo.subtract %v1776, %v1782 : tensor<64x1280x7x7xf32>
    %v1784 = stablehlo.multiply %v1783, %v1783 : tensor<64x1280x7x7xf32>
    %v1785 = stablehlo.reduce(%v1784 init: %v1777) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1786 = stablehlo.broadcast_in_dim %v1785, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1787 = stablehlo.divide %v1786, %v1778 : tensor<64x1280x7x7xf32>
    %v1788 = stablehlo.add %v1787, %v1779 : tensor<64x1280x7x7xf32>
    %v1789 = stablehlo.rsqrt %v1788 : tensor<64x1280x7x7xf32>
    %v1790 = stablehlo.multiply %v1783, %v1789 : tensor<64x1280x7x7xf32>
    %v1791 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1792 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1793 = stablehlo.multiply %v1790, %v1791 : tensor<64x1280x7x7xf32>
    %v1794 = stablehlo.add %v1793, %v1792 : tensor<64x1280x7x7xf32>
    %v1795 = stablehlo.reshape %v1794 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1796 = stablehlo.logistic %v1795 : tensor<64x62720xf32>
    %v1797 = stablehlo.multiply %v1795, %v1796 : tensor<64x62720xf32>
    %v1798 = stablehlo.reshape %v1797 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1800 = stablehlo.reduce(%v1798 init: %v1799) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<64x1280xf32>
    %v1801 = stablehlo.constant dense<49.0> : tensor<64x1280xf32>
    %v1802 = stablehlo.divide %v1800, %v1801 : tensor<64x1280xf32>
    %v1803 = stablehlo.multiply %do, %v1802 : tensor<64x1280xf32>
    %v1804 = stablehlo.dot_general %v1803, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1280xf32>, tensor<1280x1000xf32>) -> tensor<64x1000xf32>
    %v1805 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1806 = stablehlo.add %v1804, %v1805 : tensor<64x1000xf32>
    return %v1806 : tensor<64x1000xf32>
  }
}
