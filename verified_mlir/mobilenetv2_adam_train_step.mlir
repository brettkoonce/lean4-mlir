module @m {
  func.func @mobilenetv2_adam_train_step(%x: tensor<32x150528xf32>, %sW: tensor<16x3x3x3xf32>, %sb: tensor<16xf32>, %sg: tensor<16xf32>, %sbt: tensor<16xf32>, %b1eW: tensor<64x16x1x1xf32>, %b1eb: tensor<64xf32>, %b1eg: tensor<64xf32>, %b1ebt: tensor<64xf32>, %b1dW: tensor<64x1x3x3xf32>, %b1db: tensor<64xf32>, %b1dg: tensor<64xf32>, %b1dbt: tensor<64xf32>, %b1pW: tensor<24x64x1x1xf32>, %b1pb: tensor<24xf32>, %b1pg: tensor<24xf32>, %b1pbt: tensor<24xf32>, %b2eW: tensor<96x24x1x1xf32>, %b2eb: tensor<96xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2db: tensor<96xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pb: tensor<24xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<96x24x1x1xf32>, %b3eb: tensor<96xf32>, %b3eg: tensor<96xf32>, %b3ebt: tensor<96xf32>, %b3dW: tensor<96x1x3x3xf32>, %b3db: tensor<96xf32>, %b3dg: tensor<96xf32>, %b3dbt: tensor<96xf32>, %b3pW: tensor<32x96x1x1xf32>, %b3pb: tensor<32xf32>, %b3pg: tensor<32xf32>, %b3pbt: tensor<32xf32>, %b4eW: tensor<128x32x1x1xf32>, %b4eb: tensor<128xf32>, %b4eg: tensor<128xf32>, %b4ebt: tensor<128xf32>, %b4dW: tensor<128x1x3x3xf32>, %b4db: tensor<128xf32>, %b4dg: tensor<128xf32>, %b4dbt: tensor<128xf32>, %b4pW: tensor<32x128x1x1xf32>, %b4pb: tensor<32xf32>, %b4pg: tensor<32xf32>, %b4pbt: tensor<32xf32>, %b5eW: tensor<128x32x1x1xf32>, %b5eb: tensor<128xf32>, %b5eg: tensor<128xf32>, %b5ebt: tensor<128xf32>, %b5dW: tensor<128x1x3x3xf32>, %b5db: tensor<128xf32>, %b5dg: tensor<128xf32>, %b5dbt: tensor<128xf32>, %b5pW: tensor<64x128x1x1xf32>, %b5pb: tensor<64xf32>, %b5pg: tensor<64xf32>, %b5pbt: tensor<64xf32>, %b6eW: tensor<256x64x1x1xf32>, %b6eb: tensor<256xf32>, %b6eg: tensor<256xf32>, %b6ebt: tensor<256xf32>, %b6dW: tensor<256x1x3x3xf32>, %b6db: tensor<256xf32>, %b6dg: tensor<256xf32>, %b6dbt: tensor<256xf32>, %b6pW: tensor<64x256x1x1xf32>, %b6pb: tensor<64xf32>, %b6pg: tensor<64xf32>, %b6pbt: tensor<64xf32>, %hW: tensor<128x64x1x1xf32>, %hb: tensor<128xf32>, %hg: tensor<128xf32>, %hbt: tensor<128xf32>, %Wd: tensor<128x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<16x3x3x3xf32>, %sbm: tensor<16xf32>, %sgm: tensor<16xf32>, %sbtm: tensor<16xf32>, %b1eWm: tensor<64x16x1x1xf32>, %b1ebm: tensor<64xf32>, %b1egm: tensor<64xf32>, %b1ebtm: tensor<64xf32>, %b1dWm: tensor<64x1x3x3xf32>, %b1dbm: tensor<64xf32>, %b1dgm: tensor<64xf32>, %b1dbtm: tensor<64xf32>, %b1pWm: tensor<24x64x1x1xf32>, %b1pbm: tensor<24xf32>, %b1pgm: tensor<24xf32>, %b1pbtm: tensor<24xf32>, %b2eWm: tensor<96x24x1x1xf32>, %b2ebm: tensor<96xf32>, %b2egm: tensor<96xf32>, %b2ebtm: tensor<96xf32>, %b2dWm: tensor<96x1x3x3xf32>, %b2dbm: tensor<96xf32>, %b2dgm: tensor<96xf32>, %b2dbtm: tensor<96xf32>, %b2pWm: tensor<24x96x1x1xf32>, %b2pbm: tensor<24xf32>, %b2pgm: tensor<24xf32>, %b2pbtm: tensor<24xf32>, %b3eWm: tensor<96x24x1x1xf32>, %b3ebm: tensor<96xf32>, %b3egm: tensor<96xf32>, %b3ebtm: tensor<96xf32>, %b3dWm: tensor<96x1x3x3xf32>, %b3dbm: tensor<96xf32>, %b3dgm: tensor<96xf32>, %b3dbtm: tensor<96xf32>, %b3pWm: tensor<32x96x1x1xf32>, %b3pbm: tensor<32xf32>, %b3pgm: tensor<32xf32>, %b3pbtm: tensor<32xf32>, %b4eWm: tensor<128x32x1x1xf32>, %b4ebm: tensor<128xf32>, %b4egm: tensor<128xf32>, %b4ebtm: tensor<128xf32>, %b4dWm: tensor<128x1x3x3xf32>, %b4dbm: tensor<128xf32>, %b4dgm: tensor<128xf32>, %b4dbtm: tensor<128xf32>, %b4pWm: tensor<32x128x1x1xf32>, %b4pbm: tensor<32xf32>, %b4pgm: tensor<32xf32>, %b4pbtm: tensor<32xf32>, %b5eWm: tensor<128x32x1x1xf32>, %b5ebm: tensor<128xf32>, %b5egm: tensor<128xf32>, %b5ebtm: tensor<128xf32>, %b5dWm: tensor<128x1x3x3xf32>, %b5dbm: tensor<128xf32>, %b5dgm: tensor<128xf32>, %b5dbtm: tensor<128xf32>, %b5pWm: tensor<64x128x1x1xf32>, %b5pbm: tensor<64xf32>, %b5pgm: tensor<64xf32>, %b5pbtm: tensor<64xf32>, %b6eWm: tensor<256x64x1x1xf32>, %b6ebm: tensor<256xf32>, %b6egm: tensor<256xf32>, %b6ebtm: tensor<256xf32>, %b6dWm: tensor<256x1x3x3xf32>, %b6dbm: tensor<256xf32>, %b6dgm: tensor<256xf32>, %b6dbtm: tensor<256xf32>, %b6pWm: tensor<64x256x1x1xf32>, %b6pbm: tensor<64xf32>, %b6pgm: tensor<64xf32>, %b6pbtm: tensor<64xf32>, %hWm: tensor<128x64x1x1xf32>, %hbm: tensor<128xf32>, %hgm: tensor<128xf32>, %hbtm: tensor<128xf32>, %Wdm: tensor<128x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<16x3x3x3xf32>, %sbv: tensor<16xf32>, %sgv: tensor<16xf32>, %sbtv: tensor<16xf32>, %b1eWv: tensor<64x16x1x1xf32>, %b1ebv: tensor<64xf32>, %b1egv: tensor<64xf32>, %b1ebtv: tensor<64xf32>, %b1dWv: tensor<64x1x3x3xf32>, %b1dbv: tensor<64xf32>, %b1dgv: tensor<64xf32>, %b1dbtv: tensor<64xf32>, %b1pWv: tensor<24x64x1x1xf32>, %b1pbv: tensor<24xf32>, %b1pgv: tensor<24xf32>, %b1pbtv: tensor<24xf32>, %b2eWv: tensor<96x24x1x1xf32>, %b2ebv: tensor<96xf32>, %b2egv: tensor<96xf32>, %b2ebtv: tensor<96xf32>, %b2dWv: tensor<96x1x3x3xf32>, %b2dbv: tensor<96xf32>, %b2dgv: tensor<96xf32>, %b2dbtv: tensor<96xf32>, %b2pWv: tensor<24x96x1x1xf32>, %b2pbv: tensor<24xf32>, %b2pgv: tensor<24xf32>, %b2pbtv: tensor<24xf32>, %b3eWv: tensor<96x24x1x1xf32>, %b3ebv: tensor<96xf32>, %b3egv: tensor<96xf32>, %b3ebtv: tensor<96xf32>, %b3dWv: tensor<96x1x3x3xf32>, %b3dbv: tensor<96xf32>, %b3dgv: tensor<96xf32>, %b3dbtv: tensor<96xf32>, %b3pWv: tensor<32x96x1x1xf32>, %b3pbv: tensor<32xf32>, %b3pgv: tensor<32xf32>, %b3pbtv: tensor<32xf32>, %b4eWv: tensor<128x32x1x1xf32>, %b4ebv: tensor<128xf32>, %b4egv: tensor<128xf32>, %b4ebtv: tensor<128xf32>, %b4dWv: tensor<128x1x3x3xf32>, %b4dbv: tensor<128xf32>, %b4dgv: tensor<128xf32>, %b4dbtv: tensor<128xf32>, %b4pWv: tensor<32x128x1x1xf32>, %b4pbv: tensor<32xf32>, %b4pgv: tensor<32xf32>, %b4pbtv: tensor<32xf32>, %b5eWv: tensor<128x32x1x1xf32>, %b5ebv: tensor<128xf32>, %b5egv: tensor<128xf32>, %b5ebtv: tensor<128xf32>, %b5dWv: tensor<128x1x3x3xf32>, %b5dbv: tensor<128xf32>, %b5dgv: tensor<128xf32>, %b5dbtv: tensor<128xf32>, %b5pWv: tensor<64x128x1x1xf32>, %b5pbv: tensor<64xf32>, %b5pgv: tensor<64xf32>, %b5pbtv: tensor<64xf32>, %b6eWv: tensor<256x64x1x1xf32>, %b6ebv: tensor<256xf32>, %b6egv: tensor<256xf32>, %b6ebtv: tensor<256xf32>, %b6dWv: tensor<256x1x3x3xf32>, %b6dbv: tensor<256xf32>, %b6dgv: tensor<256xf32>, %b6dbtv: tensor<256xf32>, %b6pWv: tensor<64x256x1x1xf32>, %b6pbv: tensor<64xf32>, %b6pgv: tensor<64xf32>, %b6pbtv: tensor<64xf32>, %hWv: tensor<128x64x1x1xf32>, %hbv: tensor<128xf32>, %hgv: tensor<128xf32>, %hbtv: tensor<128xf32>, %Wdv: tensor<128x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<32x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %bsc = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<16x3x3x3xf32>) -> tensor<32x16x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %sb, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x16x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<32x16x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<32x16x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<32x16x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<32x16x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<32x16x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<32x16x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<32x16x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<32x16x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<32x16x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v25 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v26 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v27 = stablehlo.maximum %v24, %v25 : tensor<32x200704xf32>
    %v28 = stablehlo.minimum %v27, %v26 : tensor<32x200704xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v30 = stablehlo.convolution(%v29, %b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<64x16x1x1xf32>) -> tensor<32x64x112x112xf32>
    %v31 = stablehlo.broadcast_in_dim %b1eb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<32x64x112x112xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v35 = stablehlo.constant dense<0.0> : tensor<f32>
    %v36 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v37 = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %v38 = stablehlo.reduce(%v34 init: %v35) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v39 = stablehlo.broadcast_in_dim %v38, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v40 = stablehlo.divide %v39, %v36 : tensor<32x64x112x112xf32>
    %v41 = stablehlo.subtract %v34, %v40 : tensor<32x64x112x112xf32>
    %v42 = stablehlo.multiply %v41, %v41 : tensor<32x64x112x112xf32>
    %v43 = stablehlo.reduce(%v42 init: %v35) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v44 = stablehlo.broadcast_in_dim %v43, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v45 = stablehlo.divide %v44, %v36 : tensor<32x64x112x112xf32>
    %v46 = stablehlo.add %v45, %v37 : tensor<32x64x112x112xf32>
    %v47 = stablehlo.rsqrt %v46 : tensor<32x64x112x112xf32>
    %v48 = stablehlo.multiply %v41, %v47 : tensor<32x64x112x112xf32>
    %v49 = stablehlo.broadcast_in_dim %b1eg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v50 = stablehlo.broadcast_in_dim %b1ebt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v51 = stablehlo.multiply %v48, %v49 : tensor<32x64x112x112xf32>
    %v52 = stablehlo.add %v51, %v50 : tensor<32x64x112x112xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v54 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v55 = stablehlo.constant dense<6.0> : tensor<32x802816xf32>
    %v56 = stablehlo.maximum %v53, %v54 : tensor<32x802816xf32>
    %v57 = stablehlo.minimum %v56, %v55 : tensor<32x802816xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v59 = stablehlo.convolution(%v58, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<32x64x112x112xf32>, tensor<64x1x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v60 = stablehlo.broadcast_in_dim %b1db, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<32x64x56x56xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v66 = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<32x64x56x56xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<32x64x56x56xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<32x64x56x56xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<32x64x56x56xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<32x64x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<32x64x56x56xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<32x64x56x56xf32>
    %v78 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<32x64x56x56xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<32x64x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v83 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v84 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v85 = stablehlo.maximum %v82, %v83 : tensor<32x200704xf32>
    %v86 = stablehlo.minimum %v85, %v84 : tensor<32x200704xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v88 = stablehlo.convolution(%v87, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<24x64x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v89 = stablehlo.broadcast_in_dim %b1pb, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v90 = stablehlo.add %v88, %v89 : tensor<32x24x56x56xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v93 = stablehlo.constant dense<0.0> : tensor<f32>
    %v94 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v95 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v96 = stablehlo.reduce(%v92 init: %v93) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v97 = stablehlo.broadcast_in_dim %v96, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v98 = stablehlo.divide %v97, %v94 : tensor<32x24x56x56xf32>
    %v99 = stablehlo.subtract %v92, %v98 : tensor<32x24x56x56xf32>
    %v100 = stablehlo.multiply %v99, %v99 : tensor<32x24x56x56xf32>
    %v101 = stablehlo.reduce(%v100 init: %v93) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v102 = stablehlo.broadcast_in_dim %v101, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v103 = stablehlo.divide %v102, %v94 : tensor<32x24x56x56xf32>
    %v104 = stablehlo.add %v103, %v95 : tensor<32x24x56x56xf32>
    %v105 = stablehlo.rsqrt %v104 : tensor<32x24x56x56xf32>
    %v106 = stablehlo.multiply %v99, %v105 : tensor<32x24x56x56xf32>
    %v107 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v108 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v109 = stablehlo.multiply %v106, %v107 : tensor<32x24x56x56xf32>
    %v110 = stablehlo.add %v109, %v108 : tensor<32x24x56x56xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v113 = stablehlo.convolution(%v112, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %b2eb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v115 = stablehlo.add %v113, %v114 : tensor<32x96x56x56xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v119 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v120 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v121 = stablehlo.reduce(%v117 init: %v118) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v122 = stablehlo.broadcast_in_dim %v121, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v123 = stablehlo.divide %v122, %v119 : tensor<32x96x56x56xf32>
    %v124 = stablehlo.subtract %v117, %v123 : tensor<32x96x56x56xf32>
    %v125 = stablehlo.multiply %v124, %v124 : tensor<32x96x56x56xf32>
    %v126 = stablehlo.reduce(%v125 init: %v118) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v127 = stablehlo.broadcast_in_dim %v126, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v128 = stablehlo.divide %v127, %v119 : tensor<32x96x56x56xf32>
    %v129 = stablehlo.add %v128, %v120 : tensor<32x96x56x56xf32>
    %v130 = stablehlo.rsqrt %v129 : tensor<32x96x56x56xf32>
    %v131 = stablehlo.multiply %v124, %v130 : tensor<32x96x56x56xf32>
    %v132 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v133 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v134 = stablehlo.multiply %v131, %v132 : tensor<32x96x56x56xf32>
    %v135 = stablehlo.add %v134, %v133 : tensor<32x96x56x56xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v137 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v138 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v139 = stablehlo.maximum %v136, %v137 : tensor<32x301056xf32>
    %v140 = stablehlo.minimum %v139, %v138 : tensor<32x301056xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v142 = stablehlo.convolution(%v141, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v143 = stablehlo.broadcast_in_dim %b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v144 = stablehlo.add %v142, %v143 : tensor<32x96x56x56xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v148 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v149 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v150 = stablehlo.reduce(%v146 init: %v147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v152 = stablehlo.divide %v151, %v148 : tensor<32x96x56x56xf32>
    %v153 = stablehlo.subtract %v146, %v152 : tensor<32x96x56x56xf32>
    %v154 = stablehlo.multiply %v153, %v153 : tensor<32x96x56x56xf32>
    %v155 = stablehlo.reduce(%v154 init: %v147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v156 = stablehlo.broadcast_in_dim %v155, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v157 = stablehlo.divide %v156, %v148 : tensor<32x96x56x56xf32>
    %v158 = stablehlo.add %v157, %v149 : tensor<32x96x56x56xf32>
    %v159 = stablehlo.rsqrt %v158 : tensor<32x96x56x56xf32>
    %v160 = stablehlo.multiply %v153, %v159 : tensor<32x96x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v163 = stablehlo.multiply %v160, %v161 : tensor<32x96x56x56xf32>
    %v164 = stablehlo.add %v163, %v162 : tensor<32x96x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v166 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v167 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v168 = stablehlo.maximum %v165, %v166 : tensor<32x301056xf32>
    %v169 = stablehlo.minimum %v168, %v167 : tensor<32x301056xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v171 = stablehlo.convolution(%v170, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %b2pb, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v173 = stablehlo.add %v171, %v172 : tensor<32x24x56x56xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v177 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v178 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v179 = stablehlo.reduce(%v175 init: %v176) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v180 = stablehlo.broadcast_in_dim %v179, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v181 = stablehlo.divide %v180, %v177 : tensor<32x24x56x56xf32>
    %v182 = stablehlo.subtract %v175, %v181 : tensor<32x24x56x56xf32>
    %v183 = stablehlo.multiply %v182, %v182 : tensor<32x24x56x56xf32>
    %v184 = stablehlo.reduce(%v183 init: %v176) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v185 = stablehlo.broadcast_in_dim %v184, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v186 = stablehlo.divide %v185, %v177 : tensor<32x24x56x56xf32>
    %v187 = stablehlo.add %v186, %v178 : tensor<32x24x56x56xf32>
    %v188 = stablehlo.rsqrt %v187 : tensor<32x24x56x56xf32>
    %v189 = stablehlo.multiply %v182, %v188 : tensor<32x24x56x56xf32>
    %v190 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v191 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v192 = stablehlo.multiply %v189, %v190 : tensor<32x24x56x56xf32>
    %v193 = stablehlo.add %v192, %v191 : tensor<32x24x56x56xf32>
    %v194 = stablehlo.reshape %v193 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v195 = stablehlo.add %v194, %v111 : tensor<32x75264xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v197 = stablehlo.convolution(%v196, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v198 = stablehlo.broadcast_in_dim %b3eb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v199 = stablehlo.add %v197, %v198 : tensor<32x96x56x56xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v203 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v204 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v205 = stablehlo.reduce(%v201 init: %v202) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v206 = stablehlo.broadcast_in_dim %v205, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v207 = stablehlo.divide %v206, %v203 : tensor<32x96x56x56xf32>
    %v208 = stablehlo.subtract %v201, %v207 : tensor<32x96x56x56xf32>
    %v209 = stablehlo.multiply %v208, %v208 : tensor<32x96x56x56xf32>
    %v210 = stablehlo.reduce(%v209 init: %v202) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v212 = stablehlo.divide %v211, %v203 : tensor<32x96x56x56xf32>
    %v213 = stablehlo.add %v212, %v204 : tensor<32x96x56x56xf32>
    %v214 = stablehlo.rsqrt %v213 : tensor<32x96x56x56xf32>
    %v215 = stablehlo.multiply %v208, %v214 : tensor<32x96x56x56xf32>
    %v216 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v217 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v218 = stablehlo.multiply %v215, %v216 : tensor<32x96x56x56xf32>
    %v219 = stablehlo.add %v218, %v217 : tensor<32x96x56x56xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v221 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v222 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v223 = stablehlo.maximum %v220, %v221 : tensor<32x301056xf32>
    %v224 = stablehlo.minimum %v223, %v222 : tensor<32x301056xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v226 = stablehlo.convolution(%v225, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x28x28xf32>
    %v227 = stablehlo.broadcast_in_dim %b3db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v228 = stablehlo.add %v226, %v227 : tensor<32x96x28x28xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v232 = stablehlo.constant dense<784.0> : tensor<32x96x28x28xf32>
    %v233 = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %v234 = stablehlo.reduce(%v230 init: %v231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v235 = stablehlo.broadcast_in_dim %v234, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v236 = stablehlo.divide %v235, %v232 : tensor<32x96x28x28xf32>
    %v237 = stablehlo.subtract %v230, %v236 : tensor<32x96x28x28xf32>
    %v238 = stablehlo.multiply %v237, %v237 : tensor<32x96x28x28xf32>
    %v239 = stablehlo.reduce(%v238 init: %v231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v240 = stablehlo.broadcast_in_dim %v239, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v241 = stablehlo.divide %v240, %v232 : tensor<32x96x28x28xf32>
    %v242 = stablehlo.add %v241, %v233 : tensor<32x96x28x28xf32>
    %v243 = stablehlo.rsqrt %v242 : tensor<32x96x28x28xf32>
    %v244 = stablehlo.multiply %v237, %v243 : tensor<32x96x28x28xf32>
    %v245 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v246 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v247 = stablehlo.multiply %v244, %v245 : tensor<32x96x28x28xf32>
    %v248 = stablehlo.add %v247, %v246 : tensor<32x96x28x28xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v250 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v251 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v252 = stablehlo.maximum %v249, %v250 : tensor<32x75264xf32>
    %v253 = stablehlo.minimum %v252, %v251 : tensor<32x75264xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v255 = stablehlo.convolution(%v254, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x28x28xf32>, tensor<32x96x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v256 = stablehlo.broadcast_in_dim %b3pb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v257 = stablehlo.add %v255, %v256 : tensor<32x32x28x28xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v261 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v262 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v263 = stablehlo.reduce(%v259 init: %v260) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v264 = stablehlo.broadcast_in_dim %v263, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v265 = stablehlo.divide %v264, %v261 : tensor<32x32x28x28xf32>
    %v266 = stablehlo.subtract %v259, %v265 : tensor<32x32x28x28xf32>
    %v267 = stablehlo.multiply %v266, %v266 : tensor<32x32x28x28xf32>
    %v268 = stablehlo.reduce(%v267 init: %v260) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v269 = stablehlo.broadcast_in_dim %v268, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v270 = stablehlo.divide %v269, %v261 : tensor<32x32x28x28xf32>
    %v271 = stablehlo.add %v270, %v262 : tensor<32x32x28x28xf32>
    %v272 = stablehlo.rsqrt %v271 : tensor<32x32x28x28xf32>
    %v273 = stablehlo.multiply %v266, %v272 : tensor<32x32x28x28xf32>
    %v274 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v275 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v276 = stablehlo.multiply %v273, %v274 : tensor<32x32x28x28xf32>
    %v277 = stablehlo.add %v276, %v275 : tensor<32x32x28x28xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v280 = stablehlo.convolution(%v279, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v281 = stablehlo.broadcast_in_dim %b4eb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v282 = stablehlo.add %v280, %v281 : tensor<32x128x28x28xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v286 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v287 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v288 = stablehlo.reduce(%v284 init: %v285) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v289 = stablehlo.broadcast_in_dim %v288, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v290 = stablehlo.divide %v289, %v286 : tensor<32x128x28x28xf32>
    %v291 = stablehlo.subtract %v284, %v290 : tensor<32x128x28x28xf32>
    %v292 = stablehlo.multiply %v291, %v291 : tensor<32x128x28x28xf32>
    %v293 = stablehlo.reduce(%v292 init: %v285) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v294 = stablehlo.broadcast_in_dim %v293, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v295 = stablehlo.divide %v294, %v286 : tensor<32x128x28x28xf32>
    %v296 = stablehlo.add %v295, %v287 : tensor<32x128x28x28xf32>
    %v297 = stablehlo.rsqrt %v296 : tensor<32x128x28x28xf32>
    %v298 = stablehlo.multiply %v291, %v297 : tensor<32x128x28x28xf32>
    %v299 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v300 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v301 = stablehlo.multiply %v298, %v299 : tensor<32x128x28x28xf32>
    %v302 = stablehlo.add %v301, %v300 : tensor<32x128x28x28xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v304 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v305 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v306 = stablehlo.maximum %v303, %v304 : tensor<32x100352xf32>
    %v307 = stablehlo.minimum %v306, %v305 : tensor<32x100352xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v309 = stablehlo.convolution(%v308, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v310 = stablehlo.broadcast_in_dim %b4db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v311 = stablehlo.add %v309, %v310 : tensor<32x128x28x28xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v315 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v316 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v317 = stablehlo.reduce(%v313 init: %v314) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v318 = stablehlo.broadcast_in_dim %v317, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v319 = stablehlo.divide %v318, %v315 : tensor<32x128x28x28xf32>
    %v320 = stablehlo.subtract %v313, %v319 : tensor<32x128x28x28xf32>
    %v321 = stablehlo.multiply %v320, %v320 : tensor<32x128x28x28xf32>
    %v322 = stablehlo.reduce(%v321 init: %v314) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v323 = stablehlo.broadcast_in_dim %v322, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v324 = stablehlo.divide %v323, %v315 : tensor<32x128x28x28xf32>
    %v325 = stablehlo.add %v324, %v316 : tensor<32x128x28x28xf32>
    %v326 = stablehlo.rsqrt %v325 : tensor<32x128x28x28xf32>
    %v327 = stablehlo.multiply %v320, %v326 : tensor<32x128x28x28xf32>
    %v328 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v330 = stablehlo.multiply %v327, %v328 : tensor<32x128x28x28xf32>
    %v331 = stablehlo.add %v330, %v329 : tensor<32x128x28x28xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v333 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v334 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v335 = stablehlo.maximum %v332, %v333 : tensor<32x100352xf32>
    %v336 = stablehlo.minimum %v335, %v334 : tensor<32x100352xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v338 = stablehlo.convolution(%v337, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v339 = stablehlo.broadcast_in_dim %b4pb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<32x32x28x28xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v344 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v345 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v346 = stablehlo.reduce(%v342 init: %v343) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v347 = stablehlo.broadcast_in_dim %v346, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v348 = stablehlo.divide %v347, %v344 : tensor<32x32x28x28xf32>
    %v349 = stablehlo.subtract %v342, %v348 : tensor<32x32x28x28xf32>
    %v350 = stablehlo.multiply %v349, %v349 : tensor<32x32x28x28xf32>
    %v351 = stablehlo.reduce(%v350 init: %v343) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v352 = stablehlo.broadcast_in_dim %v351, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v353 = stablehlo.divide %v352, %v344 : tensor<32x32x28x28xf32>
    %v354 = stablehlo.add %v353, %v345 : tensor<32x32x28x28xf32>
    %v355 = stablehlo.rsqrt %v354 : tensor<32x32x28x28xf32>
    %v356 = stablehlo.multiply %v349, %v355 : tensor<32x32x28x28xf32>
    %v357 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v358 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v359 = stablehlo.multiply %v356, %v357 : tensor<32x32x28x28xf32>
    %v360 = stablehlo.add %v359, %v358 : tensor<32x32x28x28xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v362 = stablehlo.add %v361, %v278 : tensor<32x25088xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v364 = stablehlo.convolution(%v363, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v365 = stablehlo.broadcast_in_dim %b5eb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v366 = stablehlo.add %v364, %v365 : tensor<32x128x28x28xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v369 = stablehlo.constant dense<0.0> : tensor<f32>
    %v370 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v371 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v372 = stablehlo.reduce(%v368 init: %v369) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v373 = stablehlo.broadcast_in_dim %v372, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v374 = stablehlo.divide %v373, %v370 : tensor<32x128x28x28xf32>
    %v375 = stablehlo.subtract %v368, %v374 : tensor<32x128x28x28xf32>
    %v376 = stablehlo.multiply %v375, %v375 : tensor<32x128x28x28xf32>
    %v377 = stablehlo.reduce(%v376 init: %v369) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v378 = stablehlo.broadcast_in_dim %v377, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v379 = stablehlo.divide %v378, %v370 : tensor<32x128x28x28xf32>
    %v380 = stablehlo.add %v379, %v371 : tensor<32x128x28x28xf32>
    %v381 = stablehlo.rsqrt %v380 : tensor<32x128x28x28xf32>
    %v382 = stablehlo.multiply %v375, %v381 : tensor<32x128x28x28xf32>
    %v383 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v384 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v385 = stablehlo.multiply %v382, %v383 : tensor<32x128x28x28xf32>
    %v386 = stablehlo.add %v385, %v384 : tensor<32x128x28x28xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v388 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v389 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v390 = stablehlo.maximum %v387, %v388 : tensor<32x100352xf32>
    %v391 = stablehlo.minimum %v390, %v389 : tensor<32x100352xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v393 = stablehlo.convolution(%v392, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x14x14xf32>
    %v394 = stablehlo.broadcast_in_dim %b5db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v395 = stablehlo.add %v393, %v394 : tensor<32x128x14x14xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v399 = stablehlo.constant dense<196.0> : tensor<32x128x14x14xf32>
    %v400 = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %v401 = stablehlo.reduce(%v397 init: %v398) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v402 = stablehlo.broadcast_in_dim %v401, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v403 = stablehlo.divide %v402, %v399 : tensor<32x128x14x14xf32>
    %v404 = stablehlo.subtract %v397, %v403 : tensor<32x128x14x14xf32>
    %v405 = stablehlo.multiply %v404, %v404 : tensor<32x128x14x14xf32>
    %v406 = stablehlo.reduce(%v405 init: %v398) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v407 = stablehlo.broadcast_in_dim %v406, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v408 = stablehlo.divide %v407, %v399 : tensor<32x128x14x14xf32>
    %v409 = stablehlo.add %v408, %v400 : tensor<32x128x14x14xf32>
    %v410 = stablehlo.rsqrt %v409 : tensor<32x128x14x14xf32>
    %v411 = stablehlo.multiply %v404, %v410 : tensor<32x128x14x14xf32>
    %v412 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v413 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v414 = stablehlo.multiply %v411, %v412 : tensor<32x128x14x14xf32>
    %v415 = stablehlo.add %v414, %v413 : tensor<32x128x14x14xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v417 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v418 = stablehlo.constant dense<6.0> : tensor<32x25088xf32>
    %v419 = stablehlo.maximum %v416, %v417 : tensor<32x25088xf32>
    %v420 = stablehlo.minimum %v419, %v418 : tensor<32x25088xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v422 = stablehlo.convolution(%v421, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x14x14xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v423 = stablehlo.broadcast_in_dim %b5pb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v424 = stablehlo.add %v422, %v423 : tensor<32x64x14x14xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v428 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v429 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v430 = stablehlo.reduce(%v426 init: %v427) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v431 = stablehlo.broadcast_in_dim %v430, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v432 = stablehlo.divide %v431, %v428 : tensor<32x64x14x14xf32>
    %v433 = stablehlo.subtract %v426, %v432 : tensor<32x64x14x14xf32>
    %v434 = stablehlo.multiply %v433, %v433 : tensor<32x64x14x14xf32>
    %v435 = stablehlo.reduce(%v434 init: %v427) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v436 = stablehlo.broadcast_in_dim %v435, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v437 = stablehlo.divide %v436, %v428 : tensor<32x64x14x14xf32>
    %v438 = stablehlo.add %v437, %v429 : tensor<32x64x14x14xf32>
    %v439 = stablehlo.rsqrt %v438 : tensor<32x64x14x14xf32>
    %v440 = stablehlo.multiply %v433, %v439 : tensor<32x64x14x14xf32>
    %v441 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v442 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v443 = stablehlo.multiply %v440, %v441 : tensor<32x64x14x14xf32>
    %v444 = stablehlo.add %v443, %v442 : tensor<32x64x14x14xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v447 = stablehlo.convolution(%v446, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v448 = stablehlo.broadcast_in_dim %b6eb, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v449 = stablehlo.add %v447, %v448 : tensor<32x256x14x14xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v453 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v454 = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %v455 = stablehlo.reduce(%v451 init: %v452) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v456 = stablehlo.broadcast_in_dim %v455, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v457 = stablehlo.divide %v456, %v453 : tensor<32x256x14x14xf32>
    %v458 = stablehlo.subtract %v451, %v457 : tensor<32x256x14x14xf32>
    %v459 = stablehlo.multiply %v458, %v458 : tensor<32x256x14x14xf32>
    %v460 = stablehlo.reduce(%v459 init: %v452) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v461 = stablehlo.broadcast_in_dim %v460, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v462 = stablehlo.divide %v461, %v453 : tensor<32x256x14x14xf32>
    %v463 = stablehlo.add %v462, %v454 : tensor<32x256x14x14xf32>
    %v464 = stablehlo.rsqrt %v463 : tensor<32x256x14x14xf32>
    %v465 = stablehlo.multiply %v458, %v464 : tensor<32x256x14x14xf32>
    %v466 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v467 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v468 = stablehlo.multiply %v465, %v466 : tensor<32x256x14x14xf32>
    %v469 = stablehlo.add %v468, %v467 : tensor<32x256x14x14xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v471 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v472 = stablehlo.constant dense<6.0> : tensor<32x50176xf32>
    %v473 = stablehlo.maximum %v470, %v471 : tensor<32x50176xf32>
    %v474 = stablehlo.minimum %v473, %v472 : tensor<32x50176xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v476 = stablehlo.convolution(%v475, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x14x14xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v477 = stablehlo.broadcast_in_dim %b6db, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v478 = stablehlo.add %v476, %v477 : tensor<32x256x7x7xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v482 = stablehlo.constant dense<49.0> : tensor<32x256x7x7xf32>
    %v483 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v484 = stablehlo.reduce(%v480 init: %v481) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v485 = stablehlo.broadcast_in_dim %v484, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v486 = stablehlo.divide %v485, %v482 : tensor<32x256x7x7xf32>
    %v487 = stablehlo.subtract %v480, %v486 : tensor<32x256x7x7xf32>
    %v488 = stablehlo.multiply %v487, %v487 : tensor<32x256x7x7xf32>
    %v489 = stablehlo.reduce(%v488 init: %v481) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v490 = stablehlo.broadcast_in_dim %v489, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v491 = stablehlo.divide %v490, %v482 : tensor<32x256x7x7xf32>
    %v492 = stablehlo.add %v491, %v483 : tensor<32x256x7x7xf32>
    %v493 = stablehlo.rsqrt %v492 : tensor<32x256x7x7xf32>
    %v494 = stablehlo.multiply %v487, %v493 : tensor<32x256x7x7xf32>
    %v495 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v496 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v497 = stablehlo.multiply %v494, %v495 : tensor<32x256x7x7xf32>
    %v498 = stablehlo.add %v497, %v496 : tensor<32x256x7x7xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v500 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v501 = stablehlo.constant dense<6.0> : tensor<32x12544xf32>
    %v502 = stablehlo.maximum %v499, %v500 : tensor<32x12544xf32>
    %v503 = stablehlo.minimum %v502, %v501 : tensor<32x12544xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v505 = stablehlo.convolution(%v504, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<64x256x1x1xf32>) -> tensor<32x64x7x7xf32>
    %v506 = stablehlo.broadcast_in_dim %b6pb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v507 = stablehlo.add %v505, %v506 : tensor<32x64x7x7xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v510 = stablehlo.constant dense<0.0> : tensor<f32>
    %v511 = stablehlo.constant dense<49.0> : tensor<32x64x7x7xf32>
    %v512 = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %v513 = stablehlo.reduce(%v509 init: %v510) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v514 = stablehlo.broadcast_in_dim %v513, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v515 = stablehlo.divide %v514, %v511 : tensor<32x64x7x7xf32>
    %v516 = stablehlo.subtract %v509, %v515 : tensor<32x64x7x7xf32>
    %v517 = stablehlo.multiply %v516, %v516 : tensor<32x64x7x7xf32>
    %v518 = stablehlo.reduce(%v517 init: %v510) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v519 = stablehlo.broadcast_in_dim %v518, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v520 = stablehlo.divide %v519, %v511 : tensor<32x64x7x7xf32>
    %v521 = stablehlo.add %v520, %v512 : tensor<32x64x7x7xf32>
    %v522 = stablehlo.rsqrt %v521 : tensor<32x64x7x7xf32>
    %v523 = stablehlo.multiply %v516, %v522 : tensor<32x64x7x7xf32>
    %v524 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v525 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v526 = stablehlo.multiply %v523, %v524 : tensor<32x64x7x7xf32>
    %v527 = stablehlo.add %v526, %v525 : tensor<32x64x7x7xf32>
    %v528 = stablehlo.reshape %v527 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v530 = stablehlo.convolution(%v529, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x7x7xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x7x7xf32>
    %v531 = stablehlo.broadcast_in_dim %hb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<32x128x7x7xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v535 = stablehlo.constant dense<0.0> : tensor<f32>
    %v536 = stablehlo.constant dense<49.0> : tensor<32x128x7x7xf32>
    %v537 = stablehlo.constant dense<1.0e-5> : tensor<32x128x7x7xf32>
    %v538 = stablehlo.reduce(%v534 init: %v535) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v539 = stablehlo.broadcast_in_dim %v538, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v540 = stablehlo.divide %v539, %v536 : tensor<32x128x7x7xf32>
    %v541 = stablehlo.subtract %v534, %v540 : tensor<32x128x7x7xf32>
    %v542 = stablehlo.multiply %v541, %v541 : tensor<32x128x7x7xf32>
    %v543 = stablehlo.reduce(%v542 init: %v535) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v544 = stablehlo.broadcast_in_dim %v543, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v545 = stablehlo.divide %v544, %v536 : tensor<32x128x7x7xf32>
    %v546 = stablehlo.add %v545, %v537 : tensor<32x128x7x7xf32>
    %v547 = stablehlo.rsqrt %v546 : tensor<32x128x7x7xf32>
    %v548 = stablehlo.multiply %v541, %v547 : tensor<32x128x7x7xf32>
    %v549 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v550 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v551 = stablehlo.multiply %v548, %v549 : tensor<32x128x7x7xf32>
    %v552 = stablehlo.add %v551, %v550 : tensor<32x128x7x7xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<32x6272xf32>
    %v555 = stablehlo.constant dense<6.0> : tensor<32x6272xf32>
    %v556 = stablehlo.maximum %v553, %v554 : tensor<32x6272xf32>
    %v557 = stablehlo.minimum %v556, %v555 : tensor<32x6272xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v560 = stablehlo.reduce(%v558 init: %v559) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v561 = stablehlo.constant dense<49.0> : tensor<32x128xf32>
    %v562 = stablehlo.divide %v560, %v561 : tensor<32x128xf32>
    %v563 = stablehlo.dot_general %v562, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x128xf32>, tensor<128x10xf32>) -> tensor<32x10xf32>
    %v564 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v565 = stablehlo.add %v563, %v564 : tensor<32x10xf32>
    %v566 = stablehlo.exponential %v565 : tensor<32x10xf32>
    %v567 = stablehlo.constant dense<0.0> : tensor<f32>
    %v568 = stablehlo.reduce(%v566 init: %v567) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v569 = stablehlo.broadcast_in_dim %v568, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v570 = stablehlo.divide %v566, %v569 : tensor<32x10xf32>
    %dyr0 = stablehlo.subtract %v570, %onehot : tensor<32x10xf32>
    %lsa = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %lsaoh = stablehlo.multiply %lsa, %onehot : tensor<32x10xf32>
    %dyr1 = stablehlo.add %dyr0, %lsaoh : tensor<32x10xf32>
    %lsaik = stablehlo.constant dense<0.010000> : tensor<32x10xf32>
    %dyr = stablehlo.subtract %dyr1, %lsaik : tensor<32x10xf32>
    %dy = stablehlo.divide %dyr, %bsc : tensor<32x10xf32>
    %llog = stablehlo.log %v570 : tensor<32x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<32x10xf32>
    %t1s = stablehlo.reduce(%ohll init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %lls = stablehlo.reduce(%llog init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %omac = stablehlo.constant dense<0.900000> : tensor<32xf32>
    %aKc = stablehlo.constant dense<0.010000> : tensor<32xf32>
    %lt1 = stablehlo.multiply %omac, %t1s : tensor<32xf32>
    %lt2 = stablehlo.multiply %aKc, %lls : tensor<32xf32>
    %lpe = stablehlo.add %lt1, %lt2 : tensor<32xf32>
    %lsum2 = stablehlo.reduce(%lpe init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %lbfc = stablehlo.constant dense<32.0> : tensor<f32>
    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>
    %loss = stablehlo.negate %lossm : tensor<f32>
    %v571 = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<128x10xf32>) -> tensor<32x128xf32>
    %dgi = stablehlo.reshape %v571 : (tensor<32x128xf32>) -> tensor<32x128x1x1xf32>
    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<32x128x7x7xf32>
    %dgn = stablehlo.constant dense<49.0> : tensor<32x128x7x7xf32>
    %dgd = stablehlo.divide %dgb, %dgn : tensor<32x128x7x7xf32>
    %dgapf = stablehlo.reshape %dgd : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v572 = stablehlo.constant dense<0.0> : tensor<32x6272xf32>
    %v573 = stablehlo.constant dense<6.0> : tensor<32x6272xf32>
    %v574 = stablehlo.compare GT, %v553, %v572 : (tensor<32x6272xf32>, tensor<32x6272xf32>) -> tensor<32x6272xi1>
    %v575 = stablehlo.compare LT, %v553, %v573 : (tensor<32x6272xf32>, tensor<32x6272xf32>) -> tensor<32x6272xi1>
    %v576 = stablehlo.and %v574, %v575 : tensor<32x6272xi1>
    %v577 = stablehlo.select %v576, %dgapf, %v572 : tensor<32x6272xi1>, tensor<32x6272xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v579 = stablehlo.reshape %v533 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v581 = stablehlo.constant dense<49.0> : tensor<32x128x7x7xf32>
    %v582 = stablehlo.constant dense<1.0e-5> : tensor<32x128x7x7xf32>
    %v583 = stablehlo.reduce(%v579 init: %v580) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v584 = stablehlo.broadcast_in_dim %v583, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v585 = stablehlo.divide %v584, %v581 : tensor<32x128x7x7xf32>
    %v586 = stablehlo.subtract %v579, %v585 : tensor<32x128x7x7xf32>
    %v587 = stablehlo.multiply %v586, %v586 : tensor<32x128x7x7xf32>
    %v588 = stablehlo.reduce(%v587 init: %v580) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v589 = stablehlo.broadcast_in_dim %v588, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v590 = stablehlo.divide %v589, %v581 : tensor<32x128x7x7xf32>
    %v591 = stablehlo.add %v590, %v582 : tensor<32x128x7x7xf32>
    %v592 = stablehlo.rsqrt %v591 : tensor<32x128x7x7xf32>
    %v593 = stablehlo.multiply %v586, %v592 : tensor<32x128x7x7xf32>
    %v594 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v595 = stablehlo.multiply %v594, %v578 : tensor<32x128x7x7xf32>
    %v596 = stablehlo.reduce(%v595 init: %v580) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v597 = stablehlo.broadcast_in_dim %v596, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v598 = stablehlo.multiply %v593, %v595 : tensor<32x128x7x7xf32>
    %v599 = stablehlo.reduce(%v598 init: %v580) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v600 = stablehlo.broadcast_in_dim %v599, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v601 = stablehlo.multiply %v595, %v581 : tensor<32x128x7x7xf32>
    %v602 = stablehlo.subtract %v601, %v597 : tensor<32x128x7x7xf32>
    %v603 = stablehlo.multiply %v593, %v600 : tensor<32x128x7x7xf32>
    %v604 = stablehlo.subtract %v602, %v603 : tensor<32x128x7x7xf32>
    %v605 = stablehlo.divide %v592, %v581 : tensor<32x128x7x7xf32>
    %v606 = stablehlo.multiply %v605, %v604 : tensor<32x128x7x7xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v609 = stablehlo.transpose %hW, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v610 = stablehlo.reverse %v609, dims = [2, 3] : tensor<64x128x1x1xf32>
    %v611 = stablehlo.convolution(%v608, %v610)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x7x7xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x7x7xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v614 = stablehlo.reshape %v508 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v616 = stablehlo.constant dense<49.0> : tensor<32x64x7x7xf32>
    %v617 = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %v618 = stablehlo.reduce(%v614 init: %v615) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v619 = stablehlo.broadcast_in_dim %v618, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v620 = stablehlo.divide %v619, %v616 : tensor<32x64x7x7xf32>
    %v621 = stablehlo.subtract %v614, %v620 : tensor<32x64x7x7xf32>
    %v622 = stablehlo.multiply %v621, %v621 : tensor<32x64x7x7xf32>
    %v623 = stablehlo.reduce(%v622 init: %v615) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v624 = stablehlo.broadcast_in_dim %v623, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v625 = stablehlo.divide %v624, %v616 : tensor<32x64x7x7xf32>
    %v626 = stablehlo.add %v625, %v617 : tensor<32x64x7x7xf32>
    %v627 = stablehlo.rsqrt %v626 : tensor<32x64x7x7xf32>
    %v628 = stablehlo.multiply %v621, %v627 : tensor<32x64x7x7xf32>
    %v629 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v630 = stablehlo.multiply %v629, %v613 : tensor<32x64x7x7xf32>
    %v631 = stablehlo.reduce(%v630 init: %v615) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v632 = stablehlo.broadcast_in_dim %v631, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v633 = stablehlo.multiply %v628, %v630 : tensor<32x64x7x7xf32>
    %v634 = stablehlo.reduce(%v633 init: %v615) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v635 = stablehlo.broadcast_in_dim %v634, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v636 = stablehlo.multiply %v630, %v616 : tensor<32x64x7x7xf32>
    %v637 = stablehlo.subtract %v636, %v632 : tensor<32x64x7x7xf32>
    %v638 = stablehlo.multiply %v628, %v635 : tensor<32x64x7x7xf32>
    %v639 = stablehlo.subtract %v637, %v638 : tensor<32x64x7x7xf32>
    %v640 = stablehlo.divide %v627, %v616 : tensor<32x64x7x7xf32>
    %v641 = stablehlo.multiply %v640, %v639 : tensor<32x64x7x7xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v644 = stablehlo.transpose %b6pW, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v645 = stablehlo.reverse %v644, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v646 = stablehlo.convolution(%v643, %v645)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x7x7xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v648 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v649 = stablehlo.constant dense<6.0> : tensor<32x12544xf32>
    %v650 = stablehlo.compare GT, %v499, %v648 : (tensor<32x12544xf32>, tensor<32x12544xf32>) -> tensor<32x12544xi1>
    %v651 = stablehlo.compare LT, %v499, %v649 : (tensor<32x12544xf32>, tensor<32x12544xf32>) -> tensor<32x12544xi1>
    %v652 = stablehlo.and %v650, %v651 : tensor<32x12544xi1>
    %v653 = stablehlo.select %v652, %v647, %v648 : tensor<32x12544xi1>, tensor<32x12544xf32>
    %v654 = stablehlo.reshape %v653 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v655 = stablehlo.reshape %v479 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v657 = stablehlo.constant dense<49.0> : tensor<32x256x7x7xf32>
    %v658 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v659 = stablehlo.reduce(%v655 init: %v656) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v660 = stablehlo.broadcast_in_dim %v659, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v661 = stablehlo.divide %v660, %v657 : tensor<32x256x7x7xf32>
    %v662 = stablehlo.subtract %v655, %v661 : tensor<32x256x7x7xf32>
    %v663 = stablehlo.multiply %v662, %v662 : tensor<32x256x7x7xf32>
    %v664 = stablehlo.reduce(%v663 init: %v656) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v665 = stablehlo.broadcast_in_dim %v664, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v666 = stablehlo.divide %v665, %v657 : tensor<32x256x7x7xf32>
    %v667 = stablehlo.add %v666, %v658 : tensor<32x256x7x7xf32>
    %v668 = stablehlo.rsqrt %v667 : tensor<32x256x7x7xf32>
    %v669 = stablehlo.multiply %v662, %v668 : tensor<32x256x7x7xf32>
    %v670 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v671 = stablehlo.multiply %v670, %v654 : tensor<32x256x7x7xf32>
    %v672 = stablehlo.reduce(%v671 init: %v656) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v673 = stablehlo.broadcast_in_dim %v672, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v674 = stablehlo.multiply %v669, %v671 : tensor<32x256x7x7xf32>
    %v675 = stablehlo.reduce(%v674 init: %v656) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v676 = stablehlo.broadcast_in_dim %v675, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v677 = stablehlo.multiply %v671, %v657 : tensor<32x256x7x7xf32>
    %v678 = stablehlo.subtract %v677, %v673 : tensor<32x256x7x7xf32>
    %v679 = stablehlo.multiply %v669, %v676 : tensor<32x256x7x7xf32>
    %v680 = stablehlo.subtract %v678, %v679 : tensor<32x256x7x7xf32>
    %v681 = stablehlo.divide %v668, %v657 : tensor<32x256x7x7xf32>
    %v682 = stablehlo.multiply %v681, %v680 : tensor<32x256x7x7xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v685 = stablehlo.constant dense<0.0> : tensor<f32>
    %v686 = stablehlo.pad %v684, %v685, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256x14x14xf32>
    %v687 = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<256x1x3x3xf32>
    %v688 = stablehlo.convolution(%v686, %v687)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x14x14xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v690 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v691 = stablehlo.constant dense<6.0> : tensor<32x50176xf32>
    %v692 = stablehlo.compare GT, %v470, %v690 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v693 = stablehlo.compare LT, %v470, %v691 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v694 = stablehlo.and %v692, %v693 : tensor<32x50176xi1>
    %v695 = stablehlo.select %v694, %v689, %v690 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v697 = stablehlo.reshape %v450 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v699 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v700 = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %v701 = stablehlo.reduce(%v697 init: %v698) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v702 = stablehlo.broadcast_in_dim %v701, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v703 = stablehlo.divide %v702, %v699 : tensor<32x256x14x14xf32>
    %v704 = stablehlo.subtract %v697, %v703 : tensor<32x256x14x14xf32>
    %v705 = stablehlo.multiply %v704, %v704 : tensor<32x256x14x14xf32>
    %v706 = stablehlo.reduce(%v705 init: %v698) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v707 = stablehlo.broadcast_in_dim %v706, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v708 = stablehlo.divide %v707, %v699 : tensor<32x256x14x14xf32>
    %v709 = stablehlo.add %v708, %v700 : tensor<32x256x14x14xf32>
    %v710 = stablehlo.rsqrt %v709 : tensor<32x256x14x14xf32>
    %v711 = stablehlo.multiply %v704, %v710 : tensor<32x256x14x14xf32>
    %v712 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v713 = stablehlo.multiply %v712, %v696 : tensor<32x256x14x14xf32>
    %v714 = stablehlo.reduce(%v713 init: %v698) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v715 = stablehlo.broadcast_in_dim %v714, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v716 = stablehlo.multiply %v711, %v713 : tensor<32x256x14x14xf32>
    %v717 = stablehlo.reduce(%v716 init: %v698) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v718 = stablehlo.broadcast_in_dim %v717, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v719 = stablehlo.multiply %v713, %v699 : tensor<32x256x14x14xf32>
    %v720 = stablehlo.subtract %v719, %v715 : tensor<32x256x14x14xf32>
    %v721 = stablehlo.multiply %v711, %v718 : tensor<32x256x14x14xf32>
    %v722 = stablehlo.subtract %v720, %v721 : tensor<32x256x14x14xf32>
    %v723 = stablehlo.divide %v710, %v699 : tensor<32x256x14x14xf32>
    %v724 = stablehlo.multiply %v723, %v722 : tensor<32x256x14x14xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v727 = stablehlo.transpose %b6eW, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v728 = stablehlo.reverse %v727, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v729 = stablehlo.convolution(%v726, %v728)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<64x256x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v732 = stablehlo.reshape %v425 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v734 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v735 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v736 = stablehlo.reduce(%v732 init: %v733) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v737 = stablehlo.broadcast_in_dim %v736, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v738 = stablehlo.divide %v737, %v734 : tensor<32x64x14x14xf32>
    %v739 = stablehlo.subtract %v732, %v738 : tensor<32x64x14x14xf32>
    %v740 = stablehlo.multiply %v739, %v739 : tensor<32x64x14x14xf32>
    %v741 = stablehlo.reduce(%v740 init: %v733) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v742 = stablehlo.broadcast_in_dim %v741, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v743 = stablehlo.divide %v742, %v734 : tensor<32x64x14x14xf32>
    %v744 = stablehlo.add %v743, %v735 : tensor<32x64x14x14xf32>
    %v745 = stablehlo.rsqrt %v744 : tensor<32x64x14x14xf32>
    %v746 = stablehlo.multiply %v739, %v745 : tensor<32x64x14x14xf32>
    %v747 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v748 = stablehlo.multiply %v747, %v731 : tensor<32x64x14x14xf32>
    %v749 = stablehlo.reduce(%v748 init: %v733) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v750 = stablehlo.broadcast_in_dim %v749, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v751 = stablehlo.multiply %v746, %v748 : tensor<32x64x14x14xf32>
    %v752 = stablehlo.reduce(%v751 init: %v733) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v753 = stablehlo.broadcast_in_dim %v752, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v754 = stablehlo.multiply %v748, %v734 : tensor<32x64x14x14xf32>
    %v755 = stablehlo.subtract %v754, %v750 : tensor<32x64x14x14xf32>
    %v756 = stablehlo.multiply %v746, %v753 : tensor<32x64x14x14xf32>
    %v757 = stablehlo.subtract %v755, %v756 : tensor<32x64x14x14xf32>
    %v758 = stablehlo.divide %v745, %v734 : tensor<32x64x14x14xf32>
    %v759 = stablehlo.multiply %v758, %v757 : tensor<32x64x14x14xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v762 = stablehlo.transpose %b5pW, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v763 = stablehlo.reverse %v762, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v764 = stablehlo.convolution(%v761, %v763)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x14x14xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v766 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v767 = stablehlo.constant dense<6.0> : tensor<32x25088xf32>
    %v768 = stablehlo.compare GT, %v416, %v766 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v769 = stablehlo.compare LT, %v416, %v767 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v770 = stablehlo.and %v768, %v769 : tensor<32x25088xi1>
    %v771 = stablehlo.select %v770, %v765, %v766 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v773 = stablehlo.reshape %v396 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v775 = stablehlo.constant dense<196.0> : tensor<32x128x14x14xf32>
    %v776 = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %v777 = stablehlo.reduce(%v773 init: %v774) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v778 = stablehlo.broadcast_in_dim %v777, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v779 = stablehlo.divide %v778, %v775 : tensor<32x128x14x14xf32>
    %v780 = stablehlo.subtract %v773, %v779 : tensor<32x128x14x14xf32>
    %v781 = stablehlo.multiply %v780, %v780 : tensor<32x128x14x14xf32>
    %v782 = stablehlo.reduce(%v781 init: %v774) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v783 = stablehlo.broadcast_in_dim %v782, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v784 = stablehlo.divide %v783, %v775 : tensor<32x128x14x14xf32>
    %v785 = stablehlo.add %v784, %v776 : tensor<32x128x14x14xf32>
    %v786 = stablehlo.rsqrt %v785 : tensor<32x128x14x14xf32>
    %v787 = stablehlo.multiply %v780, %v786 : tensor<32x128x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v789 = stablehlo.multiply %v788, %v772 : tensor<32x128x14x14xf32>
    %v790 = stablehlo.reduce(%v789 init: %v774) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v791 = stablehlo.broadcast_in_dim %v790, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v792 = stablehlo.multiply %v787, %v789 : tensor<32x128x14x14xf32>
    %v793 = stablehlo.reduce(%v792 init: %v774) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v794 = stablehlo.broadcast_in_dim %v793, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v795 = stablehlo.multiply %v789, %v775 : tensor<32x128x14x14xf32>
    %v796 = stablehlo.subtract %v795, %v791 : tensor<32x128x14x14xf32>
    %v797 = stablehlo.multiply %v787, %v794 : tensor<32x128x14x14xf32>
    %v798 = stablehlo.subtract %v796, %v797 : tensor<32x128x14x14xf32>
    %v799 = stablehlo.divide %v786, %v775 : tensor<32x128x14x14xf32>
    %v800 = stablehlo.multiply %v799, %v798 : tensor<32x128x14x14xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v804 = stablehlo.pad %v802, %v803, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128x28x28xf32>
    %v805 = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<128x1x3x3xf32>
    %v806 = stablehlo.convolution(%v804, %v805)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v808 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v809 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v810 = stablehlo.compare GT, %v387, %v808 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v811 = stablehlo.compare LT, %v387, %v809 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v812 = stablehlo.and %v810, %v811 : tensor<32x100352xi1>
    %v813 = stablehlo.select %v812, %v807, %v808 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v815 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v817 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v818 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v819 = stablehlo.reduce(%v815 init: %v816) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v820 = stablehlo.broadcast_in_dim %v819, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v821 = stablehlo.divide %v820, %v817 : tensor<32x128x28x28xf32>
    %v822 = stablehlo.subtract %v815, %v821 : tensor<32x128x28x28xf32>
    %v823 = stablehlo.multiply %v822, %v822 : tensor<32x128x28x28xf32>
    %v824 = stablehlo.reduce(%v823 init: %v816) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v825 = stablehlo.broadcast_in_dim %v824, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v826 = stablehlo.divide %v825, %v817 : tensor<32x128x28x28xf32>
    %v827 = stablehlo.add %v826, %v818 : tensor<32x128x28x28xf32>
    %v828 = stablehlo.rsqrt %v827 : tensor<32x128x28x28xf32>
    %v829 = stablehlo.multiply %v822, %v828 : tensor<32x128x28x28xf32>
    %v830 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v831 = stablehlo.multiply %v830, %v814 : tensor<32x128x28x28xf32>
    %v832 = stablehlo.reduce(%v831 init: %v816) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v833 = stablehlo.broadcast_in_dim %v832, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v834 = stablehlo.multiply %v829, %v831 : tensor<32x128x28x28xf32>
    %v835 = stablehlo.reduce(%v834 init: %v816) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v836 = stablehlo.broadcast_in_dim %v835, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v837 = stablehlo.multiply %v831, %v817 : tensor<32x128x28x28xf32>
    %v838 = stablehlo.subtract %v837, %v833 : tensor<32x128x28x28xf32>
    %v839 = stablehlo.multiply %v829, %v836 : tensor<32x128x28x28xf32>
    %v840 = stablehlo.subtract %v838, %v839 : tensor<32x128x28x28xf32>
    %v841 = stablehlo.divide %v828, %v817 : tensor<32x128x28x28xf32>
    %v842 = stablehlo.multiply %v841, %v840 : tensor<32x128x28x28xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v845 = stablehlo.transpose %b5eW, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v846 = stablehlo.reverse %v845, dims = [2, 3] : tensor<32x128x1x1xf32>
    %v847 = stablehlo.convolution(%v844, %v846)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v850 = stablehlo.reshape %v341 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v852 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v853 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v854 = stablehlo.reduce(%v850 init: %v851) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v855 = stablehlo.broadcast_in_dim %v854, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v856 = stablehlo.divide %v855, %v852 : tensor<32x32x28x28xf32>
    %v857 = stablehlo.subtract %v850, %v856 : tensor<32x32x28x28xf32>
    %v858 = stablehlo.multiply %v857, %v857 : tensor<32x32x28x28xf32>
    %v859 = stablehlo.reduce(%v858 init: %v851) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v860 = stablehlo.broadcast_in_dim %v859, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v861 = stablehlo.divide %v860, %v852 : tensor<32x32x28x28xf32>
    %v862 = stablehlo.add %v861, %v853 : tensor<32x32x28x28xf32>
    %v863 = stablehlo.rsqrt %v862 : tensor<32x32x28x28xf32>
    %v864 = stablehlo.multiply %v857, %v863 : tensor<32x32x28x28xf32>
    %v865 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v866 = stablehlo.multiply %v865, %v849 : tensor<32x32x28x28xf32>
    %v867 = stablehlo.reduce(%v866 init: %v851) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v868 = stablehlo.broadcast_in_dim %v867, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v869 = stablehlo.multiply %v864, %v866 : tensor<32x32x28x28xf32>
    %v870 = stablehlo.reduce(%v869 init: %v851) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v871 = stablehlo.broadcast_in_dim %v870, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v872 = stablehlo.multiply %v866, %v852 : tensor<32x32x28x28xf32>
    %v873 = stablehlo.subtract %v872, %v868 : tensor<32x32x28x28xf32>
    %v874 = stablehlo.multiply %v864, %v871 : tensor<32x32x28x28xf32>
    %v875 = stablehlo.subtract %v873, %v874 : tensor<32x32x28x28xf32>
    %v876 = stablehlo.divide %v863, %v852 : tensor<32x32x28x28xf32>
    %v877 = stablehlo.multiply %v876, %v875 : tensor<32x32x28x28xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v880 = stablehlo.transpose %b4pW, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %v881 = stablehlo.reverse %v880, dims = [2, 3] : tensor<128x32x1x1xf32>
    %v882 = stablehlo.convolution(%v879, %v881)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v884 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v885 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v886 = stablehlo.compare GT, %v332, %v884 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v887 = stablehlo.compare LT, %v332, %v885 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v888 = stablehlo.and %v886, %v887 : tensor<32x100352xi1>
    %v889 = stablehlo.select %v888, %v883, %v884 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v891 = stablehlo.reshape %v312 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v892 = stablehlo.constant dense<0.0> : tensor<f32>
    %v893 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v894 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v895 = stablehlo.reduce(%v891 init: %v892) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v896 = stablehlo.broadcast_in_dim %v895, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v897 = stablehlo.divide %v896, %v893 : tensor<32x128x28x28xf32>
    %v898 = stablehlo.subtract %v891, %v897 : tensor<32x128x28x28xf32>
    %v899 = stablehlo.multiply %v898, %v898 : tensor<32x128x28x28xf32>
    %v900 = stablehlo.reduce(%v899 init: %v892) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v901 = stablehlo.broadcast_in_dim %v900, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v902 = stablehlo.divide %v901, %v893 : tensor<32x128x28x28xf32>
    %v903 = stablehlo.add %v902, %v894 : tensor<32x128x28x28xf32>
    %v904 = stablehlo.rsqrt %v903 : tensor<32x128x28x28xf32>
    %v905 = stablehlo.multiply %v898, %v904 : tensor<32x128x28x28xf32>
    %v906 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v907 = stablehlo.multiply %v906, %v890 : tensor<32x128x28x28xf32>
    %v908 = stablehlo.reduce(%v907 init: %v892) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v909 = stablehlo.broadcast_in_dim %v908, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v910 = stablehlo.multiply %v905, %v907 : tensor<32x128x28x28xf32>
    %v911 = stablehlo.reduce(%v910 init: %v892) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v912 = stablehlo.broadcast_in_dim %v911, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v913 = stablehlo.multiply %v907, %v893 : tensor<32x128x28x28xf32>
    %v914 = stablehlo.subtract %v913, %v909 : tensor<32x128x28x28xf32>
    %v915 = stablehlo.multiply %v905, %v912 : tensor<32x128x28x28xf32>
    %v916 = stablehlo.subtract %v914, %v915 : tensor<32x128x28x28xf32>
    %v917 = stablehlo.divide %v904, %v893 : tensor<32x128x28x28xf32>
    %v918 = stablehlo.multiply %v917, %v916 : tensor<32x128x28x28xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v921 = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<128x1x3x3xf32>
    %v922 = stablehlo.convolution(%v920, %v921)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v924 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v925 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v926 = stablehlo.compare GT, %v303, %v924 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v927 = stablehlo.compare LT, %v303, %v925 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v928 = stablehlo.and %v926, %v927 : tensor<32x100352xi1>
    %v929 = stablehlo.select %v928, %v923, %v924 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v931 = stablehlo.reshape %v283 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v933 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v934 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v935 = stablehlo.reduce(%v931 init: %v932) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v936 = stablehlo.broadcast_in_dim %v935, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v937 = stablehlo.divide %v936, %v933 : tensor<32x128x28x28xf32>
    %v938 = stablehlo.subtract %v931, %v937 : tensor<32x128x28x28xf32>
    %v939 = stablehlo.multiply %v938, %v938 : tensor<32x128x28x28xf32>
    %v940 = stablehlo.reduce(%v939 init: %v932) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v941 = stablehlo.broadcast_in_dim %v940, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v942 = stablehlo.divide %v941, %v933 : tensor<32x128x28x28xf32>
    %v943 = stablehlo.add %v942, %v934 : tensor<32x128x28x28xf32>
    %v944 = stablehlo.rsqrt %v943 : tensor<32x128x28x28xf32>
    %v945 = stablehlo.multiply %v938, %v944 : tensor<32x128x28x28xf32>
    %v946 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v947 = stablehlo.multiply %v946, %v930 : tensor<32x128x28x28xf32>
    %v948 = stablehlo.reduce(%v947 init: %v932) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v949 = stablehlo.broadcast_in_dim %v948, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v950 = stablehlo.multiply %v945, %v947 : tensor<32x128x28x28xf32>
    %v951 = stablehlo.reduce(%v950 init: %v932) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v952 = stablehlo.broadcast_in_dim %v951, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v953 = stablehlo.multiply %v947, %v933 : tensor<32x128x28x28xf32>
    %v954 = stablehlo.subtract %v953, %v949 : tensor<32x128x28x28xf32>
    %v955 = stablehlo.multiply %v945, %v952 : tensor<32x128x28x28xf32>
    %v956 = stablehlo.subtract %v954, %v955 : tensor<32x128x28x28xf32>
    %v957 = stablehlo.divide %v944, %v933 : tensor<32x128x28x28xf32>
    %v958 = stablehlo.multiply %v957, %v956 : tensor<32x128x28x28xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v961 = stablehlo.transpose %b4eW, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v962 = stablehlo.reverse %v961, dims = [2, 3] : tensor<32x128x1x1xf32>
    %v963 = stablehlo.convolution(%v960, %v962)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v965 = stablehlo.add %v964, %v848 : tensor<32x25088xf32>
    %v966 = stablehlo.reshape %v965 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v967 = stablehlo.reshape %v258 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v969 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v970 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v971 = stablehlo.reduce(%v967 init: %v968) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v972 = stablehlo.broadcast_in_dim %v971, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v973 = stablehlo.divide %v972, %v969 : tensor<32x32x28x28xf32>
    %v974 = stablehlo.subtract %v967, %v973 : tensor<32x32x28x28xf32>
    %v975 = stablehlo.multiply %v974, %v974 : tensor<32x32x28x28xf32>
    %v976 = stablehlo.reduce(%v975 init: %v968) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v977 = stablehlo.broadcast_in_dim %v976, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v978 = stablehlo.divide %v977, %v969 : tensor<32x32x28x28xf32>
    %v979 = stablehlo.add %v978, %v970 : tensor<32x32x28x28xf32>
    %v980 = stablehlo.rsqrt %v979 : tensor<32x32x28x28xf32>
    %v981 = stablehlo.multiply %v974, %v980 : tensor<32x32x28x28xf32>
    %v982 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v983 = stablehlo.multiply %v982, %v966 : tensor<32x32x28x28xf32>
    %v984 = stablehlo.reduce(%v983 init: %v968) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v985 = stablehlo.broadcast_in_dim %v984, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v986 = stablehlo.multiply %v981, %v983 : tensor<32x32x28x28xf32>
    %v987 = stablehlo.reduce(%v986 init: %v968) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v988 = stablehlo.broadcast_in_dim %v987, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v989 = stablehlo.multiply %v983, %v969 : tensor<32x32x28x28xf32>
    %v990 = stablehlo.subtract %v989, %v985 : tensor<32x32x28x28xf32>
    %v991 = stablehlo.multiply %v981, %v988 : tensor<32x32x28x28xf32>
    %v992 = stablehlo.subtract %v990, %v991 : tensor<32x32x28x28xf32>
    %v993 = stablehlo.divide %v980, %v969 : tensor<32x32x28x28xf32>
    %v994 = stablehlo.multiply %v993, %v992 : tensor<32x32x28x28xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v997 = stablehlo.transpose %b3pW, dims = [1, 0, 2, 3] : (tensor<32x96x1x1xf32>) -> tensor<96x32x1x1xf32>
    %v998 = stablehlo.reverse %v997, dims = [2, 3] : tensor<96x32x1x1xf32>
    %v999 = stablehlo.convolution(%v996, %v998)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<96x32x1x1xf32>) -> tensor<32x96x28x28xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v1001 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v1002 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v1003 = stablehlo.compare GT, %v249, %v1001 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v1004 = stablehlo.compare LT, %v249, %v1002 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v1005 = stablehlo.and %v1003, %v1004 : tensor<32x75264xi1>
    %v1006 = stablehlo.select %v1005, %v1000, %v1001 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1008 = stablehlo.reshape %v229 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1010 = stablehlo.constant dense<784.0> : tensor<32x96x28x28xf32>
    %v1011 = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %v1012 = stablehlo.reduce(%v1008 init: %v1009) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1013 = stablehlo.broadcast_in_dim %v1012, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1014 = stablehlo.divide %v1013, %v1010 : tensor<32x96x28x28xf32>
    %v1015 = stablehlo.subtract %v1008, %v1014 : tensor<32x96x28x28xf32>
    %v1016 = stablehlo.multiply %v1015, %v1015 : tensor<32x96x28x28xf32>
    %v1017 = stablehlo.reduce(%v1016 init: %v1009) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1018 = stablehlo.broadcast_in_dim %v1017, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1019 = stablehlo.divide %v1018, %v1010 : tensor<32x96x28x28xf32>
    %v1020 = stablehlo.add %v1019, %v1011 : tensor<32x96x28x28xf32>
    %v1021 = stablehlo.rsqrt %v1020 : tensor<32x96x28x28xf32>
    %v1022 = stablehlo.multiply %v1015, %v1021 : tensor<32x96x28x28xf32>
    %v1023 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v1024 = stablehlo.multiply %v1023, %v1007 : tensor<32x96x28x28xf32>
    %v1025 = stablehlo.reduce(%v1024 init: %v1009) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1026 = stablehlo.broadcast_in_dim %v1025, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1027 = stablehlo.multiply %v1022, %v1024 : tensor<32x96x28x28xf32>
    %v1028 = stablehlo.reduce(%v1027 init: %v1009) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1029 = stablehlo.broadcast_in_dim %v1028, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1030 = stablehlo.multiply %v1024, %v1010 : tensor<32x96x28x28xf32>
    %v1031 = stablehlo.subtract %v1030, %v1026 : tensor<32x96x28x28xf32>
    %v1032 = stablehlo.multiply %v1022, %v1029 : tensor<32x96x28x28xf32>
    %v1033 = stablehlo.subtract %v1031, %v1032 : tensor<32x96x28x28xf32>
    %v1034 = stablehlo.divide %v1021, %v1010 : tensor<32x96x28x28xf32>
    %v1035 = stablehlo.multiply %v1034, %v1033 : tensor<32x96x28x28xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1039 = stablehlo.pad %v1037, %v1038, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96x56x56xf32>
    %v1040 = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v1041 = stablehlo.convolution(%v1039, %v1040)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1043 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v1044 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v1045 = stablehlo.compare GT, %v220, %v1043 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1046 = stablehlo.compare LT, %v220, %v1044 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1047 = stablehlo.and %v1045, %v1046 : tensor<32x301056xi1>
    %v1048 = stablehlo.select %v1047, %v1042, %v1043 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1050 = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1051 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1052 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1053 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1054 = stablehlo.reduce(%v1050 init: %v1051) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1055 = stablehlo.broadcast_in_dim %v1054, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1056 = stablehlo.divide %v1055, %v1052 : tensor<32x96x56x56xf32>
    %v1057 = stablehlo.subtract %v1050, %v1056 : tensor<32x96x56x56xf32>
    %v1058 = stablehlo.multiply %v1057, %v1057 : tensor<32x96x56x56xf32>
    %v1059 = stablehlo.reduce(%v1058 init: %v1051) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1060 = stablehlo.broadcast_in_dim %v1059, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1061 = stablehlo.divide %v1060, %v1052 : tensor<32x96x56x56xf32>
    %v1062 = stablehlo.add %v1061, %v1053 : tensor<32x96x56x56xf32>
    %v1063 = stablehlo.rsqrt %v1062 : tensor<32x96x56x56xf32>
    %v1064 = stablehlo.multiply %v1057, %v1063 : tensor<32x96x56x56xf32>
    %v1065 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v1066 = stablehlo.multiply %v1065, %v1049 : tensor<32x96x56x56xf32>
    %v1067 = stablehlo.reduce(%v1066 init: %v1051) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1068 = stablehlo.broadcast_in_dim %v1067, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1069 = stablehlo.multiply %v1064, %v1066 : tensor<32x96x56x56xf32>
    %v1070 = stablehlo.reduce(%v1069 init: %v1051) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1071 = stablehlo.broadcast_in_dim %v1070, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1072 = stablehlo.multiply %v1066, %v1052 : tensor<32x96x56x56xf32>
    %v1073 = stablehlo.subtract %v1072, %v1068 : tensor<32x96x56x56xf32>
    %v1074 = stablehlo.multiply %v1064, %v1071 : tensor<32x96x56x56xf32>
    %v1075 = stablehlo.subtract %v1073, %v1074 : tensor<32x96x56x56xf32>
    %v1076 = stablehlo.divide %v1063, %v1052 : tensor<32x96x56x56xf32>
    %v1077 = stablehlo.multiply %v1076, %v1075 : tensor<32x96x56x56xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1080 = stablehlo.transpose %b3eW, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v1081 = stablehlo.reverse %v1080, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v1082 = stablehlo.convolution(%v1079, %v1081)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1085 = stablehlo.reshape %v174 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1086 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1087 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v1088 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v1089 = stablehlo.reduce(%v1085 init: %v1086) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1090 = stablehlo.broadcast_in_dim %v1089, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1091 = stablehlo.divide %v1090, %v1087 : tensor<32x24x56x56xf32>
    %v1092 = stablehlo.subtract %v1085, %v1091 : tensor<32x24x56x56xf32>
    %v1093 = stablehlo.multiply %v1092, %v1092 : tensor<32x24x56x56xf32>
    %v1094 = stablehlo.reduce(%v1093 init: %v1086) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1095 = stablehlo.broadcast_in_dim %v1094, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1096 = stablehlo.divide %v1095, %v1087 : tensor<32x24x56x56xf32>
    %v1097 = stablehlo.add %v1096, %v1088 : tensor<32x24x56x56xf32>
    %v1098 = stablehlo.rsqrt %v1097 : tensor<32x24x56x56xf32>
    %v1099 = stablehlo.multiply %v1092, %v1098 : tensor<32x24x56x56xf32>
    %v1100 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v1101 = stablehlo.multiply %v1100, %v1084 : tensor<32x24x56x56xf32>
    %v1102 = stablehlo.reduce(%v1101 init: %v1086) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1103 = stablehlo.broadcast_in_dim %v1102, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1104 = stablehlo.multiply %v1099, %v1101 : tensor<32x24x56x56xf32>
    %v1105 = stablehlo.reduce(%v1104 init: %v1086) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1106 = stablehlo.broadcast_in_dim %v1105, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1107 = stablehlo.multiply %v1101, %v1087 : tensor<32x24x56x56xf32>
    %v1108 = stablehlo.subtract %v1107, %v1103 : tensor<32x24x56x56xf32>
    %v1109 = stablehlo.multiply %v1099, %v1106 : tensor<32x24x56x56xf32>
    %v1110 = stablehlo.subtract %v1108, %v1109 : tensor<32x24x56x56xf32>
    %v1111 = stablehlo.divide %v1098, %v1087 : tensor<32x24x56x56xf32>
    %v1112 = stablehlo.multiply %v1111, %v1110 : tensor<32x24x56x56xf32>
    %v1113 = stablehlo.reshape %v1112 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1115 = stablehlo.transpose %b2pW, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v1116 = stablehlo.reverse %v1115, dims = [2, 3] : tensor<96x24x1x1xf32>
    %v1117 = stablehlo.convolution(%v1114, %v1116)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v1118 = stablehlo.reshape %v1117 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1119 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v1120 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v1121 = stablehlo.compare GT, %v165, %v1119 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1122 = stablehlo.compare LT, %v165, %v1120 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1123 = stablehlo.and %v1121, %v1122 : tensor<32x301056xi1>
    %v1124 = stablehlo.select %v1123, %v1118, %v1119 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1126 = stablehlo.reshape %v145 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1128 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1129 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1130 = stablehlo.reduce(%v1126 init: %v1127) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1131 = stablehlo.broadcast_in_dim %v1130, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1132 = stablehlo.divide %v1131, %v1128 : tensor<32x96x56x56xf32>
    %v1133 = stablehlo.subtract %v1126, %v1132 : tensor<32x96x56x56xf32>
    %v1134 = stablehlo.multiply %v1133, %v1133 : tensor<32x96x56x56xf32>
    %v1135 = stablehlo.reduce(%v1134 init: %v1127) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1136 = stablehlo.broadcast_in_dim %v1135, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1137 = stablehlo.divide %v1136, %v1128 : tensor<32x96x56x56xf32>
    %v1138 = stablehlo.add %v1137, %v1129 : tensor<32x96x56x56xf32>
    %v1139 = stablehlo.rsqrt %v1138 : tensor<32x96x56x56xf32>
    %v1140 = stablehlo.multiply %v1133, %v1139 : tensor<32x96x56x56xf32>
    %v1141 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v1142 = stablehlo.multiply %v1141, %v1125 : tensor<32x96x56x56xf32>
    %v1143 = stablehlo.reduce(%v1142 init: %v1127) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1144 = stablehlo.broadcast_in_dim %v1143, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1145 = stablehlo.multiply %v1140, %v1142 : tensor<32x96x56x56xf32>
    %v1146 = stablehlo.reduce(%v1145 init: %v1127) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1147 = stablehlo.broadcast_in_dim %v1146, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1148 = stablehlo.multiply %v1142, %v1128 : tensor<32x96x56x56xf32>
    %v1149 = stablehlo.subtract %v1148, %v1144 : tensor<32x96x56x56xf32>
    %v1150 = stablehlo.multiply %v1140, %v1147 : tensor<32x96x56x56xf32>
    %v1151 = stablehlo.subtract %v1149, %v1150 : tensor<32x96x56x56xf32>
    %v1152 = stablehlo.divide %v1139, %v1128 : tensor<32x96x56x56xf32>
    %v1153 = stablehlo.multiply %v1152, %v1151 : tensor<32x96x56x56xf32>
    %v1154 = stablehlo.reshape %v1153 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1156 = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v1157 = stablehlo.convolution(%v1155, %v1156)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v1158 = stablehlo.reshape %v1157 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1159 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v1160 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v1161 = stablehlo.compare GT, %v136, %v1159 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1162 = stablehlo.compare LT, %v136, %v1160 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1163 = stablehlo.and %v1161, %v1162 : tensor<32x301056xi1>
    %v1164 = stablehlo.select %v1163, %v1158, %v1159 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1166 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1168 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1169 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1170 = stablehlo.reduce(%v1166 init: %v1167) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1171 = stablehlo.broadcast_in_dim %v1170, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1172 = stablehlo.divide %v1171, %v1168 : tensor<32x96x56x56xf32>
    %v1173 = stablehlo.subtract %v1166, %v1172 : tensor<32x96x56x56xf32>
    %v1174 = stablehlo.multiply %v1173, %v1173 : tensor<32x96x56x56xf32>
    %v1175 = stablehlo.reduce(%v1174 init: %v1167) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1176 = stablehlo.broadcast_in_dim %v1175, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1177 = stablehlo.divide %v1176, %v1168 : tensor<32x96x56x56xf32>
    %v1178 = stablehlo.add %v1177, %v1169 : tensor<32x96x56x56xf32>
    %v1179 = stablehlo.rsqrt %v1178 : tensor<32x96x56x56xf32>
    %v1180 = stablehlo.multiply %v1173, %v1179 : tensor<32x96x56x56xf32>
    %v1181 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v1182 = stablehlo.multiply %v1181, %v1165 : tensor<32x96x56x56xf32>
    %v1183 = stablehlo.reduce(%v1182 init: %v1167) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1184 = stablehlo.broadcast_in_dim %v1183, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1185 = stablehlo.multiply %v1180, %v1182 : tensor<32x96x56x56xf32>
    %v1186 = stablehlo.reduce(%v1185 init: %v1167) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1187 = stablehlo.broadcast_in_dim %v1186, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1188 = stablehlo.multiply %v1182, %v1168 : tensor<32x96x56x56xf32>
    %v1189 = stablehlo.subtract %v1188, %v1184 : tensor<32x96x56x56xf32>
    %v1190 = stablehlo.multiply %v1180, %v1187 : tensor<32x96x56x56xf32>
    %v1191 = stablehlo.subtract %v1189, %v1190 : tensor<32x96x56x56xf32>
    %v1192 = stablehlo.divide %v1179, %v1168 : tensor<32x96x56x56xf32>
    %v1193 = stablehlo.multiply %v1192, %v1191 : tensor<32x96x56x56xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1196 = stablehlo.transpose %b2eW, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v1197 = stablehlo.reverse %v1196, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v1198 = stablehlo.convolution(%v1195, %v1197)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1200 = stablehlo.add %v1199, %v1083 : tensor<32x75264xf32>
    %v1201 = stablehlo.reshape %v1200 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1202 = stablehlo.reshape %v91 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1204 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v1205 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v1206 = stablehlo.reduce(%v1202 init: %v1203) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1207 = stablehlo.broadcast_in_dim %v1206, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1208 = stablehlo.divide %v1207, %v1204 : tensor<32x24x56x56xf32>
    %v1209 = stablehlo.subtract %v1202, %v1208 : tensor<32x24x56x56xf32>
    %v1210 = stablehlo.multiply %v1209, %v1209 : tensor<32x24x56x56xf32>
    %v1211 = stablehlo.reduce(%v1210 init: %v1203) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1212 = stablehlo.broadcast_in_dim %v1211, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1213 = stablehlo.divide %v1212, %v1204 : tensor<32x24x56x56xf32>
    %v1214 = stablehlo.add %v1213, %v1205 : tensor<32x24x56x56xf32>
    %v1215 = stablehlo.rsqrt %v1214 : tensor<32x24x56x56xf32>
    %v1216 = stablehlo.multiply %v1209, %v1215 : tensor<32x24x56x56xf32>
    %v1217 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v1218 = stablehlo.multiply %v1217, %v1201 : tensor<32x24x56x56xf32>
    %v1219 = stablehlo.reduce(%v1218 init: %v1203) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1220 = stablehlo.broadcast_in_dim %v1219, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1221 = stablehlo.multiply %v1216, %v1218 : tensor<32x24x56x56xf32>
    %v1222 = stablehlo.reduce(%v1221 init: %v1203) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1223 = stablehlo.broadcast_in_dim %v1222, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1224 = stablehlo.multiply %v1218, %v1204 : tensor<32x24x56x56xf32>
    %v1225 = stablehlo.subtract %v1224, %v1220 : tensor<32x24x56x56xf32>
    %v1226 = stablehlo.multiply %v1216, %v1223 : tensor<32x24x56x56xf32>
    %v1227 = stablehlo.subtract %v1225, %v1226 : tensor<32x24x56x56xf32>
    %v1228 = stablehlo.divide %v1215, %v1204 : tensor<32x24x56x56xf32>
    %v1229 = stablehlo.multiply %v1228, %v1227 : tensor<32x24x56x56xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1232 = stablehlo.transpose %b1pW, dims = [1, 0, 2, 3] : (tensor<24x64x1x1xf32>) -> tensor<64x24x1x1xf32>
    %v1233 = stablehlo.reverse %v1232, dims = [2, 3] : tensor<64x24x1x1xf32>
    %v1234 = stablehlo.convolution(%v1231, %v1233)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<64x24x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v1236 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v1237 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v1238 = stablehlo.compare GT, %v82, %v1236 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v1239 = stablehlo.compare LT, %v82, %v1237 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v1240 = stablehlo.and %v1238, %v1239 : tensor<32x200704xi1>
    %v1241 = stablehlo.select %v1240, %v1235, %v1236 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1243 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1245 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v1246 = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %v1247 = stablehlo.reduce(%v1243 init: %v1244) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1248 = stablehlo.broadcast_in_dim %v1247, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1249 = stablehlo.divide %v1248, %v1245 : tensor<32x64x56x56xf32>
    %v1250 = stablehlo.subtract %v1243, %v1249 : tensor<32x64x56x56xf32>
    %v1251 = stablehlo.multiply %v1250, %v1250 : tensor<32x64x56x56xf32>
    %v1252 = stablehlo.reduce(%v1251 init: %v1244) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1253 = stablehlo.broadcast_in_dim %v1252, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1254 = stablehlo.divide %v1253, %v1245 : tensor<32x64x56x56xf32>
    %v1255 = stablehlo.add %v1254, %v1246 : tensor<32x64x56x56xf32>
    %v1256 = stablehlo.rsqrt %v1255 : tensor<32x64x56x56xf32>
    %v1257 = stablehlo.multiply %v1250, %v1256 : tensor<32x64x56x56xf32>
    %v1258 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v1259 = stablehlo.multiply %v1258, %v1242 : tensor<32x64x56x56xf32>
    %v1260 = stablehlo.reduce(%v1259 init: %v1244) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1261 = stablehlo.broadcast_in_dim %v1260, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1262 = stablehlo.multiply %v1257, %v1259 : tensor<32x64x56x56xf32>
    %v1263 = stablehlo.reduce(%v1262 init: %v1244) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1264 = stablehlo.broadcast_in_dim %v1263, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1265 = stablehlo.multiply %v1259, %v1245 : tensor<32x64x56x56xf32>
    %v1266 = stablehlo.subtract %v1265, %v1261 : tensor<32x64x56x56xf32>
    %v1267 = stablehlo.multiply %v1257, %v1264 : tensor<32x64x56x56xf32>
    %v1268 = stablehlo.subtract %v1266, %v1267 : tensor<32x64x56x56xf32>
    %v1269 = stablehlo.divide %v1256, %v1245 : tensor<32x64x56x56xf32>
    %v1270 = stablehlo.multiply %v1269, %v1268 : tensor<32x64x56x56xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1274 = stablehlo.pad %v1272, %v1273, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v1275 = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<64x1x3x3xf32>
    %v1276 = stablehlo.convolution(%v1274, %v1275)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<32x64x112x112xf32>, tensor<64x1x3x3xf32>) -> tensor<32x64x112x112xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v1278 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v1279 = stablehlo.constant dense<6.0> : tensor<32x802816xf32>
    %v1280 = stablehlo.compare GT, %v53, %v1278 : (tensor<32x802816xf32>, tensor<32x802816xf32>) -> tensor<32x802816xi1>
    %v1281 = stablehlo.compare LT, %v53, %v1279 : (tensor<32x802816xf32>, tensor<32x802816xf32>) -> tensor<32x802816xi1>
    %v1282 = stablehlo.and %v1280, %v1281 : tensor<32x802816xi1>
    %v1283 = stablehlo.select %v1282, %v1277, %v1278 : tensor<32x802816xi1>, tensor<32x802816xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1285 = stablehlo.reshape %v33 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1287 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v1288 = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %v1289 = stablehlo.reduce(%v1285 init: %v1286) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1290 = stablehlo.broadcast_in_dim %v1289, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1291 = stablehlo.divide %v1290, %v1287 : tensor<32x64x112x112xf32>
    %v1292 = stablehlo.subtract %v1285, %v1291 : tensor<32x64x112x112xf32>
    %v1293 = stablehlo.multiply %v1292, %v1292 : tensor<32x64x112x112xf32>
    %v1294 = stablehlo.reduce(%v1293 init: %v1286) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1295 = stablehlo.broadcast_in_dim %v1294, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1296 = stablehlo.divide %v1295, %v1287 : tensor<32x64x112x112xf32>
    %v1297 = stablehlo.add %v1296, %v1288 : tensor<32x64x112x112xf32>
    %v1298 = stablehlo.rsqrt %v1297 : tensor<32x64x112x112xf32>
    %v1299 = stablehlo.multiply %v1292, %v1298 : tensor<32x64x112x112xf32>
    %v1300 = stablehlo.broadcast_in_dim %b1eg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v1301 = stablehlo.multiply %v1300, %v1284 : tensor<32x64x112x112xf32>
    %v1302 = stablehlo.reduce(%v1301 init: %v1286) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1303 = stablehlo.broadcast_in_dim %v1302, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1304 = stablehlo.multiply %v1299, %v1301 : tensor<32x64x112x112xf32>
    %v1305 = stablehlo.reduce(%v1304 init: %v1286) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1306 = stablehlo.broadcast_in_dim %v1305, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1307 = stablehlo.multiply %v1301, %v1287 : tensor<32x64x112x112xf32>
    %v1308 = stablehlo.subtract %v1307, %v1303 : tensor<32x64x112x112xf32>
    %v1309 = stablehlo.multiply %v1299, %v1306 : tensor<32x64x112x112xf32>
    %v1310 = stablehlo.subtract %v1308, %v1309 : tensor<32x64x112x112xf32>
    %v1311 = stablehlo.divide %v1298, %v1287 : tensor<32x64x112x112xf32>
    %v1312 = stablehlo.multiply %v1311, %v1310 : tensor<32x64x112x112xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1315 = stablehlo.transpose %b1eW, dims = [1, 0, 2, 3] : (tensor<64x16x1x1xf32>) -> tensor<16x64x1x1xf32>
    %v1316 = stablehlo.reverse %v1315, dims = [2, 3] : tensor<16x64x1x1xf32>
    %v1317 = stablehlo.convolution(%v1314, %v1316)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<16x64x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v1319 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v1320 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v1321 = stablehlo.compare GT, %v24, %v1319 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v1322 = stablehlo.compare LT, %v24, %v1320 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v1323 = stablehlo.and %v1321, %v1322 : tensor<32x200704xi1>
    %v1324 = stablehlo.select %v1323, %v1318, %v1319 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v1326 = stablehlo.reshape %v4 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v1327 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1328 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v1329 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v1330 = stablehlo.reduce(%v1326 init: %v1327) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v1331 = stablehlo.broadcast_in_dim %v1330, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v1332 = stablehlo.divide %v1331, %v1328 : tensor<32x16x112x112xf32>
    %v1333 = stablehlo.subtract %v1326, %v1332 : tensor<32x16x112x112xf32>
    %v1334 = stablehlo.multiply %v1333, %v1333 : tensor<32x16x112x112xf32>
    %v1335 = stablehlo.reduce(%v1334 init: %v1327) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v1336 = stablehlo.broadcast_in_dim %v1335, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v1337 = stablehlo.divide %v1336, %v1328 : tensor<32x16x112x112xf32>
    %v1338 = stablehlo.add %v1337, %v1329 : tensor<32x16x112x112xf32>
    %v1339 = stablehlo.rsqrt %v1338 : tensor<32x16x112x112xf32>
    %v1340 = stablehlo.multiply %v1333, %v1339 : tensor<32x16x112x112xf32>
    %v1341 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v1342 = stablehlo.multiply %v1341, %v1325 : tensor<32x16x112x112xf32>
    %v1343 = stablehlo.reduce(%v1342 init: %v1327) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v1344 = stablehlo.broadcast_in_dim %v1343, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v1345 = stablehlo.multiply %v1340, %v1342 : tensor<32x16x112x112xf32>
    %v1346 = stablehlo.reduce(%v1345 init: %v1327) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v1347 = stablehlo.broadcast_in_dim %v1346, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v1348 = stablehlo.multiply %v1342, %v1328 : tensor<32x16x112x112xf32>
    %v1349 = stablehlo.subtract %v1348, %v1344 : tensor<32x16x112x112xf32>
    %v1350 = stablehlo.multiply %v1340, %v1347 : tensor<32x16x112x112xf32>
    %v1351 = stablehlo.subtract %v1349, %v1350 : tensor<32x16x112x112xf32>
    %v1352 = stablehlo.divide %v1339, %v1328 : tensor<32x16x112x112xf32>
    %v1353 = stablehlo.multiply %v1352, %v1351 : tensor<32x16x112x112xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %b6dpWxi = stablehlo.reshape %v503 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %b6dpWdi = stablehlo.reshape %v642 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %b6dpWxt = stablehlo.transpose %b6dpWxi, dims = [1, 0, 2, 3] : (tensor<32x256x7x7xf32>) -> tensor<256x32x7x7xf32>
    %b6dpWdt = stablehlo.transpose %b6dpWdi, dims = [1, 0, 2, 3] : (tensor<32x64x7x7xf32>) -> tensor<64x32x7x7xf32>
    %b6dpWraw = stablehlo.convolution(%b6dpWxt, %b6dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x7x7xf32>, tensor<64x32x7x7xf32>) -> tensor<256x64x1x1xf32>
    %b6dpW = stablehlo.transpose %b6dpWraw, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %b6dpbi = stablehlo.reshape %v642 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %b6dpb = stablehlo.reduce(%b6dpbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %b6dpgxr = stablehlo.reshape %v508 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %b6dpgdyr = stablehlo.reshape %v612 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %b6dpgnf = stablehlo.constant dense<49.0> : tensor<32x64x7x7xf32>
    %b6dpgep = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %b6dpgsmr = stablehlo.reduce(%b6dpgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %b6dpgsm = stablehlo.broadcast_in_dim %b6dpgsmr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %b6dpgmu = stablehlo.divide %b6dpgsm, %b6dpgnf : tensor<32x64x7x7xf32>
    %b6dpgxc = stablehlo.subtract %b6dpgxr, %b6dpgmu : tensor<32x64x7x7xf32>
    %b6dpgsq = stablehlo.multiply %b6dpgxc, %b6dpgxc : tensor<32x64x7x7xf32>
    %b6dpgvsr = stablehlo.reduce(%b6dpgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %b6dpgvs = stablehlo.broadcast_in_dim %b6dpgvsr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %b6dpgvar = stablehlo.divide %b6dpgvs, %b6dpgnf : tensor<32x64x7x7xf32>
    %b6dpgve = stablehlo.add %b6dpgvar, %b6dpgep : tensor<32x64x7x7xf32>
    %b6dpgistd = stablehlo.rsqrt %b6dpgve : tensor<32x64x7x7xf32>
    %b6dpgxh = stablehlo.multiply %b6dpgxc, %b6dpgistd : tensor<32x64x7x7xf32>
    %b6dpgp = stablehlo.multiply %b6dpgdyr, %b6dpgxh : tensor<32x64x7x7xf32>
    %b6dpg = stablehlo.reduce(%b6dpgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %b6dpbt = stablehlo.reduce(%b6dpgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %b6ddui = stablehlo.reshape %v683 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %b6ddup = stablehlo.pad %b6ddui, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256x14x14xf32>
    %b6ddu = stablehlo.reshape %b6ddup : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %b6ddWxi = stablehlo.reshape %v474 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %b6ddWdi = stablehlo.reshape %b6ddu : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %b6ddWxt = stablehlo.transpose %b6ddWxi, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %b6ddWdt = stablehlo.transpose %b6ddWdi, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %b6ddWraw = stablehlo.convolution(%b6ddWxt, %b6ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 256 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<1x256x3x3xf32>
    %b6ddW = stablehlo.reshape %b6ddWraw : (tensor<1x256x3x3xf32>) -> tensor<256x1x3x3xf32>
    %b6ddbi = stablehlo.reshape %v683 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %b6ddb = stablehlo.reduce(%b6ddbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %b6ddgxr = stablehlo.reshape %v479 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %b6ddgdyr = stablehlo.reshape %v653 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %b6ddgnf = stablehlo.constant dense<49.0> : tensor<32x256x7x7xf32>
    %b6ddgep = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %b6ddgsmr = stablehlo.reduce(%b6ddgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %b6ddgsm = stablehlo.broadcast_in_dim %b6ddgsmr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %b6ddgmu = stablehlo.divide %b6ddgsm, %b6ddgnf : tensor<32x256x7x7xf32>
    %b6ddgxc = stablehlo.subtract %b6ddgxr, %b6ddgmu : tensor<32x256x7x7xf32>
    %b6ddgsq = stablehlo.multiply %b6ddgxc, %b6ddgxc : tensor<32x256x7x7xf32>
    %b6ddgvsr = stablehlo.reduce(%b6ddgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %b6ddgvs = stablehlo.broadcast_in_dim %b6ddgvsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %b6ddgvar = stablehlo.divide %b6ddgvs, %b6ddgnf : tensor<32x256x7x7xf32>
    %b6ddgve = stablehlo.add %b6ddgvar, %b6ddgep : tensor<32x256x7x7xf32>
    %b6ddgistd = stablehlo.rsqrt %b6ddgve : tensor<32x256x7x7xf32>
    %b6ddgxh = stablehlo.multiply %b6ddgxc, %b6ddgistd : tensor<32x256x7x7xf32>
    %b6ddgp = stablehlo.multiply %b6ddgdyr, %b6ddgxh : tensor<32x256x7x7xf32>
    %b6ddg = stablehlo.reduce(%b6ddgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %b6ddbt = stablehlo.reduce(%b6ddgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %b6deWxi = stablehlo.reshape %v445 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %b6deWdi = stablehlo.reshape %v725 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %b6deWxt = stablehlo.transpose %b6deWxi, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %b6deWdt = stablehlo.transpose %b6deWdi, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %b6deWraw = stablehlo.convolution(%b6deWxt, %b6deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<64x256x1x1xf32>
    %b6deW = stablehlo.transpose %b6deWraw, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %b6debi = stablehlo.reshape %v725 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %b6deb = stablehlo.reduce(%b6debi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %b6degxr = stablehlo.reshape %v450 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %b6degdyr = stablehlo.reshape %v695 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %b6degnf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %b6degep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %b6degsmr = stablehlo.reduce(%b6degxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %b6degsm = stablehlo.broadcast_in_dim %b6degsmr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %b6degmu = stablehlo.divide %b6degsm, %b6degnf : tensor<32x256x14x14xf32>
    %b6degxc = stablehlo.subtract %b6degxr, %b6degmu : tensor<32x256x14x14xf32>
    %b6degsq = stablehlo.multiply %b6degxc, %b6degxc : tensor<32x256x14x14xf32>
    %b6degvsr = stablehlo.reduce(%b6degsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %b6degvs = stablehlo.broadcast_in_dim %b6degvsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %b6degvar = stablehlo.divide %b6degvs, %b6degnf : tensor<32x256x14x14xf32>
    %b6degve = stablehlo.add %b6degvar, %b6degep : tensor<32x256x14x14xf32>
    %b6degistd = stablehlo.rsqrt %b6degve : tensor<32x256x14x14xf32>
    %b6degxh = stablehlo.multiply %b6degxc, %b6degistd : tensor<32x256x14x14xf32>
    %b6degp = stablehlo.multiply %b6degdyr, %b6degxh : tensor<32x256x14x14xf32>
    %b6deg = stablehlo.reduce(%b6degp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %b6debt = stablehlo.reduce(%b6degdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %b5dpWxi = stablehlo.reshape %v420 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %b5dpWdi = stablehlo.reshape %v760 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %b5dpWxt = stablehlo.transpose %b5dpWxi, dims = [1, 0, 2, 3] : (tensor<32x128x14x14xf32>) -> tensor<128x32x14x14xf32>
    %b5dpWdt = stablehlo.transpose %b5dpWdi, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %b5dpWraw = stablehlo.convolution(%b5dpWxt, %b5dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<128x64x1x1xf32>
    %b5dpW = stablehlo.transpose %b5dpWraw, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %b5dpbi = stablehlo.reshape %v760 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %b5dpb = stablehlo.reduce(%b5dpbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b5dpgxr = stablehlo.reshape %v425 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %b5dpgdyr = stablehlo.reshape %v730 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %b5dpgnf = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %b5dpgep = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %b5dpgsmr = stablehlo.reduce(%b5dpgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %b5dpgsm = stablehlo.broadcast_in_dim %b5dpgsmr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %b5dpgmu = stablehlo.divide %b5dpgsm, %b5dpgnf : tensor<32x64x14x14xf32>
    %b5dpgxc = stablehlo.subtract %b5dpgxr, %b5dpgmu : tensor<32x64x14x14xf32>
    %b5dpgsq = stablehlo.multiply %b5dpgxc, %b5dpgxc : tensor<32x64x14x14xf32>
    %b5dpgvsr = stablehlo.reduce(%b5dpgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %b5dpgvs = stablehlo.broadcast_in_dim %b5dpgvsr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %b5dpgvar = stablehlo.divide %b5dpgvs, %b5dpgnf : tensor<32x64x14x14xf32>
    %b5dpgve = stablehlo.add %b5dpgvar, %b5dpgep : tensor<32x64x14x14xf32>
    %b5dpgistd = stablehlo.rsqrt %b5dpgve : tensor<32x64x14x14xf32>
    %b5dpgxh = stablehlo.multiply %b5dpgxc, %b5dpgistd : tensor<32x64x14x14xf32>
    %b5dpgp = stablehlo.multiply %b5dpgdyr, %b5dpgxh : tensor<32x64x14x14xf32>
    %b5dpg = stablehlo.reduce(%b5dpgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b5dpbt = stablehlo.reduce(%b5dpgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b5ddui = stablehlo.reshape %v801 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %b5ddup = stablehlo.pad %b5ddui, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128x28x28xf32>
    %b5ddu = stablehlo.reshape %b5ddup : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %b5ddWxi = stablehlo.reshape %v391 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b5ddWdi = stablehlo.reshape %b5ddu : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b5ddWxt = stablehlo.transpose %b5ddWxi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b5ddWdt = stablehlo.transpose %b5ddWdi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b5ddWraw = stablehlo.convolution(%b5ddWxt, %b5ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 128 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<1x128x3x3xf32>
    %b5ddW = stablehlo.reshape %b5ddWraw : (tensor<1x128x3x3xf32>) -> tensor<128x1x3x3xf32>
    %b5ddbi = stablehlo.reshape %v801 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %b5ddb = stablehlo.reduce(%b5ddbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %b5ddgxr = stablehlo.reshape %v396 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %b5ddgdyr = stablehlo.reshape %v771 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %b5ddgnf = stablehlo.constant dense<196.0> : tensor<32x128x14x14xf32>
    %b5ddgep = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %b5ddgsmr = stablehlo.reduce(%b5ddgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %b5ddgsm = stablehlo.broadcast_in_dim %b5ddgsmr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %b5ddgmu = stablehlo.divide %b5ddgsm, %b5ddgnf : tensor<32x128x14x14xf32>
    %b5ddgxc = stablehlo.subtract %b5ddgxr, %b5ddgmu : tensor<32x128x14x14xf32>
    %b5ddgsq = stablehlo.multiply %b5ddgxc, %b5ddgxc : tensor<32x128x14x14xf32>
    %b5ddgvsr = stablehlo.reduce(%b5ddgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %b5ddgvs = stablehlo.broadcast_in_dim %b5ddgvsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %b5ddgvar = stablehlo.divide %b5ddgvs, %b5ddgnf : tensor<32x128x14x14xf32>
    %b5ddgve = stablehlo.add %b5ddgvar, %b5ddgep : tensor<32x128x14x14xf32>
    %b5ddgistd = stablehlo.rsqrt %b5ddgve : tensor<32x128x14x14xf32>
    %b5ddgxh = stablehlo.multiply %b5ddgxc, %b5ddgistd : tensor<32x128x14x14xf32>
    %b5ddgp = stablehlo.multiply %b5ddgdyr, %b5ddgxh : tensor<32x128x14x14xf32>
    %b5ddg = stablehlo.reduce(%b5ddgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %b5ddbt = stablehlo.reduce(%b5ddgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %b5deWxi = stablehlo.reshape %v362 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b5deWdi = stablehlo.reshape %v843 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b5deWxt = stablehlo.transpose %b5deWxi, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %b5deWdt = stablehlo.transpose %b5deWdi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b5deWraw = stablehlo.convolution(%b5deWxt, %b5deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<32x128x1x1xf32>
    %b5deW = stablehlo.transpose %b5deWraw, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %b5debi = stablehlo.reshape %v843 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b5deb = stablehlo.reduce(%b5debi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b5degxr = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b5degdyr = stablehlo.reshape %v813 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b5degnf = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %b5degep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %b5degsmr = stablehlo.reduce(%b5degxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %b5degsm = stablehlo.broadcast_in_dim %b5degsmr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %b5degmu = stablehlo.divide %b5degsm, %b5degnf : tensor<32x128x28x28xf32>
    %b5degxc = stablehlo.subtract %b5degxr, %b5degmu : tensor<32x128x28x28xf32>
    %b5degsq = stablehlo.multiply %b5degxc, %b5degxc : tensor<32x128x28x28xf32>
    %b5degvsr = stablehlo.reduce(%b5degsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %b5degvs = stablehlo.broadcast_in_dim %b5degvsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %b5degvar = stablehlo.divide %b5degvs, %b5degnf : tensor<32x128x28x28xf32>
    %b5degve = stablehlo.add %b5degvar, %b5degep : tensor<32x128x28x28xf32>
    %b5degistd = stablehlo.rsqrt %b5degve : tensor<32x128x28x28xf32>
    %b5degxh = stablehlo.multiply %b5degxc, %b5degistd : tensor<32x128x28x28xf32>
    %b5degp = stablehlo.multiply %b5degdyr, %b5degxh : tensor<32x128x28x28xf32>
    %b5deg = stablehlo.reduce(%b5degp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b5debt = stablehlo.reduce(%b5degdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4dpWxi = stablehlo.reshape %v336 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4dpWdi = stablehlo.reshape %v878 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b4dpWxt = stablehlo.transpose %b4dpWxi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b4dpWdt = stablehlo.transpose %b4dpWdi, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %b4dpWraw = stablehlo.convolution(%b4dpWxt, %b4dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<128x32x1x1xf32>
    %b4dpW = stablehlo.transpose %b4dpWraw, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %b4dpbi = stablehlo.reshape %v878 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b4dpb = stablehlo.reduce(%b4dpbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b4dpgxr = stablehlo.reshape %v341 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b4dpgdyr = stablehlo.reshape %v848 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b4dpgnf = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %b4dpgep = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %b4dpgsmr = stablehlo.reduce(%b4dpgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %b4dpgsm = stablehlo.broadcast_in_dim %b4dpgsmr, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %b4dpgmu = stablehlo.divide %b4dpgsm, %b4dpgnf : tensor<32x32x28x28xf32>
    %b4dpgxc = stablehlo.subtract %b4dpgxr, %b4dpgmu : tensor<32x32x28x28xf32>
    %b4dpgsq = stablehlo.multiply %b4dpgxc, %b4dpgxc : tensor<32x32x28x28xf32>
    %b4dpgvsr = stablehlo.reduce(%b4dpgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %b4dpgvs = stablehlo.broadcast_in_dim %b4dpgvsr, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %b4dpgvar = stablehlo.divide %b4dpgvs, %b4dpgnf : tensor<32x32x28x28xf32>
    %b4dpgve = stablehlo.add %b4dpgvar, %b4dpgep : tensor<32x32x28x28xf32>
    %b4dpgistd = stablehlo.rsqrt %b4dpgve : tensor<32x32x28x28xf32>
    %b4dpgxh = stablehlo.multiply %b4dpgxc, %b4dpgistd : tensor<32x32x28x28xf32>
    %b4dpgp = stablehlo.multiply %b4dpgdyr, %b4dpgxh : tensor<32x32x28x28xf32>
    %b4dpg = stablehlo.reduce(%b4dpgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b4dpbt = stablehlo.reduce(%b4dpgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b4ddWxi = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4ddWdi = stablehlo.reshape %v919 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4ddWxt = stablehlo.transpose %b4ddWxi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b4ddWdt = stablehlo.transpose %b4ddWdi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b4ddWraw = stablehlo.convolution(%b4ddWxt, %b4ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 128 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<1x128x3x3xf32>
    %b4ddW = stablehlo.reshape %b4ddWraw : (tensor<1x128x3x3xf32>) -> tensor<128x1x3x3xf32>
    %b4ddbi = stablehlo.reshape %v919 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4ddb = stablehlo.reduce(%b4ddbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4ddgxr = stablehlo.reshape %v312 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4ddgdyr = stablehlo.reshape %v889 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4ddgnf = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %b4ddgep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %b4ddgsmr = stablehlo.reduce(%b4ddgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %b4ddgsm = stablehlo.broadcast_in_dim %b4ddgsmr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %b4ddgmu = stablehlo.divide %b4ddgsm, %b4ddgnf : tensor<32x128x28x28xf32>
    %b4ddgxc = stablehlo.subtract %b4ddgxr, %b4ddgmu : tensor<32x128x28x28xf32>
    %b4ddgsq = stablehlo.multiply %b4ddgxc, %b4ddgxc : tensor<32x128x28x28xf32>
    %b4ddgvsr = stablehlo.reduce(%b4ddgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %b4ddgvs = stablehlo.broadcast_in_dim %b4ddgvsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %b4ddgvar = stablehlo.divide %b4ddgvs, %b4ddgnf : tensor<32x128x28x28xf32>
    %b4ddgve = stablehlo.add %b4ddgvar, %b4ddgep : tensor<32x128x28x28xf32>
    %b4ddgistd = stablehlo.rsqrt %b4ddgve : tensor<32x128x28x28xf32>
    %b4ddgxh = stablehlo.multiply %b4ddgxc, %b4ddgistd : tensor<32x128x28x28xf32>
    %b4ddgp = stablehlo.multiply %b4ddgdyr, %b4ddgxh : tensor<32x128x28x28xf32>
    %b4ddg = stablehlo.reduce(%b4ddgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4ddbt = stablehlo.reduce(%b4ddgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4deWxi = stablehlo.reshape %v278 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b4deWdi = stablehlo.reshape %v959 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4deWxt = stablehlo.transpose %b4deWxi, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %b4deWdt = stablehlo.transpose %b4deWdi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b4deWraw = stablehlo.convolution(%b4deWxt, %b4deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<32x128x1x1xf32>
    %b4deW = stablehlo.transpose %b4deWraw, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %b4debi = stablehlo.reshape %v959 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4deb = stablehlo.reduce(%b4debi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4degxr = stablehlo.reshape %v283 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4degdyr = stablehlo.reshape %v929 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4degnf = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %b4degep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %b4degsmr = stablehlo.reduce(%b4degxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %b4degsm = stablehlo.broadcast_in_dim %b4degsmr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %b4degmu = stablehlo.divide %b4degsm, %b4degnf : tensor<32x128x28x28xf32>
    %b4degxc = stablehlo.subtract %b4degxr, %b4degmu : tensor<32x128x28x28xf32>
    %b4degsq = stablehlo.multiply %b4degxc, %b4degxc : tensor<32x128x28x28xf32>
    %b4degvsr = stablehlo.reduce(%b4degsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %b4degvs = stablehlo.broadcast_in_dim %b4degvsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %b4degvar = stablehlo.divide %b4degvs, %b4degnf : tensor<32x128x28x28xf32>
    %b4degve = stablehlo.add %b4degvar, %b4degep : tensor<32x128x28x28xf32>
    %b4degistd = stablehlo.rsqrt %b4degve : tensor<32x128x28x28xf32>
    %b4degxh = stablehlo.multiply %b4degxc, %b4degistd : tensor<32x128x28x28xf32>
    %b4degp = stablehlo.multiply %b4degdyr, %b4degxh : tensor<32x128x28x28xf32>
    %b4deg = stablehlo.reduce(%b4degp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4debt = stablehlo.reduce(%b4degdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b3dpWxi = stablehlo.reshape %v253 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %b3dpWdi = stablehlo.reshape %v995 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b3dpWxt = stablehlo.transpose %b3dpWxi, dims = [1, 0, 2, 3] : (tensor<32x96x28x28xf32>) -> tensor<96x32x28x28xf32>
    %b3dpWdt = stablehlo.transpose %b3dpWdi, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %b3dpWraw = stablehlo.convolution(%b3dpWxt, %b3dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<96x32x1x1xf32>
    %b3dpW = stablehlo.transpose %b3dpWraw, dims = [1, 0, 2, 3] : (tensor<96x32x1x1xf32>) -> tensor<32x96x1x1xf32>
    %b3dpbi = stablehlo.reshape %v995 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b3dpb = stablehlo.reduce(%b3dpbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b3dpgxr = stablehlo.reshape %v258 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b3dpgdyr = stablehlo.reshape %v965 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b3dpgnf = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %b3dpgep = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %b3dpgsmr = stablehlo.reduce(%b3dpgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %b3dpgsm = stablehlo.broadcast_in_dim %b3dpgsmr, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %b3dpgmu = stablehlo.divide %b3dpgsm, %b3dpgnf : tensor<32x32x28x28xf32>
    %b3dpgxc = stablehlo.subtract %b3dpgxr, %b3dpgmu : tensor<32x32x28x28xf32>
    %b3dpgsq = stablehlo.multiply %b3dpgxc, %b3dpgxc : tensor<32x32x28x28xf32>
    %b3dpgvsr = stablehlo.reduce(%b3dpgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %b3dpgvs = stablehlo.broadcast_in_dim %b3dpgvsr, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %b3dpgvar = stablehlo.divide %b3dpgvs, %b3dpgnf : tensor<32x32x28x28xf32>
    %b3dpgve = stablehlo.add %b3dpgvar, %b3dpgep : tensor<32x32x28x28xf32>
    %b3dpgistd = stablehlo.rsqrt %b3dpgve : tensor<32x32x28x28xf32>
    %b3dpgxh = stablehlo.multiply %b3dpgxc, %b3dpgistd : tensor<32x32x28x28xf32>
    %b3dpgp = stablehlo.multiply %b3dpgdyr, %b3dpgxh : tensor<32x32x28x28xf32>
    %b3dpg = stablehlo.reduce(%b3dpgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b3dpbt = stablehlo.reduce(%b3dpgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b3ddui = stablehlo.reshape %v1036 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %b3ddup = stablehlo.pad %b3ddui, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96x56x56xf32>
    %b3ddu = stablehlo.reshape %b3ddup : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %b3ddWxi = stablehlo.reshape %v224 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b3ddWdi = stablehlo.reshape %b3ddu : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b3ddWxt = stablehlo.transpose %b3ddWxi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b3ddWdt = stablehlo.transpose %b3ddWdi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b3ddWraw = stablehlo.convolution(%b3ddWxt, %b3ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x3x3xf32>
    %b3ddW = stablehlo.reshape %b3ddWraw : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %b3ddbi = stablehlo.reshape %v1036 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %b3ddb = stablehlo.reduce(%b3ddbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %b3ddgxr = stablehlo.reshape %v229 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %b3ddgdyr = stablehlo.reshape %v1006 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %b3ddgnf = stablehlo.constant dense<784.0> : tensor<32x96x28x28xf32>
    %b3ddgep = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %b3ddgsmr = stablehlo.reduce(%b3ddgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %b3ddgsm = stablehlo.broadcast_in_dim %b3ddgsmr, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %b3ddgmu = stablehlo.divide %b3ddgsm, %b3ddgnf : tensor<32x96x28x28xf32>
    %b3ddgxc = stablehlo.subtract %b3ddgxr, %b3ddgmu : tensor<32x96x28x28xf32>
    %b3ddgsq = stablehlo.multiply %b3ddgxc, %b3ddgxc : tensor<32x96x28x28xf32>
    %b3ddgvsr = stablehlo.reduce(%b3ddgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %b3ddgvs = stablehlo.broadcast_in_dim %b3ddgvsr, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %b3ddgvar = stablehlo.divide %b3ddgvs, %b3ddgnf : tensor<32x96x28x28xf32>
    %b3ddgve = stablehlo.add %b3ddgvar, %b3ddgep : tensor<32x96x28x28xf32>
    %b3ddgistd = stablehlo.rsqrt %b3ddgve : tensor<32x96x28x28xf32>
    %b3ddgxh = stablehlo.multiply %b3ddgxc, %b3ddgistd : tensor<32x96x28x28xf32>
    %b3ddgp = stablehlo.multiply %b3ddgdyr, %b3ddgxh : tensor<32x96x28x28xf32>
    %b3ddg = stablehlo.reduce(%b3ddgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %b3ddbt = stablehlo.reduce(%b3ddgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %b3deWxi = stablehlo.reshape %v195 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b3deWdi = stablehlo.reshape %v1078 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b3deWxt = stablehlo.transpose %b3deWxi, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b3deWdt = stablehlo.transpose %b3deWdi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b3deWraw = stablehlo.convolution(%b3deWxt, %b3deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<24x96x1x1xf32>
    %b3deW = stablehlo.transpose %b3deWraw, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %b3debi = stablehlo.reshape %v1078 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b3deb = stablehlo.reduce(%b3debi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b3degxr = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b3degdyr = stablehlo.reshape %v1048 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b3degnf = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %b3degep = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %b3degsmr = stablehlo.reduce(%b3degxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %b3degsm = stablehlo.broadcast_in_dim %b3degsmr, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %b3degmu = stablehlo.divide %b3degsm, %b3degnf : tensor<32x96x56x56xf32>
    %b3degxc = stablehlo.subtract %b3degxr, %b3degmu : tensor<32x96x56x56xf32>
    %b3degsq = stablehlo.multiply %b3degxc, %b3degxc : tensor<32x96x56x56xf32>
    %b3degvsr = stablehlo.reduce(%b3degsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %b3degvs = stablehlo.broadcast_in_dim %b3degvsr, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %b3degvar = stablehlo.divide %b3degvs, %b3degnf : tensor<32x96x56x56xf32>
    %b3degve = stablehlo.add %b3degvar, %b3degep : tensor<32x96x56x56xf32>
    %b3degistd = stablehlo.rsqrt %b3degve : tensor<32x96x56x56xf32>
    %b3degxh = stablehlo.multiply %b3degxc, %b3degistd : tensor<32x96x56x56xf32>
    %b3degp = stablehlo.multiply %b3degdyr, %b3degxh : tensor<32x96x56x56xf32>
    %b3deg = stablehlo.reduce(%b3degp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b3debt = stablehlo.reduce(%b3degdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dpWxi = stablehlo.reshape %v169 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2dpWdi = stablehlo.reshape %v1113 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b2dpWxt = stablehlo.transpose %b2dpWxi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b2dpWdt = stablehlo.transpose %b2dpWdi, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b2dpWraw = stablehlo.convolution(%b2dpWxt, %b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %b2dpW = stablehlo.transpose %b2dpWraw, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %b2dpbi = stablehlo.reshape %v1113 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b2dpb = stablehlo.reduce(%b2dpbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpgxr = stablehlo.reshape %v174 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b2dpgdyr = stablehlo.reshape %v1083 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b2dpgnf = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %b2dpgep = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %b2dpgsmr = stablehlo.reduce(%b2dpgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %b2dpgsm = stablehlo.broadcast_in_dim %b2dpgsmr, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %b2dpgmu = stablehlo.divide %b2dpgsm, %b2dpgnf : tensor<32x24x56x56xf32>
    %b2dpgxc = stablehlo.subtract %b2dpgxr, %b2dpgmu : tensor<32x24x56x56xf32>
    %b2dpgsq = stablehlo.multiply %b2dpgxc, %b2dpgxc : tensor<32x24x56x56xf32>
    %b2dpgvsr = stablehlo.reduce(%b2dpgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %b2dpgvs = stablehlo.broadcast_in_dim %b2dpgvsr, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %b2dpgvar = stablehlo.divide %b2dpgvs, %b2dpgnf : tensor<32x24x56x56xf32>
    %b2dpgve = stablehlo.add %b2dpgvar, %b2dpgep : tensor<32x24x56x56xf32>
    %b2dpgistd = stablehlo.rsqrt %b2dpgve : tensor<32x24x56x56xf32>
    %b2dpgxh = stablehlo.multiply %b2dpgxc, %b2dpgistd : tensor<32x24x56x56xf32>
    %b2dpgp = stablehlo.multiply %b2dpgdyr, %b2dpgxh : tensor<32x24x56x56xf32>
    %b2dpg = stablehlo.reduce(%b2dpgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpbt = stablehlo.reduce(%b2dpgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2ddWxi = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2ddWdi = stablehlo.reshape %v1154 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2ddWxt = stablehlo.transpose %b2ddWxi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b2ddWdt = stablehlo.transpose %b2ddWdi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b2ddWraw = stablehlo.convolution(%b2ddWxt, %b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x3x3xf32>
    %b2ddW = stablehlo.reshape %b2ddWraw : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %b2ddbi = stablehlo.reshape %v1154 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2ddb = stablehlo.reduce(%b2ddbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddgxr = stablehlo.reshape %v145 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2ddgdyr = stablehlo.reshape %v1124 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2ddgnf = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %b2ddgep = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %b2ddgsmr = stablehlo.reduce(%b2ddgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %b2ddgsm = stablehlo.broadcast_in_dim %b2ddgsmr, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %b2ddgmu = stablehlo.divide %b2ddgsm, %b2ddgnf : tensor<32x96x56x56xf32>
    %b2ddgxc = stablehlo.subtract %b2ddgxr, %b2ddgmu : tensor<32x96x56x56xf32>
    %b2ddgsq = stablehlo.multiply %b2ddgxc, %b2ddgxc : tensor<32x96x56x56xf32>
    %b2ddgvsr = stablehlo.reduce(%b2ddgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %b2ddgvs = stablehlo.broadcast_in_dim %b2ddgvsr, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %b2ddgvar = stablehlo.divide %b2ddgvs, %b2ddgnf : tensor<32x96x56x56xf32>
    %b2ddgve = stablehlo.add %b2ddgvar, %b2ddgep : tensor<32x96x56x56xf32>
    %b2ddgistd = stablehlo.rsqrt %b2ddgve : tensor<32x96x56x56xf32>
    %b2ddgxh = stablehlo.multiply %b2ddgxc, %b2ddgistd : tensor<32x96x56x56xf32>
    %b2ddgp = stablehlo.multiply %b2ddgdyr, %b2ddgxh : tensor<32x96x56x56xf32>
    %b2ddg = stablehlo.reduce(%b2ddgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddbt = stablehlo.reduce(%b2ddgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2deWxi = stablehlo.reshape %v111 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b2deWdi = stablehlo.reshape %v1194 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2deWxt = stablehlo.transpose %b2deWxi, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b2deWdt = stablehlo.transpose %b2deWdi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b2deWraw = stablehlo.convolution(%b2deWxt, %b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<24x96x1x1xf32>
    %b2deW = stablehlo.transpose %b2deWraw, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %b2debi = stablehlo.reshape %v1194 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2deb = stablehlo.reduce(%b2debi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2degxr = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2degdyr = stablehlo.reshape %v1164 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2degnf = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %b2degep = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %b2degsmr = stablehlo.reduce(%b2degxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %b2degsm = stablehlo.broadcast_in_dim %b2degsmr, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %b2degmu = stablehlo.divide %b2degsm, %b2degnf : tensor<32x96x56x56xf32>
    %b2degxc = stablehlo.subtract %b2degxr, %b2degmu : tensor<32x96x56x56xf32>
    %b2degsq = stablehlo.multiply %b2degxc, %b2degxc : tensor<32x96x56x56xf32>
    %b2degvsr = stablehlo.reduce(%b2degsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %b2degvs = stablehlo.broadcast_in_dim %b2degvsr, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %b2degvar = stablehlo.divide %b2degvs, %b2degnf : tensor<32x96x56x56xf32>
    %b2degve = stablehlo.add %b2degvar, %b2degep : tensor<32x96x56x56xf32>
    %b2degistd = stablehlo.rsqrt %b2degve : tensor<32x96x56x56xf32>
    %b2degxh = stablehlo.multiply %b2degxc, %b2degistd : tensor<32x96x56x56xf32>
    %b2degp = stablehlo.multiply %b2degdyr, %b2degxh : tensor<32x96x56x56xf32>
    %b2deg = stablehlo.reduce(%b2degp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2debt = stablehlo.reduce(%b2degdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b1dpWxi = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %b1dpWdi = stablehlo.reshape %v1230 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b1dpWxt = stablehlo.transpose %b1dpWxi, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %b1dpWdt = stablehlo.transpose %b1dpWdi, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b1dpWraw = stablehlo.convolution(%b1dpWxt, %b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<64x24x1x1xf32>
    %b1dpW = stablehlo.transpose %b1dpWraw, dims = [1, 0, 2, 3] : (tensor<64x24x1x1xf32>) -> tensor<24x64x1x1xf32>
    %b1dpbi = stablehlo.reshape %v1230 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b1dpb = stablehlo.reduce(%b1dpbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b1dpgxr = stablehlo.reshape %v91 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b1dpgdyr = stablehlo.reshape %v1200 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b1dpgnf = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %b1dpgep = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %b1dpgsmr = stablehlo.reduce(%b1dpgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %b1dpgsm = stablehlo.broadcast_in_dim %b1dpgsmr, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %b1dpgmu = stablehlo.divide %b1dpgsm, %b1dpgnf : tensor<32x24x56x56xf32>
    %b1dpgxc = stablehlo.subtract %b1dpgxr, %b1dpgmu : tensor<32x24x56x56xf32>
    %b1dpgsq = stablehlo.multiply %b1dpgxc, %b1dpgxc : tensor<32x24x56x56xf32>
    %b1dpgvsr = stablehlo.reduce(%b1dpgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %b1dpgvs = stablehlo.broadcast_in_dim %b1dpgvsr, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %b1dpgvar = stablehlo.divide %b1dpgvs, %b1dpgnf : tensor<32x24x56x56xf32>
    %b1dpgve = stablehlo.add %b1dpgvar, %b1dpgep : tensor<32x24x56x56xf32>
    %b1dpgistd = stablehlo.rsqrt %b1dpgve : tensor<32x24x56x56xf32>
    %b1dpgxh = stablehlo.multiply %b1dpgxc, %b1dpgistd : tensor<32x24x56x56xf32>
    %b1dpgp = stablehlo.multiply %b1dpgdyr, %b1dpgxh : tensor<32x24x56x56xf32>
    %b1dpg = stablehlo.reduce(%b1dpgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b1dpbt = stablehlo.reduce(%b1dpgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b1ddui = stablehlo.reshape %v1271 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %b1ddup = stablehlo.pad %b1ddui, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %b1ddu = stablehlo.reshape %b1ddup : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %b1ddWxi = stablehlo.reshape %v57 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %b1ddWdi = stablehlo.reshape %b1ddu : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %b1ddWxt = stablehlo.transpose %b1ddWxi, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %b1ddWdt = stablehlo.transpose %b1ddWdi, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %b1ddWraw = stablehlo.convolution(%b1ddWxt, %b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 64 : i64, feature_group_count = 1 : i64} : (tensor<64x32x112x112xf32>, tensor<64x32x112x112xf32>) -> tensor<1x64x3x3xf32>
    %b1ddW = stablehlo.reshape %b1ddWraw : (tensor<1x64x3x3xf32>) -> tensor<64x1x3x3xf32>
    %b1ddbi = stablehlo.reshape %v1271 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %b1ddb = stablehlo.reduce(%b1ddbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %b1ddgxr = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %b1ddgdyr = stablehlo.reshape %v1241 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %b1ddgnf = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %b1ddgep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %b1ddgsmr = stablehlo.reduce(%b1ddgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %b1ddgsm = stablehlo.broadcast_in_dim %b1ddgsmr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %b1ddgmu = stablehlo.divide %b1ddgsm, %b1ddgnf : tensor<32x64x56x56xf32>
    %b1ddgxc = stablehlo.subtract %b1ddgxr, %b1ddgmu : tensor<32x64x56x56xf32>
    %b1ddgsq = stablehlo.multiply %b1ddgxc, %b1ddgxc : tensor<32x64x56x56xf32>
    %b1ddgvsr = stablehlo.reduce(%b1ddgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %b1ddgvs = stablehlo.broadcast_in_dim %b1ddgvsr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %b1ddgvar = stablehlo.divide %b1ddgvs, %b1ddgnf : tensor<32x64x56x56xf32>
    %b1ddgve = stablehlo.add %b1ddgvar, %b1ddgep : tensor<32x64x56x56xf32>
    %b1ddgistd = stablehlo.rsqrt %b1ddgve : tensor<32x64x56x56xf32>
    %b1ddgxh = stablehlo.multiply %b1ddgxc, %b1ddgistd : tensor<32x64x56x56xf32>
    %b1ddgp = stablehlo.multiply %b1ddgdyr, %b1ddgxh : tensor<32x64x56x56xf32>
    %b1ddg = stablehlo.reduce(%b1ddgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %b1ddbt = stablehlo.reduce(%b1ddgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %b1deWxi = stablehlo.reshape %v28 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %b1deWdi = stablehlo.reshape %v1313 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %b1deWxt = stablehlo.transpose %b1deWxi, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %b1deWdt = stablehlo.transpose %b1deWdi, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %b1deWraw = stablehlo.convolution(%b1deWxt, %b1deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<64x32x112x112xf32>) -> tensor<16x64x1x1xf32>
    %b1deW = stablehlo.transpose %b1deWraw, dims = [1, 0, 2, 3] : (tensor<16x64x1x1xf32>) -> tensor<64x16x1x1xf32>
    %b1debi = stablehlo.reshape %v1313 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %b1deb = stablehlo.reduce(%b1debi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %b1degxr = stablehlo.reshape %v33 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %b1degdyr = stablehlo.reshape %v1283 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %b1degnf = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %b1degep = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %b1degsmr = stablehlo.reduce(%b1degxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %b1degsm = stablehlo.broadcast_in_dim %b1degsmr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %b1degmu = stablehlo.divide %b1degsm, %b1degnf : tensor<32x64x112x112xf32>
    %b1degxc = stablehlo.subtract %b1degxr, %b1degmu : tensor<32x64x112x112xf32>
    %b1degsq = stablehlo.multiply %b1degxc, %b1degxc : tensor<32x64x112x112xf32>
    %b1degvsr = stablehlo.reduce(%b1degsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %b1degvs = stablehlo.broadcast_in_dim %b1degvsr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %b1degvar = stablehlo.divide %b1degvs, %b1degnf : tensor<32x64x112x112xf32>
    %b1degve = stablehlo.add %b1degvar, %b1degep : tensor<32x64x112x112xf32>
    %b1degistd = stablehlo.rsqrt %b1degve : tensor<32x64x112x112xf32>
    %b1degxh = stablehlo.multiply %b1degxc, %b1degistd : tensor<32x64x112x112xf32>
    %b1degp = stablehlo.multiply %b1degdyr, %b1degxh : tensor<32x64x112x112xf32>
    %b1deg = stablehlo.reduce(%b1degp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %b1debt = stablehlo.reduce(%b1degdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %dhWxi = stablehlo.reshape %v528 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %dhWdi = stablehlo.reshape %v607 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %dhWxt = stablehlo.transpose %dhWxi, dims = [1, 0, 2, 3] : (tensor<32x64x7x7xf32>) -> tensor<64x32x7x7xf32>
    %dhWdt = stablehlo.transpose %dhWdi, dims = [1, 0, 2, 3] : (tensor<32x128x7x7xf32>) -> tensor<128x32x7x7xf32>
    %dhWraw = stablehlo.convolution(%dhWxt, %dhWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x7x7xf32>, tensor<128x32x7x7xf32>) -> tensor<64x128x1x1xf32>
    %dhW = stablehlo.transpose %dhWraw, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %dhbi = stablehlo.reshape %v607 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %dhb = stablehlo.reduce(%dhbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %dhgxr = stablehlo.reshape %v533 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %dhgdyr = stablehlo.reshape %v577 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %dhgnf = stablehlo.constant dense<49.0> : tensor<32x128x7x7xf32>
    %dhgep = stablehlo.constant dense<1.0e-5> : tensor<32x128x7x7xf32>
    %dhgsmr = stablehlo.reduce(%dhgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %dhgsm = stablehlo.broadcast_in_dim %dhgsmr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %dhgmu = stablehlo.divide %dhgsm, %dhgnf : tensor<32x128x7x7xf32>
    %dhgxc = stablehlo.subtract %dhgxr, %dhgmu : tensor<32x128x7x7xf32>
    %dhgsq = stablehlo.multiply %dhgxc, %dhgxc : tensor<32x128x7x7xf32>
    %dhgvsr = stablehlo.reduce(%dhgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %dhgvs = stablehlo.broadcast_in_dim %dhgvsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %dhgvar = stablehlo.divide %dhgvs, %dhgnf : tensor<32x128x7x7xf32>
    %dhgve = stablehlo.add %dhgvar, %dhgep : tensor<32x128x7x7xf32>
    %dhgistd = stablehlo.rsqrt %dhgve : tensor<32x128x7x7xf32>
    %dhgxh = stablehlo.multiply %dhgxc, %dhgistd : tensor<32x128x7x7xf32>
    %dhgp = stablehlo.multiply %dhgdyr, %dhgxh : tensor<32x128x7x7xf32>
    %dhg = stablehlo.reduce(%dhgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %dhbt = stablehlo.reduce(%dhgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %dsui = stablehlo.reshape %v1354 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %dsup = stablehlo.pad %dsui, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16x224x224xf32>
    %dsu = stablehlo.reshape %dsup : (tensor<32x16x224x224xf32>) -> tensor<32x802816xf32>
    %dsWxi = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %dsWdi = stablehlo.reshape %dsu : (tensor<32x802816xf32>) -> tensor<32x16x224x224xf32>
    %dsWxt = stablehlo.transpose %dsWxi, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %dsWdt = stablehlo.transpose %dsWdi, dims = [1, 0, 2, 3] : (tensor<32x16x224x224xf32>) -> tensor<16x32x224x224xf32>
    %dsWraw = stablehlo.convolution(%dsWxt, %dsWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<16x32x224x224xf32>) -> tensor<3x16x3x3xf32>
    %dsW = stablehlo.transpose %dsWraw, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %dsbi = stablehlo.reshape %v1354 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %dsb = stablehlo.reduce(%dsbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %dsgxr = stablehlo.reshape %v4 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %dsgdyr = stablehlo.reshape %v1324 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %dsgnf = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %dsgep = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %dsgsmr = stablehlo.reduce(%dsgxr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %dsgsm = stablehlo.broadcast_in_dim %dsgsmr, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %dsgmu = stablehlo.divide %dsgsm, %dsgnf : tensor<32x16x112x112xf32>
    %dsgxc = stablehlo.subtract %dsgxr, %dsgmu : tensor<32x16x112x112xf32>
    %dsgsq = stablehlo.multiply %dsgxc, %dsgxc : tensor<32x16x112x112xf32>
    %dsgvsr = stablehlo.reduce(%dsgsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %dsgvs = stablehlo.broadcast_in_dim %dsgvsr, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %dsgvar = stablehlo.divide %dsgvs, %dsgnf : tensor<32x16x112x112xf32>
    %dsgve = stablehlo.add %dsgvar, %dsgep : tensor<32x16x112x112xf32>
    %dsgistd = stablehlo.rsqrt %dsgve : tensor<32x16x112x112xf32>
    %dsgxh = stablehlo.multiply %dsgxc, %dsgistd : tensor<32x16x112x112xf32>
    %dsgp = stablehlo.multiply %dsgdyr, %dsgxh : tensor<32x16x112x112xf32>
    %dsg = stablehlo.reduce(%dsgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %dsbt = stablehlo.reduce(%dsgdyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %dWd = stablehlo.dot_general %v562, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x128xf32>, tensor<32x10xf32>) -> tensor<128x10xf32>
    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %adb1sW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %adob1sW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %admssW = stablehlo.multiply %adb1sW, %sWm : tensor<16x3x3x3xf32>
    %admgsW = stablehlo.multiply %adob1sW, %dsW : tensor<16x3x3x3xf32>
    %admnsW = stablehlo.add %admssW, %admgsW : tensor<16x3x3x3xf32>
    %adb2sW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %adob2sW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %advssW = stablehlo.multiply %adb2sW, %sWv : tensor<16x3x3x3xf32>
    %adg2sW = stablehlo.multiply %dsW, %dsW : tensor<16x3x3x3xf32>
    %advgsW = stablehlo.multiply %adob2sW, %adg2sW : tensor<16x3x3x3xf32>
    %advnsW = stablehlo.add %advssW, %advgsW : tensor<16x3x3x3xf32>
    %adbc1sW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %adbc2sW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %admhsW = stablehlo.divide %admnsW, %adbc1sW : tensor<16x3x3x3xf32>
    %advhsW = stablehlo.divide %advnsW, %adbc2sW : tensor<16x3x3x3xf32>
    %adlrsW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %adepssW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %adsqsW = stablehlo.sqrt %advhsW : tensor<16x3x3x3xf32>
    %addensW = stablehlo.add %adsqsW, %adepssW : tensor<16x3x3x3xf32>
    %adratsW = stablehlo.divide %admhsW, %addensW : tensor<16x3x3x3xf32>
    %adstsW = stablehlo.multiply %adlrsW, %adratsW : tensor<16x3x3x3xf32>
    %adsubsW = stablehlo.subtract %sW, %adstsW : tensor<16x3x3x3xf32>
    %adwdsW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %adwdlrsW = stablehlo.multiply %adwdsW, %adlrsW : tensor<16x3x3x3xf32>
    %adwdpsW = stablehlo.multiply %adwdlrsW, %sW : tensor<16x3x3x3xf32>
    %adnewsW = stablehlo.subtract %adsubsW, %adwdpsW : tensor<16x3x3x3xf32>
    %adb1sb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1sb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admssb = stablehlo.multiply %adb1sb, %sbm : tensor<16xf32>
    %admgsb = stablehlo.multiply %adob1sb, %dsb : tensor<16xf32>
    %admnsb = stablehlo.add %admssb, %admgsb : tensor<16xf32>
    %adb2sb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2sb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advssb = stablehlo.multiply %adb2sb, %sbv : tensor<16xf32>
    %adg2sb = stablehlo.multiply %dsb, %dsb : tensor<16xf32>
    %advgsb = stablehlo.multiply %adob2sb, %adg2sb : tensor<16xf32>
    %advnsb = stablehlo.add %advssb, %advgsb : tensor<16xf32>
    %adbc1sb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2sb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhsb = stablehlo.divide %admnsb, %adbc1sb : tensor<16xf32>
    %advhsb = stablehlo.divide %advnsb, %adbc2sb : tensor<16xf32>
    %adlrsb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepssb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqsb = stablehlo.sqrt %advhsb : tensor<16xf32>
    %addensb = stablehlo.add %adsqsb, %adepssb : tensor<16xf32>
    %adratsb = stablehlo.divide %admhsb, %addensb : tensor<16xf32>
    %adstsb = stablehlo.multiply %adlrsb, %adratsb : tensor<16xf32>
    %adsubsb = stablehlo.subtract %sb, %adstsb : tensor<16xf32>
    %adwdsb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrsb = stablehlo.multiply %adwdsb, %adlrsb : tensor<16xf32>
    %adwdpsb = stablehlo.multiply %adwdlrsb, %sb : tensor<16xf32>
    %adnewsb = stablehlo.subtract %adsubsb, %adwdpsb : tensor<16xf32>
    %adb1sg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1sg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admssg = stablehlo.multiply %adb1sg, %sgm : tensor<16xf32>
    %admgsg = stablehlo.multiply %adob1sg, %dsg : tensor<16xf32>
    %admnsg = stablehlo.add %admssg, %admgsg : tensor<16xf32>
    %adb2sg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2sg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advssg = stablehlo.multiply %adb2sg, %sgv : tensor<16xf32>
    %adg2sg = stablehlo.multiply %dsg, %dsg : tensor<16xf32>
    %advgsg = stablehlo.multiply %adob2sg, %adg2sg : tensor<16xf32>
    %advnsg = stablehlo.add %advssg, %advgsg : tensor<16xf32>
    %adbc1sg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2sg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhsg = stablehlo.divide %admnsg, %adbc1sg : tensor<16xf32>
    %advhsg = stablehlo.divide %advnsg, %adbc2sg : tensor<16xf32>
    %adlrsg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepssg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqsg = stablehlo.sqrt %advhsg : tensor<16xf32>
    %addensg = stablehlo.add %adsqsg, %adepssg : tensor<16xf32>
    %adratsg = stablehlo.divide %admhsg, %addensg : tensor<16xf32>
    %adstsg = stablehlo.multiply %adlrsg, %adratsg : tensor<16xf32>
    %adsubsg = stablehlo.subtract %sg, %adstsg : tensor<16xf32>
    %adwdsg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrsg = stablehlo.multiply %adwdsg, %adlrsg : tensor<16xf32>
    %adwdpsg = stablehlo.multiply %adwdlrsg, %sg : tensor<16xf32>
    %adnewsg = stablehlo.subtract %adsubsg, %adwdpsg : tensor<16xf32>
    %adb1sbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1sbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admssbt = stablehlo.multiply %adb1sbt, %sbtm : tensor<16xf32>
    %admgsbt = stablehlo.multiply %adob1sbt, %dsbt : tensor<16xf32>
    %admnsbt = stablehlo.add %admssbt, %admgsbt : tensor<16xf32>
    %adb2sbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2sbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advssbt = stablehlo.multiply %adb2sbt, %sbtv : tensor<16xf32>
    %adg2sbt = stablehlo.multiply %dsbt, %dsbt : tensor<16xf32>
    %advgsbt = stablehlo.multiply %adob2sbt, %adg2sbt : tensor<16xf32>
    %advnsbt = stablehlo.add %advssbt, %advgsbt : tensor<16xf32>
    %adbc1sbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2sbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhsbt = stablehlo.divide %admnsbt, %adbc1sbt : tensor<16xf32>
    %advhsbt = stablehlo.divide %advnsbt, %adbc2sbt : tensor<16xf32>
    %adlrsbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepssbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqsbt = stablehlo.sqrt %advhsbt : tensor<16xf32>
    %addensbt = stablehlo.add %adsqsbt, %adepssbt : tensor<16xf32>
    %adratsbt = stablehlo.divide %admhsbt, %addensbt : tensor<16xf32>
    %adstsbt = stablehlo.multiply %adlrsbt, %adratsbt : tensor<16xf32>
    %adsubsbt = stablehlo.subtract %sbt, %adstsbt : tensor<16xf32>
    %adwdsbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrsbt = stablehlo.multiply %adwdsbt, %adlrsbt : tensor<16xf32>
    %adwdpsbt = stablehlo.multiply %adwdlrsbt, %sbt : tensor<16xf32>
    %adnewsbt = stablehlo.subtract %adsubsbt, %adwdpsbt : tensor<16xf32>
    %adb1b1eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x16x1x1xf32>
    %adob1b1eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x16x1x1xf32>
    %admsb1eW = stablehlo.multiply %adb1b1eW, %b1eWm : tensor<64x16x1x1xf32>
    %admgb1eW = stablehlo.multiply %adob1b1eW, %b1deW : tensor<64x16x1x1xf32>
    %admnb1eW = stablehlo.add %admsb1eW, %admgb1eW : tensor<64x16x1x1xf32>
    %adb2b1eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x16x1x1xf32>
    %adob2b1eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x16x1x1xf32>
    %advsb1eW = stablehlo.multiply %adb2b1eW, %b1eWv : tensor<64x16x1x1xf32>
    %adg2b1eW = stablehlo.multiply %b1deW, %b1deW : tensor<64x16x1x1xf32>
    %advgb1eW = stablehlo.multiply %adob2b1eW, %adg2b1eW : tensor<64x16x1x1xf32>
    %advnb1eW = stablehlo.add %advsb1eW, %advgb1eW : tensor<64x16x1x1xf32>
    %adbc1b1eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x16x1x1xf32>
    %adbc2b1eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x16x1x1xf32>
    %admhb1eW = stablehlo.divide %admnb1eW, %adbc1b1eW : tensor<64x16x1x1xf32>
    %advhb1eW = stablehlo.divide %advnb1eW, %adbc2b1eW : tensor<64x16x1x1xf32>
    %adlrb1eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x16x1x1xf32>
    %adepsb1eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x16x1x1xf32>
    %adsqb1eW = stablehlo.sqrt %advhb1eW : tensor<64x16x1x1xf32>
    %addenb1eW = stablehlo.add %adsqb1eW, %adepsb1eW : tensor<64x16x1x1xf32>
    %adratb1eW = stablehlo.divide %admhb1eW, %addenb1eW : tensor<64x16x1x1xf32>
    %adstb1eW = stablehlo.multiply %adlrb1eW, %adratb1eW : tensor<64x16x1x1xf32>
    %adsubb1eW = stablehlo.subtract %b1eW, %adstb1eW : tensor<64x16x1x1xf32>
    %adwdb1eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x16x1x1xf32>
    %adwdlrb1eW = stablehlo.multiply %adwdb1eW, %adlrb1eW : tensor<64x16x1x1xf32>
    %adwdpb1eW = stablehlo.multiply %adwdlrb1eW, %b1eW : tensor<64x16x1x1xf32>
    %adnewb1eW = stablehlo.subtract %adsubb1eW, %adwdpb1eW : tensor<64x16x1x1xf32>
    %adb1b1eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b1eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb1eb = stablehlo.multiply %adb1b1eb, %b1ebm : tensor<64xf32>
    %admgb1eb = stablehlo.multiply %adob1b1eb, %b1deb : tensor<64xf32>
    %admnb1eb = stablehlo.add %admsb1eb, %admgb1eb : tensor<64xf32>
    %adb2b1eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b1eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb1eb = stablehlo.multiply %adb2b1eb, %b1ebv : tensor<64xf32>
    %adg2b1eb = stablehlo.multiply %b1deb, %b1deb : tensor<64xf32>
    %advgb1eb = stablehlo.multiply %adob2b1eb, %adg2b1eb : tensor<64xf32>
    %advnb1eb = stablehlo.add %advsb1eb, %advgb1eb : tensor<64xf32>
    %adbc1b1eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b1eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb1eb = stablehlo.divide %admnb1eb, %adbc1b1eb : tensor<64xf32>
    %advhb1eb = stablehlo.divide %advnb1eb, %adbc2b1eb : tensor<64xf32>
    %adlrb1eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb1eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb1eb = stablehlo.sqrt %advhb1eb : tensor<64xf32>
    %addenb1eb = stablehlo.add %adsqb1eb, %adepsb1eb : tensor<64xf32>
    %adratb1eb = stablehlo.divide %admhb1eb, %addenb1eb : tensor<64xf32>
    %adstb1eb = stablehlo.multiply %adlrb1eb, %adratb1eb : tensor<64xf32>
    %adsubb1eb = stablehlo.subtract %b1eb, %adstb1eb : tensor<64xf32>
    %adwdb1eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb1eb = stablehlo.multiply %adwdb1eb, %adlrb1eb : tensor<64xf32>
    %adwdpb1eb = stablehlo.multiply %adwdlrb1eb, %b1eb : tensor<64xf32>
    %adnewb1eb = stablehlo.subtract %adsubb1eb, %adwdpb1eb : tensor<64xf32>
    %adb1b1eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b1eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb1eg = stablehlo.multiply %adb1b1eg, %b1egm : tensor<64xf32>
    %admgb1eg = stablehlo.multiply %adob1b1eg, %b1deg : tensor<64xf32>
    %admnb1eg = stablehlo.add %admsb1eg, %admgb1eg : tensor<64xf32>
    %adb2b1eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b1eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb1eg = stablehlo.multiply %adb2b1eg, %b1egv : tensor<64xf32>
    %adg2b1eg = stablehlo.multiply %b1deg, %b1deg : tensor<64xf32>
    %advgb1eg = stablehlo.multiply %adob2b1eg, %adg2b1eg : tensor<64xf32>
    %advnb1eg = stablehlo.add %advsb1eg, %advgb1eg : tensor<64xf32>
    %adbc1b1eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b1eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb1eg = stablehlo.divide %admnb1eg, %adbc1b1eg : tensor<64xf32>
    %advhb1eg = stablehlo.divide %advnb1eg, %adbc2b1eg : tensor<64xf32>
    %adlrb1eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb1eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb1eg = stablehlo.sqrt %advhb1eg : tensor<64xf32>
    %addenb1eg = stablehlo.add %adsqb1eg, %adepsb1eg : tensor<64xf32>
    %adratb1eg = stablehlo.divide %admhb1eg, %addenb1eg : tensor<64xf32>
    %adstb1eg = stablehlo.multiply %adlrb1eg, %adratb1eg : tensor<64xf32>
    %adsubb1eg = stablehlo.subtract %b1eg, %adstb1eg : tensor<64xf32>
    %adwdb1eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb1eg = stablehlo.multiply %adwdb1eg, %adlrb1eg : tensor<64xf32>
    %adwdpb1eg = stablehlo.multiply %adwdlrb1eg, %b1eg : tensor<64xf32>
    %adnewb1eg = stablehlo.subtract %adsubb1eg, %adwdpb1eg : tensor<64xf32>
    %adb1b1ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b1ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb1ebt = stablehlo.multiply %adb1b1ebt, %b1ebtm : tensor<64xf32>
    %admgb1ebt = stablehlo.multiply %adob1b1ebt, %b1debt : tensor<64xf32>
    %admnb1ebt = stablehlo.add %admsb1ebt, %admgb1ebt : tensor<64xf32>
    %adb2b1ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b1ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb1ebt = stablehlo.multiply %adb2b1ebt, %b1ebtv : tensor<64xf32>
    %adg2b1ebt = stablehlo.multiply %b1debt, %b1debt : tensor<64xf32>
    %advgb1ebt = stablehlo.multiply %adob2b1ebt, %adg2b1ebt : tensor<64xf32>
    %advnb1ebt = stablehlo.add %advsb1ebt, %advgb1ebt : tensor<64xf32>
    %adbc1b1ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b1ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb1ebt = stablehlo.divide %admnb1ebt, %adbc1b1ebt : tensor<64xf32>
    %advhb1ebt = stablehlo.divide %advnb1ebt, %adbc2b1ebt : tensor<64xf32>
    %adlrb1ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb1ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb1ebt = stablehlo.sqrt %advhb1ebt : tensor<64xf32>
    %addenb1ebt = stablehlo.add %adsqb1ebt, %adepsb1ebt : tensor<64xf32>
    %adratb1ebt = stablehlo.divide %admhb1ebt, %addenb1ebt : tensor<64xf32>
    %adstb1ebt = stablehlo.multiply %adlrb1ebt, %adratb1ebt : tensor<64xf32>
    %adsubb1ebt = stablehlo.subtract %b1ebt, %adstb1ebt : tensor<64xf32>
    %adwdb1ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb1ebt = stablehlo.multiply %adwdb1ebt, %adlrb1ebt : tensor<64xf32>
    %adwdpb1ebt = stablehlo.multiply %adwdlrb1ebt, %b1ebt : tensor<64xf32>
    %adnewb1ebt = stablehlo.subtract %adsubb1ebt, %adwdpb1ebt : tensor<64xf32>
    %adb1b1dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x1x3x3xf32>
    %adob1b1dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x1x3x3xf32>
    %admsb1dW = stablehlo.multiply %adb1b1dW, %b1dWm : tensor<64x1x3x3xf32>
    %admgb1dW = stablehlo.multiply %adob1b1dW, %b1ddW : tensor<64x1x3x3xf32>
    %admnb1dW = stablehlo.add %admsb1dW, %admgb1dW : tensor<64x1x3x3xf32>
    %adb2b1dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x1x3x3xf32>
    %adob2b1dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x1x3x3xf32>
    %advsb1dW = stablehlo.multiply %adb2b1dW, %b1dWv : tensor<64x1x3x3xf32>
    %adg2b1dW = stablehlo.multiply %b1ddW, %b1ddW : tensor<64x1x3x3xf32>
    %advgb1dW = stablehlo.multiply %adob2b1dW, %adg2b1dW : tensor<64x1x3x3xf32>
    %advnb1dW = stablehlo.add %advsb1dW, %advgb1dW : tensor<64x1x3x3xf32>
    %adbc1b1dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x1x3x3xf32>
    %adbc2b1dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x1x3x3xf32>
    %admhb1dW = stablehlo.divide %admnb1dW, %adbc1b1dW : tensor<64x1x3x3xf32>
    %advhb1dW = stablehlo.divide %advnb1dW, %adbc2b1dW : tensor<64x1x3x3xf32>
    %adlrb1dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x1x3x3xf32>
    %adepsb1dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x1x3x3xf32>
    %adsqb1dW = stablehlo.sqrt %advhb1dW : tensor<64x1x3x3xf32>
    %addenb1dW = stablehlo.add %adsqb1dW, %adepsb1dW : tensor<64x1x3x3xf32>
    %adratb1dW = stablehlo.divide %admhb1dW, %addenb1dW : tensor<64x1x3x3xf32>
    %adstb1dW = stablehlo.multiply %adlrb1dW, %adratb1dW : tensor<64x1x3x3xf32>
    %adsubb1dW = stablehlo.subtract %b1dW, %adstb1dW : tensor<64x1x3x3xf32>
    %adwdb1dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x1x3x3xf32>
    %adwdlrb1dW = stablehlo.multiply %adwdb1dW, %adlrb1dW : tensor<64x1x3x3xf32>
    %adwdpb1dW = stablehlo.multiply %adwdlrb1dW, %b1dW : tensor<64x1x3x3xf32>
    %adnewb1dW = stablehlo.subtract %adsubb1dW, %adwdpb1dW : tensor<64x1x3x3xf32>
    %adb1b1db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b1db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb1db = stablehlo.multiply %adb1b1db, %b1dbm : tensor<64xf32>
    %admgb1db = stablehlo.multiply %adob1b1db, %b1ddb : tensor<64xf32>
    %admnb1db = stablehlo.add %admsb1db, %admgb1db : tensor<64xf32>
    %adb2b1db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b1db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb1db = stablehlo.multiply %adb2b1db, %b1dbv : tensor<64xf32>
    %adg2b1db = stablehlo.multiply %b1ddb, %b1ddb : tensor<64xf32>
    %advgb1db = stablehlo.multiply %adob2b1db, %adg2b1db : tensor<64xf32>
    %advnb1db = stablehlo.add %advsb1db, %advgb1db : tensor<64xf32>
    %adbc1b1db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b1db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb1db = stablehlo.divide %admnb1db, %adbc1b1db : tensor<64xf32>
    %advhb1db = stablehlo.divide %advnb1db, %adbc2b1db : tensor<64xf32>
    %adlrb1db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb1db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb1db = stablehlo.sqrt %advhb1db : tensor<64xf32>
    %addenb1db = stablehlo.add %adsqb1db, %adepsb1db : tensor<64xf32>
    %adratb1db = stablehlo.divide %admhb1db, %addenb1db : tensor<64xf32>
    %adstb1db = stablehlo.multiply %adlrb1db, %adratb1db : tensor<64xf32>
    %adsubb1db = stablehlo.subtract %b1db, %adstb1db : tensor<64xf32>
    %adwdb1db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb1db = stablehlo.multiply %adwdb1db, %adlrb1db : tensor<64xf32>
    %adwdpb1db = stablehlo.multiply %adwdlrb1db, %b1db : tensor<64xf32>
    %adnewb1db = stablehlo.subtract %adsubb1db, %adwdpb1db : tensor<64xf32>
    %adb1b1dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b1dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb1dg = stablehlo.multiply %adb1b1dg, %b1dgm : tensor<64xf32>
    %admgb1dg = stablehlo.multiply %adob1b1dg, %b1ddg : tensor<64xf32>
    %admnb1dg = stablehlo.add %admsb1dg, %admgb1dg : tensor<64xf32>
    %adb2b1dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b1dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb1dg = stablehlo.multiply %adb2b1dg, %b1dgv : tensor<64xf32>
    %adg2b1dg = stablehlo.multiply %b1ddg, %b1ddg : tensor<64xf32>
    %advgb1dg = stablehlo.multiply %adob2b1dg, %adg2b1dg : tensor<64xf32>
    %advnb1dg = stablehlo.add %advsb1dg, %advgb1dg : tensor<64xf32>
    %adbc1b1dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b1dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb1dg = stablehlo.divide %admnb1dg, %adbc1b1dg : tensor<64xf32>
    %advhb1dg = stablehlo.divide %advnb1dg, %adbc2b1dg : tensor<64xf32>
    %adlrb1dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb1dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb1dg = stablehlo.sqrt %advhb1dg : tensor<64xf32>
    %addenb1dg = stablehlo.add %adsqb1dg, %adepsb1dg : tensor<64xf32>
    %adratb1dg = stablehlo.divide %admhb1dg, %addenb1dg : tensor<64xf32>
    %adstb1dg = stablehlo.multiply %adlrb1dg, %adratb1dg : tensor<64xf32>
    %adsubb1dg = stablehlo.subtract %b1dg, %adstb1dg : tensor<64xf32>
    %adwdb1dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb1dg = stablehlo.multiply %adwdb1dg, %adlrb1dg : tensor<64xf32>
    %adwdpb1dg = stablehlo.multiply %adwdlrb1dg, %b1dg : tensor<64xf32>
    %adnewb1dg = stablehlo.subtract %adsubb1dg, %adwdpb1dg : tensor<64xf32>
    %adb1b1dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b1dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb1dbt = stablehlo.multiply %adb1b1dbt, %b1dbtm : tensor<64xf32>
    %admgb1dbt = stablehlo.multiply %adob1b1dbt, %b1ddbt : tensor<64xf32>
    %admnb1dbt = stablehlo.add %admsb1dbt, %admgb1dbt : tensor<64xf32>
    %adb2b1dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b1dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb1dbt = stablehlo.multiply %adb2b1dbt, %b1dbtv : tensor<64xf32>
    %adg2b1dbt = stablehlo.multiply %b1ddbt, %b1ddbt : tensor<64xf32>
    %advgb1dbt = stablehlo.multiply %adob2b1dbt, %adg2b1dbt : tensor<64xf32>
    %advnb1dbt = stablehlo.add %advsb1dbt, %advgb1dbt : tensor<64xf32>
    %adbc1b1dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b1dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb1dbt = stablehlo.divide %admnb1dbt, %adbc1b1dbt : tensor<64xf32>
    %advhb1dbt = stablehlo.divide %advnb1dbt, %adbc2b1dbt : tensor<64xf32>
    %adlrb1dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb1dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb1dbt = stablehlo.sqrt %advhb1dbt : tensor<64xf32>
    %addenb1dbt = stablehlo.add %adsqb1dbt, %adepsb1dbt : tensor<64xf32>
    %adratb1dbt = stablehlo.divide %admhb1dbt, %addenb1dbt : tensor<64xf32>
    %adstb1dbt = stablehlo.multiply %adlrb1dbt, %adratb1dbt : tensor<64xf32>
    %adsubb1dbt = stablehlo.subtract %b1dbt, %adstb1dbt : tensor<64xf32>
    %adwdb1dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb1dbt = stablehlo.multiply %adwdb1dbt, %adlrb1dbt : tensor<64xf32>
    %adwdpb1dbt = stablehlo.multiply %adwdlrb1dbt, %b1dbt : tensor<64xf32>
    %adnewb1dbt = stablehlo.subtract %adsubb1dbt, %adwdpb1dbt : tensor<64xf32>
    %adb1b1pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24x64x1x1xf32>
    %adob1b1pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24x64x1x1xf32>
    %admsb1pW = stablehlo.multiply %adb1b1pW, %b1pWm : tensor<24x64x1x1xf32>
    %admgb1pW = stablehlo.multiply %adob1b1pW, %b1dpW : tensor<24x64x1x1xf32>
    %admnb1pW = stablehlo.add %admsb1pW, %admgb1pW : tensor<24x64x1x1xf32>
    %adb2b1pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24x64x1x1xf32>
    %adob2b1pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24x64x1x1xf32>
    %advsb1pW = stablehlo.multiply %adb2b1pW, %b1pWv : tensor<24x64x1x1xf32>
    %adg2b1pW = stablehlo.multiply %b1dpW, %b1dpW : tensor<24x64x1x1xf32>
    %advgb1pW = stablehlo.multiply %adob2b1pW, %adg2b1pW : tensor<24x64x1x1xf32>
    %advnb1pW = stablehlo.add %advsb1pW, %advgb1pW : tensor<24x64x1x1xf32>
    %adbc1b1pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24x64x1x1xf32>
    %adbc2b1pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24x64x1x1xf32>
    %admhb1pW = stablehlo.divide %admnb1pW, %adbc1b1pW : tensor<24x64x1x1xf32>
    %advhb1pW = stablehlo.divide %advnb1pW, %adbc2b1pW : tensor<24x64x1x1xf32>
    %adlrb1pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24x64x1x1xf32>
    %adepsb1pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24x64x1x1xf32>
    %adsqb1pW = stablehlo.sqrt %advhb1pW : tensor<24x64x1x1xf32>
    %addenb1pW = stablehlo.add %adsqb1pW, %adepsb1pW : tensor<24x64x1x1xf32>
    %adratb1pW = stablehlo.divide %admhb1pW, %addenb1pW : tensor<24x64x1x1xf32>
    %adstb1pW = stablehlo.multiply %adlrb1pW, %adratb1pW : tensor<24x64x1x1xf32>
    %adsubb1pW = stablehlo.subtract %b1pW, %adstb1pW : tensor<24x64x1x1xf32>
    %adwdb1pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24x64x1x1xf32>
    %adwdlrb1pW = stablehlo.multiply %adwdb1pW, %adlrb1pW : tensor<24x64x1x1xf32>
    %adwdpb1pW = stablehlo.multiply %adwdlrb1pW, %b1pW : tensor<24x64x1x1xf32>
    %adnewb1pW = stablehlo.subtract %adsubb1pW, %adwdpb1pW : tensor<24x64x1x1xf32>
    %adb1b1pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob1b1pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admsb1pb = stablehlo.multiply %adb1b1pb, %b1pbm : tensor<24xf32>
    %admgb1pb = stablehlo.multiply %adob1b1pb, %b1dpb : tensor<24xf32>
    %admnb1pb = stablehlo.add %admsb1pb, %admgb1pb : tensor<24xf32>
    %adb2b1pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b1pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb1pb = stablehlo.multiply %adb2b1pb, %b1pbv : tensor<24xf32>
    %adg2b1pb = stablehlo.multiply %b1dpb, %b1dpb : tensor<24xf32>
    %advgb1pb = stablehlo.multiply %adob2b1pb, %adg2b1pb : tensor<24xf32>
    %advnb1pb = stablehlo.add %advsb1pb, %advgb1pb : tensor<24xf32>
    %adbc1b1pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adbc2b1pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admhb1pb = stablehlo.divide %admnb1pb, %adbc1b1pb : tensor<24xf32>
    %advhb1pb = stablehlo.divide %advnb1pb, %adbc2b1pb : tensor<24xf32>
    %adlrb1pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adepsb1pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adsqb1pb = stablehlo.sqrt %advhb1pb : tensor<24xf32>
    %addenb1pb = stablehlo.add %adsqb1pb, %adepsb1pb : tensor<24xf32>
    %adratb1pb = stablehlo.divide %admhb1pb, %addenb1pb : tensor<24xf32>
    %adstb1pb = stablehlo.multiply %adlrb1pb, %adratb1pb : tensor<24xf32>
    %adsubb1pb = stablehlo.subtract %b1pb, %adstb1pb : tensor<24xf32>
    %adwdb1pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adwdlrb1pb = stablehlo.multiply %adwdb1pb, %adlrb1pb : tensor<24xf32>
    %adwdpb1pb = stablehlo.multiply %adwdlrb1pb, %b1pb : tensor<24xf32>
    %adnewb1pb = stablehlo.subtract %adsubb1pb, %adwdpb1pb : tensor<24xf32>
    %adb1b1pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob1b1pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admsb1pg = stablehlo.multiply %adb1b1pg, %b1pgm : tensor<24xf32>
    %admgb1pg = stablehlo.multiply %adob1b1pg, %b1dpg : tensor<24xf32>
    %admnb1pg = stablehlo.add %admsb1pg, %admgb1pg : tensor<24xf32>
    %adb2b1pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b1pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb1pg = stablehlo.multiply %adb2b1pg, %b1pgv : tensor<24xf32>
    %adg2b1pg = stablehlo.multiply %b1dpg, %b1dpg : tensor<24xf32>
    %advgb1pg = stablehlo.multiply %adob2b1pg, %adg2b1pg : tensor<24xf32>
    %advnb1pg = stablehlo.add %advsb1pg, %advgb1pg : tensor<24xf32>
    %adbc1b1pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adbc2b1pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admhb1pg = stablehlo.divide %admnb1pg, %adbc1b1pg : tensor<24xf32>
    %advhb1pg = stablehlo.divide %advnb1pg, %adbc2b1pg : tensor<24xf32>
    %adlrb1pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adepsb1pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adsqb1pg = stablehlo.sqrt %advhb1pg : tensor<24xf32>
    %addenb1pg = stablehlo.add %adsqb1pg, %adepsb1pg : tensor<24xf32>
    %adratb1pg = stablehlo.divide %admhb1pg, %addenb1pg : tensor<24xf32>
    %adstb1pg = stablehlo.multiply %adlrb1pg, %adratb1pg : tensor<24xf32>
    %adsubb1pg = stablehlo.subtract %b1pg, %adstb1pg : tensor<24xf32>
    %adwdb1pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adwdlrb1pg = stablehlo.multiply %adwdb1pg, %adlrb1pg : tensor<24xf32>
    %adwdpb1pg = stablehlo.multiply %adwdlrb1pg, %b1pg : tensor<24xf32>
    %adnewb1pg = stablehlo.subtract %adsubb1pg, %adwdpb1pg : tensor<24xf32>
    %adb1b1pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob1b1pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admsb1pbt = stablehlo.multiply %adb1b1pbt, %b1pbtm : tensor<24xf32>
    %admgb1pbt = stablehlo.multiply %adob1b1pbt, %b1dpbt : tensor<24xf32>
    %admnb1pbt = stablehlo.add %admsb1pbt, %admgb1pbt : tensor<24xf32>
    %adb2b1pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b1pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb1pbt = stablehlo.multiply %adb2b1pbt, %b1pbtv : tensor<24xf32>
    %adg2b1pbt = stablehlo.multiply %b1dpbt, %b1dpbt : tensor<24xf32>
    %advgb1pbt = stablehlo.multiply %adob2b1pbt, %adg2b1pbt : tensor<24xf32>
    %advnb1pbt = stablehlo.add %advsb1pbt, %advgb1pbt : tensor<24xf32>
    %adbc1b1pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adbc2b1pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admhb1pbt = stablehlo.divide %admnb1pbt, %adbc1b1pbt : tensor<24xf32>
    %advhb1pbt = stablehlo.divide %advnb1pbt, %adbc2b1pbt : tensor<24xf32>
    %adlrb1pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adepsb1pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adsqb1pbt = stablehlo.sqrt %advhb1pbt : tensor<24xf32>
    %addenb1pbt = stablehlo.add %adsqb1pbt, %adepsb1pbt : tensor<24xf32>
    %adratb1pbt = stablehlo.divide %admhb1pbt, %addenb1pbt : tensor<24xf32>
    %adstb1pbt = stablehlo.multiply %adlrb1pbt, %adratb1pbt : tensor<24xf32>
    %adsubb1pbt = stablehlo.subtract %b1pbt, %adstb1pbt : tensor<24xf32>
    %adwdb1pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adwdlrb1pbt = stablehlo.multiply %adwdb1pbt, %adlrb1pbt : tensor<24xf32>
    %adwdpb1pbt = stablehlo.multiply %adwdlrb1pbt, %b1pbt : tensor<24xf32>
    %adnewb1pbt = stablehlo.subtract %adsubb1pbt, %adwdpb1pbt : tensor<24xf32>
    %adb1b2eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %adob1b2eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %admsb2eW = stablehlo.multiply %adb1b2eW, %b2eWm : tensor<96x24x1x1xf32>
    %admgb2eW = stablehlo.multiply %adob1b2eW, %b2deW : tensor<96x24x1x1xf32>
    %admnb2eW = stablehlo.add %admsb2eW, %admgb2eW : tensor<96x24x1x1xf32>
    %adb2b2eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %adob2b2eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %advsb2eW = stablehlo.multiply %adb2b2eW, %b2eWv : tensor<96x24x1x1xf32>
    %adg2b2eW = stablehlo.multiply %b2deW, %b2deW : tensor<96x24x1x1xf32>
    %advgb2eW = stablehlo.multiply %adob2b2eW, %adg2b2eW : tensor<96x24x1x1xf32>
    %advnb2eW = stablehlo.add %advsb2eW, %advgb2eW : tensor<96x24x1x1xf32>
    %adbc1b2eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %adbc2b2eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %admhb2eW = stablehlo.divide %admnb2eW, %adbc1b2eW : tensor<96x24x1x1xf32>
    %advhb2eW = stablehlo.divide %advnb2eW, %adbc2b2eW : tensor<96x24x1x1xf32>
    %adlrb2eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %adepsb2eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %adsqb2eW = stablehlo.sqrt %advhb2eW : tensor<96x24x1x1xf32>
    %addenb2eW = stablehlo.add %adsqb2eW, %adepsb2eW : tensor<96x24x1x1xf32>
    %adratb2eW = stablehlo.divide %admhb2eW, %addenb2eW : tensor<96x24x1x1xf32>
    %adstb2eW = stablehlo.multiply %adlrb2eW, %adratb2eW : tensor<96x24x1x1xf32>
    %adsubb2eW = stablehlo.subtract %b2eW, %adstb2eW : tensor<96x24x1x1xf32>
    %adwdb2eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %adwdlrb2eW = stablehlo.multiply %adwdb2eW, %adlrb2eW : tensor<96x24x1x1xf32>
    %adwdpb2eW = stablehlo.multiply %adwdlrb2eW, %b2eW : tensor<96x24x1x1xf32>
    %adnewb2eW = stablehlo.subtract %adsubb2eW, %adwdpb2eW : tensor<96x24x1x1xf32>
    %adb1b2eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2eb = stablehlo.multiply %adb1b2eb, %b2ebm : tensor<96xf32>
    %admgb2eb = stablehlo.multiply %adob1b2eb, %b2deb : tensor<96xf32>
    %admnb2eb = stablehlo.add %admsb2eb, %admgb2eb : tensor<96xf32>
    %adb2b2eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2eb = stablehlo.multiply %adb2b2eb, %b2ebv : tensor<96xf32>
    %adg2b2eb = stablehlo.multiply %b2deb, %b2deb : tensor<96xf32>
    %advgb2eb = stablehlo.multiply %adob2b2eb, %adg2b2eb : tensor<96xf32>
    %advnb2eb = stablehlo.add %advsb2eb, %advgb2eb : tensor<96xf32>
    %adbc1b2eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2eb = stablehlo.divide %admnb2eb, %adbc1b2eb : tensor<96xf32>
    %advhb2eb = stablehlo.divide %advnb2eb, %adbc2b2eb : tensor<96xf32>
    %adlrb2eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2eb = stablehlo.sqrt %advhb2eb : tensor<96xf32>
    %addenb2eb = stablehlo.add %adsqb2eb, %adepsb2eb : tensor<96xf32>
    %adratb2eb = stablehlo.divide %admhb2eb, %addenb2eb : tensor<96xf32>
    %adstb2eb = stablehlo.multiply %adlrb2eb, %adratb2eb : tensor<96xf32>
    %adsubb2eb = stablehlo.subtract %b2eb, %adstb2eb : tensor<96xf32>
    %adwdb2eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2eb = stablehlo.multiply %adwdb2eb, %adlrb2eb : tensor<96xf32>
    %adwdpb2eb = stablehlo.multiply %adwdlrb2eb, %b2eb : tensor<96xf32>
    %adnewb2eb = stablehlo.subtract %adsubb2eb, %adwdpb2eb : tensor<96xf32>
    %adb1b2eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2eg = stablehlo.multiply %adb1b2eg, %b2egm : tensor<96xf32>
    %admgb2eg = stablehlo.multiply %adob1b2eg, %b2deg : tensor<96xf32>
    %admnb2eg = stablehlo.add %admsb2eg, %admgb2eg : tensor<96xf32>
    %adb2b2eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2eg = stablehlo.multiply %adb2b2eg, %b2egv : tensor<96xf32>
    %adg2b2eg = stablehlo.multiply %b2deg, %b2deg : tensor<96xf32>
    %advgb2eg = stablehlo.multiply %adob2b2eg, %adg2b2eg : tensor<96xf32>
    %advnb2eg = stablehlo.add %advsb2eg, %advgb2eg : tensor<96xf32>
    %adbc1b2eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2eg = stablehlo.divide %admnb2eg, %adbc1b2eg : tensor<96xf32>
    %advhb2eg = stablehlo.divide %advnb2eg, %adbc2b2eg : tensor<96xf32>
    %adlrb2eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2eg = stablehlo.sqrt %advhb2eg : tensor<96xf32>
    %addenb2eg = stablehlo.add %adsqb2eg, %adepsb2eg : tensor<96xf32>
    %adratb2eg = stablehlo.divide %admhb2eg, %addenb2eg : tensor<96xf32>
    %adstb2eg = stablehlo.multiply %adlrb2eg, %adratb2eg : tensor<96xf32>
    %adsubb2eg = stablehlo.subtract %b2eg, %adstb2eg : tensor<96xf32>
    %adwdb2eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2eg = stablehlo.multiply %adwdb2eg, %adlrb2eg : tensor<96xf32>
    %adwdpb2eg = stablehlo.multiply %adwdlrb2eg, %b2eg : tensor<96xf32>
    %adnewb2eg = stablehlo.subtract %adsubb2eg, %adwdpb2eg : tensor<96xf32>
    %adb1b2ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2ebt = stablehlo.multiply %adb1b2ebt, %b2ebtm : tensor<96xf32>
    %admgb2ebt = stablehlo.multiply %adob1b2ebt, %b2debt : tensor<96xf32>
    %admnb2ebt = stablehlo.add %admsb2ebt, %admgb2ebt : tensor<96xf32>
    %adb2b2ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2ebt = stablehlo.multiply %adb2b2ebt, %b2ebtv : tensor<96xf32>
    %adg2b2ebt = stablehlo.multiply %b2debt, %b2debt : tensor<96xf32>
    %advgb2ebt = stablehlo.multiply %adob2b2ebt, %adg2b2ebt : tensor<96xf32>
    %advnb2ebt = stablehlo.add %advsb2ebt, %advgb2ebt : tensor<96xf32>
    %adbc1b2ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2ebt = stablehlo.divide %admnb2ebt, %adbc1b2ebt : tensor<96xf32>
    %advhb2ebt = stablehlo.divide %advnb2ebt, %adbc2b2ebt : tensor<96xf32>
    %adlrb2ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2ebt = stablehlo.sqrt %advhb2ebt : tensor<96xf32>
    %addenb2ebt = stablehlo.add %adsqb2ebt, %adepsb2ebt : tensor<96xf32>
    %adratb2ebt = stablehlo.divide %admhb2ebt, %addenb2ebt : tensor<96xf32>
    %adstb2ebt = stablehlo.multiply %adlrb2ebt, %adratb2ebt : tensor<96xf32>
    %adsubb2ebt = stablehlo.subtract %b2ebt, %adstb2ebt : tensor<96xf32>
    %adwdb2ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2ebt = stablehlo.multiply %adwdb2ebt, %adlrb2ebt : tensor<96xf32>
    %adwdpb2ebt = stablehlo.multiply %adwdlrb2ebt, %b2ebt : tensor<96xf32>
    %adnewb2ebt = stablehlo.subtract %adsubb2ebt, %adwdpb2ebt : tensor<96xf32>
    %adb1b2dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adob1b2dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %admsb2dW = stablehlo.multiply %adb1b2dW, %b2dWm : tensor<96x1x3x3xf32>
    %admgb2dW = stablehlo.multiply %adob1b2dW, %b2ddW : tensor<96x1x3x3xf32>
    %admnb2dW = stablehlo.add %admsb2dW, %admgb2dW : tensor<96x1x3x3xf32>
    %adb2b2dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adob2b2dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %advsb2dW = stablehlo.multiply %adb2b2dW, %b2dWv : tensor<96x1x3x3xf32>
    %adg2b2dW = stablehlo.multiply %b2ddW, %b2ddW : tensor<96x1x3x3xf32>
    %advgb2dW = stablehlo.multiply %adob2b2dW, %adg2b2dW : tensor<96x1x3x3xf32>
    %advnb2dW = stablehlo.add %advsb2dW, %advgb2dW : tensor<96x1x3x3xf32>
    %adbc1b2dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adbc2b2dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %admhb2dW = stablehlo.divide %admnb2dW, %adbc1b2dW : tensor<96x1x3x3xf32>
    %advhb2dW = stablehlo.divide %advnb2dW, %adbc2b2dW : tensor<96x1x3x3xf32>
    %adlrb2dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adepsb2dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adsqb2dW = stablehlo.sqrt %advhb2dW : tensor<96x1x3x3xf32>
    %addenb2dW = stablehlo.add %adsqb2dW, %adepsb2dW : tensor<96x1x3x3xf32>
    %adratb2dW = stablehlo.divide %admhb2dW, %addenb2dW : tensor<96x1x3x3xf32>
    %adstb2dW = stablehlo.multiply %adlrb2dW, %adratb2dW : tensor<96x1x3x3xf32>
    %adsubb2dW = stablehlo.subtract %b2dW, %adstb2dW : tensor<96x1x3x3xf32>
    %adwdb2dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adwdlrb2dW = stablehlo.multiply %adwdb2dW, %adlrb2dW : tensor<96x1x3x3xf32>
    %adwdpb2dW = stablehlo.multiply %adwdlrb2dW, %b2dW : tensor<96x1x3x3xf32>
    %adnewb2dW = stablehlo.subtract %adsubb2dW, %adwdpb2dW : tensor<96x1x3x3xf32>
    %adb1b2db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2db = stablehlo.multiply %adb1b2db, %b2dbm : tensor<96xf32>
    %admgb2db = stablehlo.multiply %adob1b2db, %b2ddb : tensor<96xf32>
    %admnb2db = stablehlo.add %admsb2db, %admgb2db : tensor<96xf32>
    %adb2b2db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2db = stablehlo.multiply %adb2b2db, %b2dbv : tensor<96xf32>
    %adg2b2db = stablehlo.multiply %b2ddb, %b2ddb : tensor<96xf32>
    %advgb2db = stablehlo.multiply %adob2b2db, %adg2b2db : tensor<96xf32>
    %advnb2db = stablehlo.add %advsb2db, %advgb2db : tensor<96xf32>
    %adbc1b2db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2db = stablehlo.divide %admnb2db, %adbc1b2db : tensor<96xf32>
    %advhb2db = stablehlo.divide %advnb2db, %adbc2b2db : tensor<96xf32>
    %adlrb2db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2db = stablehlo.sqrt %advhb2db : tensor<96xf32>
    %addenb2db = stablehlo.add %adsqb2db, %adepsb2db : tensor<96xf32>
    %adratb2db = stablehlo.divide %admhb2db, %addenb2db : tensor<96xf32>
    %adstb2db = stablehlo.multiply %adlrb2db, %adratb2db : tensor<96xf32>
    %adsubb2db = stablehlo.subtract %b2db, %adstb2db : tensor<96xf32>
    %adwdb2db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2db = stablehlo.multiply %adwdb2db, %adlrb2db : tensor<96xf32>
    %adwdpb2db = stablehlo.multiply %adwdlrb2db, %b2db : tensor<96xf32>
    %adnewb2db = stablehlo.subtract %adsubb2db, %adwdpb2db : tensor<96xf32>
    %adb1b2dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2dg = stablehlo.multiply %adb1b2dg, %b2dgm : tensor<96xf32>
    %admgb2dg = stablehlo.multiply %adob1b2dg, %b2ddg : tensor<96xf32>
    %admnb2dg = stablehlo.add %admsb2dg, %admgb2dg : tensor<96xf32>
    %adb2b2dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2dg = stablehlo.multiply %adb2b2dg, %b2dgv : tensor<96xf32>
    %adg2b2dg = stablehlo.multiply %b2ddg, %b2ddg : tensor<96xf32>
    %advgb2dg = stablehlo.multiply %adob2b2dg, %adg2b2dg : tensor<96xf32>
    %advnb2dg = stablehlo.add %advsb2dg, %advgb2dg : tensor<96xf32>
    %adbc1b2dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2dg = stablehlo.divide %admnb2dg, %adbc1b2dg : tensor<96xf32>
    %advhb2dg = stablehlo.divide %advnb2dg, %adbc2b2dg : tensor<96xf32>
    %adlrb2dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2dg = stablehlo.sqrt %advhb2dg : tensor<96xf32>
    %addenb2dg = stablehlo.add %adsqb2dg, %adepsb2dg : tensor<96xf32>
    %adratb2dg = stablehlo.divide %admhb2dg, %addenb2dg : tensor<96xf32>
    %adstb2dg = stablehlo.multiply %adlrb2dg, %adratb2dg : tensor<96xf32>
    %adsubb2dg = stablehlo.subtract %b2dg, %adstb2dg : tensor<96xf32>
    %adwdb2dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2dg = stablehlo.multiply %adwdb2dg, %adlrb2dg : tensor<96xf32>
    %adwdpb2dg = stablehlo.multiply %adwdlrb2dg, %b2dg : tensor<96xf32>
    %adnewb2dg = stablehlo.subtract %adsubb2dg, %adwdpb2dg : tensor<96xf32>
    %adb1b2dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b2dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb2dbt = stablehlo.multiply %adb1b2dbt, %b2dbtm : tensor<96xf32>
    %admgb2dbt = stablehlo.multiply %adob1b2dbt, %b2ddbt : tensor<96xf32>
    %admnb2dbt = stablehlo.add %admsb2dbt, %admgb2dbt : tensor<96xf32>
    %adb2b2dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2dbt = stablehlo.multiply %adb2b2dbt, %b2dbtv : tensor<96xf32>
    %adg2b2dbt = stablehlo.multiply %b2ddbt, %b2ddbt : tensor<96xf32>
    %advgb2dbt = stablehlo.multiply %adob2b2dbt, %adg2b2dbt : tensor<96xf32>
    %advnb2dbt = stablehlo.add %advsb2dbt, %advgb2dbt : tensor<96xf32>
    %adbc1b2dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b2dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb2dbt = stablehlo.divide %admnb2dbt, %adbc1b2dbt : tensor<96xf32>
    %advhb2dbt = stablehlo.divide %advnb2dbt, %adbc2b2dbt : tensor<96xf32>
    %adlrb2dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb2dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb2dbt = stablehlo.sqrt %advhb2dbt : tensor<96xf32>
    %addenb2dbt = stablehlo.add %adsqb2dbt, %adepsb2dbt : tensor<96xf32>
    %adratb2dbt = stablehlo.divide %admhb2dbt, %addenb2dbt : tensor<96xf32>
    %adstb2dbt = stablehlo.multiply %adlrb2dbt, %adratb2dbt : tensor<96xf32>
    %adsubb2dbt = stablehlo.subtract %b2dbt, %adstb2dbt : tensor<96xf32>
    %adwdb2dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb2dbt = stablehlo.multiply %adwdb2dbt, %adlrb2dbt : tensor<96xf32>
    %adwdpb2dbt = stablehlo.multiply %adwdlrb2dbt, %b2dbt : tensor<96xf32>
    %adnewb2dbt = stablehlo.subtract %adsubb2dbt, %adwdpb2dbt : tensor<96xf32>
    %adb1b2pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %adob1b2pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %admsb2pW = stablehlo.multiply %adb1b2pW, %b2pWm : tensor<24x96x1x1xf32>
    %admgb2pW = stablehlo.multiply %adob1b2pW, %b2dpW : tensor<24x96x1x1xf32>
    %admnb2pW = stablehlo.add %admsb2pW, %admgb2pW : tensor<24x96x1x1xf32>
    %adb2b2pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %adob2b2pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %advsb2pW = stablehlo.multiply %adb2b2pW, %b2pWv : tensor<24x96x1x1xf32>
    %adg2b2pW = stablehlo.multiply %b2dpW, %b2dpW : tensor<24x96x1x1xf32>
    %advgb2pW = stablehlo.multiply %adob2b2pW, %adg2b2pW : tensor<24x96x1x1xf32>
    %advnb2pW = stablehlo.add %advsb2pW, %advgb2pW : tensor<24x96x1x1xf32>
    %adbc1b2pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %adbc2b2pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %admhb2pW = stablehlo.divide %admnb2pW, %adbc1b2pW : tensor<24x96x1x1xf32>
    %advhb2pW = stablehlo.divide %advnb2pW, %adbc2b2pW : tensor<24x96x1x1xf32>
    %adlrb2pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %adepsb2pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %adsqb2pW = stablehlo.sqrt %advhb2pW : tensor<24x96x1x1xf32>
    %addenb2pW = stablehlo.add %adsqb2pW, %adepsb2pW : tensor<24x96x1x1xf32>
    %adratb2pW = stablehlo.divide %admhb2pW, %addenb2pW : tensor<24x96x1x1xf32>
    %adstb2pW = stablehlo.multiply %adlrb2pW, %adratb2pW : tensor<24x96x1x1xf32>
    %adsubb2pW = stablehlo.subtract %b2pW, %adstb2pW : tensor<24x96x1x1xf32>
    %adwdb2pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24x96x1x1xf32>
    %adwdlrb2pW = stablehlo.multiply %adwdb2pW, %adlrb2pW : tensor<24x96x1x1xf32>
    %adwdpb2pW = stablehlo.multiply %adwdlrb2pW, %b2pW : tensor<24x96x1x1xf32>
    %adnewb2pW = stablehlo.subtract %adsubb2pW, %adwdpb2pW : tensor<24x96x1x1xf32>
    %adb1b2pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob1b2pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admsb2pb = stablehlo.multiply %adb1b2pb, %b2pbm : tensor<24xf32>
    %admgb2pb = stablehlo.multiply %adob1b2pb, %b2dpb : tensor<24xf32>
    %admnb2pb = stablehlo.add %admsb2pb, %admgb2pb : tensor<24xf32>
    %adb2b2pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b2pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb2pb = stablehlo.multiply %adb2b2pb, %b2pbv : tensor<24xf32>
    %adg2b2pb = stablehlo.multiply %b2dpb, %b2dpb : tensor<24xf32>
    %advgb2pb = stablehlo.multiply %adob2b2pb, %adg2b2pb : tensor<24xf32>
    %advnb2pb = stablehlo.add %advsb2pb, %advgb2pb : tensor<24xf32>
    %adbc1b2pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adbc2b2pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admhb2pb = stablehlo.divide %admnb2pb, %adbc1b2pb : tensor<24xf32>
    %advhb2pb = stablehlo.divide %advnb2pb, %adbc2b2pb : tensor<24xf32>
    %adlrb2pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adepsb2pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adsqb2pb = stablehlo.sqrt %advhb2pb : tensor<24xf32>
    %addenb2pb = stablehlo.add %adsqb2pb, %adepsb2pb : tensor<24xf32>
    %adratb2pb = stablehlo.divide %admhb2pb, %addenb2pb : tensor<24xf32>
    %adstb2pb = stablehlo.multiply %adlrb2pb, %adratb2pb : tensor<24xf32>
    %adsubb2pb = stablehlo.subtract %b2pb, %adstb2pb : tensor<24xf32>
    %adwdb2pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adwdlrb2pb = stablehlo.multiply %adwdb2pb, %adlrb2pb : tensor<24xf32>
    %adwdpb2pb = stablehlo.multiply %adwdlrb2pb, %b2pb : tensor<24xf32>
    %adnewb2pb = stablehlo.subtract %adsubb2pb, %adwdpb2pb : tensor<24xf32>
    %adb1b2pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob1b2pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admsb2pg = stablehlo.multiply %adb1b2pg, %b2pgm : tensor<24xf32>
    %admgb2pg = stablehlo.multiply %adob1b2pg, %b2dpg : tensor<24xf32>
    %admnb2pg = stablehlo.add %admsb2pg, %admgb2pg : tensor<24xf32>
    %adb2b2pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b2pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb2pg = stablehlo.multiply %adb2b2pg, %b2pgv : tensor<24xf32>
    %adg2b2pg = stablehlo.multiply %b2dpg, %b2dpg : tensor<24xf32>
    %advgb2pg = stablehlo.multiply %adob2b2pg, %adg2b2pg : tensor<24xf32>
    %advnb2pg = stablehlo.add %advsb2pg, %advgb2pg : tensor<24xf32>
    %adbc1b2pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adbc2b2pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admhb2pg = stablehlo.divide %admnb2pg, %adbc1b2pg : tensor<24xf32>
    %advhb2pg = stablehlo.divide %advnb2pg, %adbc2b2pg : tensor<24xf32>
    %adlrb2pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adepsb2pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adsqb2pg = stablehlo.sqrt %advhb2pg : tensor<24xf32>
    %addenb2pg = stablehlo.add %adsqb2pg, %adepsb2pg : tensor<24xf32>
    %adratb2pg = stablehlo.divide %admhb2pg, %addenb2pg : tensor<24xf32>
    %adstb2pg = stablehlo.multiply %adlrb2pg, %adratb2pg : tensor<24xf32>
    %adsubb2pg = stablehlo.subtract %b2pg, %adstb2pg : tensor<24xf32>
    %adwdb2pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adwdlrb2pg = stablehlo.multiply %adwdb2pg, %adlrb2pg : tensor<24xf32>
    %adwdpb2pg = stablehlo.multiply %adwdlrb2pg, %b2pg : tensor<24xf32>
    %adnewb2pg = stablehlo.subtract %adsubb2pg, %adwdpb2pg : tensor<24xf32>
    %adb1b2pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob1b2pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admsb2pbt = stablehlo.multiply %adb1b2pbt, %b2pbtm : tensor<24xf32>
    %admgb2pbt = stablehlo.multiply %adob1b2pbt, %b2dpbt : tensor<24xf32>
    %admnb2pbt = stablehlo.add %admsb2pbt, %admgb2pbt : tensor<24xf32>
    %adb2b2pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b2pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb2pbt = stablehlo.multiply %adb2b2pbt, %b2pbtv : tensor<24xf32>
    %adg2b2pbt = stablehlo.multiply %b2dpbt, %b2dpbt : tensor<24xf32>
    %advgb2pbt = stablehlo.multiply %adob2b2pbt, %adg2b2pbt : tensor<24xf32>
    %advnb2pbt = stablehlo.add %advsb2pbt, %advgb2pbt : tensor<24xf32>
    %adbc1b2pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adbc2b2pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %admhb2pbt = stablehlo.divide %admnb2pbt, %adbc1b2pbt : tensor<24xf32>
    %advhb2pbt = stablehlo.divide %advnb2pbt, %adbc2b2pbt : tensor<24xf32>
    %adlrb2pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adepsb2pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adsqb2pbt = stablehlo.sqrt %advhb2pbt : tensor<24xf32>
    %addenb2pbt = stablehlo.add %adsqb2pbt, %adepsb2pbt : tensor<24xf32>
    %adratb2pbt = stablehlo.divide %admhb2pbt, %addenb2pbt : tensor<24xf32>
    %adstb2pbt = stablehlo.multiply %adlrb2pbt, %adratb2pbt : tensor<24xf32>
    %adsubb2pbt = stablehlo.subtract %b2pbt, %adstb2pbt : tensor<24xf32>
    %adwdb2pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adwdlrb2pbt = stablehlo.multiply %adwdb2pbt, %adlrb2pbt : tensor<24xf32>
    %adwdpb2pbt = stablehlo.multiply %adwdlrb2pbt, %b2pbt : tensor<24xf32>
    %adnewb2pbt = stablehlo.subtract %adsubb2pbt, %adwdpb2pbt : tensor<24xf32>
    %adb1b3eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %adob1b3eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %admsb3eW = stablehlo.multiply %adb1b3eW, %b3eWm : tensor<96x24x1x1xf32>
    %admgb3eW = stablehlo.multiply %adob1b3eW, %b3deW : tensor<96x24x1x1xf32>
    %admnb3eW = stablehlo.add %admsb3eW, %admgb3eW : tensor<96x24x1x1xf32>
    %adb2b3eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %adob2b3eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %advsb3eW = stablehlo.multiply %adb2b3eW, %b3eWv : tensor<96x24x1x1xf32>
    %adg2b3eW = stablehlo.multiply %b3deW, %b3deW : tensor<96x24x1x1xf32>
    %advgb3eW = stablehlo.multiply %adob2b3eW, %adg2b3eW : tensor<96x24x1x1xf32>
    %advnb3eW = stablehlo.add %advsb3eW, %advgb3eW : tensor<96x24x1x1xf32>
    %adbc1b3eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %adbc2b3eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %admhb3eW = stablehlo.divide %admnb3eW, %adbc1b3eW : tensor<96x24x1x1xf32>
    %advhb3eW = stablehlo.divide %advnb3eW, %adbc2b3eW : tensor<96x24x1x1xf32>
    %adlrb3eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %adepsb3eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %adsqb3eW = stablehlo.sqrt %advhb3eW : tensor<96x24x1x1xf32>
    %addenb3eW = stablehlo.add %adsqb3eW, %adepsb3eW : tensor<96x24x1x1xf32>
    %adratb3eW = stablehlo.divide %admhb3eW, %addenb3eW : tensor<96x24x1x1xf32>
    %adstb3eW = stablehlo.multiply %adlrb3eW, %adratb3eW : tensor<96x24x1x1xf32>
    %adsubb3eW = stablehlo.subtract %b3eW, %adstb3eW : tensor<96x24x1x1xf32>
    %adwdb3eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x24x1x1xf32>
    %adwdlrb3eW = stablehlo.multiply %adwdb3eW, %adlrb3eW : tensor<96x24x1x1xf32>
    %adwdpb3eW = stablehlo.multiply %adwdlrb3eW, %b3eW : tensor<96x24x1x1xf32>
    %adnewb3eW = stablehlo.subtract %adsubb3eW, %adwdpb3eW : tensor<96x24x1x1xf32>
    %adb1b3eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b3eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb3eb = stablehlo.multiply %adb1b3eb, %b3ebm : tensor<96xf32>
    %admgb3eb = stablehlo.multiply %adob1b3eb, %b3deb : tensor<96xf32>
    %admnb3eb = stablehlo.add %admsb3eb, %admgb3eb : tensor<96xf32>
    %adb2b3eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b3eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb3eb = stablehlo.multiply %adb2b3eb, %b3ebv : tensor<96xf32>
    %adg2b3eb = stablehlo.multiply %b3deb, %b3deb : tensor<96xf32>
    %advgb3eb = stablehlo.multiply %adob2b3eb, %adg2b3eb : tensor<96xf32>
    %advnb3eb = stablehlo.add %advsb3eb, %advgb3eb : tensor<96xf32>
    %adbc1b3eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b3eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb3eb = stablehlo.divide %admnb3eb, %adbc1b3eb : tensor<96xf32>
    %advhb3eb = stablehlo.divide %advnb3eb, %adbc2b3eb : tensor<96xf32>
    %adlrb3eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb3eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb3eb = stablehlo.sqrt %advhb3eb : tensor<96xf32>
    %addenb3eb = stablehlo.add %adsqb3eb, %adepsb3eb : tensor<96xf32>
    %adratb3eb = stablehlo.divide %admhb3eb, %addenb3eb : tensor<96xf32>
    %adstb3eb = stablehlo.multiply %adlrb3eb, %adratb3eb : tensor<96xf32>
    %adsubb3eb = stablehlo.subtract %b3eb, %adstb3eb : tensor<96xf32>
    %adwdb3eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb3eb = stablehlo.multiply %adwdb3eb, %adlrb3eb : tensor<96xf32>
    %adwdpb3eb = stablehlo.multiply %adwdlrb3eb, %b3eb : tensor<96xf32>
    %adnewb3eb = stablehlo.subtract %adsubb3eb, %adwdpb3eb : tensor<96xf32>
    %adb1b3eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b3eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb3eg = stablehlo.multiply %adb1b3eg, %b3egm : tensor<96xf32>
    %admgb3eg = stablehlo.multiply %adob1b3eg, %b3deg : tensor<96xf32>
    %admnb3eg = stablehlo.add %admsb3eg, %admgb3eg : tensor<96xf32>
    %adb2b3eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b3eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb3eg = stablehlo.multiply %adb2b3eg, %b3egv : tensor<96xf32>
    %adg2b3eg = stablehlo.multiply %b3deg, %b3deg : tensor<96xf32>
    %advgb3eg = stablehlo.multiply %adob2b3eg, %adg2b3eg : tensor<96xf32>
    %advnb3eg = stablehlo.add %advsb3eg, %advgb3eg : tensor<96xf32>
    %adbc1b3eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b3eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb3eg = stablehlo.divide %admnb3eg, %adbc1b3eg : tensor<96xf32>
    %advhb3eg = stablehlo.divide %advnb3eg, %adbc2b3eg : tensor<96xf32>
    %adlrb3eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb3eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb3eg = stablehlo.sqrt %advhb3eg : tensor<96xf32>
    %addenb3eg = stablehlo.add %adsqb3eg, %adepsb3eg : tensor<96xf32>
    %adratb3eg = stablehlo.divide %admhb3eg, %addenb3eg : tensor<96xf32>
    %adstb3eg = stablehlo.multiply %adlrb3eg, %adratb3eg : tensor<96xf32>
    %adsubb3eg = stablehlo.subtract %b3eg, %adstb3eg : tensor<96xf32>
    %adwdb3eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb3eg = stablehlo.multiply %adwdb3eg, %adlrb3eg : tensor<96xf32>
    %adwdpb3eg = stablehlo.multiply %adwdlrb3eg, %b3eg : tensor<96xf32>
    %adnewb3eg = stablehlo.subtract %adsubb3eg, %adwdpb3eg : tensor<96xf32>
    %adb1b3ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b3ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb3ebt = stablehlo.multiply %adb1b3ebt, %b3ebtm : tensor<96xf32>
    %admgb3ebt = stablehlo.multiply %adob1b3ebt, %b3debt : tensor<96xf32>
    %admnb3ebt = stablehlo.add %admsb3ebt, %admgb3ebt : tensor<96xf32>
    %adb2b3ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b3ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb3ebt = stablehlo.multiply %adb2b3ebt, %b3ebtv : tensor<96xf32>
    %adg2b3ebt = stablehlo.multiply %b3debt, %b3debt : tensor<96xf32>
    %advgb3ebt = stablehlo.multiply %adob2b3ebt, %adg2b3ebt : tensor<96xf32>
    %advnb3ebt = stablehlo.add %advsb3ebt, %advgb3ebt : tensor<96xf32>
    %adbc1b3ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b3ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb3ebt = stablehlo.divide %admnb3ebt, %adbc1b3ebt : tensor<96xf32>
    %advhb3ebt = stablehlo.divide %advnb3ebt, %adbc2b3ebt : tensor<96xf32>
    %adlrb3ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb3ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb3ebt = stablehlo.sqrt %advhb3ebt : tensor<96xf32>
    %addenb3ebt = stablehlo.add %adsqb3ebt, %adepsb3ebt : tensor<96xf32>
    %adratb3ebt = stablehlo.divide %admhb3ebt, %addenb3ebt : tensor<96xf32>
    %adstb3ebt = stablehlo.multiply %adlrb3ebt, %adratb3ebt : tensor<96xf32>
    %adsubb3ebt = stablehlo.subtract %b3ebt, %adstb3ebt : tensor<96xf32>
    %adwdb3ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb3ebt = stablehlo.multiply %adwdb3ebt, %adlrb3ebt : tensor<96xf32>
    %adwdpb3ebt = stablehlo.multiply %adwdlrb3ebt, %b3ebt : tensor<96xf32>
    %adnewb3ebt = stablehlo.subtract %adsubb3ebt, %adwdpb3ebt : tensor<96xf32>
    %adb1b3dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adob1b3dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %admsb3dW = stablehlo.multiply %adb1b3dW, %b3dWm : tensor<96x1x3x3xf32>
    %admgb3dW = stablehlo.multiply %adob1b3dW, %b3ddW : tensor<96x1x3x3xf32>
    %admnb3dW = stablehlo.add %admsb3dW, %admgb3dW : tensor<96x1x3x3xf32>
    %adb2b3dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adob2b3dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %advsb3dW = stablehlo.multiply %adb2b3dW, %b3dWv : tensor<96x1x3x3xf32>
    %adg2b3dW = stablehlo.multiply %b3ddW, %b3ddW : tensor<96x1x3x3xf32>
    %advgb3dW = stablehlo.multiply %adob2b3dW, %adg2b3dW : tensor<96x1x3x3xf32>
    %advnb3dW = stablehlo.add %advsb3dW, %advgb3dW : tensor<96x1x3x3xf32>
    %adbc1b3dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adbc2b3dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %admhb3dW = stablehlo.divide %admnb3dW, %adbc1b3dW : tensor<96x1x3x3xf32>
    %advhb3dW = stablehlo.divide %advnb3dW, %adbc2b3dW : tensor<96x1x3x3xf32>
    %adlrb3dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adepsb3dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adsqb3dW = stablehlo.sqrt %advhb3dW : tensor<96x1x3x3xf32>
    %addenb3dW = stablehlo.add %adsqb3dW, %adepsb3dW : tensor<96x1x3x3xf32>
    %adratb3dW = stablehlo.divide %admhb3dW, %addenb3dW : tensor<96x1x3x3xf32>
    %adstb3dW = stablehlo.multiply %adlrb3dW, %adratb3dW : tensor<96x1x3x3xf32>
    %adsubb3dW = stablehlo.subtract %b3dW, %adstb3dW : tensor<96x1x3x3xf32>
    %adwdb3dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x3x3xf32>
    %adwdlrb3dW = stablehlo.multiply %adwdb3dW, %adlrb3dW : tensor<96x1x3x3xf32>
    %adwdpb3dW = stablehlo.multiply %adwdlrb3dW, %b3dW : tensor<96x1x3x3xf32>
    %adnewb3dW = stablehlo.subtract %adsubb3dW, %adwdpb3dW : tensor<96x1x3x3xf32>
    %adb1b3db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b3db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb3db = stablehlo.multiply %adb1b3db, %b3dbm : tensor<96xf32>
    %admgb3db = stablehlo.multiply %adob1b3db, %b3ddb : tensor<96xf32>
    %admnb3db = stablehlo.add %admsb3db, %admgb3db : tensor<96xf32>
    %adb2b3db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b3db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb3db = stablehlo.multiply %adb2b3db, %b3dbv : tensor<96xf32>
    %adg2b3db = stablehlo.multiply %b3ddb, %b3ddb : tensor<96xf32>
    %advgb3db = stablehlo.multiply %adob2b3db, %adg2b3db : tensor<96xf32>
    %advnb3db = stablehlo.add %advsb3db, %advgb3db : tensor<96xf32>
    %adbc1b3db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b3db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb3db = stablehlo.divide %admnb3db, %adbc1b3db : tensor<96xf32>
    %advhb3db = stablehlo.divide %advnb3db, %adbc2b3db : tensor<96xf32>
    %adlrb3db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb3db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb3db = stablehlo.sqrt %advhb3db : tensor<96xf32>
    %addenb3db = stablehlo.add %adsqb3db, %adepsb3db : tensor<96xf32>
    %adratb3db = stablehlo.divide %admhb3db, %addenb3db : tensor<96xf32>
    %adstb3db = stablehlo.multiply %adlrb3db, %adratb3db : tensor<96xf32>
    %adsubb3db = stablehlo.subtract %b3db, %adstb3db : tensor<96xf32>
    %adwdb3db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb3db = stablehlo.multiply %adwdb3db, %adlrb3db : tensor<96xf32>
    %adwdpb3db = stablehlo.multiply %adwdlrb3db, %b3db : tensor<96xf32>
    %adnewb3db = stablehlo.subtract %adsubb3db, %adwdpb3db : tensor<96xf32>
    %adb1b3dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b3dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb3dg = stablehlo.multiply %adb1b3dg, %b3dgm : tensor<96xf32>
    %admgb3dg = stablehlo.multiply %adob1b3dg, %b3ddg : tensor<96xf32>
    %admnb3dg = stablehlo.add %admsb3dg, %admgb3dg : tensor<96xf32>
    %adb2b3dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b3dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb3dg = stablehlo.multiply %adb2b3dg, %b3dgv : tensor<96xf32>
    %adg2b3dg = stablehlo.multiply %b3ddg, %b3ddg : tensor<96xf32>
    %advgb3dg = stablehlo.multiply %adob2b3dg, %adg2b3dg : tensor<96xf32>
    %advnb3dg = stablehlo.add %advsb3dg, %advgb3dg : tensor<96xf32>
    %adbc1b3dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b3dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb3dg = stablehlo.divide %admnb3dg, %adbc1b3dg : tensor<96xf32>
    %advhb3dg = stablehlo.divide %advnb3dg, %adbc2b3dg : tensor<96xf32>
    %adlrb3dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb3dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb3dg = stablehlo.sqrt %advhb3dg : tensor<96xf32>
    %addenb3dg = stablehlo.add %adsqb3dg, %adepsb3dg : tensor<96xf32>
    %adratb3dg = stablehlo.divide %admhb3dg, %addenb3dg : tensor<96xf32>
    %adstb3dg = stablehlo.multiply %adlrb3dg, %adratb3dg : tensor<96xf32>
    %adsubb3dg = stablehlo.subtract %b3dg, %adstb3dg : tensor<96xf32>
    %adwdb3dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb3dg = stablehlo.multiply %adwdb3dg, %adlrb3dg : tensor<96xf32>
    %adwdpb3dg = stablehlo.multiply %adwdlrb3dg, %b3dg : tensor<96xf32>
    %adnewb3dg = stablehlo.subtract %adsubb3dg, %adwdpb3dg : tensor<96xf32>
    %adb1b3dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1b3dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admsb3dbt = stablehlo.multiply %adb1b3dbt, %b3dbtm : tensor<96xf32>
    %admgb3dbt = stablehlo.multiply %adob1b3dbt, %b3ddbt : tensor<96xf32>
    %admnb3dbt = stablehlo.add %admsb3dbt, %admgb3dbt : tensor<96xf32>
    %adb2b3dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b3dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb3dbt = stablehlo.multiply %adb2b3dbt, %b3dbtv : tensor<96xf32>
    %adg2b3dbt = stablehlo.multiply %b3ddbt, %b3ddbt : tensor<96xf32>
    %advgb3dbt = stablehlo.multiply %adob2b3dbt, %adg2b3dbt : tensor<96xf32>
    %advnb3dbt = stablehlo.add %advsb3dbt, %advgb3dbt : tensor<96xf32>
    %adbc1b3dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2b3dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhb3dbt = stablehlo.divide %admnb3dbt, %adbc1b3dbt : tensor<96xf32>
    %advhb3dbt = stablehlo.divide %advnb3dbt, %adbc2b3dbt : tensor<96xf32>
    %adlrb3dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepsb3dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqb3dbt = stablehlo.sqrt %advhb3dbt : tensor<96xf32>
    %addenb3dbt = stablehlo.add %adsqb3dbt, %adepsb3dbt : tensor<96xf32>
    %adratb3dbt = stablehlo.divide %admhb3dbt, %addenb3dbt : tensor<96xf32>
    %adstb3dbt = stablehlo.multiply %adlrb3dbt, %adratb3dbt : tensor<96xf32>
    %adsubb3dbt = stablehlo.subtract %b3dbt, %adstb3dbt : tensor<96xf32>
    %adwdb3dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrb3dbt = stablehlo.multiply %adwdb3dbt, %adlrb3dbt : tensor<96xf32>
    %adwdpb3dbt = stablehlo.multiply %adwdlrb3dbt, %b3dbt : tensor<96xf32>
    %adnewb3dbt = stablehlo.subtract %adsubb3dbt, %adwdpb3dbt : tensor<96xf32>
    %adb1b3pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x96x1x1xf32>
    %adob1b3pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x96x1x1xf32>
    %admsb3pW = stablehlo.multiply %adb1b3pW, %b3pWm : tensor<32x96x1x1xf32>
    %admgb3pW = stablehlo.multiply %adob1b3pW, %b3dpW : tensor<32x96x1x1xf32>
    %admnb3pW = stablehlo.add %admsb3pW, %admgb3pW : tensor<32x96x1x1xf32>
    %adb2b3pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x96x1x1xf32>
    %adob2b3pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x96x1x1xf32>
    %advsb3pW = stablehlo.multiply %adb2b3pW, %b3pWv : tensor<32x96x1x1xf32>
    %adg2b3pW = stablehlo.multiply %b3dpW, %b3dpW : tensor<32x96x1x1xf32>
    %advgb3pW = stablehlo.multiply %adob2b3pW, %adg2b3pW : tensor<32x96x1x1xf32>
    %advnb3pW = stablehlo.add %advsb3pW, %advgb3pW : tensor<32x96x1x1xf32>
    %adbc1b3pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x96x1x1xf32>
    %adbc2b3pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x96x1x1xf32>
    %admhb3pW = stablehlo.divide %admnb3pW, %adbc1b3pW : tensor<32x96x1x1xf32>
    %advhb3pW = stablehlo.divide %advnb3pW, %adbc2b3pW : tensor<32x96x1x1xf32>
    %adlrb3pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x96x1x1xf32>
    %adepsb3pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x96x1x1xf32>
    %adsqb3pW = stablehlo.sqrt %advhb3pW : tensor<32x96x1x1xf32>
    %addenb3pW = stablehlo.add %adsqb3pW, %adepsb3pW : tensor<32x96x1x1xf32>
    %adratb3pW = stablehlo.divide %admhb3pW, %addenb3pW : tensor<32x96x1x1xf32>
    %adstb3pW = stablehlo.multiply %adlrb3pW, %adratb3pW : tensor<32x96x1x1xf32>
    %adsubb3pW = stablehlo.subtract %b3pW, %adstb3pW : tensor<32x96x1x1xf32>
    %adwdb3pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x96x1x1xf32>
    %adwdlrb3pW = stablehlo.multiply %adwdb3pW, %adlrb3pW : tensor<32x96x1x1xf32>
    %adwdpb3pW = stablehlo.multiply %adwdlrb3pW, %b3pW : tensor<32x96x1x1xf32>
    %adnewb3pW = stablehlo.subtract %adsubb3pW, %adwdpb3pW : tensor<32x96x1x1xf32>
    %adb1b3pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1b3pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsb3pb = stablehlo.multiply %adb1b3pb, %b3pbm : tensor<32xf32>
    %admgb3pb = stablehlo.multiply %adob1b3pb, %b3dpb : tensor<32xf32>
    %admnb3pb = stablehlo.add %admsb3pb, %admgb3pb : tensor<32xf32>
    %adb2b3pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b3pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb3pb = stablehlo.multiply %adb2b3pb, %b3pbv : tensor<32xf32>
    %adg2b3pb = stablehlo.multiply %b3dpb, %b3dpb : tensor<32xf32>
    %advgb3pb = stablehlo.multiply %adob2b3pb, %adg2b3pb : tensor<32xf32>
    %advnb3pb = stablehlo.add %advsb3pb, %advgb3pb : tensor<32xf32>
    %adbc1b3pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2b3pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhb3pb = stablehlo.divide %admnb3pb, %adbc1b3pb : tensor<32xf32>
    %advhb3pb = stablehlo.divide %advnb3pb, %adbc2b3pb : tensor<32xf32>
    %adlrb3pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsb3pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqb3pb = stablehlo.sqrt %advhb3pb : tensor<32xf32>
    %addenb3pb = stablehlo.add %adsqb3pb, %adepsb3pb : tensor<32xf32>
    %adratb3pb = stablehlo.divide %admhb3pb, %addenb3pb : tensor<32xf32>
    %adstb3pb = stablehlo.multiply %adlrb3pb, %adratb3pb : tensor<32xf32>
    %adsubb3pb = stablehlo.subtract %b3pb, %adstb3pb : tensor<32xf32>
    %adwdb3pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrb3pb = stablehlo.multiply %adwdb3pb, %adlrb3pb : tensor<32xf32>
    %adwdpb3pb = stablehlo.multiply %adwdlrb3pb, %b3pb : tensor<32xf32>
    %adnewb3pb = stablehlo.subtract %adsubb3pb, %adwdpb3pb : tensor<32xf32>
    %adb1b3pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1b3pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsb3pg = stablehlo.multiply %adb1b3pg, %b3pgm : tensor<32xf32>
    %admgb3pg = stablehlo.multiply %adob1b3pg, %b3dpg : tensor<32xf32>
    %admnb3pg = stablehlo.add %admsb3pg, %admgb3pg : tensor<32xf32>
    %adb2b3pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b3pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb3pg = stablehlo.multiply %adb2b3pg, %b3pgv : tensor<32xf32>
    %adg2b3pg = stablehlo.multiply %b3dpg, %b3dpg : tensor<32xf32>
    %advgb3pg = stablehlo.multiply %adob2b3pg, %adg2b3pg : tensor<32xf32>
    %advnb3pg = stablehlo.add %advsb3pg, %advgb3pg : tensor<32xf32>
    %adbc1b3pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2b3pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhb3pg = stablehlo.divide %admnb3pg, %adbc1b3pg : tensor<32xf32>
    %advhb3pg = stablehlo.divide %advnb3pg, %adbc2b3pg : tensor<32xf32>
    %adlrb3pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsb3pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqb3pg = stablehlo.sqrt %advhb3pg : tensor<32xf32>
    %addenb3pg = stablehlo.add %adsqb3pg, %adepsb3pg : tensor<32xf32>
    %adratb3pg = stablehlo.divide %admhb3pg, %addenb3pg : tensor<32xf32>
    %adstb3pg = stablehlo.multiply %adlrb3pg, %adratb3pg : tensor<32xf32>
    %adsubb3pg = stablehlo.subtract %b3pg, %adstb3pg : tensor<32xf32>
    %adwdb3pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrb3pg = stablehlo.multiply %adwdb3pg, %adlrb3pg : tensor<32xf32>
    %adwdpb3pg = stablehlo.multiply %adwdlrb3pg, %b3pg : tensor<32xf32>
    %adnewb3pg = stablehlo.subtract %adsubb3pg, %adwdpb3pg : tensor<32xf32>
    %adb1b3pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1b3pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsb3pbt = stablehlo.multiply %adb1b3pbt, %b3pbtm : tensor<32xf32>
    %admgb3pbt = stablehlo.multiply %adob1b3pbt, %b3dpbt : tensor<32xf32>
    %admnb3pbt = stablehlo.add %admsb3pbt, %admgb3pbt : tensor<32xf32>
    %adb2b3pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b3pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb3pbt = stablehlo.multiply %adb2b3pbt, %b3pbtv : tensor<32xf32>
    %adg2b3pbt = stablehlo.multiply %b3dpbt, %b3dpbt : tensor<32xf32>
    %advgb3pbt = stablehlo.multiply %adob2b3pbt, %adg2b3pbt : tensor<32xf32>
    %advnb3pbt = stablehlo.add %advsb3pbt, %advgb3pbt : tensor<32xf32>
    %adbc1b3pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2b3pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhb3pbt = stablehlo.divide %admnb3pbt, %adbc1b3pbt : tensor<32xf32>
    %advhb3pbt = stablehlo.divide %advnb3pbt, %adbc2b3pbt : tensor<32xf32>
    %adlrb3pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsb3pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqb3pbt = stablehlo.sqrt %advhb3pbt : tensor<32xf32>
    %addenb3pbt = stablehlo.add %adsqb3pbt, %adepsb3pbt : tensor<32xf32>
    %adratb3pbt = stablehlo.divide %admhb3pbt, %addenb3pbt : tensor<32xf32>
    %adstb3pbt = stablehlo.multiply %adlrb3pbt, %adratb3pbt : tensor<32xf32>
    %adsubb3pbt = stablehlo.subtract %b3pbt, %adstb3pbt : tensor<32xf32>
    %adwdb3pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrb3pbt = stablehlo.multiply %adwdb3pbt, %adlrb3pbt : tensor<32xf32>
    %adwdpb3pbt = stablehlo.multiply %adwdlrb3pbt, %b3pbt : tensor<32xf32>
    %adnewb3pbt = stablehlo.subtract %adsubb3pbt, %adwdpb3pbt : tensor<32xf32>
    %adb1b4eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %adob1b4eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %admsb4eW = stablehlo.multiply %adb1b4eW, %b4eWm : tensor<128x32x1x1xf32>
    %admgb4eW = stablehlo.multiply %adob1b4eW, %b4deW : tensor<128x32x1x1xf32>
    %admnb4eW = stablehlo.add %admsb4eW, %admgb4eW : tensor<128x32x1x1xf32>
    %adb2b4eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %adob2b4eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %advsb4eW = stablehlo.multiply %adb2b4eW, %b4eWv : tensor<128x32x1x1xf32>
    %adg2b4eW = stablehlo.multiply %b4deW, %b4deW : tensor<128x32x1x1xf32>
    %advgb4eW = stablehlo.multiply %adob2b4eW, %adg2b4eW : tensor<128x32x1x1xf32>
    %advnb4eW = stablehlo.add %advsb4eW, %advgb4eW : tensor<128x32x1x1xf32>
    %adbc1b4eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %adbc2b4eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %admhb4eW = stablehlo.divide %admnb4eW, %adbc1b4eW : tensor<128x32x1x1xf32>
    %advhb4eW = stablehlo.divide %advnb4eW, %adbc2b4eW : tensor<128x32x1x1xf32>
    %adlrb4eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %adepsb4eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %adsqb4eW = stablehlo.sqrt %advhb4eW : tensor<128x32x1x1xf32>
    %addenb4eW = stablehlo.add %adsqb4eW, %adepsb4eW : tensor<128x32x1x1xf32>
    %adratb4eW = stablehlo.divide %admhb4eW, %addenb4eW : tensor<128x32x1x1xf32>
    %adstb4eW = stablehlo.multiply %adlrb4eW, %adratb4eW : tensor<128x32x1x1xf32>
    %adsubb4eW = stablehlo.subtract %b4eW, %adstb4eW : tensor<128x32x1x1xf32>
    %adwdb4eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %adwdlrb4eW = stablehlo.multiply %adwdb4eW, %adlrb4eW : tensor<128x32x1x1xf32>
    %adwdpb4eW = stablehlo.multiply %adwdlrb4eW, %b4eW : tensor<128x32x1x1xf32>
    %adnewb4eW = stablehlo.subtract %adsubb4eW, %adwdpb4eW : tensor<128x32x1x1xf32>
    %adb1b4eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1b4eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsb4eb = stablehlo.multiply %adb1b4eb, %b4ebm : tensor<128xf32>
    %admgb4eb = stablehlo.multiply %adob1b4eb, %b4deb : tensor<128xf32>
    %admnb4eb = stablehlo.add %admsb4eb, %admgb4eb : tensor<128xf32>
    %adb2b4eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b4eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb4eb = stablehlo.multiply %adb2b4eb, %b4ebv : tensor<128xf32>
    %adg2b4eb = stablehlo.multiply %b4deb, %b4deb : tensor<128xf32>
    %advgb4eb = stablehlo.multiply %adob2b4eb, %adg2b4eb : tensor<128xf32>
    %advnb4eb = stablehlo.add %advsb4eb, %advgb4eb : tensor<128xf32>
    %adbc1b4eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2b4eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhb4eb = stablehlo.divide %admnb4eb, %adbc1b4eb : tensor<128xf32>
    %advhb4eb = stablehlo.divide %advnb4eb, %adbc2b4eb : tensor<128xf32>
    %adlrb4eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsb4eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqb4eb = stablehlo.sqrt %advhb4eb : tensor<128xf32>
    %addenb4eb = stablehlo.add %adsqb4eb, %adepsb4eb : tensor<128xf32>
    %adratb4eb = stablehlo.divide %admhb4eb, %addenb4eb : tensor<128xf32>
    %adstb4eb = stablehlo.multiply %adlrb4eb, %adratb4eb : tensor<128xf32>
    %adsubb4eb = stablehlo.subtract %b4eb, %adstb4eb : tensor<128xf32>
    %adwdb4eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrb4eb = stablehlo.multiply %adwdb4eb, %adlrb4eb : tensor<128xf32>
    %adwdpb4eb = stablehlo.multiply %adwdlrb4eb, %b4eb : tensor<128xf32>
    %adnewb4eb = stablehlo.subtract %adsubb4eb, %adwdpb4eb : tensor<128xf32>
    %adb1b4eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1b4eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsb4eg = stablehlo.multiply %adb1b4eg, %b4egm : tensor<128xf32>
    %admgb4eg = stablehlo.multiply %adob1b4eg, %b4deg : tensor<128xf32>
    %admnb4eg = stablehlo.add %admsb4eg, %admgb4eg : tensor<128xf32>
    %adb2b4eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b4eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb4eg = stablehlo.multiply %adb2b4eg, %b4egv : tensor<128xf32>
    %adg2b4eg = stablehlo.multiply %b4deg, %b4deg : tensor<128xf32>
    %advgb4eg = stablehlo.multiply %adob2b4eg, %adg2b4eg : tensor<128xf32>
    %advnb4eg = stablehlo.add %advsb4eg, %advgb4eg : tensor<128xf32>
    %adbc1b4eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2b4eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhb4eg = stablehlo.divide %admnb4eg, %adbc1b4eg : tensor<128xf32>
    %advhb4eg = stablehlo.divide %advnb4eg, %adbc2b4eg : tensor<128xf32>
    %adlrb4eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsb4eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqb4eg = stablehlo.sqrt %advhb4eg : tensor<128xf32>
    %addenb4eg = stablehlo.add %adsqb4eg, %adepsb4eg : tensor<128xf32>
    %adratb4eg = stablehlo.divide %admhb4eg, %addenb4eg : tensor<128xf32>
    %adstb4eg = stablehlo.multiply %adlrb4eg, %adratb4eg : tensor<128xf32>
    %adsubb4eg = stablehlo.subtract %b4eg, %adstb4eg : tensor<128xf32>
    %adwdb4eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrb4eg = stablehlo.multiply %adwdb4eg, %adlrb4eg : tensor<128xf32>
    %adwdpb4eg = stablehlo.multiply %adwdlrb4eg, %b4eg : tensor<128xf32>
    %adnewb4eg = stablehlo.subtract %adsubb4eg, %adwdpb4eg : tensor<128xf32>
    %adb1b4ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1b4ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsb4ebt = stablehlo.multiply %adb1b4ebt, %b4ebtm : tensor<128xf32>
    %admgb4ebt = stablehlo.multiply %adob1b4ebt, %b4debt : tensor<128xf32>
    %admnb4ebt = stablehlo.add %admsb4ebt, %admgb4ebt : tensor<128xf32>
    %adb2b4ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b4ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb4ebt = stablehlo.multiply %adb2b4ebt, %b4ebtv : tensor<128xf32>
    %adg2b4ebt = stablehlo.multiply %b4debt, %b4debt : tensor<128xf32>
    %advgb4ebt = stablehlo.multiply %adob2b4ebt, %adg2b4ebt : tensor<128xf32>
    %advnb4ebt = stablehlo.add %advsb4ebt, %advgb4ebt : tensor<128xf32>
    %adbc1b4ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2b4ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhb4ebt = stablehlo.divide %admnb4ebt, %adbc1b4ebt : tensor<128xf32>
    %advhb4ebt = stablehlo.divide %advnb4ebt, %adbc2b4ebt : tensor<128xf32>
    %adlrb4ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsb4ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqb4ebt = stablehlo.sqrt %advhb4ebt : tensor<128xf32>
    %addenb4ebt = stablehlo.add %adsqb4ebt, %adepsb4ebt : tensor<128xf32>
    %adratb4ebt = stablehlo.divide %admhb4ebt, %addenb4ebt : tensor<128xf32>
    %adstb4ebt = stablehlo.multiply %adlrb4ebt, %adratb4ebt : tensor<128xf32>
    %adsubb4ebt = stablehlo.subtract %b4ebt, %adstb4ebt : tensor<128xf32>
    %adwdb4ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrb4ebt = stablehlo.multiply %adwdb4ebt, %adlrb4ebt : tensor<128xf32>
    %adwdpb4ebt = stablehlo.multiply %adwdlrb4ebt, %b4ebt : tensor<128xf32>
    %adnewb4ebt = stablehlo.subtract %adsubb4ebt, %adwdpb4ebt : tensor<128xf32>
    %adb1b4dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %adob1b4dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %admsb4dW = stablehlo.multiply %adb1b4dW, %b4dWm : tensor<128x1x3x3xf32>
    %admgb4dW = stablehlo.multiply %adob1b4dW, %b4ddW : tensor<128x1x3x3xf32>
    %admnb4dW = stablehlo.add %admsb4dW, %admgb4dW : tensor<128x1x3x3xf32>
    %adb2b4dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %adob2b4dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %advsb4dW = stablehlo.multiply %adb2b4dW, %b4dWv : tensor<128x1x3x3xf32>
    %adg2b4dW = stablehlo.multiply %b4ddW, %b4ddW : tensor<128x1x3x3xf32>
    %advgb4dW = stablehlo.multiply %adob2b4dW, %adg2b4dW : tensor<128x1x3x3xf32>
    %advnb4dW = stablehlo.add %advsb4dW, %advgb4dW : tensor<128x1x3x3xf32>
    %adbc1b4dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %adbc2b4dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %admhb4dW = stablehlo.divide %admnb4dW, %adbc1b4dW : tensor<128x1x3x3xf32>
    %advhb4dW = stablehlo.divide %advnb4dW, %adbc2b4dW : tensor<128x1x3x3xf32>
    %adlrb4dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %adepsb4dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %adsqb4dW = stablehlo.sqrt %advhb4dW : tensor<128x1x3x3xf32>
    %addenb4dW = stablehlo.add %adsqb4dW, %adepsb4dW : tensor<128x1x3x3xf32>
    %adratb4dW = stablehlo.divide %admhb4dW, %addenb4dW : tensor<128x1x3x3xf32>
    %adstb4dW = stablehlo.multiply %adlrb4dW, %adratb4dW : tensor<128x1x3x3xf32>
    %adsubb4dW = stablehlo.subtract %b4dW, %adstb4dW : tensor<128x1x3x3xf32>
    %adwdb4dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %adwdlrb4dW = stablehlo.multiply %adwdb4dW, %adlrb4dW : tensor<128x1x3x3xf32>
    %adwdpb4dW = stablehlo.multiply %adwdlrb4dW, %b4dW : tensor<128x1x3x3xf32>
    %adnewb4dW = stablehlo.subtract %adsubb4dW, %adwdpb4dW : tensor<128x1x3x3xf32>
    %adb1b4db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1b4db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsb4db = stablehlo.multiply %adb1b4db, %b4dbm : tensor<128xf32>
    %admgb4db = stablehlo.multiply %adob1b4db, %b4ddb : tensor<128xf32>
    %admnb4db = stablehlo.add %admsb4db, %admgb4db : tensor<128xf32>
    %adb2b4db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b4db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb4db = stablehlo.multiply %adb2b4db, %b4dbv : tensor<128xf32>
    %adg2b4db = stablehlo.multiply %b4ddb, %b4ddb : tensor<128xf32>
    %advgb4db = stablehlo.multiply %adob2b4db, %adg2b4db : tensor<128xf32>
    %advnb4db = stablehlo.add %advsb4db, %advgb4db : tensor<128xf32>
    %adbc1b4db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2b4db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhb4db = stablehlo.divide %admnb4db, %adbc1b4db : tensor<128xf32>
    %advhb4db = stablehlo.divide %advnb4db, %adbc2b4db : tensor<128xf32>
    %adlrb4db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsb4db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqb4db = stablehlo.sqrt %advhb4db : tensor<128xf32>
    %addenb4db = stablehlo.add %adsqb4db, %adepsb4db : tensor<128xf32>
    %adratb4db = stablehlo.divide %admhb4db, %addenb4db : tensor<128xf32>
    %adstb4db = stablehlo.multiply %adlrb4db, %adratb4db : tensor<128xf32>
    %adsubb4db = stablehlo.subtract %b4db, %adstb4db : tensor<128xf32>
    %adwdb4db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrb4db = stablehlo.multiply %adwdb4db, %adlrb4db : tensor<128xf32>
    %adwdpb4db = stablehlo.multiply %adwdlrb4db, %b4db : tensor<128xf32>
    %adnewb4db = stablehlo.subtract %adsubb4db, %adwdpb4db : tensor<128xf32>
    %adb1b4dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1b4dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsb4dg = stablehlo.multiply %adb1b4dg, %b4dgm : tensor<128xf32>
    %admgb4dg = stablehlo.multiply %adob1b4dg, %b4ddg : tensor<128xf32>
    %admnb4dg = stablehlo.add %admsb4dg, %admgb4dg : tensor<128xf32>
    %adb2b4dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b4dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb4dg = stablehlo.multiply %adb2b4dg, %b4dgv : tensor<128xf32>
    %adg2b4dg = stablehlo.multiply %b4ddg, %b4ddg : tensor<128xf32>
    %advgb4dg = stablehlo.multiply %adob2b4dg, %adg2b4dg : tensor<128xf32>
    %advnb4dg = stablehlo.add %advsb4dg, %advgb4dg : tensor<128xf32>
    %adbc1b4dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2b4dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhb4dg = stablehlo.divide %admnb4dg, %adbc1b4dg : tensor<128xf32>
    %advhb4dg = stablehlo.divide %advnb4dg, %adbc2b4dg : tensor<128xf32>
    %adlrb4dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsb4dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqb4dg = stablehlo.sqrt %advhb4dg : tensor<128xf32>
    %addenb4dg = stablehlo.add %adsqb4dg, %adepsb4dg : tensor<128xf32>
    %adratb4dg = stablehlo.divide %admhb4dg, %addenb4dg : tensor<128xf32>
    %adstb4dg = stablehlo.multiply %adlrb4dg, %adratb4dg : tensor<128xf32>
    %adsubb4dg = stablehlo.subtract %b4dg, %adstb4dg : tensor<128xf32>
    %adwdb4dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrb4dg = stablehlo.multiply %adwdb4dg, %adlrb4dg : tensor<128xf32>
    %adwdpb4dg = stablehlo.multiply %adwdlrb4dg, %b4dg : tensor<128xf32>
    %adnewb4dg = stablehlo.subtract %adsubb4dg, %adwdpb4dg : tensor<128xf32>
    %adb1b4dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1b4dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsb4dbt = stablehlo.multiply %adb1b4dbt, %b4dbtm : tensor<128xf32>
    %admgb4dbt = stablehlo.multiply %adob1b4dbt, %b4ddbt : tensor<128xf32>
    %admnb4dbt = stablehlo.add %admsb4dbt, %admgb4dbt : tensor<128xf32>
    %adb2b4dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b4dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb4dbt = stablehlo.multiply %adb2b4dbt, %b4dbtv : tensor<128xf32>
    %adg2b4dbt = stablehlo.multiply %b4ddbt, %b4ddbt : tensor<128xf32>
    %advgb4dbt = stablehlo.multiply %adob2b4dbt, %adg2b4dbt : tensor<128xf32>
    %advnb4dbt = stablehlo.add %advsb4dbt, %advgb4dbt : tensor<128xf32>
    %adbc1b4dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2b4dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhb4dbt = stablehlo.divide %admnb4dbt, %adbc1b4dbt : tensor<128xf32>
    %advhb4dbt = stablehlo.divide %advnb4dbt, %adbc2b4dbt : tensor<128xf32>
    %adlrb4dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsb4dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqb4dbt = stablehlo.sqrt %advhb4dbt : tensor<128xf32>
    %addenb4dbt = stablehlo.add %adsqb4dbt, %adepsb4dbt : tensor<128xf32>
    %adratb4dbt = stablehlo.divide %admhb4dbt, %addenb4dbt : tensor<128xf32>
    %adstb4dbt = stablehlo.multiply %adlrb4dbt, %adratb4dbt : tensor<128xf32>
    %adsubb4dbt = stablehlo.subtract %b4dbt, %adstb4dbt : tensor<128xf32>
    %adwdb4dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrb4dbt = stablehlo.multiply %adwdb4dbt, %adlrb4dbt : tensor<128xf32>
    %adwdpb4dbt = stablehlo.multiply %adwdlrb4dbt, %b4dbt : tensor<128xf32>
    %adnewb4dbt = stablehlo.subtract %adsubb4dbt, %adwdpb4dbt : tensor<128xf32>
    %adb1b4pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x128x1x1xf32>
    %adob1b4pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x128x1x1xf32>
    %admsb4pW = stablehlo.multiply %adb1b4pW, %b4pWm : tensor<32x128x1x1xf32>
    %admgb4pW = stablehlo.multiply %adob1b4pW, %b4dpW : tensor<32x128x1x1xf32>
    %admnb4pW = stablehlo.add %admsb4pW, %admgb4pW : tensor<32x128x1x1xf32>
    %adb2b4pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x128x1x1xf32>
    %adob2b4pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x128x1x1xf32>
    %advsb4pW = stablehlo.multiply %adb2b4pW, %b4pWv : tensor<32x128x1x1xf32>
    %adg2b4pW = stablehlo.multiply %b4dpW, %b4dpW : tensor<32x128x1x1xf32>
    %advgb4pW = stablehlo.multiply %adob2b4pW, %adg2b4pW : tensor<32x128x1x1xf32>
    %advnb4pW = stablehlo.add %advsb4pW, %advgb4pW : tensor<32x128x1x1xf32>
    %adbc1b4pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x128x1x1xf32>
    %adbc2b4pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x128x1x1xf32>
    %admhb4pW = stablehlo.divide %admnb4pW, %adbc1b4pW : tensor<32x128x1x1xf32>
    %advhb4pW = stablehlo.divide %advnb4pW, %adbc2b4pW : tensor<32x128x1x1xf32>
    %adlrb4pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x128x1x1xf32>
    %adepsb4pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x128x1x1xf32>
    %adsqb4pW = stablehlo.sqrt %advhb4pW : tensor<32x128x1x1xf32>
    %addenb4pW = stablehlo.add %adsqb4pW, %adepsb4pW : tensor<32x128x1x1xf32>
    %adratb4pW = stablehlo.divide %admhb4pW, %addenb4pW : tensor<32x128x1x1xf32>
    %adstb4pW = stablehlo.multiply %adlrb4pW, %adratb4pW : tensor<32x128x1x1xf32>
    %adsubb4pW = stablehlo.subtract %b4pW, %adstb4pW : tensor<32x128x1x1xf32>
    %adwdb4pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x128x1x1xf32>
    %adwdlrb4pW = stablehlo.multiply %adwdb4pW, %adlrb4pW : tensor<32x128x1x1xf32>
    %adwdpb4pW = stablehlo.multiply %adwdlrb4pW, %b4pW : tensor<32x128x1x1xf32>
    %adnewb4pW = stablehlo.subtract %adsubb4pW, %adwdpb4pW : tensor<32x128x1x1xf32>
    %adb1b4pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1b4pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsb4pb = stablehlo.multiply %adb1b4pb, %b4pbm : tensor<32xf32>
    %admgb4pb = stablehlo.multiply %adob1b4pb, %b4dpb : tensor<32xf32>
    %admnb4pb = stablehlo.add %admsb4pb, %admgb4pb : tensor<32xf32>
    %adb2b4pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b4pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb4pb = stablehlo.multiply %adb2b4pb, %b4pbv : tensor<32xf32>
    %adg2b4pb = stablehlo.multiply %b4dpb, %b4dpb : tensor<32xf32>
    %advgb4pb = stablehlo.multiply %adob2b4pb, %adg2b4pb : tensor<32xf32>
    %advnb4pb = stablehlo.add %advsb4pb, %advgb4pb : tensor<32xf32>
    %adbc1b4pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2b4pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhb4pb = stablehlo.divide %admnb4pb, %adbc1b4pb : tensor<32xf32>
    %advhb4pb = stablehlo.divide %advnb4pb, %adbc2b4pb : tensor<32xf32>
    %adlrb4pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsb4pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqb4pb = stablehlo.sqrt %advhb4pb : tensor<32xf32>
    %addenb4pb = stablehlo.add %adsqb4pb, %adepsb4pb : tensor<32xf32>
    %adratb4pb = stablehlo.divide %admhb4pb, %addenb4pb : tensor<32xf32>
    %adstb4pb = stablehlo.multiply %adlrb4pb, %adratb4pb : tensor<32xf32>
    %adsubb4pb = stablehlo.subtract %b4pb, %adstb4pb : tensor<32xf32>
    %adwdb4pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrb4pb = stablehlo.multiply %adwdb4pb, %adlrb4pb : tensor<32xf32>
    %adwdpb4pb = stablehlo.multiply %adwdlrb4pb, %b4pb : tensor<32xf32>
    %adnewb4pb = stablehlo.subtract %adsubb4pb, %adwdpb4pb : tensor<32xf32>
    %adb1b4pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1b4pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsb4pg = stablehlo.multiply %adb1b4pg, %b4pgm : tensor<32xf32>
    %admgb4pg = stablehlo.multiply %adob1b4pg, %b4dpg : tensor<32xf32>
    %admnb4pg = stablehlo.add %admsb4pg, %admgb4pg : tensor<32xf32>
    %adb2b4pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b4pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb4pg = stablehlo.multiply %adb2b4pg, %b4pgv : tensor<32xf32>
    %adg2b4pg = stablehlo.multiply %b4dpg, %b4dpg : tensor<32xf32>
    %advgb4pg = stablehlo.multiply %adob2b4pg, %adg2b4pg : tensor<32xf32>
    %advnb4pg = stablehlo.add %advsb4pg, %advgb4pg : tensor<32xf32>
    %adbc1b4pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2b4pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhb4pg = stablehlo.divide %admnb4pg, %adbc1b4pg : tensor<32xf32>
    %advhb4pg = stablehlo.divide %advnb4pg, %adbc2b4pg : tensor<32xf32>
    %adlrb4pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsb4pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqb4pg = stablehlo.sqrt %advhb4pg : tensor<32xf32>
    %addenb4pg = stablehlo.add %adsqb4pg, %adepsb4pg : tensor<32xf32>
    %adratb4pg = stablehlo.divide %admhb4pg, %addenb4pg : tensor<32xf32>
    %adstb4pg = stablehlo.multiply %adlrb4pg, %adratb4pg : tensor<32xf32>
    %adsubb4pg = stablehlo.subtract %b4pg, %adstb4pg : tensor<32xf32>
    %adwdb4pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrb4pg = stablehlo.multiply %adwdb4pg, %adlrb4pg : tensor<32xf32>
    %adwdpb4pg = stablehlo.multiply %adwdlrb4pg, %b4pg : tensor<32xf32>
    %adnewb4pg = stablehlo.subtract %adsubb4pg, %adwdpb4pg : tensor<32xf32>
    %adb1b4pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1b4pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsb4pbt = stablehlo.multiply %adb1b4pbt, %b4pbtm : tensor<32xf32>
    %admgb4pbt = stablehlo.multiply %adob1b4pbt, %b4dpbt : tensor<32xf32>
    %admnb4pbt = stablehlo.add %admsb4pbt, %admgb4pbt : tensor<32xf32>
    %adb2b4pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b4pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb4pbt = stablehlo.multiply %adb2b4pbt, %b4pbtv : tensor<32xf32>
    %adg2b4pbt = stablehlo.multiply %b4dpbt, %b4dpbt : tensor<32xf32>
    %advgb4pbt = stablehlo.multiply %adob2b4pbt, %adg2b4pbt : tensor<32xf32>
    %advnb4pbt = stablehlo.add %advsb4pbt, %advgb4pbt : tensor<32xf32>
    %adbc1b4pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2b4pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhb4pbt = stablehlo.divide %admnb4pbt, %adbc1b4pbt : tensor<32xf32>
    %advhb4pbt = stablehlo.divide %advnb4pbt, %adbc2b4pbt : tensor<32xf32>
    %adlrb4pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsb4pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqb4pbt = stablehlo.sqrt %advhb4pbt : tensor<32xf32>
    %addenb4pbt = stablehlo.add %adsqb4pbt, %adepsb4pbt : tensor<32xf32>
    %adratb4pbt = stablehlo.divide %admhb4pbt, %addenb4pbt : tensor<32xf32>
    %adstb4pbt = stablehlo.multiply %adlrb4pbt, %adratb4pbt : tensor<32xf32>
    %adsubb4pbt = stablehlo.subtract %b4pbt, %adstb4pbt : tensor<32xf32>
    %adwdb4pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrb4pbt = stablehlo.multiply %adwdb4pbt, %adlrb4pbt : tensor<32xf32>
    %adwdpb4pbt = stablehlo.multiply %adwdlrb4pbt, %b4pbt : tensor<32xf32>
    %adnewb4pbt = stablehlo.subtract %adsubb4pbt, %adwdpb4pbt : tensor<32xf32>
    %adb1b5eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %adob1b5eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %admsb5eW = stablehlo.multiply %adb1b5eW, %b5eWm : tensor<128x32x1x1xf32>
    %admgb5eW = stablehlo.multiply %adob1b5eW, %b5deW : tensor<128x32x1x1xf32>
    %admnb5eW = stablehlo.add %admsb5eW, %admgb5eW : tensor<128x32x1x1xf32>
    %adb2b5eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %adob2b5eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %advsb5eW = stablehlo.multiply %adb2b5eW, %b5eWv : tensor<128x32x1x1xf32>
    %adg2b5eW = stablehlo.multiply %b5deW, %b5deW : tensor<128x32x1x1xf32>
    %advgb5eW = stablehlo.multiply %adob2b5eW, %adg2b5eW : tensor<128x32x1x1xf32>
    %advnb5eW = stablehlo.add %advsb5eW, %advgb5eW : tensor<128x32x1x1xf32>
    %adbc1b5eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %adbc2b5eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %admhb5eW = stablehlo.divide %admnb5eW, %adbc1b5eW : tensor<128x32x1x1xf32>
    %advhb5eW = stablehlo.divide %advnb5eW, %adbc2b5eW : tensor<128x32x1x1xf32>
    %adlrb5eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %adepsb5eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %adsqb5eW = stablehlo.sqrt %advhb5eW : tensor<128x32x1x1xf32>
    %addenb5eW = stablehlo.add %adsqb5eW, %adepsb5eW : tensor<128x32x1x1xf32>
    %adratb5eW = stablehlo.divide %admhb5eW, %addenb5eW : tensor<128x32x1x1xf32>
    %adstb5eW = stablehlo.multiply %adlrb5eW, %adratb5eW : tensor<128x32x1x1xf32>
    %adsubb5eW = stablehlo.subtract %b5eW, %adstb5eW : tensor<128x32x1x1xf32>
    %adwdb5eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x32x1x1xf32>
    %adwdlrb5eW = stablehlo.multiply %adwdb5eW, %adlrb5eW : tensor<128x32x1x1xf32>
    %adwdpb5eW = stablehlo.multiply %adwdlrb5eW, %b5eW : tensor<128x32x1x1xf32>
    %adnewb5eW = stablehlo.subtract %adsubb5eW, %adwdpb5eW : tensor<128x32x1x1xf32>
    %adb1b5eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1b5eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsb5eb = stablehlo.multiply %adb1b5eb, %b5ebm : tensor<128xf32>
    %admgb5eb = stablehlo.multiply %adob1b5eb, %b5deb : tensor<128xf32>
    %admnb5eb = stablehlo.add %admsb5eb, %admgb5eb : tensor<128xf32>
    %adb2b5eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b5eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb5eb = stablehlo.multiply %adb2b5eb, %b5ebv : tensor<128xf32>
    %adg2b5eb = stablehlo.multiply %b5deb, %b5deb : tensor<128xf32>
    %advgb5eb = stablehlo.multiply %adob2b5eb, %adg2b5eb : tensor<128xf32>
    %advnb5eb = stablehlo.add %advsb5eb, %advgb5eb : tensor<128xf32>
    %adbc1b5eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2b5eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhb5eb = stablehlo.divide %admnb5eb, %adbc1b5eb : tensor<128xf32>
    %advhb5eb = stablehlo.divide %advnb5eb, %adbc2b5eb : tensor<128xf32>
    %adlrb5eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsb5eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqb5eb = stablehlo.sqrt %advhb5eb : tensor<128xf32>
    %addenb5eb = stablehlo.add %adsqb5eb, %adepsb5eb : tensor<128xf32>
    %adratb5eb = stablehlo.divide %admhb5eb, %addenb5eb : tensor<128xf32>
    %adstb5eb = stablehlo.multiply %adlrb5eb, %adratb5eb : tensor<128xf32>
    %adsubb5eb = stablehlo.subtract %b5eb, %adstb5eb : tensor<128xf32>
    %adwdb5eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrb5eb = stablehlo.multiply %adwdb5eb, %adlrb5eb : tensor<128xf32>
    %adwdpb5eb = stablehlo.multiply %adwdlrb5eb, %b5eb : tensor<128xf32>
    %adnewb5eb = stablehlo.subtract %adsubb5eb, %adwdpb5eb : tensor<128xf32>
    %adb1b5eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1b5eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsb5eg = stablehlo.multiply %adb1b5eg, %b5egm : tensor<128xf32>
    %admgb5eg = stablehlo.multiply %adob1b5eg, %b5deg : tensor<128xf32>
    %admnb5eg = stablehlo.add %admsb5eg, %admgb5eg : tensor<128xf32>
    %adb2b5eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b5eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb5eg = stablehlo.multiply %adb2b5eg, %b5egv : tensor<128xf32>
    %adg2b5eg = stablehlo.multiply %b5deg, %b5deg : tensor<128xf32>
    %advgb5eg = stablehlo.multiply %adob2b5eg, %adg2b5eg : tensor<128xf32>
    %advnb5eg = stablehlo.add %advsb5eg, %advgb5eg : tensor<128xf32>
    %adbc1b5eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2b5eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhb5eg = stablehlo.divide %admnb5eg, %adbc1b5eg : tensor<128xf32>
    %advhb5eg = stablehlo.divide %advnb5eg, %adbc2b5eg : tensor<128xf32>
    %adlrb5eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsb5eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqb5eg = stablehlo.sqrt %advhb5eg : tensor<128xf32>
    %addenb5eg = stablehlo.add %adsqb5eg, %adepsb5eg : tensor<128xf32>
    %adratb5eg = stablehlo.divide %admhb5eg, %addenb5eg : tensor<128xf32>
    %adstb5eg = stablehlo.multiply %adlrb5eg, %adratb5eg : tensor<128xf32>
    %adsubb5eg = stablehlo.subtract %b5eg, %adstb5eg : tensor<128xf32>
    %adwdb5eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrb5eg = stablehlo.multiply %adwdb5eg, %adlrb5eg : tensor<128xf32>
    %adwdpb5eg = stablehlo.multiply %adwdlrb5eg, %b5eg : tensor<128xf32>
    %adnewb5eg = stablehlo.subtract %adsubb5eg, %adwdpb5eg : tensor<128xf32>
    %adb1b5ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1b5ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsb5ebt = stablehlo.multiply %adb1b5ebt, %b5ebtm : tensor<128xf32>
    %admgb5ebt = stablehlo.multiply %adob1b5ebt, %b5debt : tensor<128xf32>
    %admnb5ebt = stablehlo.add %admsb5ebt, %admgb5ebt : tensor<128xf32>
    %adb2b5ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b5ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb5ebt = stablehlo.multiply %adb2b5ebt, %b5ebtv : tensor<128xf32>
    %adg2b5ebt = stablehlo.multiply %b5debt, %b5debt : tensor<128xf32>
    %advgb5ebt = stablehlo.multiply %adob2b5ebt, %adg2b5ebt : tensor<128xf32>
    %advnb5ebt = stablehlo.add %advsb5ebt, %advgb5ebt : tensor<128xf32>
    %adbc1b5ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2b5ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhb5ebt = stablehlo.divide %admnb5ebt, %adbc1b5ebt : tensor<128xf32>
    %advhb5ebt = stablehlo.divide %advnb5ebt, %adbc2b5ebt : tensor<128xf32>
    %adlrb5ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsb5ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqb5ebt = stablehlo.sqrt %advhb5ebt : tensor<128xf32>
    %addenb5ebt = stablehlo.add %adsqb5ebt, %adepsb5ebt : tensor<128xf32>
    %adratb5ebt = stablehlo.divide %admhb5ebt, %addenb5ebt : tensor<128xf32>
    %adstb5ebt = stablehlo.multiply %adlrb5ebt, %adratb5ebt : tensor<128xf32>
    %adsubb5ebt = stablehlo.subtract %b5ebt, %adstb5ebt : tensor<128xf32>
    %adwdb5ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrb5ebt = stablehlo.multiply %adwdb5ebt, %adlrb5ebt : tensor<128xf32>
    %adwdpb5ebt = stablehlo.multiply %adwdlrb5ebt, %b5ebt : tensor<128xf32>
    %adnewb5ebt = stablehlo.subtract %adsubb5ebt, %adwdpb5ebt : tensor<128xf32>
    %adb1b5dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %adob1b5dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %admsb5dW = stablehlo.multiply %adb1b5dW, %b5dWm : tensor<128x1x3x3xf32>
    %admgb5dW = stablehlo.multiply %adob1b5dW, %b5ddW : tensor<128x1x3x3xf32>
    %admnb5dW = stablehlo.add %admsb5dW, %admgb5dW : tensor<128x1x3x3xf32>
    %adb2b5dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %adob2b5dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %advsb5dW = stablehlo.multiply %adb2b5dW, %b5dWv : tensor<128x1x3x3xf32>
    %adg2b5dW = stablehlo.multiply %b5ddW, %b5ddW : tensor<128x1x3x3xf32>
    %advgb5dW = stablehlo.multiply %adob2b5dW, %adg2b5dW : tensor<128x1x3x3xf32>
    %advnb5dW = stablehlo.add %advsb5dW, %advgb5dW : tensor<128x1x3x3xf32>
    %adbc1b5dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %adbc2b5dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %admhb5dW = stablehlo.divide %admnb5dW, %adbc1b5dW : tensor<128x1x3x3xf32>
    %advhb5dW = stablehlo.divide %advnb5dW, %adbc2b5dW : tensor<128x1x3x3xf32>
    %adlrb5dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %adepsb5dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %adsqb5dW = stablehlo.sqrt %advhb5dW : tensor<128x1x3x3xf32>
    %addenb5dW = stablehlo.add %adsqb5dW, %adepsb5dW : tensor<128x1x3x3xf32>
    %adratb5dW = stablehlo.divide %admhb5dW, %addenb5dW : tensor<128x1x3x3xf32>
    %adstb5dW = stablehlo.multiply %adlrb5dW, %adratb5dW : tensor<128x1x3x3xf32>
    %adsubb5dW = stablehlo.subtract %b5dW, %adstb5dW : tensor<128x1x3x3xf32>
    %adwdb5dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x1x3x3xf32>
    %adwdlrb5dW = stablehlo.multiply %adwdb5dW, %adlrb5dW : tensor<128x1x3x3xf32>
    %adwdpb5dW = stablehlo.multiply %adwdlrb5dW, %b5dW : tensor<128x1x3x3xf32>
    %adnewb5dW = stablehlo.subtract %adsubb5dW, %adwdpb5dW : tensor<128x1x3x3xf32>
    %adb1b5db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1b5db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsb5db = stablehlo.multiply %adb1b5db, %b5dbm : tensor<128xf32>
    %admgb5db = stablehlo.multiply %adob1b5db, %b5ddb : tensor<128xf32>
    %admnb5db = stablehlo.add %admsb5db, %admgb5db : tensor<128xf32>
    %adb2b5db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b5db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb5db = stablehlo.multiply %adb2b5db, %b5dbv : tensor<128xf32>
    %adg2b5db = stablehlo.multiply %b5ddb, %b5ddb : tensor<128xf32>
    %advgb5db = stablehlo.multiply %adob2b5db, %adg2b5db : tensor<128xf32>
    %advnb5db = stablehlo.add %advsb5db, %advgb5db : tensor<128xf32>
    %adbc1b5db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2b5db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhb5db = stablehlo.divide %admnb5db, %adbc1b5db : tensor<128xf32>
    %advhb5db = stablehlo.divide %advnb5db, %adbc2b5db : tensor<128xf32>
    %adlrb5db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsb5db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqb5db = stablehlo.sqrt %advhb5db : tensor<128xf32>
    %addenb5db = stablehlo.add %adsqb5db, %adepsb5db : tensor<128xf32>
    %adratb5db = stablehlo.divide %admhb5db, %addenb5db : tensor<128xf32>
    %adstb5db = stablehlo.multiply %adlrb5db, %adratb5db : tensor<128xf32>
    %adsubb5db = stablehlo.subtract %b5db, %adstb5db : tensor<128xf32>
    %adwdb5db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrb5db = stablehlo.multiply %adwdb5db, %adlrb5db : tensor<128xf32>
    %adwdpb5db = stablehlo.multiply %adwdlrb5db, %b5db : tensor<128xf32>
    %adnewb5db = stablehlo.subtract %adsubb5db, %adwdpb5db : tensor<128xf32>
    %adb1b5dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1b5dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsb5dg = stablehlo.multiply %adb1b5dg, %b5dgm : tensor<128xf32>
    %admgb5dg = stablehlo.multiply %adob1b5dg, %b5ddg : tensor<128xf32>
    %admnb5dg = stablehlo.add %admsb5dg, %admgb5dg : tensor<128xf32>
    %adb2b5dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b5dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb5dg = stablehlo.multiply %adb2b5dg, %b5dgv : tensor<128xf32>
    %adg2b5dg = stablehlo.multiply %b5ddg, %b5ddg : tensor<128xf32>
    %advgb5dg = stablehlo.multiply %adob2b5dg, %adg2b5dg : tensor<128xf32>
    %advnb5dg = stablehlo.add %advsb5dg, %advgb5dg : tensor<128xf32>
    %adbc1b5dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2b5dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhb5dg = stablehlo.divide %admnb5dg, %adbc1b5dg : tensor<128xf32>
    %advhb5dg = stablehlo.divide %advnb5dg, %adbc2b5dg : tensor<128xf32>
    %adlrb5dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsb5dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqb5dg = stablehlo.sqrt %advhb5dg : tensor<128xf32>
    %addenb5dg = stablehlo.add %adsqb5dg, %adepsb5dg : tensor<128xf32>
    %adratb5dg = stablehlo.divide %admhb5dg, %addenb5dg : tensor<128xf32>
    %adstb5dg = stablehlo.multiply %adlrb5dg, %adratb5dg : tensor<128xf32>
    %adsubb5dg = stablehlo.subtract %b5dg, %adstb5dg : tensor<128xf32>
    %adwdb5dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrb5dg = stablehlo.multiply %adwdb5dg, %adlrb5dg : tensor<128xf32>
    %adwdpb5dg = stablehlo.multiply %adwdlrb5dg, %b5dg : tensor<128xf32>
    %adnewb5dg = stablehlo.subtract %adsubb5dg, %adwdpb5dg : tensor<128xf32>
    %adb1b5dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1b5dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsb5dbt = stablehlo.multiply %adb1b5dbt, %b5dbtm : tensor<128xf32>
    %admgb5dbt = stablehlo.multiply %adob1b5dbt, %b5ddbt : tensor<128xf32>
    %admnb5dbt = stablehlo.add %admsb5dbt, %admgb5dbt : tensor<128xf32>
    %adb2b5dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b5dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb5dbt = stablehlo.multiply %adb2b5dbt, %b5dbtv : tensor<128xf32>
    %adg2b5dbt = stablehlo.multiply %b5ddbt, %b5ddbt : tensor<128xf32>
    %advgb5dbt = stablehlo.multiply %adob2b5dbt, %adg2b5dbt : tensor<128xf32>
    %advnb5dbt = stablehlo.add %advsb5dbt, %advgb5dbt : tensor<128xf32>
    %adbc1b5dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2b5dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhb5dbt = stablehlo.divide %admnb5dbt, %adbc1b5dbt : tensor<128xf32>
    %advhb5dbt = stablehlo.divide %advnb5dbt, %adbc2b5dbt : tensor<128xf32>
    %adlrb5dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsb5dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqb5dbt = stablehlo.sqrt %advhb5dbt : tensor<128xf32>
    %addenb5dbt = stablehlo.add %adsqb5dbt, %adepsb5dbt : tensor<128xf32>
    %adratb5dbt = stablehlo.divide %admhb5dbt, %addenb5dbt : tensor<128xf32>
    %adstb5dbt = stablehlo.multiply %adlrb5dbt, %adratb5dbt : tensor<128xf32>
    %adsubb5dbt = stablehlo.subtract %b5dbt, %adstb5dbt : tensor<128xf32>
    %adwdb5dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrb5dbt = stablehlo.multiply %adwdb5dbt, %adlrb5dbt : tensor<128xf32>
    %adwdpb5dbt = stablehlo.multiply %adwdlrb5dbt, %b5dbt : tensor<128xf32>
    %adnewb5dbt = stablehlo.subtract %adsubb5dbt, %adwdpb5dbt : tensor<128xf32>
    %adb1b5pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x128x1x1xf32>
    %adob1b5pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x128x1x1xf32>
    %admsb5pW = stablehlo.multiply %adb1b5pW, %b5pWm : tensor<64x128x1x1xf32>
    %admgb5pW = stablehlo.multiply %adob1b5pW, %b5dpW : tensor<64x128x1x1xf32>
    %admnb5pW = stablehlo.add %admsb5pW, %admgb5pW : tensor<64x128x1x1xf32>
    %adb2b5pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x128x1x1xf32>
    %adob2b5pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x128x1x1xf32>
    %advsb5pW = stablehlo.multiply %adb2b5pW, %b5pWv : tensor<64x128x1x1xf32>
    %adg2b5pW = stablehlo.multiply %b5dpW, %b5dpW : tensor<64x128x1x1xf32>
    %advgb5pW = stablehlo.multiply %adob2b5pW, %adg2b5pW : tensor<64x128x1x1xf32>
    %advnb5pW = stablehlo.add %advsb5pW, %advgb5pW : tensor<64x128x1x1xf32>
    %adbc1b5pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x128x1x1xf32>
    %adbc2b5pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x128x1x1xf32>
    %admhb5pW = stablehlo.divide %admnb5pW, %adbc1b5pW : tensor<64x128x1x1xf32>
    %advhb5pW = stablehlo.divide %advnb5pW, %adbc2b5pW : tensor<64x128x1x1xf32>
    %adlrb5pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x128x1x1xf32>
    %adepsb5pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x128x1x1xf32>
    %adsqb5pW = stablehlo.sqrt %advhb5pW : tensor<64x128x1x1xf32>
    %addenb5pW = stablehlo.add %adsqb5pW, %adepsb5pW : tensor<64x128x1x1xf32>
    %adratb5pW = stablehlo.divide %admhb5pW, %addenb5pW : tensor<64x128x1x1xf32>
    %adstb5pW = stablehlo.multiply %adlrb5pW, %adratb5pW : tensor<64x128x1x1xf32>
    %adsubb5pW = stablehlo.subtract %b5pW, %adstb5pW : tensor<64x128x1x1xf32>
    %adwdb5pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x128x1x1xf32>
    %adwdlrb5pW = stablehlo.multiply %adwdb5pW, %adlrb5pW : tensor<64x128x1x1xf32>
    %adwdpb5pW = stablehlo.multiply %adwdlrb5pW, %b5pW : tensor<64x128x1x1xf32>
    %adnewb5pW = stablehlo.subtract %adsubb5pW, %adwdpb5pW : tensor<64x128x1x1xf32>
    %adb1b5pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b5pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb5pb = stablehlo.multiply %adb1b5pb, %b5pbm : tensor<64xf32>
    %admgb5pb = stablehlo.multiply %adob1b5pb, %b5dpb : tensor<64xf32>
    %admnb5pb = stablehlo.add %admsb5pb, %admgb5pb : tensor<64xf32>
    %adb2b5pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b5pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb5pb = stablehlo.multiply %adb2b5pb, %b5pbv : tensor<64xf32>
    %adg2b5pb = stablehlo.multiply %b5dpb, %b5dpb : tensor<64xf32>
    %advgb5pb = stablehlo.multiply %adob2b5pb, %adg2b5pb : tensor<64xf32>
    %advnb5pb = stablehlo.add %advsb5pb, %advgb5pb : tensor<64xf32>
    %adbc1b5pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b5pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb5pb = stablehlo.divide %admnb5pb, %adbc1b5pb : tensor<64xf32>
    %advhb5pb = stablehlo.divide %advnb5pb, %adbc2b5pb : tensor<64xf32>
    %adlrb5pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb5pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb5pb = stablehlo.sqrt %advhb5pb : tensor<64xf32>
    %addenb5pb = stablehlo.add %adsqb5pb, %adepsb5pb : tensor<64xf32>
    %adratb5pb = stablehlo.divide %admhb5pb, %addenb5pb : tensor<64xf32>
    %adstb5pb = stablehlo.multiply %adlrb5pb, %adratb5pb : tensor<64xf32>
    %adsubb5pb = stablehlo.subtract %b5pb, %adstb5pb : tensor<64xf32>
    %adwdb5pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb5pb = stablehlo.multiply %adwdb5pb, %adlrb5pb : tensor<64xf32>
    %adwdpb5pb = stablehlo.multiply %adwdlrb5pb, %b5pb : tensor<64xf32>
    %adnewb5pb = stablehlo.subtract %adsubb5pb, %adwdpb5pb : tensor<64xf32>
    %adb1b5pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b5pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb5pg = stablehlo.multiply %adb1b5pg, %b5pgm : tensor<64xf32>
    %admgb5pg = stablehlo.multiply %adob1b5pg, %b5dpg : tensor<64xf32>
    %admnb5pg = stablehlo.add %admsb5pg, %admgb5pg : tensor<64xf32>
    %adb2b5pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b5pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb5pg = stablehlo.multiply %adb2b5pg, %b5pgv : tensor<64xf32>
    %adg2b5pg = stablehlo.multiply %b5dpg, %b5dpg : tensor<64xf32>
    %advgb5pg = stablehlo.multiply %adob2b5pg, %adg2b5pg : tensor<64xf32>
    %advnb5pg = stablehlo.add %advsb5pg, %advgb5pg : tensor<64xf32>
    %adbc1b5pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b5pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb5pg = stablehlo.divide %admnb5pg, %adbc1b5pg : tensor<64xf32>
    %advhb5pg = stablehlo.divide %advnb5pg, %adbc2b5pg : tensor<64xf32>
    %adlrb5pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb5pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb5pg = stablehlo.sqrt %advhb5pg : tensor<64xf32>
    %addenb5pg = stablehlo.add %adsqb5pg, %adepsb5pg : tensor<64xf32>
    %adratb5pg = stablehlo.divide %admhb5pg, %addenb5pg : tensor<64xf32>
    %adstb5pg = stablehlo.multiply %adlrb5pg, %adratb5pg : tensor<64xf32>
    %adsubb5pg = stablehlo.subtract %b5pg, %adstb5pg : tensor<64xf32>
    %adwdb5pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb5pg = stablehlo.multiply %adwdb5pg, %adlrb5pg : tensor<64xf32>
    %adwdpb5pg = stablehlo.multiply %adwdlrb5pg, %b5pg : tensor<64xf32>
    %adnewb5pg = stablehlo.subtract %adsubb5pg, %adwdpb5pg : tensor<64xf32>
    %adb1b5pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b5pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb5pbt = stablehlo.multiply %adb1b5pbt, %b5pbtm : tensor<64xf32>
    %admgb5pbt = stablehlo.multiply %adob1b5pbt, %b5dpbt : tensor<64xf32>
    %admnb5pbt = stablehlo.add %admsb5pbt, %admgb5pbt : tensor<64xf32>
    %adb2b5pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b5pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb5pbt = stablehlo.multiply %adb2b5pbt, %b5pbtv : tensor<64xf32>
    %adg2b5pbt = stablehlo.multiply %b5dpbt, %b5dpbt : tensor<64xf32>
    %advgb5pbt = stablehlo.multiply %adob2b5pbt, %adg2b5pbt : tensor<64xf32>
    %advnb5pbt = stablehlo.add %advsb5pbt, %advgb5pbt : tensor<64xf32>
    %adbc1b5pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b5pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb5pbt = stablehlo.divide %admnb5pbt, %adbc1b5pbt : tensor<64xf32>
    %advhb5pbt = stablehlo.divide %advnb5pbt, %adbc2b5pbt : tensor<64xf32>
    %adlrb5pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb5pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb5pbt = stablehlo.sqrt %advhb5pbt : tensor<64xf32>
    %addenb5pbt = stablehlo.add %adsqb5pbt, %adepsb5pbt : tensor<64xf32>
    %adratb5pbt = stablehlo.divide %admhb5pbt, %addenb5pbt : tensor<64xf32>
    %adstb5pbt = stablehlo.multiply %adlrb5pbt, %adratb5pbt : tensor<64xf32>
    %adsubb5pbt = stablehlo.subtract %b5pbt, %adstb5pbt : tensor<64xf32>
    %adwdb5pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb5pbt = stablehlo.multiply %adwdb5pbt, %adlrb5pbt : tensor<64xf32>
    %adwdpb5pbt = stablehlo.multiply %adwdlrb5pbt, %b5pbt : tensor<64xf32>
    %adnewb5pbt = stablehlo.subtract %adsubb5pbt, %adwdpb5pbt : tensor<64xf32>
    %adb1b6eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %adob1b6eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %admsb6eW = stablehlo.multiply %adb1b6eW, %b6eWm : tensor<256x64x1x1xf32>
    %admgb6eW = stablehlo.multiply %adob1b6eW, %b6deW : tensor<256x64x1x1xf32>
    %admnb6eW = stablehlo.add %admsb6eW, %admgb6eW : tensor<256x64x1x1xf32>
    %adb2b6eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %adob2b6eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %advsb6eW = stablehlo.multiply %adb2b6eW, %b6eWv : tensor<256x64x1x1xf32>
    %adg2b6eW = stablehlo.multiply %b6deW, %b6deW : tensor<256x64x1x1xf32>
    %advgb6eW = stablehlo.multiply %adob2b6eW, %adg2b6eW : tensor<256x64x1x1xf32>
    %advnb6eW = stablehlo.add %advsb6eW, %advgb6eW : tensor<256x64x1x1xf32>
    %adbc1b6eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %adbc2b6eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %admhb6eW = stablehlo.divide %admnb6eW, %adbc1b6eW : tensor<256x64x1x1xf32>
    %advhb6eW = stablehlo.divide %advnb6eW, %adbc2b6eW : tensor<256x64x1x1xf32>
    %adlrb6eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %adepsb6eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %adsqb6eW = stablehlo.sqrt %advhb6eW : tensor<256x64x1x1xf32>
    %addenb6eW = stablehlo.add %adsqb6eW, %adepsb6eW : tensor<256x64x1x1xf32>
    %adratb6eW = stablehlo.divide %admhb6eW, %addenb6eW : tensor<256x64x1x1xf32>
    %adstb6eW = stablehlo.multiply %adlrb6eW, %adratb6eW : tensor<256x64x1x1xf32>
    %adsubb6eW = stablehlo.subtract %b6eW, %adstb6eW : tensor<256x64x1x1xf32>
    %adwdb6eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %adwdlrb6eW = stablehlo.multiply %adwdb6eW, %adlrb6eW : tensor<256x64x1x1xf32>
    %adwdpb6eW = stablehlo.multiply %adwdlrb6eW, %b6eW : tensor<256x64x1x1xf32>
    %adnewb6eW = stablehlo.subtract %adsubb6eW, %adwdpb6eW : tensor<256x64x1x1xf32>
    %adb1b6eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1b6eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsb6eb = stablehlo.multiply %adb1b6eb, %b6ebm : tensor<256xf32>
    %admgb6eb = stablehlo.multiply %adob1b6eb, %b6deb : tensor<256xf32>
    %admnb6eb = stablehlo.add %admsb6eb, %admgb6eb : tensor<256xf32>
    %adb2b6eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2b6eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsb6eb = stablehlo.multiply %adb2b6eb, %b6ebv : tensor<256xf32>
    %adg2b6eb = stablehlo.multiply %b6deb, %b6deb : tensor<256xf32>
    %advgb6eb = stablehlo.multiply %adob2b6eb, %adg2b6eb : tensor<256xf32>
    %advnb6eb = stablehlo.add %advsb6eb, %advgb6eb : tensor<256xf32>
    %adbc1b6eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2b6eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhb6eb = stablehlo.divide %admnb6eb, %adbc1b6eb : tensor<256xf32>
    %advhb6eb = stablehlo.divide %advnb6eb, %adbc2b6eb : tensor<256xf32>
    %adlrb6eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsb6eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqb6eb = stablehlo.sqrt %advhb6eb : tensor<256xf32>
    %addenb6eb = stablehlo.add %adsqb6eb, %adepsb6eb : tensor<256xf32>
    %adratb6eb = stablehlo.divide %admhb6eb, %addenb6eb : tensor<256xf32>
    %adstb6eb = stablehlo.multiply %adlrb6eb, %adratb6eb : tensor<256xf32>
    %adsubb6eb = stablehlo.subtract %b6eb, %adstb6eb : tensor<256xf32>
    %adwdb6eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrb6eb = stablehlo.multiply %adwdb6eb, %adlrb6eb : tensor<256xf32>
    %adwdpb6eb = stablehlo.multiply %adwdlrb6eb, %b6eb : tensor<256xf32>
    %adnewb6eb = stablehlo.subtract %adsubb6eb, %adwdpb6eb : tensor<256xf32>
    %adb1b6eg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1b6eg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsb6eg = stablehlo.multiply %adb1b6eg, %b6egm : tensor<256xf32>
    %admgb6eg = stablehlo.multiply %adob1b6eg, %b6deg : tensor<256xf32>
    %admnb6eg = stablehlo.add %admsb6eg, %admgb6eg : tensor<256xf32>
    %adb2b6eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2b6eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsb6eg = stablehlo.multiply %adb2b6eg, %b6egv : tensor<256xf32>
    %adg2b6eg = stablehlo.multiply %b6deg, %b6deg : tensor<256xf32>
    %advgb6eg = stablehlo.multiply %adob2b6eg, %adg2b6eg : tensor<256xf32>
    %advnb6eg = stablehlo.add %advsb6eg, %advgb6eg : tensor<256xf32>
    %adbc1b6eg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2b6eg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhb6eg = stablehlo.divide %admnb6eg, %adbc1b6eg : tensor<256xf32>
    %advhb6eg = stablehlo.divide %advnb6eg, %adbc2b6eg : tensor<256xf32>
    %adlrb6eg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsb6eg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqb6eg = stablehlo.sqrt %advhb6eg : tensor<256xf32>
    %addenb6eg = stablehlo.add %adsqb6eg, %adepsb6eg : tensor<256xf32>
    %adratb6eg = stablehlo.divide %admhb6eg, %addenb6eg : tensor<256xf32>
    %adstb6eg = stablehlo.multiply %adlrb6eg, %adratb6eg : tensor<256xf32>
    %adsubb6eg = stablehlo.subtract %b6eg, %adstb6eg : tensor<256xf32>
    %adwdb6eg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrb6eg = stablehlo.multiply %adwdb6eg, %adlrb6eg : tensor<256xf32>
    %adwdpb6eg = stablehlo.multiply %adwdlrb6eg, %b6eg : tensor<256xf32>
    %adnewb6eg = stablehlo.subtract %adsubb6eg, %adwdpb6eg : tensor<256xf32>
    %adb1b6ebt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1b6ebt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsb6ebt = stablehlo.multiply %adb1b6ebt, %b6ebtm : tensor<256xf32>
    %admgb6ebt = stablehlo.multiply %adob1b6ebt, %b6debt : tensor<256xf32>
    %admnb6ebt = stablehlo.add %admsb6ebt, %admgb6ebt : tensor<256xf32>
    %adb2b6ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2b6ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsb6ebt = stablehlo.multiply %adb2b6ebt, %b6ebtv : tensor<256xf32>
    %adg2b6ebt = stablehlo.multiply %b6debt, %b6debt : tensor<256xf32>
    %advgb6ebt = stablehlo.multiply %adob2b6ebt, %adg2b6ebt : tensor<256xf32>
    %advnb6ebt = stablehlo.add %advsb6ebt, %advgb6ebt : tensor<256xf32>
    %adbc1b6ebt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2b6ebt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhb6ebt = stablehlo.divide %admnb6ebt, %adbc1b6ebt : tensor<256xf32>
    %advhb6ebt = stablehlo.divide %advnb6ebt, %adbc2b6ebt : tensor<256xf32>
    %adlrb6ebt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsb6ebt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqb6ebt = stablehlo.sqrt %advhb6ebt : tensor<256xf32>
    %addenb6ebt = stablehlo.add %adsqb6ebt, %adepsb6ebt : tensor<256xf32>
    %adratb6ebt = stablehlo.divide %admhb6ebt, %addenb6ebt : tensor<256xf32>
    %adstb6ebt = stablehlo.multiply %adlrb6ebt, %adratb6ebt : tensor<256xf32>
    %adsubb6ebt = stablehlo.subtract %b6ebt, %adstb6ebt : tensor<256xf32>
    %adwdb6ebt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrb6ebt = stablehlo.multiply %adwdb6ebt, %adlrb6ebt : tensor<256xf32>
    %adwdpb6ebt = stablehlo.multiply %adwdlrb6ebt, %b6ebt : tensor<256xf32>
    %adnewb6ebt = stablehlo.subtract %adsubb6ebt, %adwdpb6ebt : tensor<256xf32>
    %adb1b6dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x1x3x3xf32>
    %adob1b6dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x1x3x3xf32>
    %admsb6dW = stablehlo.multiply %adb1b6dW, %b6dWm : tensor<256x1x3x3xf32>
    %admgb6dW = stablehlo.multiply %adob1b6dW, %b6ddW : tensor<256x1x3x3xf32>
    %admnb6dW = stablehlo.add %admsb6dW, %admgb6dW : tensor<256x1x3x3xf32>
    %adb2b6dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x1x3x3xf32>
    %adob2b6dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x1x3x3xf32>
    %advsb6dW = stablehlo.multiply %adb2b6dW, %b6dWv : tensor<256x1x3x3xf32>
    %adg2b6dW = stablehlo.multiply %b6ddW, %b6ddW : tensor<256x1x3x3xf32>
    %advgb6dW = stablehlo.multiply %adob2b6dW, %adg2b6dW : tensor<256x1x3x3xf32>
    %advnb6dW = stablehlo.add %advsb6dW, %advgb6dW : tensor<256x1x3x3xf32>
    %adbc1b6dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x1x3x3xf32>
    %adbc2b6dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x1x3x3xf32>
    %admhb6dW = stablehlo.divide %admnb6dW, %adbc1b6dW : tensor<256x1x3x3xf32>
    %advhb6dW = stablehlo.divide %advnb6dW, %adbc2b6dW : tensor<256x1x3x3xf32>
    %adlrb6dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1x3x3xf32>
    %adepsb6dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x1x3x3xf32>
    %adsqb6dW = stablehlo.sqrt %advhb6dW : tensor<256x1x3x3xf32>
    %addenb6dW = stablehlo.add %adsqb6dW, %adepsb6dW : tensor<256x1x3x3xf32>
    %adratb6dW = stablehlo.divide %admhb6dW, %addenb6dW : tensor<256x1x3x3xf32>
    %adstb6dW = stablehlo.multiply %adlrb6dW, %adratb6dW : tensor<256x1x3x3xf32>
    %adsubb6dW = stablehlo.subtract %b6dW, %adstb6dW : tensor<256x1x3x3xf32>
    %adwdb6dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1x3x3xf32>
    %adwdlrb6dW = stablehlo.multiply %adwdb6dW, %adlrb6dW : tensor<256x1x3x3xf32>
    %adwdpb6dW = stablehlo.multiply %adwdlrb6dW, %b6dW : tensor<256x1x3x3xf32>
    %adnewb6dW = stablehlo.subtract %adsubb6dW, %adwdpb6dW : tensor<256x1x3x3xf32>
    %adb1b6db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1b6db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsb6db = stablehlo.multiply %adb1b6db, %b6dbm : tensor<256xf32>
    %admgb6db = stablehlo.multiply %adob1b6db, %b6ddb : tensor<256xf32>
    %admnb6db = stablehlo.add %admsb6db, %admgb6db : tensor<256xf32>
    %adb2b6db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2b6db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsb6db = stablehlo.multiply %adb2b6db, %b6dbv : tensor<256xf32>
    %adg2b6db = stablehlo.multiply %b6ddb, %b6ddb : tensor<256xf32>
    %advgb6db = stablehlo.multiply %adob2b6db, %adg2b6db : tensor<256xf32>
    %advnb6db = stablehlo.add %advsb6db, %advgb6db : tensor<256xf32>
    %adbc1b6db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2b6db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhb6db = stablehlo.divide %admnb6db, %adbc1b6db : tensor<256xf32>
    %advhb6db = stablehlo.divide %advnb6db, %adbc2b6db : tensor<256xf32>
    %adlrb6db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsb6db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqb6db = stablehlo.sqrt %advhb6db : tensor<256xf32>
    %addenb6db = stablehlo.add %adsqb6db, %adepsb6db : tensor<256xf32>
    %adratb6db = stablehlo.divide %admhb6db, %addenb6db : tensor<256xf32>
    %adstb6db = stablehlo.multiply %adlrb6db, %adratb6db : tensor<256xf32>
    %adsubb6db = stablehlo.subtract %b6db, %adstb6db : tensor<256xf32>
    %adwdb6db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrb6db = stablehlo.multiply %adwdb6db, %adlrb6db : tensor<256xf32>
    %adwdpb6db = stablehlo.multiply %adwdlrb6db, %b6db : tensor<256xf32>
    %adnewb6db = stablehlo.subtract %adsubb6db, %adwdpb6db : tensor<256xf32>
    %adb1b6dg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1b6dg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsb6dg = stablehlo.multiply %adb1b6dg, %b6dgm : tensor<256xf32>
    %admgb6dg = stablehlo.multiply %adob1b6dg, %b6ddg : tensor<256xf32>
    %admnb6dg = stablehlo.add %admsb6dg, %admgb6dg : tensor<256xf32>
    %adb2b6dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2b6dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsb6dg = stablehlo.multiply %adb2b6dg, %b6dgv : tensor<256xf32>
    %adg2b6dg = stablehlo.multiply %b6ddg, %b6ddg : tensor<256xf32>
    %advgb6dg = stablehlo.multiply %adob2b6dg, %adg2b6dg : tensor<256xf32>
    %advnb6dg = stablehlo.add %advsb6dg, %advgb6dg : tensor<256xf32>
    %adbc1b6dg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2b6dg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhb6dg = stablehlo.divide %admnb6dg, %adbc1b6dg : tensor<256xf32>
    %advhb6dg = stablehlo.divide %advnb6dg, %adbc2b6dg : tensor<256xf32>
    %adlrb6dg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsb6dg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqb6dg = stablehlo.sqrt %advhb6dg : tensor<256xf32>
    %addenb6dg = stablehlo.add %adsqb6dg, %adepsb6dg : tensor<256xf32>
    %adratb6dg = stablehlo.divide %admhb6dg, %addenb6dg : tensor<256xf32>
    %adstb6dg = stablehlo.multiply %adlrb6dg, %adratb6dg : tensor<256xf32>
    %adsubb6dg = stablehlo.subtract %b6dg, %adstb6dg : tensor<256xf32>
    %adwdb6dg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrb6dg = stablehlo.multiply %adwdb6dg, %adlrb6dg : tensor<256xf32>
    %adwdpb6dg = stablehlo.multiply %adwdlrb6dg, %b6dg : tensor<256xf32>
    %adnewb6dg = stablehlo.subtract %adsubb6dg, %adwdpb6dg : tensor<256xf32>
    %adb1b6dbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1b6dbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsb6dbt = stablehlo.multiply %adb1b6dbt, %b6dbtm : tensor<256xf32>
    %admgb6dbt = stablehlo.multiply %adob1b6dbt, %b6ddbt : tensor<256xf32>
    %admnb6dbt = stablehlo.add %admsb6dbt, %admgb6dbt : tensor<256xf32>
    %adb2b6dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2b6dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsb6dbt = stablehlo.multiply %adb2b6dbt, %b6dbtv : tensor<256xf32>
    %adg2b6dbt = stablehlo.multiply %b6ddbt, %b6ddbt : tensor<256xf32>
    %advgb6dbt = stablehlo.multiply %adob2b6dbt, %adg2b6dbt : tensor<256xf32>
    %advnb6dbt = stablehlo.add %advsb6dbt, %advgb6dbt : tensor<256xf32>
    %adbc1b6dbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2b6dbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhb6dbt = stablehlo.divide %admnb6dbt, %adbc1b6dbt : tensor<256xf32>
    %advhb6dbt = stablehlo.divide %advnb6dbt, %adbc2b6dbt : tensor<256xf32>
    %adlrb6dbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsb6dbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqb6dbt = stablehlo.sqrt %advhb6dbt : tensor<256xf32>
    %addenb6dbt = stablehlo.add %adsqb6dbt, %adepsb6dbt : tensor<256xf32>
    %adratb6dbt = stablehlo.divide %admhb6dbt, %addenb6dbt : tensor<256xf32>
    %adstb6dbt = stablehlo.multiply %adlrb6dbt, %adratb6dbt : tensor<256xf32>
    %adsubb6dbt = stablehlo.subtract %b6dbt, %adstb6dbt : tensor<256xf32>
    %adwdb6dbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrb6dbt = stablehlo.multiply %adwdb6dbt, %adlrb6dbt : tensor<256xf32>
    %adwdpb6dbt = stablehlo.multiply %adwdlrb6dbt, %b6dbt : tensor<256xf32>
    %adnewb6dbt = stablehlo.subtract %adsubb6dbt, %adwdpb6dbt : tensor<256xf32>
    %adb1b6pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %adob1b6pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %admsb6pW = stablehlo.multiply %adb1b6pW, %b6pWm : tensor<64x256x1x1xf32>
    %admgb6pW = stablehlo.multiply %adob1b6pW, %b6dpW : tensor<64x256x1x1xf32>
    %admnb6pW = stablehlo.add %admsb6pW, %admgb6pW : tensor<64x256x1x1xf32>
    %adb2b6pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %adob2b6pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %advsb6pW = stablehlo.multiply %adb2b6pW, %b6pWv : tensor<64x256x1x1xf32>
    %adg2b6pW = stablehlo.multiply %b6dpW, %b6dpW : tensor<64x256x1x1xf32>
    %advgb6pW = stablehlo.multiply %adob2b6pW, %adg2b6pW : tensor<64x256x1x1xf32>
    %advnb6pW = stablehlo.add %advsb6pW, %advgb6pW : tensor<64x256x1x1xf32>
    %adbc1b6pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %adbc2b6pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %admhb6pW = stablehlo.divide %admnb6pW, %adbc1b6pW : tensor<64x256x1x1xf32>
    %advhb6pW = stablehlo.divide %advnb6pW, %adbc2b6pW : tensor<64x256x1x1xf32>
    %adlrb6pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %adepsb6pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %adsqb6pW = stablehlo.sqrt %advhb6pW : tensor<64x256x1x1xf32>
    %addenb6pW = stablehlo.add %adsqb6pW, %adepsb6pW : tensor<64x256x1x1xf32>
    %adratb6pW = stablehlo.divide %admhb6pW, %addenb6pW : tensor<64x256x1x1xf32>
    %adstb6pW = stablehlo.multiply %adlrb6pW, %adratb6pW : tensor<64x256x1x1xf32>
    %adsubb6pW = stablehlo.subtract %b6pW, %adstb6pW : tensor<64x256x1x1xf32>
    %adwdb6pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %adwdlrb6pW = stablehlo.multiply %adwdb6pW, %adlrb6pW : tensor<64x256x1x1xf32>
    %adwdpb6pW = stablehlo.multiply %adwdlrb6pW, %b6pW : tensor<64x256x1x1xf32>
    %adnewb6pW = stablehlo.subtract %adsubb6pW, %adwdpb6pW : tensor<64x256x1x1xf32>
    %adb1b6pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b6pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb6pb = stablehlo.multiply %adb1b6pb, %b6pbm : tensor<64xf32>
    %admgb6pb = stablehlo.multiply %adob1b6pb, %b6dpb : tensor<64xf32>
    %admnb6pb = stablehlo.add %admsb6pb, %admgb6pb : tensor<64xf32>
    %adb2b6pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b6pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb6pb = stablehlo.multiply %adb2b6pb, %b6pbv : tensor<64xf32>
    %adg2b6pb = stablehlo.multiply %b6dpb, %b6dpb : tensor<64xf32>
    %advgb6pb = stablehlo.multiply %adob2b6pb, %adg2b6pb : tensor<64xf32>
    %advnb6pb = stablehlo.add %advsb6pb, %advgb6pb : tensor<64xf32>
    %adbc1b6pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b6pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb6pb = stablehlo.divide %admnb6pb, %adbc1b6pb : tensor<64xf32>
    %advhb6pb = stablehlo.divide %advnb6pb, %adbc2b6pb : tensor<64xf32>
    %adlrb6pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb6pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb6pb = stablehlo.sqrt %advhb6pb : tensor<64xf32>
    %addenb6pb = stablehlo.add %adsqb6pb, %adepsb6pb : tensor<64xf32>
    %adratb6pb = stablehlo.divide %admhb6pb, %addenb6pb : tensor<64xf32>
    %adstb6pb = stablehlo.multiply %adlrb6pb, %adratb6pb : tensor<64xf32>
    %adsubb6pb = stablehlo.subtract %b6pb, %adstb6pb : tensor<64xf32>
    %adwdb6pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb6pb = stablehlo.multiply %adwdb6pb, %adlrb6pb : tensor<64xf32>
    %adwdpb6pb = stablehlo.multiply %adwdlrb6pb, %b6pb : tensor<64xf32>
    %adnewb6pb = stablehlo.subtract %adsubb6pb, %adwdpb6pb : tensor<64xf32>
    %adb1b6pg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b6pg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb6pg = stablehlo.multiply %adb1b6pg, %b6pgm : tensor<64xf32>
    %admgb6pg = stablehlo.multiply %adob1b6pg, %b6dpg : tensor<64xf32>
    %admnb6pg = stablehlo.add %admsb6pg, %admgb6pg : tensor<64xf32>
    %adb2b6pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b6pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb6pg = stablehlo.multiply %adb2b6pg, %b6pgv : tensor<64xf32>
    %adg2b6pg = stablehlo.multiply %b6dpg, %b6dpg : tensor<64xf32>
    %advgb6pg = stablehlo.multiply %adob2b6pg, %adg2b6pg : tensor<64xf32>
    %advnb6pg = stablehlo.add %advsb6pg, %advgb6pg : tensor<64xf32>
    %adbc1b6pg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b6pg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb6pg = stablehlo.divide %admnb6pg, %adbc1b6pg : tensor<64xf32>
    %advhb6pg = stablehlo.divide %advnb6pg, %adbc2b6pg : tensor<64xf32>
    %adlrb6pg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb6pg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb6pg = stablehlo.sqrt %advhb6pg : tensor<64xf32>
    %addenb6pg = stablehlo.add %adsqb6pg, %adepsb6pg : tensor<64xf32>
    %adratb6pg = stablehlo.divide %admhb6pg, %addenb6pg : tensor<64xf32>
    %adstb6pg = stablehlo.multiply %adlrb6pg, %adratb6pg : tensor<64xf32>
    %adsubb6pg = stablehlo.subtract %b6pg, %adstb6pg : tensor<64xf32>
    %adwdb6pg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb6pg = stablehlo.multiply %adwdb6pg, %adlrb6pg : tensor<64xf32>
    %adwdpb6pg = stablehlo.multiply %adwdlrb6pg, %b6pg : tensor<64xf32>
    %adnewb6pg = stablehlo.subtract %adsubb6pg, %adwdpb6pg : tensor<64xf32>
    %adb1b6pbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b6pbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb6pbt = stablehlo.multiply %adb1b6pbt, %b6pbtm : tensor<64xf32>
    %admgb6pbt = stablehlo.multiply %adob1b6pbt, %b6dpbt : tensor<64xf32>
    %admnb6pbt = stablehlo.add %admsb6pbt, %admgb6pbt : tensor<64xf32>
    %adb2b6pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b6pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb6pbt = stablehlo.multiply %adb2b6pbt, %b6pbtv : tensor<64xf32>
    %adg2b6pbt = stablehlo.multiply %b6dpbt, %b6dpbt : tensor<64xf32>
    %advgb6pbt = stablehlo.multiply %adob2b6pbt, %adg2b6pbt : tensor<64xf32>
    %advnb6pbt = stablehlo.add %advsb6pbt, %advgb6pbt : tensor<64xf32>
    %adbc1b6pbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b6pbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb6pbt = stablehlo.divide %admnb6pbt, %adbc1b6pbt : tensor<64xf32>
    %advhb6pbt = stablehlo.divide %advnb6pbt, %adbc2b6pbt : tensor<64xf32>
    %adlrb6pbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb6pbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb6pbt = stablehlo.sqrt %advhb6pbt : tensor<64xf32>
    %addenb6pbt = stablehlo.add %adsqb6pbt, %adepsb6pbt : tensor<64xf32>
    %adratb6pbt = stablehlo.divide %admhb6pbt, %addenb6pbt : tensor<64xf32>
    %adstb6pbt = stablehlo.multiply %adlrb6pbt, %adratb6pbt : tensor<64xf32>
    %adsubb6pbt = stablehlo.subtract %b6pbt, %adstb6pbt : tensor<64xf32>
    %adwdb6pbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb6pbt = stablehlo.multiply %adwdb6pbt, %adlrb6pbt : tensor<64xf32>
    %adwdpb6pbt = stablehlo.multiply %adwdlrb6pbt, %b6pbt : tensor<64xf32>
    %adnewb6pbt = stablehlo.subtract %adsubb6pbt, %adwdpb6pbt : tensor<64xf32>
    %adb1hW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %adob1hW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %admshW = stablehlo.multiply %adb1hW, %hWm : tensor<128x64x1x1xf32>
    %admghW = stablehlo.multiply %adob1hW, %dhW : tensor<128x64x1x1xf32>
    %admnhW = stablehlo.add %admshW, %admghW : tensor<128x64x1x1xf32>
    %adb2hW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %adob2hW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %advshW = stablehlo.multiply %adb2hW, %hWv : tensor<128x64x1x1xf32>
    %adg2hW = stablehlo.multiply %dhW, %dhW : tensor<128x64x1x1xf32>
    %advghW = stablehlo.multiply %adob2hW, %adg2hW : tensor<128x64x1x1xf32>
    %advnhW = stablehlo.add %advshW, %advghW : tensor<128x64x1x1xf32>
    %adbc1hW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %adbc2hW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %admhhW = stablehlo.divide %admnhW, %adbc1hW : tensor<128x64x1x1xf32>
    %advhhW = stablehlo.divide %advnhW, %adbc2hW : tensor<128x64x1x1xf32>
    %adlrhW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %adepshW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %adsqhW = stablehlo.sqrt %advhhW : tensor<128x64x1x1xf32>
    %addenhW = stablehlo.add %adsqhW, %adepshW : tensor<128x64x1x1xf32>
    %adrathW = stablehlo.divide %admhhW, %addenhW : tensor<128x64x1x1xf32>
    %adsthW = stablehlo.multiply %adlrhW, %adrathW : tensor<128x64x1x1xf32>
    %adsubhW = stablehlo.subtract %hW, %adsthW : tensor<128x64x1x1xf32>
    %adwdhW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %adwdlrhW = stablehlo.multiply %adwdhW, %adlrhW : tensor<128x64x1x1xf32>
    %adwdphW = stablehlo.multiply %adwdlrhW, %hW : tensor<128x64x1x1xf32>
    %adnewhW = stablehlo.subtract %adsubhW, %adwdphW : tensor<128x64x1x1xf32>
    %adb1hb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1hb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admshb = stablehlo.multiply %adb1hb, %hbm : tensor<128xf32>
    %admghb = stablehlo.multiply %adob1hb, %dhb : tensor<128xf32>
    %admnhb = stablehlo.add %admshb, %admghb : tensor<128xf32>
    %adb2hb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2hb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advshb = stablehlo.multiply %adb2hb, %hbv : tensor<128xf32>
    %adg2hb = stablehlo.multiply %dhb, %dhb : tensor<128xf32>
    %advghb = stablehlo.multiply %adob2hb, %adg2hb : tensor<128xf32>
    %advnhb = stablehlo.add %advshb, %advghb : tensor<128xf32>
    %adbc1hb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2hb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhhb = stablehlo.divide %admnhb, %adbc1hb : tensor<128xf32>
    %advhhb = stablehlo.divide %advnhb, %adbc2hb : tensor<128xf32>
    %adlrhb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepshb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqhb = stablehlo.sqrt %advhhb : tensor<128xf32>
    %addenhb = stablehlo.add %adsqhb, %adepshb : tensor<128xf32>
    %adrathb = stablehlo.divide %admhhb, %addenhb : tensor<128xf32>
    %adsthb = stablehlo.multiply %adlrhb, %adrathb : tensor<128xf32>
    %adsubhb = stablehlo.subtract %hb, %adsthb : tensor<128xf32>
    %adwdhb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrhb = stablehlo.multiply %adwdhb, %adlrhb : tensor<128xf32>
    %adwdphb = stablehlo.multiply %adwdlrhb, %hb : tensor<128xf32>
    %adnewhb = stablehlo.subtract %adsubhb, %adwdphb : tensor<128xf32>
    %adb1hg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1hg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admshg = stablehlo.multiply %adb1hg, %hgm : tensor<128xf32>
    %admghg = stablehlo.multiply %adob1hg, %dhg : tensor<128xf32>
    %admnhg = stablehlo.add %admshg, %admghg : tensor<128xf32>
    %adb2hg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2hg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advshg = stablehlo.multiply %adb2hg, %hgv : tensor<128xf32>
    %adg2hg = stablehlo.multiply %dhg, %dhg : tensor<128xf32>
    %advghg = stablehlo.multiply %adob2hg, %adg2hg : tensor<128xf32>
    %advnhg = stablehlo.add %advshg, %advghg : tensor<128xf32>
    %adbc1hg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2hg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhhg = stablehlo.divide %admnhg, %adbc1hg : tensor<128xf32>
    %advhhg = stablehlo.divide %advnhg, %adbc2hg : tensor<128xf32>
    %adlrhg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepshg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqhg = stablehlo.sqrt %advhhg : tensor<128xf32>
    %addenhg = stablehlo.add %adsqhg, %adepshg : tensor<128xf32>
    %adrathg = stablehlo.divide %admhhg, %addenhg : tensor<128xf32>
    %adsthg = stablehlo.multiply %adlrhg, %adrathg : tensor<128xf32>
    %adsubhg = stablehlo.subtract %hg, %adsthg : tensor<128xf32>
    %adwdhg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrhg = stablehlo.multiply %adwdhg, %adlrhg : tensor<128xf32>
    %adwdphg = stablehlo.multiply %adwdlrhg, %hg : tensor<128xf32>
    %adnewhg = stablehlo.subtract %adsubhg, %adwdphg : tensor<128xf32>
    %adb1hbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1hbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admshbt = stablehlo.multiply %adb1hbt, %hbtm : tensor<128xf32>
    %admghbt = stablehlo.multiply %adob1hbt, %dhbt : tensor<128xf32>
    %admnhbt = stablehlo.add %admshbt, %admghbt : tensor<128xf32>
    %adb2hbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2hbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advshbt = stablehlo.multiply %adb2hbt, %hbtv : tensor<128xf32>
    %adg2hbt = stablehlo.multiply %dhbt, %dhbt : tensor<128xf32>
    %advghbt = stablehlo.multiply %adob2hbt, %adg2hbt : tensor<128xf32>
    %advnhbt = stablehlo.add %advshbt, %advghbt : tensor<128xf32>
    %adbc1hbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2hbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhhbt = stablehlo.divide %admnhbt, %adbc1hbt : tensor<128xf32>
    %advhhbt = stablehlo.divide %advnhbt, %adbc2hbt : tensor<128xf32>
    %adlrhbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepshbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqhbt = stablehlo.sqrt %advhhbt : tensor<128xf32>
    %addenhbt = stablehlo.add %adsqhbt, %adepshbt : tensor<128xf32>
    %adrathbt = stablehlo.divide %admhhbt, %addenhbt : tensor<128xf32>
    %adsthbt = stablehlo.multiply %adlrhbt, %adrathbt : tensor<128xf32>
    %adsubhbt = stablehlo.subtract %hbt, %adsthbt : tensor<128xf32>
    %adwdhbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrhbt = stablehlo.multiply %adwdhbt, %adlrhbt : tensor<128xf32>
    %adwdphbt = stablehlo.multiply %adwdlrhbt, %hbt : tensor<128xf32>
    %adnewhbt = stablehlo.subtract %adsubhbt, %adwdphbt : tensor<128xf32>
    %adb1Wd = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
    %adob1Wd = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
    %admsWd = stablehlo.multiply %adb1Wd, %Wdm : tensor<128x10xf32>
    %admgWd = stablehlo.multiply %adob1Wd, %dWd : tensor<128x10xf32>
    %admnWd = stablehlo.add %admsWd, %admgWd : tensor<128x10xf32>
    %adb2Wd = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
    %adob2Wd = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
    %advsWd = stablehlo.multiply %adb2Wd, %Wdv : tensor<128x10xf32>
    %adg2Wd = stablehlo.multiply %dWd, %dWd : tensor<128x10xf32>
    %advgWd = stablehlo.multiply %adob2Wd, %adg2Wd : tensor<128x10xf32>
    %advnWd = stablehlo.add %advsWd, %advgWd : tensor<128x10xf32>
    %adbc1Wd = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
    %adbc2Wd = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
    %admhWd = stablehlo.divide %admnWd, %adbc1Wd : tensor<128x10xf32>
    %advhWd = stablehlo.divide %advnWd, %adbc2Wd : tensor<128x10xf32>
    %adlrWd = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
    %adepsWd = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
    %adsqWd = stablehlo.sqrt %advhWd : tensor<128x10xf32>
    %addenWd = stablehlo.add %adsqWd, %adepsWd : tensor<128x10xf32>
    %adratWd = stablehlo.divide %admhWd, %addenWd : tensor<128x10xf32>
    %adstWd = stablehlo.multiply %adlrWd, %adratWd : tensor<128x10xf32>
    %adsubWd = stablehlo.subtract %Wd, %adstWd : tensor<128x10xf32>
    %adwdWd = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
    %adwdlrWd = stablehlo.multiply %adwdWd, %adlrWd : tensor<128x10xf32>
    %adwdpWd = stablehlo.multiply %adwdlrWd, %Wd : tensor<128x10xf32>
    %adnewWd = stablehlo.subtract %adsubWd, %adwdpWd : tensor<128x10xf32>
    %adb1bd = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob1bd = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admsbd = stablehlo.multiply %adb1bd, %bdm : tensor<10xf32>
    %admgbd = stablehlo.multiply %adob1bd, %dbd : tensor<10xf32>
    %admnbd = stablehlo.add %admsbd, %admgbd : tensor<10xf32>
    %adb2bd = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob2bd = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %advsbd = stablehlo.multiply %adb2bd, %bdv : tensor<10xf32>
    %adg2bd = stablehlo.multiply %dbd, %dbd : tensor<10xf32>
    %advgbd = stablehlo.multiply %adob2bd, %adg2bd : tensor<10xf32>
    %advnbd = stablehlo.add %advsbd, %advgbd : tensor<10xf32>
    %adbc1bd = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adbc2bd = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admhbd = stablehlo.divide %admnbd, %adbc1bd : tensor<10xf32>
    %advhbd = stablehlo.divide %advnbd, %adbc2bd : tensor<10xf32>
    %adlrbd = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adepsbd = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adsqbd = stablehlo.sqrt %advhbd : tensor<10xf32>
    %addenbd = stablehlo.add %adsqbd, %adepsbd : tensor<10xf32>
    %adratbd = stablehlo.divide %admhbd, %addenbd : tensor<10xf32>
    %adstbd = stablehlo.multiply %adlrbd, %adratbd : tensor<10xf32>
    %adsubbd = stablehlo.subtract %bd, %adstbd : tensor<10xf32>
    %adwdbd = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adwdlrbd = stablehlo.multiply %adwdbd, %adlrbd : tensor<10xf32>
    %adwdpbd = stablehlo.multiply %adwdlrbd, %bd : tensor<10xf32>
    %adnewbd = stablehlo.subtract %adsubbd, %adwdpbd : tensor<10xf32>
    return %adnewsW, %adnewsb, %adnewsg, %adnewsbt, %adnewb1eW, %adnewb1eb, %adnewb1eg, %adnewb1ebt, %adnewb1dW, %adnewb1db, %adnewb1dg, %adnewb1dbt, %adnewb1pW, %adnewb1pb, %adnewb1pg, %adnewb1pbt, %adnewb2eW, %adnewb2eb, %adnewb2eg, %adnewb2ebt, %adnewb2dW, %adnewb2db, %adnewb2dg, %adnewb2dbt, %adnewb2pW, %adnewb2pb, %adnewb2pg, %adnewb2pbt, %adnewb3eW, %adnewb3eb, %adnewb3eg, %adnewb3ebt, %adnewb3dW, %adnewb3db, %adnewb3dg, %adnewb3dbt, %adnewb3pW, %adnewb3pb, %adnewb3pg, %adnewb3pbt, %adnewb4eW, %adnewb4eb, %adnewb4eg, %adnewb4ebt, %adnewb4dW, %adnewb4db, %adnewb4dg, %adnewb4dbt, %adnewb4pW, %adnewb4pb, %adnewb4pg, %adnewb4pbt, %adnewb5eW, %adnewb5eb, %adnewb5eg, %adnewb5ebt, %adnewb5dW, %adnewb5db, %adnewb5dg, %adnewb5dbt, %adnewb5pW, %adnewb5pb, %adnewb5pg, %adnewb5pbt, %adnewb6eW, %adnewb6eb, %adnewb6eg, %adnewb6ebt, %adnewb6dW, %adnewb6db, %adnewb6dg, %adnewb6dbt, %adnewb6pW, %adnewb6pb, %adnewb6pg, %adnewb6pbt, %adnewhW, %adnewhb, %adnewhg, %adnewhbt, %adnewWd, %adnewbd, %admnsW, %admnsb, %admnsg, %admnsbt, %admnb1eW, %admnb1eb, %admnb1eg, %admnb1ebt, %admnb1dW, %admnb1db, %admnb1dg, %admnb1dbt, %admnb1pW, %admnb1pb, %admnb1pg, %admnb1pbt, %admnb2eW, %admnb2eb, %admnb2eg, %admnb2ebt, %admnb2dW, %admnb2db, %admnb2dg, %admnb2dbt, %admnb2pW, %admnb2pb, %admnb2pg, %admnb2pbt, %admnb3eW, %admnb3eb, %admnb3eg, %admnb3ebt, %admnb3dW, %admnb3db, %admnb3dg, %admnb3dbt, %admnb3pW, %admnb3pb, %admnb3pg, %admnb3pbt, %admnb4eW, %admnb4eb, %admnb4eg, %admnb4ebt, %admnb4dW, %admnb4db, %admnb4dg, %admnb4dbt, %admnb4pW, %admnb4pb, %admnb4pg, %admnb4pbt, %admnb5eW, %admnb5eb, %admnb5eg, %admnb5ebt, %admnb5dW, %admnb5db, %admnb5dg, %admnb5dbt, %admnb5pW, %admnb5pb, %admnb5pg, %admnb5pbt, %admnb6eW, %admnb6eb, %admnb6eg, %admnb6ebt, %admnb6dW, %admnb6db, %admnb6dg, %admnb6dbt, %admnb6pW, %admnb6pb, %admnb6pg, %admnb6pbt, %admnhW, %admnhb, %admnhg, %admnhbt, %admnWd, %admnbd, %advnsW, %advnsb, %advnsg, %advnsbt, %advnb1eW, %advnb1eb, %advnb1eg, %advnb1ebt, %advnb1dW, %advnb1db, %advnb1dg, %advnb1dbt, %advnb1pW, %advnb1pb, %advnb1pg, %advnb1pbt, %advnb2eW, %advnb2eb, %advnb2eg, %advnb2ebt, %advnb2dW, %advnb2db, %advnb2dg, %advnb2dbt, %advnb2pW, %advnb2pb, %advnb2pg, %advnb2pbt, %advnb3eW, %advnb3eb, %advnb3eg, %advnb3ebt, %advnb3dW, %advnb3db, %advnb3dg, %advnb3dbt, %advnb3pW, %advnb3pb, %advnb3pg, %advnb3pbt, %advnb4eW, %advnb4eb, %advnb4eg, %advnb4ebt, %advnb4dW, %advnb4db, %advnb4dg, %advnb4dbt, %advnb4pW, %advnb4pb, %advnb4pg, %advnb4pbt, %advnb5eW, %advnb5eb, %advnb5eg, %advnb5ebt, %advnb5dW, %advnb5db, %advnb5dg, %advnb5dbt, %advnb5pW, %advnb5pb, %advnb5pg, %advnb5pbt, %advnb6eW, %advnb6eb, %advnb6eg, %advnb6ebt, %advnb6dW, %advnb6db, %advnb6dg, %advnb6dbt, %advnb6pW, %advnb6pb, %advnb6pg, %advnb6pbt, %advnhW, %advnhb, %advnhg, %advnhbt, %advnWd, %advnbd, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
