module @m {
  func.func @mobilenetv2_adam_train_step(%x: tensor<32x150528xf32>, %sW: tensor<16x3x3x3xf32>, %sb: tensor<16xf32>, %sg: tensor<16xf32>, %sbt: tensor<16xf32>, %b1eW: tensor<64x16x1x1xf32>, %b1eb: tensor<64xf32>, %b1eg: tensor<64xf32>, %b1ebt: tensor<64xf32>, %b1dW: tensor<64x1x3x3xf32>, %b1db: tensor<64xf32>, %b1dg: tensor<64xf32>, %b1dbt: tensor<64xf32>, %b1pW: tensor<24x64x1x1xf32>, %b1pb: tensor<24xf32>, %b1pg: tensor<24xf32>, %b1pbt: tensor<24xf32>, %b2eW: tensor<96x24x1x1xf32>, %b2eb: tensor<96xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2db: tensor<96xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pb: tensor<24xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<96x24x1x1xf32>, %b3eb: tensor<96xf32>, %b3eg: tensor<96xf32>, %b3ebt: tensor<96xf32>, %b3dW: tensor<96x1x3x3xf32>, %b3db: tensor<96xf32>, %b3dg: tensor<96xf32>, %b3dbt: tensor<96xf32>, %b3pW: tensor<32x96x1x1xf32>, %b3pb: tensor<32xf32>, %b3pg: tensor<32xf32>, %b3pbt: tensor<32xf32>, %b4eW: tensor<128x32x1x1xf32>, %b4eb: tensor<128xf32>, %b4eg: tensor<128xf32>, %b4ebt: tensor<128xf32>, %b4dW: tensor<128x1x3x3xf32>, %b4db: tensor<128xf32>, %b4dg: tensor<128xf32>, %b4dbt: tensor<128xf32>, %b4pW: tensor<32x128x1x1xf32>, %b4pb: tensor<32xf32>, %b4pg: tensor<32xf32>, %b4pbt: tensor<32xf32>, %b5eW: tensor<128x32x1x1xf32>, %b5eb: tensor<128xf32>, %b5eg: tensor<128xf32>, %b5ebt: tensor<128xf32>, %b5dW: tensor<128x1x3x3xf32>, %b5db: tensor<128xf32>, %b5dg: tensor<128xf32>, %b5dbt: tensor<128xf32>, %b5pW: tensor<64x128x1x1xf32>, %b5pb: tensor<64xf32>, %b5pg: tensor<64xf32>, %b5pbt: tensor<64xf32>, %b6eW: tensor<256x64x1x1xf32>, %b6eb: tensor<256xf32>, %b6eg: tensor<256xf32>, %b6ebt: tensor<256xf32>, %b6dW: tensor<256x1x3x3xf32>, %b6db: tensor<256xf32>, %b6dg: tensor<256xf32>, %b6dbt: tensor<256xf32>, %b6pW: tensor<64x256x1x1xf32>, %b6pb: tensor<64xf32>, %b6pg: tensor<64xf32>, %b6pbt: tensor<64xf32>, %hW: tensor<128x64x1x1xf32>, %hb: tensor<128xf32>, %hg: tensor<128xf32>, %hbt: tensor<128xf32>, %Wd: tensor<128x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<16x3x3x3xf32>, %sbm: tensor<16xf32>, %sgm: tensor<16xf32>, %sbtm: tensor<16xf32>, %b1eWm: tensor<64x16x1x1xf32>, %b1ebm: tensor<64xf32>, %b1egm: tensor<64xf32>, %b1ebtm: tensor<64xf32>, %b1dWm: tensor<64x1x3x3xf32>, %b1dbm: tensor<64xf32>, %b1dgm: tensor<64xf32>, %b1dbtm: tensor<64xf32>, %b1pWm: tensor<24x64x1x1xf32>, %b1pbm: tensor<24xf32>, %b1pgm: tensor<24xf32>, %b1pbtm: tensor<24xf32>, %b2eWm: tensor<96x24x1x1xf32>, %b2ebm: tensor<96xf32>, %b2egm: tensor<96xf32>, %b2ebtm: tensor<96xf32>, %b2dWm: tensor<96x1x3x3xf32>, %b2dbm: tensor<96xf32>, %b2dgm: tensor<96xf32>, %b2dbtm: tensor<96xf32>, %b2pWm: tensor<24x96x1x1xf32>, %b2pbm: tensor<24xf32>, %b2pgm: tensor<24xf32>, %b2pbtm: tensor<24xf32>, %b3eWm: tensor<96x24x1x1xf32>, %b3ebm: tensor<96xf32>, %b3egm: tensor<96xf32>, %b3ebtm: tensor<96xf32>, %b3dWm: tensor<96x1x3x3xf32>, %b3dbm: tensor<96xf32>, %b3dgm: tensor<96xf32>, %b3dbtm: tensor<96xf32>, %b3pWm: tensor<32x96x1x1xf32>, %b3pbm: tensor<32xf32>, %b3pgm: tensor<32xf32>, %b3pbtm: tensor<32xf32>, %b4eWm: tensor<128x32x1x1xf32>, %b4ebm: tensor<128xf32>, %b4egm: tensor<128xf32>, %b4ebtm: tensor<128xf32>, %b4dWm: tensor<128x1x3x3xf32>, %b4dbm: tensor<128xf32>, %b4dgm: tensor<128xf32>, %b4dbtm: tensor<128xf32>, %b4pWm: tensor<32x128x1x1xf32>, %b4pbm: tensor<32xf32>, %b4pgm: tensor<32xf32>, %b4pbtm: tensor<32xf32>, %b5eWm: tensor<128x32x1x1xf32>, %b5ebm: tensor<128xf32>, %b5egm: tensor<128xf32>, %b5ebtm: tensor<128xf32>, %b5dWm: tensor<128x1x3x3xf32>, %b5dbm: tensor<128xf32>, %b5dgm: tensor<128xf32>, %b5dbtm: tensor<128xf32>, %b5pWm: tensor<64x128x1x1xf32>, %b5pbm: tensor<64xf32>, %b5pgm: tensor<64xf32>, %b5pbtm: tensor<64xf32>, %b6eWm: tensor<256x64x1x1xf32>, %b6ebm: tensor<256xf32>, %b6egm: tensor<256xf32>, %b6ebtm: tensor<256xf32>, %b6dWm: tensor<256x1x3x3xf32>, %b6dbm: tensor<256xf32>, %b6dgm: tensor<256xf32>, %b6dbtm: tensor<256xf32>, %b6pWm: tensor<64x256x1x1xf32>, %b6pbm: tensor<64xf32>, %b6pgm: tensor<64xf32>, %b6pbtm: tensor<64xf32>, %hWm: tensor<128x64x1x1xf32>, %hbm: tensor<128xf32>, %hgm: tensor<128xf32>, %hbtm: tensor<128xf32>, %Wdm: tensor<128x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<16x3x3x3xf32>, %sbv: tensor<16xf32>, %sgv: tensor<16xf32>, %sbtv: tensor<16xf32>, %b1eWv: tensor<64x16x1x1xf32>, %b1ebv: tensor<64xf32>, %b1egv: tensor<64xf32>, %b1ebtv: tensor<64xf32>, %b1dWv: tensor<64x1x3x3xf32>, %b1dbv: tensor<64xf32>, %b1dgv: tensor<64xf32>, %b1dbtv: tensor<64xf32>, %b1pWv: tensor<24x64x1x1xf32>, %b1pbv: tensor<24xf32>, %b1pgv: tensor<24xf32>, %b1pbtv: tensor<24xf32>, %b2eWv: tensor<96x24x1x1xf32>, %b2ebv: tensor<96xf32>, %b2egv: tensor<96xf32>, %b2ebtv: tensor<96xf32>, %b2dWv: tensor<96x1x3x3xf32>, %b2dbv: tensor<96xf32>, %b2dgv: tensor<96xf32>, %b2dbtv: tensor<96xf32>, %b2pWv: tensor<24x96x1x1xf32>, %b2pbv: tensor<24xf32>, %b2pgv: tensor<24xf32>, %b2pbtv: tensor<24xf32>, %b3eWv: tensor<96x24x1x1xf32>, %b3ebv: tensor<96xf32>, %b3egv: tensor<96xf32>, %b3ebtv: tensor<96xf32>, %b3dWv: tensor<96x1x3x3xf32>, %b3dbv: tensor<96xf32>, %b3dgv: tensor<96xf32>, %b3dbtv: tensor<96xf32>, %b3pWv: tensor<32x96x1x1xf32>, %b3pbv: tensor<32xf32>, %b3pgv: tensor<32xf32>, %b3pbtv: tensor<32xf32>, %b4eWv: tensor<128x32x1x1xf32>, %b4ebv: tensor<128xf32>, %b4egv: tensor<128xf32>, %b4ebtv: tensor<128xf32>, %b4dWv: tensor<128x1x3x3xf32>, %b4dbv: tensor<128xf32>, %b4dgv: tensor<128xf32>, %b4dbtv: tensor<128xf32>, %b4pWv: tensor<32x128x1x1xf32>, %b4pbv: tensor<32xf32>, %b4pgv: tensor<32xf32>, %b4pbtv: tensor<32xf32>, %b5eWv: tensor<128x32x1x1xf32>, %b5ebv: tensor<128xf32>, %b5egv: tensor<128xf32>, %b5ebtv: tensor<128xf32>, %b5dWv: tensor<128x1x3x3xf32>, %b5dbv: tensor<128xf32>, %b5dgv: tensor<128xf32>, %b5dbtv: tensor<128xf32>, %b5pWv: tensor<64x128x1x1xf32>, %b5pbv: tensor<64xf32>, %b5pgv: tensor<64xf32>, %b5pbtv: tensor<64xf32>, %b6eWv: tensor<256x64x1x1xf32>, %b6ebv: tensor<256xf32>, %b6egv: tensor<256xf32>, %b6ebtv: tensor<256xf32>, %b6dWv: tensor<256x1x3x3xf32>, %b6dbv: tensor<256xf32>, %b6dgv: tensor<256xf32>, %b6dbtv: tensor<256xf32>, %b6pWv: tensor<64x256x1x1xf32>, %b6pbv: tensor<64xf32>, %b6pgv: tensor<64xf32>, %b6pbtv: tensor<64xf32>, %hWv: tensor<128x64x1x1xf32>, %hbv: tensor<128xf32>, %hgv: tensor<128xf32>, %hbtv: tensor<128xf32>, %Wdv: tensor<128x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<16xf32>, %stnvari: tensor<16xf32>, %b1enmui: tensor<64xf32>, %b1envari: tensor<64xf32>, %b1dnmui: tensor<64xf32>, %b1dnvari: tensor<64xf32>, %b1pnmui: tensor<24xf32>, %b1pnvari: tensor<24xf32>, %b2enmui: tensor<96xf32>, %b2envari: tensor<96xf32>, %b2dnmui: tensor<96xf32>, %b2dnvari: tensor<96xf32>, %b2pnmui: tensor<24xf32>, %b2pnvari: tensor<24xf32>, %b3enmui: tensor<96xf32>, %b3envari: tensor<96xf32>, %b3dnmui: tensor<96xf32>, %b3dnvari: tensor<96xf32>, %b3pnmui: tensor<32xf32>, %b3pnvari: tensor<32xf32>, %b4enmui: tensor<128xf32>, %b4envari: tensor<128xf32>, %b4dnmui: tensor<128xf32>, %b4dnvari: tensor<128xf32>, %b4pnmui: tensor<32xf32>, %b4pnvari: tensor<32xf32>, %b5enmui: tensor<128xf32>, %b5envari: tensor<128xf32>, %b5dnmui: tensor<128xf32>, %b5dnvari: tensor<128xf32>, %b5pnmui: tensor<64xf32>, %b5pnvari: tensor<64xf32>, %b6enmui: tensor<256xf32>, %b6envari: tensor<256xf32>, %b6dnmui: tensor<256xf32>, %b6dnvari: tensor<256xf32>, %b6pnmui: tensor<64xf32>, %b6pnvari: tensor<64xf32>, %hnmui: tensor<128xf32>, %hnvari: tensor<128xf32>, %onehot: tensor<32x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<16xf32>, tensor<16xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>) {
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
    %stnxi = stablehlo.reshape %v4 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %stnnf = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %stnep = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %stnsmr = stablehlo.reduce(%stnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %stnsm = stablehlo.broadcast_in_dim %stnsmr, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %stnmu = stablehlo.divide %stnsm, %stnnf : tensor<32x16x112x112xf32>
    %stnxc = stablehlo.subtract %stnxi, %stnmu : tensor<32x16x112x112xf32>
    %stnsq = stablehlo.multiply %stnxc, %stnxc : tensor<32x16x112x112xf32>
    %stnvsr = stablehlo.reduce(%stnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %stnvs = stablehlo.broadcast_in_dim %stnvsr, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %stnvr = stablehlo.divide %stnvs, %stnnf : tensor<32x16x112x112xf32>
    %stnve = stablehlo.add %stnvr, %stnep : tensor<32x16x112x112xf32>
    %stnistd = stablehlo.rsqrt %stnve : tensor<32x16x112x112xf32>
    %stnxh = stablehlo.multiply %stnxc, %stnistd : tensor<32x16x112x112xf32>
    %stngb = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %stnbtb = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %stngx = stablehlo.multiply %stnxh, %stngb : tensor<32x16x112x112xf32>
    %stnn4 = stablehlo.add %stngx, %stnbtb : tensor<32x16x112x112xf32>
    %stn = stablehlo.reshape %stnn4 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v5 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v6 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v7 = stablehlo.maximum %stn, %v5 : tensor<32x200704xf32>
    %v8 = stablehlo.minimum %v7, %v6 : tensor<32x200704xf32>
    %v9 = stablehlo.reshape %v8 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v10 = stablehlo.convolution(%v9, %b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<64x16x1x1xf32>) -> tensor<32x64x112x112xf32>
    %v11 = stablehlo.broadcast_in_dim %b1eb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v12 = stablehlo.add %v10, %v11 : tensor<32x64x112x112xf32>
    %v13 = stablehlo.reshape %v12 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %b1enxi = stablehlo.reshape %v13 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %b1ennf = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %b1enep = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %b1ensmr = stablehlo.reduce(%b1enxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %b1ensm = stablehlo.broadcast_in_dim %b1ensmr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %b1enmu = stablehlo.divide %b1ensm, %b1ennf : tensor<32x64x112x112xf32>
    %b1enxc = stablehlo.subtract %b1enxi, %b1enmu : tensor<32x64x112x112xf32>
    %b1ensq = stablehlo.multiply %b1enxc, %b1enxc : tensor<32x64x112x112xf32>
    %b1envsr = stablehlo.reduce(%b1ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %b1envs = stablehlo.broadcast_in_dim %b1envsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %b1envr = stablehlo.divide %b1envs, %b1ennf : tensor<32x64x112x112xf32>
    %b1enve = stablehlo.add %b1envr, %b1enep : tensor<32x64x112x112xf32>
    %b1enistd = stablehlo.rsqrt %b1enve : tensor<32x64x112x112xf32>
    %b1enxh = stablehlo.multiply %b1enxc, %b1enistd : tensor<32x64x112x112xf32>
    %b1engb = stablehlo.broadcast_in_dim %b1eg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %b1enbtb = stablehlo.broadcast_in_dim %b1ebt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %b1engx = stablehlo.multiply %b1enxh, %b1engb : tensor<32x64x112x112xf32>
    %b1enn4 = stablehlo.add %b1engx, %b1enbtb : tensor<32x64x112x112xf32>
    %b1en = stablehlo.reshape %b1enn4 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v14 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v15 = stablehlo.constant dense<6.0> : tensor<32x802816xf32>
    %v16 = stablehlo.maximum %b1en, %v14 : tensor<32x802816xf32>
    %v17 = stablehlo.minimum %v16, %v15 : tensor<32x802816xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v19 = stablehlo.convolution(%v18, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<32x64x112x112xf32>, tensor<64x1x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v20 = stablehlo.broadcast_in_dim %b1db, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v21 = stablehlo.add %v19, %v20 : tensor<32x64x56x56xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %b1dnxi = stablehlo.reshape %v22 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %b1dnnf = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %b1dnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %b1dnsmr = stablehlo.reduce(%b1dnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %b1dnsm = stablehlo.broadcast_in_dim %b1dnsmr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %b1dnmu = stablehlo.divide %b1dnsm, %b1dnnf : tensor<32x64x56x56xf32>
    %b1dnxc = stablehlo.subtract %b1dnxi, %b1dnmu : tensor<32x64x56x56xf32>
    %b1dnsq = stablehlo.multiply %b1dnxc, %b1dnxc : tensor<32x64x56x56xf32>
    %b1dnvsr = stablehlo.reduce(%b1dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %b1dnvs = stablehlo.broadcast_in_dim %b1dnvsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %b1dnvr = stablehlo.divide %b1dnvs, %b1dnnf : tensor<32x64x56x56xf32>
    %b1dnve = stablehlo.add %b1dnvr, %b1dnep : tensor<32x64x56x56xf32>
    %b1dnistd = stablehlo.rsqrt %b1dnve : tensor<32x64x56x56xf32>
    %b1dnxh = stablehlo.multiply %b1dnxc, %b1dnistd : tensor<32x64x56x56xf32>
    %b1dngb = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %b1dnbtb = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %b1dngx = stablehlo.multiply %b1dnxh, %b1dngb : tensor<32x64x56x56xf32>
    %b1dnn4 = stablehlo.add %b1dngx, %b1dnbtb : tensor<32x64x56x56xf32>
    %b1dn = stablehlo.reshape %b1dnn4 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v23 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v24 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v25 = stablehlo.maximum %b1dn, %v23 : tensor<32x200704xf32>
    %v26 = stablehlo.minimum %v25, %v24 : tensor<32x200704xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v28 = stablehlo.convolution(%v27, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<24x64x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v29 = stablehlo.broadcast_in_dim %b1pb, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<32x24x56x56xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %b1pnxi = stablehlo.reshape %v31 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b1pnnf = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %b1pnep = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %b1pnsmr = stablehlo.reduce(%b1pnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b1pnsm = stablehlo.broadcast_in_dim %b1pnsmr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b1pnmu = stablehlo.divide %b1pnsm, %b1pnnf : tensor<32x24x56x56xf32>
    %b1pnxc = stablehlo.subtract %b1pnxi, %b1pnmu : tensor<32x24x56x56xf32>
    %b1pnsq = stablehlo.multiply %b1pnxc, %b1pnxc : tensor<32x24x56x56xf32>
    %b1pnvsr = stablehlo.reduce(%b1pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b1pnvs = stablehlo.broadcast_in_dim %b1pnvsr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b1pnvr = stablehlo.divide %b1pnvs, %b1pnnf : tensor<32x24x56x56xf32>
    %b1pnve = stablehlo.add %b1pnvr, %b1pnep : tensor<32x24x56x56xf32>
    %b1pnistd = stablehlo.rsqrt %b1pnve : tensor<32x24x56x56xf32>
    %b1pnxh = stablehlo.multiply %b1pnxc, %b1pnistd : tensor<32x24x56x56xf32>
    %b1pngb = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b1pnbtb = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b1pngx = stablehlo.multiply %b1pnxh, %b1pngb : tensor<32x24x56x56xf32>
    %b1pnn4 = stablehlo.add %b1pngx, %b1pnbtb : tensor<32x24x56x56xf32>
    %b1pn = stablehlo.reshape %b1pnn4 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v32 = stablehlo.reshape %b1pn : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v33 = stablehlo.convolution(%v32, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v34 = stablehlo.broadcast_in_dim %b2eb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v35 = stablehlo.add %v33, %v34 : tensor<32x96x56x56xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %b2enxi = stablehlo.reshape %v36 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2ennf = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %b2enep = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %b2ensmr = stablehlo.reduce(%b2enxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ensm = stablehlo.broadcast_in_dim %b2ensmr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2enmu = stablehlo.divide %b2ensm, %b2ennf : tensor<32x96x56x56xf32>
    %b2enxc = stablehlo.subtract %b2enxi, %b2enmu : tensor<32x96x56x56xf32>
    %b2ensq = stablehlo.multiply %b2enxc, %b2enxc : tensor<32x96x56x56xf32>
    %b2envsr = stablehlo.reduce(%b2ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2envs = stablehlo.broadcast_in_dim %b2envsr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2envr = stablehlo.divide %b2envs, %b2ennf : tensor<32x96x56x56xf32>
    %b2enve = stablehlo.add %b2envr, %b2enep : tensor<32x96x56x56xf32>
    %b2enistd = stablehlo.rsqrt %b2enve : tensor<32x96x56x56xf32>
    %b2enxh = stablehlo.multiply %b2enxc, %b2enistd : tensor<32x96x56x56xf32>
    %b2engb = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2enbtb = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2engx = stablehlo.multiply %b2enxh, %b2engb : tensor<32x96x56x56xf32>
    %b2enn4 = stablehlo.add %b2engx, %b2enbtb : tensor<32x96x56x56xf32>
    %b2en = stablehlo.reshape %b2enn4 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v38 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v39 = stablehlo.maximum %b2en, %v37 : tensor<32x301056xf32>
    %v40 = stablehlo.minimum %v39, %v38 : tensor<32x301056xf32>
    %v41 = stablehlo.reshape %v40 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v42 = stablehlo.convolution(%v41, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v43 = stablehlo.broadcast_in_dim %b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v44 = stablehlo.add %v42, %v43 : tensor<32x96x56x56xf32>
    %v45 = stablehlo.reshape %v44 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %b2dnxi = stablehlo.reshape %v45 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2dnnf = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %b2dnep = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %b2dnsmr = stablehlo.reduce(%b2dnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dnsm = stablehlo.broadcast_in_dim %b2dnsmr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dnmu = stablehlo.divide %b2dnsm, %b2dnnf : tensor<32x96x56x56xf32>
    %b2dnxc = stablehlo.subtract %b2dnxi, %b2dnmu : tensor<32x96x56x56xf32>
    %b2dnsq = stablehlo.multiply %b2dnxc, %b2dnxc : tensor<32x96x56x56xf32>
    %b2dnvsr = stablehlo.reduce(%b2dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dnvs = stablehlo.broadcast_in_dim %b2dnvsr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dnvr = stablehlo.divide %b2dnvs, %b2dnnf : tensor<32x96x56x56xf32>
    %b2dnve = stablehlo.add %b2dnvr, %b2dnep : tensor<32x96x56x56xf32>
    %b2dnistd = stablehlo.rsqrt %b2dnve : tensor<32x96x56x56xf32>
    %b2dnxh = stablehlo.multiply %b2dnxc, %b2dnistd : tensor<32x96x56x56xf32>
    %b2dngb = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dnbtb = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dngx = stablehlo.multiply %b2dnxh, %b2dngb : tensor<32x96x56x56xf32>
    %b2dnn4 = stablehlo.add %b2dngx, %b2dnbtb : tensor<32x96x56x56xf32>
    %b2dn = stablehlo.reshape %b2dnn4 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v46 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v47 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v48 = stablehlo.maximum %b2dn, %v46 : tensor<32x301056xf32>
    %v49 = stablehlo.minimum %v48, %v47 : tensor<32x301056xf32>
    %v50 = stablehlo.reshape %v49 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v51 = stablehlo.convolution(%v50, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v52 = stablehlo.broadcast_in_dim %b2pb, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v53 = stablehlo.add %v51, %v52 : tensor<32x24x56x56xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %b2pnxi = stablehlo.reshape %v54 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b2pnnf = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %b2pnep = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %b2pnsmr = stablehlo.reduce(%b2pnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2pnsm = stablehlo.broadcast_in_dim %b2pnsmr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pnmu = stablehlo.divide %b2pnsm, %b2pnnf : tensor<32x24x56x56xf32>
    %b2pnxc = stablehlo.subtract %b2pnxi, %b2pnmu : tensor<32x24x56x56xf32>
    %b2pnsq = stablehlo.multiply %b2pnxc, %b2pnxc : tensor<32x24x56x56xf32>
    %b2pnvsr = stablehlo.reduce(%b2pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2pnvs = stablehlo.broadcast_in_dim %b2pnvsr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pnvr = stablehlo.divide %b2pnvs, %b2pnnf : tensor<32x24x56x56xf32>
    %b2pnve = stablehlo.add %b2pnvr, %b2pnep : tensor<32x24x56x56xf32>
    %b2pnistd = stablehlo.rsqrt %b2pnve : tensor<32x24x56x56xf32>
    %b2pnxh = stablehlo.multiply %b2pnxc, %b2pnistd : tensor<32x24x56x56xf32>
    %b2pngb = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pnbtb = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2pngx = stablehlo.multiply %b2pnxh, %b2pngb : tensor<32x24x56x56xf32>
    %b2pnn4 = stablehlo.add %b2pngx, %b2pnbtb : tensor<32x24x56x56xf32>
    %b2pn = stablehlo.reshape %b2pnn4 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v55 = stablehlo.add %b2pn, %b1pn : tensor<32x75264xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v57 = stablehlo.convolution(%v56, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v58 = stablehlo.broadcast_in_dim %b3eb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v59 = stablehlo.add %v57, %v58 : tensor<32x96x56x56xf32>
    %v60 = stablehlo.reshape %v59 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %b3enxi = stablehlo.reshape %v60 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b3ennf = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %b3enep = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %b3ensmr = stablehlo.reduce(%b3enxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b3ensm = stablehlo.broadcast_in_dim %b3ensmr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b3enmu = stablehlo.divide %b3ensm, %b3ennf : tensor<32x96x56x56xf32>
    %b3enxc = stablehlo.subtract %b3enxi, %b3enmu : tensor<32x96x56x56xf32>
    %b3ensq = stablehlo.multiply %b3enxc, %b3enxc : tensor<32x96x56x56xf32>
    %b3envsr = stablehlo.reduce(%b3ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b3envs = stablehlo.broadcast_in_dim %b3envsr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b3envr = stablehlo.divide %b3envs, %b3ennf : tensor<32x96x56x56xf32>
    %b3enve = stablehlo.add %b3envr, %b3enep : tensor<32x96x56x56xf32>
    %b3enistd = stablehlo.rsqrt %b3enve : tensor<32x96x56x56xf32>
    %b3enxh = stablehlo.multiply %b3enxc, %b3enistd : tensor<32x96x56x56xf32>
    %b3engb = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b3enbtb = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b3engx = stablehlo.multiply %b3enxh, %b3engb : tensor<32x96x56x56xf32>
    %b3enn4 = stablehlo.add %b3engx, %b3enbtb : tensor<32x96x56x56xf32>
    %b3en = stablehlo.reshape %b3enn4 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v61 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v62 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v63 = stablehlo.maximum %b3en, %v61 : tensor<32x301056xf32>
    %v64 = stablehlo.minimum %v63, %v62 : tensor<32x301056xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v66 = stablehlo.convolution(%v65, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x28x28xf32>
    %v67 = stablehlo.broadcast_in_dim %b3db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v68 = stablehlo.add %v66, %v67 : tensor<32x96x28x28xf32>
    %v69 = stablehlo.reshape %v68 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %b3dnxi = stablehlo.reshape %v69 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %b3dnnf = stablehlo.constant dense<25088.0> : tensor<32x96x28x28xf32>
    %b3dnep = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %b3dnsmr = stablehlo.reduce(%b3dnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %b3dnsm = stablehlo.broadcast_in_dim %b3dnsmr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %b3dnmu = stablehlo.divide %b3dnsm, %b3dnnf : tensor<32x96x28x28xf32>
    %b3dnxc = stablehlo.subtract %b3dnxi, %b3dnmu : tensor<32x96x28x28xf32>
    %b3dnsq = stablehlo.multiply %b3dnxc, %b3dnxc : tensor<32x96x28x28xf32>
    %b3dnvsr = stablehlo.reduce(%b3dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %b3dnvs = stablehlo.broadcast_in_dim %b3dnvsr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %b3dnvr = stablehlo.divide %b3dnvs, %b3dnnf : tensor<32x96x28x28xf32>
    %b3dnve = stablehlo.add %b3dnvr, %b3dnep : tensor<32x96x28x28xf32>
    %b3dnistd = stablehlo.rsqrt %b3dnve : tensor<32x96x28x28xf32>
    %b3dnxh = stablehlo.multiply %b3dnxc, %b3dnistd : tensor<32x96x28x28xf32>
    %b3dngb = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %b3dnbtb = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %b3dngx = stablehlo.multiply %b3dnxh, %b3dngb : tensor<32x96x28x28xf32>
    %b3dnn4 = stablehlo.add %b3dngx, %b3dnbtb : tensor<32x96x28x28xf32>
    %b3dn = stablehlo.reshape %b3dnn4 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v70 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v71 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v72 = stablehlo.maximum %b3dn, %v70 : tensor<32x75264xf32>
    %v73 = stablehlo.minimum %v72, %v71 : tensor<32x75264xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v75 = stablehlo.convolution(%v74, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x28x28xf32>, tensor<32x96x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v76 = stablehlo.broadcast_in_dim %b3pb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v77 = stablehlo.add %v75, %v76 : tensor<32x32x28x28xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %b3pnxi = stablehlo.reshape %v78 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b3pnnf = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %b3pnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %b3pnsmr = stablehlo.reduce(%b3pnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b3pnsm = stablehlo.broadcast_in_dim %b3pnsmr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b3pnmu = stablehlo.divide %b3pnsm, %b3pnnf : tensor<32x32x28x28xf32>
    %b3pnxc = stablehlo.subtract %b3pnxi, %b3pnmu : tensor<32x32x28x28xf32>
    %b3pnsq = stablehlo.multiply %b3pnxc, %b3pnxc : tensor<32x32x28x28xf32>
    %b3pnvsr = stablehlo.reduce(%b3pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b3pnvs = stablehlo.broadcast_in_dim %b3pnvsr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b3pnvr = stablehlo.divide %b3pnvs, %b3pnnf : tensor<32x32x28x28xf32>
    %b3pnve = stablehlo.add %b3pnvr, %b3pnep : tensor<32x32x28x28xf32>
    %b3pnistd = stablehlo.rsqrt %b3pnve : tensor<32x32x28x28xf32>
    %b3pnxh = stablehlo.multiply %b3pnxc, %b3pnistd : tensor<32x32x28x28xf32>
    %b3pngb = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b3pnbtb = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b3pngx = stablehlo.multiply %b3pnxh, %b3pngb : tensor<32x32x28x28xf32>
    %b3pnn4 = stablehlo.add %b3pngx, %b3pnbtb : tensor<32x32x28x28xf32>
    %b3pn = stablehlo.reshape %b3pnn4 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v79 = stablehlo.reshape %b3pn : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v80 = stablehlo.convolution(%v79, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v81 = stablehlo.broadcast_in_dim %b4eb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v82 = stablehlo.add %v80, %v81 : tensor<32x128x28x28xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %b4enxi = stablehlo.reshape %v83 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4ennf = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %b4enep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %b4ensmr = stablehlo.reduce(%b4enxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4ensm = stablehlo.broadcast_in_dim %b4ensmr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4enmu = stablehlo.divide %b4ensm, %b4ennf : tensor<32x128x28x28xf32>
    %b4enxc = stablehlo.subtract %b4enxi, %b4enmu : tensor<32x128x28x28xf32>
    %b4ensq = stablehlo.multiply %b4enxc, %b4enxc : tensor<32x128x28x28xf32>
    %b4envsr = stablehlo.reduce(%b4ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4envs = stablehlo.broadcast_in_dim %b4envsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4envr = stablehlo.divide %b4envs, %b4ennf : tensor<32x128x28x28xf32>
    %b4enve = stablehlo.add %b4envr, %b4enep : tensor<32x128x28x28xf32>
    %b4enistd = stablehlo.rsqrt %b4enve : tensor<32x128x28x28xf32>
    %b4enxh = stablehlo.multiply %b4enxc, %b4enistd : tensor<32x128x28x28xf32>
    %b4engb = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4enbtb = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4engx = stablehlo.multiply %b4enxh, %b4engb : tensor<32x128x28x28xf32>
    %b4enn4 = stablehlo.add %b4engx, %b4enbtb : tensor<32x128x28x28xf32>
    %b4en = stablehlo.reshape %b4enn4 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v85 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v86 = stablehlo.maximum %b4en, %v84 : tensor<32x100352xf32>
    %v87 = stablehlo.minimum %v86, %v85 : tensor<32x100352xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v89 = stablehlo.convolution(%v88, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v90 = stablehlo.broadcast_in_dim %b4db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v91 = stablehlo.add %v89, %v90 : tensor<32x128x28x28xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %b4dnxi = stablehlo.reshape %v92 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4dnnf = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %b4dnep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %b4dnsmr = stablehlo.reduce(%b4dnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4dnsm = stablehlo.broadcast_in_dim %b4dnsmr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4dnmu = stablehlo.divide %b4dnsm, %b4dnnf : tensor<32x128x28x28xf32>
    %b4dnxc = stablehlo.subtract %b4dnxi, %b4dnmu : tensor<32x128x28x28xf32>
    %b4dnsq = stablehlo.multiply %b4dnxc, %b4dnxc : tensor<32x128x28x28xf32>
    %b4dnvsr = stablehlo.reduce(%b4dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4dnvs = stablehlo.broadcast_in_dim %b4dnvsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4dnvr = stablehlo.divide %b4dnvs, %b4dnnf : tensor<32x128x28x28xf32>
    %b4dnve = stablehlo.add %b4dnvr, %b4dnep : tensor<32x128x28x28xf32>
    %b4dnistd = stablehlo.rsqrt %b4dnve : tensor<32x128x28x28xf32>
    %b4dnxh = stablehlo.multiply %b4dnxc, %b4dnistd : tensor<32x128x28x28xf32>
    %b4dngb = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4dnbtb = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4dngx = stablehlo.multiply %b4dnxh, %b4dngb : tensor<32x128x28x28xf32>
    %b4dnn4 = stablehlo.add %b4dngx, %b4dnbtb : tensor<32x128x28x28xf32>
    %b4dn = stablehlo.reshape %b4dnn4 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v93 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v94 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v95 = stablehlo.maximum %b4dn, %v93 : tensor<32x100352xf32>
    %v96 = stablehlo.minimum %v95, %v94 : tensor<32x100352xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v98 = stablehlo.convolution(%v97, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v99 = stablehlo.broadcast_in_dim %b4pb, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v100 = stablehlo.add %v98, %v99 : tensor<32x32x28x28xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %b4pnxi = stablehlo.reshape %v101 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b4pnnf = stablehlo.constant dense<25088.0> : tensor<32x32x28x28xf32>
    %b4pnep = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %b4pnsmr = stablehlo.reduce(%b4pnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b4pnsm = stablehlo.broadcast_in_dim %b4pnsmr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4pnmu = stablehlo.divide %b4pnsm, %b4pnnf : tensor<32x32x28x28xf32>
    %b4pnxc = stablehlo.subtract %b4pnxi, %b4pnmu : tensor<32x32x28x28xf32>
    %b4pnsq = stablehlo.multiply %b4pnxc, %b4pnxc : tensor<32x32x28x28xf32>
    %b4pnvsr = stablehlo.reduce(%b4pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b4pnvs = stablehlo.broadcast_in_dim %b4pnvsr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4pnvr = stablehlo.divide %b4pnvs, %b4pnnf : tensor<32x32x28x28xf32>
    %b4pnve = stablehlo.add %b4pnvr, %b4pnep : tensor<32x32x28x28xf32>
    %b4pnistd = stablehlo.rsqrt %b4pnve : tensor<32x32x28x28xf32>
    %b4pnxh = stablehlo.multiply %b4pnxc, %b4pnistd : tensor<32x32x28x28xf32>
    %b4pngb = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4pnbtb = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4pngx = stablehlo.multiply %b4pnxh, %b4pngb : tensor<32x32x28x28xf32>
    %b4pnn4 = stablehlo.add %b4pngx, %b4pnbtb : tensor<32x32x28x28xf32>
    %b4pn = stablehlo.reshape %b4pnn4 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v102 = stablehlo.add %b4pn, %b3pn : tensor<32x25088xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v104 = stablehlo.convolution(%v103, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v105 = stablehlo.broadcast_in_dim %b5eb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v106 = stablehlo.add %v104, %v105 : tensor<32x128x28x28xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %b5enxi = stablehlo.reshape %v107 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b5ennf = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %b5enep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %b5ensmr = stablehlo.reduce(%b5enxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b5ensm = stablehlo.broadcast_in_dim %b5ensmr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b5enmu = stablehlo.divide %b5ensm, %b5ennf : tensor<32x128x28x28xf32>
    %b5enxc = stablehlo.subtract %b5enxi, %b5enmu : tensor<32x128x28x28xf32>
    %b5ensq = stablehlo.multiply %b5enxc, %b5enxc : tensor<32x128x28x28xf32>
    %b5envsr = stablehlo.reduce(%b5ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b5envs = stablehlo.broadcast_in_dim %b5envsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b5envr = stablehlo.divide %b5envs, %b5ennf : tensor<32x128x28x28xf32>
    %b5enve = stablehlo.add %b5envr, %b5enep : tensor<32x128x28x28xf32>
    %b5enistd = stablehlo.rsqrt %b5enve : tensor<32x128x28x28xf32>
    %b5enxh = stablehlo.multiply %b5enxc, %b5enistd : tensor<32x128x28x28xf32>
    %b5engb = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b5enbtb = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b5engx = stablehlo.multiply %b5enxh, %b5engb : tensor<32x128x28x28xf32>
    %b5enn4 = stablehlo.add %b5engx, %b5enbtb : tensor<32x128x28x28xf32>
    %b5en = stablehlo.reshape %b5enn4 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v108 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v109 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v110 = stablehlo.maximum %b5en, %v108 : tensor<32x100352xf32>
    %v111 = stablehlo.minimum %v110, %v109 : tensor<32x100352xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v113 = stablehlo.convolution(%v112, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x14x14xf32>
    %v114 = stablehlo.broadcast_in_dim %b5db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v115 = stablehlo.add %v113, %v114 : tensor<32x128x14x14xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %b5dnxi = stablehlo.reshape %v116 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %b5dnnf = stablehlo.constant dense<6272.0> : tensor<32x128x14x14xf32>
    %b5dnep = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %b5dnsmr = stablehlo.reduce(%b5dnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %b5dnsm = stablehlo.broadcast_in_dim %b5dnsmr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %b5dnmu = stablehlo.divide %b5dnsm, %b5dnnf : tensor<32x128x14x14xf32>
    %b5dnxc = stablehlo.subtract %b5dnxi, %b5dnmu : tensor<32x128x14x14xf32>
    %b5dnsq = stablehlo.multiply %b5dnxc, %b5dnxc : tensor<32x128x14x14xf32>
    %b5dnvsr = stablehlo.reduce(%b5dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %b5dnvs = stablehlo.broadcast_in_dim %b5dnvsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %b5dnvr = stablehlo.divide %b5dnvs, %b5dnnf : tensor<32x128x14x14xf32>
    %b5dnve = stablehlo.add %b5dnvr, %b5dnep : tensor<32x128x14x14xf32>
    %b5dnistd = stablehlo.rsqrt %b5dnve : tensor<32x128x14x14xf32>
    %b5dnxh = stablehlo.multiply %b5dnxc, %b5dnistd : tensor<32x128x14x14xf32>
    %b5dngb = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %b5dnbtb = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %b5dngx = stablehlo.multiply %b5dnxh, %b5dngb : tensor<32x128x14x14xf32>
    %b5dnn4 = stablehlo.add %b5dngx, %b5dnbtb : tensor<32x128x14x14xf32>
    %b5dn = stablehlo.reshape %b5dnn4 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v117 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v118 = stablehlo.constant dense<6.0> : tensor<32x25088xf32>
    %v119 = stablehlo.maximum %b5dn, %v117 : tensor<32x25088xf32>
    %v120 = stablehlo.minimum %v119, %v118 : tensor<32x25088xf32>
    %v121 = stablehlo.reshape %v120 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v122 = stablehlo.convolution(%v121, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x14x14xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v123 = stablehlo.broadcast_in_dim %b5pb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v124 = stablehlo.add %v122, %v123 : tensor<32x64x14x14xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %b5pnxi = stablehlo.reshape %v125 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %b5pnnf = stablehlo.constant dense<6272.0> : tensor<32x64x14x14xf32>
    %b5pnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %b5pnsmr = stablehlo.reduce(%b5pnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b5pnsm = stablehlo.broadcast_in_dim %b5pnsmr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b5pnmu = stablehlo.divide %b5pnsm, %b5pnnf : tensor<32x64x14x14xf32>
    %b5pnxc = stablehlo.subtract %b5pnxi, %b5pnmu : tensor<32x64x14x14xf32>
    %b5pnsq = stablehlo.multiply %b5pnxc, %b5pnxc : tensor<32x64x14x14xf32>
    %b5pnvsr = stablehlo.reduce(%b5pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b5pnvs = stablehlo.broadcast_in_dim %b5pnvsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b5pnvr = stablehlo.divide %b5pnvs, %b5pnnf : tensor<32x64x14x14xf32>
    %b5pnve = stablehlo.add %b5pnvr, %b5pnep : tensor<32x64x14x14xf32>
    %b5pnistd = stablehlo.rsqrt %b5pnve : tensor<32x64x14x14xf32>
    %b5pnxh = stablehlo.multiply %b5pnxc, %b5pnistd : tensor<32x64x14x14xf32>
    %b5pngb = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b5pnbtb = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b5pngx = stablehlo.multiply %b5pnxh, %b5pngb : tensor<32x64x14x14xf32>
    %b5pnn4 = stablehlo.add %b5pngx, %b5pnbtb : tensor<32x64x14x14xf32>
    %b5pn = stablehlo.reshape %b5pnn4 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v126 = stablehlo.reshape %b5pn : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v127 = stablehlo.convolution(%v126, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v128 = stablehlo.broadcast_in_dim %b6eb, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v129 = stablehlo.add %v127, %v128 : tensor<32x256x14x14xf32>
    %v130 = stablehlo.reshape %v129 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %b6enxi = stablehlo.reshape %v130 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %b6ennf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %b6enep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %b6ensmr = stablehlo.reduce(%b6enxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %b6ensm = stablehlo.broadcast_in_dim %b6ensmr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %b6enmu = stablehlo.divide %b6ensm, %b6ennf : tensor<32x256x14x14xf32>
    %b6enxc = stablehlo.subtract %b6enxi, %b6enmu : tensor<32x256x14x14xf32>
    %b6ensq = stablehlo.multiply %b6enxc, %b6enxc : tensor<32x256x14x14xf32>
    %b6envsr = stablehlo.reduce(%b6ensq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %b6envs = stablehlo.broadcast_in_dim %b6envsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %b6envr = stablehlo.divide %b6envs, %b6ennf : tensor<32x256x14x14xf32>
    %b6enve = stablehlo.add %b6envr, %b6enep : tensor<32x256x14x14xf32>
    %b6enistd = stablehlo.rsqrt %b6enve : tensor<32x256x14x14xf32>
    %b6enxh = stablehlo.multiply %b6enxc, %b6enistd : tensor<32x256x14x14xf32>
    %b6engb = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %b6enbtb = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %b6engx = stablehlo.multiply %b6enxh, %b6engb : tensor<32x256x14x14xf32>
    %b6enn4 = stablehlo.add %b6engx, %b6enbtb : tensor<32x256x14x14xf32>
    %b6en = stablehlo.reshape %b6enn4 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v131 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v132 = stablehlo.constant dense<6.0> : tensor<32x50176xf32>
    %v133 = stablehlo.maximum %b6en, %v131 : tensor<32x50176xf32>
    %v134 = stablehlo.minimum %v133, %v132 : tensor<32x50176xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v136 = stablehlo.convolution(%v135, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x14x14xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v137 = stablehlo.broadcast_in_dim %b6db, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v138 = stablehlo.add %v136, %v137 : tensor<32x256x7x7xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %b6dnxi = stablehlo.reshape %v139 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %b6dnnf = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %b6dnep = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %b6dnsmr = stablehlo.reduce(%b6dnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %b6dnsm = stablehlo.broadcast_in_dim %b6dnsmr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %b6dnmu = stablehlo.divide %b6dnsm, %b6dnnf : tensor<32x256x7x7xf32>
    %b6dnxc = stablehlo.subtract %b6dnxi, %b6dnmu : tensor<32x256x7x7xf32>
    %b6dnsq = stablehlo.multiply %b6dnxc, %b6dnxc : tensor<32x256x7x7xf32>
    %b6dnvsr = stablehlo.reduce(%b6dnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %b6dnvs = stablehlo.broadcast_in_dim %b6dnvsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %b6dnvr = stablehlo.divide %b6dnvs, %b6dnnf : tensor<32x256x7x7xf32>
    %b6dnve = stablehlo.add %b6dnvr, %b6dnep : tensor<32x256x7x7xf32>
    %b6dnistd = stablehlo.rsqrt %b6dnve : tensor<32x256x7x7xf32>
    %b6dnxh = stablehlo.multiply %b6dnxc, %b6dnistd : tensor<32x256x7x7xf32>
    %b6dngb = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %b6dnbtb = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %b6dngx = stablehlo.multiply %b6dnxh, %b6dngb : tensor<32x256x7x7xf32>
    %b6dnn4 = stablehlo.add %b6dngx, %b6dnbtb : tensor<32x256x7x7xf32>
    %b6dn = stablehlo.reshape %b6dnn4 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v140 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v141 = stablehlo.constant dense<6.0> : tensor<32x12544xf32>
    %v142 = stablehlo.maximum %b6dn, %v140 : tensor<32x12544xf32>
    %v143 = stablehlo.minimum %v142, %v141 : tensor<32x12544xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v145 = stablehlo.convolution(%v144, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<64x256x1x1xf32>) -> tensor<32x64x7x7xf32>
    %v146 = stablehlo.broadcast_in_dim %b6pb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v147 = stablehlo.add %v145, %v146 : tensor<32x64x7x7xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %b6pnxi = stablehlo.reshape %v148 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %b6pnnf = stablehlo.constant dense<1568.0> : tensor<32x64x7x7xf32>
    %b6pnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %b6pnsmr = stablehlo.reduce(%b6pnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %b6pnsm = stablehlo.broadcast_in_dim %b6pnsmr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %b6pnmu = stablehlo.divide %b6pnsm, %b6pnnf : tensor<32x64x7x7xf32>
    %b6pnxc = stablehlo.subtract %b6pnxi, %b6pnmu : tensor<32x64x7x7xf32>
    %b6pnsq = stablehlo.multiply %b6pnxc, %b6pnxc : tensor<32x64x7x7xf32>
    %b6pnvsr = stablehlo.reduce(%b6pnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %b6pnvs = stablehlo.broadcast_in_dim %b6pnvsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %b6pnvr = stablehlo.divide %b6pnvs, %b6pnnf : tensor<32x64x7x7xf32>
    %b6pnve = stablehlo.add %b6pnvr, %b6pnep : tensor<32x64x7x7xf32>
    %b6pnistd = stablehlo.rsqrt %b6pnve : tensor<32x64x7x7xf32>
    %b6pnxh = stablehlo.multiply %b6pnxc, %b6pnistd : tensor<32x64x7x7xf32>
    %b6pngb = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %b6pnbtb = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %b6pngx = stablehlo.multiply %b6pnxh, %b6pngb : tensor<32x64x7x7xf32>
    %b6pnn4 = stablehlo.add %b6pngx, %b6pnbtb : tensor<32x64x7x7xf32>
    %b6pn = stablehlo.reshape %b6pnn4 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v149 = stablehlo.reshape %b6pn : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v150 = stablehlo.convolution(%v149, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x7x7xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x7x7xf32>
    %v151 = stablehlo.broadcast_in_dim %hb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v152 = stablehlo.add %v150, %v151 : tensor<32x128x7x7xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %hnxi = stablehlo.reshape %v153 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %hnnf = stablehlo.constant dense<1568.0> : tensor<32x128x7x7xf32>
    %hnep = stablehlo.constant dense<1.0e-5> : tensor<32x128x7x7xf32>
    %hnsmr = stablehlo.reduce(%hnxi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %hnsm = stablehlo.broadcast_in_dim %hnsmr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %hnmu = stablehlo.divide %hnsm, %hnnf : tensor<32x128x7x7xf32>
    %hnxc = stablehlo.subtract %hnxi, %hnmu : tensor<32x128x7x7xf32>
    %hnsq = stablehlo.multiply %hnxc, %hnxc : tensor<32x128x7x7xf32>
    %hnvsr = stablehlo.reduce(%hnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %hnvs = stablehlo.broadcast_in_dim %hnvsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %hnvr = stablehlo.divide %hnvs, %hnnf : tensor<32x128x7x7xf32>
    %hnve = stablehlo.add %hnvr, %hnep : tensor<32x128x7x7xf32>
    %hnistd = stablehlo.rsqrt %hnve : tensor<32x128x7x7xf32>
    %hnxh = stablehlo.multiply %hnxc, %hnistd : tensor<32x128x7x7xf32>
    %hngb = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %hnbtb = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %hngx = stablehlo.multiply %hnxh, %hngb : tensor<32x128x7x7xf32>
    %hnn4 = stablehlo.add %hngx, %hnbtb : tensor<32x128x7x7xf32>
    %hn = stablehlo.reshape %hnn4 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<32x6272xf32>
    %v155 = stablehlo.constant dense<6.0> : tensor<32x6272xf32>
    %v156 = stablehlo.maximum %hn, %v154 : tensor<32x6272xf32>
    %v157 = stablehlo.minimum %v156, %v155 : tensor<32x6272xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v160 = stablehlo.reduce(%v158 init: %v159) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v161 = stablehlo.constant dense<49.0> : tensor<32x128xf32>
    %v162 = stablehlo.divide %v160, %v161 : tensor<32x128xf32>
    %v163 = stablehlo.dot_general %v162, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x128xf32>, tensor<128x10xf32>) -> tensor<32x10xf32>
    %v164 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v165 = stablehlo.add %v163, %v164 : tensor<32x10xf32>
    %v166 = stablehlo.exponential %v165 : tensor<32x10xf32>
    %v167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v168 = stablehlo.reduce(%v166 init: %v167) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v169 = stablehlo.broadcast_in_dim %v168, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v170 = stablehlo.divide %v166, %v169 : tensor<32x10xf32>
    %dyr0 = stablehlo.subtract %v170, %onehot : tensor<32x10xf32>
    %lsa = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %lsaoh = stablehlo.multiply %lsa, %onehot : tensor<32x10xf32>
    %dyr1 = stablehlo.add %dyr0, %lsaoh : tensor<32x10xf32>
    %lsaik = stablehlo.constant dense<0.010000> : tensor<32x10xf32>
    %dyr = stablehlo.subtract %dyr1, %lsaik : tensor<32x10xf32>
    %dy = stablehlo.divide %dyr, %bsc : tensor<32x10xf32>
    %llog = stablehlo.log %v170 : tensor<32x10xf32>
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
    %v171 = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<128x10xf32>) -> tensor<32x128xf32>
    %dgi = stablehlo.reshape %v171 : (tensor<32x128xf32>) -> tensor<32x128x1x1xf32>
    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<32x128x7x7xf32>
    %dgn = stablehlo.constant dense<49.0> : tensor<32x128x7x7xf32>
    %dgd = stablehlo.divide %dgb, %dgn : tensor<32x128x7x7xf32>
    %dgapf = stablehlo.reshape %dgd : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v172 = stablehlo.constant dense<0.0> : tensor<32x6272xf32>
    %v173 = stablehlo.constant dense<6.0> : tensor<32x6272xf32>
    %v174 = stablehlo.compare GT, %hn, %v172 : (tensor<32x6272xf32>, tensor<32x6272xf32>) -> tensor<32x6272xi1>
    %v175 = stablehlo.compare LT, %hn, %v173 : (tensor<32x6272xf32>, tensor<32x6272xf32>) -> tensor<32x6272xi1>
    %v176 = stablehlo.and %v174, %v175 : tensor<32x6272xi1>
    %v177 = stablehlo.select %v176, %dgapf, %v172 : tensor<32x6272xi1>, tensor<32x6272xf32>
    %dhndyi = stablehlo.reshape %v177 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %dhndxh = stablehlo.multiply %hngb, %dhndyi : tensor<32x128x7x7xf32>
    %dhnsdxr = stablehlo.reduce(%dhndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %dhnsdx = stablehlo.broadcast_in_dim %dhnsdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %dhnxd = stablehlo.multiply %hnxh, %dhndxh : tensor<32x128x7x7xf32>
    %dhnsxdr = stablehlo.reduce(%dhnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %dhnsxd = stablehlo.broadcast_in_dim %dhnsxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %dhnt1 = stablehlo.multiply %dhndxh, %hnnf : tensor<32x128x7x7xf32>
    %dhni1 = stablehlo.subtract %dhnt1, %dhnsdx : tensor<32x128x7x7xf32>
    %dhnxs = stablehlo.multiply %hnxh, %dhnsxd : tensor<32x128x7x7xf32>
    %dhni2 = stablehlo.subtract %dhni1, %dhnxs : tensor<32x128x7x7xf32>
    %dhnsN = stablehlo.divide %hnistd, %hnnf : tensor<32x128x7x7xf32>
    %dhndxn = stablehlo.multiply %dhnsN, %dhni2 : tensor<32x128x7x7xf32>
    %dhn = stablehlo.reshape %dhndxn : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %dhndgp = stablehlo.multiply %dhndyi, %hnxh : tensor<32x128x7x7xf32>
    %dhndg = stablehlo.reduce(%dhndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %dhndb = stablehlo.reduce(%dhndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %v178 = stablehlo.reshape %dhn : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v179 = stablehlo.transpose %hW, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v180 = stablehlo.reverse %v179, dims = [2, 3] : tensor<64x128x1x1xf32>
    %v181 = stablehlo.convolution(%v178, %v180)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x7x7xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x7x7xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %b6dpndyi = stablehlo.reshape %v182 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %b6dpndxh = stablehlo.multiply %b6pngb, %b6dpndyi : tensor<32x64x7x7xf32>
    %b6dpnsdxr = stablehlo.reduce(%b6dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %b6dpnsdx = stablehlo.broadcast_in_dim %b6dpnsdxr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %b6dpnxd = stablehlo.multiply %b6pnxh, %b6dpndxh : tensor<32x64x7x7xf32>
    %b6dpnsxdr = stablehlo.reduce(%b6dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %b6dpnsxd = stablehlo.broadcast_in_dim %b6dpnsxdr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %b6dpnt1 = stablehlo.multiply %b6dpndxh, %b6pnnf : tensor<32x64x7x7xf32>
    %b6dpni1 = stablehlo.subtract %b6dpnt1, %b6dpnsdx : tensor<32x64x7x7xf32>
    %b6dpnxs = stablehlo.multiply %b6pnxh, %b6dpnsxd : tensor<32x64x7x7xf32>
    %b6dpni2 = stablehlo.subtract %b6dpni1, %b6dpnxs : tensor<32x64x7x7xf32>
    %b6dpnsN = stablehlo.divide %b6pnistd, %b6pnnf : tensor<32x64x7x7xf32>
    %b6dpndxn = stablehlo.multiply %b6dpnsN, %b6dpni2 : tensor<32x64x7x7xf32>
    %b6dpn = stablehlo.reshape %b6dpndxn : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %b6dpndgp = stablehlo.multiply %b6dpndyi, %b6pnxh : tensor<32x64x7x7xf32>
    %b6dpndg = stablehlo.reduce(%b6dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %b6dpndb = stablehlo.reduce(%b6dpndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %v183 = stablehlo.reshape %b6dpn : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v184 = stablehlo.transpose %b6pW, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v185 = stablehlo.reverse %v184, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v186 = stablehlo.convolution(%v183, %v185)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x7x7xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v188 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v189 = stablehlo.constant dense<6.0> : tensor<32x12544xf32>
    %v190 = stablehlo.compare GT, %b6dn, %v188 : (tensor<32x12544xf32>, tensor<32x12544xf32>) -> tensor<32x12544xi1>
    %v191 = stablehlo.compare LT, %b6dn, %v189 : (tensor<32x12544xf32>, tensor<32x12544xf32>) -> tensor<32x12544xi1>
    %v192 = stablehlo.and %v190, %v191 : tensor<32x12544xi1>
    %v193 = stablehlo.select %v192, %v187, %v188 : tensor<32x12544xi1>, tensor<32x12544xf32>
    %b6ddndyi = stablehlo.reshape %v193 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %b6ddndxh = stablehlo.multiply %b6dngb, %b6ddndyi : tensor<32x256x7x7xf32>
    %b6ddnsdxr = stablehlo.reduce(%b6ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %b6ddnsdx = stablehlo.broadcast_in_dim %b6ddnsdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %b6ddnxd = stablehlo.multiply %b6dnxh, %b6ddndxh : tensor<32x256x7x7xf32>
    %b6ddnsxdr = stablehlo.reduce(%b6ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %b6ddnsxd = stablehlo.broadcast_in_dim %b6ddnsxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %b6ddnt1 = stablehlo.multiply %b6ddndxh, %b6dnnf : tensor<32x256x7x7xf32>
    %b6ddni1 = stablehlo.subtract %b6ddnt1, %b6ddnsdx : tensor<32x256x7x7xf32>
    %b6ddnxs = stablehlo.multiply %b6dnxh, %b6ddnsxd : tensor<32x256x7x7xf32>
    %b6ddni2 = stablehlo.subtract %b6ddni1, %b6ddnxs : tensor<32x256x7x7xf32>
    %b6ddnsN = stablehlo.divide %b6dnistd, %b6dnnf : tensor<32x256x7x7xf32>
    %b6ddndxn = stablehlo.multiply %b6ddnsN, %b6ddni2 : tensor<32x256x7x7xf32>
    %b6ddn = stablehlo.reshape %b6ddndxn : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %b6ddndgp = stablehlo.multiply %b6ddndyi, %b6dnxh : tensor<32x256x7x7xf32>
    %b6ddndg = stablehlo.reduce(%b6ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %b6ddndb = stablehlo.reduce(%b6ddndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v194 = stablehlo.reshape %b6ddn : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v195 = stablehlo.constant dense<0.0> : tensor<f32>
    %v196 = stablehlo.pad %v194, %v195, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256x14x14xf32>
    %v197 = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<256x1x3x3xf32>
    %v198 = stablehlo.convolution(%v196, %v197)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x14x14xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v200 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v201 = stablehlo.constant dense<6.0> : tensor<32x50176xf32>
    %v202 = stablehlo.compare GT, %b6en, %v200 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v203 = stablehlo.compare LT, %b6en, %v201 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v204 = stablehlo.and %v202, %v203 : tensor<32x50176xi1>
    %v205 = stablehlo.select %v204, %v199, %v200 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %b6dendyi = stablehlo.reshape %v205 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %b6dendxh = stablehlo.multiply %b6engb, %b6dendyi : tensor<32x256x14x14xf32>
    %b6densdxr = stablehlo.reduce(%b6dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %b6densdx = stablehlo.broadcast_in_dim %b6densdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %b6denxd = stablehlo.multiply %b6enxh, %b6dendxh : tensor<32x256x14x14xf32>
    %b6densxdr = stablehlo.reduce(%b6denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %b6densxd = stablehlo.broadcast_in_dim %b6densxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %b6dent1 = stablehlo.multiply %b6dendxh, %b6ennf : tensor<32x256x14x14xf32>
    %b6deni1 = stablehlo.subtract %b6dent1, %b6densdx : tensor<32x256x14x14xf32>
    %b6denxs = stablehlo.multiply %b6enxh, %b6densxd : tensor<32x256x14x14xf32>
    %b6deni2 = stablehlo.subtract %b6deni1, %b6denxs : tensor<32x256x14x14xf32>
    %b6densN = stablehlo.divide %b6enistd, %b6ennf : tensor<32x256x14x14xf32>
    %b6dendxn = stablehlo.multiply %b6densN, %b6deni2 : tensor<32x256x14x14xf32>
    %b6den = stablehlo.reshape %b6dendxn : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %b6dendgp = stablehlo.multiply %b6dendyi, %b6enxh : tensor<32x256x14x14xf32>
    %b6dendg = stablehlo.reduce(%b6dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %b6dendb = stablehlo.reduce(%b6dendyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v206 = stablehlo.reshape %b6den : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v207 = stablehlo.transpose %b6eW, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v208 = stablehlo.reverse %v207, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v209 = stablehlo.convolution(%v206, %v208)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<64x256x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %b5dpndyi = stablehlo.reshape %v210 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %b5dpndxh = stablehlo.multiply %b5pngb, %b5dpndyi : tensor<32x64x14x14xf32>
    %b5dpnsdxr = stablehlo.reduce(%b5dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b5dpnsdx = stablehlo.broadcast_in_dim %b5dpnsdxr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b5dpnxd = stablehlo.multiply %b5pnxh, %b5dpndxh : tensor<32x64x14x14xf32>
    %b5dpnsxdr = stablehlo.reduce(%b5dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b5dpnsxd = stablehlo.broadcast_in_dim %b5dpnsxdr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %b5dpnt1 = stablehlo.multiply %b5dpndxh, %b5pnnf : tensor<32x64x14x14xf32>
    %b5dpni1 = stablehlo.subtract %b5dpnt1, %b5dpnsdx : tensor<32x64x14x14xf32>
    %b5dpnxs = stablehlo.multiply %b5pnxh, %b5dpnsxd : tensor<32x64x14x14xf32>
    %b5dpni2 = stablehlo.subtract %b5dpni1, %b5dpnxs : tensor<32x64x14x14xf32>
    %b5dpnsN = stablehlo.divide %b5pnistd, %b5pnnf : tensor<32x64x14x14xf32>
    %b5dpndxn = stablehlo.multiply %b5dpnsN, %b5dpni2 : tensor<32x64x14x14xf32>
    %b5dpn = stablehlo.reshape %b5dpndxn : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %b5dpndgp = stablehlo.multiply %b5dpndyi, %b5pnxh : tensor<32x64x14x14xf32>
    %b5dpndg = stablehlo.reduce(%b5dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b5dpndb = stablehlo.reduce(%b5dpndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v211 = stablehlo.reshape %b5dpn : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v212 = stablehlo.transpose %b5pW, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v213 = stablehlo.reverse %v212, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v214 = stablehlo.convolution(%v211, %v213)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x14x14xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v216 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v217 = stablehlo.constant dense<6.0> : tensor<32x25088xf32>
    %v218 = stablehlo.compare GT, %b5dn, %v216 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v219 = stablehlo.compare LT, %b5dn, %v217 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v220 = stablehlo.and %v218, %v219 : tensor<32x25088xi1>
    %v221 = stablehlo.select %v220, %v215, %v216 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %b5ddndyi = stablehlo.reshape %v221 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %b5ddndxh = stablehlo.multiply %b5dngb, %b5ddndyi : tensor<32x128x14x14xf32>
    %b5ddnsdxr = stablehlo.reduce(%b5ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %b5ddnsdx = stablehlo.broadcast_in_dim %b5ddnsdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %b5ddnxd = stablehlo.multiply %b5dnxh, %b5ddndxh : tensor<32x128x14x14xf32>
    %b5ddnsxdr = stablehlo.reduce(%b5ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %b5ddnsxd = stablehlo.broadcast_in_dim %b5ddnsxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %b5ddnt1 = stablehlo.multiply %b5ddndxh, %b5dnnf : tensor<32x128x14x14xf32>
    %b5ddni1 = stablehlo.subtract %b5ddnt1, %b5ddnsdx : tensor<32x128x14x14xf32>
    %b5ddnxs = stablehlo.multiply %b5dnxh, %b5ddnsxd : tensor<32x128x14x14xf32>
    %b5ddni2 = stablehlo.subtract %b5ddni1, %b5ddnxs : tensor<32x128x14x14xf32>
    %b5ddnsN = stablehlo.divide %b5dnistd, %b5dnnf : tensor<32x128x14x14xf32>
    %b5ddndxn = stablehlo.multiply %b5ddnsN, %b5ddni2 : tensor<32x128x14x14xf32>
    %b5ddn = stablehlo.reshape %b5ddndxn : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %b5ddndgp = stablehlo.multiply %b5ddndyi, %b5dnxh : tensor<32x128x14x14xf32>
    %b5ddndg = stablehlo.reduce(%b5ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %b5ddndb = stablehlo.reduce(%b5ddndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %v222 = stablehlo.reshape %b5ddn : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v224 = stablehlo.pad %v222, %v223, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128x28x28xf32>
    %v225 = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<128x1x3x3xf32>
    %v226 = stablehlo.convolution(%v224, %v225)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v228 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v229 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v230 = stablehlo.compare GT, %b5en, %v228 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v231 = stablehlo.compare LT, %b5en, %v229 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v232 = stablehlo.and %v230, %v231 : tensor<32x100352xi1>
    %v233 = stablehlo.select %v232, %v227, %v228 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %b5dendyi = stablehlo.reshape %v233 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b5dendxh = stablehlo.multiply %b5engb, %b5dendyi : tensor<32x128x28x28xf32>
    %b5densdxr = stablehlo.reduce(%b5dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b5densdx = stablehlo.broadcast_in_dim %b5densdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b5denxd = stablehlo.multiply %b5enxh, %b5dendxh : tensor<32x128x28x28xf32>
    %b5densxdr = stablehlo.reduce(%b5denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b5densxd = stablehlo.broadcast_in_dim %b5densxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b5dent1 = stablehlo.multiply %b5dendxh, %b5ennf : tensor<32x128x28x28xf32>
    %b5deni1 = stablehlo.subtract %b5dent1, %b5densdx : tensor<32x128x28x28xf32>
    %b5denxs = stablehlo.multiply %b5enxh, %b5densxd : tensor<32x128x28x28xf32>
    %b5deni2 = stablehlo.subtract %b5deni1, %b5denxs : tensor<32x128x28x28xf32>
    %b5densN = stablehlo.divide %b5enistd, %b5ennf : tensor<32x128x28x28xf32>
    %b5dendxn = stablehlo.multiply %b5densN, %b5deni2 : tensor<32x128x28x28xf32>
    %b5den = stablehlo.reshape %b5dendxn : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %b5dendgp = stablehlo.multiply %b5dendyi, %b5enxh : tensor<32x128x28x28xf32>
    %b5dendg = stablehlo.reduce(%b5dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b5dendb = stablehlo.reduce(%b5dendyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v234 = stablehlo.reshape %b5den : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v235 = stablehlo.transpose %b5eW, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v236 = stablehlo.reverse %v235, dims = [2, 3] : tensor<32x128x1x1xf32>
    %v237 = stablehlo.convolution(%v234, %v236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %b4dpndyi = stablehlo.reshape %v238 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b4dpndxh = stablehlo.multiply %b4pngb, %b4dpndyi : tensor<32x32x28x28xf32>
    %b4dpnsdxr = stablehlo.reduce(%b4dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b4dpnsdx = stablehlo.broadcast_in_dim %b4dpnsdxr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4dpnxd = stablehlo.multiply %b4pnxh, %b4dpndxh : tensor<32x32x28x28xf32>
    %b4dpnsxdr = stablehlo.reduce(%b4dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b4dpnsxd = stablehlo.broadcast_in_dim %b4dpnsxdr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b4dpnt1 = stablehlo.multiply %b4dpndxh, %b4pnnf : tensor<32x32x28x28xf32>
    %b4dpni1 = stablehlo.subtract %b4dpnt1, %b4dpnsdx : tensor<32x32x28x28xf32>
    %b4dpnxs = stablehlo.multiply %b4pnxh, %b4dpnsxd : tensor<32x32x28x28xf32>
    %b4dpni2 = stablehlo.subtract %b4dpni1, %b4dpnxs : tensor<32x32x28x28xf32>
    %b4dpnsN = stablehlo.divide %b4pnistd, %b4pnnf : tensor<32x32x28x28xf32>
    %b4dpndxn = stablehlo.multiply %b4dpnsN, %b4dpni2 : tensor<32x32x28x28xf32>
    %b4dpn = stablehlo.reshape %b4dpndxn : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %b4dpndgp = stablehlo.multiply %b4dpndyi, %b4pnxh : tensor<32x32x28x28xf32>
    %b4dpndg = stablehlo.reduce(%b4dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b4dpndb = stablehlo.reduce(%b4dpndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v239 = stablehlo.reshape %b4dpn : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v240 = stablehlo.transpose %b4pW, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %v241 = stablehlo.reverse %v240, dims = [2, 3] : tensor<128x32x1x1xf32>
    %v242 = stablehlo.convolution(%v239, %v241)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v244 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v245 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v246 = stablehlo.compare GT, %b4dn, %v244 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v247 = stablehlo.compare LT, %b4dn, %v245 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v248 = stablehlo.and %v246, %v247 : tensor<32x100352xi1>
    %v249 = stablehlo.select %v248, %v243, %v244 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %b4ddndyi = stablehlo.reshape %v249 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4ddndxh = stablehlo.multiply %b4dngb, %b4ddndyi : tensor<32x128x28x28xf32>
    %b4ddnsdxr = stablehlo.reduce(%b4ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4ddnsdx = stablehlo.broadcast_in_dim %b4ddnsdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4ddnxd = stablehlo.multiply %b4dnxh, %b4ddndxh : tensor<32x128x28x28xf32>
    %b4ddnsxdr = stablehlo.reduce(%b4ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4ddnsxd = stablehlo.broadcast_in_dim %b4ddnsxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4ddnt1 = stablehlo.multiply %b4ddndxh, %b4dnnf : tensor<32x128x28x28xf32>
    %b4ddni1 = stablehlo.subtract %b4ddnt1, %b4ddnsdx : tensor<32x128x28x28xf32>
    %b4ddnxs = stablehlo.multiply %b4dnxh, %b4ddnsxd : tensor<32x128x28x28xf32>
    %b4ddni2 = stablehlo.subtract %b4ddni1, %b4ddnxs : tensor<32x128x28x28xf32>
    %b4ddnsN = stablehlo.divide %b4dnistd, %b4dnnf : tensor<32x128x28x28xf32>
    %b4ddndxn = stablehlo.multiply %b4ddnsN, %b4ddni2 : tensor<32x128x28x28xf32>
    %b4ddn = stablehlo.reshape %b4ddndxn : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %b4ddndgp = stablehlo.multiply %b4ddndyi, %b4dnxh : tensor<32x128x28x28xf32>
    %b4ddndg = stablehlo.reduce(%b4ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4ddndb = stablehlo.reduce(%b4ddndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v250 = stablehlo.reshape %b4ddn : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v251 = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<128x1x3x3xf32>
    %v252 = stablehlo.convolution(%v250, %v251)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v254 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v255 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v256 = stablehlo.compare GT, %b4en, %v254 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v257 = stablehlo.compare LT, %b4en, %v255 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v258 = stablehlo.and %v256, %v257 : tensor<32x100352xi1>
    %v259 = stablehlo.select %v258, %v253, %v254 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %b4dendyi = stablehlo.reshape %v259 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4dendxh = stablehlo.multiply %b4engb, %b4dendyi : tensor<32x128x28x28xf32>
    %b4densdxr = stablehlo.reduce(%b4dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4densdx = stablehlo.broadcast_in_dim %b4densdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4denxd = stablehlo.multiply %b4enxh, %b4dendxh : tensor<32x128x28x28xf32>
    %b4densxdr = stablehlo.reduce(%b4denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4densxd = stablehlo.broadcast_in_dim %b4densxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %b4dent1 = stablehlo.multiply %b4dendxh, %b4ennf : tensor<32x128x28x28xf32>
    %b4deni1 = stablehlo.subtract %b4dent1, %b4densdx : tensor<32x128x28x28xf32>
    %b4denxs = stablehlo.multiply %b4enxh, %b4densxd : tensor<32x128x28x28xf32>
    %b4deni2 = stablehlo.subtract %b4deni1, %b4denxs : tensor<32x128x28x28xf32>
    %b4densN = stablehlo.divide %b4enistd, %b4ennf : tensor<32x128x28x28xf32>
    %b4dendxn = stablehlo.multiply %b4densN, %b4deni2 : tensor<32x128x28x28xf32>
    %b4den = stablehlo.reshape %b4dendxn : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %b4dendgp = stablehlo.multiply %b4dendyi, %b4enxh : tensor<32x128x28x28xf32>
    %b4dendg = stablehlo.reduce(%b4dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4dendb = stablehlo.reduce(%b4dendyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v260 = stablehlo.reshape %b4den : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v261 = stablehlo.transpose %b4eW, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v262 = stablehlo.reverse %v261, dims = [2, 3] : tensor<32x128x1x1xf32>
    %v263 = stablehlo.convolution(%v260, %v262)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v265 = stablehlo.add %v264, %v238 : tensor<32x25088xf32>
    %b3dpndyi = stablehlo.reshape %v265 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b3dpndxh = stablehlo.multiply %b3pngb, %b3dpndyi : tensor<32x32x28x28xf32>
    %b3dpnsdxr = stablehlo.reduce(%b3dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b3dpnsdx = stablehlo.broadcast_in_dim %b3dpnsdxr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b3dpnxd = stablehlo.multiply %b3pnxh, %b3dpndxh : tensor<32x32x28x28xf32>
    %b3dpnsxdr = stablehlo.reduce(%b3dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b3dpnsxd = stablehlo.broadcast_in_dim %b3dpnsxdr, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %b3dpnt1 = stablehlo.multiply %b3dpndxh, %b3pnnf : tensor<32x32x28x28xf32>
    %b3dpni1 = stablehlo.subtract %b3dpnt1, %b3dpnsdx : tensor<32x32x28x28xf32>
    %b3dpnxs = stablehlo.multiply %b3pnxh, %b3dpnsxd : tensor<32x32x28x28xf32>
    %b3dpni2 = stablehlo.subtract %b3dpni1, %b3dpnxs : tensor<32x32x28x28xf32>
    %b3dpnsN = stablehlo.divide %b3pnistd, %b3pnnf : tensor<32x32x28x28xf32>
    %b3dpndxn = stablehlo.multiply %b3dpnsN, %b3dpni2 : tensor<32x32x28x28xf32>
    %b3dpn = stablehlo.reshape %b3dpndxn : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %b3dpndgp = stablehlo.multiply %b3dpndyi, %b3pnxh : tensor<32x32x28x28xf32>
    %b3dpndg = stablehlo.reduce(%b3dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b3dpndb = stablehlo.reduce(%b3dpndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v266 = stablehlo.reshape %b3dpn : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v267 = stablehlo.transpose %b3pW, dims = [1, 0, 2, 3] : (tensor<32x96x1x1xf32>) -> tensor<96x32x1x1xf32>
    %v268 = stablehlo.reverse %v267, dims = [2, 3] : tensor<96x32x1x1xf32>
    %v269 = stablehlo.convolution(%v266, %v268)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<96x32x1x1xf32>) -> tensor<32x96x28x28xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v271 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v272 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v273 = stablehlo.compare GT, %b3dn, %v271 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v274 = stablehlo.compare LT, %b3dn, %v272 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v275 = stablehlo.and %v273, %v274 : tensor<32x75264xi1>
    %v276 = stablehlo.select %v275, %v270, %v271 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %b3ddndyi = stablehlo.reshape %v276 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %b3ddndxh = stablehlo.multiply %b3dngb, %b3ddndyi : tensor<32x96x28x28xf32>
    %b3ddnsdxr = stablehlo.reduce(%b3ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %b3ddnsdx = stablehlo.broadcast_in_dim %b3ddnsdxr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %b3ddnxd = stablehlo.multiply %b3dnxh, %b3ddndxh : tensor<32x96x28x28xf32>
    %b3ddnsxdr = stablehlo.reduce(%b3ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %b3ddnsxd = stablehlo.broadcast_in_dim %b3ddnsxdr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %b3ddnt1 = stablehlo.multiply %b3ddndxh, %b3dnnf : tensor<32x96x28x28xf32>
    %b3ddni1 = stablehlo.subtract %b3ddnt1, %b3ddnsdx : tensor<32x96x28x28xf32>
    %b3ddnxs = stablehlo.multiply %b3dnxh, %b3ddnsxd : tensor<32x96x28x28xf32>
    %b3ddni2 = stablehlo.subtract %b3ddni1, %b3ddnxs : tensor<32x96x28x28xf32>
    %b3ddnsN = stablehlo.divide %b3dnistd, %b3dnnf : tensor<32x96x28x28xf32>
    %b3ddndxn = stablehlo.multiply %b3ddnsN, %b3ddni2 : tensor<32x96x28x28xf32>
    %b3ddn = stablehlo.reshape %b3ddndxn : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %b3ddndgp = stablehlo.multiply %b3ddndyi, %b3dnxh : tensor<32x96x28x28xf32>
    %b3ddndg = stablehlo.reduce(%b3ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %b3ddndb = stablehlo.reduce(%b3ddndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %v277 = stablehlo.reshape %b3ddn : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v279 = stablehlo.pad %v277, %v278, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96x56x56xf32>
    %v280 = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v281 = stablehlo.convolution(%v279, %v280)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v283 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v284 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v285 = stablehlo.compare GT, %b3en, %v283 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v286 = stablehlo.compare LT, %b3en, %v284 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v287 = stablehlo.and %v285, %v286 : tensor<32x301056xi1>
    %v288 = stablehlo.select %v287, %v282, %v283 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %b3dendyi = stablehlo.reshape %v288 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b3dendxh = stablehlo.multiply %b3engb, %b3dendyi : tensor<32x96x56x56xf32>
    %b3densdxr = stablehlo.reduce(%b3dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b3densdx = stablehlo.broadcast_in_dim %b3densdxr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b3denxd = stablehlo.multiply %b3enxh, %b3dendxh : tensor<32x96x56x56xf32>
    %b3densxdr = stablehlo.reduce(%b3denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b3densxd = stablehlo.broadcast_in_dim %b3densxdr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b3dent1 = stablehlo.multiply %b3dendxh, %b3ennf : tensor<32x96x56x56xf32>
    %b3deni1 = stablehlo.subtract %b3dent1, %b3densdx : tensor<32x96x56x56xf32>
    %b3denxs = stablehlo.multiply %b3enxh, %b3densxd : tensor<32x96x56x56xf32>
    %b3deni2 = stablehlo.subtract %b3deni1, %b3denxs : tensor<32x96x56x56xf32>
    %b3densN = stablehlo.divide %b3enistd, %b3ennf : tensor<32x96x56x56xf32>
    %b3dendxn = stablehlo.multiply %b3densN, %b3deni2 : tensor<32x96x56x56xf32>
    %b3den = stablehlo.reshape %b3dendxn : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %b3dendgp = stablehlo.multiply %b3dendyi, %b3enxh : tensor<32x96x56x56xf32>
    %b3dendg = stablehlo.reduce(%b3dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b3dendb = stablehlo.reduce(%b3dendyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v289 = stablehlo.reshape %b3den : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v290 = stablehlo.transpose %b3eW, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v291 = stablehlo.reverse %v290, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v292 = stablehlo.convolution(%v289, %v291)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %b2dpndyi = stablehlo.reshape %v293 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b2dpndxh = stablehlo.multiply %b2pngb, %b2dpndyi : tensor<32x24x56x56xf32>
    %b2dpnsdxr = stablehlo.reduce(%b2dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpnsdx = stablehlo.broadcast_in_dim %b2dpnsdxr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2dpnxd = stablehlo.multiply %b2pnxh, %b2dpndxh : tensor<32x24x56x56xf32>
    %b2dpnsxdr = stablehlo.reduce(%b2dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpnsxd = stablehlo.broadcast_in_dim %b2dpnsxdr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b2dpnt1 = stablehlo.multiply %b2dpndxh, %b2pnnf : tensor<32x24x56x56xf32>
    %b2dpni1 = stablehlo.subtract %b2dpnt1, %b2dpnsdx : tensor<32x24x56x56xf32>
    %b2dpnxs = stablehlo.multiply %b2pnxh, %b2dpnsxd : tensor<32x24x56x56xf32>
    %b2dpni2 = stablehlo.subtract %b2dpni1, %b2dpnxs : tensor<32x24x56x56xf32>
    %b2dpnsN = stablehlo.divide %b2pnistd, %b2pnnf : tensor<32x24x56x56xf32>
    %b2dpndxn = stablehlo.multiply %b2dpnsN, %b2dpni2 : tensor<32x24x56x56xf32>
    %b2dpn = stablehlo.reshape %b2dpndxn : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %b2dpndgp = stablehlo.multiply %b2dpndyi, %b2pnxh : tensor<32x24x56x56xf32>
    %b2dpndg = stablehlo.reduce(%b2dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2dpndb = stablehlo.reduce(%b2dpndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v294 = stablehlo.reshape %b2dpn : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v295 = stablehlo.transpose %b2pW, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v296 = stablehlo.reverse %v295, dims = [2, 3] : tensor<96x24x1x1xf32>
    %v297 = stablehlo.convolution(%v294, %v296)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v299 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v300 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v301 = stablehlo.compare GT, %b2dn, %v299 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v302 = stablehlo.compare LT, %b2dn, %v300 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v303 = stablehlo.and %v301, %v302 : tensor<32x301056xi1>
    %v304 = stablehlo.select %v303, %v298, %v299 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %b2ddndyi = stablehlo.reshape %v304 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2ddndxh = stablehlo.multiply %b2dngb, %b2ddndyi : tensor<32x96x56x56xf32>
    %b2ddnsdxr = stablehlo.reduce(%b2ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddnsdx = stablehlo.broadcast_in_dim %b2ddnsdxr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2ddnxd = stablehlo.multiply %b2dnxh, %b2ddndxh : tensor<32x96x56x56xf32>
    %b2ddnsxdr = stablehlo.reduce(%b2ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddnsxd = stablehlo.broadcast_in_dim %b2ddnsxdr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2ddnt1 = stablehlo.multiply %b2ddndxh, %b2dnnf : tensor<32x96x56x56xf32>
    %b2ddni1 = stablehlo.subtract %b2ddnt1, %b2ddnsdx : tensor<32x96x56x56xf32>
    %b2ddnxs = stablehlo.multiply %b2dnxh, %b2ddnsxd : tensor<32x96x56x56xf32>
    %b2ddni2 = stablehlo.subtract %b2ddni1, %b2ddnxs : tensor<32x96x56x56xf32>
    %b2ddnsN = stablehlo.divide %b2dnistd, %b2dnnf : tensor<32x96x56x56xf32>
    %b2ddndxn = stablehlo.multiply %b2ddnsN, %b2ddni2 : tensor<32x96x56x56xf32>
    %b2ddn = stablehlo.reshape %b2ddndxn : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %b2ddndgp = stablehlo.multiply %b2ddndyi, %b2dnxh : tensor<32x96x56x56xf32>
    %b2ddndg = stablehlo.reduce(%b2ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2ddndb = stablehlo.reduce(%b2ddndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v305 = stablehlo.reshape %b2ddn : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v306 = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v307 = stablehlo.convolution(%v305, %v306)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v309 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v310 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v311 = stablehlo.compare GT, %b2en, %v309 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v312 = stablehlo.compare LT, %b2en, %v310 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v313 = stablehlo.and %v311, %v312 : tensor<32x301056xi1>
    %v314 = stablehlo.select %v313, %v308, %v309 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %b2dendyi = stablehlo.reshape %v314 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2dendxh = stablehlo.multiply %b2engb, %b2dendyi : tensor<32x96x56x56xf32>
    %b2densdxr = stablehlo.reduce(%b2dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2densdx = stablehlo.broadcast_in_dim %b2densdxr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2denxd = stablehlo.multiply %b2enxh, %b2dendxh : tensor<32x96x56x56xf32>
    %b2densxdr = stablehlo.reduce(%b2denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2densxd = stablehlo.broadcast_in_dim %b2densxdr, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %b2dent1 = stablehlo.multiply %b2dendxh, %b2ennf : tensor<32x96x56x56xf32>
    %b2deni1 = stablehlo.subtract %b2dent1, %b2densdx : tensor<32x96x56x56xf32>
    %b2denxs = stablehlo.multiply %b2enxh, %b2densxd : tensor<32x96x56x56xf32>
    %b2deni2 = stablehlo.subtract %b2deni1, %b2denxs : tensor<32x96x56x56xf32>
    %b2densN = stablehlo.divide %b2enistd, %b2ennf : tensor<32x96x56x56xf32>
    %b2dendxn = stablehlo.multiply %b2densN, %b2deni2 : tensor<32x96x56x56xf32>
    %b2den = stablehlo.reshape %b2dendxn : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %b2dendgp = stablehlo.multiply %b2dendyi, %b2enxh : tensor<32x96x56x56xf32>
    %b2dendg = stablehlo.reduce(%b2dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dendb = stablehlo.reduce(%b2dendyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v315 = stablehlo.reshape %b2den : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v316 = stablehlo.transpose %b2eW, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v317 = stablehlo.reverse %v316, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v318 = stablehlo.convolution(%v315, %v317)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v320 = stablehlo.add %v319, %v293 : tensor<32x75264xf32>
    %b1dpndyi = stablehlo.reshape %v320 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b1dpndxh = stablehlo.multiply %b1pngb, %b1dpndyi : tensor<32x24x56x56xf32>
    %b1dpnsdxr = stablehlo.reduce(%b1dpndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b1dpnsdx = stablehlo.broadcast_in_dim %b1dpnsdxr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b1dpnxd = stablehlo.multiply %b1pnxh, %b1dpndxh : tensor<32x24x56x56xf32>
    %b1dpnsxdr = stablehlo.reduce(%b1dpnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b1dpnsxd = stablehlo.broadcast_in_dim %b1dpnsxdr, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %b1dpnt1 = stablehlo.multiply %b1dpndxh, %b1pnnf : tensor<32x24x56x56xf32>
    %b1dpni1 = stablehlo.subtract %b1dpnt1, %b1dpnsdx : tensor<32x24x56x56xf32>
    %b1dpnxs = stablehlo.multiply %b1pnxh, %b1dpnsxd : tensor<32x24x56x56xf32>
    %b1dpni2 = stablehlo.subtract %b1dpni1, %b1dpnxs : tensor<32x24x56x56xf32>
    %b1dpnsN = stablehlo.divide %b1pnistd, %b1pnnf : tensor<32x24x56x56xf32>
    %b1dpndxn = stablehlo.multiply %b1dpnsN, %b1dpni2 : tensor<32x24x56x56xf32>
    %b1dpn = stablehlo.reshape %b1dpndxn : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %b1dpndgp = stablehlo.multiply %b1dpndyi, %b1pnxh : tensor<32x24x56x56xf32>
    %b1dpndg = stablehlo.reduce(%b1dpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b1dpndb = stablehlo.reduce(%b1dpndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v321 = stablehlo.reshape %b1dpn : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v322 = stablehlo.transpose %b1pW, dims = [1, 0, 2, 3] : (tensor<24x64x1x1xf32>) -> tensor<64x24x1x1xf32>
    %v323 = stablehlo.reverse %v322, dims = [2, 3] : tensor<64x24x1x1xf32>
    %v324 = stablehlo.convolution(%v321, %v323)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<64x24x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v325 = stablehlo.reshape %v324 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v326 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v327 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v328 = stablehlo.compare GT, %b1dn, %v326 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v329 = stablehlo.compare LT, %b1dn, %v327 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v330 = stablehlo.and %v328, %v329 : tensor<32x200704xi1>
    %v331 = stablehlo.select %v330, %v325, %v326 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %b1ddndyi = stablehlo.reshape %v331 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %b1ddndxh = stablehlo.multiply %b1dngb, %b1ddndyi : tensor<32x64x56x56xf32>
    %b1ddnsdxr = stablehlo.reduce(%b1ddndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %b1ddnsdx = stablehlo.broadcast_in_dim %b1ddnsdxr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %b1ddnxd = stablehlo.multiply %b1dnxh, %b1ddndxh : tensor<32x64x56x56xf32>
    %b1ddnsxdr = stablehlo.reduce(%b1ddnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %b1ddnsxd = stablehlo.broadcast_in_dim %b1ddnsxdr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %b1ddnt1 = stablehlo.multiply %b1ddndxh, %b1dnnf : tensor<32x64x56x56xf32>
    %b1ddni1 = stablehlo.subtract %b1ddnt1, %b1ddnsdx : tensor<32x64x56x56xf32>
    %b1ddnxs = stablehlo.multiply %b1dnxh, %b1ddnsxd : tensor<32x64x56x56xf32>
    %b1ddni2 = stablehlo.subtract %b1ddni1, %b1ddnxs : tensor<32x64x56x56xf32>
    %b1ddnsN = stablehlo.divide %b1dnistd, %b1dnnf : tensor<32x64x56x56xf32>
    %b1ddndxn = stablehlo.multiply %b1ddnsN, %b1ddni2 : tensor<32x64x56x56xf32>
    %b1ddn = stablehlo.reshape %b1ddndxn : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %b1ddndgp = stablehlo.multiply %b1ddndyi, %b1dnxh : tensor<32x64x56x56xf32>
    %b1ddndg = stablehlo.reduce(%b1ddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %b1ddndb = stablehlo.reduce(%b1ddndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v332 = stablehlo.reshape %b1ddn : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v334 = stablehlo.pad %v332, %v333, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v335 = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<64x1x3x3xf32>
    %v336 = stablehlo.convolution(%v334, %v335)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<32x64x112x112xf32>, tensor<64x1x3x3xf32>) -> tensor<32x64x112x112xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v338 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v339 = stablehlo.constant dense<6.0> : tensor<32x802816xf32>
    %v340 = stablehlo.compare GT, %b1en, %v338 : (tensor<32x802816xf32>, tensor<32x802816xf32>) -> tensor<32x802816xi1>
    %v341 = stablehlo.compare LT, %b1en, %v339 : (tensor<32x802816xf32>, tensor<32x802816xf32>) -> tensor<32x802816xi1>
    %v342 = stablehlo.and %v340, %v341 : tensor<32x802816xi1>
    %v343 = stablehlo.select %v342, %v337, %v338 : tensor<32x802816xi1>, tensor<32x802816xf32>
    %b1dendyi = stablehlo.reshape %v343 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %b1dendxh = stablehlo.multiply %b1engb, %b1dendyi : tensor<32x64x112x112xf32>
    %b1densdxr = stablehlo.reduce(%b1dendxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %b1densdx = stablehlo.broadcast_in_dim %b1densdxr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %b1denxd = stablehlo.multiply %b1enxh, %b1dendxh : tensor<32x64x112x112xf32>
    %b1densxdr = stablehlo.reduce(%b1denxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %b1densxd = stablehlo.broadcast_in_dim %b1densxdr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %b1dent1 = stablehlo.multiply %b1dendxh, %b1ennf : tensor<32x64x112x112xf32>
    %b1deni1 = stablehlo.subtract %b1dent1, %b1densdx : tensor<32x64x112x112xf32>
    %b1denxs = stablehlo.multiply %b1enxh, %b1densxd : tensor<32x64x112x112xf32>
    %b1deni2 = stablehlo.subtract %b1deni1, %b1denxs : tensor<32x64x112x112xf32>
    %b1densN = stablehlo.divide %b1enistd, %b1ennf : tensor<32x64x112x112xf32>
    %b1dendxn = stablehlo.multiply %b1densN, %b1deni2 : tensor<32x64x112x112xf32>
    %b1den = stablehlo.reshape %b1dendxn : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %b1dendgp = stablehlo.multiply %b1dendyi, %b1enxh : tensor<32x64x112x112xf32>
    %b1dendg = stablehlo.reduce(%b1dendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %b1dendb = stablehlo.reduce(%b1dendyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v344 = stablehlo.reshape %b1den : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v345 = stablehlo.transpose %b1eW, dims = [1, 0, 2, 3] : (tensor<64x16x1x1xf32>) -> tensor<16x64x1x1xf32>
    %v346 = stablehlo.reverse %v345, dims = [2, 3] : tensor<16x64x1x1xf32>
    %v347 = stablehlo.convolution(%v344, %v346)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<16x64x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v348 = stablehlo.reshape %v347 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v349 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v350 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v351 = stablehlo.compare GT, %stn, %v349 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v352 = stablehlo.compare LT, %stn, %v350 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v353 = stablehlo.and %v351, %v352 : tensor<32x200704xi1>
    %v354 = stablehlo.select %v353, %v348, %v349 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %dstndyi = stablehlo.reshape %v354 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %dstndxh = stablehlo.multiply %stngb, %dstndyi : tensor<32x16x112x112xf32>
    %dstnsdxr = stablehlo.reduce(%dstndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %dstnsdx = stablehlo.broadcast_in_dim %dstnsdxr, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %dstnxd = stablehlo.multiply %stnxh, %dstndxh : tensor<32x16x112x112xf32>
    %dstnsxdr = stablehlo.reduce(%dstnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %dstnsxd = stablehlo.broadcast_in_dim %dstnsxdr, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %dstnt1 = stablehlo.multiply %dstndxh, %stnnf : tensor<32x16x112x112xf32>
    %dstni1 = stablehlo.subtract %dstnt1, %dstnsdx : tensor<32x16x112x112xf32>
    %dstnxs = stablehlo.multiply %stnxh, %dstnsxd : tensor<32x16x112x112xf32>
    %dstni2 = stablehlo.subtract %dstni1, %dstnxs : tensor<32x16x112x112xf32>
    %dstnsN = stablehlo.divide %stnistd, %stnnf : tensor<32x16x112x112xf32>
    %dstndxn = stablehlo.multiply %dstnsN, %dstni2 : tensor<32x16x112x112xf32>
    %dstn = stablehlo.reshape %dstndxn : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %dstndgp = stablehlo.multiply %dstndyi, %stnxh : tensor<32x16x112x112xf32>
    %dstndg = stablehlo.reduce(%dstndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %dstndb = stablehlo.reduce(%dstndyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %b6dpWxi = stablehlo.reshape %v143 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %b6dpWdi = stablehlo.reshape %b6dpn : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %b6dpWxt = stablehlo.transpose %b6dpWxi, dims = [1, 0, 2, 3] : (tensor<32x256x7x7xf32>) -> tensor<256x32x7x7xf32>
    %b6dpWdt = stablehlo.transpose %b6dpWdi, dims = [1, 0, 2, 3] : (tensor<32x64x7x7xf32>) -> tensor<64x32x7x7xf32>
    %b6dpWraw = stablehlo.convolution(%b6dpWxt, %b6dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x7x7xf32>, tensor<64x32x7x7xf32>) -> tensor<256x64x1x1xf32>
    %b6dpW = stablehlo.transpose %b6dpWraw, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %b6dpbi = stablehlo.reshape %b6dpn : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %b6dpb = stablehlo.reduce(%b6dpbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %b6ddui = stablehlo.reshape %b6ddn : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %b6ddup = stablehlo.pad %b6ddui, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256x14x14xf32>
    %b6ddu = stablehlo.reshape %b6ddup : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %b6ddWxi = stablehlo.reshape %v134 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %b6ddWdi = stablehlo.reshape %b6ddu : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %b6ddWxt = stablehlo.transpose %b6ddWxi, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %b6ddWdt = stablehlo.transpose %b6ddWdi, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %b6ddWraw = stablehlo.convolution(%b6ddWxt, %b6ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 256 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<1x256x3x3xf32>
    %b6ddW = stablehlo.reshape %b6ddWraw : (tensor<1x256x3x3xf32>) -> tensor<256x1x3x3xf32>
    %b6ddbi = stablehlo.reshape %b6ddn : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %b6ddb = stablehlo.reduce(%b6ddbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %b6deWxi = stablehlo.reshape %b5pn : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %b6deWdi = stablehlo.reshape %b6den : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %b6deWxt = stablehlo.transpose %b6deWxi, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %b6deWdt = stablehlo.transpose %b6deWdi, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %b6deWraw = stablehlo.convolution(%b6deWxt, %b6deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<64x256x1x1xf32>
    %b6deW = stablehlo.transpose %b6deWraw, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %b6debi = stablehlo.reshape %b6den : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %b6deb = stablehlo.reduce(%b6debi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %b5dpWxi = stablehlo.reshape %v120 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %b5dpWdi = stablehlo.reshape %b5dpn : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %b5dpWxt = stablehlo.transpose %b5dpWxi, dims = [1, 0, 2, 3] : (tensor<32x128x14x14xf32>) -> tensor<128x32x14x14xf32>
    %b5dpWdt = stablehlo.transpose %b5dpWdi, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %b5dpWraw = stablehlo.convolution(%b5dpWxt, %b5dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<128x64x1x1xf32>
    %b5dpW = stablehlo.transpose %b5dpWraw, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %b5dpbi = stablehlo.reshape %b5dpn : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %b5dpb = stablehlo.reduce(%b5dpbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %b5ddui = stablehlo.reshape %b5ddn : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %b5ddup = stablehlo.pad %b5ddui, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128x28x28xf32>
    %b5ddu = stablehlo.reshape %b5ddup : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %b5ddWxi = stablehlo.reshape %v111 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b5ddWdi = stablehlo.reshape %b5ddu : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b5ddWxt = stablehlo.transpose %b5ddWxi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b5ddWdt = stablehlo.transpose %b5ddWdi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b5ddWraw = stablehlo.convolution(%b5ddWxt, %b5ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 128 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<1x128x3x3xf32>
    %b5ddW = stablehlo.reshape %b5ddWraw : (tensor<1x128x3x3xf32>) -> tensor<128x1x3x3xf32>
    %b5ddbi = stablehlo.reshape %b5ddn : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %b5ddb = stablehlo.reduce(%b5ddbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %b5deWxi = stablehlo.reshape %v102 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b5deWdi = stablehlo.reshape %b5den : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b5deWxt = stablehlo.transpose %b5deWxi, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %b5deWdt = stablehlo.transpose %b5deWdi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b5deWraw = stablehlo.convolution(%b5deWxt, %b5deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<32x128x1x1xf32>
    %b5deW = stablehlo.transpose %b5deWraw, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %b5debi = stablehlo.reshape %b5den : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b5deb = stablehlo.reduce(%b5debi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4dpWxi = stablehlo.reshape %v96 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4dpWdi = stablehlo.reshape %b4dpn : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b4dpWxt = stablehlo.transpose %b4dpWxi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b4dpWdt = stablehlo.transpose %b4dpWdi, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %b4dpWraw = stablehlo.convolution(%b4dpWxt, %b4dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<128x32x1x1xf32>
    %b4dpW = stablehlo.transpose %b4dpWraw, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %b4dpbi = stablehlo.reshape %b4dpn : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b4dpb = stablehlo.reduce(%b4dpbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b4ddWxi = stablehlo.reshape %v87 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4ddWdi = stablehlo.reshape %b4ddn : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4ddWxt = stablehlo.transpose %b4ddWxi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b4ddWdt = stablehlo.transpose %b4ddWdi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b4ddWraw = stablehlo.convolution(%b4ddWxt, %b4ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 128 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<1x128x3x3xf32>
    %b4ddW = stablehlo.reshape %b4ddWraw : (tensor<1x128x3x3xf32>) -> tensor<128x1x3x3xf32>
    %b4ddbi = stablehlo.reshape %b4ddn : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4ddb = stablehlo.reduce(%b4ddbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b4deWxi = stablehlo.reshape %b3pn : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b4deWdi = stablehlo.reshape %b4den : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4deWxt = stablehlo.transpose %b4deWxi, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %b4deWdt = stablehlo.transpose %b4deWdi, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %b4deWraw = stablehlo.convolution(%b4deWxt, %b4deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<32x128x1x1xf32>
    %b4deW = stablehlo.transpose %b4deWraw, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %b4debi = stablehlo.reshape %b4den : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %b4deb = stablehlo.reduce(%b4debi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %b3dpWxi = stablehlo.reshape %v73 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %b3dpWdi = stablehlo.reshape %b3dpn : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b3dpWxt = stablehlo.transpose %b3dpWxi, dims = [1, 0, 2, 3] : (tensor<32x96x28x28xf32>) -> tensor<96x32x28x28xf32>
    %b3dpWdt = stablehlo.transpose %b3dpWdi, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %b3dpWraw = stablehlo.convolution(%b3dpWxt, %b3dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<96x32x1x1xf32>
    %b3dpW = stablehlo.transpose %b3dpWraw, dims = [1, 0, 2, 3] : (tensor<96x32x1x1xf32>) -> tensor<32x96x1x1xf32>
    %b3dpbi = stablehlo.reshape %b3dpn : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %b3dpb = stablehlo.reduce(%b3dpbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %b3ddui = stablehlo.reshape %b3ddn : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %b3ddup = stablehlo.pad %b3ddui, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96x56x56xf32>
    %b3ddu = stablehlo.reshape %b3ddup : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %b3ddWxi = stablehlo.reshape %v64 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b3ddWdi = stablehlo.reshape %b3ddu : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b3ddWxt = stablehlo.transpose %b3ddWxi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b3ddWdt = stablehlo.transpose %b3ddWdi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b3ddWraw = stablehlo.convolution(%b3ddWxt, %b3ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x3x3xf32>
    %b3ddW = stablehlo.reshape %b3ddWraw : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %b3ddbi = stablehlo.reshape %b3ddn : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %b3ddb = stablehlo.reduce(%b3ddbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %b3deWxi = stablehlo.reshape %v55 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b3deWdi = stablehlo.reshape %b3den : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b3deWxt = stablehlo.transpose %b3deWxi, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b3deWdt = stablehlo.transpose %b3deWdi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b3deWraw = stablehlo.convolution(%b3deWxt, %b3deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<24x96x1x1xf32>
    %b3deW = stablehlo.transpose %b3deWraw, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %b3debi = stablehlo.reshape %b3den : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b3deb = stablehlo.reduce(%b3debi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2dpWxi = stablehlo.reshape %v49 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2dpWdi = stablehlo.reshape %b2dpn : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b2dpWxt = stablehlo.transpose %b2dpWxi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b2dpWdt = stablehlo.transpose %b2dpWdi, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b2dpWraw = stablehlo.convolution(%b2dpWxt, %b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %b2dpW = stablehlo.transpose %b2dpWraw, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %b2dpbi = stablehlo.reshape %b2dpn : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b2dpb = stablehlo.reduce(%b2dpbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b2ddWxi = stablehlo.reshape %v40 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2ddWdi = stablehlo.reshape %b2ddn : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2ddWxt = stablehlo.transpose %b2ddWxi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b2ddWdt = stablehlo.transpose %b2ddWdi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b2ddWraw = stablehlo.convolution(%b2ddWxt, %b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x3x3xf32>
    %b2ddW = stablehlo.reshape %b2ddWraw : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %b2ddbi = stablehlo.reshape %b2ddn : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2ddb = stablehlo.reduce(%b2ddbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b2deWxi = stablehlo.reshape %b1pn : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b2deWdi = stablehlo.reshape %b2den : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2deWxt = stablehlo.transpose %b2deWxi, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b2deWdt = stablehlo.transpose %b2deWdi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %b2deWraw = stablehlo.convolution(%b2deWxt, %b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<24x96x1x1xf32>
    %b2deW = stablehlo.transpose %b2deWraw, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %b2debi = stablehlo.reshape %b2den : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %b2deb = stablehlo.reduce(%b2debi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b1dpWxi = stablehlo.reshape %v26 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %b1dpWdi = stablehlo.reshape %b1dpn : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b1dpWxt = stablehlo.transpose %b1dpWxi, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %b1dpWdt = stablehlo.transpose %b1dpWdi, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %b1dpWraw = stablehlo.convolution(%b1dpWxt, %b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<64x24x1x1xf32>
    %b1dpW = stablehlo.transpose %b1dpWraw, dims = [1, 0, 2, 3] : (tensor<64x24x1x1xf32>) -> tensor<24x64x1x1xf32>
    %b1dpbi = stablehlo.reshape %b1dpn : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %b1dpb = stablehlo.reduce(%b1dpbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %b1ddui = stablehlo.reshape %b1ddn : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %b1ddup = stablehlo.pad %b1ddui, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %b1ddu = stablehlo.reshape %b1ddup : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %b1ddWxi = stablehlo.reshape %v17 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %b1ddWdi = stablehlo.reshape %b1ddu : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %b1ddWxt = stablehlo.transpose %b1ddWxi, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %b1ddWdt = stablehlo.transpose %b1ddWdi, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %b1ddWraw = stablehlo.convolution(%b1ddWxt, %b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 64 : i64, feature_group_count = 1 : i64} : (tensor<64x32x112x112xf32>, tensor<64x32x112x112xf32>) -> tensor<1x64x3x3xf32>
    %b1ddW = stablehlo.reshape %b1ddWraw : (tensor<1x64x3x3xf32>) -> tensor<64x1x3x3xf32>
    %b1ddbi = stablehlo.reshape %b1ddn : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %b1ddb = stablehlo.reduce(%b1ddbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %b1deWxi = stablehlo.reshape %v8 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %b1deWdi = stablehlo.reshape %b1den : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %b1deWxt = stablehlo.transpose %b1deWxi, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %b1deWdt = stablehlo.transpose %b1deWdi, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %b1deWraw = stablehlo.convolution(%b1deWxt, %b1deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<64x32x112x112xf32>) -> tensor<16x64x1x1xf32>
    %b1deW = stablehlo.transpose %b1deWraw, dims = [1, 0, 2, 3] : (tensor<16x64x1x1xf32>) -> tensor<64x16x1x1xf32>
    %b1debi = stablehlo.reshape %b1den : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %b1deb = stablehlo.reduce(%b1debi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %dhWxi = stablehlo.reshape %b6pn : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %dhWdi = stablehlo.reshape %dhn : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %dhWxt = stablehlo.transpose %dhWxi, dims = [1, 0, 2, 3] : (tensor<32x64x7x7xf32>) -> tensor<64x32x7x7xf32>
    %dhWdt = stablehlo.transpose %dhWdi, dims = [1, 0, 2, 3] : (tensor<32x128x7x7xf32>) -> tensor<128x32x7x7xf32>
    %dhWraw = stablehlo.convolution(%dhWxt, %dhWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x7x7xf32>, tensor<128x32x7x7xf32>) -> tensor<64x128x1x1xf32>
    %dhW = stablehlo.transpose %dhWraw, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %dhbi = stablehlo.reshape %dhn : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %dhb = stablehlo.reduce(%dhbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %dsui = stablehlo.reshape %dstn : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
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
    %dsbi = stablehlo.reshape %dstn : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %dsb = stablehlo.reduce(%dsbi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %dWd = stablehlo.dot_general %v162, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x128xf32>, tensor<32x10xf32>) -> tensor<128x10xf32>
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
    %admgsg = stablehlo.multiply %adob1sg, %dstndg : tensor<16xf32>
    %admnsg = stablehlo.add %admssg, %admgsg : tensor<16xf32>
    %adb2sg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2sg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advssg = stablehlo.multiply %adb2sg, %sgv : tensor<16xf32>
    %adg2sg = stablehlo.multiply %dstndg, %dstndg : tensor<16xf32>
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
    %admgsbt = stablehlo.multiply %adob1sbt, %dstndb : tensor<16xf32>
    %admnsbt = stablehlo.add %admssbt, %admgsbt : tensor<16xf32>
    %adb2sbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2sbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advssbt = stablehlo.multiply %adb2sbt, %sbtv : tensor<16xf32>
    %adg2sbt = stablehlo.multiply %dstndb, %dstndb : tensor<16xf32>
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
    %admgb1eg = stablehlo.multiply %adob1b1eg, %b1dendg : tensor<64xf32>
    %admnb1eg = stablehlo.add %admsb1eg, %admgb1eg : tensor<64xf32>
    %adb2b1eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b1eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb1eg = stablehlo.multiply %adb2b1eg, %b1egv : tensor<64xf32>
    %adg2b1eg = stablehlo.multiply %b1dendg, %b1dendg : tensor<64xf32>
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
    %admgb1ebt = stablehlo.multiply %adob1b1ebt, %b1dendb : tensor<64xf32>
    %admnb1ebt = stablehlo.add %admsb1ebt, %admgb1ebt : tensor<64xf32>
    %adb2b1ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b1ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb1ebt = stablehlo.multiply %adb2b1ebt, %b1ebtv : tensor<64xf32>
    %adg2b1ebt = stablehlo.multiply %b1dendb, %b1dendb : tensor<64xf32>
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
    %admgb1dg = stablehlo.multiply %adob1b1dg, %b1ddndg : tensor<64xf32>
    %admnb1dg = stablehlo.add %admsb1dg, %admgb1dg : tensor<64xf32>
    %adb2b1dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b1dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb1dg = stablehlo.multiply %adb2b1dg, %b1dgv : tensor<64xf32>
    %adg2b1dg = stablehlo.multiply %b1ddndg, %b1ddndg : tensor<64xf32>
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
    %admgb1dbt = stablehlo.multiply %adob1b1dbt, %b1ddndb : tensor<64xf32>
    %admnb1dbt = stablehlo.add %admsb1dbt, %admgb1dbt : tensor<64xf32>
    %adb2b1dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b1dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb1dbt = stablehlo.multiply %adb2b1dbt, %b1dbtv : tensor<64xf32>
    %adg2b1dbt = stablehlo.multiply %b1ddndb, %b1ddndb : tensor<64xf32>
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
    %admgb1pg = stablehlo.multiply %adob1b1pg, %b1dpndg : tensor<24xf32>
    %admnb1pg = stablehlo.add %admsb1pg, %admgb1pg : tensor<24xf32>
    %adb2b1pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b1pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb1pg = stablehlo.multiply %adb2b1pg, %b1pgv : tensor<24xf32>
    %adg2b1pg = stablehlo.multiply %b1dpndg, %b1dpndg : tensor<24xf32>
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
    %admgb1pbt = stablehlo.multiply %adob1b1pbt, %b1dpndb : tensor<24xf32>
    %admnb1pbt = stablehlo.add %admsb1pbt, %admgb1pbt : tensor<24xf32>
    %adb2b1pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b1pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb1pbt = stablehlo.multiply %adb2b1pbt, %b1pbtv : tensor<24xf32>
    %adg2b1pbt = stablehlo.multiply %b1dpndb, %b1dpndb : tensor<24xf32>
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
    %admgb2eg = stablehlo.multiply %adob1b2eg, %b2dendg : tensor<96xf32>
    %admnb2eg = stablehlo.add %admsb2eg, %admgb2eg : tensor<96xf32>
    %adb2b2eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2eg = stablehlo.multiply %adb2b2eg, %b2egv : tensor<96xf32>
    %adg2b2eg = stablehlo.multiply %b2dendg, %b2dendg : tensor<96xf32>
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
    %admgb2ebt = stablehlo.multiply %adob1b2ebt, %b2dendb : tensor<96xf32>
    %admnb2ebt = stablehlo.add %admsb2ebt, %admgb2ebt : tensor<96xf32>
    %adb2b2ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2ebt = stablehlo.multiply %adb2b2ebt, %b2ebtv : tensor<96xf32>
    %adg2b2ebt = stablehlo.multiply %b2dendb, %b2dendb : tensor<96xf32>
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
    %admgb2dg = stablehlo.multiply %adob1b2dg, %b2ddndg : tensor<96xf32>
    %admnb2dg = stablehlo.add %admsb2dg, %admgb2dg : tensor<96xf32>
    %adb2b2dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2dg = stablehlo.multiply %adb2b2dg, %b2dgv : tensor<96xf32>
    %adg2b2dg = stablehlo.multiply %b2ddndg, %b2ddndg : tensor<96xf32>
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
    %admgb2dbt = stablehlo.multiply %adob1b2dbt, %b2ddndb : tensor<96xf32>
    %admnb2dbt = stablehlo.add %admsb2dbt, %admgb2dbt : tensor<96xf32>
    %adb2b2dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b2dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb2dbt = stablehlo.multiply %adb2b2dbt, %b2dbtv : tensor<96xf32>
    %adg2b2dbt = stablehlo.multiply %b2ddndb, %b2ddndb : tensor<96xf32>
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
    %admgb2pg = stablehlo.multiply %adob1b2pg, %b2dpndg : tensor<24xf32>
    %admnb2pg = stablehlo.add %admsb2pg, %admgb2pg : tensor<24xf32>
    %adb2b2pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b2pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb2pg = stablehlo.multiply %adb2b2pg, %b2pgv : tensor<24xf32>
    %adg2b2pg = stablehlo.multiply %b2dpndg, %b2dpndg : tensor<24xf32>
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
    %admgb2pbt = stablehlo.multiply %adob1b2pbt, %b2dpndb : tensor<24xf32>
    %admnb2pbt = stablehlo.add %admsb2pbt, %admgb2pbt : tensor<24xf32>
    %adb2b2pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %adob2b2pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<24xf32>
    %advsb2pbt = stablehlo.multiply %adb2b2pbt, %b2pbtv : tensor<24xf32>
    %adg2b2pbt = stablehlo.multiply %b2dpndb, %b2dpndb : tensor<24xf32>
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
    %admgb3eg = stablehlo.multiply %adob1b3eg, %b3dendg : tensor<96xf32>
    %admnb3eg = stablehlo.add %admsb3eg, %admgb3eg : tensor<96xf32>
    %adb2b3eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b3eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb3eg = stablehlo.multiply %adb2b3eg, %b3egv : tensor<96xf32>
    %adg2b3eg = stablehlo.multiply %b3dendg, %b3dendg : tensor<96xf32>
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
    %admgb3ebt = stablehlo.multiply %adob1b3ebt, %b3dendb : tensor<96xf32>
    %admnb3ebt = stablehlo.add %admsb3ebt, %admgb3ebt : tensor<96xf32>
    %adb2b3ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b3ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb3ebt = stablehlo.multiply %adb2b3ebt, %b3ebtv : tensor<96xf32>
    %adg2b3ebt = stablehlo.multiply %b3dendb, %b3dendb : tensor<96xf32>
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
    %admgb3dg = stablehlo.multiply %adob1b3dg, %b3ddndg : tensor<96xf32>
    %admnb3dg = stablehlo.add %admsb3dg, %admgb3dg : tensor<96xf32>
    %adb2b3dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b3dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb3dg = stablehlo.multiply %adb2b3dg, %b3dgv : tensor<96xf32>
    %adg2b3dg = stablehlo.multiply %b3ddndg, %b3ddndg : tensor<96xf32>
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
    %admgb3dbt = stablehlo.multiply %adob1b3dbt, %b3ddndb : tensor<96xf32>
    %admnb3dbt = stablehlo.add %admsb3dbt, %admgb3dbt : tensor<96xf32>
    %adb2b3dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2b3dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advsb3dbt = stablehlo.multiply %adb2b3dbt, %b3dbtv : tensor<96xf32>
    %adg2b3dbt = stablehlo.multiply %b3ddndb, %b3ddndb : tensor<96xf32>
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
    %admgb3pg = stablehlo.multiply %adob1b3pg, %b3dpndg : tensor<32xf32>
    %admnb3pg = stablehlo.add %admsb3pg, %admgb3pg : tensor<32xf32>
    %adb2b3pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b3pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb3pg = stablehlo.multiply %adb2b3pg, %b3pgv : tensor<32xf32>
    %adg2b3pg = stablehlo.multiply %b3dpndg, %b3dpndg : tensor<32xf32>
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
    %admgb3pbt = stablehlo.multiply %adob1b3pbt, %b3dpndb : tensor<32xf32>
    %admnb3pbt = stablehlo.add %admsb3pbt, %admgb3pbt : tensor<32xf32>
    %adb2b3pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b3pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb3pbt = stablehlo.multiply %adb2b3pbt, %b3pbtv : tensor<32xf32>
    %adg2b3pbt = stablehlo.multiply %b3dpndb, %b3dpndb : tensor<32xf32>
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
    %admgb4eg = stablehlo.multiply %adob1b4eg, %b4dendg : tensor<128xf32>
    %admnb4eg = stablehlo.add %admsb4eg, %admgb4eg : tensor<128xf32>
    %adb2b4eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b4eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb4eg = stablehlo.multiply %adb2b4eg, %b4egv : tensor<128xf32>
    %adg2b4eg = stablehlo.multiply %b4dendg, %b4dendg : tensor<128xf32>
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
    %admgb4ebt = stablehlo.multiply %adob1b4ebt, %b4dendb : tensor<128xf32>
    %admnb4ebt = stablehlo.add %admsb4ebt, %admgb4ebt : tensor<128xf32>
    %adb2b4ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b4ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb4ebt = stablehlo.multiply %adb2b4ebt, %b4ebtv : tensor<128xf32>
    %adg2b4ebt = stablehlo.multiply %b4dendb, %b4dendb : tensor<128xf32>
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
    %admgb4dg = stablehlo.multiply %adob1b4dg, %b4ddndg : tensor<128xf32>
    %admnb4dg = stablehlo.add %admsb4dg, %admgb4dg : tensor<128xf32>
    %adb2b4dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b4dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb4dg = stablehlo.multiply %adb2b4dg, %b4dgv : tensor<128xf32>
    %adg2b4dg = stablehlo.multiply %b4ddndg, %b4ddndg : tensor<128xf32>
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
    %admgb4dbt = stablehlo.multiply %adob1b4dbt, %b4ddndb : tensor<128xf32>
    %admnb4dbt = stablehlo.add %admsb4dbt, %admgb4dbt : tensor<128xf32>
    %adb2b4dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b4dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb4dbt = stablehlo.multiply %adb2b4dbt, %b4dbtv : tensor<128xf32>
    %adg2b4dbt = stablehlo.multiply %b4ddndb, %b4ddndb : tensor<128xf32>
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
    %admgb4pg = stablehlo.multiply %adob1b4pg, %b4dpndg : tensor<32xf32>
    %admnb4pg = stablehlo.add %admsb4pg, %admgb4pg : tensor<32xf32>
    %adb2b4pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b4pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb4pg = stablehlo.multiply %adb2b4pg, %b4pgv : tensor<32xf32>
    %adg2b4pg = stablehlo.multiply %b4dpndg, %b4dpndg : tensor<32xf32>
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
    %admgb4pbt = stablehlo.multiply %adob1b4pbt, %b4dpndb : tensor<32xf32>
    %admnb4pbt = stablehlo.add %admsb4pbt, %admgb4pbt : tensor<32xf32>
    %adb2b4pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2b4pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsb4pbt = stablehlo.multiply %adb2b4pbt, %b4pbtv : tensor<32xf32>
    %adg2b4pbt = stablehlo.multiply %b4dpndb, %b4dpndb : tensor<32xf32>
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
    %admgb5eg = stablehlo.multiply %adob1b5eg, %b5dendg : tensor<128xf32>
    %admnb5eg = stablehlo.add %admsb5eg, %admgb5eg : tensor<128xf32>
    %adb2b5eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b5eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb5eg = stablehlo.multiply %adb2b5eg, %b5egv : tensor<128xf32>
    %adg2b5eg = stablehlo.multiply %b5dendg, %b5dendg : tensor<128xf32>
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
    %admgb5ebt = stablehlo.multiply %adob1b5ebt, %b5dendb : tensor<128xf32>
    %admnb5ebt = stablehlo.add %admsb5ebt, %admgb5ebt : tensor<128xf32>
    %adb2b5ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b5ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb5ebt = stablehlo.multiply %adb2b5ebt, %b5ebtv : tensor<128xf32>
    %adg2b5ebt = stablehlo.multiply %b5dendb, %b5dendb : tensor<128xf32>
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
    %admgb5dg = stablehlo.multiply %adob1b5dg, %b5ddndg : tensor<128xf32>
    %admnb5dg = stablehlo.add %admsb5dg, %admgb5dg : tensor<128xf32>
    %adb2b5dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b5dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb5dg = stablehlo.multiply %adb2b5dg, %b5dgv : tensor<128xf32>
    %adg2b5dg = stablehlo.multiply %b5ddndg, %b5ddndg : tensor<128xf32>
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
    %admgb5dbt = stablehlo.multiply %adob1b5dbt, %b5ddndb : tensor<128xf32>
    %admnb5dbt = stablehlo.add %admsb5dbt, %admgb5dbt : tensor<128xf32>
    %adb2b5dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2b5dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsb5dbt = stablehlo.multiply %adb2b5dbt, %b5dbtv : tensor<128xf32>
    %adg2b5dbt = stablehlo.multiply %b5ddndb, %b5ddndb : tensor<128xf32>
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
    %admgb5pg = stablehlo.multiply %adob1b5pg, %b5dpndg : tensor<64xf32>
    %admnb5pg = stablehlo.add %admsb5pg, %admgb5pg : tensor<64xf32>
    %adb2b5pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b5pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb5pg = stablehlo.multiply %adb2b5pg, %b5pgv : tensor<64xf32>
    %adg2b5pg = stablehlo.multiply %b5dpndg, %b5dpndg : tensor<64xf32>
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
    %admgb5pbt = stablehlo.multiply %adob1b5pbt, %b5dpndb : tensor<64xf32>
    %admnb5pbt = stablehlo.add %admsb5pbt, %admgb5pbt : tensor<64xf32>
    %adb2b5pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b5pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb5pbt = stablehlo.multiply %adb2b5pbt, %b5pbtv : tensor<64xf32>
    %adg2b5pbt = stablehlo.multiply %b5dpndb, %b5dpndb : tensor<64xf32>
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
    %admgb6eg = stablehlo.multiply %adob1b6eg, %b6dendg : tensor<256xf32>
    %admnb6eg = stablehlo.add %admsb6eg, %admgb6eg : tensor<256xf32>
    %adb2b6eg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2b6eg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsb6eg = stablehlo.multiply %adb2b6eg, %b6egv : tensor<256xf32>
    %adg2b6eg = stablehlo.multiply %b6dendg, %b6dendg : tensor<256xf32>
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
    %admgb6ebt = stablehlo.multiply %adob1b6ebt, %b6dendb : tensor<256xf32>
    %admnb6ebt = stablehlo.add %admsb6ebt, %admgb6ebt : tensor<256xf32>
    %adb2b6ebt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2b6ebt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsb6ebt = stablehlo.multiply %adb2b6ebt, %b6ebtv : tensor<256xf32>
    %adg2b6ebt = stablehlo.multiply %b6dendb, %b6dendb : tensor<256xf32>
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
    %admgb6dg = stablehlo.multiply %adob1b6dg, %b6ddndg : tensor<256xf32>
    %admnb6dg = stablehlo.add %admsb6dg, %admgb6dg : tensor<256xf32>
    %adb2b6dg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2b6dg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsb6dg = stablehlo.multiply %adb2b6dg, %b6dgv : tensor<256xf32>
    %adg2b6dg = stablehlo.multiply %b6ddndg, %b6ddndg : tensor<256xf32>
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
    %admgb6dbt = stablehlo.multiply %adob1b6dbt, %b6ddndb : tensor<256xf32>
    %admnb6dbt = stablehlo.add %admsb6dbt, %admgb6dbt : tensor<256xf32>
    %adb2b6dbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2b6dbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsb6dbt = stablehlo.multiply %adb2b6dbt, %b6dbtv : tensor<256xf32>
    %adg2b6dbt = stablehlo.multiply %b6ddndb, %b6ddndb : tensor<256xf32>
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
    %admgb6pg = stablehlo.multiply %adob1b6pg, %b6dpndg : tensor<64xf32>
    %admnb6pg = stablehlo.add %admsb6pg, %admgb6pg : tensor<64xf32>
    %adb2b6pg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b6pg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb6pg = stablehlo.multiply %adb2b6pg, %b6pgv : tensor<64xf32>
    %adg2b6pg = stablehlo.multiply %b6dpndg, %b6dpndg : tensor<64xf32>
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
    %admgb6pbt = stablehlo.multiply %adob1b6pbt, %b6dpndb : tensor<64xf32>
    %admnb6pbt = stablehlo.add %admsb6pbt, %admgb6pbt : tensor<64xf32>
    %adb2b6pbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b6pbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb6pbt = stablehlo.multiply %adb2b6pbt, %b6pbtv : tensor<64xf32>
    %adg2b6pbt = stablehlo.multiply %b6dpndb, %b6dpndb : tensor<64xf32>
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
    %admghg = stablehlo.multiply %adob1hg, %dhndg : tensor<128xf32>
    %admnhg = stablehlo.add %admshg, %admghg : tensor<128xf32>
    %adb2hg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2hg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advshg = stablehlo.multiply %adb2hg, %hgv : tensor<128xf32>
    %adg2hg = stablehlo.multiply %dhndg, %dhndg : tensor<128xf32>
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
    %admghbt = stablehlo.multiply %adob1hbt, %dhndb : tensor<128xf32>
    %admnhbt = stablehlo.add %admshbt, %admghbt : tensor<128xf32>
    %adb2hbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2hbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advshbt = stablehlo.multiply %adb2hbt, %hbtv : tensor<128xf32>
    %adg2hbt = stablehlo.multiply %dhndb, %dhndb : tensor<128xf32>
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
    %stnbnnf = stablehlo.constant dense<401408.0> : tensor<16xf32>
    %stnbnmu = stablehlo.divide %stnsmr, %stnbnnf : tensor<16xf32>
    %stnbnvar = stablehlo.divide %stnvsr, %stnbnnf : tensor<16xf32>
    %b1enbnnf = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %b1enbnmu = stablehlo.divide %b1ensmr, %b1enbnnf : tensor<64xf32>
    %b1enbnvar = stablehlo.divide %b1envsr, %b1enbnnf : tensor<64xf32>
    %b1dnbnnf = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %b1dnbnmu = stablehlo.divide %b1dnsmr, %b1dnbnnf : tensor<64xf32>
    %b1dnbnvar = stablehlo.divide %b1dnvsr, %b1dnbnnf : tensor<64xf32>
    %b1pnbnnf = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %b1pnbnmu = stablehlo.divide %b1pnsmr, %b1pnbnnf : tensor<24xf32>
    %b1pnbnvar = stablehlo.divide %b1pnvsr, %b1pnbnnf : tensor<24xf32>
    %b2enbnnf = stablehlo.constant dense<100352.0> : tensor<96xf32>
    %b2enbnmu = stablehlo.divide %b2ensmr, %b2enbnnf : tensor<96xf32>
    %b2enbnvar = stablehlo.divide %b2envsr, %b2enbnnf : tensor<96xf32>
    %b2dnbnnf = stablehlo.constant dense<100352.0> : tensor<96xf32>
    %b2dnbnmu = stablehlo.divide %b2dnsmr, %b2dnbnnf : tensor<96xf32>
    %b2dnbnvar = stablehlo.divide %b2dnvsr, %b2dnbnnf : tensor<96xf32>
    %b2pnbnnf = stablehlo.constant dense<100352.0> : tensor<24xf32>
    %b2pnbnmu = stablehlo.divide %b2pnsmr, %b2pnbnnf : tensor<24xf32>
    %b2pnbnvar = stablehlo.divide %b2pnvsr, %b2pnbnnf : tensor<24xf32>
    %b3enbnnf = stablehlo.constant dense<100352.0> : tensor<96xf32>
    %b3enbnmu = stablehlo.divide %b3ensmr, %b3enbnnf : tensor<96xf32>
    %b3enbnvar = stablehlo.divide %b3envsr, %b3enbnnf : tensor<96xf32>
    %b3dnbnnf = stablehlo.constant dense<25088.0> : tensor<96xf32>
    %b3dnbnmu = stablehlo.divide %b3dnsmr, %b3dnbnnf : tensor<96xf32>
    %b3dnbnvar = stablehlo.divide %b3dnvsr, %b3dnbnnf : tensor<96xf32>
    %b3pnbnnf = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %b3pnbnmu = stablehlo.divide %b3pnsmr, %b3pnbnnf : tensor<32xf32>
    %b3pnbnvar = stablehlo.divide %b3pnvsr, %b3pnbnnf : tensor<32xf32>
    %b4enbnnf = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %b4enbnmu = stablehlo.divide %b4ensmr, %b4enbnnf : tensor<128xf32>
    %b4enbnvar = stablehlo.divide %b4envsr, %b4enbnnf : tensor<128xf32>
    %b4dnbnnf = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %b4dnbnmu = stablehlo.divide %b4dnsmr, %b4dnbnnf : tensor<128xf32>
    %b4dnbnvar = stablehlo.divide %b4dnvsr, %b4dnbnnf : tensor<128xf32>
    %b4pnbnnf = stablehlo.constant dense<25088.0> : tensor<32xf32>
    %b4pnbnmu = stablehlo.divide %b4pnsmr, %b4pnbnnf : tensor<32xf32>
    %b4pnbnvar = stablehlo.divide %b4pnvsr, %b4pnbnnf : tensor<32xf32>
    %b5enbnnf = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %b5enbnmu = stablehlo.divide %b5ensmr, %b5enbnnf : tensor<128xf32>
    %b5enbnvar = stablehlo.divide %b5envsr, %b5enbnnf : tensor<128xf32>
    %b5dnbnnf = stablehlo.constant dense<6272.0> : tensor<128xf32>
    %b5dnbnmu = stablehlo.divide %b5dnsmr, %b5dnbnnf : tensor<128xf32>
    %b5dnbnvar = stablehlo.divide %b5dnvsr, %b5dnbnnf : tensor<128xf32>
    %b5pnbnnf = stablehlo.constant dense<6272.0> : tensor<64xf32>
    %b5pnbnmu = stablehlo.divide %b5pnsmr, %b5pnbnnf : tensor<64xf32>
    %b5pnbnvar = stablehlo.divide %b5pnvsr, %b5pnbnnf : tensor<64xf32>
    %b6enbnnf = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %b6enbnmu = stablehlo.divide %b6ensmr, %b6enbnnf : tensor<256xf32>
    %b6enbnvar = stablehlo.divide %b6envsr, %b6enbnnf : tensor<256xf32>
    %b6dnbnnf = stablehlo.constant dense<1568.0> : tensor<256xf32>
    %b6dnbnmu = stablehlo.divide %b6dnsmr, %b6dnbnnf : tensor<256xf32>
    %b6dnbnvar = stablehlo.divide %b6dnvsr, %b6dnbnnf : tensor<256xf32>
    %b6pnbnnf = stablehlo.constant dense<1568.0> : tensor<64xf32>
    %b6pnbnmu = stablehlo.divide %b6pnsmr, %b6pnbnnf : tensor<64xf32>
    %b6pnbnvar = stablehlo.divide %b6pnvsr, %b6pnbnnf : tensor<64xf32>
    %hnbnnf = stablehlo.constant dense<1568.0> : tensor<128xf32>
    %hnbnmu = stablehlo.divide %hnsmr, %hnbnnf : tensor<128xf32>
    %hnbnvar = stablehlo.divide %hnvsr, %hnbnnf : tensor<128xf32>
    return %adnewsW, %adnewsb, %adnewsg, %adnewsbt, %adnewb1eW, %adnewb1eb, %adnewb1eg, %adnewb1ebt, %adnewb1dW, %adnewb1db, %adnewb1dg, %adnewb1dbt, %adnewb1pW, %adnewb1pb, %adnewb1pg, %adnewb1pbt, %adnewb2eW, %adnewb2eb, %adnewb2eg, %adnewb2ebt, %adnewb2dW, %adnewb2db, %adnewb2dg, %adnewb2dbt, %adnewb2pW, %adnewb2pb, %adnewb2pg, %adnewb2pbt, %adnewb3eW, %adnewb3eb, %adnewb3eg, %adnewb3ebt, %adnewb3dW, %adnewb3db, %adnewb3dg, %adnewb3dbt, %adnewb3pW, %adnewb3pb, %adnewb3pg, %adnewb3pbt, %adnewb4eW, %adnewb4eb, %adnewb4eg, %adnewb4ebt, %adnewb4dW, %adnewb4db, %adnewb4dg, %adnewb4dbt, %adnewb4pW, %adnewb4pb, %adnewb4pg, %adnewb4pbt, %adnewb5eW, %adnewb5eb, %adnewb5eg, %adnewb5ebt, %adnewb5dW, %adnewb5db, %adnewb5dg, %adnewb5dbt, %adnewb5pW, %adnewb5pb, %adnewb5pg, %adnewb5pbt, %adnewb6eW, %adnewb6eb, %adnewb6eg, %adnewb6ebt, %adnewb6dW, %adnewb6db, %adnewb6dg, %adnewb6dbt, %adnewb6pW, %adnewb6pb, %adnewb6pg, %adnewb6pbt, %adnewhW, %adnewhb, %adnewhg, %adnewhbt, %adnewWd, %adnewbd, %admnsW, %admnsb, %admnsg, %admnsbt, %admnb1eW, %admnb1eb, %admnb1eg, %admnb1ebt, %admnb1dW, %admnb1db, %admnb1dg, %admnb1dbt, %admnb1pW, %admnb1pb, %admnb1pg, %admnb1pbt, %admnb2eW, %admnb2eb, %admnb2eg, %admnb2ebt, %admnb2dW, %admnb2db, %admnb2dg, %admnb2dbt, %admnb2pW, %admnb2pb, %admnb2pg, %admnb2pbt, %admnb3eW, %admnb3eb, %admnb3eg, %admnb3ebt, %admnb3dW, %admnb3db, %admnb3dg, %admnb3dbt, %admnb3pW, %admnb3pb, %admnb3pg, %admnb3pbt, %admnb4eW, %admnb4eb, %admnb4eg, %admnb4ebt, %admnb4dW, %admnb4db, %admnb4dg, %admnb4dbt, %admnb4pW, %admnb4pb, %admnb4pg, %admnb4pbt, %admnb5eW, %admnb5eb, %admnb5eg, %admnb5ebt, %admnb5dW, %admnb5db, %admnb5dg, %admnb5dbt, %admnb5pW, %admnb5pb, %admnb5pg, %admnb5pbt, %admnb6eW, %admnb6eb, %admnb6eg, %admnb6ebt, %admnb6dW, %admnb6db, %admnb6dg, %admnb6dbt, %admnb6pW, %admnb6pb, %admnb6pg, %admnb6pbt, %admnhW, %admnhb, %admnhg, %admnhbt, %admnWd, %admnbd, %advnsW, %advnsb, %advnsg, %advnsbt, %advnb1eW, %advnb1eb, %advnb1eg, %advnb1ebt, %advnb1dW, %advnb1db, %advnb1dg, %advnb1dbt, %advnb1pW, %advnb1pb, %advnb1pg, %advnb1pbt, %advnb2eW, %advnb2eb, %advnb2eg, %advnb2ebt, %advnb2dW, %advnb2db, %advnb2dg, %advnb2dbt, %advnb2pW, %advnb2pb, %advnb2pg, %advnb2pbt, %advnb3eW, %advnb3eb, %advnb3eg, %advnb3ebt, %advnb3dW, %advnb3db, %advnb3dg, %advnb3dbt, %advnb3pW, %advnb3pb, %advnb3pg, %advnb3pbt, %advnb4eW, %advnb4eb, %advnb4eg, %advnb4ebt, %advnb4dW, %advnb4db, %advnb4dg, %advnb4dbt, %advnb4pW, %advnb4pb, %advnb4pg, %advnb4pbt, %advnb5eW, %advnb5eb, %advnb5eg, %advnb5ebt, %advnb5dW, %advnb5db, %advnb5dg, %advnb5dbt, %advnb5pW, %advnb5pb, %advnb5pg, %advnb5pbt, %advnb6eW, %advnb6eb, %advnb6eg, %advnb6ebt, %advnb6dW, %advnb6db, %advnb6dg, %advnb6dbt, %advnb6pW, %advnb6pb, %advnb6pg, %advnb6pbt, %advnhW, %advnhb, %advnhg, %advnhbt, %advnWd, %advnbd, %loss, %bc1, %bc2, %stnbnmu, %stnbnvar, %b1enbnmu, %b1enbnvar, %b1dnbnmu, %b1dnbnvar, %b1pnbnmu, %b1pnbnvar, %b2enbnmu, %b2enbnvar, %b2dnbnmu, %b2dnbnvar, %b2pnbnmu, %b2pnbnvar, %b3enbnmu, %b3enbnvar, %b3dnbnmu, %b3dnbnvar, %b3pnbnmu, %b3pnbnvar, %b4enbnmu, %b4enbnvar, %b4dnbnmu, %b4dnbnvar, %b4pnbnmu, %b4pnbnvar, %b5enbnmu, %b5enbnvar, %b5dnbnmu, %b5dnbnvar, %b5pnbnmu, %b5pnbnvar, %b6enbnmu, %b6enbnvar, %b6dnbnmu, %b6dnbnvar, %b6pnbnmu, %b6pnbnvar, %hnbnmu, %hnbnvar : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<16xf32>, tensor<16xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>
  }
}
