module @m {
  func.func @resnet34_train_step(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sb: tensor<64xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0b1: tensor<64xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0b2: tensor<64xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1b1: tensor<64xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1b2: tensor<64xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2b1: tensor<64xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2b2: tensor<64xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2b1: tensor<128xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2b2: tensor<128xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x3x3xf32>, %d2bp: tensor<128xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0b1: tensor<128xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0b2: tensor<128xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1b1: tensor<128xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1b2: tensor<128xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2b1: tensor<128xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2b2: tensor<128xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3b1: tensor<256xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3b2: tensor<256xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x3x3xf32>, %d3bp: tensor<256xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0b1: tensor<256xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0b2: tensor<256xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1b1: tensor<256xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1b2: tensor<256xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2b1: tensor<256xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2b2: tensor<256xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3b1: tensor<256xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3b2: tensor<256xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4b1: tensor<256xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4b2: tensor<256xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4b1: tensor<512xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4b2: tensor<512xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x3x3xf32>, %d4bp: tensor<512xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0b1: tensor<512xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0b2: tensor<512xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1b1: tensor<512xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1b2: tensor<512xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %stcc = stablehlo.convolution(%xr, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<32x64x112x112xf32>
    %stcbb = stablehlo.broadcast_in_dim %sb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %stc = stablehlo.add %stcc, %stcbb : tensor<32x64x112x112xf32>
    %stnnf = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %stnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %stnsmr = stablehlo.reduce(%stc init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %stnsm = stablehlo.broadcast_in_dim %stnsmr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %stnmu = stablehlo.divide %stnsm, %stnnf : tensor<32x64x112x112xf32>
    %stnxc = stablehlo.subtract %stc, %stnmu : tensor<32x64x112x112xf32>
    %stnsq = stablehlo.multiply %stnxc, %stnxc : tensor<32x64x112x112xf32>
    %stnvsr = stablehlo.reduce(%stnsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %stnvs = stablehlo.broadcast_in_dim %stnvsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %stnvr = stablehlo.divide %stnvs, %stnnf : tensor<32x64x112x112xf32>
    %stnve = stablehlo.add %stnvr, %stnep : tensor<32x64x112x112xf32>
    %stnistd = stablehlo.rsqrt %stnve : tensor<32x64x112x112xf32>
    %stnxh = stablehlo.multiply %stnxc, %stnistd : tensor<32x64x112x112xf32>
    %stngb = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %stnbtb = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %stngx = stablehlo.multiply %stnxh, %stngb : tensor<32x64x112x112xf32>
    %stn = stablehlo.add %stngx, %stnbtb : tensor<32x64x112x112xf32>
    %strz = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %str = stablehlo.maximum %stn, %strz : tensor<32x64x112x112xf32>
    %stpni = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %stp = "stablehlo.reduce_window"(%str, %stpni) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x56x56xf32>
    %s1b0c1c = stablehlo.convolution(%stp, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b0c1bb = stablehlo.broadcast_in_dim %s1b0b1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0c1 = stablehlo.add %s1b0c1c, %s1b0c1bb : tensor<32x64x56x56xf32>
    %s1b0n1nf = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %s1b0n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b0n1smr = stablehlo.reduce(%s1b0c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0n1sm = stablehlo.broadcast_in_dim %s1b0n1smr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n1mu = stablehlo.divide %s1b0n1sm, %s1b0n1nf : tensor<32x64x56x56xf32>
    %s1b0n1xc = stablehlo.subtract %s1b0c1, %s1b0n1mu : tensor<32x64x56x56xf32>
    %s1b0n1sq = stablehlo.multiply %s1b0n1xc, %s1b0n1xc : tensor<32x64x56x56xf32>
    %s1b0n1vsr = stablehlo.reduce(%s1b0n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0n1vs = stablehlo.broadcast_in_dim %s1b0n1vsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n1vr = stablehlo.divide %s1b0n1vs, %s1b0n1nf : tensor<32x64x56x56xf32>
    %s1b0n1ve = stablehlo.add %s1b0n1vr, %s1b0n1ep : tensor<32x64x56x56xf32>
    %s1b0n1istd = stablehlo.rsqrt %s1b0n1ve : tensor<32x64x56x56xf32>
    %s1b0n1xh = stablehlo.multiply %s1b0n1xc, %s1b0n1istd : tensor<32x64x56x56xf32>
    %s1b0n1gb = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n1btb = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n1gx = stablehlo.multiply %s1b0n1xh, %s1b0n1gb : tensor<32x64x56x56xf32>
    %s1b0n1 = stablehlo.add %s1b0n1gx, %s1b0n1btb : tensor<32x64x56x56xf32>
    %s1b0r1z = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b0r1 = stablehlo.maximum %s1b0n1, %s1b0r1z : tensor<32x64x56x56xf32>
    %s1b0c2c = stablehlo.convolution(%s1b0r1, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b0c2bb = stablehlo.broadcast_in_dim %s1b0b2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0c2 = stablehlo.add %s1b0c2c, %s1b0c2bb : tensor<32x64x56x56xf32>
    %s1b0n2nf = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %s1b0n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b0n2smr = stablehlo.reduce(%s1b0c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0n2sm = stablehlo.broadcast_in_dim %s1b0n2smr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n2mu = stablehlo.divide %s1b0n2sm, %s1b0n2nf : tensor<32x64x56x56xf32>
    %s1b0n2xc = stablehlo.subtract %s1b0c2, %s1b0n2mu : tensor<32x64x56x56xf32>
    %s1b0n2sq = stablehlo.multiply %s1b0n2xc, %s1b0n2xc : tensor<32x64x56x56xf32>
    %s1b0n2vsr = stablehlo.reduce(%s1b0n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0n2vs = stablehlo.broadcast_in_dim %s1b0n2vsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n2vr = stablehlo.divide %s1b0n2vs, %s1b0n2nf : tensor<32x64x56x56xf32>
    %s1b0n2ve = stablehlo.add %s1b0n2vr, %s1b0n2ep : tensor<32x64x56x56xf32>
    %s1b0n2istd = stablehlo.rsqrt %s1b0n2ve : tensor<32x64x56x56xf32>
    %s1b0n2xh = stablehlo.multiply %s1b0n2xc, %s1b0n2istd : tensor<32x64x56x56xf32>
    %s1b0n2gb = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n2btb = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n2gx = stablehlo.multiply %s1b0n2xh, %s1b0n2gb : tensor<32x64x56x56xf32>
    %s1b0n2 = stablehlo.add %s1b0n2gx, %s1b0n2btb : tensor<32x64x56x56xf32>
    %s1b0a = stablehlo.add %s1b0n2, %stp : tensor<32x64x56x56xf32>
    %s1b0oz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b0o = stablehlo.maximum %s1b0a, %s1b0oz : tensor<32x64x56x56xf32>
    %s1b1c1c = stablehlo.convolution(%s1b0o, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b1c1bb = stablehlo.broadcast_in_dim %s1b1b1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1c1 = stablehlo.add %s1b1c1c, %s1b1c1bb : tensor<32x64x56x56xf32>
    %s1b1n1nf = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %s1b1n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b1n1smr = stablehlo.reduce(%s1b1c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1n1sm = stablehlo.broadcast_in_dim %s1b1n1smr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n1mu = stablehlo.divide %s1b1n1sm, %s1b1n1nf : tensor<32x64x56x56xf32>
    %s1b1n1xc = stablehlo.subtract %s1b1c1, %s1b1n1mu : tensor<32x64x56x56xf32>
    %s1b1n1sq = stablehlo.multiply %s1b1n1xc, %s1b1n1xc : tensor<32x64x56x56xf32>
    %s1b1n1vsr = stablehlo.reduce(%s1b1n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1n1vs = stablehlo.broadcast_in_dim %s1b1n1vsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n1vr = stablehlo.divide %s1b1n1vs, %s1b1n1nf : tensor<32x64x56x56xf32>
    %s1b1n1ve = stablehlo.add %s1b1n1vr, %s1b1n1ep : tensor<32x64x56x56xf32>
    %s1b1n1istd = stablehlo.rsqrt %s1b1n1ve : tensor<32x64x56x56xf32>
    %s1b1n1xh = stablehlo.multiply %s1b1n1xc, %s1b1n1istd : tensor<32x64x56x56xf32>
    %s1b1n1gb = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n1btb = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n1gx = stablehlo.multiply %s1b1n1xh, %s1b1n1gb : tensor<32x64x56x56xf32>
    %s1b1n1 = stablehlo.add %s1b1n1gx, %s1b1n1btb : tensor<32x64x56x56xf32>
    %s1b1r1z = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b1r1 = stablehlo.maximum %s1b1n1, %s1b1r1z : tensor<32x64x56x56xf32>
    %s1b1c2c = stablehlo.convolution(%s1b1r1, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b1c2bb = stablehlo.broadcast_in_dim %s1b1b2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1c2 = stablehlo.add %s1b1c2c, %s1b1c2bb : tensor<32x64x56x56xf32>
    %s1b1n2nf = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %s1b1n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b1n2smr = stablehlo.reduce(%s1b1c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1n2sm = stablehlo.broadcast_in_dim %s1b1n2smr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n2mu = stablehlo.divide %s1b1n2sm, %s1b1n2nf : tensor<32x64x56x56xf32>
    %s1b1n2xc = stablehlo.subtract %s1b1c2, %s1b1n2mu : tensor<32x64x56x56xf32>
    %s1b1n2sq = stablehlo.multiply %s1b1n2xc, %s1b1n2xc : tensor<32x64x56x56xf32>
    %s1b1n2vsr = stablehlo.reduce(%s1b1n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1n2vs = stablehlo.broadcast_in_dim %s1b1n2vsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n2vr = stablehlo.divide %s1b1n2vs, %s1b1n2nf : tensor<32x64x56x56xf32>
    %s1b1n2ve = stablehlo.add %s1b1n2vr, %s1b1n2ep : tensor<32x64x56x56xf32>
    %s1b1n2istd = stablehlo.rsqrt %s1b1n2ve : tensor<32x64x56x56xf32>
    %s1b1n2xh = stablehlo.multiply %s1b1n2xc, %s1b1n2istd : tensor<32x64x56x56xf32>
    %s1b1n2gb = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n2btb = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n2gx = stablehlo.multiply %s1b1n2xh, %s1b1n2gb : tensor<32x64x56x56xf32>
    %s1b1n2 = stablehlo.add %s1b1n2gx, %s1b1n2btb : tensor<32x64x56x56xf32>
    %s1b1a = stablehlo.add %s1b1n2, %s1b0o : tensor<32x64x56x56xf32>
    %s1b1oz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b1o = stablehlo.maximum %s1b1a, %s1b1oz : tensor<32x64x56x56xf32>
    %s1b2c1c = stablehlo.convolution(%s1b1o, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b2c1bb = stablehlo.broadcast_in_dim %s1b2b1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2c1 = stablehlo.add %s1b2c1c, %s1b2c1bb : tensor<32x64x56x56xf32>
    %s1b2n1nf = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %s1b2n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b2n1smr = stablehlo.reduce(%s1b2c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2n1sm = stablehlo.broadcast_in_dim %s1b2n1smr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n1mu = stablehlo.divide %s1b2n1sm, %s1b2n1nf : tensor<32x64x56x56xf32>
    %s1b2n1xc = stablehlo.subtract %s1b2c1, %s1b2n1mu : tensor<32x64x56x56xf32>
    %s1b2n1sq = stablehlo.multiply %s1b2n1xc, %s1b2n1xc : tensor<32x64x56x56xf32>
    %s1b2n1vsr = stablehlo.reduce(%s1b2n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2n1vs = stablehlo.broadcast_in_dim %s1b2n1vsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n1vr = stablehlo.divide %s1b2n1vs, %s1b2n1nf : tensor<32x64x56x56xf32>
    %s1b2n1ve = stablehlo.add %s1b2n1vr, %s1b2n1ep : tensor<32x64x56x56xf32>
    %s1b2n1istd = stablehlo.rsqrt %s1b2n1ve : tensor<32x64x56x56xf32>
    %s1b2n1xh = stablehlo.multiply %s1b2n1xc, %s1b2n1istd : tensor<32x64x56x56xf32>
    %s1b2n1gb = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n1btb = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n1gx = stablehlo.multiply %s1b2n1xh, %s1b2n1gb : tensor<32x64x56x56xf32>
    %s1b2n1 = stablehlo.add %s1b2n1gx, %s1b2n1btb : tensor<32x64x56x56xf32>
    %s1b2r1z = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b2r1 = stablehlo.maximum %s1b2n1, %s1b2r1z : tensor<32x64x56x56xf32>
    %s1b2c2c = stablehlo.convolution(%s1b2r1, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b2c2bb = stablehlo.broadcast_in_dim %s1b2b2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2c2 = stablehlo.add %s1b2c2c, %s1b2c2bb : tensor<32x64x56x56xf32>
    %s1b2n2nf = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %s1b2n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b2n2smr = stablehlo.reduce(%s1b2c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2n2sm = stablehlo.broadcast_in_dim %s1b2n2smr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n2mu = stablehlo.divide %s1b2n2sm, %s1b2n2nf : tensor<32x64x56x56xf32>
    %s1b2n2xc = stablehlo.subtract %s1b2c2, %s1b2n2mu : tensor<32x64x56x56xf32>
    %s1b2n2sq = stablehlo.multiply %s1b2n2xc, %s1b2n2xc : tensor<32x64x56x56xf32>
    %s1b2n2vsr = stablehlo.reduce(%s1b2n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2n2vs = stablehlo.broadcast_in_dim %s1b2n2vsr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n2vr = stablehlo.divide %s1b2n2vs, %s1b2n2nf : tensor<32x64x56x56xf32>
    %s1b2n2ve = stablehlo.add %s1b2n2vr, %s1b2n2ep : tensor<32x64x56x56xf32>
    %s1b2n2istd = stablehlo.rsqrt %s1b2n2ve : tensor<32x64x56x56xf32>
    %s1b2n2xh = stablehlo.multiply %s1b2n2xc, %s1b2n2istd : tensor<32x64x56x56xf32>
    %s1b2n2gb = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n2btb = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n2gx = stablehlo.multiply %s1b2n2xh, %s1b2n2gb : tensor<32x64x56x56xf32>
    %s1b2n2 = stablehlo.add %s1b2n2gx, %s1b2n2btb : tensor<32x64x56x56xf32>
    %s1b2a = stablehlo.add %s1b2n2, %s1b1o : tensor<32x64x56x56xf32>
    %s1b2oz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b2o = stablehlo.maximum %s1b2a, %s1b2oz : tensor<32x64x56x56xf32>
    %d2c1c = stablehlo.convolution(%s1b2o, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %d2c1bb = stablehlo.broadcast_in_dim %d2b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2c1 = stablehlo.add %d2c1c, %d2c1bb : tensor<32x128x28x28xf32>
    %d2n1nf = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %d2n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %d2n1smr = stablehlo.reduce(%d2c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2n1sm = stablehlo.broadcast_in_dim %d2n1smr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n1mu = stablehlo.divide %d2n1sm, %d2n1nf : tensor<32x128x28x28xf32>
    %d2n1xc = stablehlo.subtract %d2c1, %d2n1mu : tensor<32x128x28x28xf32>
    %d2n1sq = stablehlo.multiply %d2n1xc, %d2n1xc : tensor<32x128x28x28xf32>
    %d2n1vsr = stablehlo.reduce(%d2n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2n1vs = stablehlo.broadcast_in_dim %d2n1vsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n1vr = stablehlo.divide %d2n1vs, %d2n1nf : tensor<32x128x28x28xf32>
    %d2n1ve = stablehlo.add %d2n1vr, %d2n1ep : tensor<32x128x28x28xf32>
    %d2n1istd = stablehlo.rsqrt %d2n1ve : tensor<32x128x28x28xf32>
    %d2n1xh = stablehlo.multiply %d2n1xc, %d2n1istd : tensor<32x128x28x28xf32>
    %d2n1gb = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n1btb = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n1gx = stablehlo.multiply %d2n1xh, %d2n1gb : tensor<32x128x28x28xf32>
    %d2n1 = stablehlo.add %d2n1gx, %d2n1btb : tensor<32x128x28x28xf32>
    %d2r1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %d2r1 = stablehlo.maximum %d2n1, %d2r1z : tensor<32x128x28x28xf32>
    %d2c2c = stablehlo.convolution(%d2r1, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %d2c2bb = stablehlo.broadcast_in_dim %d2b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2c2 = stablehlo.add %d2c2c, %d2c2bb : tensor<32x128x28x28xf32>
    %d2n2nf = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %d2n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %d2n2smr = stablehlo.reduce(%d2c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2n2sm = stablehlo.broadcast_in_dim %d2n2smr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n2mu = stablehlo.divide %d2n2sm, %d2n2nf : tensor<32x128x28x28xf32>
    %d2n2xc = stablehlo.subtract %d2c2, %d2n2mu : tensor<32x128x28x28xf32>
    %d2n2sq = stablehlo.multiply %d2n2xc, %d2n2xc : tensor<32x128x28x28xf32>
    %d2n2vsr = stablehlo.reduce(%d2n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2n2vs = stablehlo.broadcast_in_dim %d2n2vsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n2vr = stablehlo.divide %d2n2vs, %d2n2nf : tensor<32x128x28x28xf32>
    %d2n2ve = stablehlo.add %d2n2vr, %d2n2ep : tensor<32x128x28x28xf32>
    %d2n2istd = stablehlo.rsqrt %d2n2ve : tensor<32x128x28x28xf32>
    %d2n2xh = stablehlo.multiply %d2n2xc, %d2n2istd : tensor<32x128x28x28xf32>
    %d2n2gb = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n2btb = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n2gx = stablehlo.multiply %d2n2xh, %d2n2gb : tensor<32x128x28x28xf32>
    %d2n2 = stablehlo.add %d2n2gx, %d2n2btb : tensor<32x128x28x28xf32>
    %d2cpc = stablehlo.convolution(%s1b2o, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %d2cpbb = stablehlo.broadcast_in_dim %d2bp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2cp = stablehlo.add %d2cpc, %d2cpbb : tensor<32x128x28x28xf32>
    %d2npnf = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %d2npep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %d2npsmr = stablehlo.reduce(%d2cp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2npsm = stablehlo.broadcast_in_dim %d2npsmr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2npmu = stablehlo.divide %d2npsm, %d2npnf : tensor<32x128x28x28xf32>
    %d2npxc = stablehlo.subtract %d2cp, %d2npmu : tensor<32x128x28x28xf32>
    %d2npsq = stablehlo.multiply %d2npxc, %d2npxc : tensor<32x128x28x28xf32>
    %d2npvsr = stablehlo.reduce(%d2npsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2npvs = stablehlo.broadcast_in_dim %d2npvsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2npvr = stablehlo.divide %d2npvs, %d2npnf : tensor<32x128x28x28xf32>
    %d2npve = stablehlo.add %d2npvr, %d2npep : tensor<32x128x28x28xf32>
    %d2npistd = stablehlo.rsqrt %d2npve : tensor<32x128x28x28xf32>
    %d2npxh = stablehlo.multiply %d2npxc, %d2npistd : tensor<32x128x28x28xf32>
    %d2npgb = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2npbtb = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2npgx = stablehlo.multiply %d2npxh, %d2npgb : tensor<32x128x28x28xf32>
    %d2np = stablehlo.add %d2npgx, %d2npbtb : tensor<32x128x28x28xf32>
    %d2a = stablehlo.add %d2n2, %d2np : tensor<32x128x28x28xf32>
    %d2oz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %d2o = stablehlo.maximum %d2a, %d2oz : tensor<32x128x28x28xf32>
    %s2b0c1c = stablehlo.convolution(%d2o, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b0c1bb = stablehlo.broadcast_in_dim %s2b0b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0c1 = stablehlo.add %s2b0c1c, %s2b0c1bb : tensor<32x128x28x28xf32>
    %s2b0n1nf = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %s2b0n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b0n1smr = stablehlo.reduce(%s2b0c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0n1sm = stablehlo.broadcast_in_dim %s2b0n1smr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n1mu = stablehlo.divide %s2b0n1sm, %s2b0n1nf : tensor<32x128x28x28xf32>
    %s2b0n1xc = stablehlo.subtract %s2b0c1, %s2b0n1mu : tensor<32x128x28x28xf32>
    %s2b0n1sq = stablehlo.multiply %s2b0n1xc, %s2b0n1xc : tensor<32x128x28x28xf32>
    %s2b0n1vsr = stablehlo.reduce(%s2b0n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0n1vs = stablehlo.broadcast_in_dim %s2b0n1vsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n1vr = stablehlo.divide %s2b0n1vs, %s2b0n1nf : tensor<32x128x28x28xf32>
    %s2b0n1ve = stablehlo.add %s2b0n1vr, %s2b0n1ep : tensor<32x128x28x28xf32>
    %s2b0n1istd = stablehlo.rsqrt %s2b0n1ve : tensor<32x128x28x28xf32>
    %s2b0n1xh = stablehlo.multiply %s2b0n1xc, %s2b0n1istd : tensor<32x128x28x28xf32>
    %s2b0n1gb = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n1btb = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n1gx = stablehlo.multiply %s2b0n1xh, %s2b0n1gb : tensor<32x128x28x28xf32>
    %s2b0n1 = stablehlo.add %s2b0n1gx, %s2b0n1btb : tensor<32x128x28x28xf32>
    %s2b0r1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b0r1 = stablehlo.maximum %s2b0n1, %s2b0r1z : tensor<32x128x28x28xf32>
    %s2b0c2c = stablehlo.convolution(%s2b0r1, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b0c2bb = stablehlo.broadcast_in_dim %s2b0b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0c2 = stablehlo.add %s2b0c2c, %s2b0c2bb : tensor<32x128x28x28xf32>
    %s2b0n2nf = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %s2b0n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b0n2smr = stablehlo.reduce(%s2b0c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0n2sm = stablehlo.broadcast_in_dim %s2b0n2smr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n2mu = stablehlo.divide %s2b0n2sm, %s2b0n2nf : tensor<32x128x28x28xf32>
    %s2b0n2xc = stablehlo.subtract %s2b0c2, %s2b0n2mu : tensor<32x128x28x28xf32>
    %s2b0n2sq = stablehlo.multiply %s2b0n2xc, %s2b0n2xc : tensor<32x128x28x28xf32>
    %s2b0n2vsr = stablehlo.reduce(%s2b0n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0n2vs = stablehlo.broadcast_in_dim %s2b0n2vsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n2vr = stablehlo.divide %s2b0n2vs, %s2b0n2nf : tensor<32x128x28x28xf32>
    %s2b0n2ve = stablehlo.add %s2b0n2vr, %s2b0n2ep : tensor<32x128x28x28xf32>
    %s2b0n2istd = stablehlo.rsqrt %s2b0n2ve : tensor<32x128x28x28xf32>
    %s2b0n2xh = stablehlo.multiply %s2b0n2xc, %s2b0n2istd : tensor<32x128x28x28xf32>
    %s2b0n2gb = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n2btb = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n2gx = stablehlo.multiply %s2b0n2xh, %s2b0n2gb : tensor<32x128x28x28xf32>
    %s2b0n2 = stablehlo.add %s2b0n2gx, %s2b0n2btb : tensor<32x128x28x28xf32>
    %s2b0a = stablehlo.add %s2b0n2, %d2o : tensor<32x128x28x28xf32>
    %s2b0oz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b0o = stablehlo.maximum %s2b0a, %s2b0oz : tensor<32x128x28x28xf32>
    %s2b1c1c = stablehlo.convolution(%s2b0o, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b1c1bb = stablehlo.broadcast_in_dim %s2b1b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1c1 = stablehlo.add %s2b1c1c, %s2b1c1bb : tensor<32x128x28x28xf32>
    %s2b1n1nf = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %s2b1n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b1n1smr = stablehlo.reduce(%s2b1c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1n1sm = stablehlo.broadcast_in_dim %s2b1n1smr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n1mu = stablehlo.divide %s2b1n1sm, %s2b1n1nf : tensor<32x128x28x28xf32>
    %s2b1n1xc = stablehlo.subtract %s2b1c1, %s2b1n1mu : tensor<32x128x28x28xf32>
    %s2b1n1sq = stablehlo.multiply %s2b1n1xc, %s2b1n1xc : tensor<32x128x28x28xf32>
    %s2b1n1vsr = stablehlo.reduce(%s2b1n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1n1vs = stablehlo.broadcast_in_dim %s2b1n1vsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n1vr = stablehlo.divide %s2b1n1vs, %s2b1n1nf : tensor<32x128x28x28xf32>
    %s2b1n1ve = stablehlo.add %s2b1n1vr, %s2b1n1ep : tensor<32x128x28x28xf32>
    %s2b1n1istd = stablehlo.rsqrt %s2b1n1ve : tensor<32x128x28x28xf32>
    %s2b1n1xh = stablehlo.multiply %s2b1n1xc, %s2b1n1istd : tensor<32x128x28x28xf32>
    %s2b1n1gb = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n1btb = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n1gx = stablehlo.multiply %s2b1n1xh, %s2b1n1gb : tensor<32x128x28x28xf32>
    %s2b1n1 = stablehlo.add %s2b1n1gx, %s2b1n1btb : tensor<32x128x28x28xf32>
    %s2b1r1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b1r1 = stablehlo.maximum %s2b1n1, %s2b1r1z : tensor<32x128x28x28xf32>
    %s2b1c2c = stablehlo.convolution(%s2b1r1, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b1c2bb = stablehlo.broadcast_in_dim %s2b1b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1c2 = stablehlo.add %s2b1c2c, %s2b1c2bb : tensor<32x128x28x28xf32>
    %s2b1n2nf = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %s2b1n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b1n2smr = stablehlo.reduce(%s2b1c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1n2sm = stablehlo.broadcast_in_dim %s2b1n2smr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n2mu = stablehlo.divide %s2b1n2sm, %s2b1n2nf : tensor<32x128x28x28xf32>
    %s2b1n2xc = stablehlo.subtract %s2b1c2, %s2b1n2mu : tensor<32x128x28x28xf32>
    %s2b1n2sq = stablehlo.multiply %s2b1n2xc, %s2b1n2xc : tensor<32x128x28x28xf32>
    %s2b1n2vsr = stablehlo.reduce(%s2b1n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1n2vs = stablehlo.broadcast_in_dim %s2b1n2vsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n2vr = stablehlo.divide %s2b1n2vs, %s2b1n2nf : tensor<32x128x28x28xf32>
    %s2b1n2ve = stablehlo.add %s2b1n2vr, %s2b1n2ep : tensor<32x128x28x28xf32>
    %s2b1n2istd = stablehlo.rsqrt %s2b1n2ve : tensor<32x128x28x28xf32>
    %s2b1n2xh = stablehlo.multiply %s2b1n2xc, %s2b1n2istd : tensor<32x128x28x28xf32>
    %s2b1n2gb = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n2btb = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n2gx = stablehlo.multiply %s2b1n2xh, %s2b1n2gb : tensor<32x128x28x28xf32>
    %s2b1n2 = stablehlo.add %s2b1n2gx, %s2b1n2btb : tensor<32x128x28x28xf32>
    %s2b1a = stablehlo.add %s2b1n2, %s2b0o : tensor<32x128x28x28xf32>
    %s2b1oz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b1o = stablehlo.maximum %s2b1a, %s2b1oz : tensor<32x128x28x28xf32>
    %s2b2c1c = stablehlo.convolution(%s2b1o, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b2c1bb = stablehlo.broadcast_in_dim %s2b2b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2c1 = stablehlo.add %s2b2c1c, %s2b2c1bb : tensor<32x128x28x28xf32>
    %s2b2n1nf = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %s2b2n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b2n1smr = stablehlo.reduce(%s2b2c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2n1sm = stablehlo.broadcast_in_dim %s2b2n1smr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n1mu = stablehlo.divide %s2b2n1sm, %s2b2n1nf : tensor<32x128x28x28xf32>
    %s2b2n1xc = stablehlo.subtract %s2b2c1, %s2b2n1mu : tensor<32x128x28x28xf32>
    %s2b2n1sq = stablehlo.multiply %s2b2n1xc, %s2b2n1xc : tensor<32x128x28x28xf32>
    %s2b2n1vsr = stablehlo.reduce(%s2b2n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2n1vs = stablehlo.broadcast_in_dim %s2b2n1vsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n1vr = stablehlo.divide %s2b2n1vs, %s2b2n1nf : tensor<32x128x28x28xf32>
    %s2b2n1ve = stablehlo.add %s2b2n1vr, %s2b2n1ep : tensor<32x128x28x28xf32>
    %s2b2n1istd = stablehlo.rsqrt %s2b2n1ve : tensor<32x128x28x28xf32>
    %s2b2n1xh = stablehlo.multiply %s2b2n1xc, %s2b2n1istd : tensor<32x128x28x28xf32>
    %s2b2n1gb = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n1btb = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n1gx = stablehlo.multiply %s2b2n1xh, %s2b2n1gb : tensor<32x128x28x28xf32>
    %s2b2n1 = stablehlo.add %s2b2n1gx, %s2b2n1btb : tensor<32x128x28x28xf32>
    %s2b2r1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b2r1 = stablehlo.maximum %s2b2n1, %s2b2r1z : tensor<32x128x28x28xf32>
    %s2b2c2c = stablehlo.convolution(%s2b2r1, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b2c2bb = stablehlo.broadcast_in_dim %s2b2b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2c2 = stablehlo.add %s2b2c2c, %s2b2c2bb : tensor<32x128x28x28xf32>
    %s2b2n2nf = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %s2b2n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b2n2smr = stablehlo.reduce(%s2b2c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2n2sm = stablehlo.broadcast_in_dim %s2b2n2smr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n2mu = stablehlo.divide %s2b2n2sm, %s2b2n2nf : tensor<32x128x28x28xf32>
    %s2b2n2xc = stablehlo.subtract %s2b2c2, %s2b2n2mu : tensor<32x128x28x28xf32>
    %s2b2n2sq = stablehlo.multiply %s2b2n2xc, %s2b2n2xc : tensor<32x128x28x28xf32>
    %s2b2n2vsr = stablehlo.reduce(%s2b2n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2n2vs = stablehlo.broadcast_in_dim %s2b2n2vsr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n2vr = stablehlo.divide %s2b2n2vs, %s2b2n2nf : tensor<32x128x28x28xf32>
    %s2b2n2ve = stablehlo.add %s2b2n2vr, %s2b2n2ep : tensor<32x128x28x28xf32>
    %s2b2n2istd = stablehlo.rsqrt %s2b2n2ve : tensor<32x128x28x28xf32>
    %s2b2n2xh = stablehlo.multiply %s2b2n2xc, %s2b2n2istd : tensor<32x128x28x28xf32>
    %s2b2n2gb = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n2btb = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n2gx = stablehlo.multiply %s2b2n2xh, %s2b2n2gb : tensor<32x128x28x28xf32>
    %s2b2n2 = stablehlo.add %s2b2n2gx, %s2b2n2btb : tensor<32x128x28x28xf32>
    %s2b2a = stablehlo.add %s2b2n2, %s2b1o : tensor<32x128x28x28xf32>
    %s2b2oz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b2o = stablehlo.maximum %s2b2a, %s2b2oz : tensor<32x128x28x28xf32>
    %d3c1c = stablehlo.convolution(%s2b2o, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %d3c1bb = stablehlo.broadcast_in_dim %d3b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3c1 = stablehlo.add %d3c1c, %d3c1bb : tensor<32x256x14x14xf32>
    %d3n1nf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %d3n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %d3n1smr = stablehlo.reduce(%d3c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3n1sm = stablehlo.broadcast_in_dim %d3n1smr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n1mu = stablehlo.divide %d3n1sm, %d3n1nf : tensor<32x256x14x14xf32>
    %d3n1xc = stablehlo.subtract %d3c1, %d3n1mu : tensor<32x256x14x14xf32>
    %d3n1sq = stablehlo.multiply %d3n1xc, %d3n1xc : tensor<32x256x14x14xf32>
    %d3n1vsr = stablehlo.reduce(%d3n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3n1vs = stablehlo.broadcast_in_dim %d3n1vsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n1vr = stablehlo.divide %d3n1vs, %d3n1nf : tensor<32x256x14x14xf32>
    %d3n1ve = stablehlo.add %d3n1vr, %d3n1ep : tensor<32x256x14x14xf32>
    %d3n1istd = stablehlo.rsqrt %d3n1ve : tensor<32x256x14x14xf32>
    %d3n1xh = stablehlo.multiply %d3n1xc, %d3n1istd : tensor<32x256x14x14xf32>
    %d3n1gb = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n1btb = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n1gx = stablehlo.multiply %d3n1xh, %d3n1gb : tensor<32x256x14x14xf32>
    %d3n1 = stablehlo.add %d3n1gx, %d3n1btb : tensor<32x256x14x14xf32>
    %d3r1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %d3r1 = stablehlo.maximum %d3n1, %d3r1z : tensor<32x256x14x14xf32>
    %d3c2c = stablehlo.convolution(%d3r1, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %d3c2bb = stablehlo.broadcast_in_dim %d3b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3c2 = stablehlo.add %d3c2c, %d3c2bb : tensor<32x256x14x14xf32>
    %d3n2nf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %d3n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %d3n2smr = stablehlo.reduce(%d3c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3n2sm = stablehlo.broadcast_in_dim %d3n2smr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n2mu = stablehlo.divide %d3n2sm, %d3n2nf : tensor<32x256x14x14xf32>
    %d3n2xc = stablehlo.subtract %d3c2, %d3n2mu : tensor<32x256x14x14xf32>
    %d3n2sq = stablehlo.multiply %d3n2xc, %d3n2xc : tensor<32x256x14x14xf32>
    %d3n2vsr = stablehlo.reduce(%d3n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3n2vs = stablehlo.broadcast_in_dim %d3n2vsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n2vr = stablehlo.divide %d3n2vs, %d3n2nf : tensor<32x256x14x14xf32>
    %d3n2ve = stablehlo.add %d3n2vr, %d3n2ep : tensor<32x256x14x14xf32>
    %d3n2istd = stablehlo.rsqrt %d3n2ve : tensor<32x256x14x14xf32>
    %d3n2xh = stablehlo.multiply %d3n2xc, %d3n2istd : tensor<32x256x14x14xf32>
    %d3n2gb = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n2btb = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n2gx = stablehlo.multiply %d3n2xh, %d3n2gb : tensor<32x256x14x14xf32>
    %d3n2 = stablehlo.add %d3n2gx, %d3n2btb : tensor<32x256x14x14xf32>
    %d3cpc = stablehlo.convolution(%s2b2o, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %d3cpbb = stablehlo.broadcast_in_dim %d3bp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3cp = stablehlo.add %d3cpc, %d3cpbb : tensor<32x256x14x14xf32>
    %d3npnf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %d3npep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %d3npsmr = stablehlo.reduce(%d3cp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3npsm = stablehlo.broadcast_in_dim %d3npsmr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3npmu = stablehlo.divide %d3npsm, %d3npnf : tensor<32x256x14x14xf32>
    %d3npxc = stablehlo.subtract %d3cp, %d3npmu : tensor<32x256x14x14xf32>
    %d3npsq = stablehlo.multiply %d3npxc, %d3npxc : tensor<32x256x14x14xf32>
    %d3npvsr = stablehlo.reduce(%d3npsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3npvs = stablehlo.broadcast_in_dim %d3npvsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3npvr = stablehlo.divide %d3npvs, %d3npnf : tensor<32x256x14x14xf32>
    %d3npve = stablehlo.add %d3npvr, %d3npep : tensor<32x256x14x14xf32>
    %d3npistd = stablehlo.rsqrt %d3npve : tensor<32x256x14x14xf32>
    %d3npxh = stablehlo.multiply %d3npxc, %d3npistd : tensor<32x256x14x14xf32>
    %d3npgb = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3npbtb = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3npgx = stablehlo.multiply %d3npxh, %d3npgb : tensor<32x256x14x14xf32>
    %d3np = stablehlo.add %d3npgx, %d3npbtb : tensor<32x256x14x14xf32>
    %d3a = stablehlo.add %d3n2, %d3np : tensor<32x256x14x14xf32>
    %d3oz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %d3o = stablehlo.maximum %d3a, %d3oz : tensor<32x256x14x14xf32>
    %s3b0c1c = stablehlo.convolution(%d3o, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b0c1bb = stablehlo.broadcast_in_dim %s3b0b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0c1 = stablehlo.add %s3b0c1c, %s3b0c1bb : tensor<32x256x14x14xf32>
    %s3b0n1nf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %s3b0n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b0n1smr = stablehlo.reduce(%s3b0c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0n1sm = stablehlo.broadcast_in_dim %s3b0n1smr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n1mu = stablehlo.divide %s3b0n1sm, %s3b0n1nf : tensor<32x256x14x14xf32>
    %s3b0n1xc = stablehlo.subtract %s3b0c1, %s3b0n1mu : tensor<32x256x14x14xf32>
    %s3b0n1sq = stablehlo.multiply %s3b0n1xc, %s3b0n1xc : tensor<32x256x14x14xf32>
    %s3b0n1vsr = stablehlo.reduce(%s3b0n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0n1vs = stablehlo.broadcast_in_dim %s3b0n1vsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n1vr = stablehlo.divide %s3b0n1vs, %s3b0n1nf : tensor<32x256x14x14xf32>
    %s3b0n1ve = stablehlo.add %s3b0n1vr, %s3b0n1ep : tensor<32x256x14x14xf32>
    %s3b0n1istd = stablehlo.rsqrt %s3b0n1ve : tensor<32x256x14x14xf32>
    %s3b0n1xh = stablehlo.multiply %s3b0n1xc, %s3b0n1istd : tensor<32x256x14x14xf32>
    %s3b0n1gb = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n1btb = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n1gx = stablehlo.multiply %s3b0n1xh, %s3b0n1gb : tensor<32x256x14x14xf32>
    %s3b0n1 = stablehlo.add %s3b0n1gx, %s3b0n1btb : tensor<32x256x14x14xf32>
    %s3b0r1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b0r1 = stablehlo.maximum %s3b0n1, %s3b0r1z : tensor<32x256x14x14xf32>
    %s3b0c2c = stablehlo.convolution(%s3b0r1, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b0c2bb = stablehlo.broadcast_in_dim %s3b0b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0c2 = stablehlo.add %s3b0c2c, %s3b0c2bb : tensor<32x256x14x14xf32>
    %s3b0n2nf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %s3b0n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b0n2smr = stablehlo.reduce(%s3b0c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0n2sm = stablehlo.broadcast_in_dim %s3b0n2smr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n2mu = stablehlo.divide %s3b0n2sm, %s3b0n2nf : tensor<32x256x14x14xf32>
    %s3b0n2xc = stablehlo.subtract %s3b0c2, %s3b0n2mu : tensor<32x256x14x14xf32>
    %s3b0n2sq = stablehlo.multiply %s3b0n2xc, %s3b0n2xc : tensor<32x256x14x14xf32>
    %s3b0n2vsr = stablehlo.reduce(%s3b0n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0n2vs = stablehlo.broadcast_in_dim %s3b0n2vsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n2vr = stablehlo.divide %s3b0n2vs, %s3b0n2nf : tensor<32x256x14x14xf32>
    %s3b0n2ve = stablehlo.add %s3b0n2vr, %s3b0n2ep : tensor<32x256x14x14xf32>
    %s3b0n2istd = stablehlo.rsqrt %s3b0n2ve : tensor<32x256x14x14xf32>
    %s3b0n2xh = stablehlo.multiply %s3b0n2xc, %s3b0n2istd : tensor<32x256x14x14xf32>
    %s3b0n2gb = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n2btb = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n2gx = stablehlo.multiply %s3b0n2xh, %s3b0n2gb : tensor<32x256x14x14xf32>
    %s3b0n2 = stablehlo.add %s3b0n2gx, %s3b0n2btb : tensor<32x256x14x14xf32>
    %s3b0a = stablehlo.add %s3b0n2, %d3o : tensor<32x256x14x14xf32>
    %s3b0oz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b0o = stablehlo.maximum %s3b0a, %s3b0oz : tensor<32x256x14x14xf32>
    %s3b1c1c = stablehlo.convolution(%s3b0o, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b1c1bb = stablehlo.broadcast_in_dim %s3b1b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1c1 = stablehlo.add %s3b1c1c, %s3b1c1bb : tensor<32x256x14x14xf32>
    %s3b1n1nf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %s3b1n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b1n1smr = stablehlo.reduce(%s3b1c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1n1sm = stablehlo.broadcast_in_dim %s3b1n1smr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n1mu = stablehlo.divide %s3b1n1sm, %s3b1n1nf : tensor<32x256x14x14xf32>
    %s3b1n1xc = stablehlo.subtract %s3b1c1, %s3b1n1mu : tensor<32x256x14x14xf32>
    %s3b1n1sq = stablehlo.multiply %s3b1n1xc, %s3b1n1xc : tensor<32x256x14x14xf32>
    %s3b1n1vsr = stablehlo.reduce(%s3b1n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1n1vs = stablehlo.broadcast_in_dim %s3b1n1vsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n1vr = stablehlo.divide %s3b1n1vs, %s3b1n1nf : tensor<32x256x14x14xf32>
    %s3b1n1ve = stablehlo.add %s3b1n1vr, %s3b1n1ep : tensor<32x256x14x14xf32>
    %s3b1n1istd = stablehlo.rsqrt %s3b1n1ve : tensor<32x256x14x14xf32>
    %s3b1n1xh = stablehlo.multiply %s3b1n1xc, %s3b1n1istd : tensor<32x256x14x14xf32>
    %s3b1n1gb = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n1btb = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n1gx = stablehlo.multiply %s3b1n1xh, %s3b1n1gb : tensor<32x256x14x14xf32>
    %s3b1n1 = stablehlo.add %s3b1n1gx, %s3b1n1btb : tensor<32x256x14x14xf32>
    %s3b1r1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b1r1 = stablehlo.maximum %s3b1n1, %s3b1r1z : tensor<32x256x14x14xf32>
    %s3b1c2c = stablehlo.convolution(%s3b1r1, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b1c2bb = stablehlo.broadcast_in_dim %s3b1b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1c2 = stablehlo.add %s3b1c2c, %s3b1c2bb : tensor<32x256x14x14xf32>
    %s3b1n2nf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %s3b1n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b1n2smr = stablehlo.reduce(%s3b1c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1n2sm = stablehlo.broadcast_in_dim %s3b1n2smr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n2mu = stablehlo.divide %s3b1n2sm, %s3b1n2nf : tensor<32x256x14x14xf32>
    %s3b1n2xc = stablehlo.subtract %s3b1c2, %s3b1n2mu : tensor<32x256x14x14xf32>
    %s3b1n2sq = stablehlo.multiply %s3b1n2xc, %s3b1n2xc : tensor<32x256x14x14xf32>
    %s3b1n2vsr = stablehlo.reduce(%s3b1n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1n2vs = stablehlo.broadcast_in_dim %s3b1n2vsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n2vr = stablehlo.divide %s3b1n2vs, %s3b1n2nf : tensor<32x256x14x14xf32>
    %s3b1n2ve = stablehlo.add %s3b1n2vr, %s3b1n2ep : tensor<32x256x14x14xf32>
    %s3b1n2istd = stablehlo.rsqrt %s3b1n2ve : tensor<32x256x14x14xf32>
    %s3b1n2xh = stablehlo.multiply %s3b1n2xc, %s3b1n2istd : tensor<32x256x14x14xf32>
    %s3b1n2gb = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n2btb = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n2gx = stablehlo.multiply %s3b1n2xh, %s3b1n2gb : tensor<32x256x14x14xf32>
    %s3b1n2 = stablehlo.add %s3b1n2gx, %s3b1n2btb : tensor<32x256x14x14xf32>
    %s3b1a = stablehlo.add %s3b1n2, %s3b0o : tensor<32x256x14x14xf32>
    %s3b1oz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b1o = stablehlo.maximum %s3b1a, %s3b1oz : tensor<32x256x14x14xf32>
    %s3b2c1c = stablehlo.convolution(%s3b1o, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b2c1bb = stablehlo.broadcast_in_dim %s3b2b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2c1 = stablehlo.add %s3b2c1c, %s3b2c1bb : tensor<32x256x14x14xf32>
    %s3b2n1nf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %s3b2n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b2n1smr = stablehlo.reduce(%s3b2c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2n1sm = stablehlo.broadcast_in_dim %s3b2n1smr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n1mu = stablehlo.divide %s3b2n1sm, %s3b2n1nf : tensor<32x256x14x14xf32>
    %s3b2n1xc = stablehlo.subtract %s3b2c1, %s3b2n1mu : tensor<32x256x14x14xf32>
    %s3b2n1sq = stablehlo.multiply %s3b2n1xc, %s3b2n1xc : tensor<32x256x14x14xf32>
    %s3b2n1vsr = stablehlo.reduce(%s3b2n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2n1vs = stablehlo.broadcast_in_dim %s3b2n1vsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n1vr = stablehlo.divide %s3b2n1vs, %s3b2n1nf : tensor<32x256x14x14xf32>
    %s3b2n1ve = stablehlo.add %s3b2n1vr, %s3b2n1ep : tensor<32x256x14x14xf32>
    %s3b2n1istd = stablehlo.rsqrt %s3b2n1ve : tensor<32x256x14x14xf32>
    %s3b2n1xh = stablehlo.multiply %s3b2n1xc, %s3b2n1istd : tensor<32x256x14x14xf32>
    %s3b2n1gb = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n1btb = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n1gx = stablehlo.multiply %s3b2n1xh, %s3b2n1gb : tensor<32x256x14x14xf32>
    %s3b2n1 = stablehlo.add %s3b2n1gx, %s3b2n1btb : tensor<32x256x14x14xf32>
    %s3b2r1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b2r1 = stablehlo.maximum %s3b2n1, %s3b2r1z : tensor<32x256x14x14xf32>
    %s3b2c2c = stablehlo.convolution(%s3b2r1, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b2c2bb = stablehlo.broadcast_in_dim %s3b2b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2c2 = stablehlo.add %s3b2c2c, %s3b2c2bb : tensor<32x256x14x14xf32>
    %s3b2n2nf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %s3b2n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b2n2smr = stablehlo.reduce(%s3b2c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2n2sm = stablehlo.broadcast_in_dim %s3b2n2smr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n2mu = stablehlo.divide %s3b2n2sm, %s3b2n2nf : tensor<32x256x14x14xf32>
    %s3b2n2xc = stablehlo.subtract %s3b2c2, %s3b2n2mu : tensor<32x256x14x14xf32>
    %s3b2n2sq = stablehlo.multiply %s3b2n2xc, %s3b2n2xc : tensor<32x256x14x14xf32>
    %s3b2n2vsr = stablehlo.reduce(%s3b2n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2n2vs = stablehlo.broadcast_in_dim %s3b2n2vsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n2vr = stablehlo.divide %s3b2n2vs, %s3b2n2nf : tensor<32x256x14x14xf32>
    %s3b2n2ve = stablehlo.add %s3b2n2vr, %s3b2n2ep : tensor<32x256x14x14xf32>
    %s3b2n2istd = stablehlo.rsqrt %s3b2n2ve : tensor<32x256x14x14xf32>
    %s3b2n2xh = stablehlo.multiply %s3b2n2xc, %s3b2n2istd : tensor<32x256x14x14xf32>
    %s3b2n2gb = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n2btb = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n2gx = stablehlo.multiply %s3b2n2xh, %s3b2n2gb : tensor<32x256x14x14xf32>
    %s3b2n2 = stablehlo.add %s3b2n2gx, %s3b2n2btb : tensor<32x256x14x14xf32>
    %s3b2a = stablehlo.add %s3b2n2, %s3b1o : tensor<32x256x14x14xf32>
    %s3b2oz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b2o = stablehlo.maximum %s3b2a, %s3b2oz : tensor<32x256x14x14xf32>
    %s3b3c1c = stablehlo.convolution(%s3b2o, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b3c1bb = stablehlo.broadcast_in_dim %s3b3b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3c1 = stablehlo.add %s3b3c1c, %s3b3c1bb : tensor<32x256x14x14xf32>
    %s3b3n1nf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %s3b3n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b3n1smr = stablehlo.reduce(%s3b3c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3n1sm = stablehlo.broadcast_in_dim %s3b3n1smr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n1mu = stablehlo.divide %s3b3n1sm, %s3b3n1nf : tensor<32x256x14x14xf32>
    %s3b3n1xc = stablehlo.subtract %s3b3c1, %s3b3n1mu : tensor<32x256x14x14xf32>
    %s3b3n1sq = stablehlo.multiply %s3b3n1xc, %s3b3n1xc : tensor<32x256x14x14xf32>
    %s3b3n1vsr = stablehlo.reduce(%s3b3n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3n1vs = stablehlo.broadcast_in_dim %s3b3n1vsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n1vr = stablehlo.divide %s3b3n1vs, %s3b3n1nf : tensor<32x256x14x14xf32>
    %s3b3n1ve = stablehlo.add %s3b3n1vr, %s3b3n1ep : tensor<32x256x14x14xf32>
    %s3b3n1istd = stablehlo.rsqrt %s3b3n1ve : tensor<32x256x14x14xf32>
    %s3b3n1xh = stablehlo.multiply %s3b3n1xc, %s3b3n1istd : tensor<32x256x14x14xf32>
    %s3b3n1gb = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n1btb = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n1gx = stablehlo.multiply %s3b3n1xh, %s3b3n1gb : tensor<32x256x14x14xf32>
    %s3b3n1 = stablehlo.add %s3b3n1gx, %s3b3n1btb : tensor<32x256x14x14xf32>
    %s3b3r1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b3r1 = stablehlo.maximum %s3b3n1, %s3b3r1z : tensor<32x256x14x14xf32>
    %s3b3c2c = stablehlo.convolution(%s3b3r1, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b3c2bb = stablehlo.broadcast_in_dim %s3b3b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3c2 = stablehlo.add %s3b3c2c, %s3b3c2bb : tensor<32x256x14x14xf32>
    %s3b3n2nf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %s3b3n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b3n2smr = stablehlo.reduce(%s3b3c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3n2sm = stablehlo.broadcast_in_dim %s3b3n2smr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n2mu = stablehlo.divide %s3b3n2sm, %s3b3n2nf : tensor<32x256x14x14xf32>
    %s3b3n2xc = stablehlo.subtract %s3b3c2, %s3b3n2mu : tensor<32x256x14x14xf32>
    %s3b3n2sq = stablehlo.multiply %s3b3n2xc, %s3b3n2xc : tensor<32x256x14x14xf32>
    %s3b3n2vsr = stablehlo.reduce(%s3b3n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3n2vs = stablehlo.broadcast_in_dim %s3b3n2vsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n2vr = stablehlo.divide %s3b3n2vs, %s3b3n2nf : tensor<32x256x14x14xf32>
    %s3b3n2ve = stablehlo.add %s3b3n2vr, %s3b3n2ep : tensor<32x256x14x14xf32>
    %s3b3n2istd = stablehlo.rsqrt %s3b3n2ve : tensor<32x256x14x14xf32>
    %s3b3n2xh = stablehlo.multiply %s3b3n2xc, %s3b3n2istd : tensor<32x256x14x14xf32>
    %s3b3n2gb = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n2btb = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n2gx = stablehlo.multiply %s3b3n2xh, %s3b3n2gb : tensor<32x256x14x14xf32>
    %s3b3n2 = stablehlo.add %s3b3n2gx, %s3b3n2btb : tensor<32x256x14x14xf32>
    %s3b3a = stablehlo.add %s3b3n2, %s3b2o : tensor<32x256x14x14xf32>
    %s3b3oz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b3o = stablehlo.maximum %s3b3a, %s3b3oz : tensor<32x256x14x14xf32>
    %s3b4c1c = stablehlo.convolution(%s3b3o, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b4c1bb = stablehlo.broadcast_in_dim %s3b4b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4c1 = stablehlo.add %s3b4c1c, %s3b4c1bb : tensor<32x256x14x14xf32>
    %s3b4n1nf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %s3b4n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b4n1smr = stablehlo.reduce(%s3b4c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4n1sm = stablehlo.broadcast_in_dim %s3b4n1smr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n1mu = stablehlo.divide %s3b4n1sm, %s3b4n1nf : tensor<32x256x14x14xf32>
    %s3b4n1xc = stablehlo.subtract %s3b4c1, %s3b4n1mu : tensor<32x256x14x14xf32>
    %s3b4n1sq = stablehlo.multiply %s3b4n1xc, %s3b4n1xc : tensor<32x256x14x14xf32>
    %s3b4n1vsr = stablehlo.reduce(%s3b4n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4n1vs = stablehlo.broadcast_in_dim %s3b4n1vsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n1vr = stablehlo.divide %s3b4n1vs, %s3b4n1nf : tensor<32x256x14x14xf32>
    %s3b4n1ve = stablehlo.add %s3b4n1vr, %s3b4n1ep : tensor<32x256x14x14xf32>
    %s3b4n1istd = stablehlo.rsqrt %s3b4n1ve : tensor<32x256x14x14xf32>
    %s3b4n1xh = stablehlo.multiply %s3b4n1xc, %s3b4n1istd : tensor<32x256x14x14xf32>
    %s3b4n1gb = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n1btb = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n1gx = stablehlo.multiply %s3b4n1xh, %s3b4n1gb : tensor<32x256x14x14xf32>
    %s3b4n1 = stablehlo.add %s3b4n1gx, %s3b4n1btb : tensor<32x256x14x14xf32>
    %s3b4r1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b4r1 = stablehlo.maximum %s3b4n1, %s3b4r1z : tensor<32x256x14x14xf32>
    %s3b4c2c = stablehlo.convolution(%s3b4r1, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b4c2bb = stablehlo.broadcast_in_dim %s3b4b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4c2 = stablehlo.add %s3b4c2c, %s3b4c2bb : tensor<32x256x14x14xf32>
    %s3b4n2nf = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %s3b4n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b4n2smr = stablehlo.reduce(%s3b4c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4n2sm = stablehlo.broadcast_in_dim %s3b4n2smr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n2mu = stablehlo.divide %s3b4n2sm, %s3b4n2nf : tensor<32x256x14x14xf32>
    %s3b4n2xc = stablehlo.subtract %s3b4c2, %s3b4n2mu : tensor<32x256x14x14xf32>
    %s3b4n2sq = stablehlo.multiply %s3b4n2xc, %s3b4n2xc : tensor<32x256x14x14xf32>
    %s3b4n2vsr = stablehlo.reduce(%s3b4n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4n2vs = stablehlo.broadcast_in_dim %s3b4n2vsr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n2vr = stablehlo.divide %s3b4n2vs, %s3b4n2nf : tensor<32x256x14x14xf32>
    %s3b4n2ve = stablehlo.add %s3b4n2vr, %s3b4n2ep : tensor<32x256x14x14xf32>
    %s3b4n2istd = stablehlo.rsqrt %s3b4n2ve : tensor<32x256x14x14xf32>
    %s3b4n2xh = stablehlo.multiply %s3b4n2xc, %s3b4n2istd : tensor<32x256x14x14xf32>
    %s3b4n2gb = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n2btb = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n2gx = stablehlo.multiply %s3b4n2xh, %s3b4n2gb : tensor<32x256x14x14xf32>
    %s3b4n2 = stablehlo.add %s3b4n2gx, %s3b4n2btb : tensor<32x256x14x14xf32>
    %s3b4a = stablehlo.add %s3b4n2, %s3b3o : tensor<32x256x14x14xf32>
    %s3b4oz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b4o = stablehlo.maximum %s3b4a, %s3b4oz : tensor<32x256x14x14xf32>
    %d4c1c = stablehlo.convolution(%s3b4o, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %d4c1bb = stablehlo.broadcast_in_dim %d4b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4c1 = stablehlo.add %d4c1c, %d4c1bb : tensor<32x512x7x7xf32>
    %d4n1nf = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %d4n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %d4n1smr = stablehlo.reduce(%d4c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4n1sm = stablehlo.broadcast_in_dim %d4n1smr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n1mu = stablehlo.divide %d4n1sm, %d4n1nf : tensor<32x512x7x7xf32>
    %d4n1xc = stablehlo.subtract %d4c1, %d4n1mu : tensor<32x512x7x7xf32>
    %d4n1sq = stablehlo.multiply %d4n1xc, %d4n1xc : tensor<32x512x7x7xf32>
    %d4n1vsr = stablehlo.reduce(%d4n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4n1vs = stablehlo.broadcast_in_dim %d4n1vsr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n1vr = stablehlo.divide %d4n1vs, %d4n1nf : tensor<32x512x7x7xf32>
    %d4n1ve = stablehlo.add %d4n1vr, %d4n1ep : tensor<32x512x7x7xf32>
    %d4n1istd = stablehlo.rsqrt %d4n1ve : tensor<32x512x7x7xf32>
    %d4n1xh = stablehlo.multiply %d4n1xc, %d4n1istd : tensor<32x512x7x7xf32>
    %d4n1gb = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n1btb = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n1gx = stablehlo.multiply %d4n1xh, %d4n1gb : tensor<32x512x7x7xf32>
    %d4n1 = stablehlo.add %d4n1gx, %d4n1btb : tensor<32x512x7x7xf32>
    %d4r1z = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %d4r1 = stablehlo.maximum %d4n1, %d4r1z : tensor<32x512x7x7xf32>
    %d4c2c = stablehlo.convolution(%d4r1, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %d4c2bb = stablehlo.broadcast_in_dim %d4b2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4c2 = stablehlo.add %d4c2c, %d4c2bb : tensor<32x512x7x7xf32>
    %d4n2nf = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %d4n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %d4n2smr = stablehlo.reduce(%d4c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4n2sm = stablehlo.broadcast_in_dim %d4n2smr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n2mu = stablehlo.divide %d4n2sm, %d4n2nf : tensor<32x512x7x7xf32>
    %d4n2xc = stablehlo.subtract %d4c2, %d4n2mu : tensor<32x512x7x7xf32>
    %d4n2sq = stablehlo.multiply %d4n2xc, %d4n2xc : tensor<32x512x7x7xf32>
    %d4n2vsr = stablehlo.reduce(%d4n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4n2vs = stablehlo.broadcast_in_dim %d4n2vsr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n2vr = stablehlo.divide %d4n2vs, %d4n2nf : tensor<32x512x7x7xf32>
    %d4n2ve = stablehlo.add %d4n2vr, %d4n2ep : tensor<32x512x7x7xf32>
    %d4n2istd = stablehlo.rsqrt %d4n2ve : tensor<32x512x7x7xf32>
    %d4n2xh = stablehlo.multiply %d4n2xc, %d4n2istd : tensor<32x512x7x7xf32>
    %d4n2gb = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n2btb = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n2gx = stablehlo.multiply %d4n2xh, %d4n2gb : tensor<32x512x7x7xf32>
    %d4n2 = stablehlo.add %d4n2gx, %d4n2btb : tensor<32x512x7x7xf32>
    %d4cpc = stablehlo.convolution(%s3b4o, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %d4cpbb = stablehlo.broadcast_in_dim %d4bp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4cp = stablehlo.add %d4cpc, %d4cpbb : tensor<32x512x7x7xf32>
    %d4npnf = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %d4npep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %d4npsmr = stablehlo.reduce(%d4cp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4npsm = stablehlo.broadcast_in_dim %d4npsmr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4npmu = stablehlo.divide %d4npsm, %d4npnf : tensor<32x512x7x7xf32>
    %d4npxc = stablehlo.subtract %d4cp, %d4npmu : tensor<32x512x7x7xf32>
    %d4npsq = stablehlo.multiply %d4npxc, %d4npxc : tensor<32x512x7x7xf32>
    %d4npvsr = stablehlo.reduce(%d4npsq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4npvs = stablehlo.broadcast_in_dim %d4npvsr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4npvr = stablehlo.divide %d4npvs, %d4npnf : tensor<32x512x7x7xf32>
    %d4npve = stablehlo.add %d4npvr, %d4npep : tensor<32x512x7x7xf32>
    %d4npistd = stablehlo.rsqrt %d4npve : tensor<32x512x7x7xf32>
    %d4npxh = stablehlo.multiply %d4npxc, %d4npistd : tensor<32x512x7x7xf32>
    %d4npgb = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4npbtb = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4npgx = stablehlo.multiply %d4npxh, %d4npgb : tensor<32x512x7x7xf32>
    %d4np = stablehlo.add %d4npgx, %d4npbtb : tensor<32x512x7x7xf32>
    %d4a = stablehlo.add %d4n2, %d4np : tensor<32x512x7x7xf32>
    %d4oz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %d4o = stablehlo.maximum %d4a, %d4oz : tensor<32x512x7x7xf32>
    %s4b0c1c = stablehlo.convolution(%d4o, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b0c1bb = stablehlo.broadcast_in_dim %s4b0b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0c1 = stablehlo.add %s4b0c1c, %s4b0c1bb : tensor<32x512x7x7xf32>
    %s4b0n1nf = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %s4b0n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %s4b0n1smr = stablehlo.reduce(%s4b0c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0n1sm = stablehlo.broadcast_in_dim %s4b0n1smr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n1mu = stablehlo.divide %s4b0n1sm, %s4b0n1nf : tensor<32x512x7x7xf32>
    %s4b0n1xc = stablehlo.subtract %s4b0c1, %s4b0n1mu : tensor<32x512x7x7xf32>
    %s4b0n1sq = stablehlo.multiply %s4b0n1xc, %s4b0n1xc : tensor<32x512x7x7xf32>
    %s4b0n1vsr = stablehlo.reduce(%s4b0n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0n1vs = stablehlo.broadcast_in_dim %s4b0n1vsr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n1vr = stablehlo.divide %s4b0n1vs, %s4b0n1nf : tensor<32x512x7x7xf32>
    %s4b0n1ve = stablehlo.add %s4b0n1vr, %s4b0n1ep : tensor<32x512x7x7xf32>
    %s4b0n1istd = stablehlo.rsqrt %s4b0n1ve : tensor<32x512x7x7xf32>
    %s4b0n1xh = stablehlo.multiply %s4b0n1xc, %s4b0n1istd : tensor<32x512x7x7xf32>
    %s4b0n1gb = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n1btb = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n1gx = stablehlo.multiply %s4b0n1xh, %s4b0n1gb : tensor<32x512x7x7xf32>
    %s4b0n1 = stablehlo.add %s4b0n1gx, %s4b0n1btb : tensor<32x512x7x7xf32>
    %s4b0r1z = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b0r1 = stablehlo.maximum %s4b0n1, %s4b0r1z : tensor<32x512x7x7xf32>
    %s4b0c2c = stablehlo.convolution(%s4b0r1, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b0c2bb = stablehlo.broadcast_in_dim %s4b0b2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0c2 = stablehlo.add %s4b0c2c, %s4b0c2bb : tensor<32x512x7x7xf32>
    %s4b0n2nf = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %s4b0n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %s4b0n2smr = stablehlo.reduce(%s4b0c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0n2sm = stablehlo.broadcast_in_dim %s4b0n2smr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n2mu = stablehlo.divide %s4b0n2sm, %s4b0n2nf : tensor<32x512x7x7xf32>
    %s4b0n2xc = stablehlo.subtract %s4b0c2, %s4b0n2mu : tensor<32x512x7x7xf32>
    %s4b0n2sq = stablehlo.multiply %s4b0n2xc, %s4b0n2xc : tensor<32x512x7x7xf32>
    %s4b0n2vsr = stablehlo.reduce(%s4b0n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0n2vs = stablehlo.broadcast_in_dim %s4b0n2vsr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n2vr = stablehlo.divide %s4b0n2vs, %s4b0n2nf : tensor<32x512x7x7xf32>
    %s4b0n2ve = stablehlo.add %s4b0n2vr, %s4b0n2ep : tensor<32x512x7x7xf32>
    %s4b0n2istd = stablehlo.rsqrt %s4b0n2ve : tensor<32x512x7x7xf32>
    %s4b0n2xh = stablehlo.multiply %s4b0n2xc, %s4b0n2istd : tensor<32x512x7x7xf32>
    %s4b0n2gb = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n2btb = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n2gx = stablehlo.multiply %s4b0n2xh, %s4b0n2gb : tensor<32x512x7x7xf32>
    %s4b0n2 = stablehlo.add %s4b0n2gx, %s4b0n2btb : tensor<32x512x7x7xf32>
    %s4b0a = stablehlo.add %s4b0n2, %d4o : tensor<32x512x7x7xf32>
    %s4b0oz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b0o = stablehlo.maximum %s4b0a, %s4b0oz : tensor<32x512x7x7xf32>
    %s4b1c1c = stablehlo.convolution(%s4b0o, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b1c1bb = stablehlo.broadcast_in_dim %s4b1b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1c1 = stablehlo.add %s4b1c1c, %s4b1c1bb : tensor<32x512x7x7xf32>
    %s4b1n1nf = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %s4b1n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %s4b1n1smr = stablehlo.reduce(%s4b1c1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1n1sm = stablehlo.broadcast_in_dim %s4b1n1smr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n1mu = stablehlo.divide %s4b1n1sm, %s4b1n1nf : tensor<32x512x7x7xf32>
    %s4b1n1xc = stablehlo.subtract %s4b1c1, %s4b1n1mu : tensor<32x512x7x7xf32>
    %s4b1n1sq = stablehlo.multiply %s4b1n1xc, %s4b1n1xc : tensor<32x512x7x7xf32>
    %s4b1n1vsr = stablehlo.reduce(%s4b1n1sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1n1vs = stablehlo.broadcast_in_dim %s4b1n1vsr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n1vr = stablehlo.divide %s4b1n1vs, %s4b1n1nf : tensor<32x512x7x7xf32>
    %s4b1n1ve = stablehlo.add %s4b1n1vr, %s4b1n1ep : tensor<32x512x7x7xf32>
    %s4b1n1istd = stablehlo.rsqrt %s4b1n1ve : tensor<32x512x7x7xf32>
    %s4b1n1xh = stablehlo.multiply %s4b1n1xc, %s4b1n1istd : tensor<32x512x7x7xf32>
    %s4b1n1gb = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n1btb = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n1gx = stablehlo.multiply %s4b1n1xh, %s4b1n1gb : tensor<32x512x7x7xf32>
    %s4b1n1 = stablehlo.add %s4b1n1gx, %s4b1n1btb : tensor<32x512x7x7xf32>
    %s4b1r1z = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b1r1 = stablehlo.maximum %s4b1n1, %s4b1r1z : tensor<32x512x7x7xf32>
    %s4b1c2c = stablehlo.convolution(%s4b1r1, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b1c2bb = stablehlo.broadcast_in_dim %s4b1b2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1c2 = stablehlo.add %s4b1c2c, %s4b1c2bb : tensor<32x512x7x7xf32>
    %s4b1n2nf = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %s4b1n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %s4b1n2smr = stablehlo.reduce(%s4b1c2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1n2sm = stablehlo.broadcast_in_dim %s4b1n2smr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n2mu = stablehlo.divide %s4b1n2sm, %s4b1n2nf : tensor<32x512x7x7xf32>
    %s4b1n2xc = stablehlo.subtract %s4b1c2, %s4b1n2mu : tensor<32x512x7x7xf32>
    %s4b1n2sq = stablehlo.multiply %s4b1n2xc, %s4b1n2xc : tensor<32x512x7x7xf32>
    %s4b1n2vsr = stablehlo.reduce(%s4b1n2sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1n2vs = stablehlo.broadcast_in_dim %s4b1n2vsr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n2vr = stablehlo.divide %s4b1n2vs, %s4b1n2nf : tensor<32x512x7x7xf32>
    %s4b1n2ve = stablehlo.add %s4b1n2vr, %s4b1n2ep : tensor<32x512x7x7xf32>
    %s4b1n2istd = stablehlo.rsqrt %s4b1n2ve : tensor<32x512x7x7xf32>
    %s4b1n2xh = stablehlo.multiply %s4b1n2xc, %s4b1n2istd : tensor<32x512x7x7xf32>
    %s4b1n2gb = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n2btb = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n2gx = stablehlo.multiply %s4b1n2xh, %s4b1n2gb : tensor<32x512x7x7xf32>
    %s4b1n2 = stablehlo.add %s4b1n2gx, %s4b1n2btb : tensor<32x512x7x7xf32>
    %s4b1a = stablehlo.add %s4b1n2, %s4b0o : tensor<32x512x7x7xf32>
    %s4b1oz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b1o = stablehlo.maximum %s4b1a, %s4b1oz : tensor<32x512x7x7xf32>
    %gaps = stablehlo.reduce(%s4b1o init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %gapnf = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %gap = stablehlo.divide %gaps, %gapnf : tensor<32x512xf32>
    %ld = stablehlo.dot_general %gap, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<512x10xf32>) -> tensor<32x10xf32>
    %ldb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %logits = stablehlo.add %ld, %ldb : tensor<32x10xf32>
    %le = stablehlo.exponential %logits : tensor<32x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<32x10xf32>
    %dyr = stablehlo.subtract %lsm, %onehot : tensor<32x10xf32>
    %bnc = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %dy = stablehlo.divide %dyr, %bnc : tensor<32x10xf32>
    %dgap = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<512x10xf32>) -> tensor<32x512xf32>
    %dWd = stablehlo.dot_general %gap, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<32x10xf32>) -> tensor<512x10xf32>
    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dgnf = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %dgs = stablehlo.divide %dgap, %dgnf : tensor<32x512xf32>
    %dgapin = stablehlo.broadcast_in_dim %dgs, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1daz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b1dam = stablehlo.compare GT, %s4b1a, %s4b1daz : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %s4b1da = stablehlo.select %s4b1dam, %dgapin, %s4b1daz : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %s4b1dn2dxh = stablehlo.multiply %s4b1n2gb, %s4b1da : tensor<32x512x7x7xf32>
    %s4b1dn2sdxr = stablehlo.reduce(%s4b1dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dn2sdx = stablehlo.broadcast_in_dim %s4b1dn2sdxr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1dn2xd = stablehlo.multiply %s4b1n2xh, %s4b1dn2dxh : tensor<32x512x7x7xf32>
    %s4b1dn2sxdr = stablehlo.reduce(%s4b1dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dn2sxd = stablehlo.broadcast_in_dim %s4b1dn2sxdr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1dn2t1 = stablehlo.multiply %s4b1dn2dxh, %s4b1n2nf : tensor<32x512x7x7xf32>
    %s4b1dn2i1 = stablehlo.subtract %s4b1dn2t1, %s4b1dn2sdx : tensor<32x512x7x7xf32>
    %s4b1dn2xs = stablehlo.multiply %s4b1n2xh, %s4b1dn2sxd : tensor<32x512x7x7xf32>
    %s4b1dn2i2 = stablehlo.subtract %s4b1dn2i1, %s4b1dn2xs : tensor<32x512x7x7xf32>
    %s4b1dn2sN = stablehlo.divide %s4b1n2istd, %s4b1n2nf : tensor<32x512x7x7xf32>
    %s4b1dn2 = stablehlo.multiply %s4b1dn2sN, %s4b1dn2i2 : tensor<32x512x7x7xf32>
    %s4b1dn2dgp = stablehlo.multiply %s4b1da, %s4b1n2xh : tensor<32x512x7x7xf32>
    %s4b1dn2dg = stablehlo.reduce(%s4b1dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dn2db = stablehlo.reduce(%s4b1da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dc2t = stablehlo.transpose %s4b1W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b1dc2r = stablehlo.reverse %s4b1dc2t, dims = [2, 3] : tensor<512x512x3x3xf32>
    %s4b1dc2 = stablehlo.convolution(%s4b1dn2, %s4b1dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b1dW2xt = stablehlo.transpose %s4b1r1, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b1dW2dt = stablehlo.transpose %s4b1dn2, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b1dW2raw = stablehlo.convolution(%s4b1dW2xt, %s4b1dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %s4b1dW2 = stablehlo.transpose %s4b1dW2raw, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b1db2 = stablehlo.reduce(%s4b1dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dr1z = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b1dr1m = stablehlo.compare GT, %s4b1n1, %s4b1dr1z : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %s4b1dr1 = stablehlo.select %s4b1dr1m, %s4b1dc2, %s4b1dr1z : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %s4b1dn1dxh = stablehlo.multiply %s4b1n1gb, %s4b1dr1 : tensor<32x512x7x7xf32>
    %s4b1dn1sdxr = stablehlo.reduce(%s4b1dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dn1sdx = stablehlo.broadcast_in_dim %s4b1dn1sdxr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1dn1xd = stablehlo.multiply %s4b1n1xh, %s4b1dn1dxh : tensor<32x512x7x7xf32>
    %s4b1dn1sxdr = stablehlo.reduce(%s4b1dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dn1sxd = stablehlo.broadcast_in_dim %s4b1dn1sxdr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1dn1t1 = stablehlo.multiply %s4b1dn1dxh, %s4b1n1nf : tensor<32x512x7x7xf32>
    %s4b1dn1i1 = stablehlo.subtract %s4b1dn1t1, %s4b1dn1sdx : tensor<32x512x7x7xf32>
    %s4b1dn1xs = stablehlo.multiply %s4b1n1xh, %s4b1dn1sxd : tensor<32x512x7x7xf32>
    %s4b1dn1i2 = stablehlo.subtract %s4b1dn1i1, %s4b1dn1xs : tensor<32x512x7x7xf32>
    %s4b1dn1sN = stablehlo.divide %s4b1n1istd, %s4b1n1nf : tensor<32x512x7x7xf32>
    %s4b1dn1 = stablehlo.multiply %s4b1dn1sN, %s4b1dn1i2 : tensor<32x512x7x7xf32>
    %s4b1dn1dgp = stablehlo.multiply %s4b1dr1, %s4b1n1xh : tensor<32x512x7x7xf32>
    %s4b1dn1dg = stablehlo.reduce(%s4b1dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dn1db = stablehlo.reduce(%s4b1dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dc1t = stablehlo.transpose %s4b1W1, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b1dc1r = stablehlo.reverse %s4b1dc1t, dims = [2, 3] : tensor<512x512x3x3xf32>
    %s4b1dc1 = stablehlo.convolution(%s4b1dn1, %s4b1dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b1dW1xt = stablehlo.transpose %s4b0o, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b1dW1dt = stablehlo.transpose %s4b1dn1, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b1dW1raw = stablehlo.convolution(%s4b1dW1xt, %s4b1dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %s4b1dW1 = stablehlo.transpose %s4b1dW1raw, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b1db1 = stablehlo.reduce(%s4b1dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dx = stablehlo.add %s4b1dc1, %s4b1da : tensor<32x512x7x7xf32>
    %s4b0daz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b0dam = stablehlo.compare GT, %s4b0a, %s4b0daz : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %s4b0da = stablehlo.select %s4b0dam, %s4b1dx, %s4b0daz : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %s4b0dn2dxh = stablehlo.multiply %s4b0n2gb, %s4b0da : tensor<32x512x7x7xf32>
    %s4b0dn2sdxr = stablehlo.reduce(%s4b0dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dn2sdx = stablehlo.broadcast_in_dim %s4b0dn2sdxr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0dn2xd = stablehlo.multiply %s4b0n2xh, %s4b0dn2dxh : tensor<32x512x7x7xf32>
    %s4b0dn2sxdr = stablehlo.reduce(%s4b0dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dn2sxd = stablehlo.broadcast_in_dim %s4b0dn2sxdr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0dn2t1 = stablehlo.multiply %s4b0dn2dxh, %s4b0n2nf : tensor<32x512x7x7xf32>
    %s4b0dn2i1 = stablehlo.subtract %s4b0dn2t1, %s4b0dn2sdx : tensor<32x512x7x7xf32>
    %s4b0dn2xs = stablehlo.multiply %s4b0n2xh, %s4b0dn2sxd : tensor<32x512x7x7xf32>
    %s4b0dn2i2 = stablehlo.subtract %s4b0dn2i1, %s4b0dn2xs : tensor<32x512x7x7xf32>
    %s4b0dn2sN = stablehlo.divide %s4b0n2istd, %s4b0n2nf : tensor<32x512x7x7xf32>
    %s4b0dn2 = stablehlo.multiply %s4b0dn2sN, %s4b0dn2i2 : tensor<32x512x7x7xf32>
    %s4b0dn2dgp = stablehlo.multiply %s4b0da, %s4b0n2xh : tensor<32x512x7x7xf32>
    %s4b0dn2dg = stablehlo.reduce(%s4b0dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dn2db = stablehlo.reduce(%s4b0da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dc2t = stablehlo.transpose %s4b0W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b0dc2r = stablehlo.reverse %s4b0dc2t, dims = [2, 3] : tensor<512x512x3x3xf32>
    %s4b0dc2 = stablehlo.convolution(%s4b0dn2, %s4b0dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b0dW2xt = stablehlo.transpose %s4b0r1, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b0dW2dt = stablehlo.transpose %s4b0dn2, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b0dW2raw = stablehlo.convolution(%s4b0dW2xt, %s4b0dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %s4b0dW2 = stablehlo.transpose %s4b0dW2raw, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b0db2 = stablehlo.reduce(%s4b0dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dr1z = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b0dr1m = stablehlo.compare GT, %s4b0n1, %s4b0dr1z : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %s4b0dr1 = stablehlo.select %s4b0dr1m, %s4b0dc2, %s4b0dr1z : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %s4b0dn1dxh = stablehlo.multiply %s4b0n1gb, %s4b0dr1 : tensor<32x512x7x7xf32>
    %s4b0dn1sdxr = stablehlo.reduce(%s4b0dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dn1sdx = stablehlo.broadcast_in_dim %s4b0dn1sdxr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0dn1xd = stablehlo.multiply %s4b0n1xh, %s4b0dn1dxh : tensor<32x512x7x7xf32>
    %s4b0dn1sxdr = stablehlo.reduce(%s4b0dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dn1sxd = stablehlo.broadcast_in_dim %s4b0dn1sxdr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0dn1t1 = stablehlo.multiply %s4b0dn1dxh, %s4b0n1nf : tensor<32x512x7x7xf32>
    %s4b0dn1i1 = stablehlo.subtract %s4b0dn1t1, %s4b0dn1sdx : tensor<32x512x7x7xf32>
    %s4b0dn1xs = stablehlo.multiply %s4b0n1xh, %s4b0dn1sxd : tensor<32x512x7x7xf32>
    %s4b0dn1i2 = stablehlo.subtract %s4b0dn1i1, %s4b0dn1xs : tensor<32x512x7x7xf32>
    %s4b0dn1sN = stablehlo.divide %s4b0n1istd, %s4b0n1nf : tensor<32x512x7x7xf32>
    %s4b0dn1 = stablehlo.multiply %s4b0dn1sN, %s4b0dn1i2 : tensor<32x512x7x7xf32>
    %s4b0dn1dgp = stablehlo.multiply %s4b0dr1, %s4b0n1xh : tensor<32x512x7x7xf32>
    %s4b0dn1dg = stablehlo.reduce(%s4b0dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dn1db = stablehlo.reduce(%s4b0dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dc1t = stablehlo.transpose %s4b0W1, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b0dc1r = stablehlo.reverse %s4b0dc1t, dims = [2, 3] : tensor<512x512x3x3xf32>
    %s4b0dc1 = stablehlo.convolution(%s4b0dn1, %s4b0dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b0dW1xt = stablehlo.transpose %d4o, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b0dW1dt = stablehlo.transpose %s4b0dn1, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b0dW1raw = stablehlo.convolution(%s4b0dW1xt, %s4b0dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %s4b0dW1 = stablehlo.transpose %s4b0dW1raw, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b0db1 = stablehlo.reduce(%s4b0dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dx = stablehlo.add %s4b0dc1, %s4b0da : tensor<32x512x7x7xf32>
    %d4daz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %d4dam = stablehlo.compare GT, %d4a, %d4daz : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %d4da = stablehlo.select %d4dam, %s4b0dx, %d4daz : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %d4dn2dxh = stablehlo.multiply %d4n2gb, %d4da : tensor<32x512x7x7xf32>
    %d4dn2sdxr = stablehlo.reduce(%d4dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dn2sdx = stablehlo.broadcast_in_dim %d4dn2sdxr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4dn2xd = stablehlo.multiply %d4n2xh, %d4dn2dxh : tensor<32x512x7x7xf32>
    %d4dn2sxdr = stablehlo.reduce(%d4dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dn2sxd = stablehlo.broadcast_in_dim %d4dn2sxdr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4dn2t1 = stablehlo.multiply %d4dn2dxh, %d4n2nf : tensor<32x512x7x7xf32>
    %d4dn2i1 = stablehlo.subtract %d4dn2t1, %d4dn2sdx : tensor<32x512x7x7xf32>
    %d4dn2xs = stablehlo.multiply %d4n2xh, %d4dn2sxd : tensor<32x512x7x7xf32>
    %d4dn2i2 = stablehlo.subtract %d4dn2i1, %d4dn2xs : tensor<32x512x7x7xf32>
    %d4dn2sN = stablehlo.divide %d4n2istd, %d4n2nf : tensor<32x512x7x7xf32>
    %d4dn2 = stablehlo.multiply %d4dn2sN, %d4dn2i2 : tensor<32x512x7x7xf32>
    %d4dn2dgp = stablehlo.multiply %d4da, %d4n2xh : tensor<32x512x7x7xf32>
    %d4dn2dg = stablehlo.reduce(%d4dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dn2db = stablehlo.reduce(%d4da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dc2t = stablehlo.transpose %d4W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %d4dc2r = stablehlo.reverse %d4dc2t, dims = [2, 3] : tensor<512x512x3x3xf32>
    %d4dc2 = stablehlo.convolution(%d4dn2, %d4dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %d4dW2xt = stablehlo.transpose %d4r1, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %d4dW2dt = stablehlo.transpose %d4dn2, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %d4dW2raw = stablehlo.convolution(%d4dW2xt, %d4dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %d4dW2 = stablehlo.transpose %d4dW2raw, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %d4db2 = stablehlo.reduce(%d4dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dr1z = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %d4dr1m = stablehlo.compare GT, %d4n1, %d4dr1z : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %d4dr1 = stablehlo.select %d4dr1m, %d4dc2, %d4dr1z : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %d4dn1dxh = stablehlo.multiply %d4n1gb, %d4dr1 : tensor<32x512x7x7xf32>
    %d4dn1sdxr = stablehlo.reduce(%d4dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dn1sdx = stablehlo.broadcast_in_dim %d4dn1sdxr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4dn1xd = stablehlo.multiply %d4n1xh, %d4dn1dxh : tensor<32x512x7x7xf32>
    %d4dn1sxdr = stablehlo.reduce(%d4dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dn1sxd = stablehlo.broadcast_in_dim %d4dn1sxdr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4dn1t1 = stablehlo.multiply %d4dn1dxh, %d4n1nf : tensor<32x512x7x7xf32>
    %d4dn1i1 = stablehlo.subtract %d4dn1t1, %d4dn1sdx : tensor<32x512x7x7xf32>
    %d4dn1xs = stablehlo.multiply %d4n1xh, %d4dn1sxd : tensor<32x512x7x7xf32>
    %d4dn1i2 = stablehlo.subtract %d4dn1i1, %d4dn1xs : tensor<32x512x7x7xf32>
    %d4dn1sN = stablehlo.divide %d4n1istd, %d4n1nf : tensor<32x512x7x7xf32>
    %d4dn1 = stablehlo.multiply %d4dn1sN, %d4dn1i2 : tensor<32x512x7x7xf32>
    %d4dn1dgp = stablehlo.multiply %d4dr1, %d4n1xh : tensor<32x512x7x7xf32>
    %d4dn1dg = stablehlo.reduce(%d4dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dn1db = stablehlo.reduce(%d4dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dc1u = stablehlo.pad %d4dn1, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %d4dc1t = stablehlo.transpose %d4W1, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %d4dc1r = stablehlo.reverse %d4dc1t, dims = [2, 3] : tensor<256x512x3x3xf32>
    %d4dc1 = stablehlo.convolution(%d4dc1u, %d4dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %d4dW1u = stablehlo.pad %d4dn1, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %d4dW1xt = stablehlo.transpose %s3b4o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %d4dW1dt = stablehlo.transpose %d4dW1u, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %d4dW1raw = stablehlo.convolution(%d4dW1xt, %d4dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %d4dW1 = stablehlo.transpose %d4dW1raw, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %d4db1 = stablehlo.reduce(%d4dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dnpdxh = stablehlo.multiply %d4npgb, %d4da : tensor<32x512x7x7xf32>
    %d4dnpsdxr = stablehlo.reduce(%d4dnpdxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dnpsdx = stablehlo.broadcast_in_dim %d4dnpsdxr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4dnpxd = stablehlo.multiply %d4npxh, %d4dnpdxh : tensor<32x512x7x7xf32>
    %d4dnpsxdr = stablehlo.reduce(%d4dnpxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dnpsxd = stablehlo.broadcast_in_dim %d4dnpsxdr, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4dnpt1 = stablehlo.multiply %d4dnpdxh, %d4npnf : tensor<32x512x7x7xf32>
    %d4dnpi1 = stablehlo.subtract %d4dnpt1, %d4dnpsdx : tensor<32x512x7x7xf32>
    %d4dnpxs = stablehlo.multiply %d4npxh, %d4dnpsxd : tensor<32x512x7x7xf32>
    %d4dnpi2 = stablehlo.subtract %d4dnpi1, %d4dnpxs : tensor<32x512x7x7xf32>
    %d4dnpsN = stablehlo.divide %d4npistd, %d4npnf : tensor<32x512x7x7xf32>
    %d4dnp = stablehlo.multiply %d4dnpsN, %d4dnpi2 : tensor<32x512x7x7xf32>
    %d4dnpdgp = stablehlo.multiply %d4da, %d4npxh : tensor<32x512x7x7xf32>
    %d4dnpdg = stablehlo.reduce(%d4dnpdgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dnpdb = stablehlo.reduce(%d4da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dcpu = stablehlo.pad %d4dnp, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %d4dcpt = stablehlo.transpose %d4Wp, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %d4dcpr = stablehlo.reverse %d4dcpt, dims = [2, 3] : tensor<256x512x3x3xf32>
    %d4dcp = stablehlo.convolution(%d4dcpu, %d4dcpr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %d4dWpu = stablehlo.pad %d4dnp, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %d4dWpxt = stablehlo.transpose %s3b4o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %d4dWpdt = stablehlo.transpose %d4dWpu, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %d4dWpraw = stablehlo.convolution(%d4dWpxt, %d4dWpdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %d4dWp = stablehlo.transpose %d4dWpraw, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %d4dbp = stablehlo.reduce(%d4dnp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dx = stablehlo.add %d4dc1, %d4dcp : tensor<32x256x14x14xf32>
    %s3b4daz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b4dam = stablehlo.compare GT, %s3b4a, %s3b4daz : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b4da = stablehlo.select %s3b4dam, %d4dx, %s3b4daz : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b4dn2dxh = stablehlo.multiply %s3b4n2gb, %s3b4da : tensor<32x256x14x14xf32>
    %s3b4dn2sdxr = stablehlo.reduce(%s3b4dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dn2sdx = stablehlo.broadcast_in_dim %s3b4dn2sdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4dn2xd = stablehlo.multiply %s3b4n2xh, %s3b4dn2dxh : tensor<32x256x14x14xf32>
    %s3b4dn2sxdr = stablehlo.reduce(%s3b4dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dn2sxd = stablehlo.broadcast_in_dim %s3b4dn2sxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4dn2t1 = stablehlo.multiply %s3b4dn2dxh, %s3b4n2nf : tensor<32x256x14x14xf32>
    %s3b4dn2i1 = stablehlo.subtract %s3b4dn2t1, %s3b4dn2sdx : tensor<32x256x14x14xf32>
    %s3b4dn2xs = stablehlo.multiply %s3b4n2xh, %s3b4dn2sxd : tensor<32x256x14x14xf32>
    %s3b4dn2i2 = stablehlo.subtract %s3b4dn2i1, %s3b4dn2xs : tensor<32x256x14x14xf32>
    %s3b4dn2sN = stablehlo.divide %s3b4n2istd, %s3b4n2nf : tensor<32x256x14x14xf32>
    %s3b4dn2 = stablehlo.multiply %s3b4dn2sN, %s3b4dn2i2 : tensor<32x256x14x14xf32>
    %s3b4dn2dgp = stablehlo.multiply %s3b4da, %s3b4n2xh : tensor<32x256x14x14xf32>
    %s3b4dn2dg = stablehlo.reduce(%s3b4dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dn2db = stablehlo.reduce(%s3b4da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dc2t = stablehlo.transpose %s3b4W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b4dc2r = stablehlo.reverse %s3b4dc2t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b4dc2 = stablehlo.convolution(%s3b4dn2, %s3b4dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b4dW2xt = stablehlo.transpose %s3b4r1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b4dW2dt = stablehlo.transpose %s3b4dn2, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b4dW2raw = stablehlo.convolution(%s3b4dW2xt, %s3b4dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b4dW2 = stablehlo.transpose %s3b4dW2raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b4db2 = stablehlo.reduce(%s3b4dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dr1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b4dr1m = stablehlo.compare GT, %s3b4n1, %s3b4dr1z : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b4dr1 = stablehlo.select %s3b4dr1m, %s3b4dc2, %s3b4dr1z : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b4dn1dxh = stablehlo.multiply %s3b4n1gb, %s3b4dr1 : tensor<32x256x14x14xf32>
    %s3b4dn1sdxr = stablehlo.reduce(%s3b4dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dn1sdx = stablehlo.broadcast_in_dim %s3b4dn1sdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4dn1xd = stablehlo.multiply %s3b4n1xh, %s3b4dn1dxh : tensor<32x256x14x14xf32>
    %s3b4dn1sxdr = stablehlo.reduce(%s3b4dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dn1sxd = stablehlo.broadcast_in_dim %s3b4dn1sxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4dn1t1 = stablehlo.multiply %s3b4dn1dxh, %s3b4n1nf : tensor<32x256x14x14xf32>
    %s3b4dn1i1 = stablehlo.subtract %s3b4dn1t1, %s3b4dn1sdx : tensor<32x256x14x14xf32>
    %s3b4dn1xs = stablehlo.multiply %s3b4n1xh, %s3b4dn1sxd : tensor<32x256x14x14xf32>
    %s3b4dn1i2 = stablehlo.subtract %s3b4dn1i1, %s3b4dn1xs : tensor<32x256x14x14xf32>
    %s3b4dn1sN = stablehlo.divide %s3b4n1istd, %s3b4n1nf : tensor<32x256x14x14xf32>
    %s3b4dn1 = stablehlo.multiply %s3b4dn1sN, %s3b4dn1i2 : tensor<32x256x14x14xf32>
    %s3b4dn1dgp = stablehlo.multiply %s3b4dr1, %s3b4n1xh : tensor<32x256x14x14xf32>
    %s3b4dn1dg = stablehlo.reduce(%s3b4dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dn1db = stablehlo.reduce(%s3b4dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dc1t = stablehlo.transpose %s3b4W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b4dc1r = stablehlo.reverse %s3b4dc1t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b4dc1 = stablehlo.convolution(%s3b4dn1, %s3b4dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b4dW1xt = stablehlo.transpose %s3b3o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b4dW1dt = stablehlo.transpose %s3b4dn1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b4dW1raw = stablehlo.convolution(%s3b4dW1xt, %s3b4dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b4dW1 = stablehlo.transpose %s3b4dW1raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b4db1 = stablehlo.reduce(%s3b4dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dx = stablehlo.add %s3b4dc1, %s3b4da : tensor<32x256x14x14xf32>
    %s3b3daz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b3dam = stablehlo.compare GT, %s3b3a, %s3b3daz : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b3da = stablehlo.select %s3b3dam, %s3b4dx, %s3b3daz : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b3dn2dxh = stablehlo.multiply %s3b3n2gb, %s3b3da : tensor<32x256x14x14xf32>
    %s3b3dn2sdxr = stablehlo.reduce(%s3b3dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dn2sdx = stablehlo.broadcast_in_dim %s3b3dn2sdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3dn2xd = stablehlo.multiply %s3b3n2xh, %s3b3dn2dxh : tensor<32x256x14x14xf32>
    %s3b3dn2sxdr = stablehlo.reduce(%s3b3dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dn2sxd = stablehlo.broadcast_in_dim %s3b3dn2sxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3dn2t1 = stablehlo.multiply %s3b3dn2dxh, %s3b3n2nf : tensor<32x256x14x14xf32>
    %s3b3dn2i1 = stablehlo.subtract %s3b3dn2t1, %s3b3dn2sdx : tensor<32x256x14x14xf32>
    %s3b3dn2xs = stablehlo.multiply %s3b3n2xh, %s3b3dn2sxd : tensor<32x256x14x14xf32>
    %s3b3dn2i2 = stablehlo.subtract %s3b3dn2i1, %s3b3dn2xs : tensor<32x256x14x14xf32>
    %s3b3dn2sN = stablehlo.divide %s3b3n2istd, %s3b3n2nf : tensor<32x256x14x14xf32>
    %s3b3dn2 = stablehlo.multiply %s3b3dn2sN, %s3b3dn2i2 : tensor<32x256x14x14xf32>
    %s3b3dn2dgp = stablehlo.multiply %s3b3da, %s3b3n2xh : tensor<32x256x14x14xf32>
    %s3b3dn2dg = stablehlo.reduce(%s3b3dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dn2db = stablehlo.reduce(%s3b3da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dc2t = stablehlo.transpose %s3b3W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b3dc2r = stablehlo.reverse %s3b3dc2t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b3dc2 = stablehlo.convolution(%s3b3dn2, %s3b3dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b3dW2xt = stablehlo.transpose %s3b3r1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b3dW2dt = stablehlo.transpose %s3b3dn2, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b3dW2raw = stablehlo.convolution(%s3b3dW2xt, %s3b3dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b3dW2 = stablehlo.transpose %s3b3dW2raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b3db2 = stablehlo.reduce(%s3b3dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dr1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b3dr1m = stablehlo.compare GT, %s3b3n1, %s3b3dr1z : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b3dr1 = stablehlo.select %s3b3dr1m, %s3b3dc2, %s3b3dr1z : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b3dn1dxh = stablehlo.multiply %s3b3n1gb, %s3b3dr1 : tensor<32x256x14x14xf32>
    %s3b3dn1sdxr = stablehlo.reduce(%s3b3dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dn1sdx = stablehlo.broadcast_in_dim %s3b3dn1sdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3dn1xd = stablehlo.multiply %s3b3n1xh, %s3b3dn1dxh : tensor<32x256x14x14xf32>
    %s3b3dn1sxdr = stablehlo.reduce(%s3b3dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dn1sxd = stablehlo.broadcast_in_dim %s3b3dn1sxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3dn1t1 = stablehlo.multiply %s3b3dn1dxh, %s3b3n1nf : tensor<32x256x14x14xf32>
    %s3b3dn1i1 = stablehlo.subtract %s3b3dn1t1, %s3b3dn1sdx : tensor<32x256x14x14xf32>
    %s3b3dn1xs = stablehlo.multiply %s3b3n1xh, %s3b3dn1sxd : tensor<32x256x14x14xf32>
    %s3b3dn1i2 = stablehlo.subtract %s3b3dn1i1, %s3b3dn1xs : tensor<32x256x14x14xf32>
    %s3b3dn1sN = stablehlo.divide %s3b3n1istd, %s3b3n1nf : tensor<32x256x14x14xf32>
    %s3b3dn1 = stablehlo.multiply %s3b3dn1sN, %s3b3dn1i2 : tensor<32x256x14x14xf32>
    %s3b3dn1dgp = stablehlo.multiply %s3b3dr1, %s3b3n1xh : tensor<32x256x14x14xf32>
    %s3b3dn1dg = stablehlo.reduce(%s3b3dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dn1db = stablehlo.reduce(%s3b3dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dc1t = stablehlo.transpose %s3b3W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b3dc1r = stablehlo.reverse %s3b3dc1t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b3dc1 = stablehlo.convolution(%s3b3dn1, %s3b3dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b3dW1xt = stablehlo.transpose %s3b2o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b3dW1dt = stablehlo.transpose %s3b3dn1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b3dW1raw = stablehlo.convolution(%s3b3dW1xt, %s3b3dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b3dW1 = stablehlo.transpose %s3b3dW1raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b3db1 = stablehlo.reduce(%s3b3dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dx = stablehlo.add %s3b3dc1, %s3b3da : tensor<32x256x14x14xf32>
    %s3b2daz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b2dam = stablehlo.compare GT, %s3b2a, %s3b2daz : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b2da = stablehlo.select %s3b2dam, %s3b3dx, %s3b2daz : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b2dn2dxh = stablehlo.multiply %s3b2n2gb, %s3b2da : tensor<32x256x14x14xf32>
    %s3b2dn2sdxr = stablehlo.reduce(%s3b2dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dn2sdx = stablehlo.broadcast_in_dim %s3b2dn2sdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2dn2xd = stablehlo.multiply %s3b2n2xh, %s3b2dn2dxh : tensor<32x256x14x14xf32>
    %s3b2dn2sxdr = stablehlo.reduce(%s3b2dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dn2sxd = stablehlo.broadcast_in_dim %s3b2dn2sxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2dn2t1 = stablehlo.multiply %s3b2dn2dxh, %s3b2n2nf : tensor<32x256x14x14xf32>
    %s3b2dn2i1 = stablehlo.subtract %s3b2dn2t1, %s3b2dn2sdx : tensor<32x256x14x14xf32>
    %s3b2dn2xs = stablehlo.multiply %s3b2n2xh, %s3b2dn2sxd : tensor<32x256x14x14xf32>
    %s3b2dn2i2 = stablehlo.subtract %s3b2dn2i1, %s3b2dn2xs : tensor<32x256x14x14xf32>
    %s3b2dn2sN = stablehlo.divide %s3b2n2istd, %s3b2n2nf : tensor<32x256x14x14xf32>
    %s3b2dn2 = stablehlo.multiply %s3b2dn2sN, %s3b2dn2i2 : tensor<32x256x14x14xf32>
    %s3b2dn2dgp = stablehlo.multiply %s3b2da, %s3b2n2xh : tensor<32x256x14x14xf32>
    %s3b2dn2dg = stablehlo.reduce(%s3b2dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dn2db = stablehlo.reduce(%s3b2da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dc2t = stablehlo.transpose %s3b2W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b2dc2r = stablehlo.reverse %s3b2dc2t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b2dc2 = stablehlo.convolution(%s3b2dn2, %s3b2dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b2dW2xt = stablehlo.transpose %s3b2r1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b2dW2dt = stablehlo.transpose %s3b2dn2, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b2dW2raw = stablehlo.convolution(%s3b2dW2xt, %s3b2dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b2dW2 = stablehlo.transpose %s3b2dW2raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b2db2 = stablehlo.reduce(%s3b2dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dr1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b2dr1m = stablehlo.compare GT, %s3b2n1, %s3b2dr1z : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b2dr1 = stablehlo.select %s3b2dr1m, %s3b2dc2, %s3b2dr1z : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b2dn1dxh = stablehlo.multiply %s3b2n1gb, %s3b2dr1 : tensor<32x256x14x14xf32>
    %s3b2dn1sdxr = stablehlo.reduce(%s3b2dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dn1sdx = stablehlo.broadcast_in_dim %s3b2dn1sdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2dn1xd = stablehlo.multiply %s3b2n1xh, %s3b2dn1dxh : tensor<32x256x14x14xf32>
    %s3b2dn1sxdr = stablehlo.reduce(%s3b2dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dn1sxd = stablehlo.broadcast_in_dim %s3b2dn1sxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2dn1t1 = stablehlo.multiply %s3b2dn1dxh, %s3b2n1nf : tensor<32x256x14x14xf32>
    %s3b2dn1i1 = stablehlo.subtract %s3b2dn1t1, %s3b2dn1sdx : tensor<32x256x14x14xf32>
    %s3b2dn1xs = stablehlo.multiply %s3b2n1xh, %s3b2dn1sxd : tensor<32x256x14x14xf32>
    %s3b2dn1i2 = stablehlo.subtract %s3b2dn1i1, %s3b2dn1xs : tensor<32x256x14x14xf32>
    %s3b2dn1sN = stablehlo.divide %s3b2n1istd, %s3b2n1nf : tensor<32x256x14x14xf32>
    %s3b2dn1 = stablehlo.multiply %s3b2dn1sN, %s3b2dn1i2 : tensor<32x256x14x14xf32>
    %s3b2dn1dgp = stablehlo.multiply %s3b2dr1, %s3b2n1xh : tensor<32x256x14x14xf32>
    %s3b2dn1dg = stablehlo.reduce(%s3b2dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dn1db = stablehlo.reduce(%s3b2dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dc1t = stablehlo.transpose %s3b2W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b2dc1r = stablehlo.reverse %s3b2dc1t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b2dc1 = stablehlo.convolution(%s3b2dn1, %s3b2dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b2dW1xt = stablehlo.transpose %s3b1o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b2dW1dt = stablehlo.transpose %s3b2dn1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b2dW1raw = stablehlo.convolution(%s3b2dW1xt, %s3b2dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b2dW1 = stablehlo.transpose %s3b2dW1raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b2db1 = stablehlo.reduce(%s3b2dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dx = stablehlo.add %s3b2dc1, %s3b2da : tensor<32x256x14x14xf32>
    %s3b1daz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b1dam = stablehlo.compare GT, %s3b1a, %s3b1daz : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b1da = stablehlo.select %s3b1dam, %s3b2dx, %s3b1daz : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b1dn2dxh = stablehlo.multiply %s3b1n2gb, %s3b1da : tensor<32x256x14x14xf32>
    %s3b1dn2sdxr = stablehlo.reduce(%s3b1dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dn2sdx = stablehlo.broadcast_in_dim %s3b1dn2sdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1dn2xd = stablehlo.multiply %s3b1n2xh, %s3b1dn2dxh : tensor<32x256x14x14xf32>
    %s3b1dn2sxdr = stablehlo.reduce(%s3b1dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dn2sxd = stablehlo.broadcast_in_dim %s3b1dn2sxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1dn2t1 = stablehlo.multiply %s3b1dn2dxh, %s3b1n2nf : tensor<32x256x14x14xf32>
    %s3b1dn2i1 = stablehlo.subtract %s3b1dn2t1, %s3b1dn2sdx : tensor<32x256x14x14xf32>
    %s3b1dn2xs = stablehlo.multiply %s3b1n2xh, %s3b1dn2sxd : tensor<32x256x14x14xf32>
    %s3b1dn2i2 = stablehlo.subtract %s3b1dn2i1, %s3b1dn2xs : tensor<32x256x14x14xf32>
    %s3b1dn2sN = stablehlo.divide %s3b1n2istd, %s3b1n2nf : tensor<32x256x14x14xf32>
    %s3b1dn2 = stablehlo.multiply %s3b1dn2sN, %s3b1dn2i2 : tensor<32x256x14x14xf32>
    %s3b1dn2dgp = stablehlo.multiply %s3b1da, %s3b1n2xh : tensor<32x256x14x14xf32>
    %s3b1dn2dg = stablehlo.reduce(%s3b1dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dn2db = stablehlo.reduce(%s3b1da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dc2t = stablehlo.transpose %s3b1W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b1dc2r = stablehlo.reverse %s3b1dc2t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b1dc2 = stablehlo.convolution(%s3b1dn2, %s3b1dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b1dW2xt = stablehlo.transpose %s3b1r1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b1dW2dt = stablehlo.transpose %s3b1dn2, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b1dW2raw = stablehlo.convolution(%s3b1dW2xt, %s3b1dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b1dW2 = stablehlo.transpose %s3b1dW2raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b1db2 = stablehlo.reduce(%s3b1dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dr1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b1dr1m = stablehlo.compare GT, %s3b1n1, %s3b1dr1z : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b1dr1 = stablehlo.select %s3b1dr1m, %s3b1dc2, %s3b1dr1z : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b1dn1dxh = stablehlo.multiply %s3b1n1gb, %s3b1dr1 : tensor<32x256x14x14xf32>
    %s3b1dn1sdxr = stablehlo.reduce(%s3b1dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dn1sdx = stablehlo.broadcast_in_dim %s3b1dn1sdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1dn1xd = stablehlo.multiply %s3b1n1xh, %s3b1dn1dxh : tensor<32x256x14x14xf32>
    %s3b1dn1sxdr = stablehlo.reduce(%s3b1dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dn1sxd = stablehlo.broadcast_in_dim %s3b1dn1sxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1dn1t1 = stablehlo.multiply %s3b1dn1dxh, %s3b1n1nf : tensor<32x256x14x14xf32>
    %s3b1dn1i1 = stablehlo.subtract %s3b1dn1t1, %s3b1dn1sdx : tensor<32x256x14x14xf32>
    %s3b1dn1xs = stablehlo.multiply %s3b1n1xh, %s3b1dn1sxd : tensor<32x256x14x14xf32>
    %s3b1dn1i2 = stablehlo.subtract %s3b1dn1i1, %s3b1dn1xs : tensor<32x256x14x14xf32>
    %s3b1dn1sN = stablehlo.divide %s3b1n1istd, %s3b1n1nf : tensor<32x256x14x14xf32>
    %s3b1dn1 = stablehlo.multiply %s3b1dn1sN, %s3b1dn1i2 : tensor<32x256x14x14xf32>
    %s3b1dn1dgp = stablehlo.multiply %s3b1dr1, %s3b1n1xh : tensor<32x256x14x14xf32>
    %s3b1dn1dg = stablehlo.reduce(%s3b1dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dn1db = stablehlo.reduce(%s3b1dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dc1t = stablehlo.transpose %s3b1W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b1dc1r = stablehlo.reverse %s3b1dc1t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b1dc1 = stablehlo.convolution(%s3b1dn1, %s3b1dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b1dW1xt = stablehlo.transpose %s3b0o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b1dW1dt = stablehlo.transpose %s3b1dn1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b1dW1raw = stablehlo.convolution(%s3b1dW1xt, %s3b1dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b1dW1 = stablehlo.transpose %s3b1dW1raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b1db1 = stablehlo.reduce(%s3b1dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dx = stablehlo.add %s3b1dc1, %s3b1da : tensor<32x256x14x14xf32>
    %s3b0daz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b0dam = stablehlo.compare GT, %s3b0a, %s3b0daz : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b0da = stablehlo.select %s3b0dam, %s3b1dx, %s3b0daz : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b0dn2dxh = stablehlo.multiply %s3b0n2gb, %s3b0da : tensor<32x256x14x14xf32>
    %s3b0dn2sdxr = stablehlo.reduce(%s3b0dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dn2sdx = stablehlo.broadcast_in_dim %s3b0dn2sdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0dn2xd = stablehlo.multiply %s3b0n2xh, %s3b0dn2dxh : tensor<32x256x14x14xf32>
    %s3b0dn2sxdr = stablehlo.reduce(%s3b0dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dn2sxd = stablehlo.broadcast_in_dim %s3b0dn2sxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0dn2t1 = stablehlo.multiply %s3b0dn2dxh, %s3b0n2nf : tensor<32x256x14x14xf32>
    %s3b0dn2i1 = stablehlo.subtract %s3b0dn2t1, %s3b0dn2sdx : tensor<32x256x14x14xf32>
    %s3b0dn2xs = stablehlo.multiply %s3b0n2xh, %s3b0dn2sxd : tensor<32x256x14x14xf32>
    %s3b0dn2i2 = stablehlo.subtract %s3b0dn2i1, %s3b0dn2xs : tensor<32x256x14x14xf32>
    %s3b0dn2sN = stablehlo.divide %s3b0n2istd, %s3b0n2nf : tensor<32x256x14x14xf32>
    %s3b0dn2 = stablehlo.multiply %s3b0dn2sN, %s3b0dn2i2 : tensor<32x256x14x14xf32>
    %s3b0dn2dgp = stablehlo.multiply %s3b0da, %s3b0n2xh : tensor<32x256x14x14xf32>
    %s3b0dn2dg = stablehlo.reduce(%s3b0dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dn2db = stablehlo.reduce(%s3b0da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dc2t = stablehlo.transpose %s3b0W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b0dc2r = stablehlo.reverse %s3b0dc2t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b0dc2 = stablehlo.convolution(%s3b0dn2, %s3b0dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b0dW2xt = stablehlo.transpose %s3b0r1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b0dW2dt = stablehlo.transpose %s3b0dn2, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b0dW2raw = stablehlo.convolution(%s3b0dW2xt, %s3b0dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b0dW2 = stablehlo.transpose %s3b0dW2raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b0db2 = stablehlo.reduce(%s3b0dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dr1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b0dr1m = stablehlo.compare GT, %s3b0n1, %s3b0dr1z : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b0dr1 = stablehlo.select %s3b0dr1m, %s3b0dc2, %s3b0dr1z : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b0dn1dxh = stablehlo.multiply %s3b0n1gb, %s3b0dr1 : tensor<32x256x14x14xf32>
    %s3b0dn1sdxr = stablehlo.reduce(%s3b0dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dn1sdx = stablehlo.broadcast_in_dim %s3b0dn1sdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0dn1xd = stablehlo.multiply %s3b0n1xh, %s3b0dn1dxh : tensor<32x256x14x14xf32>
    %s3b0dn1sxdr = stablehlo.reduce(%s3b0dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dn1sxd = stablehlo.broadcast_in_dim %s3b0dn1sxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0dn1t1 = stablehlo.multiply %s3b0dn1dxh, %s3b0n1nf : tensor<32x256x14x14xf32>
    %s3b0dn1i1 = stablehlo.subtract %s3b0dn1t1, %s3b0dn1sdx : tensor<32x256x14x14xf32>
    %s3b0dn1xs = stablehlo.multiply %s3b0n1xh, %s3b0dn1sxd : tensor<32x256x14x14xf32>
    %s3b0dn1i2 = stablehlo.subtract %s3b0dn1i1, %s3b0dn1xs : tensor<32x256x14x14xf32>
    %s3b0dn1sN = stablehlo.divide %s3b0n1istd, %s3b0n1nf : tensor<32x256x14x14xf32>
    %s3b0dn1 = stablehlo.multiply %s3b0dn1sN, %s3b0dn1i2 : tensor<32x256x14x14xf32>
    %s3b0dn1dgp = stablehlo.multiply %s3b0dr1, %s3b0n1xh : tensor<32x256x14x14xf32>
    %s3b0dn1dg = stablehlo.reduce(%s3b0dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dn1db = stablehlo.reduce(%s3b0dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dc1t = stablehlo.transpose %s3b0W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b0dc1r = stablehlo.reverse %s3b0dc1t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b0dc1 = stablehlo.convolution(%s3b0dn1, %s3b0dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b0dW1xt = stablehlo.transpose %d3o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b0dW1dt = stablehlo.transpose %s3b0dn1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b0dW1raw = stablehlo.convolution(%s3b0dW1xt, %s3b0dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b0dW1 = stablehlo.transpose %s3b0dW1raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b0db1 = stablehlo.reduce(%s3b0dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dx = stablehlo.add %s3b0dc1, %s3b0da : tensor<32x256x14x14xf32>
    %d3daz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %d3dam = stablehlo.compare GT, %d3a, %d3daz : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %d3da = stablehlo.select %d3dam, %s3b0dx, %d3daz : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %d3dn2dxh = stablehlo.multiply %d3n2gb, %d3da : tensor<32x256x14x14xf32>
    %d3dn2sdxr = stablehlo.reduce(%d3dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dn2sdx = stablehlo.broadcast_in_dim %d3dn2sdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3dn2xd = stablehlo.multiply %d3n2xh, %d3dn2dxh : tensor<32x256x14x14xf32>
    %d3dn2sxdr = stablehlo.reduce(%d3dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dn2sxd = stablehlo.broadcast_in_dim %d3dn2sxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3dn2t1 = stablehlo.multiply %d3dn2dxh, %d3n2nf : tensor<32x256x14x14xf32>
    %d3dn2i1 = stablehlo.subtract %d3dn2t1, %d3dn2sdx : tensor<32x256x14x14xf32>
    %d3dn2xs = stablehlo.multiply %d3n2xh, %d3dn2sxd : tensor<32x256x14x14xf32>
    %d3dn2i2 = stablehlo.subtract %d3dn2i1, %d3dn2xs : tensor<32x256x14x14xf32>
    %d3dn2sN = stablehlo.divide %d3n2istd, %d3n2nf : tensor<32x256x14x14xf32>
    %d3dn2 = stablehlo.multiply %d3dn2sN, %d3dn2i2 : tensor<32x256x14x14xf32>
    %d3dn2dgp = stablehlo.multiply %d3da, %d3n2xh : tensor<32x256x14x14xf32>
    %d3dn2dg = stablehlo.reduce(%d3dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dn2db = stablehlo.reduce(%d3da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dc2t = stablehlo.transpose %d3W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %d3dc2r = stablehlo.reverse %d3dc2t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %d3dc2 = stablehlo.convolution(%d3dn2, %d3dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %d3dW2xt = stablehlo.transpose %d3r1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %d3dW2dt = stablehlo.transpose %d3dn2, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %d3dW2raw = stablehlo.convolution(%d3dW2xt, %d3dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %d3dW2 = stablehlo.transpose %d3dW2raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %d3db2 = stablehlo.reduce(%d3dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dr1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %d3dr1m = stablehlo.compare GT, %d3n1, %d3dr1z : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %d3dr1 = stablehlo.select %d3dr1m, %d3dc2, %d3dr1z : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %d3dn1dxh = stablehlo.multiply %d3n1gb, %d3dr1 : tensor<32x256x14x14xf32>
    %d3dn1sdxr = stablehlo.reduce(%d3dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dn1sdx = stablehlo.broadcast_in_dim %d3dn1sdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3dn1xd = stablehlo.multiply %d3n1xh, %d3dn1dxh : tensor<32x256x14x14xf32>
    %d3dn1sxdr = stablehlo.reduce(%d3dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dn1sxd = stablehlo.broadcast_in_dim %d3dn1sxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3dn1t1 = stablehlo.multiply %d3dn1dxh, %d3n1nf : tensor<32x256x14x14xf32>
    %d3dn1i1 = stablehlo.subtract %d3dn1t1, %d3dn1sdx : tensor<32x256x14x14xf32>
    %d3dn1xs = stablehlo.multiply %d3n1xh, %d3dn1sxd : tensor<32x256x14x14xf32>
    %d3dn1i2 = stablehlo.subtract %d3dn1i1, %d3dn1xs : tensor<32x256x14x14xf32>
    %d3dn1sN = stablehlo.divide %d3n1istd, %d3n1nf : tensor<32x256x14x14xf32>
    %d3dn1 = stablehlo.multiply %d3dn1sN, %d3dn1i2 : tensor<32x256x14x14xf32>
    %d3dn1dgp = stablehlo.multiply %d3dr1, %d3n1xh : tensor<32x256x14x14xf32>
    %d3dn1dg = stablehlo.reduce(%d3dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dn1db = stablehlo.reduce(%d3dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dc1u = stablehlo.pad %d3dn1, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %d3dc1t = stablehlo.transpose %d3W1, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %d3dc1r = stablehlo.reverse %d3dc1t, dims = [2, 3] : tensor<128x256x3x3xf32>
    %d3dc1 = stablehlo.convolution(%d3dc1u, %d3dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %d3dW1u = stablehlo.pad %d3dn1, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %d3dW1xt = stablehlo.transpose %s2b2o, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %d3dW1dt = stablehlo.transpose %d3dW1u, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %d3dW1raw = stablehlo.convolution(%d3dW1xt, %d3dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %d3dW1 = stablehlo.transpose %d3dW1raw, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %d3db1 = stablehlo.reduce(%d3dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dnpdxh = stablehlo.multiply %d3npgb, %d3da : tensor<32x256x14x14xf32>
    %d3dnpsdxr = stablehlo.reduce(%d3dnpdxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dnpsdx = stablehlo.broadcast_in_dim %d3dnpsdxr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3dnpxd = stablehlo.multiply %d3npxh, %d3dnpdxh : tensor<32x256x14x14xf32>
    %d3dnpsxdr = stablehlo.reduce(%d3dnpxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dnpsxd = stablehlo.broadcast_in_dim %d3dnpsxdr, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3dnpt1 = stablehlo.multiply %d3dnpdxh, %d3npnf : tensor<32x256x14x14xf32>
    %d3dnpi1 = stablehlo.subtract %d3dnpt1, %d3dnpsdx : tensor<32x256x14x14xf32>
    %d3dnpxs = stablehlo.multiply %d3npxh, %d3dnpsxd : tensor<32x256x14x14xf32>
    %d3dnpi2 = stablehlo.subtract %d3dnpi1, %d3dnpxs : tensor<32x256x14x14xf32>
    %d3dnpsN = stablehlo.divide %d3npistd, %d3npnf : tensor<32x256x14x14xf32>
    %d3dnp = stablehlo.multiply %d3dnpsN, %d3dnpi2 : tensor<32x256x14x14xf32>
    %d3dnpdgp = stablehlo.multiply %d3da, %d3npxh : tensor<32x256x14x14xf32>
    %d3dnpdg = stablehlo.reduce(%d3dnpdgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dnpdb = stablehlo.reduce(%d3da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dcpu = stablehlo.pad %d3dnp, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %d3dcpt = stablehlo.transpose %d3Wp, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %d3dcpr = stablehlo.reverse %d3dcpt, dims = [2, 3] : tensor<128x256x3x3xf32>
    %d3dcp = stablehlo.convolution(%d3dcpu, %d3dcpr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %d3dWpu = stablehlo.pad %d3dnp, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %d3dWpxt = stablehlo.transpose %s2b2o, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %d3dWpdt = stablehlo.transpose %d3dWpu, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %d3dWpraw = stablehlo.convolution(%d3dWpxt, %d3dWpdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %d3dWp = stablehlo.transpose %d3dWpraw, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %d3dbp = stablehlo.reduce(%d3dnp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dx = stablehlo.add %d3dc1, %d3dcp : tensor<32x128x28x28xf32>
    %s2b2daz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b2dam = stablehlo.compare GT, %s2b2a, %s2b2daz : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %s2b2da = stablehlo.select %s2b2dam, %d3dx, %s2b2daz : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %s2b2dn2dxh = stablehlo.multiply %s2b2n2gb, %s2b2da : tensor<32x128x28x28xf32>
    %s2b2dn2sdxr = stablehlo.reduce(%s2b2dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dn2sdx = stablehlo.broadcast_in_dim %s2b2dn2sdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2dn2xd = stablehlo.multiply %s2b2n2xh, %s2b2dn2dxh : tensor<32x128x28x28xf32>
    %s2b2dn2sxdr = stablehlo.reduce(%s2b2dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dn2sxd = stablehlo.broadcast_in_dim %s2b2dn2sxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2dn2t1 = stablehlo.multiply %s2b2dn2dxh, %s2b2n2nf : tensor<32x128x28x28xf32>
    %s2b2dn2i1 = stablehlo.subtract %s2b2dn2t1, %s2b2dn2sdx : tensor<32x128x28x28xf32>
    %s2b2dn2xs = stablehlo.multiply %s2b2n2xh, %s2b2dn2sxd : tensor<32x128x28x28xf32>
    %s2b2dn2i2 = stablehlo.subtract %s2b2dn2i1, %s2b2dn2xs : tensor<32x128x28x28xf32>
    %s2b2dn2sN = stablehlo.divide %s2b2n2istd, %s2b2n2nf : tensor<32x128x28x28xf32>
    %s2b2dn2 = stablehlo.multiply %s2b2dn2sN, %s2b2dn2i2 : tensor<32x128x28x28xf32>
    %s2b2dn2dgp = stablehlo.multiply %s2b2da, %s2b2n2xh : tensor<32x128x28x28xf32>
    %s2b2dn2dg = stablehlo.reduce(%s2b2dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dn2db = stablehlo.reduce(%s2b2da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dc2t = stablehlo.transpose %s2b2W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b2dc2r = stablehlo.reverse %s2b2dc2t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %s2b2dc2 = stablehlo.convolution(%s2b2dn2, %s2b2dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b2dW2xt = stablehlo.transpose %s2b2r1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b2dW2dt = stablehlo.transpose %s2b2dn2, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b2dW2raw = stablehlo.convolution(%s2b2dW2xt, %s2b2dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %s2b2dW2 = stablehlo.transpose %s2b2dW2raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b2db2 = stablehlo.reduce(%s2b2dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dr1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b2dr1m = stablehlo.compare GT, %s2b2n1, %s2b2dr1z : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %s2b2dr1 = stablehlo.select %s2b2dr1m, %s2b2dc2, %s2b2dr1z : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %s2b2dn1dxh = stablehlo.multiply %s2b2n1gb, %s2b2dr1 : tensor<32x128x28x28xf32>
    %s2b2dn1sdxr = stablehlo.reduce(%s2b2dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dn1sdx = stablehlo.broadcast_in_dim %s2b2dn1sdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2dn1xd = stablehlo.multiply %s2b2n1xh, %s2b2dn1dxh : tensor<32x128x28x28xf32>
    %s2b2dn1sxdr = stablehlo.reduce(%s2b2dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dn1sxd = stablehlo.broadcast_in_dim %s2b2dn1sxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2dn1t1 = stablehlo.multiply %s2b2dn1dxh, %s2b2n1nf : tensor<32x128x28x28xf32>
    %s2b2dn1i1 = stablehlo.subtract %s2b2dn1t1, %s2b2dn1sdx : tensor<32x128x28x28xf32>
    %s2b2dn1xs = stablehlo.multiply %s2b2n1xh, %s2b2dn1sxd : tensor<32x128x28x28xf32>
    %s2b2dn1i2 = stablehlo.subtract %s2b2dn1i1, %s2b2dn1xs : tensor<32x128x28x28xf32>
    %s2b2dn1sN = stablehlo.divide %s2b2n1istd, %s2b2n1nf : tensor<32x128x28x28xf32>
    %s2b2dn1 = stablehlo.multiply %s2b2dn1sN, %s2b2dn1i2 : tensor<32x128x28x28xf32>
    %s2b2dn1dgp = stablehlo.multiply %s2b2dr1, %s2b2n1xh : tensor<32x128x28x28xf32>
    %s2b2dn1dg = stablehlo.reduce(%s2b2dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dn1db = stablehlo.reduce(%s2b2dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dc1t = stablehlo.transpose %s2b2W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b2dc1r = stablehlo.reverse %s2b2dc1t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %s2b2dc1 = stablehlo.convolution(%s2b2dn1, %s2b2dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b2dW1xt = stablehlo.transpose %s2b1o, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b2dW1dt = stablehlo.transpose %s2b2dn1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b2dW1raw = stablehlo.convolution(%s2b2dW1xt, %s2b2dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %s2b2dW1 = stablehlo.transpose %s2b2dW1raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b2db1 = stablehlo.reduce(%s2b2dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dx = stablehlo.add %s2b2dc1, %s2b2da : tensor<32x128x28x28xf32>
    %s2b1daz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b1dam = stablehlo.compare GT, %s2b1a, %s2b1daz : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %s2b1da = stablehlo.select %s2b1dam, %s2b2dx, %s2b1daz : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %s2b1dn2dxh = stablehlo.multiply %s2b1n2gb, %s2b1da : tensor<32x128x28x28xf32>
    %s2b1dn2sdxr = stablehlo.reduce(%s2b1dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dn2sdx = stablehlo.broadcast_in_dim %s2b1dn2sdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1dn2xd = stablehlo.multiply %s2b1n2xh, %s2b1dn2dxh : tensor<32x128x28x28xf32>
    %s2b1dn2sxdr = stablehlo.reduce(%s2b1dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dn2sxd = stablehlo.broadcast_in_dim %s2b1dn2sxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1dn2t1 = stablehlo.multiply %s2b1dn2dxh, %s2b1n2nf : tensor<32x128x28x28xf32>
    %s2b1dn2i1 = stablehlo.subtract %s2b1dn2t1, %s2b1dn2sdx : tensor<32x128x28x28xf32>
    %s2b1dn2xs = stablehlo.multiply %s2b1n2xh, %s2b1dn2sxd : tensor<32x128x28x28xf32>
    %s2b1dn2i2 = stablehlo.subtract %s2b1dn2i1, %s2b1dn2xs : tensor<32x128x28x28xf32>
    %s2b1dn2sN = stablehlo.divide %s2b1n2istd, %s2b1n2nf : tensor<32x128x28x28xf32>
    %s2b1dn2 = stablehlo.multiply %s2b1dn2sN, %s2b1dn2i2 : tensor<32x128x28x28xf32>
    %s2b1dn2dgp = stablehlo.multiply %s2b1da, %s2b1n2xh : tensor<32x128x28x28xf32>
    %s2b1dn2dg = stablehlo.reduce(%s2b1dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dn2db = stablehlo.reduce(%s2b1da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dc2t = stablehlo.transpose %s2b1W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b1dc2r = stablehlo.reverse %s2b1dc2t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %s2b1dc2 = stablehlo.convolution(%s2b1dn2, %s2b1dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b1dW2xt = stablehlo.transpose %s2b1r1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b1dW2dt = stablehlo.transpose %s2b1dn2, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b1dW2raw = stablehlo.convolution(%s2b1dW2xt, %s2b1dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %s2b1dW2 = stablehlo.transpose %s2b1dW2raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b1db2 = stablehlo.reduce(%s2b1dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dr1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b1dr1m = stablehlo.compare GT, %s2b1n1, %s2b1dr1z : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %s2b1dr1 = stablehlo.select %s2b1dr1m, %s2b1dc2, %s2b1dr1z : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %s2b1dn1dxh = stablehlo.multiply %s2b1n1gb, %s2b1dr1 : tensor<32x128x28x28xf32>
    %s2b1dn1sdxr = stablehlo.reduce(%s2b1dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dn1sdx = stablehlo.broadcast_in_dim %s2b1dn1sdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1dn1xd = stablehlo.multiply %s2b1n1xh, %s2b1dn1dxh : tensor<32x128x28x28xf32>
    %s2b1dn1sxdr = stablehlo.reduce(%s2b1dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dn1sxd = stablehlo.broadcast_in_dim %s2b1dn1sxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1dn1t1 = stablehlo.multiply %s2b1dn1dxh, %s2b1n1nf : tensor<32x128x28x28xf32>
    %s2b1dn1i1 = stablehlo.subtract %s2b1dn1t1, %s2b1dn1sdx : tensor<32x128x28x28xf32>
    %s2b1dn1xs = stablehlo.multiply %s2b1n1xh, %s2b1dn1sxd : tensor<32x128x28x28xf32>
    %s2b1dn1i2 = stablehlo.subtract %s2b1dn1i1, %s2b1dn1xs : tensor<32x128x28x28xf32>
    %s2b1dn1sN = stablehlo.divide %s2b1n1istd, %s2b1n1nf : tensor<32x128x28x28xf32>
    %s2b1dn1 = stablehlo.multiply %s2b1dn1sN, %s2b1dn1i2 : tensor<32x128x28x28xf32>
    %s2b1dn1dgp = stablehlo.multiply %s2b1dr1, %s2b1n1xh : tensor<32x128x28x28xf32>
    %s2b1dn1dg = stablehlo.reduce(%s2b1dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dn1db = stablehlo.reduce(%s2b1dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dc1t = stablehlo.transpose %s2b1W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b1dc1r = stablehlo.reverse %s2b1dc1t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %s2b1dc1 = stablehlo.convolution(%s2b1dn1, %s2b1dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b1dW1xt = stablehlo.transpose %s2b0o, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b1dW1dt = stablehlo.transpose %s2b1dn1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b1dW1raw = stablehlo.convolution(%s2b1dW1xt, %s2b1dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %s2b1dW1 = stablehlo.transpose %s2b1dW1raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b1db1 = stablehlo.reduce(%s2b1dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dx = stablehlo.add %s2b1dc1, %s2b1da : tensor<32x128x28x28xf32>
    %s2b0daz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b0dam = stablehlo.compare GT, %s2b0a, %s2b0daz : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %s2b0da = stablehlo.select %s2b0dam, %s2b1dx, %s2b0daz : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %s2b0dn2dxh = stablehlo.multiply %s2b0n2gb, %s2b0da : tensor<32x128x28x28xf32>
    %s2b0dn2sdxr = stablehlo.reduce(%s2b0dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dn2sdx = stablehlo.broadcast_in_dim %s2b0dn2sdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0dn2xd = stablehlo.multiply %s2b0n2xh, %s2b0dn2dxh : tensor<32x128x28x28xf32>
    %s2b0dn2sxdr = stablehlo.reduce(%s2b0dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dn2sxd = stablehlo.broadcast_in_dim %s2b0dn2sxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0dn2t1 = stablehlo.multiply %s2b0dn2dxh, %s2b0n2nf : tensor<32x128x28x28xf32>
    %s2b0dn2i1 = stablehlo.subtract %s2b0dn2t1, %s2b0dn2sdx : tensor<32x128x28x28xf32>
    %s2b0dn2xs = stablehlo.multiply %s2b0n2xh, %s2b0dn2sxd : tensor<32x128x28x28xf32>
    %s2b0dn2i2 = stablehlo.subtract %s2b0dn2i1, %s2b0dn2xs : tensor<32x128x28x28xf32>
    %s2b0dn2sN = stablehlo.divide %s2b0n2istd, %s2b0n2nf : tensor<32x128x28x28xf32>
    %s2b0dn2 = stablehlo.multiply %s2b0dn2sN, %s2b0dn2i2 : tensor<32x128x28x28xf32>
    %s2b0dn2dgp = stablehlo.multiply %s2b0da, %s2b0n2xh : tensor<32x128x28x28xf32>
    %s2b0dn2dg = stablehlo.reduce(%s2b0dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dn2db = stablehlo.reduce(%s2b0da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dc2t = stablehlo.transpose %s2b0W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b0dc2r = stablehlo.reverse %s2b0dc2t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %s2b0dc2 = stablehlo.convolution(%s2b0dn2, %s2b0dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b0dW2xt = stablehlo.transpose %s2b0r1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b0dW2dt = stablehlo.transpose %s2b0dn2, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b0dW2raw = stablehlo.convolution(%s2b0dW2xt, %s2b0dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %s2b0dW2 = stablehlo.transpose %s2b0dW2raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b0db2 = stablehlo.reduce(%s2b0dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dr1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b0dr1m = stablehlo.compare GT, %s2b0n1, %s2b0dr1z : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %s2b0dr1 = stablehlo.select %s2b0dr1m, %s2b0dc2, %s2b0dr1z : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %s2b0dn1dxh = stablehlo.multiply %s2b0n1gb, %s2b0dr1 : tensor<32x128x28x28xf32>
    %s2b0dn1sdxr = stablehlo.reduce(%s2b0dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dn1sdx = stablehlo.broadcast_in_dim %s2b0dn1sdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0dn1xd = stablehlo.multiply %s2b0n1xh, %s2b0dn1dxh : tensor<32x128x28x28xf32>
    %s2b0dn1sxdr = stablehlo.reduce(%s2b0dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dn1sxd = stablehlo.broadcast_in_dim %s2b0dn1sxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0dn1t1 = stablehlo.multiply %s2b0dn1dxh, %s2b0n1nf : tensor<32x128x28x28xf32>
    %s2b0dn1i1 = stablehlo.subtract %s2b0dn1t1, %s2b0dn1sdx : tensor<32x128x28x28xf32>
    %s2b0dn1xs = stablehlo.multiply %s2b0n1xh, %s2b0dn1sxd : tensor<32x128x28x28xf32>
    %s2b0dn1i2 = stablehlo.subtract %s2b0dn1i1, %s2b0dn1xs : tensor<32x128x28x28xf32>
    %s2b0dn1sN = stablehlo.divide %s2b0n1istd, %s2b0n1nf : tensor<32x128x28x28xf32>
    %s2b0dn1 = stablehlo.multiply %s2b0dn1sN, %s2b0dn1i2 : tensor<32x128x28x28xf32>
    %s2b0dn1dgp = stablehlo.multiply %s2b0dr1, %s2b0n1xh : tensor<32x128x28x28xf32>
    %s2b0dn1dg = stablehlo.reduce(%s2b0dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dn1db = stablehlo.reduce(%s2b0dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dc1t = stablehlo.transpose %s2b0W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b0dc1r = stablehlo.reverse %s2b0dc1t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %s2b0dc1 = stablehlo.convolution(%s2b0dn1, %s2b0dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b0dW1xt = stablehlo.transpose %d2o, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b0dW1dt = stablehlo.transpose %s2b0dn1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b0dW1raw = stablehlo.convolution(%s2b0dW1xt, %s2b0dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %s2b0dW1 = stablehlo.transpose %s2b0dW1raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b0db1 = stablehlo.reduce(%s2b0dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dx = stablehlo.add %s2b0dc1, %s2b0da : tensor<32x128x28x28xf32>
    %d2daz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %d2dam = stablehlo.compare GT, %d2a, %d2daz : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %d2da = stablehlo.select %d2dam, %s2b0dx, %d2daz : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %d2dn2dxh = stablehlo.multiply %d2n2gb, %d2da : tensor<32x128x28x28xf32>
    %d2dn2sdxr = stablehlo.reduce(%d2dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dn2sdx = stablehlo.broadcast_in_dim %d2dn2sdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2dn2xd = stablehlo.multiply %d2n2xh, %d2dn2dxh : tensor<32x128x28x28xf32>
    %d2dn2sxdr = stablehlo.reduce(%d2dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dn2sxd = stablehlo.broadcast_in_dim %d2dn2sxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2dn2t1 = stablehlo.multiply %d2dn2dxh, %d2n2nf : tensor<32x128x28x28xf32>
    %d2dn2i1 = stablehlo.subtract %d2dn2t1, %d2dn2sdx : tensor<32x128x28x28xf32>
    %d2dn2xs = stablehlo.multiply %d2n2xh, %d2dn2sxd : tensor<32x128x28x28xf32>
    %d2dn2i2 = stablehlo.subtract %d2dn2i1, %d2dn2xs : tensor<32x128x28x28xf32>
    %d2dn2sN = stablehlo.divide %d2n2istd, %d2n2nf : tensor<32x128x28x28xf32>
    %d2dn2 = stablehlo.multiply %d2dn2sN, %d2dn2i2 : tensor<32x128x28x28xf32>
    %d2dn2dgp = stablehlo.multiply %d2da, %d2n2xh : tensor<32x128x28x28xf32>
    %d2dn2dg = stablehlo.reduce(%d2dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dn2db = stablehlo.reduce(%d2da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dc2t = stablehlo.transpose %d2W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %d2dc2r = stablehlo.reverse %d2dc2t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %d2dc2 = stablehlo.convolution(%d2dn2, %d2dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %d2dW2xt = stablehlo.transpose %d2r1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %d2dW2dt = stablehlo.transpose %d2dn2, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %d2dW2raw = stablehlo.convolution(%d2dW2xt, %d2dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %d2dW2 = stablehlo.transpose %d2dW2raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %d2db2 = stablehlo.reduce(%d2dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dr1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %d2dr1m = stablehlo.compare GT, %d2n1, %d2dr1z : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %d2dr1 = stablehlo.select %d2dr1m, %d2dc2, %d2dr1z : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %d2dn1dxh = stablehlo.multiply %d2n1gb, %d2dr1 : tensor<32x128x28x28xf32>
    %d2dn1sdxr = stablehlo.reduce(%d2dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dn1sdx = stablehlo.broadcast_in_dim %d2dn1sdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2dn1xd = stablehlo.multiply %d2n1xh, %d2dn1dxh : tensor<32x128x28x28xf32>
    %d2dn1sxdr = stablehlo.reduce(%d2dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dn1sxd = stablehlo.broadcast_in_dim %d2dn1sxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2dn1t1 = stablehlo.multiply %d2dn1dxh, %d2n1nf : tensor<32x128x28x28xf32>
    %d2dn1i1 = stablehlo.subtract %d2dn1t1, %d2dn1sdx : tensor<32x128x28x28xf32>
    %d2dn1xs = stablehlo.multiply %d2n1xh, %d2dn1sxd : tensor<32x128x28x28xf32>
    %d2dn1i2 = stablehlo.subtract %d2dn1i1, %d2dn1xs : tensor<32x128x28x28xf32>
    %d2dn1sN = stablehlo.divide %d2n1istd, %d2n1nf : tensor<32x128x28x28xf32>
    %d2dn1 = stablehlo.multiply %d2dn1sN, %d2dn1i2 : tensor<32x128x28x28xf32>
    %d2dn1dgp = stablehlo.multiply %d2dr1, %d2n1xh : tensor<32x128x28x28xf32>
    %d2dn1dg = stablehlo.reduce(%d2dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dn1db = stablehlo.reduce(%d2dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dc1u = stablehlo.pad %d2dn1, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %d2dc1t = stablehlo.transpose %d2W1, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %d2dc1r = stablehlo.reverse %d2dc1t, dims = [2, 3] : tensor<64x128x3x3xf32>
    %d2dc1 = stablehlo.convolution(%d2dc1u, %d2dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %d2dW1u = stablehlo.pad %d2dn1, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %d2dW1xt = stablehlo.transpose %s1b2o, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %d2dW1dt = stablehlo.transpose %d2dW1u, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %d2dW1raw = stablehlo.convolution(%d2dW1xt, %d2dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %d2dW1 = stablehlo.transpose %d2dW1raw, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %d2db1 = stablehlo.reduce(%d2dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dnpdxh = stablehlo.multiply %d2npgb, %d2da : tensor<32x128x28x28xf32>
    %d2dnpsdxr = stablehlo.reduce(%d2dnpdxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dnpsdx = stablehlo.broadcast_in_dim %d2dnpsdxr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2dnpxd = stablehlo.multiply %d2npxh, %d2dnpdxh : tensor<32x128x28x28xf32>
    %d2dnpsxdr = stablehlo.reduce(%d2dnpxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dnpsxd = stablehlo.broadcast_in_dim %d2dnpsxdr, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2dnpt1 = stablehlo.multiply %d2dnpdxh, %d2npnf : tensor<32x128x28x28xf32>
    %d2dnpi1 = stablehlo.subtract %d2dnpt1, %d2dnpsdx : tensor<32x128x28x28xf32>
    %d2dnpxs = stablehlo.multiply %d2npxh, %d2dnpsxd : tensor<32x128x28x28xf32>
    %d2dnpi2 = stablehlo.subtract %d2dnpi1, %d2dnpxs : tensor<32x128x28x28xf32>
    %d2dnpsN = stablehlo.divide %d2npistd, %d2npnf : tensor<32x128x28x28xf32>
    %d2dnp = stablehlo.multiply %d2dnpsN, %d2dnpi2 : tensor<32x128x28x28xf32>
    %d2dnpdgp = stablehlo.multiply %d2da, %d2npxh : tensor<32x128x28x28xf32>
    %d2dnpdg = stablehlo.reduce(%d2dnpdgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dnpdb = stablehlo.reduce(%d2da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dcpu = stablehlo.pad %d2dnp, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %d2dcpt = stablehlo.transpose %d2Wp, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %d2dcpr = stablehlo.reverse %d2dcpt, dims = [2, 3] : tensor<64x128x3x3xf32>
    %d2dcp = stablehlo.convolution(%d2dcpu, %d2dcpr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %d2dWpu = stablehlo.pad %d2dnp, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %d2dWpxt = stablehlo.transpose %s1b2o, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %d2dWpdt = stablehlo.transpose %d2dWpu, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %d2dWpraw = stablehlo.convolution(%d2dWpxt, %d2dWpdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %d2dWp = stablehlo.transpose %d2dWpraw, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %d2dbp = stablehlo.reduce(%d2dnp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dx = stablehlo.add %d2dc1, %d2dcp : tensor<32x64x56x56xf32>
    %s1b2daz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b2dam = stablehlo.compare GT, %s1b2a, %s1b2daz : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %s1b2da = stablehlo.select %s1b2dam, %d2dx, %s1b2daz : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %s1b2dn2dxh = stablehlo.multiply %s1b2n2gb, %s1b2da : tensor<32x64x56x56xf32>
    %s1b2dn2sdxr = stablehlo.reduce(%s1b2dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dn2sdx = stablehlo.broadcast_in_dim %s1b2dn2sdxr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2dn2xd = stablehlo.multiply %s1b2n2xh, %s1b2dn2dxh : tensor<32x64x56x56xf32>
    %s1b2dn2sxdr = stablehlo.reduce(%s1b2dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dn2sxd = stablehlo.broadcast_in_dim %s1b2dn2sxdr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2dn2t1 = stablehlo.multiply %s1b2dn2dxh, %s1b2n2nf : tensor<32x64x56x56xf32>
    %s1b2dn2i1 = stablehlo.subtract %s1b2dn2t1, %s1b2dn2sdx : tensor<32x64x56x56xf32>
    %s1b2dn2xs = stablehlo.multiply %s1b2n2xh, %s1b2dn2sxd : tensor<32x64x56x56xf32>
    %s1b2dn2i2 = stablehlo.subtract %s1b2dn2i1, %s1b2dn2xs : tensor<32x64x56x56xf32>
    %s1b2dn2sN = stablehlo.divide %s1b2n2istd, %s1b2n2nf : tensor<32x64x56x56xf32>
    %s1b2dn2 = stablehlo.multiply %s1b2dn2sN, %s1b2dn2i2 : tensor<32x64x56x56xf32>
    %s1b2dn2dgp = stablehlo.multiply %s1b2da, %s1b2n2xh : tensor<32x64x56x56xf32>
    %s1b2dn2dg = stablehlo.reduce(%s1b2dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dn2db = stablehlo.reduce(%s1b2da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dc2t = stablehlo.transpose %s1b2W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b2dc2r = stablehlo.reverse %s1b2dc2t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %s1b2dc2 = stablehlo.convolution(%s1b2dn2, %s1b2dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b2dW2xt = stablehlo.transpose %s1b2r1, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b2dW2dt = stablehlo.transpose %s1b2dn2, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b2dW2raw = stablehlo.convolution(%s1b2dW2xt, %s1b2dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %s1b2dW2 = stablehlo.transpose %s1b2dW2raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b2db2 = stablehlo.reduce(%s1b2dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dr1z = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b2dr1m = stablehlo.compare GT, %s1b2n1, %s1b2dr1z : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %s1b2dr1 = stablehlo.select %s1b2dr1m, %s1b2dc2, %s1b2dr1z : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %s1b2dn1dxh = stablehlo.multiply %s1b2n1gb, %s1b2dr1 : tensor<32x64x56x56xf32>
    %s1b2dn1sdxr = stablehlo.reduce(%s1b2dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dn1sdx = stablehlo.broadcast_in_dim %s1b2dn1sdxr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2dn1xd = stablehlo.multiply %s1b2n1xh, %s1b2dn1dxh : tensor<32x64x56x56xf32>
    %s1b2dn1sxdr = stablehlo.reduce(%s1b2dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dn1sxd = stablehlo.broadcast_in_dim %s1b2dn1sxdr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2dn1t1 = stablehlo.multiply %s1b2dn1dxh, %s1b2n1nf : tensor<32x64x56x56xf32>
    %s1b2dn1i1 = stablehlo.subtract %s1b2dn1t1, %s1b2dn1sdx : tensor<32x64x56x56xf32>
    %s1b2dn1xs = stablehlo.multiply %s1b2n1xh, %s1b2dn1sxd : tensor<32x64x56x56xf32>
    %s1b2dn1i2 = stablehlo.subtract %s1b2dn1i1, %s1b2dn1xs : tensor<32x64x56x56xf32>
    %s1b2dn1sN = stablehlo.divide %s1b2n1istd, %s1b2n1nf : tensor<32x64x56x56xf32>
    %s1b2dn1 = stablehlo.multiply %s1b2dn1sN, %s1b2dn1i2 : tensor<32x64x56x56xf32>
    %s1b2dn1dgp = stablehlo.multiply %s1b2dr1, %s1b2n1xh : tensor<32x64x56x56xf32>
    %s1b2dn1dg = stablehlo.reduce(%s1b2dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dn1db = stablehlo.reduce(%s1b2dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dc1t = stablehlo.transpose %s1b2W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b2dc1r = stablehlo.reverse %s1b2dc1t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %s1b2dc1 = stablehlo.convolution(%s1b2dn1, %s1b2dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b2dW1xt = stablehlo.transpose %s1b1o, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b2dW1dt = stablehlo.transpose %s1b2dn1, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b2dW1raw = stablehlo.convolution(%s1b2dW1xt, %s1b2dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %s1b2dW1 = stablehlo.transpose %s1b2dW1raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b2db1 = stablehlo.reduce(%s1b2dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dx = stablehlo.add %s1b2dc1, %s1b2da : tensor<32x64x56x56xf32>
    %s1b1daz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b1dam = stablehlo.compare GT, %s1b1a, %s1b1daz : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %s1b1da = stablehlo.select %s1b1dam, %s1b2dx, %s1b1daz : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %s1b1dn2dxh = stablehlo.multiply %s1b1n2gb, %s1b1da : tensor<32x64x56x56xf32>
    %s1b1dn2sdxr = stablehlo.reduce(%s1b1dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dn2sdx = stablehlo.broadcast_in_dim %s1b1dn2sdxr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1dn2xd = stablehlo.multiply %s1b1n2xh, %s1b1dn2dxh : tensor<32x64x56x56xf32>
    %s1b1dn2sxdr = stablehlo.reduce(%s1b1dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dn2sxd = stablehlo.broadcast_in_dim %s1b1dn2sxdr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1dn2t1 = stablehlo.multiply %s1b1dn2dxh, %s1b1n2nf : tensor<32x64x56x56xf32>
    %s1b1dn2i1 = stablehlo.subtract %s1b1dn2t1, %s1b1dn2sdx : tensor<32x64x56x56xf32>
    %s1b1dn2xs = stablehlo.multiply %s1b1n2xh, %s1b1dn2sxd : tensor<32x64x56x56xf32>
    %s1b1dn2i2 = stablehlo.subtract %s1b1dn2i1, %s1b1dn2xs : tensor<32x64x56x56xf32>
    %s1b1dn2sN = stablehlo.divide %s1b1n2istd, %s1b1n2nf : tensor<32x64x56x56xf32>
    %s1b1dn2 = stablehlo.multiply %s1b1dn2sN, %s1b1dn2i2 : tensor<32x64x56x56xf32>
    %s1b1dn2dgp = stablehlo.multiply %s1b1da, %s1b1n2xh : tensor<32x64x56x56xf32>
    %s1b1dn2dg = stablehlo.reduce(%s1b1dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dn2db = stablehlo.reduce(%s1b1da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dc2t = stablehlo.transpose %s1b1W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b1dc2r = stablehlo.reverse %s1b1dc2t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %s1b1dc2 = stablehlo.convolution(%s1b1dn2, %s1b1dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b1dW2xt = stablehlo.transpose %s1b1r1, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b1dW2dt = stablehlo.transpose %s1b1dn2, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b1dW2raw = stablehlo.convolution(%s1b1dW2xt, %s1b1dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %s1b1dW2 = stablehlo.transpose %s1b1dW2raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b1db2 = stablehlo.reduce(%s1b1dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dr1z = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b1dr1m = stablehlo.compare GT, %s1b1n1, %s1b1dr1z : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %s1b1dr1 = stablehlo.select %s1b1dr1m, %s1b1dc2, %s1b1dr1z : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %s1b1dn1dxh = stablehlo.multiply %s1b1n1gb, %s1b1dr1 : tensor<32x64x56x56xf32>
    %s1b1dn1sdxr = stablehlo.reduce(%s1b1dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dn1sdx = stablehlo.broadcast_in_dim %s1b1dn1sdxr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1dn1xd = stablehlo.multiply %s1b1n1xh, %s1b1dn1dxh : tensor<32x64x56x56xf32>
    %s1b1dn1sxdr = stablehlo.reduce(%s1b1dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dn1sxd = stablehlo.broadcast_in_dim %s1b1dn1sxdr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1dn1t1 = stablehlo.multiply %s1b1dn1dxh, %s1b1n1nf : tensor<32x64x56x56xf32>
    %s1b1dn1i1 = stablehlo.subtract %s1b1dn1t1, %s1b1dn1sdx : tensor<32x64x56x56xf32>
    %s1b1dn1xs = stablehlo.multiply %s1b1n1xh, %s1b1dn1sxd : tensor<32x64x56x56xf32>
    %s1b1dn1i2 = stablehlo.subtract %s1b1dn1i1, %s1b1dn1xs : tensor<32x64x56x56xf32>
    %s1b1dn1sN = stablehlo.divide %s1b1n1istd, %s1b1n1nf : tensor<32x64x56x56xf32>
    %s1b1dn1 = stablehlo.multiply %s1b1dn1sN, %s1b1dn1i2 : tensor<32x64x56x56xf32>
    %s1b1dn1dgp = stablehlo.multiply %s1b1dr1, %s1b1n1xh : tensor<32x64x56x56xf32>
    %s1b1dn1dg = stablehlo.reduce(%s1b1dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dn1db = stablehlo.reduce(%s1b1dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dc1t = stablehlo.transpose %s1b1W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b1dc1r = stablehlo.reverse %s1b1dc1t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %s1b1dc1 = stablehlo.convolution(%s1b1dn1, %s1b1dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b1dW1xt = stablehlo.transpose %s1b0o, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b1dW1dt = stablehlo.transpose %s1b1dn1, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b1dW1raw = stablehlo.convolution(%s1b1dW1xt, %s1b1dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %s1b1dW1 = stablehlo.transpose %s1b1dW1raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b1db1 = stablehlo.reduce(%s1b1dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dx = stablehlo.add %s1b1dc1, %s1b1da : tensor<32x64x56x56xf32>
    %s1b0daz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b0dam = stablehlo.compare GT, %s1b0a, %s1b0daz : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %s1b0da = stablehlo.select %s1b0dam, %s1b1dx, %s1b0daz : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %s1b0dn2dxh = stablehlo.multiply %s1b0n2gb, %s1b0da : tensor<32x64x56x56xf32>
    %s1b0dn2sdxr = stablehlo.reduce(%s1b0dn2dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dn2sdx = stablehlo.broadcast_in_dim %s1b0dn2sdxr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0dn2xd = stablehlo.multiply %s1b0n2xh, %s1b0dn2dxh : tensor<32x64x56x56xf32>
    %s1b0dn2sxdr = stablehlo.reduce(%s1b0dn2xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dn2sxd = stablehlo.broadcast_in_dim %s1b0dn2sxdr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0dn2t1 = stablehlo.multiply %s1b0dn2dxh, %s1b0n2nf : tensor<32x64x56x56xf32>
    %s1b0dn2i1 = stablehlo.subtract %s1b0dn2t1, %s1b0dn2sdx : tensor<32x64x56x56xf32>
    %s1b0dn2xs = stablehlo.multiply %s1b0n2xh, %s1b0dn2sxd : tensor<32x64x56x56xf32>
    %s1b0dn2i2 = stablehlo.subtract %s1b0dn2i1, %s1b0dn2xs : tensor<32x64x56x56xf32>
    %s1b0dn2sN = stablehlo.divide %s1b0n2istd, %s1b0n2nf : tensor<32x64x56x56xf32>
    %s1b0dn2 = stablehlo.multiply %s1b0dn2sN, %s1b0dn2i2 : tensor<32x64x56x56xf32>
    %s1b0dn2dgp = stablehlo.multiply %s1b0da, %s1b0n2xh : tensor<32x64x56x56xf32>
    %s1b0dn2dg = stablehlo.reduce(%s1b0dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dn2db = stablehlo.reduce(%s1b0da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dc2t = stablehlo.transpose %s1b0W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b0dc2r = stablehlo.reverse %s1b0dc2t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %s1b0dc2 = stablehlo.convolution(%s1b0dn2, %s1b0dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b0dW2xt = stablehlo.transpose %s1b0r1, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b0dW2dt = stablehlo.transpose %s1b0dn2, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b0dW2raw = stablehlo.convolution(%s1b0dW2xt, %s1b0dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %s1b0dW2 = stablehlo.transpose %s1b0dW2raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b0db2 = stablehlo.reduce(%s1b0dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dr1z = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b0dr1m = stablehlo.compare GT, %s1b0n1, %s1b0dr1z : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %s1b0dr1 = stablehlo.select %s1b0dr1m, %s1b0dc2, %s1b0dr1z : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %s1b0dn1dxh = stablehlo.multiply %s1b0n1gb, %s1b0dr1 : tensor<32x64x56x56xf32>
    %s1b0dn1sdxr = stablehlo.reduce(%s1b0dn1dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dn1sdx = stablehlo.broadcast_in_dim %s1b0dn1sdxr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0dn1xd = stablehlo.multiply %s1b0n1xh, %s1b0dn1dxh : tensor<32x64x56x56xf32>
    %s1b0dn1sxdr = stablehlo.reduce(%s1b0dn1xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dn1sxd = stablehlo.broadcast_in_dim %s1b0dn1sxdr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0dn1t1 = stablehlo.multiply %s1b0dn1dxh, %s1b0n1nf : tensor<32x64x56x56xf32>
    %s1b0dn1i1 = stablehlo.subtract %s1b0dn1t1, %s1b0dn1sdx : tensor<32x64x56x56xf32>
    %s1b0dn1xs = stablehlo.multiply %s1b0n1xh, %s1b0dn1sxd : tensor<32x64x56x56xf32>
    %s1b0dn1i2 = stablehlo.subtract %s1b0dn1i1, %s1b0dn1xs : tensor<32x64x56x56xf32>
    %s1b0dn1sN = stablehlo.divide %s1b0n1istd, %s1b0n1nf : tensor<32x64x56x56xf32>
    %s1b0dn1 = stablehlo.multiply %s1b0dn1sN, %s1b0dn1i2 : tensor<32x64x56x56xf32>
    %s1b0dn1dgp = stablehlo.multiply %s1b0dr1, %s1b0n1xh : tensor<32x64x56x56xf32>
    %s1b0dn1dg = stablehlo.reduce(%s1b0dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dn1db = stablehlo.reduce(%s1b0dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dc1t = stablehlo.transpose %s1b0W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b0dc1r = stablehlo.reverse %s1b0dc1t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %s1b0dc1 = stablehlo.convolution(%s1b0dn1, %s1b0dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b0dW1xt = stablehlo.transpose %stp, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b0dW1dt = stablehlo.transpose %s1b0dn1, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b0dW1raw = stablehlo.convolution(%s1b0dW1xt, %s1b0dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %s1b0dW1 = stablehlo.transpose %s1b0dW1raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b0db1 = stablehlo.reduce(%s1b0dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dx = stablehlo.add %s1b0dc1, %s1b0da : tensor<32x64x56x56xf32>
    %dmp = "stablehlo.select_and_scatter"(%str, %s1b0dx, %sc) ({
      ^bb0(%qa: tensor<f32>, %qb: tensor<f32>):
        %qge = stablehlo.compare GE, %qa, %qb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %qge : tensor<i1>
    }, {
      ^bb0(%qc: tensor<f32>, %qd: tensor<f32>):
        %qs = stablehlo.add %qc, %qd : tensor<f32>
        stablehlo.return %qs : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<32x64x112x112xf32>, tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %dstrz = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %dstrm = stablehlo.compare GT, %stn, %dstrz : (tensor<32x64x112x112xf32>, tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xi1>
    %dstr = stablehlo.select %dstrm, %dmp, %dstrz : tensor<32x64x112x112xi1>, tensor<32x64x112x112xf32>
    %dstndxh = stablehlo.multiply %stngb, %dstr : tensor<32x64x112x112xf32>
    %dstnsdxr = stablehlo.reduce(%dstndxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %dstnsdx = stablehlo.broadcast_in_dim %dstnsdxr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %dstnxd = stablehlo.multiply %stnxh, %dstndxh : tensor<32x64x112x112xf32>
    %dstnsxdr = stablehlo.reduce(%dstnxd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %dstnsxd = stablehlo.broadcast_in_dim %dstnsxdr, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %dstnt1 = stablehlo.multiply %dstndxh, %stnnf : tensor<32x64x112x112xf32>
    %dstni1 = stablehlo.subtract %dstnt1, %dstnsdx : tensor<32x64x112x112xf32>
    %dstnxs = stablehlo.multiply %stnxh, %dstnsxd : tensor<32x64x112x112xf32>
    %dstni2 = stablehlo.subtract %dstni1, %dstnxs : tensor<32x64x112x112xf32>
    %dstnsN = stablehlo.divide %stnistd, %stnnf : tensor<32x64x112x112xf32>
    %dstn = stablehlo.multiply %dstnsN, %dstni2 : tensor<32x64x112x112xf32>
    %dstndgp = stablehlo.multiply %dstr, %stnxh : tensor<32x64x112x112xf32>
    %dstndg = stablehlo.reduce(%dstndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %dstndb = stablehlo.reduce(%dstr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %dsb = stablehlo.reduce(%dstn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %dsWu = stablehlo.pad %dstn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x224x224xf32>
    %dsWxt = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %dsWdt = stablehlo.transpose %dsWu, dims = [1, 0, 2, 3] : (tensor<32x64x224x224xf32>) -> tensor<64x32x224x224xf32>
    %dsWraw = stablehlo.convolution(%dsWxt, %dsWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<64x32x224x224xf32>) -> tensor<3x64x7x7xf32>
    %dsW = stablehlo.transpose %dsWraw, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %sWl = stablehlo.constant dense<0.1> : tensor<64x3x7x7xf32>
    %sWs = stablehlo.multiply %dsW, %sWl : tensor<64x3x7x7xf32>
    %sWn = stablehlo.subtract %sW, %sWs : tensor<64x3x7x7xf32>
    %sbl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %sbs = stablehlo.multiply %dsb, %sbl : tensor<64xf32>
    %sbn = stablehlo.subtract %sb, %sbs : tensor<64xf32>
    %sgl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %sgs = stablehlo.multiply %dstndg, %sgl : tensor<64xf32>
    %sgn = stablehlo.subtract %sg, %sgs : tensor<64xf32>
    %sbtl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %sbts = stablehlo.multiply %dstndb, %sbtl : tensor<64xf32>
    %sbtn = stablehlo.subtract %sbt, %sbts : tensor<64xf32>
    %s1b0W1l = stablehlo.constant dense<0.1> : tensor<64x64x3x3xf32>
    %s1b0W1s = stablehlo.multiply %s1b0dW1, %s1b0W1l : tensor<64x64x3x3xf32>
    %s1b0W1n = stablehlo.subtract %s1b0W1, %s1b0W1s : tensor<64x64x3x3xf32>
    %s1b0b1l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b0b1s = stablehlo.multiply %s1b0db1, %s1b0b1l : tensor<64xf32>
    %s1b0b1n = stablehlo.subtract %s1b0b1, %s1b0b1s : tensor<64xf32>
    %s1b0g1l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b0g1s = stablehlo.multiply %s1b0dn1dg, %s1b0g1l : tensor<64xf32>
    %s1b0g1n = stablehlo.subtract %s1b0g1, %s1b0g1s : tensor<64xf32>
    %s1b0bt1l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b0bt1s = stablehlo.multiply %s1b0dn1db, %s1b0bt1l : tensor<64xf32>
    %s1b0bt1n = stablehlo.subtract %s1b0bt1, %s1b0bt1s : tensor<64xf32>
    %s1b0W2l = stablehlo.constant dense<0.1> : tensor<64x64x3x3xf32>
    %s1b0W2s = stablehlo.multiply %s1b0dW2, %s1b0W2l : tensor<64x64x3x3xf32>
    %s1b0W2n = stablehlo.subtract %s1b0W2, %s1b0W2s : tensor<64x64x3x3xf32>
    %s1b0b2l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b0b2s = stablehlo.multiply %s1b0db2, %s1b0b2l : tensor<64xf32>
    %s1b0b2n = stablehlo.subtract %s1b0b2, %s1b0b2s : tensor<64xf32>
    %s1b0g2l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b0g2s = stablehlo.multiply %s1b0dn2dg, %s1b0g2l : tensor<64xf32>
    %s1b0g2n = stablehlo.subtract %s1b0g2, %s1b0g2s : tensor<64xf32>
    %s1b0bt2l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b0bt2s = stablehlo.multiply %s1b0dn2db, %s1b0bt2l : tensor<64xf32>
    %s1b0bt2n = stablehlo.subtract %s1b0bt2, %s1b0bt2s : tensor<64xf32>
    %s1b1W1l = stablehlo.constant dense<0.1> : tensor<64x64x3x3xf32>
    %s1b1W1s = stablehlo.multiply %s1b1dW1, %s1b1W1l : tensor<64x64x3x3xf32>
    %s1b1W1n = stablehlo.subtract %s1b1W1, %s1b1W1s : tensor<64x64x3x3xf32>
    %s1b1b1l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b1b1s = stablehlo.multiply %s1b1db1, %s1b1b1l : tensor<64xf32>
    %s1b1b1n = stablehlo.subtract %s1b1b1, %s1b1b1s : tensor<64xf32>
    %s1b1g1l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b1g1s = stablehlo.multiply %s1b1dn1dg, %s1b1g1l : tensor<64xf32>
    %s1b1g1n = stablehlo.subtract %s1b1g1, %s1b1g1s : tensor<64xf32>
    %s1b1bt1l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b1bt1s = stablehlo.multiply %s1b1dn1db, %s1b1bt1l : tensor<64xf32>
    %s1b1bt1n = stablehlo.subtract %s1b1bt1, %s1b1bt1s : tensor<64xf32>
    %s1b1W2l = stablehlo.constant dense<0.1> : tensor<64x64x3x3xf32>
    %s1b1W2s = stablehlo.multiply %s1b1dW2, %s1b1W2l : tensor<64x64x3x3xf32>
    %s1b1W2n = stablehlo.subtract %s1b1W2, %s1b1W2s : tensor<64x64x3x3xf32>
    %s1b1b2l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b1b2s = stablehlo.multiply %s1b1db2, %s1b1b2l : tensor<64xf32>
    %s1b1b2n = stablehlo.subtract %s1b1b2, %s1b1b2s : tensor<64xf32>
    %s1b1g2l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b1g2s = stablehlo.multiply %s1b1dn2dg, %s1b1g2l : tensor<64xf32>
    %s1b1g2n = stablehlo.subtract %s1b1g2, %s1b1g2s : tensor<64xf32>
    %s1b1bt2l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b1bt2s = stablehlo.multiply %s1b1dn2db, %s1b1bt2l : tensor<64xf32>
    %s1b1bt2n = stablehlo.subtract %s1b1bt2, %s1b1bt2s : tensor<64xf32>
    %s1b2W1l = stablehlo.constant dense<0.1> : tensor<64x64x3x3xf32>
    %s1b2W1s = stablehlo.multiply %s1b2dW1, %s1b2W1l : tensor<64x64x3x3xf32>
    %s1b2W1n = stablehlo.subtract %s1b2W1, %s1b2W1s : tensor<64x64x3x3xf32>
    %s1b2b1l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b2b1s = stablehlo.multiply %s1b2db1, %s1b2b1l : tensor<64xf32>
    %s1b2b1n = stablehlo.subtract %s1b2b1, %s1b2b1s : tensor<64xf32>
    %s1b2g1l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b2g1s = stablehlo.multiply %s1b2dn1dg, %s1b2g1l : tensor<64xf32>
    %s1b2g1n = stablehlo.subtract %s1b2g1, %s1b2g1s : tensor<64xf32>
    %s1b2bt1l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b2bt1s = stablehlo.multiply %s1b2dn1db, %s1b2bt1l : tensor<64xf32>
    %s1b2bt1n = stablehlo.subtract %s1b2bt1, %s1b2bt1s : tensor<64xf32>
    %s1b2W2l = stablehlo.constant dense<0.1> : tensor<64x64x3x3xf32>
    %s1b2W2s = stablehlo.multiply %s1b2dW2, %s1b2W2l : tensor<64x64x3x3xf32>
    %s1b2W2n = stablehlo.subtract %s1b2W2, %s1b2W2s : tensor<64x64x3x3xf32>
    %s1b2b2l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b2b2s = stablehlo.multiply %s1b2db2, %s1b2b2l : tensor<64xf32>
    %s1b2b2n = stablehlo.subtract %s1b2b2, %s1b2b2s : tensor<64xf32>
    %s1b2g2l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b2g2s = stablehlo.multiply %s1b2dn2dg, %s1b2g2l : tensor<64xf32>
    %s1b2g2n = stablehlo.subtract %s1b2g2, %s1b2g2s : tensor<64xf32>
    %s1b2bt2l = stablehlo.constant dense<0.1> : tensor<64xf32>
    %s1b2bt2s = stablehlo.multiply %s1b2dn2db, %s1b2bt2l : tensor<64xf32>
    %s1b2bt2n = stablehlo.subtract %s1b2bt2, %s1b2bt2s : tensor<64xf32>
    %d2W1l = stablehlo.constant dense<0.1> : tensor<128x64x3x3xf32>
    %d2W1s = stablehlo.multiply %d2dW1, %d2W1l : tensor<128x64x3x3xf32>
    %d2W1n = stablehlo.subtract %d2W1, %d2W1s : tensor<128x64x3x3xf32>
    %d2b1l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %d2b1s = stablehlo.multiply %d2db1, %d2b1l : tensor<128xf32>
    %d2b1n = stablehlo.subtract %d2b1, %d2b1s : tensor<128xf32>
    %d2g1l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %d2g1s = stablehlo.multiply %d2dn1dg, %d2g1l : tensor<128xf32>
    %d2g1n = stablehlo.subtract %d2g1, %d2g1s : tensor<128xf32>
    %d2bt1l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %d2bt1s = stablehlo.multiply %d2dn1db, %d2bt1l : tensor<128xf32>
    %d2bt1n = stablehlo.subtract %d2bt1, %d2bt1s : tensor<128xf32>
    %d2W2l = stablehlo.constant dense<0.1> : tensor<128x128x3x3xf32>
    %d2W2s = stablehlo.multiply %d2dW2, %d2W2l : tensor<128x128x3x3xf32>
    %d2W2n = stablehlo.subtract %d2W2, %d2W2s : tensor<128x128x3x3xf32>
    %d2b2l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %d2b2s = stablehlo.multiply %d2db2, %d2b2l : tensor<128xf32>
    %d2b2n = stablehlo.subtract %d2b2, %d2b2s : tensor<128xf32>
    %d2g2l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %d2g2s = stablehlo.multiply %d2dn2dg, %d2g2l : tensor<128xf32>
    %d2g2n = stablehlo.subtract %d2g2, %d2g2s : tensor<128xf32>
    %d2bt2l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %d2bt2s = stablehlo.multiply %d2dn2db, %d2bt2l : tensor<128xf32>
    %d2bt2n = stablehlo.subtract %d2bt2, %d2bt2s : tensor<128xf32>
    %d2Wpl = stablehlo.constant dense<0.1> : tensor<128x64x3x3xf32>
    %d2Wps = stablehlo.multiply %d2dWp, %d2Wpl : tensor<128x64x3x3xf32>
    %d2Wpn = stablehlo.subtract %d2Wp, %d2Wps : tensor<128x64x3x3xf32>
    %d2bpl = stablehlo.constant dense<0.1> : tensor<128xf32>
    %d2bps = stablehlo.multiply %d2dbp, %d2bpl : tensor<128xf32>
    %d2bpn = stablehlo.subtract %d2bp, %d2bps : tensor<128xf32>
    %d2gpl = stablehlo.constant dense<0.1> : tensor<128xf32>
    %d2gps = stablehlo.multiply %d2dnpdg, %d2gpl : tensor<128xf32>
    %d2gpn = stablehlo.subtract %d2gp, %d2gps : tensor<128xf32>
    %d2btpl = stablehlo.constant dense<0.1> : tensor<128xf32>
    %d2btps = stablehlo.multiply %d2dnpdb, %d2btpl : tensor<128xf32>
    %d2btpn = stablehlo.subtract %d2btp, %d2btps : tensor<128xf32>
    %s2b0W1l = stablehlo.constant dense<0.1> : tensor<128x128x3x3xf32>
    %s2b0W1s = stablehlo.multiply %s2b0dW1, %s2b0W1l : tensor<128x128x3x3xf32>
    %s2b0W1n = stablehlo.subtract %s2b0W1, %s2b0W1s : tensor<128x128x3x3xf32>
    %s2b0b1l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b0b1s = stablehlo.multiply %s2b0db1, %s2b0b1l : tensor<128xf32>
    %s2b0b1n = stablehlo.subtract %s2b0b1, %s2b0b1s : tensor<128xf32>
    %s2b0g1l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b0g1s = stablehlo.multiply %s2b0dn1dg, %s2b0g1l : tensor<128xf32>
    %s2b0g1n = stablehlo.subtract %s2b0g1, %s2b0g1s : tensor<128xf32>
    %s2b0bt1l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b0bt1s = stablehlo.multiply %s2b0dn1db, %s2b0bt1l : tensor<128xf32>
    %s2b0bt1n = stablehlo.subtract %s2b0bt1, %s2b0bt1s : tensor<128xf32>
    %s2b0W2l = stablehlo.constant dense<0.1> : tensor<128x128x3x3xf32>
    %s2b0W2s = stablehlo.multiply %s2b0dW2, %s2b0W2l : tensor<128x128x3x3xf32>
    %s2b0W2n = stablehlo.subtract %s2b0W2, %s2b0W2s : tensor<128x128x3x3xf32>
    %s2b0b2l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b0b2s = stablehlo.multiply %s2b0db2, %s2b0b2l : tensor<128xf32>
    %s2b0b2n = stablehlo.subtract %s2b0b2, %s2b0b2s : tensor<128xf32>
    %s2b0g2l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b0g2s = stablehlo.multiply %s2b0dn2dg, %s2b0g2l : tensor<128xf32>
    %s2b0g2n = stablehlo.subtract %s2b0g2, %s2b0g2s : tensor<128xf32>
    %s2b0bt2l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b0bt2s = stablehlo.multiply %s2b0dn2db, %s2b0bt2l : tensor<128xf32>
    %s2b0bt2n = stablehlo.subtract %s2b0bt2, %s2b0bt2s : tensor<128xf32>
    %s2b1W1l = stablehlo.constant dense<0.1> : tensor<128x128x3x3xf32>
    %s2b1W1s = stablehlo.multiply %s2b1dW1, %s2b1W1l : tensor<128x128x3x3xf32>
    %s2b1W1n = stablehlo.subtract %s2b1W1, %s2b1W1s : tensor<128x128x3x3xf32>
    %s2b1b1l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b1b1s = stablehlo.multiply %s2b1db1, %s2b1b1l : tensor<128xf32>
    %s2b1b1n = stablehlo.subtract %s2b1b1, %s2b1b1s : tensor<128xf32>
    %s2b1g1l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b1g1s = stablehlo.multiply %s2b1dn1dg, %s2b1g1l : tensor<128xf32>
    %s2b1g1n = stablehlo.subtract %s2b1g1, %s2b1g1s : tensor<128xf32>
    %s2b1bt1l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b1bt1s = stablehlo.multiply %s2b1dn1db, %s2b1bt1l : tensor<128xf32>
    %s2b1bt1n = stablehlo.subtract %s2b1bt1, %s2b1bt1s : tensor<128xf32>
    %s2b1W2l = stablehlo.constant dense<0.1> : tensor<128x128x3x3xf32>
    %s2b1W2s = stablehlo.multiply %s2b1dW2, %s2b1W2l : tensor<128x128x3x3xf32>
    %s2b1W2n = stablehlo.subtract %s2b1W2, %s2b1W2s : tensor<128x128x3x3xf32>
    %s2b1b2l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b1b2s = stablehlo.multiply %s2b1db2, %s2b1b2l : tensor<128xf32>
    %s2b1b2n = stablehlo.subtract %s2b1b2, %s2b1b2s : tensor<128xf32>
    %s2b1g2l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b1g2s = stablehlo.multiply %s2b1dn2dg, %s2b1g2l : tensor<128xf32>
    %s2b1g2n = stablehlo.subtract %s2b1g2, %s2b1g2s : tensor<128xf32>
    %s2b1bt2l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b1bt2s = stablehlo.multiply %s2b1dn2db, %s2b1bt2l : tensor<128xf32>
    %s2b1bt2n = stablehlo.subtract %s2b1bt2, %s2b1bt2s : tensor<128xf32>
    %s2b2W1l = stablehlo.constant dense<0.1> : tensor<128x128x3x3xf32>
    %s2b2W1s = stablehlo.multiply %s2b2dW1, %s2b2W1l : tensor<128x128x3x3xf32>
    %s2b2W1n = stablehlo.subtract %s2b2W1, %s2b2W1s : tensor<128x128x3x3xf32>
    %s2b2b1l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b2b1s = stablehlo.multiply %s2b2db1, %s2b2b1l : tensor<128xf32>
    %s2b2b1n = stablehlo.subtract %s2b2b1, %s2b2b1s : tensor<128xf32>
    %s2b2g1l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b2g1s = stablehlo.multiply %s2b2dn1dg, %s2b2g1l : tensor<128xf32>
    %s2b2g1n = stablehlo.subtract %s2b2g1, %s2b2g1s : tensor<128xf32>
    %s2b2bt1l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b2bt1s = stablehlo.multiply %s2b2dn1db, %s2b2bt1l : tensor<128xf32>
    %s2b2bt1n = stablehlo.subtract %s2b2bt1, %s2b2bt1s : tensor<128xf32>
    %s2b2W2l = stablehlo.constant dense<0.1> : tensor<128x128x3x3xf32>
    %s2b2W2s = stablehlo.multiply %s2b2dW2, %s2b2W2l : tensor<128x128x3x3xf32>
    %s2b2W2n = stablehlo.subtract %s2b2W2, %s2b2W2s : tensor<128x128x3x3xf32>
    %s2b2b2l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b2b2s = stablehlo.multiply %s2b2db2, %s2b2b2l : tensor<128xf32>
    %s2b2b2n = stablehlo.subtract %s2b2b2, %s2b2b2s : tensor<128xf32>
    %s2b2g2l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b2g2s = stablehlo.multiply %s2b2dn2dg, %s2b2g2l : tensor<128xf32>
    %s2b2g2n = stablehlo.subtract %s2b2g2, %s2b2g2s : tensor<128xf32>
    %s2b2bt2l = stablehlo.constant dense<0.1> : tensor<128xf32>
    %s2b2bt2s = stablehlo.multiply %s2b2dn2db, %s2b2bt2l : tensor<128xf32>
    %s2b2bt2n = stablehlo.subtract %s2b2bt2, %s2b2bt2s : tensor<128xf32>
    %d3W1l = stablehlo.constant dense<0.1> : tensor<256x128x3x3xf32>
    %d3W1s = stablehlo.multiply %d3dW1, %d3W1l : tensor<256x128x3x3xf32>
    %d3W1n = stablehlo.subtract %d3W1, %d3W1s : tensor<256x128x3x3xf32>
    %d3b1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %d3b1s = stablehlo.multiply %d3db1, %d3b1l : tensor<256xf32>
    %d3b1n = stablehlo.subtract %d3b1, %d3b1s : tensor<256xf32>
    %d3g1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %d3g1s = stablehlo.multiply %d3dn1dg, %d3g1l : tensor<256xf32>
    %d3g1n = stablehlo.subtract %d3g1, %d3g1s : tensor<256xf32>
    %d3bt1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %d3bt1s = stablehlo.multiply %d3dn1db, %d3bt1l : tensor<256xf32>
    %d3bt1n = stablehlo.subtract %d3bt1, %d3bt1s : tensor<256xf32>
    %d3W2l = stablehlo.constant dense<0.1> : tensor<256x256x3x3xf32>
    %d3W2s = stablehlo.multiply %d3dW2, %d3W2l : tensor<256x256x3x3xf32>
    %d3W2n = stablehlo.subtract %d3W2, %d3W2s : tensor<256x256x3x3xf32>
    %d3b2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %d3b2s = stablehlo.multiply %d3db2, %d3b2l : tensor<256xf32>
    %d3b2n = stablehlo.subtract %d3b2, %d3b2s : tensor<256xf32>
    %d3g2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %d3g2s = stablehlo.multiply %d3dn2dg, %d3g2l : tensor<256xf32>
    %d3g2n = stablehlo.subtract %d3g2, %d3g2s : tensor<256xf32>
    %d3bt2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %d3bt2s = stablehlo.multiply %d3dn2db, %d3bt2l : tensor<256xf32>
    %d3bt2n = stablehlo.subtract %d3bt2, %d3bt2s : tensor<256xf32>
    %d3Wpl = stablehlo.constant dense<0.1> : tensor<256x128x3x3xf32>
    %d3Wps = stablehlo.multiply %d3dWp, %d3Wpl : tensor<256x128x3x3xf32>
    %d3Wpn = stablehlo.subtract %d3Wp, %d3Wps : tensor<256x128x3x3xf32>
    %d3bpl = stablehlo.constant dense<0.1> : tensor<256xf32>
    %d3bps = stablehlo.multiply %d3dbp, %d3bpl : tensor<256xf32>
    %d3bpn = stablehlo.subtract %d3bp, %d3bps : tensor<256xf32>
    %d3gpl = stablehlo.constant dense<0.1> : tensor<256xf32>
    %d3gps = stablehlo.multiply %d3dnpdg, %d3gpl : tensor<256xf32>
    %d3gpn = stablehlo.subtract %d3gp, %d3gps : tensor<256xf32>
    %d3btpl = stablehlo.constant dense<0.1> : tensor<256xf32>
    %d3btps = stablehlo.multiply %d3dnpdb, %d3btpl : tensor<256xf32>
    %d3btpn = stablehlo.subtract %d3btp, %d3btps : tensor<256xf32>
    %s3b0W1l = stablehlo.constant dense<0.1> : tensor<256x256x3x3xf32>
    %s3b0W1s = stablehlo.multiply %s3b0dW1, %s3b0W1l : tensor<256x256x3x3xf32>
    %s3b0W1n = stablehlo.subtract %s3b0W1, %s3b0W1s : tensor<256x256x3x3xf32>
    %s3b0b1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b0b1s = stablehlo.multiply %s3b0db1, %s3b0b1l : tensor<256xf32>
    %s3b0b1n = stablehlo.subtract %s3b0b1, %s3b0b1s : tensor<256xf32>
    %s3b0g1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b0g1s = stablehlo.multiply %s3b0dn1dg, %s3b0g1l : tensor<256xf32>
    %s3b0g1n = stablehlo.subtract %s3b0g1, %s3b0g1s : tensor<256xf32>
    %s3b0bt1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b0bt1s = stablehlo.multiply %s3b0dn1db, %s3b0bt1l : tensor<256xf32>
    %s3b0bt1n = stablehlo.subtract %s3b0bt1, %s3b0bt1s : tensor<256xf32>
    %s3b0W2l = stablehlo.constant dense<0.1> : tensor<256x256x3x3xf32>
    %s3b0W2s = stablehlo.multiply %s3b0dW2, %s3b0W2l : tensor<256x256x3x3xf32>
    %s3b0W2n = stablehlo.subtract %s3b0W2, %s3b0W2s : tensor<256x256x3x3xf32>
    %s3b0b2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b0b2s = stablehlo.multiply %s3b0db2, %s3b0b2l : tensor<256xf32>
    %s3b0b2n = stablehlo.subtract %s3b0b2, %s3b0b2s : tensor<256xf32>
    %s3b0g2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b0g2s = stablehlo.multiply %s3b0dn2dg, %s3b0g2l : tensor<256xf32>
    %s3b0g2n = stablehlo.subtract %s3b0g2, %s3b0g2s : tensor<256xf32>
    %s3b0bt2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b0bt2s = stablehlo.multiply %s3b0dn2db, %s3b0bt2l : tensor<256xf32>
    %s3b0bt2n = stablehlo.subtract %s3b0bt2, %s3b0bt2s : tensor<256xf32>
    %s3b1W1l = stablehlo.constant dense<0.1> : tensor<256x256x3x3xf32>
    %s3b1W1s = stablehlo.multiply %s3b1dW1, %s3b1W1l : tensor<256x256x3x3xf32>
    %s3b1W1n = stablehlo.subtract %s3b1W1, %s3b1W1s : tensor<256x256x3x3xf32>
    %s3b1b1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b1b1s = stablehlo.multiply %s3b1db1, %s3b1b1l : tensor<256xf32>
    %s3b1b1n = stablehlo.subtract %s3b1b1, %s3b1b1s : tensor<256xf32>
    %s3b1g1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b1g1s = stablehlo.multiply %s3b1dn1dg, %s3b1g1l : tensor<256xf32>
    %s3b1g1n = stablehlo.subtract %s3b1g1, %s3b1g1s : tensor<256xf32>
    %s3b1bt1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b1bt1s = stablehlo.multiply %s3b1dn1db, %s3b1bt1l : tensor<256xf32>
    %s3b1bt1n = stablehlo.subtract %s3b1bt1, %s3b1bt1s : tensor<256xf32>
    %s3b1W2l = stablehlo.constant dense<0.1> : tensor<256x256x3x3xf32>
    %s3b1W2s = stablehlo.multiply %s3b1dW2, %s3b1W2l : tensor<256x256x3x3xf32>
    %s3b1W2n = stablehlo.subtract %s3b1W2, %s3b1W2s : tensor<256x256x3x3xf32>
    %s3b1b2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b1b2s = stablehlo.multiply %s3b1db2, %s3b1b2l : tensor<256xf32>
    %s3b1b2n = stablehlo.subtract %s3b1b2, %s3b1b2s : tensor<256xf32>
    %s3b1g2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b1g2s = stablehlo.multiply %s3b1dn2dg, %s3b1g2l : tensor<256xf32>
    %s3b1g2n = stablehlo.subtract %s3b1g2, %s3b1g2s : tensor<256xf32>
    %s3b1bt2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b1bt2s = stablehlo.multiply %s3b1dn2db, %s3b1bt2l : tensor<256xf32>
    %s3b1bt2n = stablehlo.subtract %s3b1bt2, %s3b1bt2s : tensor<256xf32>
    %s3b2W1l = stablehlo.constant dense<0.1> : tensor<256x256x3x3xf32>
    %s3b2W1s = stablehlo.multiply %s3b2dW1, %s3b2W1l : tensor<256x256x3x3xf32>
    %s3b2W1n = stablehlo.subtract %s3b2W1, %s3b2W1s : tensor<256x256x3x3xf32>
    %s3b2b1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b2b1s = stablehlo.multiply %s3b2db1, %s3b2b1l : tensor<256xf32>
    %s3b2b1n = stablehlo.subtract %s3b2b1, %s3b2b1s : tensor<256xf32>
    %s3b2g1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b2g1s = stablehlo.multiply %s3b2dn1dg, %s3b2g1l : tensor<256xf32>
    %s3b2g1n = stablehlo.subtract %s3b2g1, %s3b2g1s : tensor<256xf32>
    %s3b2bt1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b2bt1s = stablehlo.multiply %s3b2dn1db, %s3b2bt1l : tensor<256xf32>
    %s3b2bt1n = stablehlo.subtract %s3b2bt1, %s3b2bt1s : tensor<256xf32>
    %s3b2W2l = stablehlo.constant dense<0.1> : tensor<256x256x3x3xf32>
    %s3b2W2s = stablehlo.multiply %s3b2dW2, %s3b2W2l : tensor<256x256x3x3xf32>
    %s3b2W2n = stablehlo.subtract %s3b2W2, %s3b2W2s : tensor<256x256x3x3xf32>
    %s3b2b2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b2b2s = stablehlo.multiply %s3b2db2, %s3b2b2l : tensor<256xf32>
    %s3b2b2n = stablehlo.subtract %s3b2b2, %s3b2b2s : tensor<256xf32>
    %s3b2g2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b2g2s = stablehlo.multiply %s3b2dn2dg, %s3b2g2l : tensor<256xf32>
    %s3b2g2n = stablehlo.subtract %s3b2g2, %s3b2g2s : tensor<256xf32>
    %s3b2bt2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b2bt2s = stablehlo.multiply %s3b2dn2db, %s3b2bt2l : tensor<256xf32>
    %s3b2bt2n = stablehlo.subtract %s3b2bt2, %s3b2bt2s : tensor<256xf32>
    %s3b3W1l = stablehlo.constant dense<0.1> : tensor<256x256x3x3xf32>
    %s3b3W1s = stablehlo.multiply %s3b3dW1, %s3b3W1l : tensor<256x256x3x3xf32>
    %s3b3W1n = stablehlo.subtract %s3b3W1, %s3b3W1s : tensor<256x256x3x3xf32>
    %s3b3b1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b3b1s = stablehlo.multiply %s3b3db1, %s3b3b1l : tensor<256xf32>
    %s3b3b1n = stablehlo.subtract %s3b3b1, %s3b3b1s : tensor<256xf32>
    %s3b3g1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b3g1s = stablehlo.multiply %s3b3dn1dg, %s3b3g1l : tensor<256xf32>
    %s3b3g1n = stablehlo.subtract %s3b3g1, %s3b3g1s : tensor<256xf32>
    %s3b3bt1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b3bt1s = stablehlo.multiply %s3b3dn1db, %s3b3bt1l : tensor<256xf32>
    %s3b3bt1n = stablehlo.subtract %s3b3bt1, %s3b3bt1s : tensor<256xf32>
    %s3b3W2l = stablehlo.constant dense<0.1> : tensor<256x256x3x3xf32>
    %s3b3W2s = stablehlo.multiply %s3b3dW2, %s3b3W2l : tensor<256x256x3x3xf32>
    %s3b3W2n = stablehlo.subtract %s3b3W2, %s3b3W2s : tensor<256x256x3x3xf32>
    %s3b3b2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b3b2s = stablehlo.multiply %s3b3db2, %s3b3b2l : tensor<256xf32>
    %s3b3b2n = stablehlo.subtract %s3b3b2, %s3b3b2s : tensor<256xf32>
    %s3b3g2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b3g2s = stablehlo.multiply %s3b3dn2dg, %s3b3g2l : tensor<256xf32>
    %s3b3g2n = stablehlo.subtract %s3b3g2, %s3b3g2s : tensor<256xf32>
    %s3b3bt2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b3bt2s = stablehlo.multiply %s3b3dn2db, %s3b3bt2l : tensor<256xf32>
    %s3b3bt2n = stablehlo.subtract %s3b3bt2, %s3b3bt2s : tensor<256xf32>
    %s3b4W1l = stablehlo.constant dense<0.1> : tensor<256x256x3x3xf32>
    %s3b4W1s = stablehlo.multiply %s3b4dW1, %s3b4W1l : tensor<256x256x3x3xf32>
    %s3b4W1n = stablehlo.subtract %s3b4W1, %s3b4W1s : tensor<256x256x3x3xf32>
    %s3b4b1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b4b1s = stablehlo.multiply %s3b4db1, %s3b4b1l : tensor<256xf32>
    %s3b4b1n = stablehlo.subtract %s3b4b1, %s3b4b1s : tensor<256xf32>
    %s3b4g1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b4g1s = stablehlo.multiply %s3b4dn1dg, %s3b4g1l : tensor<256xf32>
    %s3b4g1n = stablehlo.subtract %s3b4g1, %s3b4g1s : tensor<256xf32>
    %s3b4bt1l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b4bt1s = stablehlo.multiply %s3b4dn1db, %s3b4bt1l : tensor<256xf32>
    %s3b4bt1n = stablehlo.subtract %s3b4bt1, %s3b4bt1s : tensor<256xf32>
    %s3b4W2l = stablehlo.constant dense<0.1> : tensor<256x256x3x3xf32>
    %s3b4W2s = stablehlo.multiply %s3b4dW2, %s3b4W2l : tensor<256x256x3x3xf32>
    %s3b4W2n = stablehlo.subtract %s3b4W2, %s3b4W2s : tensor<256x256x3x3xf32>
    %s3b4b2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b4b2s = stablehlo.multiply %s3b4db2, %s3b4b2l : tensor<256xf32>
    %s3b4b2n = stablehlo.subtract %s3b4b2, %s3b4b2s : tensor<256xf32>
    %s3b4g2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b4g2s = stablehlo.multiply %s3b4dn2dg, %s3b4g2l : tensor<256xf32>
    %s3b4g2n = stablehlo.subtract %s3b4g2, %s3b4g2s : tensor<256xf32>
    %s3b4bt2l = stablehlo.constant dense<0.1> : tensor<256xf32>
    %s3b4bt2s = stablehlo.multiply %s3b4dn2db, %s3b4bt2l : tensor<256xf32>
    %s3b4bt2n = stablehlo.subtract %s3b4bt2, %s3b4bt2s : tensor<256xf32>
    %d4W1l = stablehlo.constant dense<0.1> : tensor<512x256x3x3xf32>
    %d4W1s = stablehlo.multiply %d4dW1, %d4W1l : tensor<512x256x3x3xf32>
    %d4W1n = stablehlo.subtract %d4W1, %d4W1s : tensor<512x256x3x3xf32>
    %d4b1l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %d4b1s = stablehlo.multiply %d4db1, %d4b1l : tensor<512xf32>
    %d4b1n = stablehlo.subtract %d4b1, %d4b1s : tensor<512xf32>
    %d4g1l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %d4g1s = stablehlo.multiply %d4dn1dg, %d4g1l : tensor<512xf32>
    %d4g1n = stablehlo.subtract %d4g1, %d4g1s : tensor<512xf32>
    %d4bt1l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %d4bt1s = stablehlo.multiply %d4dn1db, %d4bt1l : tensor<512xf32>
    %d4bt1n = stablehlo.subtract %d4bt1, %d4bt1s : tensor<512xf32>
    %d4W2l = stablehlo.constant dense<0.1> : tensor<512x512x3x3xf32>
    %d4W2s = stablehlo.multiply %d4dW2, %d4W2l : tensor<512x512x3x3xf32>
    %d4W2n = stablehlo.subtract %d4W2, %d4W2s : tensor<512x512x3x3xf32>
    %d4b2l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %d4b2s = stablehlo.multiply %d4db2, %d4b2l : tensor<512xf32>
    %d4b2n = stablehlo.subtract %d4b2, %d4b2s : tensor<512xf32>
    %d4g2l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %d4g2s = stablehlo.multiply %d4dn2dg, %d4g2l : tensor<512xf32>
    %d4g2n = stablehlo.subtract %d4g2, %d4g2s : tensor<512xf32>
    %d4bt2l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %d4bt2s = stablehlo.multiply %d4dn2db, %d4bt2l : tensor<512xf32>
    %d4bt2n = stablehlo.subtract %d4bt2, %d4bt2s : tensor<512xf32>
    %d4Wpl = stablehlo.constant dense<0.1> : tensor<512x256x3x3xf32>
    %d4Wps = stablehlo.multiply %d4dWp, %d4Wpl : tensor<512x256x3x3xf32>
    %d4Wpn = stablehlo.subtract %d4Wp, %d4Wps : tensor<512x256x3x3xf32>
    %d4bpl = stablehlo.constant dense<0.1> : tensor<512xf32>
    %d4bps = stablehlo.multiply %d4dbp, %d4bpl : tensor<512xf32>
    %d4bpn = stablehlo.subtract %d4bp, %d4bps : tensor<512xf32>
    %d4gpl = stablehlo.constant dense<0.1> : tensor<512xf32>
    %d4gps = stablehlo.multiply %d4dnpdg, %d4gpl : tensor<512xf32>
    %d4gpn = stablehlo.subtract %d4gp, %d4gps : tensor<512xf32>
    %d4btpl = stablehlo.constant dense<0.1> : tensor<512xf32>
    %d4btps = stablehlo.multiply %d4dnpdb, %d4btpl : tensor<512xf32>
    %d4btpn = stablehlo.subtract %d4btp, %d4btps : tensor<512xf32>
    %s4b0W1l = stablehlo.constant dense<0.1> : tensor<512x512x3x3xf32>
    %s4b0W1s = stablehlo.multiply %s4b0dW1, %s4b0W1l : tensor<512x512x3x3xf32>
    %s4b0W1n = stablehlo.subtract %s4b0W1, %s4b0W1s : tensor<512x512x3x3xf32>
    %s4b0b1l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %s4b0b1s = stablehlo.multiply %s4b0db1, %s4b0b1l : tensor<512xf32>
    %s4b0b1n = stablehlo.subtract %s4b0b1, %s4b0b1s : tensor<512xf32>
    %s4b0g1l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %s4b0g1s = stablehlo.multiply %s4b0dn1dg, %s4b0g1l : tensor<512xf32>
    %s4b0g1n = stablehlo.subtract %s4b0g1, %s4b0g1s : tensor<512xf32>
    %s4b0bt1l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %s4b0bt1s = stablehlo.multiply %s4b0dn1db, %s4b0bt1l : tensor<512xf32>
    %s4b0bt1n = stablehlo.subtract %s4b0bt1, %s4b0bt1s : tensor<512xf32>
    %s4b0W2l = stablehlo.constant dense<0.1> : tensor<512x512x3x3xf32>
    %s4b0W2s = stablehlo.multiply %s4b0dW2, %s4b0W2l : tensor<512x512x3x3xf32>
    %s4b0W2n = stablehlo.subtract %s4b0W2, %s4b0W2s : tensor<512x512x3x3xf32>
    %s4b0b2l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %s4b0b2s = stablehlo.multiply %s4b0db2, %s4b0b2l : tensor<512xf32>
    %s4b0b2n = stablehlo.subtract %s4b0b2, %s4b0b2s : tensor<512xf32>
    %s4b0g2l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %s4b0g2s = stablehlo.multiply %s4b0dn2dg, %s4b0g2l : tensor<512xf32>
    %s4b0g2n = stablehlo.subtract %s4b0g2, %s4b0g2s : tensor<512xf32>
    %s4b0bt2l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %s4b0bt2s = stablehlo.multiply %s4b0dn2db, %s4b0bt2l : tensor<512xf32>
    %s4b0bt2n = stablehlo.subtract %s4b0bt2, %s4b0bt2s : tensor<512xf32>
    %s4b1W1l = stablehlo.constant dense<0.1> : tensor<512x512x3x3xf32>
    %s4b1W1s = stablehlo.multiply %s4b1dW1, %s4b1W1l : tensor<512x512x3x3xf32>
    %s4b1W1n = stablehlo.subtract %s4b1W1, %s4b1W1s : tensor<512x512x3x3xf32>
    %s4b1b1l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %s4b1b1s = stablehlo.multiply %s4b1db1, %s4b1b1l : tensor<512xf32>
    %s4b1b1n = stablehlo.subtract %s4b1b1, %s4b1b1s : tensor<512xf32>
    %s4b1g1l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %s4b1g1s = stablehlo.multiply %s4b1dn1dg, %s4b1g1l : tensor<512xf32>
    %s4b1g1n = stablehlo.subtract %s4b1g1, %s4b1g1s : tensor<512xf32>
    %s4b1bt1l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %s4b1bt1s = stablehlo.multiply %s4b1dn1db, %s4b1bt1l : tensor<512xf32>
    %s4b1bt1n = stablehlo.subtract %s4b1bt1, %s4b1bt1s : tensor<512xf32>
    %s4b1W2l = stablehlo.constant dense<0.1> : tensor<512x512x3x3xf32>
    %s4b1W2s = stablehlo.multiply %s4b1dW2, %s4b1W2l : tensor<512x512x3x3xf32>
    %s4b1W2n = stablehlo.subtract %s4b1W2, %s4b1W2s : tensor<512x512x3x3xf32>
    %s4b1b2l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %s4b1b2s = stablehlo.multiply %s4b1db2, %s4b1b2l : tensor<512xf32>
    %s4b1b2n = stablehlo.subtract %s4b1b2, %s4b1b2s : tensor<512xf32>
    %s4b1g2l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %s4b1g2s = stablehlo.multiply %s4b1dn2dg, %s4b1g2l : tensor<512xf32>
    %s4b1g2n = stablehlo.subtract %s4b1g2, %s4b1g2s : tensor<512xf32>
    %s4b1bt2l = stablehlo.constant dense<0.1> : tensor<512xf32>
    %s4b1bt2s = stablehlo.multiply %s4b1dn2db, %s4b1bt2l : tensor<512xf32>
    %s4b1bt2n = stablehlo.subtract %s4b1bt2, %s4b1bt2s : tensor<512xf32>
    %Wdl = stablehlo.constant dense<0.1> : tensor<512x10xf32>
    %Wds = stablehlo.multiply %dWd, %Wdl : tensor<512x10xf32>
    %Wdn = stablehlo.subtract %Wd, %Wds : tensor<512x10xf32>
    %bdl = stablehlo.constant dense<0.1> : tensor<10xf32>
    %bds = stablehlo.multiply %dbd, %bdl : tensor<10xf32>
    %bdn = stablehlo.subtract %bd, %bds : tensor<10xf32>
    return %sWn, %sbn, %sgn, %sbtn, %s1b0W1n, %s1b0b1n, %s1b0g1n, %s1b0bt1n, %s1b0W2n, %s1b0b2n, %s1b0g2n, %s1b0bt2n, %s1b1W1n, %s1b1b1n, %s1b1g1n, %s1b1bt1n, %s1b1W2n, %s1b1b2n, %s1b1g2n, %s1b1bt2n, %s1b2W1n, %s1b2b1n, %s1b2g1n, %s1b2bt1n, %s1b2W2n, %s1b2b2n, %s1b2g2n, %s1b2bt2n, %d2W1n, %d2b1n, %d2g1n, %d2bt1n, %d2W2n, %d2b2n, %d2g2n, %d2bt2n, %d2Wpn, %d2bpn, %d2gpn, %d2btpn, %s2b0W1n, %s2b0b1n, %s2b0g1n, %s2b0bt1n, %s2b0W2n, %s2b0b2n, %s2b0g2n, %s2b0bt2n, %s2b1W1n, %s2b1b1n, %s2b1g1n, %s2b1bt1n, %s2b1W2n, %s2b1b2n, %s2b1g2n, %s2b1bt2n, %s2b2W1n, %s2b2b1n, %s2b2g1n, %s2b2bt1n, %s2b2W2n, %s2b2b2n, %s2b2g2n, %s2b2bt2n, %d3W1n, %d3b1n, %d3g1n, %d3bt1n, %d3W2n, %d3b2n, %d3g2n, %d3bt2n, %d3Wpn, %d3bpn, %d3gpn, %d3btpn, %s3b0W1n, %s3b0b1n, %s3b0g1n, %s3b0bt1n, %s3b0W2n, %s3b0b2n, %s3b0g2n, %s3b0bt2n, %s3b1W1n, %s3b1b1n, %s3b1g1n, %s3b1bt1n, %s3b1W2n, %s3b1b2n, %s3b1g2n, %s3b1bt2n, %s3b2W1n, %s3b2b1n, %s3b2g1n, %s3b2bt1n, %s3b2W2n, %s3b2b2n, %s3b2g2n, %s3b2bt2n, %s3b3W1n, %s3b3b1n, %s3b3g1n, %s3b3bt1n, %s3b3W2n, %s3b3b2n, %s3b3g2n, %s3b3bt2n, %s3b4W1n, %s3b4b1n, %s3b4g1n, %s3b4bt1n, %s3b4W2n, %s3b4b2n, %s3b4g2n, %s3b4bt2n, %d4W1n, %d4b1n, %d4g1n, %d4bt1n, %d4W2n, %d4b2n, %d4g2n, %d4bt2n, %d4Wpn, %d4bpn, %d4gpn, %d4btpn, %s4b0W1n, %s4b0b1n, %s4b0g1n, %s4b0bt1n, %s4b0W2n, %s4b0b2n, %s4b0g2n, %s4b0bt2n, %s4b1W1n, %s4b1b1n, %s4b1g1n, %s4b1bt1n, %s4b1W2n, %s4b1b2n, %s4b1g2n, %s4b1bt2n, %Wdn, %bdn : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}
