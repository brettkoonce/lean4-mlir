module @m {
  func.func @resnet34_fwd_eval(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sb: tensor<64xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0b1: tensor<64xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0b2: tensor<64xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1b1: tensor<64xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1b2: tensor<64xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2b1: tensor<64xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2b2: tensor<64xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2b1: tensor<128xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2b2: tensor<128xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x3x3xf32>, %d2bp: tensor<128xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0b1: tensor<128xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0b2: tensor<128xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1b1: tensor<128xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1b2: tensor<128xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2b1: tensor<128xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2b2: tensor<128xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3b1: tensor<256xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3b2: tensor<256xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x3x3xf32>, %d3bp: tensor<256xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0b1: tensor<256xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0b2: tensor<256xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1b1: tensor<256xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1b2: tensor<256xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2b1: tensor<256xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2b2: tensor<256xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3b1: tensor<256xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3b2: tensor<256xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4b1: tensor<256xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4b2: tensor<256xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4b1: tensor<512xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4b2: tensor<512xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x3x3xf32>, %d4bp: tensor<512xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0b1: tensor<512xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0b2: tensor<512xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1b1: tensor<512xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1b2: tensor<512xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %stnmu: tensor<64xf32>, %stnvar: tensor<64xf32>, %s1b0n1mu: tensor<64xf32>, %s1b0n1var: tensor<64xf32>, %s1b0n2mu: tensor<64xf32>, %s1b0n2var: tensor<64xf32>, %s1b1n1mu: tensor<64xf32>, %s1b1n1var: tensor<64xf32>, %s1b1n2mu: tensor<64xf32>, %s1b1n2var: tensor<64xf32>, %s1b2n1mu: tensor<64xf32>, %s1b2n1var: tensor<64xf32>, %s1b2n2mu: tensor<64xf32>, %s1b2n2var: tensor<64xf32>, %d2n1mu: tensor<128xf32>, %d2n1var: tensor<128xf32>, %d2n2mu: tensor<128xf32>, %d2n2var: tensor<128xf32>, %d2npmu: tensor<128xf32>, %d2npvar: tensor<128xf32>, %s2b0n1mu: tensor<128xf32>, %s2b0n1var: tensor<128xf32>, %s2b0n2mu: tensor<128xf32>, %s2b0n2var: tensor<128xf32>, %s2b1n1mu: tensor<128xf32>, %s2b1n1var: tensor<128xf32>, %s2b1n2mu: tensor<128xf32>, %s2b1n2var: tensor<128xf32>, %s2b2n1mu: tensor<128xf32>, %s2b2n1var: tensor<128xf32>, %s2b2n2mu: tensor<128xf32>, %s2b2n2var: tensor<128xf32>, %d3n1mu: tensor<256xf32>, %d3n1var: tensor<256xf32>, %d3n2mu: tensor<256xf32>, %d3n2var: tensor<256xf32>, %d3npmu: tensor<256xf32>, %d3npvar: tensor<256xf32>, %s3b0n1mu: tensor<256xf32>, %s3b0n1var: tensor<256xf32>, %s3b0n2mu: tensor<256xf32>, %s3b0n2var: tensor<256xf32>, %s3b1n1mu: tensor<256xf32>, %s3b1n1var: tensor<256xf32>, %s3b1n2mu: tensor<256xf32>, %s3b1n2var: tensor<256xf32>, %s3b2n1mu: tensor<256xf32>, %s3b2n1var: tensor<256xf32>, %s3b2n2mu: tensor<256xf32>, %s3b2n2var: tensor<256xf32>, %s3b3n1mu: tensor<256xf32>, %s3b3n1var: tensor<256xf32>, %s3b3n2mu: tensor<256xf32>, %s3b3n2var: tensor<256xf32>, %s3b4n1mu: tensor<256xf32>, %s3b4n1var: tensor<256xf32>, %s3b4n2mu: tensor<256xf32>, %s3b4n2var: tensor<256xf32>, %d4n1mu: tensor<512xf32>, %d4n1var: tensor<512xf32>, %d4n2mu: tensor<512xf32>, %d4n2var: tensor<512xf32>, %d4npmu: tensor<512xf32>, %d4npvar: tensor<512xf32>, %s4b0n1mu: tensor<512xf32>, %s4b0n1var: tensor<512xf32>, %s4b0n2mu: tensor<512xf32>, %s4b0n2var: tensor<512xf32>, %s4b1n1mu: tensor<512xf32>, %s4b1n1var: tensor<512xf32>, %s4b1n2mu: tensor<512xf32>, %s4b1n2var: tensor<512xf32>) -> tensor<32x10xf32> {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %stcc = stablehlo.convolution(%xr, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<32x64x112x112xf32>
    %stcbb = stablehlo.broadcast_in_dim %sb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %stc = stablehlo.add %stcc, %stcbb : tensor<32x64x112x112xf32>
    %stnmub = stablehlo.broadcast_in_dim %stnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %stnxc = stablehlo.subtract %stc, %stnmub : tensor<32x64x112x112xf32>
    %stnvb = stablehlo.broadcast_in_dim %stnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %stnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %stnve = stablehlo.add %stnvb, %stnep : tensor<32x64x112x112xf32>
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
    %s1b0n1mub = stablehlo.broadcast_in_dim %s1b0n1mu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n1xc = stablehlo.subtract %s1b0c1, %s1b0n1mub : tensor<32x64x56x56xf32>
    %s1b0n1vb = stablehlo.broadcast_in_dim %s1b0n1var, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b0n1ve = stablehlo.add %s1b0n1vb, %s1b0n1ep : tensor<32x64x56x56xf32>
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
    %s1b0n2mub = stablehlo.broadcast_in_dim %s1b0n2mu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n2xc = stablehlo.subtract %s1b0c2, %s1b0n2mub : tensor<32x64x56x56xf32>
    %s1b0n2vb = stablehlo.broadcast_in_dim %s1b0n2var, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b0n2ve = stablehlo.add %s1b0n2vb, %s1b0n2ep : tensor<32x64x56x56xf32>
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
    %s1b1n1mub = stablehlo.broadcast_in_dim %s1b1n1mu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n1xc = stablehlo.subtract %s1b1c1, %s1b1n1mub : tensor<32x64x56x56xf32>
    %s1b1n1vb = stablehlo.broadcast_in_dim %s1b1n1var, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b1n1ve = stablehlo.add %s1b1n1vb, %s1b1n1ep : tensor<32x64x56x56xf32>
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
    %s1b1n2mub = stablehlo.broadcast_in_dim %s1b1n2mu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n2xc = stablehlo.subtract %s1b1c2, %s1b1n2mub : tensor<32x64x56x56xf32>
    %s1b1n2vb = stablehlo.broadcast_in_dim %s1b1n2var, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b1n2ve = stablehlo.add %s1b1n2vb, %s1b1n2ep : tensor<32x64x56x56xf32>
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
    %s1b2n1mub = stablehlo.broadcast_in_dim %s1b2n1mu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n1xc = stablehlo.subtract %s1b2c1, %s1b2n1mub : tensor<32x64x56x56xf32>
    %s1b2n1vb = stablehlo.broadcast_in_dim %s1b2n1var, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b2n1ve = stablehlo.add %s1b2n1vb, %s1b2n1ep : tensor<32x64x56x56xf32>
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
    %s1b2n2mub = stablehlo.broadcast_in_dim %s1b2n2mu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n2xc = stablehlo.subtract %s1b2c2, %s1b2n2mub : tensor<32x64x56x56xf32>
    %s1b2n2vb = stablehlo.broadcast_in_dim %s1b2n2var, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b2n2ve = stablehlo.add %s1b2n2vb, %s1b2n2ep : tensor<32x64x56x56xf32>
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
    %d2n1mub = stablehlo.broadcast_in_dim %d2n1mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n1xc = stablehlo.subtract %d2c1, %d2n1mub : tensor<32x128x28x28xf32>
    %d2n1vb = stablehlo.broadcast_in_dim %d2n1var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %d2n1ve = stablehlo.add %d2n1vb, %d2n1ep : tensor<32x128x28x28xf32>
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
    %d2n2mub = stablehlo.broadcast_in_dim %d2n2mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n2xc = stablehlo.subtract %d2c2, %d2n2mub : tensor<32x128x28x28xf32>
    %d2n2vb = stablehlo.broadcast_in_dim %d2n2var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %d2n2ve = stablehlo.add %d2n2vb, %d2n2ep : tensor<32x128x28x28xf32>
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
    %d2npmub = stablehlo.broadcast_in_dim %d2npmu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2npxc = stablehlo.subtract %d2cp, %d2npmub : tensor<32x128x28x28xf32>
    %d2npvb = stablehlo.broadcast_in_dim %d2npvar, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2npep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %d2npve = stablehlo.add %d2npvb, %d2npep : tensor<32x128x28x28xf32>
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
    %s2b0n1mub = stablehlo.broadcast_in_dim %s2b0n1mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n1xc = stablehlo.subtract %s2b0c1, %s2b0n1mub : tensor<32x128x28x28xf32>
    %s2b0n1vb = stablehlo.broadcast_in_dim %s2b0n1var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b0n1ve = stablehlo.add %s2b0n1vb, %s2b0n1ep : tensor<32x128x28x28xf32>
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
    %s2b0n2mub = stablehlo.broadcast_in_dim %s2b0n2mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n2xc = stablehlo.subtract %s2b0c2, %s2b0n2mub : tensor<32x128x28x28xf32>
    %s2b0n2vb = stablehlo.broadcast_in_dim %s2b0n2var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b0n2ve = stablehlo.add %s2b0n2vb, %s2b0n2ep : tensor<32x128x28x28xf32>
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
    %s2b1n1mub = stablehlo.broadcast_in_dim %s2b1n1mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n1xc = stablehlo.subtract %s2b1c1, %s2b1n1mub : tensor<32x128x28x28xf32>
    %s2b1n1vb = stablehlo.broadcast_in_dim %s2b1n1var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b1n1ve = stablehlo.add %s2b1n1vb, %s2b1n1ep : tensor<32x128x28x28xf32>
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
    %s2b1n2mub = stablehlo.broadcast_in_dim %s2b1n2mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n2xc = stablehlo.subtract %s2b1c2, %s2b1n2mub : tensor<32x128x28x28xf32>
    %s2b1n2vb = stablehlo.broadcast_in_dim %s2b1n2var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b1n2ve = stablehlo.add %s2b1n2vb, %s2b1n2ep : tensor<32x128x28x28xf32>
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
    %s2b2n1mub = stablehlo.broadcast_in_dim %s2b2n1mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n1xc = stablehlo.subtract %s2b2c1, %s2b2n1mub : tensor<32x128x28x28xf32>
    %s2b2n1vb = stablehlo.broadcast_in_dim %s2b2n1var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b2n1ve = stablehlo.add %s2b2n1vb, %s2b2n1ep : tensor<32x128x28x28xf32>
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
    %s2b2n2mub = stablehlo.broadcast_in_dim %s2b2n2mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n2xc = stablehlo.subtract %s2b2c2, %s2b2n2mub : tensor<32x128x28x28xf32>
    %s2b2n2vb = stablehlo.broadcast_in_dim %s2b2n2var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b2n2ve = stablehlo.add %s2b2n2vb, %s2b2n2ep : tensor<32x128x28x28xf32>
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
    %d3n1mub = stablehlo.broadcast_in_dim %d3n1mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n1xc = stablehlo.subtract %d3c1, %d3n1mub : tensor<32x256x14x14xf32>
    %d3n1vb = stablehlo.broadcast_in_dim %d3n1var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %d3n1ve = stablehlo.add %d3n1vb, %d3n1ep : tensor<32x256x14x14xf32>
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
    %d3n2mub = stablehlo.broadcast_in_dim %d3n2mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n2xc = stablehlo.subtract %d3c2, %d3n2mub : tensor<32x256x14x14xf32>
    %d3n2vb = stablehlo.broadcast_in_dim %d3n2var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %d3n2ve = stablehlo.add %d3n2vb, %d3n2ep : tensor<32x256x14x14xf32>
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
    %d3npmub = stablehlo.broadcast_in_dim %d3npmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3npxc = stablehlo.subtract %d3cp, %d3npmub : tensor<32x256x14x14xf32>
    %d3npvb = stablehlo.broadcast_in_dim %d3npvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3npep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %d3npve = stablehlo.add %d3npvb, %d3npep : tensor<32x256x14x14xf32>
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
    %s3b0n1mub = stablehlo.broadcast_in_dim %s3b0n1mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n1xc = stablehlo.subtract %s3b0c1, %s3b0n1mub : tensor<32x256x14x14xf32>
    %s3b0n1vb = stablehlo.broadcast_in_dim %s3b0n1var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b0n1ve = stablehlo.add %s3b0n1vb, %s3b0n1ep : tensor<32x256x14x14xf32>
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
    %s3b0n2mub = stablehlo.broadcast_in_dim %s3b0n2mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n2xc = stablehlo.subtract %s3b0c2, %s3b0n2mub : tensor<32x256x14x14xf32>
    %s3b0n2vb = stablehlo.broadcast_in_dim %s3b0n2var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b0n2ve = stablehlo.add %s3b0n2vb, %s3b0n2ep : tensor<32x256x14x14xf32>
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
    %s3b1n1mub = stablehlo.broadcast_in_dim %s3b1n1mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n1xc = stablehlo.subtract %s3b1c1, %s3b1n1mub : tensor<32x256x14x14xf32>
    %s3b1n1vb = stablehlo.broadcast_in_dim %s3b1n1var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b1n1ve = stablehlo.add %s3b1n1vb, %s3b1n1ep : tensor<32x256x14x14xf32>
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
    %s3b1n2mub = stablehlo.broadcast_in_dim %s3b1n2mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n2xc = stablehlo.subtract %s3b1c2, %s3b1n2mub : tensor<32x256x14x14xf32>
    %s3b1n2vb = stablehlo.broadcast_in_dim %s3b1n2var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b1n2ve = stablehlo.add %s3b1n2vb, %s3b1n2ep : tensor<32x256x14x14xf32>
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
    %s3b2n1mub = stablehlo.broadcast_in_dim %s3b2n1mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n1xc = stablehlo.subtract %s3b2c1, %s3b2n1mub : tensor<32x256x14x14xf32>
    %s3b2n1vb = stablehlo.broadcast_in_dim %s3b2n1var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b2n1ve = stablehlo.add %s3b2n1vb, %s3b2n1ep : tensor<32x256x14x14xf32>
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
    %s3b2n2mub = stablehlo.broadcast_in_dim %s3b2n2mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n2xc = stablehlo.subtract %s3b2c2, %s3b2n2mub : tensor<32x256x14x14xf32>
    %s3b2n2vb = stablehlo.broadcast_in_dim %s3b2n2var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b2n2ve = stablehlo.add %s3b2n2vb, %s3b2n2ep : tensor<32x256x14x14xf32>
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
    %s3b3n1mub = stablehlo.broadcast_in_dim %s3b3n1mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n1xc = stablehlo.subtract %s3b3c1, %s3b3n1mub : tensor<32x256x14x14xf32>
    %s3b3n1vb = stablehlo.broadcast_in_dim %s3b3n1var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b3n1ve = stablehlo.add %s3b3n1vb, %s3b3n1ep : tensor<32x256x14x14xf32>
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
    %s3b3n2mub = stablehlo.broadcast_in_dim %s3b3n2mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n2xc = stablehlo.subtract %s3b3c2, %s3b3n2mub : tensor<32x256x14x14xf32>
    %s3b3n2vb = stablehlo.broadcast_in_dim %s3b3n2var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b3n2ve = stablehlo.add %s3b3n2vb, %s3b3n2ep : tensor<32x256x14x14xf32>
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
    %s3b4n1mub = stablehlo.broadcast_in_dim %s3b4n1mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n1xc = stablehlo.subtract %s3b4c1, %s3b4n1mub : tensor<32x256x14x14xf32>
    %s3b4n1vb = stablehlo.broadcast_in_dim %s3b4n1var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b4n1ve = stablehlo.add %s3b4n1vb, %s3b4n1ep : tensor<32x256x14x14xf32>
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
    %s3b4n2mub = stablehlo.broadcast_in_dim %s3b4n2mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n2xc = stablehlo.subtract %s3b4c2, %s3b4n2mub : tensor<32x256x14x14xf32>
    %s3b4n2vb = stablehlo.broadcast_in_dim %s3b4n2var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b4n2ve = stablehlo.add %s3b4n2vb, %s3b4n2ep : tensor<32x256x14x14xf32>
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
    %d4n1mub = stablehlo.broadcast_in_dim %d4n1mu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n1xc = stablehlo.subtract %d4c1, %d4n1mub : tensor<32x512x7x7xf32>
    %d4n1vb = stablehlo.broadcast_in_dim %d4n1var, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %d4n1ve = stablehlo.add %d4n1vb, %d4n1ep : tensor<32x512x7x7xf32>
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
    %d4n2mub = stablehlo.broadcast_in_dim %d4n2mu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n2xc = stablehlo.subtract %d4c2, %d4n2mub : tensor<32x512x7x7xf32>
    %d4n2vb = stablehlo.broadcast_in_dim %d4n2var, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %d4n2ve = stablehlo.add %d4n2vb, %d4n2ep : tensor<32x512x7x7xf32>
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
    %d4npmub = stablehlo.broadcast_in_dim %d4npmu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4npxc = stablehlo.subtract %d4cp, %d4npmub : tensor<32x512x7x7xf32>
    %d4npvb = stablehlo.broadcast_in_dim %d4npvar, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4npep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %d4npve = stablehlo.add %d4npvb, %d4npep : tensor<32x512x7x7xf32>
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
    %s4b0n1mub = stablehlo.broadcast_in_dim %s4b0n1mu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n1xc = stablehlo.subtract %s4b0c1, %s4b0n1mub : tensor<32x512x7x7xf32>
    %s4b0n1vb = stablehlo.broadcast_in_dim %s4b0n1var, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %s4b0n1ve = stablehlo.add %s4b0n1vb, %s4b0n1ep : tensor<32x512x7x7xf32>
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
    %s4b0n2mub = stablehlo.broadcast_in_dim %s4b0n2mu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n2xc = stablehlo.subtract %s4b0c2, %s4b0n2mub : tensor<32x512x7x7xf32>
    %s4b0n2vb = stablehlo.broadcast_in_dim %s4b0n2var, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %s4b0n2ve = stablehlo.add %s4b0n2vb, %s4b0n2ep : tensor<32x512x7x7xf32>
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
    %s4b1n1mub = stablehlo.broadcast_in_dim %s4b1n1mu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n1xc = stablehlo.subtract %s4b1c1, %s4b1n1mub : tensor<32x512x7x7xf32>
    %s4b1n1vb = stablehlo.broadcast_in_dim %s4b1n1var, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %s4b1n1ve = stablehlo.add %s4b1n1vb, %s4b1n1ep : tensor<32x512x7x7xf32>
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
    %s4b1n2mub = stablehlo.broadcast_in_dim %s4b1n2mu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n2xc = stablehlo.subtract %s4b1c2, %s4b1n2mub : tensor<32x512x7x7xf32>
    %s4b1n2vb = stablehlo.broadcast_in_dim %s4b1n2var, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %s4b1n2ve = stablehlo.add %s4b1n2vb, %s4b1n2ep : tensor<32x512x7x7xf32>
    %s4b1n2istd = stablehlo.rsqrt %s4b1n2ve : tensor<32x512x7x7xf32>
    %s4b1n2xh = stablehlo.multiply %s4b1n2xc, %s4b1n2istd : tensor<32x512x7x7xf32>
    %s4b1n2gb = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n2btb = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n2gx = stablehlo.multiply %s4b1n2xh, %s4b1n2gb : tensor<32x512x7x7xf32>
    %s4b1n2 = stablehlo.add %s4b1n2gx, %s4b1n2btb : tensor<32x512x7x7xf32>
    %s4b1a = stablehlo.add %s4b1n2, %s4b0o : tensor<32x512x7x7xf32>
    %s4b1oz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b1o = stablehlo.maximum %s4b1a, %s4b1oz : tensor<32x512x7x7xf32>
    %outgs = stablehlo.reduce(%s4b1o init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %outgnf = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %outg = stablehlo.divide %outgs, %outgnf : tensor<32x512xf32>
    %outdd = stablehlo.dot_general %outg, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<512x10xf32>) -> tensor<32x10xf32>
    %outdb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %out = stablehlo.add %outdd, %outdb : tensor<32x10xf32>
    return %out : tensor<32x10xf32>
  }
}
