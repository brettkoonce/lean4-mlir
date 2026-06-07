module @m {
  func.func @mobilenetv2_fwd(%x: tensor<128x3072xf32>, %sW: tensor<32x3x3x3xf32>, %sb: tensor<32xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %iraeW: tensor<64x32x1x1xf32>, %iraeb: tensor<64xf32>, %iraeg: tensor<64xf32>, %iraebt: tensor<64xf32>, %iradW: tensor<64x1x3x3xf32>, %iradb: tensor<64xf32>, %iradg: tensor<64xf32>, %iradbt: tensor<64xf32>, %irapW: tensor<32x64x1x1xf32>, %irapb: tensor<32xf32>, %irapg: tensor<32xf32>, %irapbt: tensor<32xf32>, %irbeW: tensor<64x32x1x1xf32>, %irbeb: tensor<64xf32>, %irbeg: tensor<64xf32>, %irbebt: tensor<64xf32>, %irbdW: tensor<64x1x3x3xf32>, %irbdb: tensor<64xf32>, %irbdg: tensor<64xf32>, %irbdbt: tensor<64xf32>, %irbpW: tensor<64x64x1x1xf32>, %irbpb: tensor<64xf32>, %irbpg: tensor<64xf32>, %irbpbt: tensor<64xf32>, %hW: tensor<128x64x1x1xf32>, %hb: tensor<128xf32>, %hg: tensor<128xf32>, %hbt: tensor<128xf32>, %Wd: tensor<128x10xf32>, %bd: tensor<10xf32>) -> tensor<128x10xf32> {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %stcc = stablehlo.convolution(%xr, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x16x16xf32>
    %stcbb = stablehlo.broadcast_in_dim %sb, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16x16xf32>
    %stc = stablehlo.add %stcc, %stcbb : tensor<128x32x16x16xf32>
    %stnnf = stablehlo.constant dense<256.0> : tensor<128x32x16x16xf32>
    %stnep = stablehlo.constant dense<1.0e-5> : tensor<128x32x16x16xf32>
    %stnsmr = stablehlo.reduce(%stc init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32xf32>
    %stnsm = stablehlo.broadcast_in_dim %stnsmr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x16x16xf32>
    %stnmu = stablehlo.divide %stnsm, %stnnf : tensor<128x32x16x16xf32>
    %stnxc = stablehlo.subtract %stc, %stnmu : tensor<128x32x16x16xf32>
    %stnsq = stablehlo.multiply %stnxc, %stnxc : tensor<128x32x16x16xf32>
    %stnvsr = stablehlo.reduce(%stnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32xf32>
    %stnvs = stablehlo.broadcast_in_dim %stnvsr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x16x16xf32>
    %stnvr = stablehlo.divide %stnvs, %stnnf : tensor<128x32x16x16xf32>
    %stnve = stablehlo.add %stnvr, %stnep : tensor<128x32x16x16xf32>
    %stnistd = stablehlo.rsqrt %stnve : tensor<128x32x16x16xf32>
    %stnxh = stablehlo.multiply %stnxc, %stnistd : tensor<128x32x16x16xf32>
    %stngb = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16x16xf32>
    %stnbtb = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16x16xf32>
    %stngx = stablehlo.multiply %stnxh, %stngb : tensor<128x32x16x16xf32>
    %stn = stablehlo.add %stngx, %stnbtb : tensor<128x32x16x16xf32>
    %strz = stablehlo.constant dense<0.0> : tensor<128x32x16x16xf32>
    %strsix = stablehlo.constant dense<6.0> : tensor<128x32x16x16xf32>
    %strmx = stablehlo.maximum %stn, %strz : tensor<128x32x16x16xf32>
    %str = stablehlo.minimum %strmx, %strsix : tensor<128x32x16x16xf32>
    %stpni = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %stp = "stablehlo.reduce_window"(%str, %stpni) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %iraec = stablehlo.convolution(%stp, %iraeW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<64x32x1x1xf32>) -> tensor<128x64x8x8xf32>
    %iraebb = stablehlo.broadcast_in_dim %iraeb, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %irae = stablehlo.add %iraec, %iraebb : tensor<128x64x8x8xf32>
    %iraennf = stablehlo.constant dense<64.0> : tensor<128x64x8x8xf32>
    %iraenep = stablehlo.constant dense<1.0e-5> : tensor<128x64x8x8xf32>
    %iraensmr = stablehlo.reduce(%irae init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %iraensm = stablehlo.broadcast_in_dim %iraensmr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %iraenmu = stablehlo.divide %iraensm, %iraennf : tensor<128x64x8x8xf32>
    %iraenxc = stablehlo.subtract %irae, %iraenmu : tensor<128x64x8x8xf32>
    %iraensq = stablehlo.multiply %iraenxc, %iraenxc : tensor<128x64x8x8xf32>
    %iraenvsr = stablehlo.reduce(%iraensq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %iraenvs = stablehlo.broadcast_in_dim %iraenvsr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %iraenvr = stablehlo.divide %iraenvs, %iraennf : tensor<128x64x8x8xf32>
    %iraenve = stablehlo.add %iraenvr, %iraenep : tensor<128x64x8x8xf32>
    %iraenistd = stablehlo.rsqrt %iraenve : tensor<128x64x8x8xf32>
    %iraenxh = stablehlo.multiply %iraenxc, %iraenistd : tensor<128x64x8x8xf32>
    %iraengb = stablehlo.broadcast_in_dim %iraeg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %iraenbtb = stablehlo.broadcast_in_dim %iraebt, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %iraengx = stablehlo.multiply %iraenxh, %iraengb : tensor<128x64x8x8xf32>
    %iraen = stablehlo.add %iraengx, %iraenbtb : tensor<128x64x8x8xf32>
    %iraerz = stablehlo.constant dense<0.0> : tensor<128x64x8x8xf32>
    %iraersix = stablehlo.constant dense<6.0> : tensor<128x64x8x8xf32>
    %iraermx = stablehlo.maximum %iraen, %iraerz : tensor<128x64x8x8xf32>
    %iraer = stablehlo.minimum %iraermx, %iraersix : tensor<128x64x8x8xf32>
    %iradc = stablehlo.convolution(%iraer, %iradW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<128x64x8x8xf32>, tensor<64x1x3x3xf32>) -> tensor<128x64x8x8xf32>
    %iradbb = stablehlo.broadcast_in_dim %iradb, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %irad = stablehlo.add %iradc, %iradbb : tensor<128x64x8x8xf32>
    %iradnnf = stablehlo.constant dense<64.0> : tensor<128x64x8x8xf32>
    %iradnep = stablehlo.constant dense<1.0e-5> : tensor<128x64x8x8xf32>
    %iradnsmr = stablehlo.reduce(%irad init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %iradnsm = stablehlo.broadcast_in_dim %iradnsmr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %iradnmu = stablehlo.divide %iradnsm, %iradnnf : tensor<128x64x8x8xf32>
    %iradnxc = stablehlo.subtract %irad, %iradnmu : tensor<128x64x8x8xf32>
    %iradnsq = stablehlo.multiply %iradnxc, %iradnxc : tensor<128x64x8x8xf32>
    %iradnvsr = stablehlo.reduce(%iradnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %iradnvs = stablehlo.broadcast_in_dim %iradnvsr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %iradnvr = stablehlo.divide %iradnvs, %iradnnf : tensor<128x64x8x8xf32>
    %iradnve = stablehlo.add %iradnvr, %iradnep : tensor<128x64x8x8xf32>
    %iradnistd = stablehlo.rsqrt %iradnve : tensor<128x64x8x8xf32>
    %iradnxh = stablehlo.multiply %iradnxc, %iradnistd : tensor<128x64x8x8xf32>
    %iradngb = stablehlo.broadcast_in_dim %iradg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %iradnbtb = stablehlo.broadcast_in_dim %iradbt, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %iradngx = stablehlo.multiply %iradnxh, %iradngb : tensor<128x64x8x8xf32>
    %iradn = stablehlo.add %iradngx, %iradnbtb : tensor<128x64x8x8xf32>
    %iradrz = stablehlo.constant dense<0.0> : tensor<128x64x8x8xf32>
    %iradrsix = stablehlo.constant dense<6.0> : tensor<128x64x8x8xf32>
    %iradrmx = stablehlo.maximum %iradn, %iradrz : tensor<128x64x8x8xf32>
    %iradr = stablehlo.minimum %iradrmx, %iradrsix : tensor<128x64x8x8xf32>
    %irapc = stablehlo.convolution(%iradr, %irapW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x8x8xf32>, tensor<32x64x1x1xf32>) -> tensor<128x32x8x8xf32>
    %irapbb = stablehlo.broadcast_in_dim %irapb, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %irap = stablehlo.add %irapc, %irapbb : tensor<128x32x8x8xf32>
    %irapnnf = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %irapnep = stablehlo.constant dense<1.0e-5> : tensor<128x32x8x8xf32>
    %irapnsmr = stablehlo.reduce(%irap init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %irapnsm = stablehlo.broadcast_in_dim %irapnsmr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %irapnmu = stablehlo.divide %irapnsm, %irapnnf : tensor<128x32x8x8xf32>
    %irapnxc = stablehlo.subtract %irap, %irapnmu : tensor<128x32x8x8xf32>
    %irapnsq = stablehlo.multiply %irapnxc, %irapnxc : tensor<128x32x8x8xf32>
    %irapnvsr = stablehlo.reduce(%irapnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %irapnvs = stablehlo.broadcast_in_dim %irapnvsr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %irapnvr = stablehlo.divide %irapnvs, %irapnnf : tensor<128x32x8x8xf32>
    %irapnve = stablehlo.add %irapnvr, %irapnep : tensor<128x32x8x8xf32>
    %irapnistd = stablehlo.rsqrt %irapnve : tensor<128x32x8x8xf32>
    %irapnxh = stablehlo.multiply %irapnxc, %irapnistd : tensor<128x32x8x8xf32>
    %irapngb = stablehlo.broadcast_in_dim %irapg, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %irapnbtb = stablehlo.broadcast_in_dim %irapbt, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %irapngx = stablehlo.multiply %irapnxh, %irapngb : tensor<128x32x8x8xf32>
    %irapn = stablehlo.add %irapngx, %irapnbtb : tensor<128x32x8x8xf32>
    %irao = stablehlo.add %irapn, %stp : tensor<128x32x8x8xf32>
    %irbec = stablehlo.convolution(%irao, %irbeW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<64x32x1x1xf32>) -> tensor<128x64x8x8xf32>
    %irbebb = stablehlo.broadcast_in_dim %irbeb, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %irbe = stablehlo.add %irbec, %irbebb : tensor<128x64x8x8xf32>
    %irbennf = stablehlo.constant dense<64.0> : tensor<128x64x8x8xf32>
    %irbenep = stablehlo.constant dense<1.0e-5> : tensor<128x64x8x8xf32>
    %irbensmr = stablehlo.reduce(%irbe init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %irbensm = stablehlo.broadcast_in_dim %irbensmr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %irbenmu = stablehlo.divide %irbensm, %irbennf : tensor<128x64x8x8xf32>
    %irbenxc = stablehlo.subtract %irbe, %irbenmu : tensor<128x64x8x8xf32>
    %irbensq = stablehlo.multiply %irbenxc, %irbenxc : tensor<128x64x8x8xf32>
    %irbenvsr = stablehlo.reduce(%irbensq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %irbenvs = stablehlo.broadcast_in_dim %irbenvsr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %irbenvr = stablehlo.divide %irbenvs, %irbennf : tensor<128x64x8x8xf32>
    %irbenve = stablehlo.add %irbenvr, %irbenep : tensor<128x64x8x8xf32>
    %irbenistd = stablehlo.rsqrt %irbenve : tensor<128x64x8x8xf32>
    %irbenxh = stablehlo.multiply %irbenxc, %irbenistd : tensor<128x64x8x8xf32>
    %irbengb = stablehlo.broadcast_in_dim %irbeg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %irbenbtb = stablehlo.broadcast_in_dim %irbebt, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %irbengx = stablehlo.multiply %irbenxh, %irbengb : tensor<128x64x8x8xf32>
    %irben = stablehlo.add %irbengx, %irbenbtb : tensor<128x64x8x8xf32>
    %irberz = stablehlo.constant dense<0.0> : tensor<128x64x8x8xf32>
    %irbersix = stablehlo.constant dense<6.0> : tensor<128x64x8x8xf32>
    %irbermx = stablehlo.maximum %irben, %irberz : tensor<128x64x8x8xf32>
    %irber = stablehlo.minimum %irbermx, %irbersix : tensor<128x64x8x8xf32>
    %irbdc = stablehlo.convolution(%irber, %irbdW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<128x64x8x8xf32>, tensor<64x1x3x3xf32>) -> tensor<128x64x8x8xf32>
    %irbdbb = stablehlo.broadcast_in_dim %irbdb, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %irbd = stablehlo.add %irbdc, %irbdbb : tensor<128x64x8x8xf32>
    %irbdnnf = stablehlo.constant dense<64.0> : tensor<128x64x8x8xf32>
    %irbdnep = stablehlo.constant dense<1.0e-5> : tensor<128x64x8x8xf32>
    %irbdnsmr = stablehlo.reduce(%irbd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %irbdnsm = stablehlo.broadcast_in_dim %irbdnsmr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %irbdnmu = stablehlo.divide %irbdnsm, %irbdnnf : tensor<128x64x8x8xf32>
    %irbdnxc = stablehlo.subtract %irbd, %irbdnmu : tensor<128x64x8x8xf32>
    %irbdnsq = stablehlo.multiply %irbdnxc, %irbdnxc : tensor<128x64x8x8xf32>
    %irbdnvsr = stablehlo.reduce(%irbdnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %irbdnvs = stablehlo.broadcast_in_dim %irbdnvsr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %irbdnvr = stablehlo.divide %irbdnvs, %irbdnnf : tensor<128x64x8x8xf32>
    %irbdnve = stablehlo.add %irbdnvr, %irbdnep : tensor<128x64x8x8xf32>
    %irbdnistd = stablehlo.rsqrt %irbdnve : tensor<128x64x8x8xf32>
    %irbdnxh = stablehlo.multiply %irbdnxc, %irbdnistd : tensor<128x64x8x8xf32>
    %irbdngb = stablehlo.broadcast_in_dim %irbdg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %irbdnbtb = stablehlo.broadcast_in_dim %irbdbt, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %irbdngx = stablehlo.multiply %irbdnxh, %irbdngb : tensor<128x64x8x8xf32>
    %irbdn = stablehlo.add %irbdngx, %irbdnbtb : tensor<128x64x8x8xf32>
    %irbdrz = stablehlo.constant dense<0.0> : tensor<128x64x8x8xf32>
    %irbdrsix = stablehlo.constant dense<6.0> : tensor<128x64x8x8xf32>
    %irbdrmx = stablehlo.maximum %irbdn, %irbdrz : tensor<128x64x8x8xf32>
    %irbdr = stablehlo.minimum %irbdrmx, %irbdrsix : tensor<128x64x8x8xf32>
    %irbpc = stablehlo.convolution(%irbdr, %irbpW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x8x8xf32>, tensor<64x64x1x1xf32>) -> tensor<128x64x8x8xf32>
    %irbpbb = stablehlo.broadcast_in_dim %irbpb, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %irbp = stablehlo.add %irbpc, %irbpbb : tensor<128x64x8x8xf32>
    %irbpnnf = stablehlo.constant dense<64.0> : tensor<128x64x8x8xf32>
    %irbpnep = stablehlo.constant dense<1.0e-5> : tensor<128x64x8x8xf32>
    %irbpnsmr = stablehlo.reduce(%irbp init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %irbpnsm = stablehlo.broadcast_in_dim %irbpnsmr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %irbpnmu = stablehlo.divide %irbpnsm, %irbpnnf : tensor<128x64x8x8xf32>
    %irbpnxc = stablehlo.subtract %irbp, %irbpnmu : tensor<128x64x8x8xf32>
    %irbpnsq = stablehlo.multiply %irbpnxc, %irbpnxc : tensor<128x64x8x8xf32>
    %irbpnvsr = stablehlo.reduce(%irbpnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %irbpnvs = stablehlo.broadcast_in_dim %irbpnvsr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %irbpnvr = stablehlo.divide %irbpnvs, %irbpnnf : tensor<128x64x8x8xf32>
    %irbpnve = stablehlo.add %irbpnvr, %irbpnep : tensor<128x64x8x8xf32>
    %irbpnistd = stablehlo.rsqrt %irbpnve : tensor<128x64x8x8xf32>
    %irbpnxh = stablehlo.multiply %irbpnxc, %irbpnistd : tensor<128x64x8x8xf32>
    %irbpngb = stablehlo.broadcast_in_dim %irbpg, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %irbpnbtb = stablehlo.broadcast_in_dim %irbpbt, dims = [1] : (tensor<64xf32>) -> tensor<128x64x8x8xf32>
    %irbpngx = stablehlo.multiply %irbpnxh, %irbpngb : tensor<128x64x8x8xf32>
    %irbpn = stablehlo.add %irbpngx, %irbpnbtb : tensor<128x64x8x8xf32>
    %hc = stablehlo.convolution(%irbpn, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x8x8xf32>, tensor<128x64x1x1xf32>) -> tensor<128x128x8x8xf32>
    %hbb = stablehlo.broadcast_in_dim %hb, dims = [1] : (tensor<128xf32>) -> tensor<128x128x8x8xf32>
    %h = stablehlo.add %hc, %hbb : tensor<128x128x8x8xf32>
    %hnnf = stablehlo.constant dense<64.0> : tensor<128x128x8x8xf32>
    %hnep = stablehlo.constant dense<1.0e-5> : tensor<128x128x8x8xf32>
    %hnsmr = stablehlo.reduce(%h init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x8x8xf32>, tensor<f32>) -> tensor<128x128xf32>
    %hnsm = stablehlo.broadcast_in_dim %hnsmr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x8x8xf32>
    %hnmu = stablehlo.divide %hnsm, %hnnf : tensor<128x128x8x8xf32>
    %hnxc = stablehlo.subtract %h, %hnmu : tensor<128x128x8x8xf32>
    %hnsq = stablehlo.multiply %hnxc, %hnxc : tensor<128x128x8x8xf32>
    %hnvsr = stablehlo.reduce(%hnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x8x8xf32>, tensor<f32>) -> tensor<128x128xf32>
    %hnvs = stablehlo.broadcast_in_dim %hnvsr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x8x8xf32>
    %hnvr = stablehlo.divide %hnvs, %hnnf : tensor<128x128x8x8xf32>
    %hnve = stablehlo.add %hnvr, %hnep : tensor<128x128x8x8xf32>
    %hnistd = stablehlo.rsqrt %hnve : tensor<128x128x8x8xf32>
    %hnxh = stablehlo.multiply %hnxc, %hnistd : tensor<128x128x8x8xf32>
    %hngb = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<128xf32>) -> tensor<128x128x8x8xf32>
    %hnbtb = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<128xf32>) -> tensor<128x128x8x8xf32>
    %hngx = stablehlo.multiply %hnxh, %hngb : tensor<128x128x8x8xf32>
    %hn = stablehlo.add %hngx, %hnbtb : tensor<128x128x8x8xf32>
    %hrz = stablehlo.constant dense<0.0> : tensor<128x128x8x8xf32>
    %hrsix = stablehlo.constant dense<6.0> : tensor<128x128x8x8xf32>
    %hrmx = stablehlo.maximum %hn, %hrz : tensor<128x128x8x8xf32>
    %hr = stablehlo.minimum %hrmx, %hrsix : tensor<128x128x8x8xf32>
    %outgs = stablehlo.reduce(%hr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x8x8xf32>, tensor<f32>) -> tensor<128x128xf32>
    %outgnf = stablehlo.constant dense<64.0> : tensor<128x128xf32>
    %outg = stablehlo.divide %outgs, %outgnf : tensor<128x128xf32>
    %outdd = stablehlo.dot_general %outg, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x10xf32>) -> tensor<128x10xf32>
    %outdb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %out = stablehlo.add %outdd, %outdb : tensor<128x10xf32>
    return %out : tensor<128x10xf32>
  }
}
