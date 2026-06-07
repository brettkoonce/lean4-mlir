module @m {
  func.func @mobilenetv2_train_step(%x: tensor<128x3072xf32>, %sW: tensor<32x3x3x3xf32>, %sb: tensor<32xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %iraeW: tensor<64x32x1x1xf32>, %iraeb: tensor<64xf32>, %iraeg: tensor<64xf32>, %iraebt: tensor<64xf32>, %iradW: tensor<64x1x3x3xf32>, %iradb: tensor<64xf32>, %iradg: tensor<64xf32>, %iradbt: tensor<64xf32>, %irapW: tensor<32x64x1x1xf32>, %irapb: tensor<32xf32>, %irapg: tensor<32xf32>, %irapbt: tensor<32xf32>, %irbeW: tensor<64x32x1x1xf32>, %irbeb: tensor<64xf32>, %irbeg: tensor<64xf32>, %irbebt: tensor<64xf32>, %irbdW: tensor<64x1x3x3xf32>, %irbdb: tensor<64xf32>, %irbdg: tensor<64xf32>, %irbdbt: tensor<64xf32>, %irbpW: tensor<64x64x1x1xf32>, %irbpb: tensor<64xf32>, %irbpg: tensor<64xf32>, %irbpbt: tensor<64xf32>, %hW: tensor<128x64x1x1xf32>, %hb: tensor<128xf32>, %hg: tensor<128xf32>, %hbt: tensor<128xf32>, %Wd: tensor<128x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x32x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x32x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>) {
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
    %gaps = stablehlo.reduce(%hr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x8x8xf32>, tensor<f32>) -> tensor<128x128xf32>
    %gapnf = stablehlo.constant dense<64.0> : tensor<128x128xf32>
    %gap = stablehlo.divide %gaps, %gapnf : tensor<128x128xf32>
    %ld = stablehlo.dot_general %gap, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x10xf32>) -> tensor<128x10xf32>
    %ldb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %logits = stablehlo.add %ld, %ldb : tensor<128x10xf32>
    %le = stablehlo.exponential %logits : tensor<128x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<128x10xf32>
    %dyr = stablehlo.subtract %lsm, %onehot : tensor<128x10xf32>
    %bnc = stablehlo.constant dense<128.0> : tensor<128x10xf32>
    %dy = stablehlo.divide %dyr, %bnc : tensor<128x10xf32>
    %dgap = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<128x10xf32>) -> tensor<128x128xf32>
    %dWd = stablehlo.dot_general %gap, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x10xf32>) -> tensor<128x10xf32>
    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dgnf = stablehlo.constant dense<64.0> : tensor<128x128xf32>
    %dgs = stablehlo.divide %dgap, %dgnf : tensor<128x128xf32>
    %dgapin = stablehlo.broadcast_in_dim %dgs, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x8x8xf32>
    %dhrz = stablehlo.constant dense<0.0> : tensor<128x128x8x8xf32>
    %dhrsix = stablehlo.constant dense<6.0> : tensor<128x128x8x8xf32>
    %dhrg0 = stablehlo.compare GT, %hn, %dhrz : (tensor<128x128x8x8xf32>, tensor<128x128x8x8xf32>) -> tensor<128x128x8x8xi1>
    %dhrl6 = stablehlo.compare LT, %hn, %dhrsix : (tensor<128x128x8x8xf32>, tensor<128x128x8x8xf32>) -> tensor<128x128x8x8xi1>
    %dhrm = stablehlo.and %dhrg0, %dhrl6 : tensor<128x128x8x8xi1>
    %dhr = stablehlo.select %dhrm, %dgapin, %dhrz : tensor<128x128x8x8xi1>, tensor<128x128x8x8xf32>
    %dhndxh = stablehlo.multiply %hngb, %dhr : tensor<128x128x8x8xf32>
    %dhnsdxr = stablehlo.reduce(%dhndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x8x8xf32>, tensor<f32>) -> tensor<128x128xf32>
    %dhnsdx = stablehlo.broadcast_in_dim %dhnsdxr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x8x8xf32>
    %dhnxd = stablehlo.multiply %hnxh, %dhndxh : tensor<128x128x8x8xf32>
    %dhnsxdr = stablehlo.reduce(%dhnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x128x8x8xf32>, tensor<f32>) -> tensor<128x128xf32>
    %dhnsxd = stablehlo.broadcast_in_dim %dhnsxdr, dims = [0, 1] : (tensor<128x128xf32>) -> tensor<128x128x8x8xf32>
    %dhnt1 = stablehlo.multiply %dhndxh, %hnnf : tensor<128x128x8x8xf32>
    %dhni1 = stablehlo.subtract %dhnt1, %dhnsdx : tensor<128x128x8x8xf32>
    %dhnxs = stablehlo.multiply %hnxh, %dhnsxd : tensor<128x128x8x8xf32>
    %dhni2 = stablehlo.subtract %dhni1, %dhnxs : tensor<128x128x8x8xf32>
    %dhnsN = stablehlo.divide %hnistd, %hnnf : tensor<128x128x8x8xf32>
    %dhn = stablehlo.multiply %dhnsN, %dhni2 : tensor<128x128x8x8xf32>
    %dhndgp = stablehlo.multiply %dhr, %hnxh : tensor<128x128x8x8xf32>
    %dhndg = stablehlo.reduce(%dhndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x8x8xf32>, tensor<f32>) -> tensor<128xf32>
    %dhndb = stablehlo.reduce(%dhr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x8x8xf32>, tensor<f32>) -> tensor<128xf32>
    %dht = stablehlo.transpose %hW, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %dh = stablehlo.convolution(%dhn, %dht)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x128x8x8xf32>, tensor<64x128x1x1xf32>) -> tensor<128x64x8x8xf32>
    %dhWxt = stablehlo.transpose %irbpn, dims = [1, 0, 2, 3] : (tensor<128x64x8x8xf32>) -> tensor<64x128x8x8xf32>
    %dhWdt = stablehlo.transpose %dhn, dims = [1, 0, 2, 3] : (tensor<128x128x8x8xf32>) -> tensor<128x128x8x8xf32>
    %dhWraw = stablehlo.convolution(%dhWxt, %dhWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x8x8xf32>, tensor<128x128x8x8xf32>) -> tensor<64x128x1x1xf32>
    %dhW = stablehlo.transpose %dhWraw, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %dhb = stablehlo.reduce(%dhn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x128x8x8xf32>, tensor<f32>) -> tensor<128xf32>
    %irbdpndxh = stablehlo.multiply %irbpngb, %dh : tensor<128x64x8x8xf32>
    %irbdpnsdxr = stablehlo.reduce(%irbdpndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %irbdpnsdx = stablehlo.broadcast_in_dim %irbdpnsdxr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %irbdpnxd = stablehlo.multiply %irbpnxh, %irbdpndxh : tensor<128x64x8x8xf32>
    %irbdpnsxdr = stablehlo.reduce(%irbdpnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %irbdpnsxd = stablehlo.broadcast_in_dim %irbdpnsxdr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %irbdpnt1 = stablehlo.multiply %irbdpndxh, %irbpnnf : tensor<128x64x8x8xf32>
    %irbdpni1 = stablehlo.subtract %irbdpnt1, %irbdpnsdx : tensor<128x64x8x8xf32>
    %irbdpnxs = stablehlo.multiply %irbpnxh, %irbdpnsxd : tensor<128x64x8x8xf32>
    %irbdpni2 = stablehlo.subtract %irbdpni1, %irbdpnxs : tensor<128x64x8x8xf32>
    %irbdpnsN = stablehlo.divide %irbpnistd, %irbpnnf : tensor<128x64x8x8xf32>
    %irbdpn = stablehlo.multiply %irbdpnsN, %irbdpni2 : tensor<128x64x8x8xf32>
    %irbdpndgp = stablehlo.multiply %dh, %irbpnxh : tensor<128x64x8x8xf32>
    %irbdpndg = stablehlo.reduce(%irbdpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %irbdpndb = stablehlo.reduce(%dh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %irbdpt = stablehlo.transpose %irbpW, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %irbdp = stablehlo.convolution(%irbdpn, %irbdpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x8x8xf32>, tensor<64x64x1x1xf32>) -> tensor<128x64x8x8xf32>
    %irbdpWxt = stablehlo.transpose %irbdr, dims = [1, 0, 2, 3] : (tensor<128x64x8x8xf32>) -> tensor<64x128x8x8xf32>
    %irbdpWdt = stablehlo.transpose %irbdpn, dims = [1, 0, 2, 3] : (tensor<128x64x8x8xf32>) -> tensor<64x128x8x8xf32>
    %irbdpWraw = stablehlo.convolution(%irbdpWxt, %irbdpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x8x8xf32>, tensor<64x128x8x8xf32>) -> tensor<64x64x1x1xf32>
    %irbdpW = stablehlo.transpose %irbdpWraw, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %irbdpb = stablehlo.reduce(%irbdpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %irbddrz = stablehlo.constant dense<0.0> : tensor<128x64x8x8xf32>
    %irbddrsix = stablehlo.constant dense<6.0> : tensor<128x64x8x8xf32>
    %irbddrg0 = stablehlo.compare GT, %irbdn, %irbddrz : (tensor<128x64x8x8xf32>, tensor<128x64x8x8xf32>) -> tensor<128x64x8x8xi1>
    %irbddrl6 = stablehlo.compare LT, %irbdn, %irbddrsix : (tensor<128x64x8x8xf32>, tensor<128x64x8x8xf32>) -> tensor<128x64x8x8xi1>
    %irbddrm = stablehlo.and %irbddrg0, %irbddrl6 : tensor<128x64x8x8xi1>
    %irbddr = stablehlo.select %irbddrm, %irbdp, %irbddrz : tensor<128x64x8x8xi1>, tensor<128x64x8x8xf32>
    %irbddndxh = stablehlo.multiply %irbdngb, %irbddr : tensor<128x64x8x8xf32>
    %irbddnsdxr = stablehlo.reduce(%irbddndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %irbddnsdx = stablehlo.broadcast_in_dim %irbddnsdxr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %irbddnxd = stablehlo.multiply %irbdnxh, %irbddndxh : tensor<128x64x8x8xf32>
    %irbddnsxdr = stablehlo.reduce(%irbddnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %irbddnsxd = stablehlo.broadcast_in_dim %irbddnsxdr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %irbddnt1 = stablehlo.multiply %irbddndxh, %irbdnnf : tensor<128x64x8x8xf32>
    %irbddni1 = stablehlo.subtract %irbddnt1, %irbddnsdx : tensor<128x64x8x8xf32>
    %irbddnxs = stablehlo.multiply %irbdnxh, %irbddnsxd : tensor<128x64x8x8xf32>
    %irbddni2 = stablehlo.subtract %irbddni1, %irbddnxs : tensor<128x64x8x8xf32>
    %irbddnsN = stablehlo.divide %irbdnistd, %irbdnnf : tensor<128x64x8x8xf32>
    %irbddn = stablehlo.multiply %irbddnsN, %irbddni2 : tensor<128x64x8x8xf32>
    %irbddndgp = stablehlo.multiply %irbddr, %irbdnxh : tensor<128x64x8x8xf32>
    %irbddndg = stablehlo.reduce(%irbddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %irbddndb = stablehlo.reduce(%irbddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %irbddrev = stablehlo.reverse %irbdW, dims = [2, 3] : tensor<64x1x3x3xf32>
    %irbdd = stablehlo.convolution(%irbddn, %irbddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<128x64x8x8xf32>, tensor<64x1x3x3xf32>) -> tensor<128x64x8x8xf32>
    %irbddWxt = stablehlo.transpose %irber, dims = [1, 0, 2, 3] : (tensor<128x64x8x8xf32>) -> tensor<64x128x8x8xf32>
    %irbddWdt = stablehlo.transpose %irbddn, dims = [1, 0, 2, 3] : (tensor<128x64x8x8xf32>) -> tensor<64x128x8x8xf32>
    %irbddWraw = stablehlo.convolution(%irbddWxt, %irbddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 64 : i64, feature_group_count = 1 : i64} : (tensor<64x128x8x8xf32>, tensor<64x128x8x8xf32>) -> tensor<1x64x3x3xf32>
    %irbddW = stablehlo.reshape %irbddWraw : (tensor<1x64x3x3xf32>) -> tensor<64x1x3x3xf32>
    %irbddb = stablehlo.reduce(%irbddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %irbderz = stablehlo.constant dense<0.0> : tensor<128x64x8x8xf32>
    %irbdersix = stablehlo.constant dense<6.0> : tensor<128x64x8x8xf32>
    %irbderg0 = stablehlo.compare GT, %irben, %irbderz : (tensor<128x64x8x8xf32>, tensor<128x64x8x8xf32>) -> tensor<128x64x8x8xi1>
    %irbderl6 = stablehlo.compare LT, %irben, %irbdersix : (tensor<128x64x8x8xf32>, tensor<128x64x8x8xf32>) -> tensor<128x64x8x8xi1>
    %irbderm = stablehlo.and %irbderg0, %irbderl6 : tensor<128x64x8x8xi1>
    %irbder = stablehlo.select %irbderm, %irbdd, %irbderz : tensor<128x64x8x8xi1>, tensor<128x64x8x8xf32>
    %irbdendxh = stablehlo.multiply %irbengb, %irbder : tensor<128x64x8x8xf32>
    %irbdensdxr = stablehlo.reduce(%irbdendxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %irbdensdx = stablehlo.broadcast_in_dim %irbdensdxr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %irbdenxd = stablehlo.multiply %irbenxh, %irbdendxh : tensor<128x64x8x8xf32>
    %irbdensxdr = stablehlo.reduce(%irbdenxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %irbdensxd = stablehlo.broadcast_in_dim %irbdensxdr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %irbdent1 = stablehlo.multiply %irbdendxh, %irbennf : tensor<128x64x8x8xf32>
    %irbdeni1 = stablehlo.subtract %irbdent1, %irbdensdx : tensor<128x64x8x8xf32>
    %irbdenxs = stablehlo.multiply %irbenxh, %irbdensxd : tensor<128x64x8x8xf32>
    %irbdeni2 = stablehlo.subtract %irbdeni1, %irbdenxs : tensor<128x64x8x8xf32>
    %irbdensN = stablehlo.divide %irbenistd, %irbennf : tensor<128x64x8x8xf32>
    %irbden = stablehlo.multiply %irbdensN, %irbdeni2 : tensor<128x64x8x8xf32>
    %irbdendgp = stablehlo.multiply %irbder, %irbenxh : tensor<128x64x8x8xf32>
    %irbdendg = stablehlo.reduce(%irbdendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %irbdendb = stablehlo.reduce(%irbder init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %irbdet = stablehlo.transpose %irbeW, dims = [1, 0, 2, 3] : (tensor<64x32x1x1xf32>) -> tensor<32x64x1x1xf32>
    %irbde = stablehlo.convolution(%irbden, %irbdet)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x8x8xf32>, tensor<32x64x1x1xf32>) -> tensor<128x32x8x8xf32>
    %irbdeWxt = stablehlo.transpose %irao, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %irbdeWdt = stablehlo.transpose %irbden, dims = [1, 0, 2, 3] : (tensor<128x64x8x8xf32>) -> tensor<64x128x8x8xf32>
    %irbdeWraw = stablehlo.convolution(%irbdeWxt, %irbdeWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<64x128x8x8xf32>) -> tensor<32x64x1x1xf32>
    %irbdeW = stablehlo.transpose %irbdeWraw, dims = [1, 0, 2, 3] : (tensor<32x64x1x1xf32>) -> tensor<64x32x1x1xf32>
    %irbdeb = stablehlo.reduce(%irbden init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %iradpndxh = stablehlo.multiply %irapngb, %irbde : tensor<128x32x8x8xf32>
    %iradpnsdxr = stablehlo.reduce(%iradpndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %iradpnsdx = stablehlo.broadcast_in_dim %iradpnsdxr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %iradpnxd = stablehlo.multiply %irapnxh, %iradpndxh : tensor<128x32x8x8xf32>
    %iradpnsxdr = stablehlo.reduce(%iradpnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %iradpnsxd = stablehlo.broadcast_in_dim %iradpnsxdr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %iradpnt1 = stablehlo.multiply %iradpndxh, %irapnnf : tensor<128x32x8x8xf32>
    %iradpni1 = stablehlo.subtract %iradpnt1, %iradpnsdx : tensor<128x32x8x8xf32>
    %iradpnxs = stablehlo.multiply %irapnxh, %iradpnsxd : tensor<128x32x8x8xf32>
    %iradpni2 = stablehlo.subtract %iradpni1, %iradpnxs : tensor<128x32x8x8xf32>
    %iradpnsN = stablehlo.divide %irapnistd, %irapnnf : tensor<128x32x8x8xf32>
    %iradpn = stablehlo.multiply %iradpnsN, %iradpni2 : tensor<128x32x8x8xf32>
    %iradpndgp = stablehlo.multiply %irbde, %irapnxh : tensor<128x32x8x8xf32>
    %iradpndg = stablehlo.reduce(%iradpndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %iradpndb = stablehlo.reduce(%irbde init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %iradpt = stablehlo.transpose %irapW, dims = [1, 0, 2, 3] : (tensor<32x64x1x1xf32>) -> tensor<64x32x1x1xf32>
    %iradp = stablehlo.convolution(%iradpn, %iradpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<64x32x1x1xf32>) -> tensor<128x64x8x8xf32>
    %iradpWxt = stablehlo.transpose %iradr, dims = [1, 0, 2, 3] : (tensor<128x64x8x8xf32>) -> tensor<64x128x8x8xf32>
    %iradpWdt = stablehlo.transpose %iradpn, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %iradpWraw = stablehlo.convolution(%iradpWxt, %iradpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<64x32x1x1xf32>
    %iradpW = stablehlo.transpose %iradpWraw, dims = [1, 0, 2, 3] : (tensor<64x32x1x1xf32>) -> tensor<32x64x1x1xf32>
    %iradpb = stablehlo.reduce(%iradpn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %iraddrz = stablehlo.constant dense<0.0> : tensor<128x64x8x8xf32>
    %iraddrsix = stablehlo.constant dense<6.0> : tensor<128x64x8x8xf32>
    %iraddrg0 = stablehlo.compare GT, %iradn, %iraddrz : (tensor<128x64x8x8xf32>, tensor<128x64x8x8xf32>) -> tensor<128x64x8x8xi1>
    %iraddrl6 = stablehlo.compare LT, %iradn, %iraddrsix : (tensor<128x64x8x8xf32>, tensor<128x64x8x8xf32>) -> tensor<128x64x8x8xi1>
    %iraddrm = stablehlo.and %iraddrg0, %iraddrl6 : tensor<128x64x8x8xi1>
    %iraddr = stablehlo.select %iraddrm, %iradp, %iraddrz : tensor<128x64x8x8xi1>, tensor<128x64x8x8xf32>
    %iraddndxh = stablehlo.multiply %iradngb, %iraddr : tensor<128x64x8x8xf32>
    %iraddnsdxr = stablehlo.reduce(%iraddndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %iraddnsdx = stablehlo.broadcast_in_dim %iraddnsdxr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %iraddnxd = stablehlo.multiply %iradnxh, %iraddndxh : tensor<128x64x8x8xf32>
    %iraddnsxdr = stablehlo.reduce(%iraddnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %iraddnsxd = stablehlo.broadcast_in_dim %iraddnsxdr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %iraddnt1 = stablehlo.multiply %iraddndxh, %iradnnf : tensor<128x64x8x8xf32>
    %iraddni1 = stablehlo.subtract %iraddnt1, %iraddnsdx : tensor<128x64x8x8xf32>
    %iraddnxs = stablehlo.multiply %iradnxh, %iraddnsxd : tensor<128x64x8x8xf32>
    %iraddni2 = stablehlo.subtract %iraddni1, %iraddnxs : tensor<128x64x8x8xf32>
    %iraddnsN = stablehlo.divide %iradnistd, %iradnnf : tensor<128x64x8x8xf32>
    %iraddn = stablehlo.multiply %iraddnsN, %iraddni2 : tensor<128x64x8x8xf32>
    %iraddndgp = stablehlo.multiply %iraddr, %iradnxh : tensor<128x64x8x8xf32>
    %iraddndg = stablehlo.reduce(%iraddndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %iraddndb = stablehlo.reduce(%iraddr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %iraddrev = stablehlo.reverse %iradW, dims = [2, 3] : tensor<64x1x3x3xf32>
    %iradd = stablehlo.convolution(%iraddn, %iraddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<128x64x8x8xf32>, tensor<64x1x3x3xf32>) -> tensor<128x64x8x8xf32>
    %iraddWxt = stablehlo.transpose %iraer, dims = [1, 0, 2, 3] : (tensor<128x64x8x8xf32>) -> tensor<64x128x8x8xf32>
    %iraddWdt = stablehlo.transpose %iraddn, dims = [1, 0, 2, 3] : (tensor<128x64x8x8xf32>) -> tensor<64x128x8x8xf32>
    %iraddWraw = stablehlo.convolution(%iraddWxt, %iraddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 64 : i64, feature_group_count = 1 : i64} : (tensor<64x128x8x8xf32>, tensor<64x128x8x8xf32>) -> tensor<1x64x3x3xf32>
    %iraddW = stablehlo.reshape %iraddWraw : (tensor<1x64x3x3xf32>) -> tensor<64x1x3x3xf32>
    %iraddb = stablehlo.reduce(%iraddn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %iraderz = stablehlo.constant dense<0.0> : tensor<128x64x8x8xf32>
    %iradersix = stablehlo.constant dense<6.0> : tensor<128x64x8x8xf32>
    %iraderg0 = stablehlo.compare GT, %iraen, %iraderz : (tensor<128x64x8x8xf32>, tensor<128x64x8x8xf32>) -> tensor<128x64x8x8xi1>
    %iraderl6 = stablehlo.compare LT, %iraen, %iradersix : (tensor<128x64x8x8xf32>, tensor<128x64x8x8xf32>) -> tensor<128x64x8x8xi1>
    %iraderm = stablehlo.and %iraderg0, %iraderl6 : tensor<128x64x8x8xi1>
    %irader = stablehlo.select %iraderm, %iradd, %iraderz : tensor<128x64x8x8xi1>, tensor<128x64x8x8xf32>
    %iradendxh = stablehlo.multiply %iraengb, %irader : tensor<128x64x8x8xf32>
    %iradensdxr = stablehlo.reduce(%iradendxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %iradensdx = stablehlo.broadcast_in_dim %iradensdxr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %iradenxd = stablehlo.multiply %iraenxh, %iradendxh : tensor<128x64x8x8xf32>
    %iradensxdr = stablehlo.reduce(%iradenxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64xf32>
    %iradensxd = stablehlo.broadcast_in_dim %iradensxdr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x8x8xf32>
    %iradent1 = stablehlo.multiply %iradendxh, %iraennf : tensor<128x64x8x8xf32>
    %iradeni1 = stablehlo.subtract %iradent1, %iradensdx : tensor<128x64x8x8xf32>
    %iradenxs = stablehlo.multiply %iraenxh, %iradensxd : tensor<128x64x8x8xf32>
    %iradeni2 = stablehlo.subtract %iradeni1, %iradenxs : tensor<128x64x8x8xf32>
    %iradensN = stablehlo.divide %iraenistd, %iraennf : tensor<128x64x8x8xf32>
    %iraden = stablehlo.multiply %iradensN, %iradeni2 : tensor<128x64x8x8xf32>
    %iradendgp = stablehlo.multiply %irader, %iraenxh : tensor<128x64x8x8xf32>
    %iradendg = stablehlo.reduce(%iradendgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %iradendb = stablehlo.reduce(%irader init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %iradet = stablehlo.transpose %iraeW, dims = [1, 0, 2, 3] : (tensor<64x32x1x1xf32>) -> tensor<32x64x1x1xf32>
    %irade = stablehlo.convolution(%iraden, %iradet)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x8x8xf32>, tensor<32x64x1x1xf32>) -> tensor<128x32x8x8xf32>
    %iradeWxt = stablehlo.transpose %stp, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %iradeWdt = stablehlo.transpose %iraden, dims = [1, 0, 2, 3] : (tensor<128x64x8x8xf32>) -> tensor<64x128x8x8xf32>
    %iradeWraw = stablehlo.convolution(%iradeWxt, %iradeWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<64x128x8x8xf32>) -> tensor<32x64x1x1xf32>
    %iradeW = stablehlo.transpose %iradeWraw, dims = [1, 0, 2, 3] : (tensor<32x64x1x1xf32>) -> tensor<64x32x1x1xf32>
    %iradeb = stablehlo.reduce(%iraden init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<64xf32>
    %iradx = stablehlo.add %irade, %irbde : tensor<128x32x8x8xf32>
    %dmp = "stablehlo.select_and_scatter"(%str, %iradx, %sc) ({
      ^bb0(%qa: tensor<f32>, %qb: tensor<f32>):
        %qge = stablehlo.compare GE, %qa, %qb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %qge : tensor<i1>
    }, {
      ^bb0(%qc: tensor<f32>, %qd: tensor<f32>):
        %qs = stablehlo.add %qc, %qd : tensor<f32>
        stablehlo.return %qs : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x16x16xf32>, tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    %dstrz = stablehlo.constant dense<0.0> : tensor<128x32x16x16xf32>
    %dstrsix = stablehlo.constant dense<6.0> : tensor<128x32x16x16xf32>
    %dstrg0 = stablehlo.compare GT, %stn, %dstrz : (tensor<128x32x16x16xf32>, tensor<128x32x16x16xf32>) -> tensor<128x32x16x16xi1>
    %dstrl6 = stablehlo.compare LT, %stn, %dstrsix : (tensor<128x32x16x16xf32>, tensor<128x32x16x16xf32>) -> tensor<128x32x16x16xi1>
    %dstrm = stablehlo.and %dstrg0, %dstrl6 : tensor<128x32x16x16xi1>
    %dstr = stablehlo.select %dstrm, %dmp, %dstrz : tensor<128x32x16x16xi1>, tensor<128x32x16x16xf32>
    %dstndxh = stablehlo.multiply %stngb, %dstr : tensor<128x32x16x16xf32>
    %dstnsdxr = stablehlo.reduce(%dstndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dstnsdx = stablehlo.broadcast_in_dim %dstnsdxr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x16x16xf32>
    %dstnxd = stablehlo.multiply %stnxh, %dstndxh : tensor<128x32x16x16xf32>
    %dstnsxdr = stablehlo.reduce(%dstnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dstnsxd = stablehlo.broadcast_in_dim %dstnsxdr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x16x16xf32>
    %dstnt1 = stablehlo.multiply %dstndxh, %stnnf : tensor<128x32x16x16xf32>
    %dstni1 = stablehlo.subtract %dstnt1, %dstnsdx : tensor<128x32x16x16xf32>
    %dstnxs = stablehlo.multiply %stnxh, %dstnsxd : tensor<128x32x16x16xf32>
    %dstni2 = stablehlo.subtract %dstni1, %dstnxs : tensor<128x32x16x16xf32>
    %dstnsN = stablehlo.divide %stnistd, %stnnf : tensor<128x32x16x16xf32>
    %dstn = stablehlo.multiply %dstnsN, %dstni2 : tensor<128x32x16x16xf32>
    %dstndgp = stablehlo.multiply %dstr, %stnxh : tensor<128x32x16x16xf32>
    %dstndg = stablehlo.reduce(%dstndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<32xf32>
    %dstndb = stablehlo.reduce(%dstr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<32xf32>
    %dsb = stablehlo.reduce(%dstn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<32xf32>
    %dsWu = stablehlo.pad %dstn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32x32x32xf32>
    %dsWxt = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %dsWdt = stablehlo.transpose %dsWu, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %dsWraw = stablehlo.convolution(%dsWxt, %dsWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<3x32x3x3xf32>
    %dsW = stablehlo.transpose %dsWraw, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %sWl = stablehlo.constant dense<0.1> : tensor<32x3x3x3xf32>
    %sWs = stablehlo.multiply %dsW, %sWl : tensor<32x3x3x3xf32>
    %sWn = stablehlo.subtract %sW, %sWs : tensor<32x3x3x3xf32>
    %sbl = stablehlo.constant dense<0.1> : tensor<32xf32>
    %sbs = stablehlo.multiply %dsb, %sbl : tensor<32xf32>
    %sbn = stablehlo.subtract %sb, %sbs : tensor<32xf32>
    %sgl = stablehlo.constant dense<0.1> : tensor<32xf32>
    %sgs = stablehlo.multiply %dstndg, %sgl : tensor<32xf32>
    %sgn = stablehlo.subtract %sg, %sgs : tensor<32xf32>
    %sbtl = stablehlo.constant dense<0.1> : tensor<32xf32>
    %sbts = stablehlo.multiply %dstndb, %sbtl : tensor<32xf32>
    %sbtn = stablehlo.subtract %sbt, %sbts : tensor<32xf32>
    %iraeWl = stablehlo.constant dense<0.1> : tensor<64x32x1x1xf32>
    %iraeWs = stablehlo.multiply %iradeW, %iraeWl : tensor<64x32x1x1xf32>
    %iraeWn = stablehlo.subtract %iraeW, %iraeWs : tensor<64x32x1x1xf32>
    %iraebl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %iraebs = stablehlo.multiply %iradeb, %iraebl : tensor<64xf32>
    %iraebn = stablehlo.subtract %iraeb, %iraebs : tensor<64xf32>
    %iraegl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %iraegs = stablehlo.multiply %iradendg, %iraegl : tensor<64xf32>
    %iraegn = stablehlo.subtract %iraeg, %iraegs : tensor<64xf32>
    %iraebtl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %iraebts = stablehlo.multiply %iradendb, %iraebtl : tensor<64xf32>
    %iraebtn = stablehlo.subtract %iraebt, %iraebts : tensor<64xf32>
    %iradWl = stablehlo.constant dense<0.1> : tensor<64x1x3x3xf32>
    %iradWs = stablehlo.multiply %iraddW, %iradWl : tensor<64x1x3x3xf32>
    %iradWn = stablehlo.subtract %iradW, %iradWs : tensor<64x1x3x3xf32>
    %iradbl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %iradbs = stablehlo.multiply %iraddb, %iradbl : tensor<64xf32>
    %iradbn = stablehlo.subtract %iradb, %iradbs : tensor<64xf32>
    %iradgl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %iradgs = stablehlo.multiply %iraddndg, %iradgl : tensor<64xf32>
    %iradgn = stablehlo.subtract %iradg, %iradgs : tensor<64xf32>
    %iradbtl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %iradbts = stablehlo.multiply %iraddndb, %iradbtl : tensor<64xf32>
    %iradbtn = stablehlo.subtract %iradbt, %iradbts : tensor<64xf32>
    %irapWl = stablehlo.constant dense<0.1> : tensor<32x64x1x1xf32>
    %irapWs = stablehlo.multiply %iradpW, %irapWl : tensor<32x64x1x1xf32>
    %irapWn = stablehlo.subtract %irapW, %irapWs : tensor<32x64x1x1xf32>
    %irapbl = stablehlo.constant dense<0.1> : tensor<32xf32>
    %irapbs = stablehlo.multiply %iradpb, %irapbl : tensor<32xf32>
    %irapbn = stablehlo.subtract %irapb, %irapbs : tensor<32xf32>
    %irapgl = stablehlo.constant dense<0.1> : tensor<32xf32>
    %irapgs = stablehlo.multiply %iradpndg, %irapgl : tensor<32xf32>
    %irapgn = stablehlo.subtract %irapg, %irapgs : tensor<32xf32>
    %irapbtl = stablehlo.constant dense<0.1> : tensor<32xf32>
    %irapbts = stablehlo.multiply %iradpndb, %irapbtl : tensor<32xf32>
    %irapbtn = stablehlo.subtract %irapbt, %irapbts : tensor<32xf32>
    %irbeWl = stablehlo.constant dense<0.1> : tensor<64x32x1x1xf32>
    %irbeWs = stablehlo.multiply %irbdeW, %irbeWl : tensor<64x32x1x1xf32>
    %irbeWn = stablehlo.subtract %irbeW, %irbeWs : tensor<64x32x1x1xf32>
    %irbebl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %irbebs = stablehlo.multiply %irbdeb, %irbebl : tensor<64xf32>
    %irbebn = stablehlo.subtract %irbeb, %irbebs : tensor<64xf32>
    %irbegl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %irbegs = stablehlo.multiply %irbdendg, %irbegl : tensor<64xf32>
    %irbegn = stablehlo.subtract %irbeg, %irbegs : tensor<64xf32>
    %irbebtl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %irbebts = stablehlo.multiply %irbdendb, %irbebtl : tensor<64xf32>
    %irbebtn = stablehlo.subtract %irbebt, %irbebts : tensor<64xf32>
    %irbdWl = stablehlo.constant dense<0.1> : tensor<64x1x3x3xf32>
    %irbdWs = stablehlo.multiply %irbddW, %irbdWl : tensor<64x1x3x3xf32>
    %irbdWn = stablehlo.subtract %irbdW, %irbdWs : tensor<64x1x3x3xf32>
    %irbdbl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %irbdbs = stablehlo.multiply %irbddb, %irbdbl : tensor<64xf32>
    %irbdbn = stablehlo.subtract %irbdb, %irbdbs : tensor<64xf32>
    %irbdgl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %irbdgs = stablehlo.multiply %irbddndg, %irbdgl : tensor<64xf32>
    %irbdgn = stablehlo.subtract %irbdg, %irbdgs : tensor<64xf32>
    %irbdbtl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %irbdbts = stablehlo.multiply %irbddndb, %irbdbtl : tensor<64xf32>
    %irbdbtn = stablehlo.subtract %irbdbt, %irbdbts : tensor<64xf32>
    %irbpWl = stablehlo.constant dense<0.1> : tensor<64x64x1x1xf32>
    %irbpWs = stablehlo.multiply %irbdpW, %irbpWl : tensor<64x64x1x1xf32>
    %irbpWn = stablehlo.subtract %irbpW, %irbpWs : tensor<64x64x1x1xf32>
    %irbpbl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %irbpbs = stablehlo.multiply %irbdpb, %irbpbl : tensor<64xf32>
    %irbpbn = stablehlo.subtract %irbpb, %irbpbs : tensor<64xf32>
    %irbpgl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %irbpgs = stablehlo.multiply %irbdpndg, %irbpgl : tensor<64xf32>
    %irbpgn = stablehlo.subtract %irbpg, %irbpgs : tensor<64xf32>
    %irbpbtl = stablehlo.constant dense<0.1> : tensor<64xf32>
    %irbpbts = stablehlo.multiply %irbdpndb, %irbpbtl : tensor<64xf32>
    %irbpbtn = stablehlo.subtract %irbpbt, %irbpbts : tensor<64xf32>
    %hWl = stablehlo.constant dense<0.1> : tensor<128x64x1x1xf32>
    %hWs = stablehlo.multiply %dhW, %hWl : tensor<128x64x1x1xf32>
    %hWn = stablehlo.subtract %hW, %hWs : tensor<128x64x1x1xf32>
    %hbl = stablehlo.constant dense<0.1> : tensor<128xf32>
    %hbs = stablehlo.multiply %dhb, %hbl : tensor<128xf32>
    %hbn = stablehlo.subtract %hb, %hbs : tensor<128xf32>
    %hgl = stablehlo.constant dense<0.1> : tensor<128xf32>
    %hgs = stablehlo.multiply %dhndg, %hgl : tensor<128xf32>
    %hgn = stablehlo.subtract %hg, %hgs : tensor<128xf32>
    %hbtl = stablehlo.constant dense<0.1> : tensor<128xf32>
    %hbts = stablehlo.multiply %dhndb, %hbtl : tensor<128xf32>
    %hbtn = stablehlo.subtract %hbt, %hbts : tensor<128xf32>
    %Wdl = stablehlo.constant dense<0.1> : tensor<128x10xf32>
    %Wds = stablehlo.multiply %dWd, %Wdl : tensor<128x10xf32>
    %Wdn = stablehlo.subtract %Wd, %Wds : tensor<128x10xf32>
    %bdl = stablehlo.constant dense<0.1> : tensor<10xf32>
    %bds = stablehlo.multiply %dbd, %bdl : tensor<10xf32>
    %bdn = stablehlo.subtract %bd, %bds : tensor<10xf32>
    return %sWn, %sbn, %sgn, %sbtn, %iraeWn, %iraebn, %iraegn, %iraebtn, %iradWn, %iradbn, %iradgn, %iradbtn, %irapWn, %irapbn, %irapgn, %irapbtn, %irbeWn, %irbebn, %irbegn, %irbebtn, %irbdWn, %irbdbn, %irbdgn, %irbdbtn, %irbpWn, %irbpbn, %irbpgn, %irbpbtn, %hWn, %hbn, %hgn, %hbtn, %Wdn, %bdn : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x32x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x32x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>
  }
}
