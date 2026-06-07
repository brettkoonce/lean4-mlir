module @m {
  func.func @convnext_train_step(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<f32>, %s0b0nbt: tensor<f32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<f32>, %s0b1nbt: tensor<f32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<f32>, %s0b2nbt: tensor<f32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<f32>, %d0nbt: tensor<f32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<f32>, %s1b0nbt: tensor<f32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<f32>, %s1b1nbt: tensor<f32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<f32>, %s1b2nbt: tensor<f32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<f32>, %d1nbt: tensor<f32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<f32>, %s2b0nbt: tensor<f32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<f32>, %s2b1nbt: tensor<f32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<f32>, %s2b2nbt: tensor<f32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<f32>, %s2b3nbt: tensor<f32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<f32>, %s2b4nbt: tensor<f32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<f32>, %s2b5nbt: tensor<f32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<f32>, %s2b6nbt: tensor<f32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<f32>, %s2b7nbt: tensor<f32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<f32>, %s2b8nbt: tensor<f32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<f32>, %d2nbt: tensor<f32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<f32>, %s3b0nbt: tensor<f32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<f32>, %s3b1nbt: tensor<f32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<f32>, %s3b2nbt: tensor<f32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<f32>, %hnbt: tensor<f32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %psc = stablehlo.convolution(%xr, %psW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<96x3x4x4xf32>) -> tensor<32x96x56x56xf32>
    %psbb = stablehlo.broadcast_in_dim %psb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %ps = stablehlo.add %psc, %psbb : tensor<32x96x56x56xf32>
    %s0b0dc = stablehlo.convolution(%ps, %s0b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %s0b0dbb = stablehlo.broadcast_in_dim %s0b0db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b0d = stablehlo.add %s0b0dc, %s0b0dbb : tensor<32x96x56x56xf32>
    %s0b0nri = stablehlo.reshape %s0b0d : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b0nnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %s0b0nep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %s0b0nsmr = stablehlo.reduce(%s0b0nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b0nsm = stablehlo.broadcast_in_dim %s0b0nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b0nmu = stablehlo.divide %s0b0nsm, %s0b0nnf : tensor<32x301056xf32>
    %s0b0nxc = stablehlo.subtract %s0b0nri, %s0b0nmu : tensor<32x301056xf32>
    %s0b0nsq = stablehlo.multiply %s0b0nxc, %s0b0nxc : tensor<32x301056xf32>
    %s0b0nvsr = stablehlo.reduce(%s0b0nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b0nvs = stablehlo.broadcast_in_dim %s0b0nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b0nvr = stablehlo.divide %s0b0nvs, %s0b0nnf : tensor<32x301056xf32>
    %s0b0nve = stablehlo.add %s0b0nvr, %s0b0nep : tensor<32x301056xf32>
    %s0b0nistd = stablehlo.rsqrt %s0b0nve : tensor<32x301056xf32>
    %s0b0nxh = stablehlo.multiply %s0b0nxc, %s0b0nistd : tensor<32x301056xf32>
    %s0b0ngb = stablehlo.broadcast_in_dim %s0b0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b0nbtb = stablehlo.broadcast_in_dim %s0b0nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b0ngx = stablehlo.multiply %s0b0nxh, %s0b0ngb : tensor<32x301056xf32>
    %s0b0nfl = stablehlo.add %s0b0ngx, %s0b0nbtb : tensor<32x301056xf32>
    %s0b0n = stablehlo.reshape %s0b0nfl : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %s0b0ec = stablehlo.convolution(%s0b0n, %s0b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %s0b0ebb = stablehlo.broadcast_in_dim %s0b0eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %s0b0e = stablehlo.add %s0b0ec, %s0b0ebb : tensor<32x384x56x56xf32>
    %s0b0gx2 = stablehlo.multiply %s0b0e, %s0b0e : tensor<32x384x56x56xf32>
    %s0b0gx3 = stablehlo.multiply %s0b0gx2, %s0b0e : tensor<32x384x56x56xf32>
    %s0b0gck = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %s0b0gkx3 = stablehlo.multiply %s0b0gck, %s0b0gx3 : tensor<32x384x56x56xf32>
    %s0b0ginn = stablehlo.add %s0b0e, %s0b0gkx3 : tensor<32x384x56x56xf32>
    %s0b0gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %s0b0gu = stablehlo.multiply %s0b0gcs, %s0b0ginn : tensor<32x384x56x56xf32>
    %s0b0gt = stablehlo.tanh %s0b0gu : tensor<32x384x56x56xf32>
    %s0b0gone = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %s0b0gopt = stablehlo.add %s0b0gone, %s0b0gt : tensor<32x384x56x56xf32>
    %s0b0ghalf = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %s0b0ghx = stablehlo.multiply %s0b0ghalf, %s0b0e : tensor<32x384x56x56xf32>
    %s0b0g = stablehlo.multiply %s0b0ghx, %s0b0gopt : tensor<32x384x56x56xf32>
    %s0b0pc = stablehlo.convolution(%s0b0g, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %s0b0pbb = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b0p = stablehlo.add %s0b0pc, %s0b0pbb : tensor<32x96x56x56xf32>
    %s0b0lsgb = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b0ls = stablehlo.multiply %s0b0p, %s0b0lsgb : tensor<32x96x56x56xf32>
    %s0b0o = stablehlo.add %s0b0ls, %ps : tensor<32x96x56x56xf32>
    %s0b1dc = stablehlo.convolution(%s0b0o, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %s0b1dbb = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b1d = stablehlo.add %s0b1dc, %s0b1dbb : tensor<32x96x56x56xf32>
    %s0b1nri = stablehlo.reshape %s0b1d : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b1nnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %s0b1nep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %s0b1nsmr = stablehlo.reduce(%s0b1nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b1nsm = stablehlo.broadcast_in_dim %s0b1nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b1nmu = stablehlo.divide %s0b1nsm, %s0b1nnf : tensor<32x301056xf32>
    %s0b1nxc = stablehlo.subtract %s0b1nri, %s0b1nmu : tensor<32x301056xf32>
    %s0b1nsq = stablehlo.multiply %s0b1nxc, %s0b1nxc : tensor<32x301056xf32>
    %s0b1nvsr = stablehlo.reduce(%s0b1nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b1nvs = stablehlo.broadcast_in_dim %s0b1nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b1nvr = stablehlo.divide %s0b1nvs, %s0b1nnf : tensor<32x301056xf32>
    %s0b1nve = stablehlo.add %s0b1nvr, %s0b1nep : tensor<32x301056xf32>
    %s0b1nistd = stablehlo.rsqrt %s0b1nve : tensor<32x301056xf32>
    %s0b1nxh = stablehlo.multiply %s0b1nxc, %s0b1nistd : tensor<32x301056xf32>
    %s0b1ngb = stablehlo.broadcast_in_dim %s0b1ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b1nbtb = stablehlo.broadcast_in_dim %s0b1nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b1ngx = stablehlo.multiply %s0b1nxh, %s0b1ngb : tensor<32x301056xf32>
    %s0b1nfl = stablehlo.add %s0b1ngx, %s0b1nbtb : tensor<32x301056xf32>
    %s0b1n = stablehlo.reshape %s0b1nfl : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %s0b1ec = stablehlo.convolution(%s0b1n, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %s0b1ebb = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %s0b1e = stablehlo.add %s0b1ec, %s0b1ebb : tensor<32x384x56x56xf32>
    %s0b1gx2 = stablehlo.multiply %s0b1e, %s0b1e : tensor<32x384x56x56xf32>
    %s0b1gx3 = stablehlo.multiply %s0b1gx2, %s0b1e : tensor<32x384x56x56xf32>
    %s0b1gck = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %s0b1gkx3 = stablehlo.multiply %s0b1gck, %s0b1gx3 : tensor<32x384x56x56xf32>
    %s0b1ginn = stablehlo.add %s0b1e, %s0b1gkx3 : tensor<32x384x56x56xf32>
    %s0b1gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %s0b1gu = stablehlo.multiply %s0b1gcs, %s0b1ginn : tensor<32x384x56x56xf32>
    %s0b1gt = stablehlo.tanh %s0b1gu : tensor<32x384x56x56xf32>
    %s0b1gone = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %s0b1gopt = stablehlo.add %s0b1gone, %s0b1gt : tensor<32x384x56x56xf32>
    %s0b1ghalf = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %s0b1ghx = stablehlo.multiply %s0b1ghalf, %s0b1e : tensor<32x384x56x56xf32>
    %s0b1g = stablehlo.multiply %s0b1ghx, %s0b1gopt : tensor<32x384x56x56xf32>
    %s0b1pc = stablehlo.convolution(%s0b1g, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %s0b1pbb = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b1p = stablehlo.add %s0b1pc, %s0b1pbb : tensor<32x96x56x56xf32>
    %s0b1lsgb = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b1ls = stablehlo.multiply %s0b1p, %s0b1lsgb : tensor<32x96x56x56xf32>
    %s0b1o = stablehlo.add %s0b1ls, %s0b0o : tensor<32x96x56x56xf32>
    %s0b2dc = stablehlo.convolution(%s0b1o, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %s0b2dbb = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b2d = stablehlo.add %s0b2dc, %s0b2dbb : tensor<32x96x56x56xf32>
    %s0b2nri = stablehlo.reshape %s0b2d : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b2nnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %s0b2nep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %s0b2nsmr = stablehlo.reduce(%s0b2nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b2nsm = stablehlo.broadcast_in_dim %s0b2nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b2nmu = stablehlo.divide %s0b2nsm, %s0b2nnf : tensor<32x301056xf32>
    %s0b2nxc = stablehlo.subtract %s0b2nri, %s0b2nmu : tensor<32x301056xf32>
    %s0b2nsq = stablehlo.multiply %s0b2nxc, %s0b2nxc : tensor<32x301056xf32>
    %s0b2nvsr = stablehlo.reduce(%s0b2nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b2nvs = stablehlo.broadcast_in_dim %s0b2nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b2nvr = stablehlo.divide %s0b2nvs, %s0b2nnf : tensor<32x301056xf32>
    %s0b2nve = stablehlo.add %s0b2nvr, %s0b2nep : tensor<32x301056xf32>
    %s0b2nistd = stablehlo.rsqrt %s0b2nve : tensor<32x301056xf32>
    %s0b2nxh = stablehlo.multiply %s0b2nxc, %s0b2nistd : tensor<32x301056xf32>
    %s0b2ngb = stablehlo.broadcast_in_dim %s0b2ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b2nbtb = stablehlo.broadcast_in_dim %s0b2nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b2ngx = stablehlo.multiply %s0b2nxh, %s0b2ngb : tensor<32x301056xf32>
    %s0b2nfl = stablehlo.add %s0b2ngx, %s0b2nbtb : tensor<32x301056xf32>
    %s0b2n = stablehlo.reshape %s0b2nfl : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %s0b2ec = stablehlo.convolution(%s0b2n, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %s0b2ebb = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %s0b2e = stablehlo.add %s0b2ec, %s0b2ebb : tensor<32x384x56x56xf32>
    %s0b2gx2 = stablehlo.multiply %s0b2e, %s0b2e : tensor<32x384x56x56xf32>
    %s0b2gx3 = stablehlo.multiply %s0b2gx2, %s0b2e : tensor<32x384x56x56xf32>
    %s0b2gck = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %s0b2gkx3 = stablehlo.multiply %s0b2gck, %s0b2gx3 : tensor<32x384x56x56xf32>
    %s0b2ginn = stablehlo.add %s0b2e, %s0b2gkx3 : tensor<32x384x56x56xf32>
    %s0b2gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %s0b2gu = stablehlo.multiply %s0b2gcs, %s0b2ginn : tensor<32x384x56x56xf32>
    %s0b2gt = stablehlo.tanh %s0b2gu : tensor<32x384x56x56xf32>
    %s0b2gone = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %s0b2gopt = stablehlo.add %s0b2gone, %s0b2gt : tensor<32x384x56x56xf32>
    %s0b2ghalf = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %s0b2ghx = stablehlo.multiply %s0b2ghalf, %s0b2e : tensor<32x384x56x56xf32>
    %s0b2g = stablehlo.multiply %s0b2ghx, %s0b2gopt : tensor<32x384x56x56xf32>
    %s0b2pc = stablehlo.convolution(%s0b2g, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %s0b2pbb = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b2p = stablehlo.add %s0b2pc, %s0b2pbb : tensor<32x96x56x56xf32>
    %s0b2lsgb = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b2ls = stablehlo.multiply %s0b2p, %s0b2lsgb : tensor<32x96x56x56xf32>
    %s0b2o = stablehlo.add %s0b2ls, %s0b1o : tensor<32x96x56x56xf32>
    %d0nri = stablehlo.reshape %s0b2o : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %d0nnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %d0nep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %d0nsmr = stablehlo.reduce(%d0nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %d0nsm = stablehlo.broadcast_in_dim %d0nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %d0nmu = stablehlo.divide %d0nsm, %d0nnf : tensor<32x301056xf32>
    %d0nxc = stablehlo.subtract %d0nri, %d0nmu : tensor<32x301056xf32>
    %d0nsq = stablehlo.multiply %d0nxc, %d0nxc : tensor<32x301056xf32>
    %d0nvsr = stablehlo.reduce(%d0nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %d0nvs = stablehlo.broadcast_in_dim %d0nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %d0nvr = stablehlo.divide %d0nvs, %d0nnf : tensor<32x301056xf32>
    %d0nve = stablehlo.add %d0nvr, %d0nep : tensor<32x301056xf32>
    %d0nistd = stablehlo.rsqrt %d0nve : tensor<32x301056xf32>
    %d0nxh = stablehlo.multiply %d0nxc, %d0nistd : tensor<32x301056xf32>
    %d0ngb = stablehlo.broadcast_in_dim %d0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %d0nbtb = stablehlo.broadcast_in_dim %d0nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %d0ngx = stablehlo.multiply %d0nxh, %d0ngb : tensor<32x301056xf32>
    %d0nfl = stablehlo.add %d0ngx, %d0nbtb : tensor<32x301056xf32>
    %d0n = stablehlo.reshape %d0nfl : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %d0cc = stablehlo.convolution(%d0n, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %d0cbb = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %d0c = stablehlo.add %d0cc, %d0cbb : tensor<32x192x28x28xf32>
    %s1b0dc = stablehlo.convolution(%d0c, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %s1b0dbb = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b0d = stablehlo.add %s1b0dc, %s1b0dbb : tensor<32x192x28x28xf32>
    %s1b0nri = stablehlo.reshape %s1b0d : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b0nnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %s1b0nep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %s1b0nsmr = stablehlo.reduce(%s1b0nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b0nsm = stablehlo.broadcast_in_dim %s1b0nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b0nmu = stablehlo.divide %s1b0nsm, %s1b0nnf : tensor<32x150528xf32>
    %s1b0nxc = stablehlo.subtract %s1b0nri, %s1b0nmu : tensor<32x150528xf32>
    %s1b0nsq = stablehlo.multiply %s1b0nxc, %s1b0nxc : tensor<32x150528xf32>
    %s1b0nvsr = stablehlo.reduce(%s1b0nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b0nvs = stablehlo.broadcast_in_dim %s1b0nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b0nvr = stablehlo.divide %s1b0nvs, %s1b0nnf : tensor<32x150528xf32>
    %s1b0nve = stablehlo.add %s1b0nvr, %s1b0nep : tensor<32x150528xf32>
    %s1b0nistd = stablehlo.rsqrt %s1b0nve : tensor<32x150528xf32>
    %s1b0nxh = stablehlo.multiply %s1b0nxc, %s1b0nistd : tensor<32x150528xf32>
    %s1b0ngb = stablehlo.broadcast_in_dim %s1b0ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b0nbtb = stablehlo.broadcast_in_dim %s1b0nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b0ngx = stablehlo.multiply %s1b0nxh, %s1b0ngb : tensor<32x150528xf32>
    %s1b0nfl = stablehlo.add %s1b0ngx, %s1b0nbtb : tensor<32x150528xf32>
    %s1b0n = stablehlo.reshape %s1b0nfl : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %s1b0ec = stablehlo.convolution(%s1b0n, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %s1b0ebb = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %s1b0e = stablehlo.add %s1b0ec, %s1b0ebb : tensor<32x768x28x28xf32>
    %s1b0gx2 = stablehlo.multiply %s1b0e, %s1b0e : tensor<32x768x28x28xf32>
    %s1b0gx3 = stablehlo.multiply %s1b0gx2, %s1b0e : tensor<32x768x28x28xf32>
    %s1b0gck = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %s1b0gkx3 = stablehlo.multiply %s1b0gck, %s1b0gx3 : tensor<32x768x28x28xf32>
    %s1b0ginn = stablehlo.add %s1b0e, %s1b0gkx3 : tensor<32x768x28x28xf32>
    %s1b0gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %s1b0gu = stablehlo.multiply %s1b0gcs, %s1b0ginn : tensor<32x768x28x28xf32>
    %s1b0gt = stablehlo.tanh %s1b0gu : tensor<32x768x28x28xf32>
    %s1b0gone = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %s1b0gopt = stablehlo.add %s1b0gone, %s1b0gt : tensor<32x768x28x28xf32>
    %s1b0ghalf = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %s1b0ghx = stablehlo.multiply %s1b0ghalf, %s1b0e : tensor<32x768x28x28xf32>
    %s1b0g = stablehlo.multiply %s1b0ghx, %s1b0gopt : tensor<32x768x28x28xf32>
    %s1b0pc = stablehlo.convolution(%s1b0g, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %s1b0pbb = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b0p = stablehlo.add %s1b0pc, %s1b0pbb : tensor<32x192x28x28xf32>
    %s1b0lsgb = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b0ls = stablehlo.multiply %s1b0p, %s1b0lsgb : tensor<32x192x28x28xf32>
    %s1b0o = stablehlo.add %s1b0ls, %d0c : tensor<32x192x28x28xf32>
    %s1b1dc = stablehlo.convolution(%s1b0o, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %s1b1dbb = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b1d = stablehlo.add %s1b1dc, %s1b1dbb : tensor<32x192x28x28xf32>
    %s1b1nri = stablehlo.reshape %s1b1d : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b1nnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %s1b1nep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %s1b1nsmr = stablehlo.reduce(%s1b1nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b1nsm = stablehlo.broadcast_in_dim %s1b1nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b1nmu = stablehlo.divide %s1b1nsm, %s1b1nnf : tensor<32x150528xf32>
    %s1b1nxc = stablehlo.subtract %s1b1nri, %s1b1nmu : tensor<32x150528xf32>
    %s1b1nsq = stablehlo.multiply %s1b1nxc, %s1b1nxc : tensor<32x150528xf32>
    %s1b1nvsr = stablehlo.reduce(%s1b1nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b1nvs = stablehlo.broadcast_in_dim %s1b1nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b1nvr = stablehlo.divide %s1b1nvs, %s1b1nnf : tensor<32x150528xf32>
    %s1b1nve = stablehlo.add %s1b1nvr, %s1b1nep : tensor<32x150528xf32>
    %s1b1nistd = stablehlo.rsqrt %s1b1nve : tensor<32x150528xf32>
    %s1b1nxh = stablehlo.multiply %s1b1nxc, %s1b1nistd : tensor<32x150528xf32>
    %s1b1ngb = stablehlo.broadcast_in_dim %s1b1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b1nbtb = stablehlo.broadcast_in_dim %s1b1nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b1ngx = stablehlo.multiply %s1b1nxh, %s1b1ngb : tensor<32x150528xf32>
    %s1b1nfl = stablehlo.add %s1b1ngx, %s1b1nbtb : tensor<32x150528xf32>
    %s1b1n = stablehlo.reshape %s1b1nfl : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %s1b1ec = stablehlo.convolution(%s1b1n, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %s1b1ebb = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %s1b1e = stablehlo.add %s1b1ec, %s1b1ebb : tensor<32x768x28x28xf32>
    %s1b1gx2 = stablehlo.multiply %s1b1e, %s1b1e : tensor<32x768x28x28xf32>
    %s1b1gx3 = stablehlo.multiply %s1b1gx2, %s1b1e : tensor<32x768x28x28xf32>
    %s1b1gck = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %s1b1gkx3 = stablehlo.multiply %s1b1gck, %s1b1gx3 : tensor<32x768x28x28xf32>
    %s1b1ginn = stablehlo.add %s1b1e, %s1b1gkx3 : tensor<32x768x28x28xf32>
    %s1b1gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %s1b1gu = stablehlo.multiply %s1b1gcs, %s1b1ginn : tensor<32x768x28x28xf32>
    %s1b1gt = stablehlo.tanh %s1b1gu : tensor<32x768x28x28xf32>
    %s1b1gone = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %s1b1gopt = stablehlo.add %s1b1gone, %s1b1gt : tensor<32x768x28x28xf32>
    %s1b1ghalf = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %s1b1ghx = stablehlo.multiply %s1b1ghalf, %s1b1e : tensor<32x768x28x28xf32>
    %s1b1g = stablehlo.multiply %s1b1ghx, %s1b1gopt : tensor<32x768x28x28xf32>
    %s1b1pc = stablehlo.convolution(%s1b1g, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %s1b1pbb = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b1p = stablehlo.add %s1b1pc, %s1b1pbb : tensor<32x192x28x28xf32>
    %s1b1lsgb = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b1ls = stablehlo.multiply %s1b1p, %s1b1lsgb : tensor<32x192x28x28xf32>
    %s1b1o = stablehlo.add %s1b1ls, %s1b0o : tensor<32x192x28x28xf32>
    %s1b2dc = stablehlo.convolution(%s1b1o, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %s1b2dbb = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b2d = stablehlo.add %s1b2dc, %s1b2dbb : tensor<32x192x28x28xf32>
    %s1b2nri = stablehlo.reshape %s1b2d : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b2nnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %s1b2nep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %s1b2nsmr = stablehlo.reduce(%s1b2nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b2nsm = stablehlo.broadcast_in_dim %s1b2nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b2nmu = stablehlo.divide %s1b2nsm, %s1b2nnf : tensor<32x150528xf32>
    %s1b2nxc = stablehlo.subtract %s1b2nri, %s1b2nmu : tensor<32x150528xf32>
    %s1b2nsq = stablehlo.multiply %s1b2nxc, %s1b2nxc : tensor<32x150528xf32>
    %s1b2nvsr = stablehlo.reduce(%s1b2nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b2nvs = stablehlo.broadcast_in_dim %s1b2nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b2nvr = stablehlo.divide %s1b2nvs, %s1b2nnf : tensor<32x150528xf32>
    %s1b2nve = stablehlo.add %s1b2nvr, %s1b2nep : tensor<32x150528xf32>
    %s1b2nistd = stablehlo.rsqrt %s1b2nve : tensor<32x150528xf32>
    %s1b2nxh = stablehlo.multiply %s1b2nxc, %s1b2nistd : tensor<32x150528xf32>
    %s1b2ngb = stablehlo.broadcast_in_dim %s1b2ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b2nbtb = stablehlo.broadcast_in_dim %s1b2nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b2ngx = stablehlo.multiply %s1b2nxh, %s1b2ngb : tensor<32x150528xf32>
    %s1b2nfl = stablehlo.add %s1b2ngx, %s1b2nbtb : tensor<32x150528xf32>
    %s1b2n = stablehlo.reshape %s1b2nfl : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %s1b2ec = stablehlo.convolution(%s1b2n, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %s1b2ebb = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %s1b2e = stablehlo.add %s1b2ec, %s1b2ebb : tensor<32x768x28x28xf32>
    %s1b2gx2 = stablehlo.multiply %s1b2e, %s1b2e : tensor<32x768x28x28xf32>
    %s1b2gx3 = stablehlo.multiply %s1b2gx2, %s1b2e : tensor<32x768x28x28xf32>
    %s1b2gck = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %s1b2gkx3 = stablehlo.multiply %s1b2gck, %s1b2gx3 : tensor<32x768x28x28xf32>
    %s1b2ginn = stablehlo.add %s1b2e, %s1b2gkx3 : tensor<32x768x28x28xf32>
    %s1b2gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %s1b2gu = stablehlo.multiply %s1b2gcs, %s1b2ginn : tensor<32x768x28x28xf32>
    %s1b2gt = stablehlo.tanh %s1b2gu : tensor<32x768x28x28xf32>
    %s1b2gone = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %s1b2gopt = stablehlo.add %s1b2gone, %s1b2gt : tensor<32x768x28x28xf32>
    %s1b2ghalf = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %s1b2ghx = stablehlo.multiply %s1b2ghalf, %s1b2e : tensor<32x768x28x28xf32>
    %s1b2g = stablehlo.multiply %s1b2ghx, %s1b2gopt : tensor<32x768x28x28xf32>
    %s1b2pc = stablehlo.convolution(%s1b2g, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %s1b2pbb = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b2p = stablehlo.add %s1b2pc, %s1b2pbb : tensor<32x192x28x28xf32>
    %s1b2lsgb = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b2ls = stablehlo.multiply %s1b2p, %s1b2lsgb : tensor<32x192x28x28xf32>
    %s1b2o = stablehlo.add %s1b2ls, %s1b1o : tensor<32x192x28x28xf32>
    %d1nri = stablehlo.reshape %s1b2o : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %d1nnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %d1nep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %d1nsmr = stablehlo.reduce(%d1nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %d1nsm = stablehlo.broadcast_in_dim %d1nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %d1nmu = stablehlo.divide %d1nsm, %d1nnf : tensor<32x150528xf32>
    %d1nxc = stablehlo.subtract %d1nri, %d1nmu : tensor<32x150528xf32>
    %d1nsq = stablehlo.multiply %d1nxc, %d1nxc : tensor<32x150528xf32>
    %d1nvsr = stablehlo.reduce(%d1nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %d1nvs = stablehlo.broadcast_in_dim %d1nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %d1nvr = stablehlo.divide %d1nvs, %d1nnf : tensor<32x150528xf32>
    %d1nve = stablehlo.add %d1nvr, %d1nep : tensor<32x150528xf32>
    %d1nistd = stablehlo.rsqrt %d1nve : tensor<32x150528xf32>
    %d1nxh = stablehlo.multiply %d1nxc, %d1nistd : tensor<32x150528xf32>
    %d1ngb = stablehlo.broadcast_in_dim %d1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %d1nbtb = stablehlo.broadcast_in_dim %d1nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %d1ngx = stablehlo.multiply %d1nxh, %d1ngb : tensor<32x150528xf32>
    %d1nfl = stablehlo.add %d1ngx, %d1nbtb : tensor<32x150528xf32>
    %d1n = stablehlo.reshape %d1nfl : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %d1cc = stablehlo.convolution(%d1n, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %d1cbb = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %d1c = stablehlo.add %d1cc, %d1cbb : tensor<32x384x14x14xf32>
    %s2b0dc = stablehlo.convolution(%d1c, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b0dbb = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b0d = stablehlo.add %s2b0dc, %s2b0dbb : tensor<32x384x14x14xf32>
    %s2b0nri = stablehlo.reshape %s2b0d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b0nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b0nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b0nsmr = stablehlo.reduce(%s2b0nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b0nsm = stablehlo.broadcast_in_dim %s2b0nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b0nmu = stablehlo.divide %s2b0nsm, %s2b0nnf : tensor<32x75264xf32>
    %s2b0nxc = stablehlo.subtract %s2b0nri, %s2b0nmu : tensor<32x75264xf32>
    %s2b0nsq = stablehlo.multiply %s2b0nxc, %s2b0nxc : tensor<32x75264xf32>
    %s2b0nvsr = stablehlo.reduce(%s2b0nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b0nvs = stablehlo.broadcast_in_dim %s2b0nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b0nvr = stablehlo.divide %s2b0nvs, %s2b0nnf : tensor<32x75264xf32>
    %s2b0nve = stablehlo.add %s2b0nvr, %s2b0nep : tensor<32x75264xf32>
    %s2b0nistd = stablehlo.rsqrt %s2b0nve : tensor<32x75264xf32>
    %s2b0nxh = stablehlo.multiply %s2b0nxc, %s2b0nistd : tensor<32x75264xf32>
    %s2b0ngb = stablehlo.broadcast_in_dim %s2b0ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b0nbtb = stablehlo.broadcast_in_dim %s2b0nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b0ngx = stablehlo.multiply %s2b0nxh, %s2b0ngb : tensor<32x75264xf32>
    %s2b0nfl = stablehlo.add %s2b0ngx, %s2b0nbtb : tensor<32x75264xf32>
    %s2b0n = stablehlo.reshape %s2b0nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b0ec = stablehlo.convolution(%s2b0n, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b0ebb = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b0e = stablehlo.add %s2b0ec, %s2b0ebb : tensor<32x1536x14x14xf32>
    %s2b0gx2 = stablehlo.multiply %s2b0e, %s2b0e : tensor<32x1536x14x14xf32>
    %s2b0gx3 = stablehlo.multiply %s2b0gx2, %s2b0e : tensor<32x1536x14x14xf32>
    %s2b0gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b0gkx3 = stablehlo.multiply %s2b0gck, %s2b0gx3 : tensor<32x1536x14x14xf32>
    %s2b0ginn = stablehlo.add %s2b0e, %s2b0gkx3 : tensor<32x1536x14x14xf32>
    %s2b0gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b0gu = stablehlo.multiply %s2b0gcs, %s2b0ginn : tensor<32x1536x14x14xf32>
    %s2b0gt = stablehlo.tanh %s2b0gu : tensor<32x1536x14x14xf32>
    %s2b0gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b0gopt = stablehlo.add %s2b0gone, %s2b0gt : tensor<32x1536x14x14xf32>
    %s2b0ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b0ghx = stablehlo.multiply %s2b0ghalf, %s2b0e : tensor<32x1536x14x14xf32>
    %s2b0g = stablehlo.multiply %s2b0ghx, %s2b0gopt : tensor<32x1536x14x14xf32>
    %s2b0pc = stablehlo.convolution(%s2b0g, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b0pbb = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b0p = stablehlo.add %s2b0pc, %s2b0pbb : tensor<32x384x14x14xf32>
    %s2b0lsgb = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b0ls = stablehlo.multiply %s2b0p, %s2b0lsgb : tensor<32x384x14x14xf32>
    %s2b0o = stablehlo.add %s2b0ls, %d1c : tensor<32x384x14x14xf32>
    %s2b1dc = stablehlo.convolution(%s2b0o, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b1dbb = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b1d = stablehlo.add %s2b1dc, %s2b1dbb : tensor<32x384x14x14xf32>
    %s2b1nri = stablehlo.reshape %s2b1d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b1nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b1nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b1nsmr = stablehlo.reduce(%s2b1nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b1nsm = stablehlo.broadcast_in_dim %s2b1nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b1nmu = stablehlo.divide %s2b1nsm, %s2b1nnf : tensor<32x75264xf32>
    %s2b1nxc = stablehlo.subtract %s2b1nri, %s2b1nmu : tensor<32x75264xf32>
    %s2b1nsq = stablehlo.multiply %s2b1nxc, %s2b1nxc : tensor<32x75264xf32>
    %s2b1nvsr = stablehlo.reduce(%s2b1nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b1nvs = stablehlo.broadcast_in_dim %s2b1nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b1nvr = stablehlo.divide %s2b1nvs, %s2b1nnf : tensor<32x75264xf32>
    %s2b1nve = stablehlo.add %s2b1nvr, %s2b1nep : tensor<32x75264xf32>
    %s2b1nistd = stablehlo.rsqrt %s2b1nve : tensor<32x75264xf32>
    %s2b1nxh = stablehlo.multiply %s2b1nxc, %s2b1nistd : tensor<32x75264xf32>
    %s2b1ngb = stablehlo.broadcast_in_dim %s2b1ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b1nbtb = stablehlo.broadcast_in_dim %s2b1nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b1ngx = stablehlo.multiply %s2b1nxh, %s2b1ngb : tensor<32x75264xf32>
    %s2b1nfl = stablehlo.add %s2b1ngx, %s2b1nbtb : tensor<32x75264xf32>
    %s2b1n = stablehlo.reshape %s2b1nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b1ec = stablehlo.convolution(%s2b1n, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b1ebb = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b1e = stablehlo.add %s2b1ec, %s2b1ebb : tensor<32x1536x14x14xf32>
    %s2b1gx2 = stablehlo.multiply %s2b1e, %s2b1e : tensor<32x1536x14x14xf32>
    %s2b1gx3 = stablehlo.multiply %s2b1gx2, %s2b1e : tensor<32x1536x14x14xf32>
    %s2b1gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b1gkx3 = stablehlo.multiply %s2b1gck, %s2b1gx3 : tensor<32x1536x14x14xf32>
    %s2b1ginn = stablehlo.add %s2b1e, %s2b1gkx3 : tensor<32x1536x14x14xf32>
    %s2b1gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b1gu = stablehlo.multiply %s2b1gcs, %s2b1ginn : tensor<32x1536x14x14xf32>
    %s2b1gt = stablehlo.tanh %s2b1gu : tensor<32x1536x14x14xf32>
    %s2b1gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b1gopt = stablehlo.add %s2b1gone, %s2b1gt : tensor<32x1536x14x14xf32>
    %s2b1ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b1ghx = stablehlo.multiply %s2b1ghalf, %s2b1e : tensor<32x1536x14x14xf32>
    %s2b1g = stablehlo.multiply %s2b1ghx, %s2b1gopt : tensor<32x1536x14x14xf32>
    %s2b1pc = stablehlo.convolution(%s2b1g, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b1pbb = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b1p = stablehlo.add %s2b1pc, %s2b1pbb : tensor<32x384x14x14xf32>
    %s2b1lsgb = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b1ls = stablehlo.multiply %s2b1p, %s2b1lsgb : tensor<32x384x14x14xf32>
    %s2b1o = stablehlo.add %s2b1ls, %s2b0o : tensor<32x384x14x14xf32>
    %s2b2dc = stablehlo.convolution(%s2b1o, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b2dbb = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b2d = stablehlo.add %s2b2dc, %s2b2dbb : tensor<32x384x14x14xf32>
    %s2b2nri = stablehlo.reshape %s2b2d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b2nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b2nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b2nsmr = stablehlo.reduce(%s2b2nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b2nsm = stablehlo.broadcast_in_dim %s2b2nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b2nmu = stablehlo.divide %s2b2nsm, %s2b2nnf : tensor<32x75264xf32>
    %s2b2nxc = stablehlo.subtract %s2b2nri, %s2b2nmu : tensor<32x75264xf32>
    %s2b2nsq = stablehlo.multiply %s2b2nxc, %s2b2nxc : tensor<32x75264xf32>
    %s2b2nvsr = stablehlo.reduce(%s2b2nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b2nvs = stablehlo.broadcast_in_dim %s2b2nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b2nvr = stablehlo.divide %s2b2nvs, %s2b2nnf : tensor<32x75264xf32>
    %s2b2nve = stablehlo.add %s2b2nvr, %s2b2nep : tensor<32x75264xf32>
    %s2b2nistd = stablehlo.rsqrt %s2b2nve : tensor<32x75264xf32>
    %s2b2nxh = stablehlo.multiply %s2b2nxc, %s2b2nistd : tensor<32x75264xf32>
    %s2b2ngb = stablehlo.broadcast_in_dim %s2b2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b2nbtb = stablehlo.broadcast_in_dim %s2b2nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b2ngx = stablehlo.multiply %s2b2nxh, %s2b2ngb : tensor<32x75264xf32>
    %s2b2nfl = stablehlo.add %s2b2ngx, %s2b2nbtb : tensor<32x75264xf32>
    %s2b2n = stablehlo.reshape %s2b2nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b2ec = stablehlo.convolution(%s2b2n, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b2ebb = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b2e = stablehlo.add %s2b2ec, %s2b2ebb : tensor<32x1536x14x14xf32>
    %s2b2gx2 = stablehlo.multiply %s2b2e, %s2b2e : tensor<32x1536x14x14xf32>
    %s2b2gx3 = stablehlo.multiply %s2b2gx2, %s2b2e : tensor<32x1536x14x14xf32>
    %s2b2gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b2gkx3 = stablehlo.multiply %s2b2gck, %s2b2gx3 : tensor<32x1536x14x14xf32>
    %s2b2ginn = stablehlo.add %s2b2e, %s2b2gkx3 : tensor<32x1536x14x14xf32>
    %s2b2gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b2gu = stablehlo.multiply %s2b2gcs, %s2b2ginn : tensor<32x1536x14x14xf32>
    %s2b2gt = stablehlo.tanh %s2b2gu : tensor<32x1536x14x14xf32>
    %s2b2gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b2gopt = stablehlo.add %s2b2gone, %s2b2gt : tensor<32x1536x14x14xf32>
    %s2b2ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b2ghx = stablehlo.multiply %s2b2ghalf, %s2b2e : tensor<32x1536x14x14xf32>
    %s2b2g = stablehlo.multiply %s2b2ghx, %s2b2gopt : tensor<32x1536x14x14xf32>
    %s2b2pc = stablehlo.convolution(%s2b2g, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b2pbb = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b2p = stablehlo.add %s2b2pc, %s2b2pbb : tensor<32x384x14x14xf32>
    %s2b2lsgb = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b2ls = stablehlo.multiply %s2b2p, %s2b2lsgb : tensor<32x384x14x14xf32>
    %s2b2o = stablehlo.add %s2b2ls, %s2b1o : tensor<32x384x14x14xf32>
    %s2b3dc = stablehlo.convolution(%s2b2o, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b3dbb = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b3d = stablehlo.add %s2b3dc, %s2b3dbb : tensor<32x384x14x14xf32>
    %s2b3nri = stablehlo.reshape %s2b3d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b3nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b3nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b3nsmr = stablehlo.reduce(%s2b3nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b3nsm = stablehlo.broadcast_in_dim %s2b3nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b3nmu = stablehlo.divide %s2b3nsm, %s2b3nnf : tensor<32x75264xf32>
    %s2b3nxc = stablehlo.subtract %s2b3nri, %s2b3nmu : tensor<32x75264xf32>
    %s2b3nsq = stablehlo.multiply %s2b3nxc, %s2b3nxc : tensor<32x75264xf32>
    %s2b3nvsr = stablehlo.reduce(%s2b3nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b3nvs = stablehlo.broadcast_in_dim %s2b3nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b3nvr = stablehlo.divide %s2b3nvs, %s2b3nnf : tensor<32x75264xf32>
    %s2b3nve = stablehlo.add %s2b3nvr, %s2b3nep : tensor<32x75264xf32>
    %s2b3nistd = stablehlo.rsqrt %s2b3nve : tensor<32x75264xf32>
    %s2b3nxh = stablehlo.multiply %s2b3nxc, %s2b3nistd : tensor<32x75264xf32>
    %s2b3ngb = stablehlo.broadcast_in_dim %s2b3ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b3nbtb = stablehlo.broadcast_in_dim %s2b3nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b3ngx = stablehlo.multiply %s2b3nxh, %s2b3ngb : tensor<32x75264xf32>
    %s2b3nfl = stablehlo.add %s2b3ngx, %s2b3nbtb : tensor<32x75264xf32>
    %s2b3n = stablehlo.reshape %s2b3nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b3ec = stablehlo.convolution(%s2b3n, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b3ebb = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b3e = stablehlo.add %s2b3ec, %s2b3ebb : tensor<32x1536x14x14xf32>
    %s2b3gx2 = stablehlo.multiply %s2b3e, %s2b3e : tensor<32x1536x14x14xf32>
    %s2b3gx3 = stablehlo.multiply %s2b3gx2, %s2b3e : tensor<32x1536x14x14xf32>
    %s2b3gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b3gkx3 = stablehlo.multiply %s2b3gck, %s2b3gx3 : tensor<32x1536x14x14xf32>
    %s2b3ginn = stablehlo.add %s2b3e, %s2b3gkx3 : tensor<32x1536x14x14xf32>
    %s2b3gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b3gu = stablehlo.multiply %s2b3gcs, %s2b3ginn : tensor<32x1536x14x14xf32>
    %s2b3gt = stablehlo.tanh %s2b3gu : tensor<32x1536x14x14xf32>
    %s2b3gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b3gopt = stablehlo.add %s2b3gone, %s2b3gt : tensor<32x1536x14x14xf32>
    %s2b3ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b3ghx = stablehlo.multiply %s2b3ghalf, %s2b3e : tensor<32x1536x14x14xf32>
    %s2b3g = stablehlo.multiply %s2b3ghx, %s2b3gopt : tensor<32x1536x14x14xf32>
    %s2b3pc = stablehlo.convolution(%s2b3g, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b3pbb = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b3p = stablehlo.add %s2b3pc, %s2b3pbb : tensor<32x384x14x14xf32>
    %s2b3lsgb = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b3ls = stablehlo.multiply %s2b3p, %s2b3lsgb : tensor<32x384x14x14xf32>
    %s2b3o = stablehlo.add %s2b3ls, %s2b2o : tensor<32x384x14x14xf32>
    %s2b4dc = stablehlo.convolution(%s2b3o, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b4dbb = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b4d = stablehlo.add %s2b4dc, %s2b4dbb : tensor<32x384x14x14xf32>
    %s2b4nri = stablehlo.reshape %s2b4d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b4nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b4nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b4nsmr = stablehlo.reduce(%s2b4nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b4nsm = stablehlo.broadcast_in_dim %s2b4nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b4nmu = stablehlo.divide %s2b4nsm, %s2b4nnf : tensor<32x75264xf32>
    %s2b4nxc = stablehlo.subtract %s2b4nri, %s2b4nmu : tensor<32x75264xf32>
    %s2b4nsq = stablehlo.multiply %s2b4nxc, %s2b4nxc : tensor<32x75264xf32>
    %s2b4nvsr = stablehlo.reduce(%s2b4nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b4nvs = stablehlo.broadcast_in_dim %s2b4nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b4nvr = stablehlo.divide %s2b4nvs, %s2b4nnf : tensor<32x75264xf32>
    %s2b4nve = stablehlo.add %s2b4nvr, %s2b4nep : tensor<32x75264xf32>
    %s2b4nistd = stablehlo.rsqrt %s2b4nve : tensor<32x75264xf32>
    %s2b4nxh = stablehlo.multiply %s2b4nxc, %s2b4nistd : tensor<32x75264xf32>
    %s2b4ngb = stablehlo.broadcast_in_dim %s2b4ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b4nbtb = stablehlo.broadcast_in_dim %s2b4nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b4ngx = stablehlo.multiply %s2b4nxh, %s2b4ngb : tensor<32x75264xf32>
    %s2b4nfl = stablehlo.add %s2b4ngx, %s2b4nbtb : tensor<32x75264xf32>
    %s2b4n = stablehlo.reshape %s2b4nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b4ec = stablehlo.convolution(%s2b4n, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b4ebb = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b4e = stablehlo.add %s2b4ec, %s2b4ebb : tensor<32x1536x14x14xf32>
    %s2b4gx2 = stablehlo.multiply %s2b4e, %s2b4e : tensor<32x1536x14x14xf32>
    %s2b4gx3 = stablehlo.multiply %s2b4gx2, %s2b4e : tensor<32x1536x14x14xf32>
    %s2b4gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b4gkx3 = stablehlo.multiply %s2b4gck, %s2b4gx3 : tensor<32x1536x14x14xf32>
    %s2b4ginn = stablehlo.add %s2b4e, %s2b4gkx3 : tensor<32x1536x14x14xf32>
    %s2b4gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b4gu = stablehlo.multiply %s2b4gcs, %s2b4ginn : tensor<32x1536x14x14xf32>
    %s2b4gt = stablehlo.tanh %s2b4gu : tensor<32x1536x14x14xf32>
    %s2b4gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b4gopt = stablehlo.add %s2b4gone, %s2b4gt : tensor<32x1536x14x14xf32>
    %s2b4ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b4ghx = stablehlo.multiply %s2b4ghalf, %s2b4e : tensor<32x1536x14x14xf32>
    %s2b4g = stablehlo.multiply %s2b4ghx, %s2b4gopt : tensor<32x1536x14x14xf32>
    %s2b4pc = stablehlo.convolution(%s2b4g, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b4pbb = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b4p = stablehlo.add %s2b4pc, %s2b4pbb : tensor<32x384x14x14xf32>
    %s2b4lsgb = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b4ls = stablehlo.multiply %s2b4p, %s2b4lsgb : tensor<32x384x14x14xf32>
    %s2b4o = stablehlo.add %s2b4ls, %s2b3o : tensor<32x384x14x14xf32>
    %s2b5dc = stablehlo.convolution(%s2b4o, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b5dbb = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b5d = stablehlo.add %s2b5dc, %s2b5dbb : tensor<32x384x14x14xf32>
    %s2b5nri = stablehlo.reshape %s2b5d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b5nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b5nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b5nsmr = stablehlo.reduce(%s2b5nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b5nsm = stablehlo.broadcast_in_dim %s2b5nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b5nmu = stablehlo.divide %s2b5nsm, %s2b5nnf : tensor<32x75264xf32>
    %s2b5nxc = stablehlo.subtract %s2b5nri, %s2b5nmu : tensor<32x75264xf32>
    %s2b5nsq = stablehlo.multiply %s2b5nxc, %s2b5nxc : tensor<32x75264xf32>
    %s2b5nvsr = stablehlo.reduce(%s2b5nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b5nvs = stablehlo.broadcast_in_dim %s2b5nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b5nvr = stablehlo.divide %s2b5nvs, %s2b5nnf : tensor<32x75264xf32>
    %s2b5nve = stablehlo.add %s2b5nvr, %s2b5nep : tensor<32x75264xf32>
    %s2b5nistd = stablehlo.rsqrt %s2b5nve : tensor<32x75264xf32>
    %s2b5nxh = stablehlo.multiply %s2b5nxc, %s2b5nistd : tensor<32x75264xf32>
    %s2b5ngb = stablehlo.broadcast_in_dim %s2b5ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b5nbtb = stablehlo.broadcast_in_dim %s2b5nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b5ngx = stablehlo.multiply %s2b5nxh, %s2b5ngb : tensor<32x75264xf32>
    %s2b5nfl = stablehlo.add %s2b5ngx, %s2b5nbtb : tensor<32x75264xf32>
    %s2b5n = stablehlo.reshape %s2b5nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b5ec = stablehlo.convolution(%s2b5n, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b5ebb = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b5e = stablehlo.add %s2b5ec, %s2b5ebb : tensor<32x1536x14x14xf32>
    %s2b5gx2 = stablehlo.multiply %s2b5e, %s2b5e : tensor<32x1536x14x14xf32>
    %s2b5gx3 = stablehlo.multiply %s2b5gx2, %s2b5e : tensor<32x1536x14x14xf32>
    %s2b5gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b5gkx3 = stablehlo.multiply %s2b5gck, %s2b5gx3 : tensor<32x1536x14x14xf32>
    %s2b5ginn = stablehlo.add %s2b5e, %s2b5gkx3 : tensor<32x1536x14x14xf32>
    %s2b5gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b5gu = stablehlo.multiply %s2b5gcs, %s2b5ginn : tensor<32x1536x14x14xf32>
    %s2b5gt = stablehlo.tanh %s2b5gu : tensor<32x1536x14x14xf32>
    %s2b5gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b5gopt = stablehlo.add %s2b5gone, %s2b5gt : tensor<32x1536x14x14xf32>
    %s2b5ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b5ghx = stablehlo.multiply %s2b5ghalf, %s2b5e : tensor<32x1536x14x14xf32>
    %s2b5g = stablehlo.multiply %s2b5ghx, %s2b5gopt : tensor<32x1536x14x14xf32>
    %s2b5pc = stablehlo.convolution(%s2b5g, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b5pbb = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b5p = stablehlo.add %s2b5pc, %s2b5pbb : tensor<32x384x14x14xf32>
    %s2b5lsgb = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b5ls = stablehlo.multiply %s2b5p, %s2b5lsgb : tensor<32x384x14x14xf32>
    %s2b5o = stablehlo.add %s2b5ls, %s2b4o : tensor<32x384x14x14xf32>
    %s2b6dc = stablehlo.convolution(%s2b5o, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b6dbb = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b6d = stablehlo.add %s2b6dc, %s2b6dbb : tensor<32x384x14x14xf32>
    %s2b6nri = stablehlo.reshape %s2b6d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b6nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b6nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b6nsmr = stablehlo.reduce(%s2b6nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b6nsm = stablehlo.broadcast_in_dim %s2b6nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b6nmu = stablehlo.divide %s2b6nsm, %s2b6nnf : tensor<32x75264xf32>
    %s2b6nxc = stablehlo.subtract %s2b6nri, %s2b6nmu : tensor<32x75264xf32>
    %s2b6nsq = stablehlo.multiply %s2b6nxc, %s2b6nxc : tensor<32x75264xf32>
    %s2b6nvsr = stablehlo.reduce(%s2b6nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b6nvs = stablehlo.broadcast_in_dim %s2b6nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b6nvr = stablehlo.divide %s2b6nvs, %s2b6nnf : tensor<32x75264xf32>
    %s2b6nve = stablehlo.add %s2b6nvr, %s2b6nep : tensor<32x75264xf32>
    %s2b6nistd = stablehlo.rsqrt %s2b6nve : tensor<32x75264xf32>
    %s2b6nxh = stablehlo.multiply %s2b6nxc, %s2b6nistd : tensor<32x75264xf32>
    %s2b6ngb = stablehlo.broadcast_in_dim %s2b6ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b6nbtb = stablehlo.broadcast_in_dim %s2b6nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b6ngx = stablehlo.multiply %s2b6nxh, %s2b6ngb : tensor<32x75264xf32>
    %s2b6nfl = stablehlo.add %s2b6ngx, %s2b6nbtb : tensor<32x75264xf32>
    %s2b6n = stablehlo.reshape %s2b6nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b6ec = stablehlo.convolution(%s2b6n, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b6ebb = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b6e = stablehlo.add %s2b6ec, %s2b6ebb : tensor<32x1536x14x14xf32>
    %s2b6gx2 = stablehlo.multiply %s2b6e, %s2b6e : tensor<32x1536x14x14xf32>
    %s2b6gx3 = stablehlo.multiply %s2b6gx2, %s2b6e : tensor<32x1536x14x14xf32>
    %s2b6gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b6gkx3 = stablehlo.multiply %s2b6gck, %s2b6gx3 : tensor<32x1536x14x14xf32>
    %s2b6ginn = stablehlo.add %s2b6e, %s2b6gkx3 : tensor<32x1536x14x14xf32>
    %s2b6gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b6gu = stablehlo.multiply %s2b6gcs, %s2b6ginn : tensor<32x1536x14x14xf32>
    %s2b6gt = stablehlo.tanh %s2b6gu : tensor<32x1536x14x14xf32>
    %s2b6gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b6gopt = stablehlo.add %s2b6gone, %s2b6gt : tensor<32x1536x14x14xf32>
    %s2b6ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b6ghx = stablehlo.multiply %s2b6ghalf, %s2b6e : tensor<32x1536x14x14xf32>
    %s2b6g = stablehlo.multiply %s2b6ghx, %s2b6gopt : tensor<32x1536x14x14xf32>
    %s2b6pc = stablehlo.convolution(%s2b6g, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b6pbb = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b6p = stablehlo.add %s2b6pc, %s2b6pbb : tensor<32x384x14x14xf32>
    %s2b6lsgb = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b6ls = stablehlo.multiply %s2b6p, %s2b6lsgb : tensor<32x384x14x14xf32>
    %s2b6o = stablehlo.add %s2b6ls, %s2b5o : tensor<32x384x14x14xf32>
    %s2b7dc = stablehlo.convolution(%s2b6o, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b7dbb = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b7d = stablehlo.add %s2b7dc, %s2b7dbb : tensor<32x384x14x14xf32>
    %s2b7nri = stablehlo.reshape %s2b7d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b7nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b7nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b7nsmr = stablehlo.reduce(%s2b7nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b7nsm = stablehlo.broadcast_in_dim %s2b7nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b7nmu = stablehlo.divide %s2b7nsm, %s2b7nnf : tensor<32x75264xf32>
    %s2b7nxc = stablehlo.subtract %s2b7nri, %s2b7nmu : tensor<32x75264xf32>
    %s2b7nsq = stablehlo.multiply %s2b7nxc, %s2b7nxc : tensor<32x75264xf32>
    %s2b7nvsr = stablehlo.reduce(%s2b7nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b7nvs = stablehlo.broadcast_in_dim %s2b7nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b7nvr = stablehlo.divide %s2b7nvs, %s2b7nnf : tensor<32x75264xf32>
    %s2b7nve = stablehlo.add %s2b7nvr, %s2b7nep : tensor<32x75264xf32>
    %s2b7nistd = stablehlo.rsqrt %s2b7nve : tensor<32x75264xf32>
    %s2b7nxh = stablehlo.multiply %s2b7nxc, %s2b7nistd : tensor<32x75264xf32>
    %s2b7ngb = stablehlo.broadcast_in_dim %s2b7ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b7nbtb = stablehlo.broadcast_in_dim %s2b7nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b7ngx = stablehlo.multiply %s2b7nxh, %s2b7ngb : tensor<32x75264xf32>
    %s2b7nfl = stablehlo.add %s2b7ngx, %s2b7nbtb : tensor<32x75264xf32>
    %s2b7n = stablehlo.reshape %s2b7nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b7ec = stablehlo.convolution(%s2b7n, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b7ebb = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b7e = stablehlo.add %s2b7ec, %s2b7ebb : tensor<32x1536x14x14xf32>
    %s2b7gx2 = stablehlo.multiply %s2b7e, %s2b7e : tensor<32x1536x14x14xf32>
    %s2b7gx3 = stablehlo.multiply %s2b7gx2, %s2b7e : tensor<32x1536x14x14xf32>
    %s2b7gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b7gkx3 = stablehlo.multiply %s2b7gck, %s2b7gx3 : tensor<32x1536x14x14xf32>
    %s2b7ginn = stablehlo.add %s2b7e, %s2b7gkx3 : tensor<32x1536x14x14xf32>
    %s2b7gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b7gu = stablehlo.multiply %s2b7gcs, %s2b7ginn : tensor<32x1536x14x14xf32>
    %s2b7gt = stablehlo.tanh %s2b7gu : tensor<32x1536x14x14xf32>
    %s2b7gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b7gopt = stablehlo.add %s2b7gone, %s2b7gt : tensor<32x1536x14x14xf32>
    %s2b7ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b7ghx = stablehlo.multiply %s2b7ghalf, %s2b7e : tensor<32x1536x14x14xf32>
    %s2b7g = stablehlo.multiply %s2b7ghx, %s2b7gopt : tensor<32x1536x14x14xf32>
    %s2b7pc = stablehlo.convolution(%s2b7g, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b7pbb = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b7p = stablehlo.add %s2b7pc, %s2b7pbb : tensor<32x384x14x14xf32>
    %s2b7lsgb = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b7ls = stablehlo.multiply %s2b7p, %s2b7lsgb : tensor<32x384x14x14xf32>
    %s2b7o = stablehlo.add %s2b7ls, %s2b6o : tensor<32x384x14x14xf32>
    %s2b8dc = stablehlo.convolution(%s2b7o, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b8dbb = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b8d = stablehlo.add %s2b8dc, %s2b8dbb : tensor<32x384x14x14xf32>
    %s2b8nri = stablehlo.reshape %s2b8d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b8nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b8nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b8nsmr = stablehlo.reduce(%s2b8nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b8nsm = stablehlo.broadcast_in_dim %s2b8nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b8nmu = stablehlo.divide %s2b8nsm, %s2b8nnf : tensor<32x75264xf32>
    %s2b8nxc = stablehlo.subtract %s2b8nri, %s2b8nmu : tensor<32x75264xf32>
    %s2b8nsq = stablehlo.multiply %s2b8nxc, %s2b8nxc : tensor<32x75264xf32>
    %s2b8nvsr = stablehlo.reduce(%s2b8nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b8nvs = stablehlo.broadcast_in_dim %s2b8nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b8nvr = stablehlo.divide %s2b8nvs, %s2b8nnf : tensor<32x75264xf32>
    %s2b8nve = stablehlo.add %s2b8nvr, %s2b8nep : tensor<32x75264xf32>
    %s2b8nistd = stablehlo.rsqrt %s2b8nve : tensor<32x75264xf32>
    %s2b8nxh = stablehlo.multiply %s2b8nxc, %s2b8nistd : tensor<32x75264xf32>
    %s2b8ngb = stablehlo.broadcast_in_dim %s2b8ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b8nbtb = stablehlo.broadcast_in_dim %s2b8nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b8ngx = stablehlo.multiply %s2b8nxh, %s2b8ngb : tensor<32x75264xf32>
    %s2b8nfl = stablehlo.add %s2b8ngx, %s2b8nbtb : tensor<32x75264xf32>
    %s2b8n = stablehlo.reshape %s2b8nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b8ec = stablehlo.convolution(%s2b8n, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b8ebb = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b8e = stablehlo.add %s2b8ec, %s2b8ebb : tensor<32x1536x14x14xf32>
    %s2b8gx2 = stablehlo.multiply %s2b8e, %s2b8e : tensor<32x1536x14x14xf32>
    %s2b8gx3 = stablehlo.multiply %s2b8gx2, %s2b8e : tensor<32x1536x14x14xf32>
    %s2b8gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b8gkx3 = stablehlo.multiply %s2b8gck, %s2b8gx3 : tensor<32x1536x14x14xf32>
    %s2b8ginn = stablehlo.add %s2b8e, %s2b8gkx3 : tensor<32x1536x14x14xf32>
    %s2b8gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b8gu = stablehlo.multiply %s2b8gcs, %s2b8ginn : tensor<32x1536x14x14xf32>
    %s2b8gt = stablehlo.tanh %s2b8gu : tensor<32x1536x14x14xf32>
    %s2b8gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b8gopt = stablehlo.add %s2b8gone, %s2b8gt : tensor<32x1536x14x14xf32>
    %s2b8ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b8ghx = stablehlo.multiply %s2b8ghalf, %s2b8e : tensor<32x1536x14x14xf32>
    %s2b8g = stablehlo.multiply %s2b8ghx, %s2b8gopt : tensor<32x1536x14x14xf32>
    %s2b8pc = stablehlo.convolution(%s2b8g, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b8pbb = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b8p = stablehlo.add %s2b8pc, %s2b8pbb : tensor<32x384x14x14xf32>
    %s2b8lsgb = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b8ls = stablehlo.multiply %s2b8p, %s2b8lsgb : tensor<32x384x14x14xf32>
    %s2b8o = stablehlo.add %s2b8ls, %s2b7o : tensor<32x384x14x14xf32>
    %d2nri = stablehlo.reshape %s2b8o : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %d2nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %d2nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %d2nsmr = stablehlo.reduce(%d2nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %d2nsm = stablehlo.broadcast_in_dim %d2nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %d2nmu = stablehlo.divide %d2nsm, %d2nnf : tensor<32x75264xf32>
    %d2nxc = stablehlo.subtract %d2nri, %d2nmu : tensor<32x75264xf32>
    %d2nsq = stablehlo.multiply %d2nxc, %d2nxc : tensor<32x75264xf32>
    %d2nvsr = stablehlo.reduce(%d2nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %d2nvs = stablehlo.broadcast_in_dim %d2nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %d2nvr = stablehlo.divide %d2nvs, %d2nnf : tensor<32x75264xf32>
    %d2nve = stablehlo.add %d2nvr, %d2nep : tensor<32x75264xf32>
    %d2nistd = stablehlo.rsqrt %d2nve : tensor<32x75264xf32>
    %d2nxh = stablehlo.multiply %d2nxc, %d2nistd : tensor<32x75264xf32>
    %d2ngb = stablehlo.broadcast_in_dim %d2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %d2nbtb = stablehlo.broadcast_in_dim %d2nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %d2ngx = stablehlo.multiply %d2nxh, %d2ngb : tensor<32x75264xf32>
    %d2nfl = stablehlo.add %d2ngx, %d2nbtb : tensor<32x75264xf32>
    %d2n = stablehlo.reshape %d2nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %d2cc = stablehlo.convolution(%d2n, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %d2cbb = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %d2c = stablehlo.add %d2cc, %d2cbb : tensor<32x768x7x7xf32>
    %s3b0dc = stablehlo.convolution(%d2c, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %s3b0dbb = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b0d = stablehlo.add %s3b0dc, %s3b0dbb : tensor<32x768x7x7xf32>
    %s3b0nri = stablehlo.reshape %s3b0d : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b0nnf = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %s3b0nep = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %s3b0nsmr = stablehlo.reduce(%s3b0nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b0nsm = stablehlo.broadcast_in_dim %s3b0nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b0nmu = stablehlo.divide %s3b0nsm, %s3b0nnf : tensor<32x37632xf32>
    %s3b0nxc = stablehlo.subtract %s3b0nri, %s3b0nmu : tensor<32x37632xf32>
    %s3b0nsq = stablehlo.multiply %s3b0nxc, %s3b0nxc : tensor<32x37632xf32>
    %s3b0nvsr = stablehlo.reduce(%s3b0nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b0nvs = stablehlo.broadcast_in_dim %s3b0nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b0nvr = stablehlo.divide %s3b0nvs, %s3b0nnf : tensor<32x37632xf32>
    %s3b0nve = stablehlo.add %s3b0nvr, %s3b0nep : tensor<32x37632xf32>
    %s3b0nistd = stablehlo.rsqrt %s3b0nve : tensor<32x37632xf32>
    %s3b0nxh = stablehlo.multiply %s3b0nxc, %s3b0nistd : tensor<32x37632xf32>
    %s3b0ngb = stablehlo.broadcast_in_dim %s3b0ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b0nbtb = stablehlo.broadcast_in_dim %s3b0nbt, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b0ngx = stablehlo.multiply %s3b0nxh, %s3b0ngb : tensor<32x37632xf32>
    %s3b0nfl = stablehlo.add %s3b0ngx, %s3b0nbtb : tensor<32x37632xf32>
    %s3b0n = stablehlo.reshape %s3b0nfl : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %s3b0ec = stablehlo.convolution(%s3b0n, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %s3b0ebb = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %s3b0e = stablehlo.add %s3b0ec, %s3b0ebb : tensor<32x3072x7x7xf32>
    %s3b0gx2 = stablehlo.multiply %s3b0e, %s3b0e : tensor<32x3072x7x7xf32>
    %s3b0gx3 = stablehlo.multiply %s3b0gx2, %s3b0e : tensor<32x3072x7x7xf32>
    %s3b0gck = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %s3b0gkx3 = stablehlo.multiply %s3b0gck, %s3b0gx3 : tensor<32x3072x7x7xf32>
    %s3b0ginn = stablehlo.add %s3b0e, %s3b0gkx3 : tensor<32x3072x7x7xf32>
    %s3b0gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %s3b0gu = stablehlo.multiply %s3b0gcs, %s3b0ginn : tensor<32x3072x7x7xf32>
    %s3b0gt = stablehlo.tanh %s3b0gu : tensor<32x3072x7x7xf32>
    %s3b0gone = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %s3b0gopt = stablehlo.add %s3b0gone, %s3b0gt : tensor<32x3072x7x7xf32>
    %s3b0ghalf = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %s3b0ghx = stablehlo.multiply %s3b0ghalf, %s3b0e : tensor<32x3072x7x7xf32>
    %s3b0g = stablehlo.multiply %s3b0ghx, %s3b0gopt : tensor<32x3072x7x7xf32>
    %s3b0pc = stablehlo.convolution(%s3b0g, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %s3b0pbb = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b0p = stablehlo.add %s3b0pc, %s3b0pbb : tensor<32x768x7x7xf32>
    %s3b0lsgb = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b0ls = stablehlo.multiply %s3b0p, %s3b0lsgb : tensor<32x768x7x7xf32>
    %s3b0o = stablehlo.add %s3b0ls, %d2c : tensor<32x768x7x7xf32>
    %s3b1dc = stablehlo.convolution(%s3b0o, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %s3b1dbb = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b1d = stablehlo.add %s3b1dc, %s3b1dbb : tensor<32x768x7x7xf32>
    %s3b1nri = stablehlo.reshape %s3b1d : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b1nnf = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %s3b1nep = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %s3b1nsmr = stablehlo.reduce(%s3b1nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b1nsm = stablehlo.broadcast_in_dim %s3b1nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b1nmu = stablehlo.divide %s3b1nsm, %s3b1nnf : tensor<32x37632xf32>
    %s3b1nxc = stablehlo.subtract %s3b1nri, %s3b1nmu : tensor<32x37632xf32>
    %s3b1nsq = stablehlo.multiply %s3b1nxc, %s3b1nxc : tensor<32x37632xf32>
    %s3b1nvsr = stablehlo.reduce(%s3b1nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b1nvs = stablehlo.broadcast_in_dim %s3b1nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b1nvr = stablehlo.divide %s3b1nvs, %s3b1nnf : tensor<32x37632xf32>
    %s3b1nve = stablehlo.add %s3b1nvr, %s3b1nep : tensor<32x37632xf32>
    %s3b1nistd = stablehlo.rsqrt %s3b1nve : tensor<32x37632xf32>
    %s3b1nxh = stablehlo.multiply %s3b1nxc, %s3b1nistd : tensor<32x37632xf32>
    %s3b1ngb = stablehlo.broadcast_in_dim %s3b1ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b1nbtb = stablehlo.broadcast_in_dim %s3b1nbt, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b1ngx = stablehlo.multiply %s3b1nxh, %s3b1ngb : tensor<32x37632xf32>
    %s3b1nfl = stablehlo.add %s3b1ngx, %s3b1nbtb : tensor<32x37632xf32>
    %s3b1n = stablehlo.reshape %s3b1nfl : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %s3b1ec = stablehlo.convolution(%s3b1n, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %s3b1ebb = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %s3b1e = stablehlo.add %s3b1ec, %s3b1ebb : tensor<32x3072x7x7xf32>
    %s3b1gx2 = stablehlo.multiply %s3b1e, %s3b1e : tensor<32x3072x7x7xf32>
    %s3b1gx3 = stablehlo.multiply %s3b1gx2, %s3b1e : tensor<32x3072x7x7xf32>
    %s3b1gck = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %s3b1gkx3 = stablehlo.multiply %s3b1gck, %s3b1gx3 : tensor<32x3072x7x7xf32>
    %s3b1ginn = stablehlo.add %s3b1e, %s3b1gkx3 : tensor<32x3072x7x7xf32>
    %s3b1gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %s3b1gu = stablehlo.multiply %s3b1gcs, %s3b1ginn : tensor<32x3072x7x7xf32>
    %s3b1gt = stablehlo.tanh %s3b1gu : tensor<32x3072x7x7xf32>
    %s3b1gone = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %s3b1gopt = stablehlo.add %s3b1gone, %s3b1gt : tensor<32x3072x7x7xf32>
    %s3b1ghalf = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %s3b1ghx = stablehlo.multiply %s3b1ghalf, %s3b1e : tensor<32x3072x7x7xf32>
    %s3b1g = stablehlo.multiply %s3b1ghx, %s3b1gopt : tensor<32x3072x7x7xf32>
    %s3b1pc = stablehlo.convolution(%s3b1g, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %s3b1pbb = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b1p = stablehlo.add %s3b1pc, %s3b1pbb : tensor<32x768x7x7xf32>
    %s3b1lsgb = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b1ls = stablehlo.multiply %s3b1p, %s3b1lsgb : tensor<32x768x7x7xf32>
    %s3b1o = stablehlo.add %s3b1ls, %s3b0o : tensor<32x768x7x7xf32>
    %s3b2dc = stablehlo.convolution(%s3b1o, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %s3b2dbb = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b2d = stablehlo.add %s3b2dc, %s3b2dbb : tensor<32x768x7x7xf32>
    %s3b2nri = stablehlo.reshape %s3b2d : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b2nnf = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %s3b2nep = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %s3b2nsmr = stablehlo.reduce(%s3b2nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b2nsm = stablehlo.broadcast_in_dim %s3b2nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b2nmu = stablehlo.divide %s3b2nsm, %s3b2nnf : tensor<32x37632xf32>
    %s3b2nxc = stablehlo.subtract %s3b2nri, %s3b2nmu : tensor<32x37632xf32>
    %s3b2nsq = stablehlo.multiply %s3b2nxc, %s3b2nxc : tensor<32x37632xf32>
    %s3b2nvsr = stablehlo.reduce(%s3b2nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b2nvs = stablehlo.broadcast_in_dim %s3b2nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b2nvr = stablehlo.divide %s3b2nvs, %s3b2nnf : tensor<32x37632xf32>
    %s3b2nve = stablehlo.add %s3b2nvr, %s3b2nep : tensor<32x37632xf32>
    %s3b2nistd = stablehlo.rsqrt %s3b2nve : tensor<32x37632xf32>
    %s3b2nxh = stablehlo.multiply %s3b2nxc, %s3b2nistd : tensor<32x37632xf32>
    %s3b2ngb = stablehlo.broadcast_in_dim %s3b2ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b2nbtb = stablehlo.broadcast_in_dim %s3b2nbt, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b2ngx = stablehlo.multiply %s3b2nxh, %s3b2ngb : tensor<32x37632xf32>
    %s3b2nfl = stablehlo.add %s3b2ngx, %s3b2nbtb : tensor<32x37632xf32>
    %s3b2n = stablehlo.reshape %s3b2nfl : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %s3b2ec = stablehlo.convolution(%s3b2n, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %s3b2ebb = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %s3b2e = stablehlo.add %s3b2ec, %s3b2ebb : tensor<32x3072x7x7xf32>
    %s3b2gx2 = stablehlo.multiply %s3b2e, %s3b2e : tensor<32x3072x7x7xf32>
    %s3b2gx3 = stablehlo.multiply %s3b2gx2, %s3b2e : tensor<32x3072x7x7xf32>
    %s3b2gck = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %s3b2gkx3 = stablehlo.multiply %s3b2gck, %s3b2gx3 : tensor<32x3072x7x7xf32>
    %s3b2ginn = stablehlo.add %s3b2e, %s3b2gkx3 : tensor<32x3072x7x7xf32>
    %s3b2gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %s3b2gu = stablehlo.multiply %s3b2gcs, %s3b2ginn : tensor<32x3072x7x7xf32>
    %s3b2gt = stablehlo.tanh %s3b2gu : tensor<32x3072x7x7xf32>
    %s3b2gone = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %s3b2gopt = stablehlo.add %s3b2gone, %s3b2gt : tensor<32x3072x7x7xf32>
    %s3b2ghalf = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %s3b2ghx = stablehlo.multiply %s3b2ghalf, %s3b2e : tensor<32x3072x7x7xf32>
    %s3b2g = stablehlo.multiply %s3b2ghx, %s3b2gopt : tensor<32x3072x7x7xf32>
    %s3b2pc = stablehlo.convolution(%s3b2g, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %s3b2pbb = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b2p = stablehlo.add %s3b2pc, %s3b2pbb : tensor<32x768x7x7xf32>
    %s3b2lsgb = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b2ls = stablehlo.multiply %s3b2p, %s3b2lsgb : tensor<32x768x7x7xf32>
    %s3b2o = stablehlo.add %s3b2ls, %s3b1o : tensor<32x768x7x7xf32>
    %gaps = stablehlo.reduce(%s3b2o init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %gapnf = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %gap = stablehlo.divide %gaps, %gapnf : tensor<32x768xf32>
    %gapr = stablehlo.reshape %gap : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %hnri = stablehlo.reshape %gapr : (tensor<32x768x1x1xf32>) -> tensor<32x768xf32>
    %hnnf = stablehlo.constant dense<768.0> : tensor<32x768xf32>
    %hnep = stablehlo.constant dense<1.0e-6> : tensor<32x768xf32>
    %hnsmr = stablehlo.reduce(%hnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %hnsm = stablehlo.broadcast_in_dim %hnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %hnmu = stablehlo.divide %hnsm, %hnnf : tensor<32x768xf32>
    %hnxc = stablehlo.subtract %hnri, %hnmu : tensor<32x768xf32>
    %hnsq = stablehlo.multiply %hnxc, %hnxc : tensor<32x768xf32>
    %hnvsr = stablehlo.reduce(%hnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %hnvs = stablehlo.broadcast_in_dim %hnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %hnvr = stablehlo.divide %hnvs, %hnnf : tensor<32x768xf32>
    %hnve = stablehlo.add %hnvr, %hnep : tensor<32x768xf32>
    %hnistd = stablehlo.rsqrt %hnve : tensor<32x768xf32>
    %hnxh = stablehlo.multiply %hnxc, %hnistd : tensor<32x768xf32>
    %hngb = stablehlo.broadcast_in_dim %hng, dims = [] : (tensor<f32>) -> tensor<32x768xf32>
    %hnbtb = stablehlo.broadcast_in_dim %hnbt, dims = [] : (tensor<f32>) -> tensor<32x768xf32>
    %hngx = stablehlo.multiply %hnxh, %hngb : tensor<32x768xf32>
    %hnfl = stablehlo.add %hngx, %hnbtb : tensor<32x768xf32>
    %hn = stablehlo.reshape %hnfl : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %hnf = stablehlo.reshape %hn : (tensor<32x768x1x1xf32>) -> tensor<32x768xf32>
    %ld = stablehlo.dot_general %hnf, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x10xf32>) -> tensor<32x10xf32>
    %ldb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %logits = stablehlo.add %ld, %ldb : tensor<32x10xf32>
    %le = stablehlo.exponential %logits : tensor<32x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<32x10xf32>
    %dyr = stablehlo.subtract %lsm, %onehot : tensor<32x10xf32>
    %bnc = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %dy = stablehlo.divide %dyr, %bnc : tensor<32x10xf32>
    %dhnf = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<768x10xf32>) -> tensor<32x768xf32>
    %dWd = stablehlo.dot_general %hnf, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<32x10xf32>) -> tensor<768x10xf32>
    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dhnr = stablehlo.reshape %dhnf : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %hdri = stablehlo.reshape %gapr : (tensor<32x768x1x1xf32>) -> tensor<32x768xf32>
    %hdrdy = stablehlo.reshape %dhnr : (tensor<32x768x1x1xf32>) -> tensor<32x768xf32>
    %hdnf = stablehlo.constant dense<768.0> : tensor<32x768xf32>
    %hdep = stablehlo.constant dense<1.0e-6> : tensor<32x768xf32>
    %hdsmr = stablehlo.reduce(%hdri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %hdsm = stablehlo.broadcast_in_dim %hdsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %hdmu = stablehlo.divide %hdsm, %hdnf : tensor<32x768xf32>
    %hdxc = stablehlo.subtract %hdri, %hdmu : tensor<32x768xf32>
    %hdsq = stablehlo.multiply %hdxc, %hdxc : tensor<32x768xf32>
    %hdvsr = stablehlo.reduce(%hdsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %hdvs = stablehlo.broadcast_in_dim %hdvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %hdvr = stablehlo.divide %hdvs, %hdnf : tensor<32x768xf32>
    %hdve = stablehlo.add %hdvr, %hdep : tensor<32x768xf32>
    %hdistd = stablehlo.rsqrt %hdve : tensor<32x768xf32>
    %hdxh = stablehlo.multiply %hdxc, %hdistd : tensor<32x768xf32>
    %hdgb = stablehlo.broadcast_in_dim %hng, dims = [] : (tensor<f32>) -> tensor<32x768xf32>
    %hddxh = stablehlo.multiply %hdgb, %hdrdy : tensor<32x768xf32>
    %hdsdxr = stablehlo.reduce(%hddxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %hdsdx = stablehlo.broadcast_in_dim %hdsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %hdxd = stablehlo.multiply %hdxh, %hddxh : tensor<32x768xf32>
    %hdsxdr = stablehlo.reduce(%hdxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %hdsxd = stablehlo.broadcast_in_dim %hdsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %hdt1 = stablehlo.multiply %hddxh, %hdnf : tensor<32x768xf32>
    %hdi1 = stablehlo.subtract %hdt1, %hdsdx : tensor<32x768xf32>
    %hdxs = stablehlo.multiply %hdxh, %hdsxd : tensor<32x768xf32>
    %hdi2 = stablehlo.subtract %hdi1, %hdxs : tensor<32x768xf32>
    %hdsN = stablehlo.divide %hdistd, %hdnf : tensor<32x768xf32>
    %hdgin = stablehlo.multiply %hdsN, %hdi2 : tensor<32x768xf32>
    %hd = stablehlo.reshape %hdgin : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %hddgp = stablehlo.multiply %hdrdy, %hdxh : tensor<32x768xf32>
    %hddg = stablehlo.reduce(%hddgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<f32>
    %hddb = stablehlo.reduce(%hdrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<f32>
    %hdf = stablehlo.reshape %hd : (tensor<32x768x1x1xf32>) -> tensor<32x768xf32>
    %dgd = stablehlo.divide %hdf, %gapnf : tensor<32x768xf32>
    %dgap = stablehlo.broadcast_in_dim %dgd, dims = [0, 1] : (tensor<32x768xf32>) -> tensor<32x768x7x7xf32>
    %s3b2dlsgb = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b2dls = stablehlo.multiply %s3b2dlsgb, %dgap : tensor<32x768x7x7xf32>
    %s3b2dlsxdy = stablehlo.multiply %s3b2p, %dgap : tensor<32x768x7x7xf32>
    %s3b2dlsdg = stablehlo.reduce(%s3b2dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b2dpt = stablehlo.transpose %s3b2pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %s3b2dp = stablehlo.convolution(%s3b2dls, %s3b2dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %s3b2dpWxt = stablehlo.transpose %s3b2g, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %s3b2dpWdt = stablehlo.transpose %s3b2dls, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b2dpWraw = stablehlo.convolution(%s3b2dpWxt, %s3b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %s3b2dpW = stablehlo.transpose %s3b2dpWraw, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %s3b2dpb = stablehlo.reduce(%s3b2dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b2dgx2 = stablehlo.multiply %s3b2e, %s3b2e : tensor<32x3072x7x7xf32>
    %s3b2dgx3 = stablehlo.multiply %s3b2dgx2, %s3b2e : tensor<32x3072x7x7xf32>
    %s3b2dgck = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %s3b2dgkx3 = stablehlo.multiply %s3b2dgck, %s3b2dgx3 : tensor<32x3072x7x7xf32>
    %s3b2dginn = stablehlo.add %s3b2e, %s3b2dgkx3 : tensor<32x3072x7x7xf32>
    %s3b2dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %s3b2dgu = stablehlo.multiply %s3b2dgcs, %s3b2dginn : tensor<32x3072x7x7xf32>
    %s3b2dgt = stablehlo.tanh %s3b2dgu : tensor<32x3072x7x7xf32>
    %s3b2dgone = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %s3b2dgopt = stablehlo.add %s3b2dgone, %s3b2dgt : tensor<32x3072x7x7xf32>
    %s3b2dghalf = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %s3b2dgterm1 = stablehlo.multiply %s3b2dghalf, %s3b2dgopt : tensor<32x3072x7x7xf32>
    %s3b2dgt2 = stablehlo.multiply %s3b2dgt, %s3b2dgt : tensor<32x3072x7x7xf32>
    %s3b2dgomt2 = stablehlo.subtract %s3b2dgone, %s3b2dgt2 : tensor<32x3072x7x7xf32>
    %s3b2dghx = stablehlo.multiply %s3b2dghalf, %s3b2e : tensor<32x3072x7x7xf32>
    %s3b2dghxo = stablehlo.multiply %s3b2dghx, %s3b2dgomt2 : tensor<32x3072x7x7xf32>
    %s3b2dgc3b = stablehlo.constant dense<0.134145> : tensor<32x3072x7x7xf32>
    %s3b2dga3x2 = stablehlo.multiply %s3b2dgc3b, %s3b2dgx2 : tensor<32x3072x7x7xf32>
    %s3b2dgin2 = stablehlo.add %s3b2dgone, %s3b2dga3x2 : tensor<32x3072x7x7xf32>
    %s3b2dgup = stablehlo.multiply %s3b2dgcs, %s3b2dgin2 : tensor<32x3072x7x7xf32>
    %s3b2dgterm2 = stablehlo.multiply %s3b2dghxo, %s3b2dgup : tensor<32x3072x7x7xf32>
    %s3b2dggp = stablehlo.add %s3b2dgterm1, %s3b2dgterm2 : tensor<32x3072x7x7xf32>
    %s3b2dg = stablehlo.multiply %s3b2dp, %s3b2dggp : tensor<32x3072x7x7xf32>
    %s3b2det = stablehlo.transpose %s3b2eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %s3b2de = stablehlo.convolution(%s3b2dg, %s3b2det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %s3b2deWxt = stablehlo.transpose %s3b2n, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b2deWdt = stablehlo.transpose %s3b2dg, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %s3b2deWraw = stablehlo.convolution(%s3b2deWxt, %s3b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %s3b2deW = stablehlo.transpose %s3b2deWraw, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %s3b2deb = stablehlo.reduce(%s3b2dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %s3b2dnri = stablehlo.reshape %s3b2d : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b2dnrdy = stablehlo.reshape %s3b2de : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b2dnnf = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %s3b2dnep = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %s3b2dnsmr = stablehlo.reduce(%s3b2dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b2dnsm = stablehlo.broadcast_in_dim %s3b2dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b2dnmu = stablehlo.divide %s3b2dnsm, %s3b2dnnf : tensor<32x37632xf32>
    %s3b2dnxc = stablehlo.subtract %s3b2dnri, %s3b2dnmu : tensor<32x37632xf32>
    %s3b2dnsq = stablehlo.multiply %s3b2dnxc, %s3b2dnxc : tensor<32x37632xf32>
    %s3b2dnvsr = stablehlo.reduce(%s3b2dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b2dnvs = stablehlo.broadcast_in_dim %s3b2dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b2dnvr = stablehlo.divide %s3b2dnvs, %s3b2dnnf : tensor<32x37632xf32>
    %s3b2dnve = stablehlo.add %s3b2dnvr, %s3b2dnep : tensor<32x37632xf32>
    %s3b2dnistd = stablehlo.rsqrt %s3b2dnve : tensor<32x37632xf32>
    %s3b2dnxh = stablehlo.multiply %s3b2dnxc, %s3b2dnistd : tensor<32x37632xf32>
    %s3b2dngb = stablehlo.broadcast_in_dim %s3b2ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b2dndxh = stablehlo.multiply %s3b2dngb, %s3b2dnrdy : tensor<32x37632xf32>
    %s3b2dnsdxr = stablehlo.reduce(%s3b2dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b2dnsdx = stablehlo.broadcast_in_dim %s3b2dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b2dnxd = stablehlo.multiply %s3b2dnxh, %s3b2dndxh : tensor<32x37632xf32>
    %s3b2dnsxdr = stablehlo.reduce(%s3b2dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b2dnsxd = stablehlo.broadcast_in_dim %s3b2dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b2dnt1 = stablehlo.multiply %s3b2dndxh, %s3b2dnnf : tensor<32x37632xf32>
    %s3b2dni1 = stablehlo.subtract %s3b2dnt1, %s3b2dnsdx : tensor<32x37632xf32>
    %s3b2dnxs = stablehlo.multiply %s3b2dnxh, %s3b2dnsxd : tensor<32x37632xf32>
    %s3b2dni2 = stablehlo.subtract %s3b2dni1, %s3b2dnxs : tensor<32x37632xf32>
    %s3b2dnsN = stablehlo.divide %s3b2dnistd, %s3b2dnnf : tensor<32x37632xf32>
    %s3b2dngin = stablehlo.multiply %s3b2dnsN, %s3b2dni2 : tensor<32x37632xf32>
    %s3b2dn = stablehlo.reshape %s3b2dngin : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %s3b2dndgp = stablehlo.multiply %s3b2dnrdy, %s3b2dnxh : tensor<32x37632xf32>
    %s3b2dndg = stablehlo.reduce(%s3b2dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %s3b2dndb = stablehlo.reduce(%s3b2dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %s3b2ddrev = stablehlo.reverse %s3b2dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %s3b2dd = stablehlo.convolution(%s3b2dn, %s3b2ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %s3b2ddWxt = stablehlo.transpose %s3b1o, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b2ddWdt = stablehlo.transpose %s3b2dn, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b2ddWraw = stablehlo.convolution(%s3b2ddWxt, %s3b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %s3b2ddW = stablehlo.reshape %s3b2ddWraw : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %s3b2ddb = stablehlo.reduce(%s3b2dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b2dx = stablehlo.add %s3b2dd, %dgap : tensor<32x768x7x7xf32>
    %s3b1dlsgb = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b1dls = stablehlo.multiply %s3b1dlsgb, %s3b2dx : tensor<32x768x7x7xf32>
    %s3b1dlsxdy = stablehlo.multiply %s3b1p, %s3b2dx : tensor<32x768x7x7xf32>
    %s3b1dlsdg = stablehlo.reduce(%s3b1dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b1dpt = stablehlo.transpose %s3b1pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %s3b1dp = stablehlo.convolution(%s3b1dls, %s3b1dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %s3b1dpWxt = stablehlo.transpose %s3b1g, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %s3b1dpWdt = stablehlo.transpose %s3b1dls, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b1dpWraw = stablehlo.convolution(%s3b1dpWxt, %s3b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %s3b1dpW = stablehlo.transpose %s3b1dpWraw, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %s3b1dpb = stablehlo.reduce(%s3b1dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b1dgx2 = stablehlo.multiply %s3b1e, %s3b1e : tensor<32x3072x7x7xf32>
    %s3b1dgx3 = stablehlo.multiply %s3b1dgx2, %s3b1e : tensor<32x3072x7x7xf32>
    %s3b1dgck = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %s3b1dgkx3 = stablehlo.multiply %s3b1dgck, %s3b1dgx3 : tensor<32x3072x7x7xf32>
    %s3b1dginn = stablehlo.add %s3b1e, %s3b1dgkx3 : tensor<32x3072x7x7xf32>
    %s3b1dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %s3b1dgu = stablehlo.multiply %s3b1dgcs, %s3b1dginn : tensor<32x3072x7x7xf32>
    %s3b1dgt = stablehlo.tanh %s3b1dgu : tensor<32x3072x7x7xf32>
    %s3b1dgone = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %s3b1dgopt = stablehlo.add %s3b1dgone, %s3b1dgt : tensor<32x3072x7x7xf32>
    %s3b1dghalf = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %s3b1dgterm1 = stablehlo.multiply %s3b1dghalf, %s3b1dgopt : tensor<32x3072x7x7xf32>
    %s3b1dgt2 = stablehlo.multiply %s3b1dgt, %s3b1dgt : tensor<32x3072x7x7xf32>
    %s3b1dgomt2 = stablehlo.subtract %s3b1dgone, %s3b1dgt2 : tensor<32x3072x7x7xf32>
    %s3b1dghx = stablehlo.multiply %s3b1dghalf, %s3b1e : tensor<32x3072x7x7xf32>
    %s3b1dghxo = stablehlo.multiply %s3b1dghx, %s3b1dgomt2 : tensor<32x3072x7x7xf32>
    %s3b1dgc3b = stablehlo.constant dense<0.134145> : tensor<32x3072x7x7xf32>
    %s3b1dga3x2 = stablehlo.multiply %s3b1dgc3b, %s3b1dgx2 : tensor<32x3072x7x7xf32>
    %s3b1dgin2 = stablehlo.add %s3b1dgone, %s3b1dga3x2 : tensor<32x3072x7x7xf32>
    %s3b1dgup = stablehlo.multiply %s3b1dgcs, %s3b1dgin2 : tensor<32x3072x7x7xf32>
    %s3b1dgterm2 = stablehlo.multiply %s3b1dghxo, %s3b1dgup : tensor<32x3072x7x7xf32>
    %s3b1dggp = stablehlo.add %s3b1dgterm1, %s3b1dgterm2 : tensor<32x3072x7x7xf32>
    %s3b1dg = stablehlo.multiply %s3b1dp, %s3b1dggp : tensor<32x3072x7x7xf32>
    %s3b1det = stablehlo.transpose %s3b1eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %s3b1de = stablehlo.convolution(%s3b1dg, %s3b1det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %s3b1deWxt = stablehlo.transpose %s3b1n, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b1deWdt = stablehlo.transpose %s3b1dg, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %s3b1deWraw = stablehlo.convolution(%s3b1deWxt, %s3b1deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %s3b1deW = stablehlo.transpose %s3b1deWraw, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %s3b1deb = stablehlo.reduce(%s3b1dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %s3b1dnri = stablehlo.reshape %s3b1d : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b1dnrdy = stablehlo.reshape %s3b1de : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b1dnnf = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %s3b1dnep = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %s3b1dnsmr = stablehlo.reduce(%s3b1dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b1dnsm = stablehlo.broadcast_in_dim %s3b1dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b1dnmu = stablehlo.divide %s3b1dnsm, %s3b1dnnf : tensor<32x37632xf32>
    %s3b1dnxc = stablehlo.subtract %s3b1dnri, %s3b1dnmu : tensor<32x37632xf32>
    %s3b1dnsq = stablehlo.multiply %s3b1dnxc, %s3b1dnxc : tensor<32x37632xf32>
    %s3b1dnvsr = stablehlo.reduce(%s3b1dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b1dnvs = stablehlo.broadcast_in_dim %s3b1dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b1dnvr = stablehlo.divide %s3b1dnvs, %s3b1dnnf : tensor<32x37632xf32>
    %s3b1dnve = stablehlo.add %s3b1dnvr, %s3b1dnep : tensor<32x37632xf32>
    %s3b1dnistd = stablehlo.rsqrt %s3b1dnve : tensor<32x37632xf32>
    %s3b1dnxh = stablehlo.multiply %s3b1dnxc, %s3b1dnistd : tensor<32x37632xf32>
    %s3b1dngb = stablehlo.broadcast_in_dim %s3b1ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b1dndxh = stablehlo.multiply %s3b1dngb, %s3b1dnrdy : tensor<32x37632xf32>
    %s3b1dnsdxr = stablehlo.reduce(%s3b1dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b1dnsdx = stablehlo.broadcast_in_dim %s3b1dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b1dnxd = stablehlo.multiply %s3b1dnxh, %s3b1dndxh : tensor<32x37632xf32>
    %s3b1dnsxdr = stablehlo.reduce(%s3b1dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b1dnsxd = stablehlo.broadcast_in_dim %s3b1dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b1dnt1 = stablehlo.multiply %s3b1dndxh, %s3b1dnnf : tensor<32x37632xf32>
    %s3b1dni1 = stablehlo.subtract %s3b1dnt1, %s3b1dnsdx : tensor<32x37632xf32>
    %s3b1dnxs = stablehlo.multiply %s3b1dnxh, %s3b1dnsxd : tensor<32x37632xf32>
    %s3b1dni2 = stablehlo.subtract %s3b1dni1, %s3b1dnxs : tensor<32x37632xf32>
    %s3b1dnsN = stablehlo.divide %s3b1dnistd, %s3b1dnnf : tensor<32x37632xf32>
    %s3b1dngin = stablehlo.multiply %s3b1dnsN, %s3b1dni2 : tensor<32x37632xf32>
    %s3b1dn = stablehlo.reshape %s3b1dngin : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %s3b1dndgp = stablehlo.multiply %s3b1dnrdy, %s3b1dnxh : tensor<32x37632xf32>
    %s3b1dndg = stablehlo.reduce(%s3b1dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %s3b1dndb = stablehlo.reduce(%s3b1dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %s3b1ddrev = stablehlo.reverse %s3b1dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %s3b1dd = stablehlo.convolution(%s3b1dn, %s3b1ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %s3b1ddWxt = stablehlo.transpose %s3b0o, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b1ddWdt = stablehlo.transpose %s3b1dn, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b1ddWraw = stablehlo.convolution(%s3b1ddWxt, %s3b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %s3b1ddW = stablehlo.reshape %s3b1ddWraw : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %s3b1ddb = stablehlo.reduce(%s3b1dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b1dx = stablehlo.add %s3b1dd, %s3b2dx : tensor<32x768x7x7xf32>
    %s3b0dlsgb = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b0dls = stablehlo.multiply %s3b0dlsgb, %s3b1dx : tensor<32x768x7x7xf32>
    %s3b0dlsxdy = stablehlo.multiply %s3b0p, %s3b1dx : tensor<32x768x7x7xf32>
    %s3b0dlsdg = stablehlo.reduce(%s3b0dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b0dpt = stablehlo.transpose %s3b0pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %s3b0dp = stablehlo.convolution(%s3b0dls, %s3b0dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %s3b0dpWxt = stablehlo.transpose %s3b0g, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %s3b0dpWdt = stablehlo.transpose %s3b0dls, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b0dpWraw = stablehlo.convolution(%s3b0dpWxt, %s3b0dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %s3b0dpW = stablehlo.transpose %s3b0dpWraw, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %s3b0dpb = stablehlo.reduce(%s3b0dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b0dgx2 = stablehlo.multiply %s3b0e, %s3b0e : tensor<32x3072x7x7xf32>
    %s3b0dgx3 = stablehlo.multiply %s3b0dgx2, %s3b0e : tensor<32x3072x7x7xf32>
    %s3b0dgck = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %s3b0dgkx3 = stablehlo.multiply %s3b0dgck, %s3b0dgx3 : tensor<32x3072x7x7xf32>
    %s3b0dginn = stablehlo.add %s3b0e, %s3b0dgkx3 : tensor<32x3072x7x7xf32>
    %s3b0dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %s3b0dgu = stablehlo.multiply %s3b0dgcs, %s3b0dginn : tensor<32x3072x7x7xf32>
    %s3b0dgt = stablehlo.tanh %s3b0dgu : tensor<32x3072x7x7xf32>
    %s3b0dgone = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %s3b0dgopt = stablehlo.add %s3b0dgone, %s3b0dgt : tensor<32x3072x7x7xf32>
    %s3b0dghalf = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %s3b0dgterm1 = stablehlo.multiply %s3b0dghalf, %s3b0dgopt : tensor<32x3072x7x7xf32>
    %s3b0dgt2 = stablehlo.multiply %s3b0dgt, %s3b0dgt : tensor<32x3072x7x7xf32>
    %s3b0dgomt2 = stablehlo.subtract %s3b0dgone, %s3b0dgt2 : tensor<32x3072x7x7xf32>
    %s3b0dghx = stablehlo.multiply %s3b0dghalf, %s3b0e : tensor<32x3072x7x7xf32>
    %s3b0dghxo = stablehlo.multiply %s3b0dghx, %s3b0dgomt2 : tensor<32x3072x7x7xf32>
    %s3b0dgc3b = stablehlo.constant dense<0.134145> : tensor<32x3072x7x7xf32>
    %s3b0dga3x2 = stablehlo.multiply %s3b0dgc3b, %s3b0dgx2 : tensor<32x3072x7x7xf32>
    %s3b0dgin2 = stablehlo.add %s3b0dgone, %s3b0dga3x2 : tensor<32x3072x7x7xf32>
    %s3b0dgup = stablehlo.multiply %s3b0dgcs, %s3b0dgin2 : tensor<32x3072x7x7xf32>
    %s3b0dgterm2 = stablehlo.multiply %s3b0dghxo, %s3b0dgup : tensor<32x3072x7x7xf32>
    %s3b0dggp = stablehlo.add %s3b0dgterm1, %s3b0dgterm2 : tensor<32x3072x7x7xf32>
    %s3b0dg = stablehlo.multiply %s3b0dp, %s3b0dggp : tensor<32x3072x7x7xf32>
    %s3b0det = stablehlo.transpose %s3b0eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %s3b0de = stablehlo.convolution(%s3b0dg, %s3b0det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %s3b0deWxt = stablehlo.transpose %s3b0n, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b0deWdt = stablehlo.transpose %s3b0dg, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %s3b0deWraw = stablehlo.convolution(%s3b0deWxt, %s3b0deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %s3b0deW = stablehlo.transpose %s3b0deWraw, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %s3b0deb = stablehlo.reduce(%s3b0dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %s3b0dnri = stablehlo.reshape %s3b0d : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b0dnrdy = stablehlo.reshape %s3b0de : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b0dnnf = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %s3b0dnep = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %s3b0dnsmr = stablehlo.reduce(%s3b0dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b0dnsm = stablehlo.broadcast_in_dim %s3b0dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b0dnmu = stablehlo.divide %s3b0dnsm, %s3b0dnnf : tensor<32x37632xf32>
    %s3b0dnxc = stablehlo.subtract %s3b0dnri, %s3b0dnmu : tensor<32x37632xf32>
    %s3b0dnsq = stablehlo.multiply %s3b0dnxc, %s3b0dnxc : tensor<32x37632xf32>
    %s3b0dnvsr = stablehlo.reduce(%s3b0dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b0dnvs = stablehlo.broadcast_in_dim %s3b0dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b0dnvr = stablehlo.divide %s3b0dnvs, %s3b0dnnf : tensor<32x37632xf32>
    %s3b0dnve = stablehlo.add %s3b0dnvr, %s3b0dnep : tensor<32x37632xf32>
    %s3b0dnistd = stablehlo.rsqrt %s3b0dnve : tensor<32x37632xf32>
    %s3b0dnxh = stablehlo.multiply %s3b0dnxc, %s3b0dnistd : tensor<32x37632xf32>
    %s3b0dngb = stablehlo.broadcast_in_dim %s3b0ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b0dndxh = stablehlo.multiply %s3b0dngb, %s3b0dnrdy : tensor<32x37632xf32>
    %s3b0dnsdxr = stablehlo.reduce(%s3b0dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b0dnsdx = stablehlo.broadcast_in_dim %s3b0dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b0dnxd = stablehlo.multiply %s3b0dnxh, %s3b0dndxh : tensor<32x37632xf32>
    %s3b0dnsxdr = stablehlo.reduce(%s3b0dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b0dnsxd = stablehlo.broadcast_in_dim %s3b0dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b0dnt1 = stablehlo.multiply %s3b0dndxh, %s3b0dnnf : tensor<32x37632xf32>
    %s3b0dni1 = stablehlo.subtract %s3b0dnt1, %s3b0dnsdx : tensor<32x37632xf32>
    %s3b0dnxs = stablehlo.multiply %s3b0dnxh, %s3b0dnsxd : tensor<32x37632xf32>
    %s3b0dni2 = stablehlo.subtract %s3b0dni1, %s3b0dnxs : tensor<32x37632xf32>
    %s3b0dnsN = stablehlo.divide %s3b0dnistd, %s3b0dnnf : tensor<32x37632xf32>
    %s3b0dngin = stablehlo.multiply %s3b0dnsN, %s3b0dni2 : tensor<32x37632xf32>
    %s3b0dn = stablehlo.reshape %s3b0dngin : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %s3b0dndgp = stablehlo.multiply %s3b0dnrdy, %s3b0dnxh : tensor<32x37632xf32>
    %s3b0dndg = stablehlo.reduce(%s3b0dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %s3b0dndb = stablehlo.reduce(%s3b0dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %s3b0ddrev = stablehlo.reverse %s3b0dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %s3b0dd = stablehlo.convolution(%s3b0dn, %s3b0ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %s3b0ddWxt = stablehlo.transpose %d2c, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b0ddWdt = stablehlo.transpose %s3b0dn, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b0ddWraw = stablehlo.convolution(%s3b0ddWxt, %s3b0ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %s3b0ddW = stablehlo.reshape %s3b0ddWraw : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %s3b0ddb = stablehlo.reduce(%s3b0dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b0dx = stablehlo.add %s3b0dd, %s3b1dx : tensor<32x768x7x7xf32>
    %d2dcu = stablehlo.pad %s3b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x14x14xf32>
    %d2dct = stablehlo.transpose %d2W, dims = [1, 0, 2, 3] : (tensor<768x384x2x2xf32>) -> tensor<384x768x2x2xf32>
    %d2dcr = stablehlo.reverse %d2dct, dims = [2, 3] : tensor<384x768x2x2xf32>
    %d2dc = stablehlo.convolution(%d2dcu, %d2dcr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x14x14xf32>, tensor<384x768x2x2xf32>) -> tensor<32x384x14x14xf32>
    %d2dWu = stablehlo.pad %s3b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x13x13xf32>
    %d2dWxt = stablehlo.transpose %d2n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %d2dWdt = stablehlo.transpose %d2dWu, dims = [1, 0, 2, 3] : (tensor<32x768x13x13xf32>) -> tensor<768x32x13x13xf32>
    %d2dWraw = stablehlo.convolution(%d2dWxt, %d2dWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<768x32x13x13xf32>) -> tensor<384x768x2x2xf32>
    %d2dW = stablehlo.transpose %d2dWraw, dims = [1, 0, 2, 3] : (tensor<384x768x2x2xf32>) -> tensor<768x384x2x2xf32>
    %d2db = stablehlo.reduce(%s3b0dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %d2dnri = stablehlo.reshape %s2b8o : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %d2dnrdy = stablehlo.reshape %d2dc : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %d2dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %d2dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %d2dnsmr = stablehlo.reduce(%d2dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %d2dnsm = stablehlo.broadcast_in_dim %d2dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %d2dnmu = stablehlo.divide %d2dnsm, %d2dnnf : tensor<32x75264xf32>
    %d2dnxc = stablehlo.subtract %d2dnri, %d2dnmu : tensor<32x75264xf32>
    %d2dnsq = stablehlo.multiply %d2dnxc, %d2dnxc : tensor<32x75264xf32>
    %d2dnvsr = stablehlo.reduce(%d2dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %d2dnvs = stablehlo.broadcast_in_dim %d2dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %d2dnvr = stablehlo.divide %d2dnvs, %d2dnnf : tensor<32x75264xf32>
    %d2dnve = stablehlo.add %d2dnvr, %d2dnep : tensor<32x75264xf32>
    %d2dnistd = stablehlo.rsqrt %d2dnve : tensor<32x75264xf32>
    %d2dnxh = stablehlo.multiply %d2dnxc, %d2dnistd : tensor<32x75264xf32>
    %d2dngb = stablehlo.broadcast_in_dim %d2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %d2dndxh = stablehlo.multiply %d2dngb, %d2dnrdy : tensor<32x75264xf32>
    %d2dnsdxr = stablehlo.reduce(%d2dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %d2dnsdx = stablehlo.broadcast_in_dim %d2dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %d2dnxd = stablehlo.multiply %d2dnxh, %d2dndxh : tensor<32x75264xf32>
    %d2dnsxdr = stablehlo.reduce(%d2dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %d2dnsxd = stablehlo.broadcast_in_dim %d2dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %d2dnt1 = stablehlo.multiply %d2dndxh, %d2dnnf : tensor<32x75264xf32>
    %d2dni1 = stablehlo.subtract %d2dnt1, %d2dnsdx : tensor<32x75264xf32>
    %d2dnxs = stablehlo.multiply %d2dnxh, %d2dnsxd : tensor<32x75264xf32>
    %d2dni2 = stablehlo.subtract %d2dni1, %d2dnxs : tensor<32x75264xf32>
    %d2dnsN = stablehlo.divide %d2dnistd, %d2dnnf : tensor<32x75264xf32>
    %d2dngin = stablehlo.multiply %d2dnsN, %d2dni2 : tensor<32x75264xf32>
    %d2dn = stablehlo.reshape %d2dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %d2dndgp = stablehlo.multiply %d2dnrdy, %d2dnxh : tensor<32x75264xf32>
    %d2dndg = stablehlo.reduce(%d2dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %d2dndb = stablehlo.reduce(%d2dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b8dlsgb = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b8dls = stablehlo.multiply %s2b8dlsgb, %d2dn : tensor<32x384x14x14xf32>
    %s2b8dlsxdy = stablehlo.multiply %s2b8p, %d2dn : tensor<32x384x14x14xf32>
    %s2b8dlsdg = stablehlo.reduce(%s2b8dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b8dpt = stablehlo.transpose %s2b8pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b8dp = stablehlo.convolution(%s2b8dls, %s2b8dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b8dpWxt = stablehlo.transpose %s2b8g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b8dpWdt = stablehlo.transpose %s2b8dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b8dpWraw = stablehlo.convolution(%s2b8dpWxt, %s2b8dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b8dpW = stablehlo.transpose %s2b8dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b8dpb = stablehlo.reduce(%s2b8dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b8dgx2 = stablehlo.multiply %s2b8e, %s2b8e : tensor<32x1536x14x14xf32>
    %s2b8dgx3 = stablehlo.multiply %s2b8dgx2, %s2b8e : tensor<32x1536x14x14xf32>
    %s2b8dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b8dgkx3 = stablehlo.multiply %s2b8dgck, %s2b8dgx3 : tensor<32x1536x14x14xf32>
    %s2b8dginn = stablehlo.add %s2b8e, %s2b8dgkx3 : tensor<32x1536x14x14xf32>
    %s2b8dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b8dgu = stablehlo.multiply %s2b8dgcs, %s2b8dginn : tensor<32x1536x14x14xf32>
    %s2b8dgt = stablehlo.tanh %s2b8dgu : tensor<32x1536x14x14xf32>
    %s2b8dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b8dgopt = stablehlo.add %s2b8dgone, %s2b8dgt : tensor<32x1536x14x14xf32>
    %s2b8dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b8dgterm1 = stablehlo.multiply %s2b8dghalf, %s2b8dgopt : tensor<32x1536x14x14xf32>
    %s2b8dgt2 = stablehlo.multiply %s2b8dgt, %s2b8dgt : tensor<32x1536x14x14xf32>
    %s2b8dgomt2 = stablehlo.subtract %s2b8dgone, %s2b8dgt2 : tensor<32x1536x14x14xf32>
    %s2b8dghx = stablehlo.multiply %s2b8dghalf, %s2b8e : tensor<32x1536x14x14xf32>
    %s2b8dghxo = stablehlo.multiply %s2b8dghx, %s2b8dgomt2 : tensor<32x1536x14x14xf32>
    %s2b8dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b8dga3x2 = stablehlo.multiply %s2b8dgc3b, %s2b8dgx2 : tensor<32x1536x14x14xf32>
    %s2b8dgin2 = stablehlo.add %s2b8dgone, %s2b8dga3x2 : tensor<32x1536x14x14xf32>
    %s2b8dgup = stablehlo.multiply %s2b8dgcs, %s2b8dgin2 : tensor<32x1536x14x14xf32>
    %s2b8dgterm2 = stablehlo.multiply %s2b8dghxo, %s2b8dgup : tensor<32x1536x14x14xf32>
    %s2b8dggp = stablehlo.add %s2b8dgterm1, %s2b8dgterm2 : tensor<32x1536x14x14xf32>
    %s2b8dg = stablehlo.multiply %s2b8dp, %s2b8dggp : tensor<32x1536x14x14xf32>
    %s2b8det = stablehlo.transpose %s2b8eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b8de = stablehlo.convolution(%s2b8dg, %s2b8det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b8deWxt = stablehlo.transpose %s2b8n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b8deWdt = stablehlo.transpose %s2b8dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b8deWraw = stablehlo.convolution(%s2b8deWxt, %s2b8deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b8deW = stablehlo.transpose %s2b8deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b8deb = stablehlo.reduce(%s2b8dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b8dnri = stablehlo.reshape %s2b8d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b8dnrdy = stablehlo.reshape %s2b8de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b8dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b8dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b8dnsmr = stablehlo.reduce(%s2b8dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b8dnsm = stablehlo.broadcast_in_dim %s2b8dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b8dnmu = stablehlo.divide %s2b8dnsm, %s2b8dnnf : tensor<32x75264xf32>
    %s2b8dnxc = stablehlo.subtract %s2b8dnri, %s2b8dnmu : tensor<32x75264xf32>
    %s2b8dnsq = stablehlo.multiply %s2b8dnxc, %s2b8dnxc : tensor<32x75264xf32>
    %s2b8dnvsr = stablehlo.reduce(%s2b8dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b8dnvs = stablehlo.broadcast_in_dim %s2b8dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b8dnvr = stablehlo.divide %s2b8dnvs, %s2b8dnnf : tensor<32x75264xf32>
    %s2b8dnve = stablehlo.add %s2b8dnvr, %s2b8dnep : tensor<32x75264xf32>
    %s2b8dnistd = stablehlo.rsqrt %s2b8dnve : tensor<32x75264xf32>
    %s2b8dnxh = stablehlo.multiply %s2b8dnxc, %s2b8dnistd : tensor<32x75264xf32>
    %s2b8dngb = stablehlo.broadcast_in_dim %s2b8ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b8dndxh = stablehlo.multiply %s2b8dngb, %s2b8dnrdy : tensor<32x75264xf32>
    %s2b8dnsdxr = stablehlo.reduce(%s2b8dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b8dnsdx = stablehlo.broadcast_in_dim %s2b8dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b8dnxd = stablehlo.multiply %s2b8dnxh, %s2b8dndxh : tensor<32x75264xf32>
    %s2b8dnsxdr = stablehlo.reduce(%s2b8dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b8dnsxd = stablehlo.broadcast_in_dim %s2b8dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b8dnt1 = stablehlo.multiply %s2b8dndxh, %s2b8dnnf : tensor<32x75264xf32>
    %s2b8dni1 = stablehlo.subtract %s2b8dnt1, %s2b8dnsdx : tensor<32x75264xf32>
    %s2b8dnxs = stablehlo.multiply %s2b8dnxh, %s2b8dnsxd : tensor<32x75264xf32>
    %s2b8dni2 = stablehlo.subtract %s2b8dni1, %s2b8dnxs : tensor<32x75264xf32>
    %s2b8dnsN = stablehlo.divide %s2b8dnistd, %s2b8dnnf : tensor<32x75264xf32>
    %s2b8dngin = stablehlo.multiply %s2b8dnsN, %s2b8dni2 : tensor<32x75264xf32>
    %s2b8dn = stablehlo.reshape %s2b8dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b8dndgp = stablehlo.multiply %s2b8dnrdy, %s2b8dnxh : tensor<32x75264xf32>
    %s2b8dndg = stablehlo.reduce(%s2b8dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b8dndb = stablehlo.reduce(%s2b8dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b8ddrev = stablehlo.reverse %s2b8dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b8dd = stablehlo.convolution(%s2b8dn, %s2b8ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b8ddWxt = stablehlo.transpose %s2b7o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b8ddWdt = stablehlo.transpose %s2b8dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b8ddWraw = stablehlo.convolution(%s2b8ddWxt, %s2b8ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b8ddW = stablehlo.reshape %s2b8ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b8ddb = stablehlo.reduce(%s2b8dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b8dx = stablehlo.add %s2b8dd, %d2dn : tensor<32x384x14x14xf32>
    %s2b7dlsgb = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b7dls = stablehlo.multiply %s2b7dlsgb, %s2b8dx : tensor<32x384x14x14xf32>
    %s2b7dlsxdy = stablehlo.multiply %s2b7p, %s2b8dx : tensor<32x384x14x14xf32>
    %s2b7dlsdg = stablehlo.reduce(%s2b7dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b7dpt = stablehlo.transpose %s2b7pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b7dp = stablehlo.convolution(%s2b7dls, %s2b7dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b7dpWxt = stablehlo.transpose %s2b7g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b7dpWdt = stablehlo.transpose %s2b7dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b7dpWraw = stablehlo.convolution(%s2b7dpWxt, %s2b7dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b7dpW = stablehlo.transpose %s2b7dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b7dpb = stablehlo.reduce(%s2b7dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b7dgx2 = stablehlo.multiply %s2b7e, %s2b7e : tensor<32x1536x14x14xf32>
    %s2b7dgx3 = stablehlo.multiply %s2b7dgx2, %s2b7e : tensor<32x1536x14x14xf32>
    %s2b7dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b7dgkx3 = stablehlo.multiply %s2b7dgck, %s2b7dgx3 : tensor<32x1536x14x14xf32>
    %s2b7dginn = stablehlo.add %s2b7e, %s2b7dgkx3 : tensor<32x1536x14x14xf32>
    %s2b7dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b7dgu = stablehlo.multiply %s2b7dgcs, %s2b7dginn : tensor<32x1536x14x14xf32>
    %s2b7dgt = stablehlo.tanh %s2b7dgu : tensor<32x1536x14x14xf32>
    %s2b7dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b7dgopt = stablehlo.add %s2b7dgone, %s2b7dgt : tensor<32x1536x14x14xf32>
    %s2b7dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b7dgterm1 = stablehlo.multiply %s2b7dghalf, %s2b7dgopt : tensor<32x1536x14x14xf32>
    %s2b7dgt2 = stablehlo.multiply %s2b7dgt, %s2b7dgt : tensor<32x1536x14x14xf32>
    %s2b7dgomt2 = stablehlo.subtract %s2b7dgone, %s2b7dgt2 : tensor<32x1536x14x14xf32>
    %s2b7dghx = stablehlo.multiply %s2b7dghalf, %s2b7e : tensor<32x1536x14x14xf32>
    %s2b7dghxo = stablehlo.multiply %s2b7dghx, %s2b7dgomt2 : tensor<32x1536x14x14xf32>
    %s2b7dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b7dga3x2 = stablehlo.multiply %s2b7dgc3b, %s2b7dgx2 : tensor<32x1536x14x14xf32>
    %s2b7dgin2 = stablehlo.add %s2b7dgone, %s2b7dga3x2 : tensor<32x1536x14x14xf32>
    %s2b7dgup = stablehlo.multiply %s2b7dgcs, %s2b7dgin2 : tensor<32x1536x14x14xf32>
    %s2b7dgterm2 = stablehlo.multiply %s2b7dghxo, %s2b7dgup : tensor<32x1536x14x14xf32>
    %s2b7dggp = stablehlo.add %s2b7dgterm1, %s2b7dgterm2 : tensor<32x1536x14x14xf32>
    %s2b7dg = stablehlo.multiply %s2b7dp, %s2b7dggp : tensor<32x1536x14x14xf32>
    %s2b7det = stablehlo.transpose %s2b7eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b7de = stablehlo.convolution(%s2b7dg, %s2b7det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b7deWxt = stablehlo.transpose %s2b7n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b7deWdt = stablehlo.transpose %s2b7dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b7deWraw = stablehlo.convolution(%s2b7deWxt, %s2b7deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b7deW = stablehlo.transpose %s2b7deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b7deb = stablehlo.reduce(%s2b7dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b7dnri = stablehlo.reshape %s2b7d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b7dnrdy = stablehlo.reshape %s2b7de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b7dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b7dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b7dnsmr = stablehlo.reduce(%s2b7dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b7dnsm = stablehlo.broadcast_in_dim %s2b7dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b7dnmu = stablehlo.divide %s2b7dnsm, %s2b7dnnf : tensor<32x75264xf32>
    %s2b7dnxc = stablehlo.subtract %s2b7dnri, %s2b7dnmu : tensor<32x75264xf32>
    %s2b7dnsq = stablehlo.multiply %s2b7dnxc, %s2b7dnxc : tensor<32x75264xf32>
    %s2b7dnvsr = stablehlo.reduce(%s2b7dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b7dnvs = stablehlo.broadcast_in_dim %s2b7dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b7dnvr = stablehlo.divide %s2b7dnvs, %s2b7dnnf : tensor<32x75264xf32>
    %s2b7dnve = stablehlo.add %s2b7dnvr, %s2b7dnep : tensor<32x75264xf32>
    %s2b7dnistd = stablehlo.rsqrt %s2b7dnve : tensor<32x75264xf32>
    %s2b7dnxh = stablehlo.multiply %s2b7dnxc, %s2b7dnistd : tensor<32x75264xf32>
    %s2b7dngb = stablehlo.broadcast_in_dim %s2b7ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b7dndxh = stablehlo.multiply %s2b7dngb, %s2b7dnrdy : tensor<32x75264xf32>
    %s2b7dnsdxr = stablehlo.reduce(%s2b7dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b7dnsdx = stablehlo.broadcast_in_dim %s2b7dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b7dnxd = stablehlo.multiply %s2b7dnxh, %s2b7dndxh : tensor<32x75264xf32>
    %s2b7dnsxdr = stablehlo.reduce(%s2b7dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b7dnsxd = stablehlo.broadcast_in_dim %s2b7dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b7dnt1 = stablehlo.multiply %s2b7dndxh, %s2b7dnnf : tensor<32x75264xf32>
    %s2b7dni1 = stablehlo.subtract %s2b7dnt1, %s2b7dnsdx : tensor<32x75264xf32>
    %s2b7dnxs = stablehlo.multiply %s2b7dnxh, %s2b7dnsxd : tensor<32x75264xf32>
    %s2b7dni2 = stablehlo.subtract %s2b7dni1, %s2b7dnxs : tensor<32x75264xf32>
    %s2b7dnsN = stablehlo.divide %s2b7dnistd, %s2b7dnnf : tensor<32x75264xf32>
    %s2b7dngin = stablehlo.multiply %s2b7dnsN, %s2b7dni2 : tensor<32x75264xf32>
    %s2b7dn = stablehlo.reshape %s2b7dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b7dndgp = stablehlo.multiply %s2b7dnrdy, %s2b7dnxh : tensor<32x75264xf32>
    %s2b7dndg = stablehlo.reduce(%s2b7dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b7dndb = stablehlo.reduce(%s2b7dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b7ddrev = stablehlo.reverse %s2b7dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b7dd = stablehlo.convolution(%s2b7dn, %s2b7ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b7ddWxt = stablehlo.transpose %s2b6o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b7ddWdt = stablehlo.transpose %s2b7dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b7ddWraw = stablehlo.convolution(%s2b7ddWxt, %s2b7ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b7ddW = stablehlo.reshape %s2b7ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b7ddb = stablehlo.reduce(%s2b7dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b7dx = stablehlo.add %s2b7dd, %s2b8dx : tensor<32x384x14x14xf32>
    %s2b6dlsgb = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b6dls = stablehlo.multiply %s2b6dlsgb, %s2b7dx : tensor<32x384x14x14xf32>
    %s2b6dlsxdy = stablehlo.multiply %s2b6p, %s2b7dx : tensor<32x384x14x14xf32>
    %s2b6dlsdg = stablehlo.reduce(%s2b6dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b6dpt = stablehlo.transpose %s2b6pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b6dp = stablehlo.convolution(%s2b6dls, %s2b6dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b6dpWxt = stablehlo.transpose %s2b6g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b6dpWdt = stablehlo.transpose %s2b6dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b6dpWraw = stablehlo.convolution(%s2b6dpWxt, %s2b6dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b6dpW = stablehlo.transpose %s2b6dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b6dpb = stablehlo.reduce(%s2b6dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b6dgx2 = stablehlo.multiply %s2b6e, %s2b6e : tensor<32x1536x14x14xf32>
    %s2b6dgx3 = stablehlo.multiply %s2b6dgx2, %s2b6e : tensor<32x1536x14x14xf32>
    %s2b6dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b6dgkx3 = stablehlo.multiply %s2b6dgck, %s2b6dgx3 : tensor<32x1536x14x14xf32>
    %s2b6dginn = stablehlo.add %s2b6e, %s2b6dgkx3 : tensor<32x1536x14x14xf32>
    %s2b6dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b6dgu = stablehlo.multiply %s2b6dgcs, %s2b6dginn : tensor<32x1536x14x14xf32>
    %s2b6dgt = stablehlo.tanh %s2b6dgu : tensor<32x1536x14x14xf32>
    %s2b6dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b6dgopt = stablehlo.add %s2b6dgone, %s2b6dgt : tensor<32x1536x14x14xf32>
    %s2b6dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b6dgterm1 = stablehlo.multiply %s2b6dghalf, %s2b6dgopt : tensor<32x1536x14x14xf32>
    %s2b6dgt2 = stablehlo.multiply %s2b6dgt, %s2b6dgt : tensor<32x1536x14x14xf32>
    %s2b6dgomt2 = stablehlo.subtract %s2b6dgone, %s2b6dgt2 : tensor<32x1536x14x14xf32>
    %s2b6dghx = stablehlo.multiply %s2b6dghalf, %s2b6e : tensor<32x1536x14x14xf32>
    %s2b6dghxo = stablehlo.multiply %s2b6dghx, %s2b6dgomt2 : tensor<32x1536x14x14xf32>
    %s2b6dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b6dga3x2 = stablehlo.multiply %s2b6dgc3b, %s2b6dgx2 : tensor<32x1536x14x14xf32>
    %s2b6dgin2 = stablehlo.add %s2b6dgone, %s2b6dga3x2 : tensor<32x1536x14x14xf32>
    %s2b6dgup = stablehlo.multiply %s2b6dgcs, %s2b6dgin2 : tensor<32x1536x14x14xf32>
    %s2b6dgterm2 = stablehlo.multiply %s2b6dghxo, %s2b6dgup : tensor<32x1536x14x14xf32>
    %s2b6dggp = stablehlo.add %s2b6dgterm1, %s2b6dgterm2 : tensor<32x1536x14x14xf32>
    %s2b6dg = stablehlo.multiply %s2b6dp, %s2b6dggp : tensor<32x1536x14x14xf32>
    %s2b6det = stablehlo.transpose %s2b6eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b6de = stablehlo.convolution(%s2b6dg, %s2b6det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b6deWxt = stablehlo.transpose %s2b6n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b6deWdt = stablehlo.transpose %s2b6dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b6deWraw = stablehlo.convolution(%s2b6deWxt, %s2b6deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b6deW = stablehlo.transpose %s2b6deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b6deb = stablehlo.reduce(%s2b6dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b6dnri = stablehlo.reshape %s2b6d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b6dnrdy = stablehlo.reshape %s2b6de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b6dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b6dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b6dnsmr = stablehlo.reduce(%s2b6dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b6dnsm = stablehlo.broadcast_in_dim %s2b6dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b6dnmu = stablehlo.divide %s2b6dnsm, %s2b6dnnf : tensor<32x75264xf32>
    %s2b6dnxc = stablehlo.subtract %s2b6dnri, %s2b6dnmu : tensor<32x75264xf32>
    %s2b6dnsq = stablehlo.multiply %s2b6dnxc, %s2b6dnxc : tensor<32x75264xf32>
    %s2b6dnvsr = stablehlo.reduce(%s2b6dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b6dnvs = stablehlo.broadcast_in_dim %s2b6dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b6dnvr = stablehlo.divide %s2b6dnvs, %s2b6dnnf : tensor<32x75264xf32>
    %s2b6dnve = stablehlo.add %s2b6dnvr, %s2b6dnep : tensor<32x75264xf32>
    %s2b6dnistd = stablehlo.rsqrt %s2b6dnve : tensor<32x75264xf32>
    %s2b6dnxh = stablehlo.multiply %s2b6dnxc, %s2b6dnistd : tensor<32x75264xf32>
    %s2b6dngb = stablehlo.broadcast_in_dim %s2b6ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b6dndxh = stablehlo.multiply %s2b6dngb, %s2b6dnrdy : tensor<32x75264xf32>
    %s2b6dnsdxr = stablehlo.reduce(%s2b6dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b6dnsdx = stablehlo.broadcast_in_dim %s2b6dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b6dnxd = stablehlo.multiply %s2b6dnxh, %s2b6dndxh : tensor<32x75264xf32>
    %s2b6dnsxdr = stablehlo.reduce(%s2b6dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b6dnsxd = stablehlo.broadcast_in_dim %s2b6dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b6dnt1 = stablehlo.multiply %s2b6dndxh, %s2b6dnnf : tensor<32x75264xf32>
    %s2b6dni1 = stablehlo.subtract %s2b6dnt1, %s2b6dnsdx : tensor<32x75264xf32>
    %s2b6dnxs = stablehlo.multiply %s2b6dnxh, %s2b6dnsxd : tensor<32x75264xf32>
    %s2b6dni2 = stablehlo.subtract %s2b6dni1, %s2b6dnxs : tensor<32x75264xf32>
    %s2b6dnsN = stablehlo.divide %s2b6dnistd, %s2b6dnnf : tensor<32x75264xf32>
    %s2b6dngin = stablehlo.multiply %s2b6dnsN, %s2b6dni2 : tensor<32x75264xf32>
    %s2b6dn = stablehlo.reshape %s2b6dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b6dndgp = stablehlo.multiply %s2b6dnrdy, %s2b6dnxh : tensor<32x75264xf32>
    %s2b6dndg = stablehlo.reduce(%s2b6dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b6dndb = stablehlo.reduce(%s2b6dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b6ddrev = stablehlo.reverse %s2b6dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b6dd = stablehlo.convolution(%s2b6dn, %s2b6ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b6ddWxt = stablehlo.transpose %s2b5o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b6ddWdt = stablehlo.transpose %s2b6dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b6ddWraw = stablehlo.convolution(%s2b6ddWxt, %s2b6ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b6ddW = stablehlo.reshape %s2b6ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b6ddb = stablehlo.reduce(%s2b6dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b6dx = stablehlo.add %s2b6dd, %s2b7dx : tensor<32x384x14x14xf32>
    %s2b5dlsgb = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b5dls = stablehlo.multiply %s2b5dlsgb, %s2b6dx : tensor<32x384x14x14xf32>
    %s2b5dlsxdy = stablehlo.multiply %s2b5p, %s2b6dx : tensor<32x384x14x14xf32>
    %s2b5dlsdg = stablehlo.reduce(%s2b5dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b5dpt = stablehlo.transpose %s2b5pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b5dp = stablehlo.convolution(%s2b5dls, %s2b5dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b5dpWxt = stablehlo.transpose %s2b5g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b5dpWdt = stablehlo.transpose %s2b5dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b5dpWraw = stablehlo.convolution(%s2b5dpWxt, %s2b5dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b5dpW = stablehlo.transpose %s2b5dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b5dpb = stablehlo.reduce(%s2b5dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b5dgx2 = stablehlo.multiply %s2b5e, %s2b5e : tensor<32x1536x14x14xf32>
    %s2b5dgx3 = stablehlo.multiply %s2b5dgx2, %s2b5e : tensor<32x1536x14x14xf32>
    %s2b5dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b5dgkx3 = stablehlo.multiply %s2b5dgck, %s2b5dgx3 : tensor<32x1536x14x14xf32>
    %s2b5dginn = stablehlo.add %s2b5e, %s2b5dgkx3 : tensor<32x1536x14x14xf32>
    %s2b5dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b5dgu = stablehlo.multiply %s2b5dgcs, %s2b5dginn : tensor<32x1536x14x14xf32>
    %s2b5dgt = stablehlo.tanh %s2b5dgu : tensor<32x1536x14x14xf32>
    %s2b5dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b5dgopt = stablehlo.add %s2b5dgone, %s2b5dgt : tensor<32x1536x14x14xf32>
    %s2b5dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b5dgterm1 = stablehlo.multiply %s2b5dghalf, %s2b5dgopt : tensor<32x1536x14x14xf32>
    %s2b5dgt2 = stablehlo.multiply %s2b5dgt, %s2b5dgt : tensor<32x1536x14x14xf32>
    %s2b5dgomt2 = stablehlo.subtract %s2b5dgone, %s2b5dgt2 : tensor<32x1536x14x14xf32>
    %s2b5dghx = stablehlo.multiply %s2b5dghalf, %s2b5e : tensor<32x1536x14x14xf32>
    %s2b5dghxo = stablehlo.multiply %s2b5dghx, %s2b5dgomt2 : tensor<32x1536x14x14xf32>
    %s2b5dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b5dga3x2 = stablehlo.multiply %s2b5dgc3b, %s2b5dgx2 : tensor<32x1536x14x14xf32>
    %s2b5dgin2 = stablehlo.add %s2b5dgone, %s2b5dga3x2 : tensor<32x1536x14x14xf32>
    %s2b5dgup = stablehlo.multiply %s2b5dgcs, %s2b5dgin2 : tensor<32x1536x14x14xf32>
    %s2b5dgterm2 = stablehlo.multiply %s2b5dghxo, %s2b5dgup : tensor<32x1536x14x14xf32>
    %s2b5dggp = stablehlo.add %s2b5dgterm1, %s2b5dgterm2 : tensor<32x1536x14x14xf32>
    %s2b5dg = stablehlo.multiply %s2b5dp, %s2b5dggp : tensor<32x1536x14x14xf32>
    %s2b5det = stablehlo.transpose %s2b5eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b5de = stablehlo.convolution(%s2b5dg, %s2b5det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b5deWxt = stablehlo.transpose %s2b5n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b5deWdt = stablehlo.transpose %s2b5dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b5deWraw = stablehlo.convolution(%s2b5deWxt, %s2b5deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b5deW = stablehlo.transpose %s2b5deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b5deb = stablehlo.reduce(%s2b5dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b5dnri = stablehlo.reshape %s2b5d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b5dnrdy = stablehlo.reshape %s2b5de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b5dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b5dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b5dnsmr = stablehlo.reduce(%s2b5dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b5dnsm = stablehlo.broadcast_in_dim %s2b5dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b5dnmu = stablehlo.divide %s2b5dnsm, %s2b5dnnf : tensor<32x75264xf32>
    %s2b5dnxc = stablehlo.subtract %s2b5dnri, %s2b5dnmu : tensor<32x75264xf32>
    %s2b5dnsq = stablehlo.multiply %s2b5dnxc, %s2b5dnxc : tensor<32x75264xf32>
    %s2b5dnvsr = stablehlo.reduce(%s2b5dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b5dnvs = stablehlo.broadcast_in_dim %s2b5dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b5dnvr = stablehlo.divide %s2b5dnvs, %s2b5dnnf : tensor<32x75264xf32>
    %s2b5dnve = stablehlo.add %s2b5dnvr, %s2b5dnep : tensor<32x75264xf32>
    %s2b5dnistd = stablehlo.rsqrt %s2b5dnve : tensor<32x75264xf32>
    %s2b5dnxh = stablehlo.multiply %s2b5dnxc, %s2b5dnistd : tensor<32x75264xf32>
    %s2b5dngb = stablehlo.broadcast_in_dim %s2b5ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b5dndxh = stablehlo.multiply %s2b5dngb, %s2b5dnrdy : tensor<32x75264xf32>
    %s2b5dnsdxr = stablehlo.reduce(%s2b5dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b5dnsdx = stablehlo.broadcast_in_dim %s2b5dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b5dnxd = stablehlo.multiply %s2b5dnxh, %s2b5dndxh : tensor<32x75264xf32>
    %s2b5dnsxdr = stablehlo.reduce(%s2b5dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b5dnsxd = stablehlo.broadcast_in_dim %s2b5dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b5dnt1 = stablehlo.multiply %s2b5dndxh, %s2b5dnnf : tensor<32x75264xf32>
    %s2b5dni1 = stablehlo.subtract %s2b5dnt1, %s2b5dnsdx : tensor<32x75264xf32>
    %s2b5dnxs = stablehlo.multiply %s2b5dnxh, %s2b5dnsxd : tensor<32x75264xf32>
    %s2b5dni2 = stablehlo.subtract %s2b5dni1, %s2b5dnxs : tensor<32x75264xf32>
    %s2b5dnsN = stablehlo.divide %s2b5dnistd, %s2b5dnnf : tensor<32x75264xf32>
    %s2b5dngin = stablehlo.multiply %s2b5dnsN, %s2b5dni2 : tensor<32x75264xf32>
    %s2b5dn = stablehlo.reshape %s2b5dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b5dndgp = stablehlo.multiply %s2b5dnrdy, %s2b5dnxh : tensor<32x75264xf32>
    %s2b5dndg = stablehlo.reduce(%s2b5dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b5dndb = stablehlo.reduce(%s2b5dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b5ddrev = stablehlo.reverse %s2b5dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b5dd = stablehlo.convolution(%s2b5dn, %s2b5ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b5ddWxt = stablehlo.transpose %s2b4o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b5ddWdt = stablehlo.transpose %s2b5dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b5ddWraw = stablehlo.convolution(%s2b5ddWxt, %s2b5ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b5ddW = stablehlo.reshape %s2b5ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b5ddb = stablehlo.reduce(%s2b5dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b5dx = stablehlo.add %s2b5dd, %s2b6dx : tensor<32x384x14x14xf32>
    %s2b4dlsgb = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b4dls = stablehlo.multiply %s2b4dlsgb, %s2b5dx : tensor<32x384x14x14xf32>
    %s2b4dlsxdy = stablehlo.multiply %s2b4p, %s2b5dx : tensor<32x384x14x14xf32>
    %s2b4dlsdg = stablehlo.reduce(%s2b4dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b4dpt = stablehlo.transpose %s2b4pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b4dp = stablehlo.convolution(%s2b4dls, %s2b4dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b4dpWxt = stablehlo.transpose %s2b4g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b4dpWdt = stablehlo.transpose %s2b4dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b4dpWraw = stablehlo.convolution(%s2b4dpWxt, %s2b4dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b4dpW = stablehlo.transpose %s2b4dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b4dpb = stablehlo.reduce(%s2b4dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b4dgx2 = stablehlo.multiply %s2b4e, %s2b4e : tensor<32x1536x14x14xf32>
    %s2b4dgx3 = stablehlo.multiply %s2b4dgx2, %s2b4e : tensor<32x1536x14x14xf32>
    %s2b4dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b4dgkx3 = stablehlo.multiply %s2b4dgck, %s2b4dgx3 : tensor<32x1536x14x14xf32>
    %s2b4dginn = stablehlo.add %s2b4e, %s2b4dgkx3 : tensor<32x1536x14x14xf32>
    %s2b4dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b4dgu = stablehlo.multiply %s2b4dgcs, %s2b4dginn : tensor<32x1536x14x14xf32>
    %s2b4dgt = stablehlo.tanh %s2b4dgu : tensor<32x1536x14x14xf32>
    %s2b4dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b4dgopt = stablehlo.add %s2b4dgone, %s2b4dgt : tensor<32x1536x14x14xf32>
    %s2b4dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b4dgterm1 = stablehlo.multiply %s2b4dghalf, %s2b4dgopt : tensor<32x1536x14x14xf32>
    %s2b4dgt2 = stablehlo.multiply %s2b4dgt, %s2b4dgt : tensor<32x1536x14x14xf32>
    %s2b4dgomt2 = stablehlo.subtract %s2b4dgone, %s2b4dgt2 : tensor<32x1536x14x14xf32>
    %s2b4dghx = stablehlo.multiply %s2b4dghalf, %s2b4e : tensor<32x1536x14x14xf32>
    %s2b4dghxo = stablehlo.multiply %s2b4dghx, %s2b4dgomt2 : tensor<32x1536x14x14xf32>
    %s2b4dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b4dga3x2 = stablehlo.multiply %s2b4dgc3b, %s2b4dgx2 : tensor<32x1536x14x14xf32>
    %s2b4dgin2 = stablehlo.add %s2b4dgone, %s2b4dga3x2 : tensor<32x1536x14x14xf32>
    %s2b4dgup = stablehlo.multiply %s2b4dgcs, %s2b4dgin2 : tensor<32x1536x14x14xf32>
    %s2b4dgterm2 = stablehlo.multiply %s2b4dghxo, %s2b4dgup : tensor<32x1536x14x14xf32>
    %s2b4dggp = stablehlo.add %s2b4dgterm1, %s2b4dgterm2 : tensor<32x1536x14x14xf32>
    %s2b4dg = stablehlo.multiply %s2b4dp, %s2b4dggp : tensor<32x1536x14x14xf32>
    %s2b4det = stablehlo.transpose %s2b4eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b4de = stablehlo.convolution(%s2b4dg, %s2b4det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b4deWxt = stablehlo.transpose %s2b4n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b4deWdt = stablehlo.transpose %s2b4dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b4deWraw = stablehlo.convolution(%s2b4deWxt, %s2b4deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b4deW = stablehlo.transpose %s2b4deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b4deb = stablehlo.reduce(%s2b4dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b4dnri = stablehlo.reshape %s2b4d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b4dnrdy = stablehlo.reshape %s2b4de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b4dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b4dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b4dnsmr = stablehlo.reduce(%s2b4dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b4dnsm = stablehlo.broadcast_in_dim %s2b4dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b4dnmu = stablehlo.divide %s2b4dnsm, %s2b4dnnf : tensor<32x75264xf32>
    %s2b4dnxc = stablehlo.subtract %s2b4dnri, %s2b4dnmu : tensor<32x75264xf32>
    %s2b4dnsq = stablehlo.multiply %s2b4dnxc, %s2b4dnxc : tensor<32x75264xf32>
    %s2b4dnvsr = stablehlo.reduce(%s2b4dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b4dnvs = stablehlo.broadcast_in_dim %s2b4dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b4dnvr = stablehlo.divide %s2b4dnvs, %s2b4dnnf : tensor<32x75264xf32>
    %s2b4dnve = stablehlo.add %s2b4dnvr, %s2b4dnep : tensor<32x75264xf32>
    %s2b4dnistd = stablehlo.rsqrt %s2b4dnve : tensor<32x75264xf32>
    %s2b4dnxh = stablehlo.multiply %s2b4dnxc, %s2b4dnistd : tensor<32x75264xf32>
    %s2b4dngb = stablehlo.broadcast_in_dim %s2b4ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b4dndxh = stablehlo.multiply %s2b4dngb, %s2b4dnrdy : tensor<32x75264xf32>
    %s2b4dnsdxr = stablehlo.reduce(%s2b4dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b4dnsdx = stablehlo.broadcast_in_dim %s2b4dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b4dnxd = stablehlo.multiply %s2b4dnxh, %s2b4dndxh : tensor<32x75264xf32>
    %s2b4dnsxdr = stablehlo.reduce(%s2b4dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b4dnsxd = stablehlo.broadcast_in_dim %s2b4dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b4dnt1 = stablehlo.multiply %s2b4dndxh, %s2b4dnnf : tensor<32x75264xf32>
    %s2b4dni1 = stablehlo.subtract %s2b4dnt1, %s2b4dnsdx : tensor<32x75264xf32>
    %s2b4dnxs = stablehlo.multiply %s2b4dnxh, %s2b4dnsxd : tensor<32x75264xf32>
    %s2b4dni2 = stablehlo.subtract %s2b4dni1, %s2b4dnxs : tensor<32x75264xf32>
    %s2b4dnsN = stablehlo.divide %s2b4dnistd, %s2b4dnnf : tensor<32x75264xf32>
    %s2b4dngin = stablehlo.multiply %s2b4dnsN, %s2b4dni2 : tensor<32x75264xf32>
    %s2b4dn = stablehlo.reshape %s2b4dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b4dndgp = stablehlo.multiply %s2b4dnrdy, %s2b4dnxh : tensor<32x75264xf32>
    %s2b4dndg = stablehlo.reduce(%s2b4dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b4dndb = stablehlo.reduce(%s2b4dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b4ddrev = stablehlo.reverse %s2b4dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b4dd = stablehlo.convolution(%s2b4dn, %s2b4ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b4ddWxt = stablehlo.transpose %s2b3o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b4ddWdt = stablehlo.transpose %s2b4dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b4ddWraw = stablehlo.convolution(%s2b4ddWxt, %s2b4ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b4ddW = stablehlo.reshape %s2b4ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b4ddb = stablehlo.reduce(%s2b4dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b4dx = stablehlo.add %s2b4dd, %s2b5dx : tensor<32x384x14x14xf32>
    %s2b3dlsgb = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b3dls = stablehlo.multiply %s2b3dlsgb, %s2b4dx : tensor<32x384x14x14xf32>
    %s2b3dlsxdy = stablehlo.multiply %s2b3p, %s2b4dx : tensor<32x384x14x14xf32>
    %s2b3dlsdg = stablehlo.reduce(%s2b3dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b3dpt = stablehlo.transpose %s2b3pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b3dp = stablehlo.convolution(%s2b3dls, %s2b3dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b3dpWxt = stablehlo.transpose %s2b3g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b3dpWdt = stablehlo.transpose %s2b3dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b3dpWraw = stablehlo.convolution(%s2b3dpWxt, %s2b3dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b3dpW = stablehlo.transpose %s2b3dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b3dpb = stablehlo.reduce(%s2b3dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b3dgx2 = stablehlo.multiply %s2b3e, %s2b3e : tensor<32x1536x14x14xf32>
    %s2b3dgx3 = stablehlo.multiply %s2b3dgx2, %s2b3e : tensor<32x1536x14x14xf32>
    %s2b3dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b3dgkx3 = stablehlo.multiply %s2b3dgck, %s2b3dgx3 : tensor<32x1536x14x14xf32>
    %s2b3dginn = stablehlo.add %s2b3e, %s2b3dgkx3 : tensor<32x1536x14x14xf32>
    %s2b3dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b3dgu = stablehlo.multiply %s2b3dgcs, %s2b3dginn : tensor<32x1536x14x14xf32>
    %s2b3dgt = stablehlo.tanh %s2b3dgu : tensor<32x1536x14x14xf32>
    %s2b3dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b3dgopt = stablehlo.add %s2b3dgone, %s2b3dgt : tensor<32x1536x14x14xf32>
    %s2b3dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b3dgterm1 = stablehlo.multiply %s2b3dghalf, %s2b3dgopt : tensor<32x1536x14x14xf32>
    %s2b3dgt2 = stablehlo.multiply %s2b3dgt, %s2b3dgt : tensor<32x1536x14x14xf32>
    %s2b3dgomt2 = stablehlo.subtract %s2b3dgone, %s2b3dgt2 : tensor<32x1536x14x14xf32>
    %s2b3dghx = stablehlo.multiply %s2b3dghalf, %s2b3e : tensor<32x1536x14x14xf32>
    %s2b3dghxo = stablehlo.multiply %s2b3dghx, %s2b3dgomt2 : tensor<32x1536x14x14xf32>
    %s2b3dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b3dga3x2 = stablehlo.multiply %s2b3dgc3b, %s2b3dgx2 : tensor<32x1536x14x14xf32>
    %s2b3dgin2 = stablehlo.add %s2b3dgone, %s2b3dga3x2 : tensor<32x1536x14x14xf32>
    %s2b3dgup = stablehlo.multiply %s2b3dgcs, %s2b3dgin2 : tensor<32x1536x14x14xf32>
    %s2b3dgterm2 = stablehlo.multiply %s2b3dghxo, %s2b3dgup : tensor<32x1536x14x14xf32>
    %s2b3dggp = stablehlo.add %s2b3dgterm1, %s2b3dgterm2 : tensor<32x1536x14x14xf32>
    %s2b3dg = stablehlo.multiply %s2b3dp, %s2b3dggp : tensor<32x1536x14x14xf32>
    %s2b3det = stablehlo.transpose %s2b3eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b3de = stablehlo.convolution(%s2b3dg, %s2b3det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b3deWxt = stablehlo.transpose %s2b3n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b3deWdt = stablehlo.transpose %s2b3dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b3deWraw = stablehlo.convolution(%s2b3deWxt, %s2b3deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b3deW = stablehlo.transpose %s2b3deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b3deb = stablehlo.reduce(%s2b3dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b3dnri = stablehlo.reshape %s2b3d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b3dnrdy = stablehlo.reshape %s2b3de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b3dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b3dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b3dnsmr = stablehlo.reduce(%s2b3dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b3dnsm = stablehlo.broadcast_in_dim %s2b3dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b3dnmu = stablehlo.divide %s2b3dnsm, %s2b3dnnf : tensor<32x75264xf32>
    %s2b3dnxc = stablehlo.subtract %s2b3dnri, %s2b3dnmu : tensor<32x75264xf32>
    %s2b3dnsq = stablehlo.multiply %s2b3dnxc, %s2b3dnxc : tensor<32x75264xf32>
    %s2b3dnvsr = stablehlo.reduce(%s2b3dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b3dnvs = stablehlo.broadcast_in_dim %s2b3dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b3dnvr = stablehlo.divide %s2b3dnvs, %s2b3dnnf : tensor<32x75264xf32>
    %s2b3dnve = stablehlo.add %s2b3dnvr, %s2b3dnep : tensor<32x75264xf32>
    %s2b3dnistd = stablehlo.rsqrt %s2b3dnve : tensor<32x75264xf32>
    %s2b3dnxh = stablehlo.multiply %s2b3dnxc, %s2b3dnistd : tensor<32x75264xf32>
    %s2b3dngb = stablehlo.broadcast_in_dim %s2b3ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b3dndxh = stablehlo.multiply %s2b3dngb, %s2b3dnrdy : tensor<32x75264xf32>
    %s2b3dnsdxr = stablehlo.reduce(%s2b3dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b3dnsdx = stablehlo.broadcast_in_dim %s2b3dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b3dnxd = stablehlo.multiply %s2b3dnxh, %s2b3dndxh : tensor<32x75264xf32>
    %s2b3dnsxdr = stablehlo.reduce(%s2b3dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b3dnsxd = stablehlo.broadcast_in_dim %s2b3dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b3dnt1 = stablehlo.multiply %s2b3dndxh, %s2b3dnnf : tensor<32x75264xf32>
    %s2b3dni1 = stablehlo.subtract %s2b3dnt1, %s2b3dnsdx : tensor<32x75264xf32>
    %s2b3dnxs = stablehlo.multiply %s2b3dnxh, %s2b3dnsxd : tensor<32x75264xf32>
    %s2b3dni2 = stablehlo.subtract %s2b3dni1, %s2b3dnxs : tensor<32x75264xf32>
    %s2b3dnsN = stablehlo.divide %s2b3dnistd, %s2b3dnnf : tensor<32x75264xf32>
    %s2b3dngin = stablehlo.multiply %s2b3dnsN, %s2b3dni2 : tensor<32x75264xf32>
    %s2b3dn = stablehlo.reshape %s2b3dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b3dndgp = stablehlo.multiply %s2b3dnrdy, %s2b3dnxh : tensor<32x75264xf32>
    %s2b3dndg = stablehlo.reduce(%s2b3dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b3dndb = stablehlo.reduce(%s2b3dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b3ddrev = stablehlo.reverse %s2b3dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b3dd = stablehlo.convolution(%s2b3dn, %s2b3ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b3ddWxt = stablehlo.transpose %s2b2o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b3ddWdt = stablehlo.transpose %s2b3dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b3ddWraw = stablehlo.convolution(%s2b3ddWxt, %s2b3ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b3ddW = stablehlo.reshape %s2b3ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b3ddb = stablehlo.reduce(%s2b3dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b3dx = stablehlo.add %s2b3dd, %s2b4dx : tensor<32x384x14x14xf32>
    %s2b2dlsgb = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b2dls = stablehlo.multiply %s2b2dlsgb, %s2b3dx : tensor<32x384x14x14xf32>
    %s2b2dlsxdy = stablehlo.multiply %s2b2p, %s2b3dx : tensor<32x384x14x14xf32>
    %s2b2dlsdg = stablehlo.reduce(%s2b2dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b2dpt = stablehlo.transpose %s2b2pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b2dp = stablehlo.convolution(%s2b2dls, %s2b2dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b2dpWxt = stablehlo.transpose %s2b2g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b2dpWdt = stablehlo.transpose %s2b2dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b2dpWraw = stablehlo.convolution(%s2b2dpWxt, %s2b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b2dpW = stablehlo.transpose %s2b2dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b2dpb = stablehlo.reduce(%s2b2dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b2dgx2 = stablehlo.multiply %s2b2e, %s2b2e : tensor<32x1536x14x14xf32>
    %s2b2dgx3 = stablehlo.multiply %s2b2dgx2, %s2b2e : tensor<32x1536x14x14xf32>
    %s2b2dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b2dgkx3 = stablehlo.multiply %s2b2dgck, %s2b2dgx3 : tensor<32x1536x14x14xf32>
    %s2b2dginn = stablehlo.add %s2b2e, %s2b2dgkx3 : tensor<32x1536x14x14xf32>
    %s2b2dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b2dgu = stablehlo.multiply %s2b2dgcs, %s2b2dginn : tensor<32x1536x14x14xf32>
    %s2b2dgt = stablehlo.tanh %s2b2dgu : tensor<32x1536x14x14xf32>
    %s2b2dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b2dgopt = stablehlo.add %s2b2dgone, %s2b2dgt : tensor<32x1536x14x14xf32>
    %s2b2dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b2dgterm1 = stablehlo.multiply %s2b2dghalf, %s2b2dgopt : tensor<32x1536x14x14xf32>
    %s2b2dgt2 = stablehlo.multiply %s2b2dgt, %s2b2dgt : tensor<32x1536x14x14xf32>
    %s2b2dgomt2 = stablehlo.subtract %s2b2dgone, %s2b2dgt2 : tensor<32x1536x14x14xf32>
    %s2b2dghx = stablehlo.multiply %s2b2dghalf, %s2b2e : tensor<32x1536x14x14xf32>
    %s2b2dghxo = stablehlo.multiply %s2b2dghx, %s2b2dgomt2 : tensor<32x1536x14x14xf32>
    %s2b2dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b2dga3x2 = stablehlo.multiply %s2b2dgc3b, %s2b2dgx2 : tensor<32x1536x14x14xf32>
    %s2b2dgin2 = stablehlo.add %s2b2dgone, %s2b2dga3x2 : tensor<32x1536x14x14xf32>
    %s2b2dgup = stablehlo.multiply %s2b2dgcs, %s2b2dgin2 : tensor<32x1536x14x14xf32>
    %s2b2dgterm2 = stablehlo.multiply %s2b2dghxo, %s2b2dgup : tensor<32x1536x14x14xf32>
    %s2b2dggp = stablehlo.add %s2b2dgterm1, %s2b2dgterm2 : tensor<32x1536x14x14xf32>
    %s2b2dg = stablehlo.multiply %s2b2dp, %s2b2dggp : tensor<32x1536x14x14xf32>
    %s2b2det = stablehlo.transpose %s2b2eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b2de = stablehlo.convolution(%s2b2dg, %s2b2det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b2deWxt = stablehlo.transpose %s2b2n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b2deWdt = stablehlo.transpose %s2b2dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b2deWraw = stablehlo.convolution(%s2b2deWxt, %s2b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b2deW = stablehlo.transpose %s2b2deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b2deb = stablehlo.reduce(%s2b2dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b2dnri = stablehlo.reshape %s2b2d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b2dnrdy = stablehlo.reshape %s2b2de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b2dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b2dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b2dnsmr = stablehlo.reduce(%s2b2dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b2dnsm = stablehlo.broadcast_in_dim %s2b2dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b2dnmu = stablehlo.divide %s2b2dnsm, %s2b2dnnf : tensor<32x75264xf32>
    %s2b2dnxc = stablehlo.subtract %s2b2dnri, %s2b2dnmu : tensor<32x75264xf32>
    %s2b2dnsq = stablehlo.multiply %s2b2dnxc, %s2b2dnxc : tensor<32x75264xf32>
    %s2b2dnvsr = stablehlo.reduce(%s2b2dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b2dnvs = stablehlo.broadcast_in_dim %s2b2dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b2dnvr = stablehlo.divide %s2b2dnvs, %s2b2dnnf : tensor<32x75264xf32>
    %s2b2dnve = stablehlo.add %s2b2dnvr, %s2b2dnep : tensor<32x75264xf32>
    %s2b2dnistd = stablehlo.rsqrt %s2b2dnve : tensor<32x75264xf32>
    %s2b2dnxh = stablehlo.multiply %s2b2dnxc, %s2b2dnistd : tensor<32x75264xf32>
    %s2b2dngb = stablehlo.broadcast_in_dim %s2b2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b2dndxh = stablehlo.multiply %s2b2dngb, %s2b2dnrdy : tensor<32x75264xf32>
    %s2b2dnsdxr = stablehlo.reduce(%s2b2dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b2dnsdx = stablehlo.broadcast_in_dim %s2b2dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b2dnxd = stablehlo.multiply %s2b2dnxh, %s2b2dndxh : tensor<32x75264xf32>
    %s2b2dnsxdr = stablehlo.reduce(%s2b2dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b2dnsxd = stablehlo.broadcast_in_dim %s2b2dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b2dnt1 = stablehlo.multiply %s2b2dndxh, %s2b2dnnf : tensor<32x75264xf32>
    %s2b2dni1 = stablehlo.subtract %s2b2dnt1, %s2b2dnsdx : tensor<32x75264xf32>
    %s2b2dnxs = stablehlo.multiply %s2b2dnxh, %s2b2dnsxd : tensor<32x75264xf32>
    %s2b2dni2 = stablehlo.subtract %s2b2dni1, %s2b2dnxs : tensor<32x75264xf32>
    %s2b2dnsN = stablehlo.divide %s2b2dnistd, %s2b2dnnf : tensor<32x75264xf32>
    %s2b2dngin = stablehlo.multiply %s2b2dnsN, %s2b2dni2 : tensor<32x75264xf32>
    %s2b2dn = stablehlo.reshape %s2b2dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b2dndgp = stablehlo.multiply %s2b2dnrdy, %s2b2dnxh : tensor<32x75264xf32>
    %s2b2dndg = stablehlo.reduce(%s2b2dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b2dndb = stablehlo.reduce(%s2b2dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b2ddrev = stablehlo.reverse %s2b2dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b2dd = stablehlo.convolution(%s2b2dn, %s2b2ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b2ddWxt = stablehlo.transpose %s2b1o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b2ddWdt = stablehlo.transpose %s2b2dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b2ddWraw = stablehlo.convolution(%s2b2ddWxt, %s2b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b2ddW = stablehlo.reshape %s2b2ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b2ddb = stablehlo.reduce(%s2b2dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b2dx = stablehlo.add %s2b2dd, %s2b3dx : tensor<32x384x14x14xf32>
    %s2b1dlsgb = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b1dls = stablehlo.multiply %s2b1dlsgb, %s2b2dx : tensor<32x384x14x14xf32>
    %s2b1dlsxdy = stablehlo.multiply %s2b1p, %s2b2dx : tensor<32x384x14x14xf32>
    %s2b1dlsdg = stablehlo.reduce(%s2b1dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b1dpt = stablehlo.transpose %s2b1pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b1dp = stablehlo.convolution(%s2b1dls, %s2b1dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b1dpWxt = stablehlo.transpose %s2b1g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b1dpWdt = stablehlo.transpose %s2b1dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b1dpWraw = stablehlo.convolution(%s2b1dpWxt, %s2b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b1dpW = stablehlo.transpose %s2b1dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b1dpb = stablehlo.reduce(%s2b1dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b1dgx2 = stablehlo.multiply %s2b1e, %s2b1e : tensor<32x1536x14x14xf32>
    %s2b1dgx3 = stablehlo.multiply %s2b1dgx2, %s2b1e : tensor<32x1536x14x14xf32>
    %s2b1dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b1dgkx3 = stablehlo.multiply %s2b1dgck, %s2b1dgx3 : tensor<32x1536x14x14xf32>
    %s2b1dginn = stablehlo.add %s2b1e, %s2b1dgkx3 : tensor<32x1536x14x14xf32>
    %s2b1dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b1dgu = stablehlo.multiply %s2b1dgcs, %s2b1dginn : tensor<32x1536x14x14xf32>
    %s2b1dgt = stablehlo.tanh %s2b1dgu : tensor<32x1536x14x14xf32>
    %s2b1dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b1dgopt = stablehlo.add %s2b1dgone, %s2b1dgt : tensor<32x1536x14x14xf32>
    %s2b1dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b1dgterm1 = stablehlo.multiply %s2b1dghalf, %s2b1dgopt : tensor<32x1536x14x14xf32>
    %s2b1dgt2 = stablehlo.multiply %s2b1dgt, %s2b1dgt : tensor<32x1536x14x14xf32>
    %s2b1dgomt2 = stablehlo.subtract %s2b1dgone, %s2b1dgt2 : tensor<32x1536x14x14xf32>
    %s2b1dghx = stablehlo.multiply %s2b1dghalf, %s2b1e : tensor<32x1536x14x14xf32>
    %s2b1dghxo = stablehlo.multiply %s2b1dghx, %s2b1dgomt2 : tensor<32x1536x14x14xf32>
    %s2b1dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b1dga3x2 = stablehlo.multiply %s2b1dgc3b, %s2b1dgx2 : tensor<32x1536x14x14xf32>
    %s2b1dgin2 = stablehlo.add %s2b1dgone, %s2b1dga3x2 : tensor<32x1536x14x14xf32>
    %s2b1dgup = stablehlo.multiply %s2b1dgcs, %s2b1dgin2 : tensor<32x1536x14x14xf32>
    %s2b1dgterm2 = stablehlo.multiply %s2b1dghxo, %s2b1dgup : tensor<32x1536x14x14xf32>
    %s2b1dggp = stablehlo.add %s2b1dgterm1, %s2b1dgterm2 : tensor<32x1536x14x14xf32>
    %s2b1dg = stablehlo.multiply %s2b1dp, %s2b1dggp : tensor<32x1536x14x14xf32>
    %s2b1det = stablehlo.transpose %s2b1eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b1de = stablehlo.convolution(%s2b1dg, %s2b1det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b1deWxt = stablehlo.transpose %s2b1n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b1deWdt = stablehlo.transpose %s2b1dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b1deWraw = stablehlo.convolution(%s2b1deWxt, %s2b1deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b1deW = stablehlo.transpose %s2b1deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b1deb = stablehlo.reduce(%s2b1dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b1dnri = stablehlo.reshape %s2b1d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b1dnrdy = stablehlo.reshape %s2b1de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b1dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b1dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b1dnsmr = stablehlo.reduce(%s2b1dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b1dnsm = stablehlo.broadcast_in_dim %s2b1dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b1dnmu = stablehlo.divide %s2b1dnsm, %s2b1dnnf : tensor<32x75264xf32>
    %s2b1dnxc = stablehlo.subtract %s2b1dnri, %s2b1dnmu : tensor<32x75264xf32>
    %s2b1dnsq = stablehlo.multiply %s2b1dnxc, %s2b1dnxc : tensor<32x75264xf32>
    %s2b1dnvsr = stablehlo.reduce(%s2b1dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b1dnvs = stablehlo.broadcast_in_dim %s2b1dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b1dnvr = stablehlo.divide %s2b1dnvs, %s2b1dnnf : tensor<32x75264xf32>
    %s2b1dnve = stablehlo.add %s2b1dnvr, %s2b1dnep : tensor<32x75264xf32>
    %s2b1dnistd = stablehlo.rsqrt %s2b1dnve : tensor<32x75264xf32>
    %s2b1dnxh = stablehlo.multiply %s2b1dnxc, %s2b1dnistd : tensor<32x75264xf32>
    %s2b1dngb = stablehlo.broadcast_in_dim %s2b1ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b1dndxh = stablehlo.multiply %s2b1dngb, %s2b1dnrdy : tensor<32x75264xf32>
    %s2b1dnsdxr = stablehlo.reduce(%s2b1dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b1dnsdx = stablehlo.broadcast_in_dim %s2b1dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b1dnxd = stablehlo.multiply %s2b1dnxh, %s2b1dndxh : tensor<32x75264xf32>
    %s2b1dnsxdr = stablehlo.reduce(%s2b1dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b1dnsxd = stablehlo.broadcast_in_dim %s2b1dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b1dnt1 = stablehlo.multiply %s2b1dndxh, %s2b1dnnf : tensor<32x75264xf32>
    %s2b1dni1 = stablehlo.subtract %s2b1dnt1, %s2b1dnsdx : tensor<32x75264xf32>
    %s2b1dnxs = stablehlo.multiply %s2b1dnxh, %s2b1dnsxd : tensor<32x75264xf32>
    %s2b1dni2 = stablehlo.subtract %s2b1dni1, %s2b1dnxs : tensor<32x75264xf32>
    %s2b1dnsN = stablehlo.divide %s2b1dnistd, %s2b1dnnf : tensor<32x75264xf32>
    %s2b1dngin = stablehlo.multiply %s2b1dnsN, %s2b1dni2 : tensor<32x75264xf32>
    %s2b1dn = stablehlo.reshape %s2b1dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b1dndgp = stablehlo.multiply %s2b1dnrdy, %s2b1dnxh : tensor<32x75264xf32>
    %s2b1dndg = stablehlo.reduce(%s2b1dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b1dndb = stablehlo.reduce(%s2b1dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b1ddrev = stablehlo.reverse %s2b1dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b1dd = stablehlo.convolution(%s2b1dn, %s2b1ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b1ddWxt = stablehlo.transpose %s2b0o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b1ddWdt = stablehlo.transpose %s2b1dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b1ddWraw = stablehlo.convolution(%s2b1ddWxt, %s2b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b1ddW = stablehlo.reshape %s2b1ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b1ddb = stablehlo.reduce(%s2b1dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b1dx = stablehlo.add %s2b1dd, %s2b2dx : tensor<32x384x14x14xf32>
    %s2b0dlsgb = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b0dls = stablehlo.multiply %s2b0dlsgb, %s2b1dx : tensor<32x384x14x14xf32>
    %s2b0dlsxdy = stablehlo.multiply %s2b0p, %s2b1dx : tensor<32x384x14x14xf32>
    %s2b0dlsdg = stablehlo.reduce(%s2b0dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b0dpt = stablehlo.transpose %s2b0pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b0dp = stablehlo.convolution(%s2b0dls, %s2b0dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b0dpWxt = stablehlo.transpose %s2b0g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b0dpWdt = stablehlo.transpose %s2b0dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b0dpWraw = stablehlo.convolution(%s2b0dpWxt, %s2b0dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b0dpW = stablehlo.transpose %s2b0dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b0dpb = stablehlo.reduce(%s2b0dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b0dgx2 = stablehlo.multiply %s2b0e, %s2b0e : tensor<32x1536x14x14xf32>
    %s2b0dgx3 = stablehlo.multiply %s2b0dgx2, %s2b0e : tensor<32x1536x14x14xf32>
    %s2b0dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b0dgkx3 = stablehlo.multiply %s2b0dgck, %s2b0dgx3 : tensor<32x1536x14x14xf32>
    %s2b0dginn = stablehlo.add %s2b0e, %s2b0dgkx3 : tensor<32x1536x14x14xf32>
    %s2b0dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b0dgu = stablehlo.multiply %s2b0dgcs, %s2b0dginn : tensor<32x1536x14x14xf32>
    %s2b0dgt = stablehlo.tanh %s2b0dgu : tensor<32x1536x14x14xf32>
    %s2b0dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b0dgopt = stablehlo.add %s2b0dgone, %s2b0dgt : tensor<32x1536x14x14xf32>
    %s2b0dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b0dgterm1 = stablehlo.multiply %s2b0dghalf, %s2b0dgopt : tensor<32x1536x14x14xf32>
    %s2b0dgt2 = stablehlo.multiply %s2b0dgt, %s2b0dgt : tensor<32x1536x14x14xf32>
    %s2b0dgomt2 = stablehlo.subtract %s2b0dgone, %s2b0dgt2 : tensor<32x1536x14x14xf32>
    %s2b0dghx = stablehlo.multiply %s2b0dghalf, %s2b0e : tensor<32x1536x14x14xf32>
    %s2b0dghxo = stablehlo.multiply %s2b0dghx, %s2b0dgomt2 : tensor<32x1536x14x14xf32>
    %s2b0dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b0dga3x2 = stablehlo.multiply %s2b0dgc3b, %s2b0dgx2 : tensor<32x1536x14x14xf32>
    %s2b0dgin2 = stablehlo.add %s2b0dgone, %s2b0dga3x2 : tensor<32x1536x14x14xf32>
    %s2b0dgup = stablehlo.multiply %s2b0dgcs, %s2b0dgin2 : tensor<32x1536x14x14xf32>
    %s2b0dgterm2 = stablehlo.multiply %s2b0dghxo, %s2b0dgup : tensor<32x1536x14x14xf32>
    %s2b0dggp = stablehlo.add %s2b0dgterm1, %s2b0dgterm2 : tensor<32x1536x14x14xf32>
    %s2b0dg = stablehlo.multiply %s2b0dp, %s2b0dggp : tensor<32x1536x14x14xf32>
    %s2b0det = stablehlo.transpose %s2b0eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b0de = stablehlo.convolution(%s2b0dg, %s2b0det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b0deWxt = stablehlo.transpose %s2b0n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b0deWdt = stablehlo.transpose %s2b0dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b0deWraw = stablehlo.convolution(%s2b0deWxt, %s2b0deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b0deW = stablehlo.transpose %s2b0deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b0deb = stablehlo.reduce(%s2b0dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b0dnri = stablehlo.reshape %s2b0d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b0dnrdy = stablehlo.reshape %s2b0de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b0dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b0dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b0dnsmr = stablehlo.reduce(%s2b0dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b0dnsm = stablehlo.broadcast_in_dim %s2b0dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b0dnmu = stablehlo.divide %s2b0dnsm, %s2b0dnnf : tensor<32x75264xf32>
    %s2b0dnxc = stablehlo.subtract %s2b0dnri, %s2b0dnmu : tensor<32x75264xf32>
    %s2b0dnsq = stablehlo.multiply %s2b0dnxc, %s2b0dnxc : tensor<32x75264xf32>
    %s2b0dnvsr = stablehlo.reduce(%s2b0dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b0dnvs = stablehlo.broadcast_in_dim %s2b0dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b0dnvr = stablehlo.divide %s2b0dnvs, %s2b0dnnf : tensor<32x75264xf32>
    %s2b0dnve = stablehlo.add %s2b0dnvr, %s2b0dnep : tensor<32x75264xf32>
    %s2b0dnistd = stablehlo.rsqrt %s2b0dnve : tensor<32x75264xf32>
    %s2b0dnxh = stablehlo.multiply %s2b0dnxc, %s2b0dnistd : tensor<32x75264xf32>
    %s2b0dngb = stablehlo.broadcast_in_dim %s2b0ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b0dndxh = stablehlo.multiply %s2b0dngb, %s2b0dnrdy : tensor<32x75264xf32>
    %s2b0dnsdxr = stablehlo.reduce(%s2b0dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b0dnsdx = stablehlo.broadcast_in_dim %s2b0dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b0dnxd = stablehlo.multiply %s2b0dnxh, %s2b0dndxh : tensor<32x75264xf32>
    %s2b0dnsxdr = stablehlo.reduce(%s2b0dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b0dnsxd = stablehlo.broadcast_in_dim %s2b0dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b0dnt1 = stablehlo.multiply %s2b0dndxh, %s2b0dnnf : tensor<32x75264xf32>
    %s2b0dni1 = stablehlo.subtract %s2b0dnt1, %s2b0dnsdx : tensor<32x75264xf32>
    %s2b0dnxs = stablehlo.multiply %s2b0dnxh, %s2b0dnsxd : tensor<32x75264xf32>
    %s2b0dni2 = stablehlo.subtract %s2b0dni1, %s2b0dnxs : tensor<32x75264xf32>
    %s2b0dnsN = stablehlo.divide %s2b0dnistd, %s2b0dnnf : tensor<32x75264xf32>
    %s2b0dngin = stablehlo.multiply %s2b0dnsN, %s2b0dni2 : tensor<32x75264xf32>
    %s2b0dn = stablehlo.reshape %s2b0dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b0dndgp = stablehlo.multiply %s2b0dnrdy, %s2b0dnxh : tensor<32x75264xf32>
    %s2b0dndg = stablehlo.reduce(%s2b0dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b0dndb = stablehlo.reduce(%s2b0dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b0ddrev = stablehlo.reverse %s2b0dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b0dd = stablehlo.convolution(%s2b0dn, %s2b0ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b0ddWxt = stablehlo.transpose %d1c, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b0ddWdt = stablehlo.transpose %s2b0dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b0ddWraw = stablehlo.convolution(%s2b0ddWxt, %s2b0ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b0ddW = stablehlo.reshape %s2b0ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b0ddb = stablehlo.reduce(%s2b0dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b0dx = stablehlo.add %s2b0dd, %s2b1dx : tensor<32x384x14x14xf32>
    %d1dcu = stablehlo.pad %s2b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x28x28xf32>
    %d1dct = stablehlo.transpose %d1W, dims = [1, 0, 2, 3] : (tensor<384x192x2x2xf32>) -> tensor<192x384x2x2xf32>
    %d1dcr = stablehlo.reverse %d1dct, dims = [2, 3] : tensor<192x384x2x2xf32>
    %d1dc = stablehlo.convolution(%d1dcu, %d1dcr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x28x28xf32>, tensor<192x384x2x2xf32>) -> tensor<32x192x28x28xf32>
    %d1dWu = stablehlo.pad %s2b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x27x27xf32>
    %d1dWxt = stablehlo.transpose %d1n, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %d1dWdt = stablehlo.transpose %d1dWu, dims = [1, 0, 2, 3] : (tensor<32x384x27x27xf32>) -> tensor<384x32x27x27xf32>
    %d1dWraw = stablehlo.convolution(%d1dWxt, %d1dWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<384x32x27x27xf32>) -> tensor<192x384x2x2xf32>
    %d1dW = stablehlo.transpose %d1dWraw, dims = [1, 0, 2, 3] : (tensor<192x384x2x2xf32>) -> tensor<384x192x2x2xf32>
    %d1db = stablehlo.reduce(%s2b0dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %d1dnri = stablehlo.reshape %s1b2o : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %d1dnrdy = stablehlo.reshape %d1dc : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %d1dnnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %d1dnep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %d1dnsmr = stablehlo.reduce(%d1dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %d1dnsm = stablehlo.broadcast_in_dim %d1dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %d1dnmu = stablehlo.divide %d1dnsm, %d1dnnf : tensor<32x150528xf32>
    %d1dnxc = stablehlo.subtract %d1dnri, %d1dnmu : tensor<32x150528xf32>
    %d1dnsq = stablehlo.multiply %d1dnxc, %d1dnxc : tensor<32x150528xf32>
    %d1dnvsr = stablehlo.reduce(%d1dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %d1dnvs = stablehlo.broadcast_in_dim %d1dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %d1dnvr = stablehlo.divide %d1dnvs, %d1dnnf : tensor<32x150528xf32>
    %d1dnve = stablehlo.add %d1dnvr, %d1dnep : tensor<32x150528xf32>
    %d1dnistd = stablehlo.rsqrt %d1dnve : tensor<32x150528xf32>
    %d1dnxh = stablehlo.multiply %d1dnxc, %d1dnistd : tensor<32x150528xf32>
    %d1dngb = stablehlo.broadcast_in_dim %d1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %d1dndxh = stablehlo.multiply %d1dngb, %d1dnrdy : tensor<32x150528xf32>
    %d1dnsdxr = stablehlo.reduce(%d1dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %d1dnsdx = stablehlo.broadcast_in_dim %d1dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %d1dnxd = stablehlo.multiply %d1dnxh, %d1dndxh : tensor<32x150528xf32>
    %d1dnsxdr = stablehlo.reduce(%d1dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %d1dnsxd = stablehlo.broadcast_in_dim %d1dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %d1dnt1 = stablehlo.multiply %d1dndxh, %d1dnnf : tensor<32x150528xf32>
    %d1dni1 = stablehlo.subtract %d1dnt1, %d1dnsdx : tensor<32x150528xf32>
    %d1dnxs = stablehlo.multiply %d1dnxh, %d1dnsxd : tensor<32x150528xf32>
    %d1dni2 = stablehlo.subtract %d1dni1, %d1dnxs : tensor<32x150528xf32>
    %d1dnsN = stablehlo.divide %d1dnistd, %d1dnnf : tensor<32x150528xf32>
    %d1dngin = stablehlo.multiply %d1dnsN, %d1dni2 : tensor<32x150528xf32>
    %d1dn = stablehlo.reshape %d1dngin : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %d1dndgp = stablehlo.multiply %d1dnrdy, %d1dnxh : tensor<32x150528xf32>
    %d1dndg = stablehlo.reduce(%d1dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %d1dndb = stablehlo.reduce(%d1dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b2dlsgb = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b2dls = stablehlo.multiply %s1b2dlsgb, %d1dn : tensor<32x192x28x28xf32>
    %s1b2dlsxdy = stablehlo.multiply %s1b2p, %d1dn : tensor<32x192x28x28xf32>
    %s1b2dlsdg = stablehlo.reduce(%s1b2dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b2dpt = stablehlo.transpose %s1b2pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %s1b2dp = stablehlo.convolution(%s1b2dls, %s1b2dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %s1b2dpWxt = stablehlo.transpose %s1b2g, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %s1b2dpWdt = stablehlo.transpose %s1b2dls, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b2dpWraw = stablehlo.convolution(%s1b2dpWxt, %s1b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %s1b2dpW = stablehlo.transpose %s1b2dpWraw, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %s1b2dpb = stablehlo.reduce(%s1b2dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b2dgx2 = stablehlo.multiply %s1b2e, %s1b2e : tensor<32x768x28x28xf32>
    %s1b2dgx3 = stablehlo.multiply %s1b2dgx2, %s1b2e : tensor<32x768x28x28xf32>
    %s1b2dgck = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %s1b2dgkx3 = stablehlo.multiply %s1b2dgck, %s1b2dgx3 : tensor<32x768x28x28xf32>
    %s1b2dginn = stablehlo.add %s1b2e, %s1b2dgkx3 : tensor<32x768x28x28xf32>
    %s1b2dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %s1b2dgu = stablehlo.multiply %s1b2dgcs, %s1b2dginn : tensor<32x768x28x28xf32>
    %s1b2dgt = stablehlo.tanh %s1b2dgu : tensor<32x768x28x28xf32>
    %s1b2dgone = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %s1b2dgopt = stablehlo.add %s1b2dgone, %s1b2dgt : tensor<32x768x28x28xf32>
    %s1b2dghalf = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %s1b2dgterm1 = stablehlo.multiply %s1b2dghalf, %s1b2dgopt : tensor<32x768x28x28xf32>
    %s1b2dgt2 = stablehlo.multiply %s1b2dgt, %s1b2dgt : tensor<32x768x28x28xf32>
    %s1b2dgomt2 = stablehlo.subtract %s1b2dgone, %s1b2dgt2 : tensor<32x768x28x28xf32>
    %s1b2dghx = stablehlo.multiply %s1b2dghalf, %s1b2e : tensor<32x768x28x28xf32>
    %s1b2dghxo = stablehlo.multiply %s1b2dghx, %s1b2dgomt2 : tensor<32x768x28x28xf32>
    %s1b2dgc3b = stablehlo.constant dense<0.134145> : tensor<32x768x28x28xf32>
    %s1b2dga3x2 = stablehlo.multiply %s1b2dgc3b, %s1b2dgx2 : tensor<32x768x28x28xf32>
    %s1b2dgin2 = stablehlo.add %s1b2dgone, %s1b2dga3x2 : tensor<32x768x28x28xf32>
    %s1b2dgup = stablehlo.multiply %s1b2dgcs, %s1b2dgin2 : tensor<32x768x28x28xf32>
    %s1b2dgterm2 = stablehlo.multiply %s1b2dghxo, %s1b2dgup : tensor<32x768x28x28xf32>
    %s1b2dggp = stablehlo.add %s1b2dgterm1, %s1b2dgterm2 : tensor<32x768x28x28xf32>
    %s1b2dg = stablehlo.multiply %s1b2dp, %s1b2dggp : tensor<32x768x28x28xf32>
    %s1b2det = stablehlo.transpose %s1b2eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %s1b2de = stablehlo.convolution(%s1b2dg, %s1b2det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %s1b2deWxt = stablehlo.transpose %s1b2n, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b2deWdt = stablehlo.transpose %s1b2dg, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %s1b2deWraw = stablehlo.convolution(%s1b2deWxt, %s1b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %s1b2deW = stablehlo.transpose %s1b2deWraw, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %s1b2deb = stablehlo.reduce(%s1b2dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %s1b2dnri = stablehlo.reshape %s1b2d : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b2dnrdy = stablehlo.reshape %s1b2de : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b2dnnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %s1b2dnep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %s1b2dnsmr = stablehlo.reduce(%s1b2dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b2dnsm = stablehlo.broadcast_in_dim %s1b2dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b2dnmu = stablehlo.divide %s1b2dnsm, %s1b2dnnf : tensor<32x150528xf32>
    %s1b2dnxc = stablehlo.subtract %s1b2dnri, %s1b2dnmu : tensor<32x150528xf32>
    %s1b2dnsq = stablehlo.multiply %s1b2dnxc, %s1b2dnxc : tensor<32x150528xf32>
    %s1b2dnvsr = stablehlo.reduce(%s1b2dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b2dnvs = stablehlo.broadcast_in_dim %s1b2dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b2dnvr = stablehlo.divide %s1b2dnvs, %s1b2dnnf : tensor<32x150528xf32>
    %s1b2dnve = stablehlo.add %s1b2dnvr, %s1b2dnep : tensor<32x150528xf32>
    %s1b2dnistd = stablehlo.rsqrt %s1b2dnve : tensor<32x150528xf32>
    %s1b2dnxh = stablehlo.multiply %s1b2dnxc, %s1b2dnistd : tensor<32x150528xf32>
    %s1b2dngb = stablehlo.broadcast_in_dim %s1b2ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b2dndxh = stablehlo.multiply %s1b2dngb, %s1b2dnrdy : tensor<32x150528xf32>
    %s1b2dnsdxr = stablehlo.reduce(%s1b2dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b2dnsdx = stablehlo.broadcast_in_dim %s1b2dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b2dnxd = stablehlo.multiply %s1b2dnxh, %s1b2dndxh : tensor<32x150528xf32>
    %s1b2dnsxdr = stablehlo.reduce(%s1b2dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b2dnsxd = stablehlo.broadcast_in_dim %s1b2dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b2dnt1 = stablehlo.multiply %s1b2dndxh, %s1b2dnnf : tensor<32x150528xf32>
    %s1b2dni1 = stablehlo.subtract %s1b2dnt1, %s1b2dnsdx : tensor<32x150528xf32>
    %s1b2dnxs = stablehlo.multiply %s1b2dnxh, %s1b2dnsxd : tensor<32x150528xf32>
    %s1b2dni2 = stablehlo.subtract %s1b2dni1, %s1b2dnxs : tensor<32x150528xf32>
    %s1b2dnsN = stablehlo.divide %s1b2dnistd, %s1b2dnnf : tensor<32x150528xf32>
    %s1b2dngin = stablehlo.multiply %s1b2dnsN, %s1b2dni2 : tensor<32x150528xf32>
    %s1b2dn = stablehlo.reshape %s1b2dngin : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %s1b2dndgp = stablehlo.multiply %s1b2dnrdy, %s1b2dnxh : tensor<32x150528xf32>
    %s1b2dndg = stablehlo.reduce(%s1b2dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b2dndb = stablehlo.reduce(%s1b2dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b2ddrev = stablehlo.reverse %s1b2dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %s1b2dd = stablehlo.convolution(%s1b2dn, %s1b2ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %s1b2ddWxt = stablehlo.transpose %s1b1o, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b2ddWdt = stablehlo.transpose %s1b2dn, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b2ddWraw = stablehlo.convolution(%s1b2ddWxt, %s1b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %s1b2ddW = stablehlo.reshape %s1b2ddWraw : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %s1b2ddb = stablehlo.reduce(%s1b2dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b2dx = stablehlo.add %s1b2dd, %d1dn : tensor<32x192x28x28xf32>
    %s1b1dlsgb = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b1dls = stablehlo.multiply %s1b1dlsgb, %s1b2dx : tensor<32x192x28x28xf32>
    %s1b1dlsxdy = stablehlo.multiply %s1b1p, %s1b2dx : tensor<32x192x28x28xf32>
    %s1b1dlsdg = stablehlo.reduce(%s1b1dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b1dpt = stablehlo.transpose %s1b1pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %s1b1dp = stablehlo.convolution(%s1b1dls, %s1b1dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %s1b1dpWxt = stablehlo.transpose %s1b1g, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %s1b1dpWdt = stablehlo.transpose %s1b1dls, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b1dpWraw = stablehlo.convolution(%s1b1dpWxt, %s1b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %s1b1dpW = stablehlo.transpose %s1b1dpWraw, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %s1b1dpb = stablehlo.reduce(%s1b1dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b1dgx2 = stablehlo.multiply %s1b1e, %s1b1e : tensor<32x768x28x28xf32>
    %s1b1dgx3 = stablehlo.multiply %s1b1dgx2, %s1b1e : tensor<32x768x28x28xf32>
    %s1b1dgck = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %s1b1dgkx3 = stablehlo.multiply %s1b1dgck, %s1b1dgx3 : tensor<32x768x28x28xf32>
    %s1b1dginn = stablehlo.add %s1b1e, %s1b1dgkx3 : tensor<32x768x28x28xf32>
    %s1b1dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %s1b1dgu = stablehlo.multiply %s1b1dgcs, %s1b1dginn : tensor<32x768x28x28xf32>
    %s1b1dgt = stablehlo.tanh %s1b1dgu : tensor<32x768x28x28xf32>
    %s1b1dgone = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %s1b1dgopt = stablehlo.add %s1b1dgone, %s1b1dgt : tensor<32x768x28x28xf32>
    %s1b1dghalf = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %s1b1dgterm1 = stablehlo.multiply %s1b1dghalf, %s1b1dgopt : tensor<32x768x28x28xf32>
    %s1b1dgt2 = stablehlo.multiply %s1b1dgt, %s1b1dgt : tensor<32x768x28x28xf32>
    %s1b1dgomt2 = stablehlo.subtract %s1b1dgone, %s1b1dgt2 : tensor<32x768x28x28xf32>
    %s1b1dghx = stablehlo.multiply %s1b1dghalf, %s1b1e : tensor<32x768x28x28xf32>
    %s1b1dghxo = stablehlo.multiply %s1b1dghx, %s1b1dgomt2 : tensor<32x768x28x28xf32>
    %s1b1dgc3b = stablehlo.constant dense<0.134145> : tensor<32x768x28x28xf32>
    %s1b1dga3x2 = stablehlo.multiply %s1b1dgc3b, %s1b1dgx2 : tensor<32x768x28x28xf32>
    %s1b1dgin2 = stablehlo.add %s1b1dgone, %s1b1dga3x2 : tensor<32x768x28x28xf32>
    %s1b1dgup = stablehlo.multiply %s1b1dgcs, %s1b1dgin2 : tensor<32x768x28x28xf32>
    %s1b1dgterm2 = stablehlo.multiply %s1b1dghxo, %s1b1dgup : tensor<32x768x28x28xf32>
    %s1b1dggp = stablehlo.add %s1b1dgterm1, %s1b1dgterm2 : tensor<32x768x28x28xf32>
    %s1b1dg = stablehlo.multiply %s1b1dp, %s1b1dggp : tensor<32x768x28x28xf32>
    %s1b1det = stablehlo.transpose %s1b1eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %s1b1de = stablehlo.convolution(%s1b1dg, %s1b1det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %s1b1deWxt = stablehlo.transpose %s1b1n, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b1deWdt = stablehlo.transpose %s1b1dg, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %s1b1deWraw = stablehlo.convolution(%s1b1deWxt, %s1b1deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %s1b1deW = stablehlo.transpose %s1b1deWraw, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %s1b1deb = stablehlo.reduce(%s1b1dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %s1b1dnri = stablehlo.reshape %s1b1d : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b1dnrdy = stablehlo.reshape %s1b1de : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b1dnnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %s1b1dnep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %s1b1dnsmr = stablehlo.reduce(%s1b1dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b1dnsm = stablehlo.broadcast_in_dim %s1b1dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b1dnmu = stablehlo.divide %s1b1dnsm, %s1b1dnnf : tensor<32x150528xf32>
    %s1b1dnxc = stablehlo.subtract %s1b1dnri, %s1b1dnmu : tensor<32x150528xf32>
    %s1b1dnsq = stablehlo.multiply %s1b1dnxc, %s1b1dnxc : tensor<32x150528xf32>
    %s1b1dnvsr = stablehlo.reduce(%s1b1dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b1dnvs = stablehlo.broadcast_in_dim %s1b1dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b1dnvr = stablehlo.divide %s1b1dnvs, %s1b1dnnf : tensor<32x150528xf32>
    %s1b1dnve = stablehlo.add %s1b1dnvr, %s1b1dnep : tensor<32x150528xf32>
    %s1b1dnistd = stablehlo.rsqrt %s1b1dnve : tensor<32x150528xf32>
    %s1b1dnxh = stablehlo.multiply %s1b1dnxc, %s1b1dnistd : tensor<32x150528xf32>
    %s1b1dngb = stablehlo.broadcast_in_dim %s1b1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b1dndxh = stablehlo.multiply %s1b1dngb, %s1b1dnrdy : tensor<32x150528xf32>
    %s1b1dnsdxr = stablehlo.reduce(%s1b1dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b1dnsdx = stablehlo.broadcast_in_dim %s1b1dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b1dnxd = stablehlo.multiply %s1b1dnxh, %s1b1dndxh : tensor<32x150528xf32>
    %s1b1dnsxdr = stablehlo.reduce(%s1b1dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b1dnsxd = stablehlo.broadcast_in_dim %s1b1dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b1dnt1 = stablehlo.multiply %s1b1dndxh, %s1b1dnnf : tensor<32x150528xf32>
    %s1b1dni1 = stablehlo.subtract %s1b1dnt1, %s1b1dnsdx : tensor<32x150528xf32>
    %s1b1dnxs = stablehlo.multiply %s1b1dnxh, %s1b1dnsxd : tensor<32x150528xf32>
    %s1b1dni2 = stablehlo.subtract %s1b1dni1, %s1b1dnxs : tensor<32x150528xf32>
    %s1b1dnsN = stablehlo.divide %s1b1dnistd, %s1b1dnnf : tensor<32x150528xf32>
    %s1b1dngin = stablehlo.multiply %s1b1dnsN, %s1b1dni2 : tensor<32x150528xf32>
    %s1b1dn = stablehlo.reshape %s1b1dngin : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %s1b1dndgp = stablehlo.multiply %s1b1dnrdy, %s1b1dnxh : tensor<32x150528xf32>
    %s1b1dndg = stablehlo.reduce(%s1b1dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b1dndb = stablehlo.reduce(%s1b1dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b1ddrev = stablehlo.reverse %s1b1dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %s1b1dd = stablehlo.convolution(%s1b1dn, %s1b1ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %s1b1ddWxt = stablehlo.transpose %s1b0o, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b1ddWdt = stablehlo.transpose %s1b1dn, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b1ddWraw = stablehlo.convolution(%s1b1ddWxt, %s1b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %s1b1ddW = stablehlo.reshape %s1b1ddWraw : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %s1b1ddb = stablehlo.reduce(%s1b1dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b1dx = stablehlo.add %s1b1dd, %s1b2dx : tensor<32x192x28x28xf32>
    %s1b0dlsgb = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b0dls = stablehlo.multiply %s1b0dlsgb, %s1b1dx : tensor<32x192x28x28xf32>
    %s1b0dlsxdy = stablehlo.multiply %s1b0p, %s1b1dx : tensor<32x192x28x28xf32>
    %s1b0dlsdg = stablehlo.reduce(%s1b0dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b0dpt = stablehlo.transpose %s1b0pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %s1b0dp = stablehlo.convolution(%s1b0dls, %s1b0dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %s1b0dpWxt = stablehlo.transpose %s1b0g, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %s1b0dpWdt = stablehlo.transpose %s1b0dls, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b0dpWraw = stablehlo.convolution(%s1b0dpWxt, %s1b0dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %s1b0dpW = stablehlo.transpose %s1b0dpWraw, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %s1b0dpb = stablehlo.reduce(%s1b0dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b0dgx2 = stablehlo.multiply %s1b0e, %s1b0e : tensor<32x768x28x28xf32>
    %s1b0dgx3 = stablehlo.multiply %s1b0dgx2, %s1b0e : tensor<32x768x28x28xf32>
    %s1b0dgck = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %s1b0dgkx3 = stablehlo.multiply %s1b0dgck, %s1b0dgx3 : tensor<32x768x28x28xf32>
    %s1b0dginn = stablehlo.add %s1b0e, %s1b0dgkx3 : tensor<32x768x28x28xf32>
    %s1b0dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %s1b0dgu = stablehlo.multiply %s1b0dgcs, %s1b0dginn : tensor<32x768x28x28xf32>
    %s1b0dgt = stablehlo.tanh %s1b0dgu : tensor<32x768x28x28xf32>
    %s1b0dgone = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %s1b0dgopt = stablehlo.add %s1b0dgone, %s1b0dgt : tensor<32x768x28x28xf32>
    %s1b0dghalf = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %s1b0dgterm1 = stablehlo.multiply %s1b0dghalf, %s1b0dgopt : tensor<32x768x28x28xf32>
    %s1b0dgt2 = stablehlo.multiply %s1b0dgt, %s1b0dgt : tensor<32x768x28x28xf32>
    %s1b0dgomt2 = stablehlo.subtract %s1b0dgone, %s1b0dgt2 : tensor<32x768x28x28xf32>
    %s1b0dghx = stablehlo.multiply %s1b0dghalf, %s1b0e : tensor<32x768x28x28xf32>
    %s1b0dghxo = stablehlo.multiply %s1b0dghx, %s1b0dgomt2 : tensor<32x768x28x28xf32>
    %s1b0dgc3b = stablehlo.constant dense<0.134145> : tensor<32x768x28x28xf32>
    %s1b0dga3x2 = stablehlo.multiply %s1b0dgc3b, %s1b0dgx2 : tensor<32x768x28x28xf32>
    %s1b0dgin2 = stablehlo.add %s1b0dgone, %s1b0dga3x2 : tensor<32x768x28x28xf32>
    %s1b0dgup = stablehlo.multiply %s1b0dgcs, %s1b0dgin2 : tensor<32x768x28x28xf32>
    %s1b0dgterm2 = stablehlo.multiply %s1b0dghxo, %s1b0dgup : tensor<32x768x28x28xf32>
    %s1b0dggp = stablehlo.add %s1b0dgterm1, %s1b0dgterm2 : tensor<32x768x28x28xf32>
    %s1b0dg = stablehlo.multiply %s1b0dp, %s1b0dggp : tensor<32x768x28x28xf32>
    %s1b0det = stablehlo.transpose %s1b0eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %s1b0de = stablehlo.convolution(%s1b0dg, %s1b0det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %s1b0deWxt = stablehlo.transpose %s1b0n, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b0deWdt = stablehlo.transpose %s1b0dg, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %s1b0deWraw = stablehlo.convolution(%s1b0deWxt, %s1b0deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %s1b0deW = stablehlo.transpose %s1b0deWraw, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %s1b0deb = stablehlo.reduce(%s1b0dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %s1b0dnri = stablehlo.reshape %s1b0d : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b0dnrdy = stablehlo.reshape %s1b0de : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b0dnnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %s1b0dnep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %s1b0dnsmr = stablehlo.reduce(%s1b0dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b0dnsm = stablehlo.broadcast_in_dim %s1b0dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b0dnmu = stablehlo.divide %s1b0dnsm, %s1b0dnnf : tensor<32x150528xf32>
    %s1b0dnxc = stablehlo.subtract %s1b0dnri, %s1b0dnmu : tensor<32x150528xf32>
    %s1b0dnsq = stablehlo.multiply %s1b0dnxc, %s1b0dnxc : tensor<32x150528xf32>
    %s1b0dnvsr = stablehlo.reduce(%s1b0dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b0dnvs = stablehlo.broadcast_in_dim %s1b0dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b0dnvr = stablehlo.divide %s1b0dnvs, %s1b0dnnf : tensor<32x150528xf32>
    %s1b0dnve = stablehlo.add %s1b0dnvr, %s1b0dnep : tensor<32x150528xf32>
    %s1b0dnistd = stablehlo.rsqrt %s1b0dnve : tensor<32x150528xf32>
    %s1b0dnxh = stablehlo.multiply %s1b0dnxc, %s1b0dnistd : tensor<32x150528xf32>
    %s1b0dngb = stablehlo.broadcast_in_dim %s1b0ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b0dndxh = stablehlo.multiply %s1b0dngb, %s1b0dnrdy : tensor<32x150528xf32>
    %s1b0dnsdxr = stablehlo.reduce(%s1b0dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b0dnsdx = stablehlo.broadcast_in_dim %s1b0dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b0dnxd = stablehlo.multiply %s1b0dnxh, %s1b0dndxh : tensor<32x150528xf32>
    %s1b0dnsxdr = stablehlo.reduce(%s1b0dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b0dnsxd = stablehlo.broadcast_in_dim %s1b0dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b0dnt1 = stablehlo.multiply %s1b0dndxh, %s1b0dnnf : tensor<32x150528xf32>
    %s1b0dni1 = stablehlo.subtract %s1b0dnt1, %s1b0dnsdx : tensor<32x150528xf32>
    %s1b0dnxs = stablehlo.multiply %s1b0dnxh, %s1b0dnsxd : tensor<32x150528xf32>
    %s1b0dni2 = stablehlo.subtract %s1b0dni1, %s1b0dnxs : tensor<32x150528xf32>
    %s1b0dnsN = stablehlo.divide %s1b0dnistd, %s1b0dnnf : tensor<32x150528xf32>
    %s1b0dngin = stablehlo.multiply %s1b0dnsN, %s1b0dni2 : tensor<32x150528xf32>
    %s1b0dn = stablehlo.reshape %s1b0dngin : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %s1b0dndgp = stablehlo.multiply %s1b0dnrdy, %s1b0dnxh : tensor<32x150528xf32>
    %s1b0dndg = stablehlo.reduce(%s1b0dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b0dndb = stablehlo.reduce(%s1b0dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b0ddrev = stablehlo.reverse %s1b0dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %s1b0dd = stablehlo.convolution(%s1b0dn, %s1b0ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %s1b0ddWxt = stablehlo.transpose %d0c, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b0ddWdt = stablehlo.transpose %s1b0dn, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b0ddWraw = stablehlo.convolution(%s1b0ddWxt, %s1b0ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %s1b0ddW = stablehlo.reshape %s1b0ddWraw : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %s1b0ddb = stablehlo.reduce(%s1b0dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b0dx = stablehlo.add %s1b0dd, %s1b1dx : tensor<32x192x28x28xf32>
    %d0dcu = stablehlo.pad %s1b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x56x56xf32>
    %d0dct = stablehlo.transpose %d0W, dims = [1, 0, 2, 3] : (tensor<192x96x2x2xf32>) -> tensor<96x192x2x2xf32>
    %d0dcr = stablehlo.reverse %d0dct, dims = [2, 3] : tensor<96x192x2x2xf32>
    %d0dc = stablehlo.convolution(%d0dcu, %d0dcr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x56x56xf32>, tensor<96x192x2x2xf32>) -> tensor<32x96x56x56xf32>
    %d0dWu = stablehlo.pad %s1b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x55x55xf32>
    %d0dWxt = stablehlo.transpose %d0n, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %d0dWdt = stablehlo.transpose %d0dWu, dims = [1, 0, 2, 3] : (tensor<32x192x55x55xf32>) -> tensor<192x32x55x55xf32>
    %d0dWraw = stablehlo.convolution(%d0dWxt, %d0dWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<192x32x55x55xf32>) -> tensor<96x192x2x2xf32>
    %d0dW = stablehlo.transpose %d0dWraw, dims = [1, 0, 2, 3] : (tensor<96x192x2x2xf32>) -> tensor<192x96x2x2xf32>
    %d0db = stablehlo.reduce(%s1b0dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %d0dnri = stablehlo.reshape %s0b2o : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %d0dnrdy = stablehlo.reshape %d0dc : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %d0dnnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %d0dnep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %d0dnsmr = stablehlo.reduce(%d0dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %d0dnsm = stablehlo.broadcast_in_dim %d0dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %d0dnmu = stablehlo.divide %d0dnsm, %d0dnnf : tensor<32x301056xf32>
    %d0dnxc = stablehlo.subtract %d0dnri, %d0dnmu : tensor<32x301056xf32>
    %d0dnsq = stablehlo.multiply %d0dnxc, %d0dnxc : tensor<32x301056xf32>
    %d0dnvsr = stablehlo.reduce(%d0dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %d0dnvs = stablehlo.broadcast_in_dim %d0dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %d0dnvr = stablehlo.divide %d0dnvs, %d0dnnf : tensor<32x301056xf32>
    %d0dnve = stablehlo.add %d0dnvr, %d0dnep : tensor<32x301056xf32>
    %d0dnistd = stablehlo.rsqrt %d0dnve : tensor<32x301056xf32>
    %d0dnxh = stablehlo.multiply %d0dnxc, %d0dnistd : tensor<32x301056xf32>
    %d0dngb = stablehlo.broadcast_in_dim %d0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %d0dndxh = stablehlo.multiply %d0dngb, %d0dnrdy : tensor<32x301056xf32>
    %d0dnsdxr = stablehlo.reduce(%d0dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %d0dnsdx = stablehlo.broadcast_in_dim %d0dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %d0dnxd = stablehlo.multiply %d0dnxh, %d0dndxh : tensor<32x301056xf32>
    %d0dnsxdr = stablehlo.reduce(%d0dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %d0dnsxd = stablehlo.broadcast_in_dim %d0dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %d0dnt1 = stablehlo.multiply %d0dndxh, %d0dnnf : tensor<32x301056xf32>
    %d0dni1 = stablehlo.subtract %d0dnt1, %d0dnsdx : tensor<32x301056xf32>
    %d0dnxs = stablehlo.multiply %d0dnxh, %d0dnsxd : tensor<32x301056xf32>
    %d0dni2 = stablehlo.subtract %d0dni1, %d0dnxs : tensor<32x301056xf32>
    %d0dnsN = stablehlo.divide %d0dnistd, %d0dnnf : tensor<32x301056xf32>
    %d0dngin = stablehlo.multiply %d0dnsN, %d0dni2 : tensor<32x301056xf32>
    %d0dn = stablehlo.reshape %d0dngin : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %d0dndgp = stablehlo.multiply %d0dnrdy, %d0dnxh : tensor<32x301056xf32>
    %d0dndg = stablehlo.reduce(%d0dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %d0dndb = stablehlo.reduce(%d0dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b2dlsgb = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b2dls = stablehlo.multiply %s0b2dlsgb, %d0dn : tensor<32x96x56x56xf32>
    %s0b2dlsxdy = stablehlo.multiply %s0b2p, %d0dn : tensor<32x96x56x56xf32>
    %s0b2dlsdg = stablehlo.reduce(%s0b2dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b2dpt = stablehlo.transpose %s0b2pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %s0b2dp = stablehlo.convolution(%s0b2dls, %s0b2dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %s0b2dpWxt = stablehlo.transpose %s0b2g, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %s0b2dpWdt = stablehlo.transpose %s0b2dls, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b2dpWraw = stablehlo.convolution(%s0b2dpWxt, %s0b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %s0b2dpW = stablehlo.transpose %s0b2dpWraw, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %s0b2dpb = stablehlo.reduce(%s0b2dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b2dgx2 = stablehlo.multiply %s0b2e, %s0b2e : tensor<32x384x56x56xf32>
    %s0b2dgx3 = stablehlo.multiply %s0b2dgx2, %s0b2e : tensor<32x384x56x56xf32>
    %s0b2dgck = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %s0b2dgkx3 = stablehlo.multiply %s0b2dgck, %s0b2dgx3 : tensor<32x384x56x56xf32>
    %s0b2dginn = stablehlo.add %s0b2e, %s0b2dgkx3 : tensor<32x384x56x56xf32>
    %s0b2dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %s0b2dgu = stablehlo.multiply %s0b2dgcs, %s0b2dginn : tensor<32x384x56x56xf32>
    %s0b2dgt = stablehlo.tanh %s0b2dgu : tensor<32x384x56x56xf32>
    %s0b2dgone = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %s0b2dgopt = stablehlo.add %s0b2dgone, %s0b2dgt : tensor<32x384x56x56xf32>
    %s0b2dghalf = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %s0b2dgterm1 = stablehlo.multiply %s0b2dghalf, %s0b2dgopt : tensor<32x384x56x56xf32>
    %s0b2dgt2 = stablehlo.multiply %s0b2dgt, %s0b2dgt : tensor<32x384x56x56xf32>
    %s0b2dgomt2 = stablehlo.subtract %s0b2dgone, %s0b2dgt2 : tensor<32x384x56x56xf32>
    %s0b2dghx = stablehlo.multiply %s0b2dghalf, %s0b2e : tensor<32x384x56x56xf32>
    %s0b2dghxo = stablehlo.multiply %s0b2dghx, %s0b2dgomt2 : tensor<32x384x56x56xf32>
    %s0b2dgc3b = stablehlo.constant dense<0.134145> : tensor<32x384x56x56xf32>
    %s0b2dga3x2 = stablehlo.multiply %s0b2dgc3b, %s0b2dgx2 : tensor<32x384x56x56xf32>
    %s0b2dgin2 = stablehlo.add %s0b2dgone, %s0b2dga3x2 : tensor<32x384x56x56xf32>
    %s0b2dgup = stablehlo.multiply %s0b2dgcs, %s0b2dgin2 : tensor<32x384x56x56xf32>
    %s0b2dgterm2 = stablehlo.multiply %s0b2dghxo, %s0b2dgup : tensor<32x384x56x56xf32>
    %s0b2dggp = stablehlo.add %s0b2dgterm1, %s0b2dgterm2 : tensor<32x384x56x56xf32>
    %s0b2dg = stablehlo.multiply %s0b2dp, %s0b2dggp : tensor<32x384x56x56xf32>
    %s0b2det = stablehlo.transpose %s0b2eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %s0b2de = stablehlo.convolution(%s0b2dg, %s0b2det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %s0b2deWxt = stablehlo.transpose %s0b2n, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b2deWdt = stablehlo.transpose %s0b2dg, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %s0b2deWraw = stablehlo.convolution(%s0b2deWxt, %s0b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %s0b2deW = stablehlo.transpose %s0b2deWraw, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %s0b2deb = stablehlo.reduce(%s0b2dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %s0b2dnri = stablehlo.reshape %s0b2d : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b2dnrdy = stablehlo.reshape %s0b2de : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b2dnnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %s0b2dnep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %s0b2dnsmr = stablehlo.reduce(%s0b2dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b2dnsm = stablehlo.broadcast_in_dim %s0b2dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b2dnmu = stablehlo.divide %s0b2dnsm, %s0b2dnnf : tensor<32x301056xf32>
    %s0b2dnxc = stablehlo.subtract %s0b2dnri, %s0b2dnmu : tensor<32x301056xf32>
    %s0b2dnsq = stablehlo.multiply %s0b2dnxc, %s0b2dnxc : tensor<32x301056xf32>
    %s0b2dnvsr = stablehlo.reduce(%s0b2dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b2dnvs = stablehlo.broadcast_in_dim %s0b2dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b2dnvr = stablehlo.divide %s0b2dnvs, %s0b2dnnf : tensor<32x301056xf32>
    %s0b2dnve = stablehlo.add %s0b2dnvr, %s0b2dnep : tensor<32x301056xf32>
    %s0b2dnistd = stablehlo.rsqrt %s0b2dnve : tensor<32x301056xf32>
    %s0b2dnxh = stablehlo.multiply %s0b2dnxc, %s0b2dnistd : tensor<32x301056xf32>
    %s0b2dngb = stablehlo.broadcast_in_dim %s0b2ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b2dndxh = stablehlo.multiply %s0b2dngb, %s0b2dnrdy : tensor<32x301056xf32>
    %s0b2dnsdxr = stablehlo.reduce(%s0b2dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b2dnsdx = stablehlo.broadcast_in_dim %s0b2dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b2dnxd = stablehlo.multiply %s0b2dnxh, %s0b2dndxh : tensor<32x301056xf32>
    %s0b2dnsxdr = stablehlo.reduce(%s0b2dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b2dnsxd = stablehlo.broadcast_in_dim %s0b2dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b2dnt1 = stablehlo.multiply %s0b2dndxh, %s0b2dnnf : tensor<32x301056xf32>
    %s0b2dni1 = stablehlo.subtract %s0b2dnt1, %s0b2dnsdx : tensor<32x301056xf32>
    %s0b2dnxs = stablehlo.multiply %s0b2dnxh, %s0b2dnsxd : tensor<32x301056xf32>
    %s0b2dni2 = stablehlo.subtract %s0b2dni1, %s0b2dnxs : tensor<32x301056xf32>
    %s0b2dnsN = stablehlo.divide %s0b2dnistd, %s0b2dnnf : tensor<32x301056xf32>
    %s0b2dngin = stablehlo.multiply %s0b2dnsN, %s0b2dni2 : tensor<32x301056xf32>
    %s0b2dn = stablehlo.reshape %s0b2dngin : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %s0b2dndgp = stablehlo.multiply %s0b2dnrdy, %s0b2dnxh : tensor<32x301056xf32>
    %s0b2dndg = stablehlo.reduce(%s0b2dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b2dndb = stablehlo.reduce(%s0b2dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b2ddrev = stablehlo.reverse %s0b2dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %s0b2dd = stablehlo.convolution(%s0b2dn, %s0b2ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %s0b2ddWxt = stablehlo.transpose %s0b1o, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b2ddWdt = stablehlo.transpose %s0b2dn, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b2ddWraw = stablehlo.convolution(%s0b2ddWxt, %s0b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %s0b2ddW = stablehlo.reshape %s0b2ddWraw : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %s0b2ddb = stablehlo.reduce(%s0b2dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b2dx = stablehlo.add %s0b2dd, %d0dn : tensor<32x96x56x56xf32>
    %s0b1dlsgb = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b1dls = stablehlo.multiply %s0b1dlsgb, %s0b2dx : tensor<32x96x56x56xf32>
    %s0b1dlsxdy = stablehlo.multiply %s0b1p, %s0b2dx : tensor<32x96x56x56xf32>
    %s0b1dlsdg = stablehlo.reduce(%s0b1dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b1dpt = stablehlo.transpose %s0b1pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %s0b1dp = stablehlo.convolution(%s0b1dls, %s0b1dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %s0b1dpWxt = stablehlo.transpose %s0b1g, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %s0b1dpWdt = stablehlo.transpose %s0b1dls, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b1dpWraw = stablehlo.convolution(%s0b1dpWxt, %s0b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %s0b1dpW = stablehlo.transpose %s0b1dpWraw, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %s0b1dpb = stablehlo.reduce(%s0b1dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b1dgx2 = stablehlo.multiply %s0b1e, %s0b1e : tensor<32x384x56x56xf32>
    %s0b1dgx3 = stablehlo.multiply %s0b1dgx2, %s0b1e : tensor<32x384x56x56xf32>
    %s0b1dgck = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %s0b1dgkx3 = stablehlo.multiply %s0b1dgck, %s0b1dgx3 : tensor<32x384x56x56xf32>
    %s0b1dginn = stablehlo.add %s0b1e, %s0b1dgkx3 : tensor<32x384x56x56xf32>
    %s0b1dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %s0b1dgu = stablehlo.multiply %s0b1dgcs, %s0b1dginn : tensor<32x384x56x56xf32>
    %s0b1dgt = stablehlo.tanh %s0b1dgu : tensor<32x384x56x56xf32>
    %s0b1dgone = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %s0b1dgopt = stablehlo.add %s0b1dgone, %s0b1dgt : tensor<32x384x56x56xf32>
    %s0b1dghalf = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %s0b1dgterm1 = stablehlo.multiply %s0b1dghalf, %s0b1dgopt : tensor<32x384x56x56xf32>
    %s0b1dgt2 = stablehlo.multiply %s0b1dgt, %s0b1dgt : tensor<32x384x56x56xf32>
    %s0b1dgomt2 = stablehlo.subtract %s0b1dgone, %s0b1dgt2 : tensor<32x384x56x56xf32>
    %s0b1dghx = stablehlo.multiply %s0b1dghalf, %s0b1e : tensor<32x384x56x56xf32>
    %s0b1dghxo = stablehlo.multiply %s0b1dghx, %s0b1dgomt2 : tensor<32x384x56x56xf32>
    %s0b1dgc3b = stablehlo.constant dense<0.134145> : tensor<32x384x56x56xf32>
    %s0b1dga3x2 = stablehlo.multiply %s0b1dgc3b, %s0b1dgx2 : tensor<32x384x56x56xf32>
    %s0b1dgin2 = stablehlo.add %s0b1dgone, %s0b1dga3x2 : tensor<32x384x56x56xf32>
    %s0b1dgup = stablehlo.multiply %s0b1dgcs, %s0b1dgin2 : tensor<32x384x56x56xf32>
    %s0b1dgterm2 = stablehlo.multiply %s0b1dghxo, %s0b1dgup : tensor<32x384x56x56xf32>
    %s0b1dggp = stablehlo.add %s0b1dgterm1, %s0b1dgterm2 : tensor<32x384x56x56xf32>
    %s0b1dg = stablehlo.multiply %s0b1dp, %s0b1dggp : tensor<32x384x56x56xf32>
    %s0b1det = stablehlo.transpose %s0b1eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %s0b1de = stablehlo.convolution(%s0b1dg, %s0b1det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %s0b1deWxt = stablehlo.transpose %s0b1n, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b1deWdt = stablehlo.transpose %s0b1dg, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %s0b1deWraw = stablehlo.convolution(%s0b1deWxt, %s0b1deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %s0b1deW = stablehlo.transpose %s0b1deWraw, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %s0b1deb = stablehlo.reduce(%s0b1dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %s0b1dnri = stablehlo.reshape %s0b1d : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b1dnrdy = stablehlo.reshape %s0b1de : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b1dnnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %s0b1dnep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %s0b1dnsmr = stablehlo.reduce(%s0b1dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b1dnsm = stablehlo.broadcast_in_dim %s0b1dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b1dnmu = stablehlo.divide %s0b1dnsm, %s0b1dnnf : tensor<32x301056xf32>
    %s0b1dnxc = stablehlo.subtract %s0b1dnri, %s0b1dnmu : tensor<32x301056xf32>
    %s0b1dnsq = stablehlo.multiply %s0b1dnxc, %s0b1dnxc : tensor<32x301056xf32>
    %s0b1dnvsr = stablehlo.reduce(%s0b1dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b1dnvs = stablehlo.broadcast_in_dim %s0b1dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b1dnvr = stablehlo.divide %s0b1dnvs, %s0b1dnnf : tensor<32x301056xf32>
    %s0b1dnve = stablehlo.add %s0b1dnvr, %s0b1dnep : tensor<32x301056xf32>
    %s0b1dnistd = stablehlo.rsqrt %s0b1dnve : tensor<32x301056xf32>
    %s0b1dnxh = stablehlo.multiply %s0b1dnxc, %s0b1dnistd : tensor<32x301056xf32>
    %s0b1dngb = stablehlo.broadcast_in_dim %s0b1ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b1dndxh = stablehlo.multiply %s0b1dngb, %s0b1dnrdy : tensor<32x301056xf32>
    %s0b1dnsdxr = stablehlo.reduce(%s0b1dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b1dnsdx = stablehlo.broadcast_in_dim %s0b1dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b1dnxd = stablehlo.multiply %s0b1dnxh, %s0b1dndxh : tensor<32x301056xf32>
    %s0b1dnsxdr = stablehlo.reduce(%s0b1dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b1dnsxd = stablehlo.broadcast_in_dim %s0b1dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b1dnt1 = stablehlo.multiply %s0b1dndxh, %s0b1dnnf : tensor<32x301056xf32>
    %s0b1dni1 = stablehlo.subtract %s0b1dnt1, %s0b1dnsdx : tensor<32x301056xf32>
    %s0b1dnxs = stablehlo.multiply %s0b1dnxh, %s0b1dnsxd : tensor<32x301056xf32>
    %s0b1dni2 = stablehlo.subtract %s0b1dni1, %s0b1dnxs : tensor<32x301056xf32>
    %s0b1dnsN = stablehlo.divide %s0b1dnistd, %s0b1dnnf : tensor<32x301056xf32>
    %s0b1dngin = stablehlo.multiply %s0b1dnsN, %s0b1dni2 : tensor<32x301056xf32>
    %s0b1dn = stablehlo.reshape %s0b1dngin : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %s0b1dndgp = stablehlo.multiply %s0b1dnrdy, %s0b1dnxh : tensor<32x301056xf32>
    %s0b1dndg = stablehlo.reduce(%s0b1dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b1dndb = stablehlo.reduce(%s0b1dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b1ddrev = stablehlo.reverse %s0b1dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %s0b1dd = stablehlo.convolution(%s0b1dn, %s0b1ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %s0b1ddWxt = stablehlo.transpose %s0b0o, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b1ddWdt = stablehlo.transpose %s0b1dn, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b1ddWraw = stablehlo.convolution(%s0b1ddWxt, %s0b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %s0b1ddW = stablehlo.reshape %s0b1ddWraw : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %s0b1ddb = stablehlo.reduce(%s0b1dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b1dx = stablehlo.add %s0b1dd, %s0b2dx : tensor<32x96x56x56xf32>
    %s0b0dlsgb = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b0dls = stablehlo.multiply %s0b0dlsgb, %s0b1dx : tensor<32x96x56x56xf32>
    %s0b0dlsxdy = stablehlo.multiply %s0b0p, %s0b1dx : tensor<32x96x56x56xf32>
    %s0b0dlsdg = stablehlo.reduce(%s0b0dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b0dpt = stablehlo.transpose %s0b0pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %s0b0dp = stablehlo.convolution(%s0b0dls, %s0b0dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %s0b0dpWxt = stablehlo.transpose %s0b0g, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %s0b0dpWdt = stablehlo.transpose %s0b0dls, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b0dpWraw = stablehlo.convolution(%s0b0dpWxt, %s0b0dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %s0b0dpW = stablehlo.transpose %s0b0dpWraw, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %s0b0dpb = stablehlo.reduce(%s0b0dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b0dgx2 = stablehlo.multiply %s0b0e, %s0b0e : tensor<32x384x56x56xf32>
    %s0b0dgx3 = stablehlo.multiply %s0b0dgx2, %s0b0e : tensor<32x384x56x56xf32>
    %s0b0dgck = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %s0b0dgkx3 = stablehlo.multiply %s0b0dgck, %s0b0dgx3 : tensor<32x384x56x56xf32>
    %s0b0dginn = stablehlo.add %s0b0e, %s0b0dgkx3 : tensor<32x384x56x56xf32>
    %s0b0dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %s0b0dgu = stablehlo.multiply %s0b0dgcs, %s0b0dginn : tensor<32x384x56x56xf32>
    %s0b0dgt = stablehlo.tanh %s0b0dgu : tensor<32x384x56x56xf32>
    %s0b0dgone = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %s0b0dgopt = stablehlo.add %s0b0dgone, %s0b0dgt : tensor<32x384x56x56xf32>
    %s0b0dghalf = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %s0b0dgterm1 = stablehlo.multiply %s0b0dghalf, %s0b0dgopt : tensor<32x384x56x56xf32>
    %s0b0dgt2 = stablehlo.multiply %s0b0dgt, %s0b0dgt : tensor<32x384x56x56xf32>
    %s0b0dgomt2 = stablehlo.subtract %s0b0dgone, %s0b0dgt2 : tensor<32x384x56x56xf32>
    %s0b0dghx = stablehlo.multiply %s0b0dghalf, %s0b0e : tensor<32x384x56x56xf32>
    %s0b0dghxo = stablehlo.multiply %s0b0dghx, %s0b0dgomt2 : tensor<32x384x56x56xf32>
    %s0b0dgc3b = stablehlo.constant dense<0.134145> : tensor<32x384x56x56xf32>
    %s0b0dga3x2 = stablehlo.multiply %s0b0dgc3b, %s0b0dgx2 : tensor<32x384x56x56xf32>
    %s0b0dgin2 = stablehlo.add %s0b0dgone, %s0b0dga3x2 : tensor<32x384x56x56xf32>
    %s0b0dgup = stablehlo.multiply %s0b0dgcs, %s0b0dgin2 : tensor<32x384x56x56xf32>
    %s0b0dgterm2 = stablehlo.multiply %s0b0dghxo, %s0b0dgup : tensor<32x384x56x56xf32>
    %s0b0dggp = stablehlo.add %s0b0dgterm1, %s0b0dgterm2 : tensor<32x384x56x56xf32>
    %s0b0dg = stablehlo.multiply %s0b0dp, %s0b0dggp : tensor<32x384x56x56xf32>
    %s0b0det = stablehlo.transpose %s0b0eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %s0b0de = stablehlo.convolution(%s0b0dg, %s0b0det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %s0b0deWxt = stablehlo.transpose %s0b0n, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b0deWdt = stablehlo.transpose %s0b0dg, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %s0b0deWraw = stablehlo.convolution(%s0b0deWxt, %s0b0deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %s0b0deW = stablehlo.transpose %s0b0deWraw, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %s0b0deb = stablehlo.reduce(%s0b0dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %s0b0dnri = stablehlo.reshape %s0b0d : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b0dnrdy = stablehlo.reshape %s0b0de : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b0dnnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %s0b0dnep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %s0b0dnsmr = stablehlo.reduce(%s0b0dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b0dnsm = stablehlo.broadcast_in_dim %s0b0dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b0dnmu = stablehlo.divide %s0b0dnsm, %s0b0dnnf : tensor<32x301056xf32>
    %s0b0dnxc = stablehlo.subtract %s0b0dnri, %s0b0dnmu : tensor<32x301056xf32>
    %s0b0dnsq = stablehlo.multiply %s0b0dnxc, %s0b0dnxc : tensor<32x301056xf32>
    %s0b0dnvsr = stablehlo.reduce(%s0b0dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b0dnvs = stablehlo.broadcast_in_dim %s0b0dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b0dnvr = stablehlo.divide %s0b0dnvs, %s0b0dnnf : tensor<32x301056xf32>
    %s0b0dnve = stablehlo.add %s0b0dnvr, %s0b0dnep : tensor<32x301056xf32>
    %s0b0dnistd = stablehlo.rsqrt %s0b0dnve : tensor<32x301056xf32>
    %s0b0dnxh = stablehlo.multiply %s0b0dnxc, %s0b0dnistd : tensor<32x301056xf32>
    %s0b0dngb = stablehlo.broadcast_in_dim %s0b0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b0dndxh = stablehlo.multiply %s0b0dngb, %s0b0dnrdy : tensor<32x301056xf32>
    %s0b0dnsdxr = stablehlo.reduce(%s0b0dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b0dnsdx = stablehlo.broadcast_in_dim %s0b0dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b0dnxd = stablehlo.multiply %s0b0dnxh, %s0b0dndxh : tensor<32x301056xf32>
    %s0b0dnsxdr = stablehlo.reduce(%s0b0dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b0dnsxd = stablehlo.broadcast_in_dim %s0b0dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b0dnt1 = stablehlo.multiply %s0b0dndxh, %s0b0dnnf : tensor<32x301056xf32>
    %s0b0dni1 = stablehlo.subtract %s0b0dnt1, %s0b0dnsdx : tensor<32x301056xf32>
    %s0b0dnxs = stablehlo.multiply %s0b0dnxh, %s0b0dnsxd : tensor<32x301056xf32>
    %s0b0dni2 = stablehlo.subtract %s0b0dni1, %s0b0dnxs : tensor<32x301056xf32>
    %s0b0dnsN = stablehlo.divide %s0b0dnistd, %s0b0dnnf : tensor<32x301056xf32>
    %s0b0dngin = stablehlo.multiply %s0b0dnsN, %s0b0dni2 : tensor<32x301056xf32>
    %s0b0dn = stablehlo.reshape %s0b0dngin : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %s0b0dndgp = stablehlo.multiply %s0b0dnrdy, %s0b0dnxh : tensor<32x301056xf32>
    %s0b0dndg = stablehlo.reduce(%s0b0dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b0dndb = stablehlo.reduce(%s0b0dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b0ddrev = stablehlo.reverse %s0b0dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %s0b0dd = stablehlo.convolution(%s0b0dn, %s0b0ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %s0b0ddWxt = stablehlo.transpose %ps, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b0ddWdt = stablehlo.transpose %s0b0dn, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b0ddWraw = stablehlo.convolution(%s0b0ddWxt, %s0b0ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %s0b0ddW = stablehlo.reshape %s0b0ddWraw : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %s0b0ddb = stablehlo.reduce(%s0b0dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b0dx = stablehlo.add %s0b0dd, %s0b1dx : tensor<32x96x56x56xf32>
    %psdWu = stablehlo.pad %s0b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x221x221xf32>
    %psdWxt = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %psdWdt = stablehlo.transpose %psdWu, dims = [1, 0, 2, 3] : (tensor<32x96x221x221xf32>) -> tensor<96x32x221x221xf32>
    %psdWraw = stablehlo.convolution(%psdWxt, %psdWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<96x32x221x221xf32>) -> tensor<3x96x4x4xf32>
    %psdW = stablehlo.transpose %psdWraw, dims = [1, 0, 2, 3] : (tensor<3x96x4x4xf32>) -> tensor<96x3x4x4xf32>
    %psdb = stablehlo.reduce(%s0b0dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %psWl = stablehlo.constant dense<0.1> : tensor<96x3x4x4xf32>
    %psWs = stablehlo.multiply %psdW, %psWl : tensor<96x3x4x4xf32>
    %psWn = stablehlo.subtract %psW, %psWs : tensor<96x3x4x4xf32>
    %psbl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %psbs = stablehlo.multiply %psdb, %psbl : tensor<96xf32>
    %psbn = stablehlo.subtract %psb, %psbs : tensor<96xf32>
    %s0b0dWl = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %s0b0dWs = stablehlo.multiply %s0b0ddW, %s0b0dWl : tensor<96x1x7x7xf32>
    %s0b0dWn = stablehlo.subtract %s0b0dW, %s0b0dWs : tensor<96x1x7x7xf32>
    %s0b0dbl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %s0b0dbs = stablehlo.multiply %s0b0ddb, %s0b0dbl : tensor<96xf32>
    %s0b0dbn = stablehlo.subtract %s0b0db, %s0b0dbs : tensor<96xf32>
    %s0b0ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s0b0ngs = stablehlo.multiply %s0b0dndg, %s0b0ngl : tensor<f32>
    %s0b0ngn = stablehlo.subtract %s0b0ng, %s0b0ngs : tensor<f32>
    %s0b0nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s0b0nbts = stablehlo.multiply %s0b0dndb, %s0b0nbtl : tensor<f32>
    %s0b0nbtn = stablehlo.subtract %s0b0nbt, %s0b0nbts : tensor<f32>
    %s0b0eWl = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %s0b0eWs = stablehlo.multiply %s0b0deW, %s0b0eWl : tensor<384x96x1x1xf32>
    %s0b0eWn = stablehlo.subtract %s0b0eW, %s0b0eWs : tensor<384x96x1x1xf32>
    %s0b0ebl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s0b0ebs = stablehlo.multiply %s0b0deb, %s0b0ebl : tensor<384xf32>
    %s0b0ebn = stablehlo.subtract %s0b0eb, %s0b0ebs : tensor<384xf32>
    %s0b0pWl = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %s0b0pWs = stablehlo.multiply %s0b0dpW, %s0b0pWl : tensor<96x384x1x1xf32>
    %s0b0pWn = stablehlo.subtract %s0b0pW, %s0b0pWs : tensor<96x384x1x1xf32>
    %s0b0pbl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %s0b0pbs = stablehlo.multiply %s0b0dpb, %s0b0pbl : tensor<96xf32>
    %s0b0pbn = stablehlo.subtract %s0b0pb, %s0b0pbs : tensor<96xf32>
    %s0b0lgl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %s0b0lgs = stablehlo.multiply %s0b0dlsdg, %s0b0lgl : tensor<96xf32>
    %s0b0lgn = stablehlo.subtract %s0b0lg, %s0b0lgs : tensor<96xf32>
    %s0b1dWl = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %s0b1dWs = stablehlo.multiply %s0b1ddW, %s0b1dWl : tensor<96x1x7x7xf32>
    %s0b1dWn = stablehlo.subtract %s0b1dW, %s0b1dWs : tensor<96x1x7x7xf32>
    %s0b1dbl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %s0b1dbs = stablehlo.multiply %s0b1ddb, %s0b1dbl : tensor<96xf32>
    %s0b1dbn = stablehlo.subtract %s0b1db, %s0b1dbs : tensor<96xf32>
    %s0b1ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s0b1ngs = stablehlo.multiply %s0b1dndg, %s0b1ngl : tensor<f32>
    %s0b1ngn = stablehlo.subtract %s0b1ng, %s0b1ngs : tensor<f32>
    %s0b1nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s0b1nbts = stablehlo.multiply %s0b1dndb, %s0b1nbtl : tensor<f32>
    %s0b1nbtn = stablehlo.subtract %s0b1nbt, %s0b1nbts : tensor<f32>
    %s0b1eWl = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %s0b1eWs = stablehlo.multiply %s0b1deW, %s0b1eWl : tensor<384x96x1x1xf32>
    %s0b1eWn = stablehlo.subtract %s0b1eW, %s0b1eWs : tensor<384x96x1x1xf32>
    %s0b1ebl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s0b1ebs = stablehlo.multiply %s0b1deb, %s0b1ebl : tensor<384xf32>
    %s0b1ebn = stablehlo.subtract %s0b1eb, %s0b1ebs : tensor<384xf32>
    %s0b1pWl = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %s0b1pWs = stablehlo.multiply %s0b1dpW, %s0b1pWl : tensor<96x384x1x1xf32>
    %s0b1pWn = stablehlo.subtract %s0b1pW, %s0b1pWs : tensor<96x384x1x1xf32>
    %s0b1pbl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %s0b1pbs = stablehlo.multiply %s0b1dpb, %s0b1pbl : tensor<96xf32>
    %s0b1pbn = stablehlo.subtract %s0b1pb, %s0b1pbs : tensor<96xf32>
    %s0b1lgl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %s0b1lgs = stablehlo.multiply %s0b1dlsdg, %s0b1lgl : tensor<96xf32>
    %s0b1lgn = stablehlo.subtract %s0b1lg, %s0b1lgs : tensor<96xf32>
    %s0b2dWl = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %s0b2dWs = stablehlo.multiply %s0b2ddW, %s0b2dWl : tensor<96x1x7x7xf32>
    %s0b2dWn = stablehlo.subtract %s0b2dW, %s0b2dWs : tensor<96x1x7x7xf32>
    %s0b2dbl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %s0b2dbs = stablehlo.multiply %s0b2ddb, %s0b2dbl : tensor<96xf32>
    %s0b2dbn = stablehlo.subtract %s0b2db, %s0b2dbs : tensor<96xf32>
    %s0b2ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s0b2ngs = stablehlo.multiply %s0b2dndg, %s0b2ngl : tensor<f32>
    %s0b2ngn = stablehlo.subtract %s0b2ng, %s0b2ngs : tensor<f32>
    %s0b2nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s0b2nbts = stablehlo.multiply %s0b2dndb, %s0b2nbtl : tensor<f32>
    %s0b2nbtn = stablehlo.subtract %s0b2nbt, %s0b2nbts : tensor<f32>
    %s0b2eWl = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %s0b2eWs = stablehlo.multiply %s0b2deW, %s0b2eWl : tensor<384x96x1x1xf32>
    %s0b2eWn = stablehlo.subtract %s0b2eW, %s0b2eWs : tensor<384x96x1x1xf32>
    %s0b2ebl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s0b2ebs = stablehlo.multiply %s0b2deb, %s0b2ebl : tensor<384xf32>
    %s0b2ebn = stablehlo.subtract %s0b2eb, %s0b2ebs : tensor<384xf32>
    %s0b2pWl = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %s0b2pWs = stablehlo.multiply %s0b2dpW, %s0b2pWl : tensor<96x384x1x1xf32>
    %s0b2pWn = stablehlo.subtract %s0b2pW, %s0b2pWs : tensor<96x384x1x1xf32>
    %s0b2pbl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %s0b2pbs = stablehlo.multiply %s0b2dpb, %s0b2pbl : tensor<96xf32>
    %s0b2pbn = stablehlo.subtract %s0b2pb, %s0b2pbs : tensor<96xf32>
    %s0b2lgl = stablehlo.constant dense<0.1> : tensor<96xf32>
    %s0b2lgs = stablehlo.multiply %s0b2dlsdg, %s0b2lgl : tensor<96xf32>
    %s0b2lgn = stablehlo.subtract %s0b2lg, %s0b2lgs : tensor<96xf32>
    %d0ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %d0ngs = stablehlo.multiply %d0dndg, %d0ngl : tensor<f32>
    %d0ngn = stablehlo.subtract %d0ng, %d0ngs : tensor<f32>
    %d0nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %d0nbts = stablehlo.multiply %d0dndb, %d0nbtl : tensor<f32>
    %d0nbtn = stablehlo.subtract %d0nbt, %d0nbts : tensor<f32>
    %d0Wl = stablehlo.constant dense<0.1> : tensor<192x96x2x2xf32>
    %d0Ws = stablehlo.multiply %d0dW, %d0Wl : tensor<192x96x2x2xf32>
    %d0Wn = stablehlo.subtract %d0W, %d0Ws : tensor<192x96x2x2xf32>
    %d0bl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %d0bs = stablehlo.multiply %d0db, %d0bl : tensor<192xf32>
    %d0bn = stablehlo.subtract %d0b, %d0bs : tensor<192xf32>
    %s1b0dWl = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %s1b0dWs = stablehlo.multiply %s1b0ddW, %s1b0dWl : tensor<192x1x7x7xf32>
    %s1b0dWn = stablehlo.subtract %s1b0dW, %s1b0dWs : tensor<192x1x7x7xf32>
    %s1b0dbl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %s1b0dbs = stablehlo.multiply %s1b0ddb, %s1b0dbl : tensor<192xf32>
    %s1b0dbn = stablehlo.subtract %s1b0db, %s1b0dbs : tensor<192xf32>
    %s1b0ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s1b0ngs = stablehlo.multiply %s1b0dndg, %s1b0ngl : tensor<f32>
    %s1b0ngn = stablehlo.subtract %s1b0ng, %s1b0ngs : tensor<f32>
    %s1b0nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s1b0nbts = stablehlo.multiply %s1b0dndb, %s1b0nbtl : tensor<f32>
    %s1b0nbtn = stablehlo.subtract %s1b0nbt, %s1b0nbts : tensor<f32>
    %s1b0eWl = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %s1b0eWs = stablehlo.multiply %s1b0deW, %s1b0eWl : tensor<768x192x1x1xf32>
    %s1b0eWn = stablehlo.subtract %s1b0eW, %s1b0eWs : tensor<768x192x1x1xf32>
    %s1b0ebl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %s1b0ebs = stablehlo.multiply %s1b0deb, %s1b0ebl : tensor<768xf32>
    %s1b0ebn = stablehlo.subtract %s1b0eb, %s1b0ebs : tensor<768xf32>
    %s1b0pWl = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %s1b0pWs = stablehlo.multiply %s1b0dpW, %s1b0pWl : tensor<192x768x1x1xf32>
    %s1b0pWn = stablehlo.subtract %s1b0pW, %s1b0pWs : tensor<192x768x1x1xf32>
    %s1b0pbl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %s1b0pbs = stablehlo.multiply %s1b0dpb, %s1b0pbl : tensor<192xf32>
    %s1b0pbn = stablehlo.subtract %s1b0pb, %s1b0pbs : tensor<192xf32>
    %s1b0lgl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %s1b0lgs = stablehlo.multiply %s1b0dlsdg, %s1b0lgl : tensor<192xf32>
    %s1b0lgn = stablehlo.subtract %s1b0lg, %s1b0lgs : tensor<192xf32>
    %s1b1dWl = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %s1b1dWs = stablehlo.multiply %s1b1ddW, %s1b1dWl : tensor<192x1x7x7xf32>
    %s1b1dWn = stablehlo.subtract %s1b1dW, %s1b1dWs : tensor<192x1x7x7xf32>
    %s1b1dbl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %s1b1dbs = stablehlo.multiply %s1b1ddb, %s1b1dbl : tensor<192xf32>
    %s1b1dbn = stablehlo.subtract %s1b1db, %s1b1dbs : tensor<192xf32>
    %s1b1ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s1b1ngs = stablehlo.multiply %s1b1dndg, %s1b1ngl : tensor<f32>
    %s1b1ngn = stablehlo.subtract %s1b1ng, %s1b1ngs : tensor<f32>
    %s1b1nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s1b1nbts = stablehlo.multiply %s1b1dndb, %s1b1nbtl : tensor<f32>
    %s1b1nbtn = stablehlo.subtract %s1b1nbt, %s1b1nbts : tensor<f32>
    %s1b1eWl = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %s1b1eWs = stablehlo.multiply %s1b1deW, %s1b1eWl : tensor<768x192x1x1xf32>
    %s1b1eWn = stablehlo.subtract %s1b1eW, %s1b1eWs : tensor<768x192x1x1xf32>
    %s1b1ebl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %s1b1ebs = stablehlo.multiply %s1b1deb, %s1b1ebl : tensor<768xf32>
    %s1b1ebn = stablehlo.subtract %s1b1eb, %s1b1ebs : tensor<768xf32>
    %s1b1pWl = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %s1b1pWs = stablehlo.multiply %s1b1dpW, %s1b1pWl : tensor<192x768x1x1xf32>
    %s1b1pWn = stablehlo.subtract %s1b1pW, %s1b1pWs : tensor<192x768x1x1xf32>
    %s1b1pbl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %s1b1pbs = stablehlo.multiply %s1b1dpb, %s1b1pbl : tensor<192xf32>
    %s1b1pbn = stablehlo.subtract %s1b1pb, %s1b1pbs : tensor<192xf32>
    %s1b1lgl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %s1b1lgs = stablehlo.multiply %s1b1dlsdg, %s1b1lgl : tensor<192xf32>
    %s1b1lgn = stablehlo.subtract %s1b1lg, %s1b1lgs : tensor<192xf32>
    %s1b2dWl = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %s1b2dWs = stablehlo.multiply %s1b2ddW, %s1b2dWl : tensor<192x1x7x7xf32>
    %s1b2dWn = stablehlo.subtract %s1b2dW, %s1b2dWs : tensor<192x1x7x7xf32>
    %s1b2dbl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %s1b2dbs = stablehlo.multiply %s1b2ddb, %s1b2dbl : tensor<192xf32>
    %s1b2dbn = stablehlo.subtract %s1b2db, %s1b2dbs : tensor<192xf32>
    %s1b2ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s1b2ngs = stablehlo.multiply %s1b2dndg, %s1b2ngl : tensor<f32>
    %s1b2ngn = stablehlo.subtract %s1b2ng, %s1b2ngs : tensor<f32>
    %s1b2nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s1b2nbts = stablehlo.multiply %s1b2dndb, %s1b2nbtl : tensor<f32>
    %s1b2nbtn = stablehlo.subtract %s1b2nbt, %s1b2nbts : tensor<f32>
    %s1b2eWl = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %s1b2eWs = stablehlo.multiply %s1b2deW, %s1b2eWl : tensor<768x192x1x1xf32>
    %s1b2eWn = stablehlo.subtract %s1b2eW, %s1b2eWs : tensor<768x192x1x1xf32>
    %s1b2ebl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %s1b2ebs = stablehlo.multiply %s1b2deb, %s1b2ebl : tensor<768xf32>
    %s1b2ebn = stablehlo.subtract %s1b2eb, %s1b2ebs : tensor<768xf32>
    %s1b2pWl = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %s1b2pWs = stablehlo.multiply %s1b2dpW, %s1b2pWl : tensor<192x768x1x1xf32>
    %s1b2pWn = stablehlo.subtract %s1b2pW, %s1b2pWs : tensor<192x768x1x1xf32>
    %s1b2pbl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %s1b2pbs = stablehlo.multiply %s1b2dpb, %s1b2pbl : tensor<192xf32>
    %s1b2pbn = stablehlo.subtract %s1b2pb, %s1b2pbs : tensor<192xf32>
    %s1b2lgl = stablehlo.constant dense<0.1> : tensor<192xf32>
    %s1b2lgs = stablehlo.multiply %s1b2dlsdg, %s1b2lgl : tensor<192xf32>
    %s1b2lgn = stablehlo.subtract %s1b2lg, %s1b2lgs : tensor<192xf32>
    %d1ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %d1ngs = stablehlo.multiply %d1dndg, %d1ngl : tensor<f32>
    %d1ngn = stablehlo.subtract %d1ng, %d1ngs : tensor<f32>
    %d1nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %d1nbts = stablehlo.multiply %d1dndb, %d1nbtl : tensor<f32>
    %d1nbtn = stablehlo.subtract %d1nbt, %d1nbts : tensor<f32>
    %d1Wl = stablehlo.constant dense<0.1> : tensor<384x192x2x2xf32>
    %d1Ws = stablehlo.multiply %d1dW, %d1Wl : tensor<384x192x2x2xf32>
    %d1Wn = stablehlo.subtract %d1W, %d1Ws : tensor<384x192x2x2xf32>
    %d1bl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %d1bs = stablehlo.multiply %d1db, %d1bl : tensor<384xf32>
    %d1bn = stablehlo.subtract %d1b, %d1bs : tensor<384xf32>
    %s2b0dWl = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %s2b0dWs = stablehlo.multiply %s2b0ddW, %s2b0dWl : tensor<384x1x7x7xf32>
    %s2b0dWn = stablehlo.subtract %s2b0dW, %s2b0dWs : tensor<384x1x7x7xf32>
    %s2b0dbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b0dbs = stablehlo.multiply %s2b0ddb, %s2b0dbl : tensor<384xf32>
    %s2b0dbn = stablehlo.subtract %s2b0db, %s2b0dbs : tensor<384xf32>
    %s2b0ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b0ngs = stablehlo.multiply %s2b0dndg, %s2b0ngl : tensor<f32>
    %s2b0ngn = stablehlo.subtract %s2b0ng, %s2b0ngs : tensor<f32>
    %s2b0nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b0nbts = stablehlo.multiply %s2b0dndb, %s2b0nbtl : tensor<f32>
    %s2b0nbtn = stablehlo.subtract %s2b0nbt, %s2b0nbts : tensor<f32>
    %s2b0eWl = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %s2b0eWs = stablehlo.multiply %s2b0deW, %s2b0eWl : tensor<1536x384x1x1xf32>
    %s2b0eWn = stablehlo.subtract %s2b0eW, %s2b0eWs : tensor<1536x384x1x1xf32>
    %s2b0ebl = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %s2b0ebs = stablehlo.multiply %s2b0deb, %s2b0ebl : tensor<1536xf32>
    %s2b0ebn = stablehlo.subtract %s2b0eb, %s2b0ebs : tensor<1536xf32>
    %s2b0pWl = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %s2b0pWs = stablehlo.multiply %s2b0dpW, %s2b0pWl : tensor<384x1536x1x1xf32>
    %s2b0pWn = stablehlo.subtract %s2b0pW, %s2b0pWs : tensor<384x1536x1x1xf32>
    %s2b0pbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b0pbs = stablehlo.multiply %s2b0dpb, %s2b0pbl : tensor<384xf32>
    %s2b0pbn = stablehlo.subtract %s2b0pb, %s2b0pbs : tensor<384xf32>
    %s2b0lgl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b0lgs = stablehlo.multiply %s2b0dlsdg, %s2b0lgl : tensor<384xf32>
    %s2b0lgn = stablehlo.subtract %s2b0lg, %s2b0lgs : tensor<384xf32>
    %s2b1dWl = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %s2b1dWs = stablehlo.multiply %s2b1ddW, %s2b1dWl : tensor<384x1x7x7xf32>
    %s2b1dWn = stablehlo.subtract %s2b1dW, %s2b1dWs : tensor<384x1x7x7xf32>
    %s2b1dbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b1dbs = stablehlo.multiply %s2b1ddb, %s2b1dbl : tensor<384xf32>
    %s2b1dbn = stablehlo.subtract %s2b1db, %s2b1dbs : tensor<384xf32>
    %s2b1ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b1ngs = stablehlo.multiply %s2b1dndg, %s2b1ngl : tensor<f32>
    %s2b1ngn = stablehlo.subtract %s2b1ng, %s2b1ngs : tensor<f32>
    %s2b1nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b1nbts = stablehlo.multiply %s2b1dndb, %s2b1nbtl : tensor<f32>
    %s2b1nbtn = stablehlo.subtract %s2b1nbt, %s2b1nbts : tensor<f32>
    %s2b1eWl = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %s2b1eWs = stablehlo.multiply %s2b1deW, %s2b1eWl : tensor<1536x384x1x1xf32>
    %s2b1eWn = stablehlo.subtract %s2b1eW, %s2b1eWs : tensor<1536x384x1x1xf32>
    %s2b1ebl = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %s2b1ebs = stablehlo.multiply %s2b1deb, %s2b1ebl : tensor<1536xf32>
    %s2b1ebn = stablehlo.subtract %s2b1eb, %s2b1ebs : tensor<1536xf32>
    %s2b1pWl = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %s2b1pWs = stablehlo.multiply %s2b1dpW, %s2b1pWl : tensor<384x1536x1x1xf32>
    %s2b1pWn = stablehlo.subtract %s2b1pW, %s2b1pWs : tensor<384x1536x1x1xf32>
    %s2b1pbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b1pbs = stablehlo.multiply %s2b1dpb, %s2b1pbl : tensor<384xf32>
    %s2b1pbn = stablehlo.subtract %s2b1pb, %s2b1pbs : tensor<384xf32>
    %s2b1lgl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b1lgs = stablehlo.multiply %s2b1dlsdg, %s2b1lgl : tensor<384xf32>
    %s2b1lgn = stablehlo.subtract %s2b1lg, %s2b1lgs : tensor<384xf32>
    %s2b2dWl = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %s2b2dWs = stablehlo.multiply %s2b2ddW, %s2b2dWl : tensor<384x1x7x7xf32>
    %s2b2dWn = stablehlo.subtract %s2b2dW, %s2b2dWs : tensor<384x1x7x7xf32>
    %s2b2dbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b2dbs = stablehlo.multiply %s2b2ddb, %s2b2dbl : tensor<384xf32>
    %s2b2dbn = stablehlo.subtract %s2b2db, %s2b2dbs : tensor<384xf32>
    %s2b2ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b2ngs = stablehlo.multiply %s2b2dndg, %s2b2ngl : tensor<f32>
    %s2b2ngn = stablehlo.subtract %s2b2ng, %s2b2ngs : tensor<f32>
    %s2b2nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b2nbts = stablehlo.multiply %s2b2dndb, %s2b2nbtl : tensor<f32>
    %s2b2nbtn = stablehlo.subtract %s2b2nbt, %s2b2nbts : tensor<f32>
    %s2b2eWl = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %s2b2eWs = stablehlo.multiply %s2b2deW, %s2b2eWl : tensor<1536x384x1x1xf32>
    %s2b2eWn = stablehlo.subtract %s2b2eW, %s2b2eWs : tensor<1536x384x1x1xf32>
    %s2b2ebl = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %s2b2ebs = stablehlo.multiply %s2b2deb, %s2b2ebl : tensor<1536xf32>
    %s2b2ebn = stablehlo.subtract %s2b2eb, %s2b2ebs : tensor<1536xf32>
    %s2b2pWl = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %s2b2pWs = stablehlo.multiply %s2b2dpW, %s2b2pWl : tensor<384x1536x1x1xf32>
    %s2b2pWn = stablehlo.subtract %s2b2pW, %s2b2pWs : tensor<384x1536x1x1xf32>
    %s2b2pbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b2pbs = stablehlo.multiply %s2b2dpb, %s2b2pbl : tensor<384xf32>
    %s2b2pbn = stablehlo.subtract %s2b2pb, %s2b2pbs : tensor<384xf32>
    %s2b2lgl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b2lgs = stablehlo.multiply %s2b2dlsdg, %s2b2lgl : tensor<384xf32>
    %s2b2lgn = stablehlo.subtract %s2b2lg, %s2b2lgs : tensor<384xf32>
    %s2b3dWl = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %s2b3dWs = stablehlo.multiply %s2b3ddW, %s2b3dWl : tensor<384x1x7x7xf32>
    %s2b3dWn = stablehlo.subtract %s2b3dW, %s2b3dWs : tensor<384x1x7x7xf32>
    %s2b3dbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b3dbs = stablehlo.multiply %s2b3ddb, %s2b3dbl : tensor<384xf32>
    %s2b3dbn = stablehlo.subtract %s2b3db, %s2b3dbs : tensor<384xf32>
    %s2b3ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b3ngs = stablehlo.multiply %s2b3dndg, %s2b3ngl : tensor<f32>
    %s2b3ngn = stablehlo.subtract %s2b3ng, %s2b3ngs : tensor<f32>
    %s2b3nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b3nbts = stablehlo.multiply %s2b3dndb, %s2b3nbtl : tensor<f32>
    %s2b3nbtn = stablehlo.subtract %s2b3nbt, %s2b3nbts : tensor<f32>
    %s2b3eWl = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %s2b3eWs = stablehlo.multiply %s2b3deW, %s2b3eWl : tensor<1536x384x1x1xf32>
    %s2b3eWn = stablehlo.subtract %s2b3eW, %s2b3eWs : tensor<1536x384x1x1xf32>
    %s2b3ebl = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %s2b3ebs = stablehlo.multiply %s2b3deb, %s2b3ebl : tensor<1536xf32>
    %s2b3ebn = stablehlo.subtract %s2b3eb, %s2b3ebs : tensor<1536xf32>
    %s2b3pWl = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %s2b3pWs = stablehlo.multiply %s2b3dpW, %s2b3pWl : tensor<384x1536x1x1xf32>
    %s2b3pWn = stablehlo.subtract %s2b3pW, %s2b3pWs : tensor<384x1536x1x1xf32>
    %s2b3pbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b3pbs = stablehlo.multiply %s2b3dpb, %s2b3pbl : tensor<384xf32>
    %s2b3pbn = stablehlo.subtract %s2b3pb, %s2b3pbs : tensor<384xf32>
    %s2b3lgl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b3lgs = stablehlo.multiply %s2b3dlsdg, %s2b3lgl : tensor<384xf32>
    %s2b3lgn = stablehlo.subtract %s2b3lg, %s2b3lgs : tensor<384xf32>
    %s2b4dWl = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %s2b4dWs = stablehlo.multiply %s2b4ddW, %s2b4dWl : tensor<384x1x7x7xf32>
    %s2b4dWn = stablehlo.subtract %s2b4dW, %s2b4dWs : tensor<384x1x7x7xf32>
    %s2b4dbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b4dbs = stablehlo.multiply %s2b4ddb, %s2b4dbl : tensor<384xf32>
    %s2b4dbn = stablehlo.subtract %s2b4db, %s2b4dbs : tensor<384xf32>
    %s2b4ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b4ngs = stablehlo.multiply %s2b4dndg, %s2b4ngl : tensor<f32>
    %s2b4ngn = stablehlo.subtract %s2b4ng, %s2b4ngs : tensor<f32>
    %s2b4nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b4nbts = stablehlo.multiply %s2b4dndb, %s2b4nbtl : tensor<f32>
    %s2b4nbtn = stablehlo.subtract %s2b4nbt, %s2b4nbts : tensor<f32>
    %s2b4eWl = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %s2b4eWs = stablehlo.multiply %s2b4deW, %s2b4eWl : tensor<1536x384x1x1xf32>
    %s2b4eWn = stablehlo.subtract %s2b4eW, %s2b4eWs : tensor<1536x384x1x1xf32>
    %s2b4ebl = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %s2b4ebs = stablehlo.multiply %s2b4deb, %s2b4ebl : tensor<1536xf32>
    %s2b4ebn = stablehlo.subtract %s2b4eb, %s2b4ebs : tensor<1536xf32>
    %s2b4pWl = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %s2b4pWs = stablehlo.multiply %s2b4dpW, %s2b4pWl : tensor<384x1536x1x1xf32>
    %s2b4pWn = stablehlo.subtract %s2b4pW, %s2b4pWs : tensor<384x1536x1x1xf32>
    %s2b4pbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b4pbs = stablehlo.multiply %s2b4dpb, %s2b4pbl : tensor<384xf32>
    %s2b4pbn = stablehlo.subtract %s2b4pb, %s2b4pbs : tensor<384xf32>
    %s2b4lgl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b4lgs = stablehlo.multiply %s2b4dlsdg, %s2b4lgl : tensor<384xf32>
    %s2b4lgn = stablehlo.subtract %s2b4lg, %s2b4lgs : tensor<384xf32>
    %s2b5dWl = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %s2b5dWs = stablehlo.multiply %s2b5ddW, %s2b5dWl : tensor<384x1x7x7xf32>
    %s2b5dWn = stablehlo.subtract %s2b5dW, %s2b5dWs : tensor<384x1x7x7xf32>
    %s2b5dbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b5dbs = stablehlo.multiply %s2b5ddb, %s2b5dbl : tensor<384xf32>
    %s2b5dbn = stablehlo.subtract %s2b5db, %s2b5dbs : tensor<384xf32>
    %s2b5ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b5ngs = stablehlo.multiply %s2b5dndg, %s2b5ngl : tensor<f32>
    %s2b5ngn = stablehlo.subtract %s2b5ng, %s2b5ngs : tensor<f32>
    %s2b5nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b5nbts = stablehlo.multiply %s2b5dndb, %s2b5nbtl : tensor<f32>
    %s2b5nbtn = stablehlo.subtract %s2b5nbt, %s2b5nbts : tensor<f32>
    %s2b5eWl = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %s2b5eWs = stablehlo.multiply %s2b5deW, %s2b5eWl : tensor<1536x384x1x1xf32>
    %s2b5eWn = stablehlo.subtract %s2b5eW, %s2b5eWs : tensor<1536x384x1x1xf32>
    %s2b5ebl = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %s2b5ebs = stablehlo.multiply %s2b5deb, %s2b5ebl : tensor<1536xf32>
    %s2b5ebn = stablehlo.subtract %s2b5eb, %s2b5ebs : tensor<1536xf32>
    %s2b5pWl = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %s2b5pWs = stablehlo.multiply %s2b5dpW, %s2b5pWl : tensor<384x1536x1x1xf32>
    %s2b5pWn = stablehlo.subtract %s2b5pW, %s2b5pWs : tensor<384x1536x1x1xf32>
    %s2b5pbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b5pbs = stablehlo.multiply %s2b5dpb, %s2b5pbl : tensor<384xf32>
    %s2b5pbn = stablehlo.subtract %s2b5pb, %s2b5pbs : tensor<384xf32>
    %s2b5lgl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b5lgs = stablehlo.multiply %s2b5dlsdg, %s2b5lgl : tensor<384xf32>
    %s2b5lgn = stablehlo.subtract %s2b5lg, %s2b5lgs : tensor<384xf32>
    %s2b6dWl = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %s2b6dWs = stablehlo.multiply %s2b6ddW, %s2b6dWl : tensor<384x1x7x7xf32>
    %s2b6dWn = stablehlo.subtract %s2b6dW, %s2b6dWs : tensor<384x1x7x7xf32>
    %s2b6dbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b6dbs = stablehlo.multiply %s2b6ddb, %s2b6dbl : tensor<384xf32>
    %s2b6dbn = stablehlo.subtract %s2b6db, %s2b6dbs : tensor<384xf32>
    %s2b6ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b6ngs = stablehlo.multiply %s2b6dndg, %s2b6ngl : tensor<f32>
    %s2b6ngn = stablehlo.subtract %s2b6ng, %s2b6ngs : tensor<f32>
    %s2b6nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b6nbts = stablehlo.multiply %s2b6dndb, %s2b6nbtl : tensor<f32>
    %s2b6nbtn = stablehlo.subtract %s2b6nbt, %s2b6nbts : tensor<f32>
    %s2b6eWl = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %s2b6eWs = stablehlo.multiply %s2b6deW, %s2b6eWl : tensor<1536x384x1x1xf32>
    %s2b6eWn = stablehlo.subtract %s2b6eW, %s2b6eWs : tensor<1536x384x1x1xf32>
    %s2b6ebl = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %s2b6ebs = stablehlo.multiply %s2b6deb, %s2b6ebl : tensor<1536xf32>
    %s2b6ebn = stablehlo.subtract %s2b6eb, %s2b6ebs : tensor<1536xf32>
    %s2b6pWl = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %s2b6pWs = stablehlo.multiply %s2b6dpW, %s2b6pWl : tensor<384x1536x1x1xf32>
    %s2b6pWn = stablehlo.subtract %s2b6pW, %s2b6pWs : tensor<384x1536x1x1xf32>
    %s2b6pbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b6pbs = stablehlo.multiply %s2b6dpb, %s2b6pbl : tensor<384xf32>
    %s2b6pbn = stablehlo.subtract %s2b6pb, %s2b6pbs : tensor<384xf32>
    %s2b6lgl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b6lgs = stablehlo.multiply %s2b6dlsdg, %s2b6lgl : tensor<384xf32>
    %s2b6lgn = stablehlo.subtract %s2b6lg, %s2b6lgs : tensor<384xf32>
    %s2b7dWl = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %s2b7dWs = stablehlo.multiply %s2b7ddW, %s2b7dWl : tensor<384x1x7x7xf32>
    %s2b7dWn = stablehlo.subtract %s2b7dW, %s2b7dWs : tensor<384x1x7x7xf32>
    %s2b7dbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b7dbs = stablehlo.multiply %s2b7ddb, %s2b7dbl : tensor<384xf32>
    %s2b7dbn = stablehlo.subtract %s2b7db, %s2b7dbs : tensor<384xf32>
    %s2b7ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b7ngs = stablehlo.multiply %s2b7dndg, %s2b7ngl : tensor<f32>
    %s2b7ngn = stablehlo.subtract %s2b7ng, %s2b7ngs : tensor<f32>
    %s2b7nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b7nbts = stablehlo.multiply %s2b7dndb, %s2b7nbtl : tensor<f32>
    %s2b7nbtn = stablehlo.subtract %s2b7nbt, %s2b7nbts : tensor<f32>
    %s2b7eWl = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %s2b7eWs = stablehlo.multiply %s2b7deW, %s2b7eWl : tensor<1536x384x1x1xf32>
    %s2b7eWn = stablehlo.subtract %s2b7eW, %s2b7eWs : tensor<1536x384x1x1xf32>
    %s2b7ebl = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %s2b7ebs = stablehlo.multiply %s2b7deb, %s2b7ebl : tensor<1536xf32>
    %s2b7ebn = stablehlo.subtract %s2b7eb, %s2b7ebs : tensor<1536xf32>
    %s2b7pWl = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %s2b7pWs = stablehlo.multiply %s2b7dpW, %s2b7pWl : tensor<384x1536x1x1xf32>
    %s2b7pWn = stablehlo.subtract %s2b7pW, %s2b7pWs : tensor<384x1536x1x1xf32>
    %s2b7pbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b7pbs = stablehlo.multiply %s2b7dpb, %s2b7pbl : tensor<384xf32>
    %s2b7pbn = stablehlo.subtract %s2b7pb, %s2b7pbs : tensor<384xf32>
    %s2b7lgl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b7lgs = stablehlo.multiply %s2b7dlsdg, %s2b7lgl : tensor<384xf32>
    %s2b7lgn = stablehlo.subtract %s2b7lg, %s2b7lgs : tensor<384xf32>
    %s2b8dWl = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %s2b8dWs = stablehlo.multiply %s2b8ddW, %s2b8dWl : tensor<384x1x7x7xf32>
    %s2b8dWn = stablehlo.subtract %s2b8dW, %s2b8dWs : tensor<384x1x7x7xf32>
    %s2b8dbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b8dbs = stablehlo.multiply %s2b8ddb, %s2b8dbl : tensor<384xf32>
    %s2b8dbn = stablehlo.subtract %s2b8db, %s2b8dbs : tensor<384xf32>
    %s2b8ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b8ngs = stablehlo.multiply %s2b8dndg, %s2b8ngl : tensor<f32>
    %s2b8ngn = stablehlo.subtract %s2b8ng, %s2b8ngs : tensor<f32>
    %s2b8nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s2b8nbts = stablehlo.multiply %s2b8dndb, %s2b8nbtl : tensor<f32>
    %s2b8nbtn = stablehlo.subtract %s2b8nbt, %s2b8nbts : tensor<f32>
    %s2b8eWl = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %s2b8eWs = stablehlo.multiply %s2b8deW, %s2b8eWl : tensor<1536x384x1x1xf32>
    %s2b8eWn = stablehlo.subtract %s2b8eW, %s2b8eWs : tensor<1536x384x1x1xf32>
    %s2b8ebl = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %s2b8ebs = stablehlo.multiply %s2b8deb, %s2b8ebl : tensor<1536xf32>
    %s2b8ebn = stablehlo.subtract %s2b8eb, %s2b8ebs : tensor<1536xf32>
    %s2b8pWl = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %s2b8pWs = stablehlo.multiply %s2b8dpW, %s2b8pWl : tensor<384x1536x1x1xf32>
    %s2b8pWn = stablehlo.subtract %s2b8pW, %s2b8pWs : tensor<384x1536x1x1xf32>
    %s2b8pbl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b8pbs = stablehlo.multiply %s2b8dpb, %s2b8pbl : tensor<384xf32>
    %s2b8pbn = stablehlo.subtract %s2b8pb, %s2b8pbs : tensor<384xf32>
    %s2b8lgl = stablehlo.constant dense<0.1> : tensor<384xf32>
    %s2b8lgs = stablehlo.multiply %s2b8dlsdg, %s2b8lgl : tensor<384xf32>
    %s2b8lgn = stablehlo.subtract %s2b8lg, %s2b8lgs : tensor<384xf32>
    %d2ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %d2ngs = stablehlo.multiply %d2dndg, %d2ngl : tensor<f32>
    %d2ngn = stablehlo.subtract %d2ng, %d2ngs : tensor<f32>
    %d2nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %d2nbts = stablehlo.multiply %d2dndb, %d2nbtl : tensor<f32>
    %d2nbtn = stablehlo.subtract %d2nbt, %d2nbts : tensor<f32>
    %d2Wl = stablehlo.constant dense<0.1> : tensor<768x384x2x2xf32>
    %d2Ws = stablehlo.multiply %d2dW, %d2Wl : tensor<768x384x2x2xf32>
    %d2Wn = stablehlo.subtract %d2W, %d2Ws : tensor<768x384x2x2xf32>
    %d2bl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %d2bs = stablehlo.multiply %d2db, %d2bl : tensor<768xf32>
    %d2bn = stablehlo.subtract %d2b, %d2bs : tensor<768xf32>
    %s3b0dWl = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %s3b0dWs = stablehlo.multiply %s3b0ddW, %s3b0dWl : tensor<768x1x7x7xf32>
    %s3b0dWn = stablehlo.subtract %s3b0dW, %s3b0dWs : tensor<768x1x7x7xf32>
    %s3b0dbl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %s3b0dbs = stablehlo.multiply %s3b0ddb, %s3b0dbl : tensor<768xf32>
    %s3b0dbn = stablehlo.subtract %s3b0db, %s3b0dbs : tensor<768xf32>
    %s3b0ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s3b0ngs = stablehlo.multiply %s3b0dndg, %s3b0ngl : tensor<f32>
    %s3b0ngn = stablehlo.subtract %s3b0ng, %s3b0ngs : tensor<f32>
    %s3b0nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s3b0nbts = stablehlo.multiply %s3b0dndb, %s3b0nbtl : tensor<f32>
    %s3b0nbtn = stablehlo.subtract %s3b0nbt, %s3b0nbts : tensor<f32>
    %s3b0eWl = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %s3b0eWs = stablehlo.multiply %s3b0deW, %s3b0eWl : tensor<3072x768x1x1xf32>
    %s3b0eWn = stablehlo.subtract %s3b0eW, %s3b0eWs : tensor<3072x768x1x1xf32>
    %s3b0ebl = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %s3b0ebs = stablehlo.multiply %s3b0deb, %s3b0ebl : tensor<3072xf32>
    %s3b0ebn = stablehlo.subtract %s3b0eb, %s3b0ebs : tensor<3072xf32>
    %s3b0pWl = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %s3b0pWs = stablehlo.multiply %s3b0dpW, %s3b0pWl : tensor<768x3072x1x1xf32>
    %s3b0pWn = stablehlo.subtract %s3b0pW, %s3b0pWs : tensor<768x3072x1x1xf32>
    %s3b0pbl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %s3b0pbs = stablehlo.multiply %s3b0dpb, %s3b0pbl : tensor<768xf32>
    %s3b0pbn = stablehlo.subtract %s3b0pb, %s3b0pbs : tensor<768xf32>
    %s3b0lgl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %s3b0lgs = stablehlo.multiply %s3b0dlsdg, %s3b0lgl : tensor<768xf32>
    %s3b0lgn = stablehlo.subtract %s3b0lg, %s3b0lgs : tensor<768xf32>
    %s3b1dWl = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %s3b1dWs = stablehlo.multiply %s3b1ddW, %s3b1dWl : tensor<768x1x7x7xf32>
    %s3b1dWn = stablehlo.subtract %s3b1dW, %s3b1dWs : tensor<768x1x7x7xf32>
    %s3b1dbl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %s3b1dbs = stablehlo.multiply %s3b1ddb, %s3b1dbl : tensor<768xf32>
    %s3b1dbn = stablehlo.subtract %s3b1db, %s3b1dbs : tensor<768xf32>
    %s3b1ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s3b1ngs = stablehlo.multiply %s3b1dndg, %s3b1ngl : tensor<f32>
    %s3b1ngn = stablehlo.subtract %s3b1ng, %s3b1ngs : tensor<f32>
    %s3b1nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s3b1nbts = stablehlo.multiply %s3b1dndb, %s3b1nbtl : tensor<f32>
    %s3b1nbtn = stablehlo.subtract %s3b1nbt, %s3b1nbts : tensor<f32>
    %s3b1eWl = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %s3b1eWs = stablehlo.multiply %s3b1deW, %s3b1eWl : tensor<3072x768x1x1xf32>
    %s3b1eWn = stablehlo.subtract %s3b1eW, %s3b1eWs : tensor<3072x768x1x1xf32>
    %s3b1ebl = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %s3b1ebs = stablehlo.multiply %s3b1deb, %s3b1ebl : tensor<3072xf32>
    %s3b1ebn = stablehlo.subtract %s3b1eb, %s3b1ebs : tensor<3072xf32>
    %s3b1pWl = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %s3b1pWs = stablehlo.multiply %s3b1dpW, %s3b1pWl : tensor<768x3072x1x1xf32>
    %s3b1pWn = stablehlo.subtract %s3b1pW, %s3b1pWs : tensor<768x3072x1x1xf32>
    %s3b1pbl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %s3b1pbs = stablehlo.multiply %s3b1dpb, %s3b1pbl : tensor<768xf32>
    %s3b1pbn = stablehlo.subtract %s3b1pb, %s3b1pbs : tensor<768xf32>
    %s3b1lgl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %s3b1lgs = stablehlo.multiply %s3b1dlsdg, %s3b1lgl : tensor<768xf32>
    %s3b1lgn = stablehlo.subtract %s3b1lg, %s3b1lgs : tensor<768xf32>
    %s3b2dWl = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %s3b2dWs = stablehlo.multiply %s3b2ddW, %s3b2dWl : tensor<768x1x7x7xf32>
    %s3b2dWn = stablehlo.subtract %s3b2dW, %s3b2dWs : tensor<768x1x7x7xf32>
    %s3b2dbl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %s3b2dbs = stablehlo.multiply %s3b2ddb, %s3b2dbl : tensor<768xf32>
    %s3b2dbn = stablehlo.subtract %s3b2db, %s3b2dbs : tensor<768xf32>
    %s3b2ngl = stablehlo.constant dense<0.1> : tensor<f32>
    %s3b2ngs = stablehlo.multiply %s3b2dndg, %s3b2ngl : tensor<f32>
    %s3b2ngn = stablehlo.subtract %s3b2ng, %s3b2ngs : tensor<f32>
    %s3b2nbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %s3b2nbts = stablehlo.multiply %s3b2dndb, %s3b2nbtl : tensor<f32>
    %s3b2nbtn = stablehlo.subtract %s3b2nbt, %s3b2nbts : tensor<f32>
    %s3b2eWl = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %s3b2eWs = stablehlo.multiply %s3b2deW, %s3b2eWl : tensor<3072x768x1x1xf32>
    %s3b2eWn = stablehlo.subtract %s3b2eW, %s3b2eWs : tensor<3072x768x1x1xf32>
    %s3b2ebl = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %s3b2ebs = stablehlo.multiply %s3b2deb, %s3b2ebl : tensor<3072xf32>
    %s3b2ebn = stablehlo.subtract %s3b2eb, %s3b2ebs : tensor<3072xf32>
    %s3b2pWl = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %s3b2pWs = stablehlo.multiply %s3b2dpW, %s3b2pWl : tensor<768x3072x1x1xf32>
    %s3b2pWn = stablehlo.subtract %s3b2pW, %s3b2pWs : tensor<768x3072x1x1xf32>
    %s3b2pbl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %s3b2pbs = stablehlo.multiply %s3b2dpb, %s3b2pbl : tensor<768xf32>
    %s3b2pbn = stablehlo.subtract %s3b2pb, %s3b2pbs : tensor<768xf32>
    %s3b2lgl = stablehlo.constant dense<0.1> : tensor<768xf32>
    %s3b2lgs = stablehlo.multiply %s3b2dlsdg, %s3b2lgl : tensor<768xf32>
    %s3b2lgn = stablehlo.subtract %s3b2lg, %s3b2lgs : tensor<768xf32>
    %hngl = stablehlo.constant dense<0.1> : tensor<f32>
    %hngs = stablehlo.multiply %hddg, %hngl : tensor<f32>
    %hngn = stablehlo.subtract %hng, %hngs : tensor<f32>
    %hnbtl = stablehlo.constant dense<0.1> : tensor<f32>
    %hnbts = stablehlo.multiply %hddb, %hnbtl : tensor<f32>
    %hnbtn = stablehlo.subtract %hnbt, %hnbts : tensor<f32>
    %Wdl = stablehlo.constant dense<0.1> : tensor<768x10xf32>
    %Wds = stablehlo.multiply %dWd, %Wdl : tensor<768x10xf32>
    %Wdn = stablehlo.subtract %Wd, %Wds : tensor<768x10xf32>
    %bdl = stablehlo.constant dense<0.1> : tensor<10xf32>
    %bds = stablehlo.multiply %dbd, %bdl : tensor<10xf32>
    %bdn = stablehlo.subtract %bd, %bds : tensor<10xf32>
    return %psWn, %psbn, %s0b0dWn, %s0b0dbn, %s0b0ngn, %s0b0nbtn, %s0b0eWn, %s0b0ebn, %s0b0pWn, %s0b0pbn, %s0b0lgn, %s0b1dWn, %s0b1dbn, %s0b1ngn, %s0b1nbtn, %s0b1eWn, %s0b1ebn, %s0b1pWn, %s0b1pbn, %s0b1lgn, %s0b2dWn, %s0b2dbn, %s0b2ngn, %s0b2nbtn, %s0b2eWn, %s0b2ebn, %s0b2pWn, %s0b2pbn, %s0b2lgn, %d0ngn, %d0nbtn, %d0Wn, %d0bn, %s1b0dWn, %s1b0dbn, %s1b0ngn, %s1b0nbtn, %s1b0eWn, %s1b0ebn, %s1b0pWn, %s1b0pbn, %s1b0lgn, %s1b1dWn, %s1b1dbn, %s1b1ngn, %s1b1nbtn, %s1b1eWn, %s1b1ebn, %s1b1pWn, %s1b1pbn, %s1b1lgn, %s1b2dWn, %s1b2dbn, %s1b2ngn, %s1b2nbtn, %s1b2eWn, %s1b2ebn, %s1b2pWn, %s1b2pbn, %s1b2lgn, %d1ngn, %d1nbtn, %d1Wn, %d1bn, %s2b0dWn, %s2b0dbn, %s2b0ngn, %s2b0nbtn, %s2b0eWn, %s2b0ebn, %s2b0pWn, %s2b0pbn, %s2b0lgn, %s2b1dWn, %s2b1dbn, %s2b1ngn, %s2b1nbtn, %s2b1eWn, %s2b1ebn, %s2b1pWn, %s2b1pbn, %s2b1lgn, %s2b2dWn, %s2b2dbn, %s2b2ngn, %s2b2nbtn, %s2b2eWn, %s2b2ebn, %s2b2pWn, %s2b2pbn, %s2b2lgn, %s2b3dWn, %s2b3dbn, %s2b3ngn, %s2b3nbtn, %s2b3eWn, %s2b3ebn, %s2b3pWn, %s2b3pbn, %s2b3lgn, %s2b4dWn, %s2b4dbn, %s2b4ngn, %s2b4nbtn, %s2b4eWn, %s2b4ebn, %s2b4pWn, %s2b4pbn, %s2b4lgn, %s2b5dWn, %s2b5dbn, %s2b5ngn, %s2b5nbtn, %s2b5eWn, %s2b5ebn, %s2b5pWn, %s2b5pbn, %s2b5lgn, %s2b6dWn, %s2b6dbn, %s2b6ngn, %s2b6nbtn, %s2b6eWn, %s2b6ebn, %s2b6pWn, %s2b6pbn, %s2b6lgn, %s2b7dWn, %s2b7dbn, %s2b7ngn, %s2b7nbtn, %s2b7eWn, %s2b7ebn, %s2b7pWn, %s2b7pbn, %s2b7lgn, %s2b8dWn, %s2b8dbn, %s2b8ngn, %s2b8nbtn, %s2b8eWn, %s2b8ebn, %s2b8pWn, %s2b8pbn, %s2b8lgn, %d2ngn, %d2nbtn, %d2Wn, %d2bn, %s3b0dWn, %s3b0dbn, %s3b0ngn, %s3b0nbtn, %s3b0eWn, %s3b0ebn, %s3b0pWn, %s3b0pbn, %s3b0lgn, %s3b1dWn, %s3b1dbn, %s3b1ngn, %s3b1nbtn, %s3b1eWn, %s3b1ebn, %s3b1pWn, %s3b1pbn, %s3b1lgn, %s3b2dWn, %s3b2dbn, %s3b2ngn, %s3b2nbtn, %s3b2eWn, %s3b2ebn, %s3b2pWn, %s3b2pbn, %s3b2lgn, %hngn, %hnbtn, %Wdn, %bdn : tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>
  }
}
