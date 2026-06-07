module @m {
  func.func @convnext_fwd(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<f32>, %s0b0nbt: tensor<f32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<f32>, %s0b1nbt: tensor<f32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<f32>, %s0b2nbt: tensor<f32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<f32>, %d0nbt: tensor<f32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<f32>, %s1b0nbt: tensor<f32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<f32>, %s1b1nbt: tensor<f32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<f32>, %s1b2nbt: tensor<f32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<f32>, %d1nbt: tensor<f32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<f32>, %s2b0nbt: tensor<f32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<f32>, %s2b1nbt: tensor<f32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<f32>, %s2b2nbt: tensor<f32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<f32>, %s2b3nbt: tensor<f32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<f32>, %s2b4nbt: tensor<f32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<f32>, %s2b5nbt: tensor<f32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<f32>, %s2b6nbt: tensor<f32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<f32>, %s2b7nbt: tensor<f32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<f32>, %s2b8nbt: tensor<f32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<f32>, %d2nbt: tensor<f32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<f32>, %s3b0nbt: tensor<f32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<f32>, %s3b1nbt: tensor<f32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<f32>, %s3b2nbt: tensor<f32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<f32>, %hnbt: tensor<f32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>) -> tensor<32x10xf32> {
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
    %out = stablehlo.add %ld, %ldb : tensor<32x10xf32>
    return %out : tensor<32x10xf32>
  }
}
