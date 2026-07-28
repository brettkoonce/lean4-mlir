module @m {
  func.func @convnext_train_step(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<f32>, %s0b0nbt: tensor<f32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<f32>, %s0b1nbt: tensor<f32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<f32>, %s0b2nbt: tensor<f32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<f32>, %d0nbt: tensor<f32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<f32>, %s1b0nbt: tensor<f32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<f32>, %s1b1nbt: tensor<f32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<f32>, %s1b2nbt: tensor<f32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<f32>, %d1nbt: tensor<f32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<f32>, %s2b0nbt: tensor<f32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<f32>, %s2b1nbt: tensor<f32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<f32>, %s2b2nbt: tensor<f32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<f32>, %s2b3nbt: tensor<f32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<f32>, %s2b4nbt: tensor<f32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<f32>, %s2b5nbt: tensor<f32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<f32>, %s2b6nbt: tensor<f32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<f32>, %s2b7nbt: tensor<f32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<f32>, %s2b8nbt: tensor<f32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<f32>, %d2nbt: tensor<f32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<f32>, %s3b0nbt: tensor<f32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<f32>, %s3b1nbt: tensor<f32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<f32>, %s3b2nbt: tensor<f32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<f32>, %hnbt: tensor<f32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %bsc = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %psW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<96x3x4x4xf32>) -> tensor<32x96x56x56xf32>
    %v2 = stablehlo.broadcast_in_dim %psb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x96x56x56xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6 = stablehlo.convolution(%v5, %s0b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v7 = stablehlo.broadcast_in_dim %s0b0db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v8 = stablehlo.add %v6, %v7 : tensor<32x96x56x56xf32>
    %v9 = stablehlo.reshape %v8 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v10 = stablehlo.constant dense<0.0> : tensor<f32>
    %v11 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v12 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v13 = stablehlo.reduce(%v9 init: %v10) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v14 = stablehlo.broadcast_in_dim %v13, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v15 = stablehlo.divide %v14, %v11 : tensor<32x301056xf32>
    %v16 = stablehlo.subtract %v9, %v15 : tensor<32x301056xf32>
    %v17 = stablehlo.multiply %v16, %v16 : tensor<32x301056xf32>
    %v18 = stablehlo.reduce(%v17 init: %v10) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v19 = stablehlo.broadcast_in_dim %v18, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v20 = stablehlo.divide %v19, %v11 : tensor<32x301056xf32>
    %v21 = stablehlo.add %v20, %v12 : tensor<32x301056xf32>
    %v22 = stablehlo.rsqrt %v21 : tensor<32x301056xf32>
    %v23 = stablehlo.multiply %v16, %v22 : tensor<32x301056xf32>
    %v24 = stablehlo.broadcast_in_dim %s0b0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v25 = stablehlo.broadcast_in_dim %s0b0nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v26 = stablehlo.multiply %v23, %v24 : tensor<32x301056xf32>
    %v27 = stablehlo.add %v26, %v25 : tensor<32x301056xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v29 = stablehlo.convolution(%v28, %s0b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v30 = stablehlo.broadcast_in_dim %s0b0eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v31 = stablehlo.add %v29, %v30 : tensor<32x384x56x56xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v33 = stablehlo.multiply %v32, %v32 : tensor<32x1204224xf32>
    %v34 = stablehlo.multiply %v33, %v32 : tensor<32x1204224xf32>
    %v35 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v36 = stablehlo.multiply %v35, %v34 : tensor<32x1204224xf32>
    %v37 = stablehlo.add %v32, %v36 : tensor<32x1204224xf32>
    %v38 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v39 = stablehlo.multiply %v38, %v37 : tensor<32x1204224xf32>
    %v40 = stablehlo.tanh %v39 : tensor<32x1204224xf32>
    %v41 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v42 = stablehlo.add %v41, %v40 : tensor<32x1204224xf32>
    %v43 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v44 = stablehlo.multiply %v43, %v32 : tensor<32x1204224xf32>
    %v45 = stablehlo.multiply %v44, %v42 : tensor<32x1204224xf32>
    %v46 = stablehlo.reshape %v45 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v47 = stablehlo.convolution(%v46, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v48 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v49 = stablehlo.add %v47, %v48 : tensor<32x96x56x56xf32>
    %v50 = stablehlo.reshape %v49 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v52 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v53 = stablehlo.multiply %v51, %v52 : tensor<32x96x56x56xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v55 = stablehlo.add %v54, %v4 : tensor<32x301056xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v57 = stablehlo.convolution(%v56, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v58 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v59 = stablehlo.add %v57, %v58 : tensor<32x96x56x56xf32>
    %v60 = stablehlo.reshape %v59 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v61 = stablehlo.constant dense<0.0> : tensor<f32>
    %v62 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v63 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v64 = stablehlo.reduce(%v60 init: %v61) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v65 = stablehlo.broadcast_in_dim %v64, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v66 = stablehlo.divide %v65, %v62 : tensor<32x301056xf32>
    %v67 = stablehlo.subtract %v60, %v66 : tensor<32x301056xf32>
    %v68 = stablehlo.multiply %v67, %v67 : tensor<32x301056xf32>
    %v69 = stablehlo.reduce(%v68 init: %v61) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v70 = stablehlo.broadcast_in_dim %v69, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v71 = stablehlo.divide %v70, %v62 : tensor<32x301056xf32>
    %v72 = stablehlo.add %v71, %v63 : tensor<32x301056xf32>
    %v73 = stablehlo.rsqrt %v72 : tensor<32x301056xf32>
    %v74 = stablehlo.multiply %v67, %v73 : tensor<32x301056xf32>
    %v75 = stablehlo.broadcast_in_dim %s0b1ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v76 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v77 = stablehlo.multiply %v74, %v75 : tensor<32x301056xf32>
    %v78 = stablehlo.add %v77, %v76 : tensor<32x301056xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v80 = stablehlo.convolution(%v79, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v81 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v82 = stablehlo.add %v80, %v81 : tensor<32x384x56x56xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v84 = stablehlo.multiply %v83, %v83 : tensor<32x1204224xf32>
    %v85 = stablehlo.multiply %v84, %v83 : tensor<32x1204224xf32>
    %v86 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v87 = stablehlo.multiply %v86, %v85 : tensor<32x1204224xf32>
    %v88 = stablehlo.add %v83, %v87 : tensor<32x1204224xf32>
    %v89 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v90 = stablehlo.multiply %v89, %v88 : tensor<32x1204224xf32>
    %v91 = stablehlo.tanh %v90 : tensor<32x1204224xf32>
    %v92 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v93 = stablehlo.add %v92, %v91 : tensor<32x1204224xf32>
    %v94 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v95 = stablehlo.multiply %v94, %v83 : tensor<32x1204224xf32>
    %v96 = stablehlo.multiply %v95, %v93 : tensor<32x1204224xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v98 = stablehlo.convolution(%v97, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v99 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v100 = stablehlo.add %v98, %v99 : tensor<32x96x56x56xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v103 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v104 = stablehlo.multiply %v102, %v103 : tensor<32x96x56x56xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v106 = stablehlo.add %v105, %v55 : tensor<32x301056xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v108 = stablehlo.convolution(%v107, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v109 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v110 = stablehlo.add %v108, %v109 : tensor<32x96x56x56xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v113 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v114 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v115 = stablehlo.reduce(%v111 init: %v112) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v116 = stablehlo.broadcast_in_dim %v115, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v117 = stablehlo.divide %v116, %v113 : tensor<32x301056xf32>
    %v118 = stablehlo.subtract %v111, %v117 : tensor<32x301056xf32>
    %v119 = stablehlo.multiply %v118, %v118 : tensor<32x301056xf32>
    %v120 = stablehlo.reduce(%v119 init: %v112) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v121 = stablehlo.broadcast_in_dim %v120, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v122 = stablehlo.divide %v121, %v113 : tensor<32x301056xf32>
    %v123 = stablehlo.add %v122, %v114 : tensor<32x301056xf32>
    %v124 = stablehlo.rsqrt %v123 : tensor<32x301056xf32>
    %v125 = stablehlo.multiply %v118, %v124 : tensor<32x301056xf32>
    %v126 = stablehlo.broadcast_in_dim %s0b2ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v127 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v128 = stablehlo.multiply %v125, %v126 : tensor<32x301056xf32>
    %v129 = stablehlo.add %v128, %v127 : tensor<32x301056xf32>
    %v130 = stablehlo.reshape %v129 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v131 = stablehlo.convolution(%v130, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v132 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v133 = stablehlo.add %v131, %v132 : tensor<32x384x56x56xf32>
    %v134 = stablehlo.reshape %v133 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v135 = stablehlo.multiply %v134, %v134 : tensor<32x1204224xf32>
    %v136 = stablehlo.multiply %v135, %v134 : tensor<32x1204224xf32>
    %v137 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v138 = stablehlo.multiply %v137, %v136 : tensor<32x1204224xf32>
    %v139 = stablehlo.add %v134, %v138 : tensor<32x1204224xf32>
    %v140 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v141 = stablehlo.multiply %v140, %v139 : tensor<32x1204224xf32>
    %v142 = stablehlo.tanh %v141 : tensor<32x1204224xf32>
    %v143 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v144 = stablehlo.add %v143, %v142 : tensor<32x1204224xf32>
    %v145 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v146 = stablehlo.multiply %v145, %v134 : tensor<32x1204224xf32>
    %v147 = stablehlo.multiply %v146, %v144 : tensor<32x1204224xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v149 = stablehlo.convolution(%v148, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v150 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v151 = stablehlo.add %v149, %v150 : tensor<32x96x56x56xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v154 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v155 = stablehlo.multiply %v153, %v154 : tensor<32x96x56x56xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v157 = stablehlo.add %v156, %v106 : tensor<32x301056xf32>
    %v158 = stablehlo.constant dense<0.0> : tensor<f32>
    %v159 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v160 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v161 = stablehlo.reduce(%v157 init: %v158) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v162 = stablehlo.broadcast_in_dim %v161, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v163 = stablehlo.divide %v162, %v159 : tensor<32x301056xf32>
    %v164 = stablehlo.subtract %v157, %v163 : tensor<32x301056xf32>
    %v165 = stablehlo.multiply %v164, %v164 : tensor<32x301056xf32>
    %v166 = stablehlo.reduce(%v165 init: %v158) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v167 = stablehlo.broadcast_in_dim %v166, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v168 = stablehlo.divide %v167, %v159 : tensor<32x301056xf32>
    %v169 = stablehlo.add %v168, %v160 : tensor<32x301056xf32>
    %v170 = stablehlo.rsqrt %v169 : tensor<32x301056xf32>
    %v171 = stablehlo.multiply %v164, %v170 : tensor<32x301056xf32>
    %v172 = stablehlo.broadcast_in_dim %d0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v173 = stablehlo.broadcast_in_dim %d0nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v174 = stablehlo.multiply %v171, %v172 : tensor<32x301056xf32>
    %v175 = stablehlo.add %v174, %v173 : tensor<32x301056xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v177 = stablehlo.convolution(%v176, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v178 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v179 = stablehlo.add %v177, %v178 : tensor<32x192x28x28xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v182 = stablehlo.convolution(%v181, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v183 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v184 = stablehlo.add %v182, %v183 : tensor<32x192x28x28xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v187 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v188 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v189 = stablehlo.reduce(%v185 init: %v186) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v190 = stablehlo.broadcast_in_dim %v189, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v191 = stablehlo.divide %v190, %v187 : tensor<32x150528xf32>
    %v192 = stablehlo.subtract %v185, %v191 : tensor<32x150528xf32>
    %v193 = stablehlo.multiply %v192, %v192 : tensor<32x150528xf32>
    %v194 = stablehlo.reduce(%v193 init: %v186) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v195 = stablehlo.broadcast_in_dim %v194, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v196 = stablehlo.divide %v195, %v187 : tensor<32x150528xf32>
    %v197 = stablehlo.add %v196, %v188 : tensor<32x150528xf32>
    %v198 = stablehlo.rsqrt %v197 : tensor<32x150528xf32>
    %v199 = stablehlo.multiply %v192, %v198 : tensor<32x150528xf32>
    %v200 = stablehlo.broadcast_in_dim %s1b0ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v201 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v202 = stablehlo.multiply %v199, %v200 : tensor<32x150528xf32>
    %v203 = stablehlo.add %v202, %v201 : tensor<32x150528xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v205 = stablehlo.convolution(%v204, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v206 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v207 = stablehlo.add %v205, %v206 : tensor<32x768x28x28xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v209 = stablehlo.multiply %v208, %v208 : tensor<32x602112xf32>
    %v210 = stablehlo.multiply %v209, %v208 : tensor<32x602112xf32>
    %v211 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v212 = stablehlo.multiply %v211, %v210 : tensor<32x602112xf32>
    %v213 = stablehlo.add %v208, %v212 : tensor<32x602112xf32>
    %v214 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v215 = stablehlo.multiply %v214, %v213 : tensor<32x602112xf32>
    %v216 = stablehlo.tanh %v215 : tensor<32x602112xf32>
    %v217 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v218 = stablehlo.add %v217, %v216 : tensor<32x602112xf32>
    %v219 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v220 = stablehlo.multiply %v219, %v208 : tensor<32x602112xf32>
    %v221 = stablehlo.multiply %v220, %v218 : tensor<32x602112xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v223 = stablehlo.convolution(%v222, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v224 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v225 = stablehlo.add %v223, %v224 : tensor<32x192x28x28xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v228 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v229 = stablehlo.multiply %v227, %v228 : tensor<32x192x28x28xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v231 = stablehlo.add %v230, %v180 : tensor<32x150528xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v233 = stablehlo.convolution(%v232, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v234 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v235 = stablehlo.add %v233, %v234 : tensor<32x192x28x28xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v238 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v239 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v240 = stablehlo.reduce(%v236 init: %v237) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v241 = stablehlo.broadcast_in_dim %v240, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v242 = stablehlo.divide %v241, %v238 : tensor<32x150528xf32>
    %v243 = stablehlo.subtract %v236, %v242 : tensor<32x150528xf32>
    %v244 = stablehlo.multiply %v243, %v243 : tensor<32x150528xf32>
    %v245 = stablehlo.reduce(%v244 init: %v237) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v246 = stablehlo.broadcast_in_dim %v245, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v247 = stablehlo.divide %v246, %v238 : tensor<32x150528xf32>
    %v248 = stablehlo.add %v247, %v239 : tensor<32x150528xf32>
    %v249 = stablehlo.rsqrt %v248 : tensor<32x150528xf32>
    %v250 = stablehlo.multiply %v243, %v249 : tensor<32x150528xf32>
    %v251 = stablehlo.broadcast_in_dim %s1b1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v252 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v253 = stablehlo.multiply %v250, %v251 : tensor<32x150528xf32>
    %v254 = stablehlo.add %v253, %v252 : tensor<32x150528xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v256 = stablehlo.convolution(%v255, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v257 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v258 = stablehlo.add %v256, %v257 : tensor<32x768x28x28xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v260 = stablehlo.multiply %v259, %v259 : tensor<32x602112xf32>
    %v261 = stablehlo.multiply %v260, %v259 : tensor<32x602112xf32>
    %v262 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v263 = stablehlo.multiply %v262, %v261 : tensor<32x602112xf32>
    %v264 = stablehlo.add %v259, %v263 : tensor<32x602112xf32>
    %v265 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v266 = stablehlo.multiply %v265, %v264 : tensor<32x602112xf32>
    %v267 = stablehlo.tanh %v266 : tensor<32x602112xf32>
    %v268 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v269 = stablehlo.add %v268, %v267 : tensor<32x602112xf32>
    %v270 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v271 = stablehlo.multiply %v270, %v259 : tensor<32x602112xf32>
    %v272 = stablehlo.multiply %v271, %v269 : tensor<32x602112xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v274 = stablehlo.convolution(%v273, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v275 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v276 = stablehlo.add %v274, %v275 : tensor<32x192x28x28xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v279 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v280 = stablehlo.multiply %v278, %v279 : tensor<32x192x28x28xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v282 = stablehlo.add %v281, %v231 : tensor<32x150528xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v284 = stablehlo.convolution(%v283, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v285 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v286 = stablehlo.add %v284, %v285 : tensor<32x192x28x28xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v289 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v290 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v291 = stablehlo.reduce(%v287 init: %v288) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v292 = stablehlo.broadcast_in_dim %v291, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v293 = stablehlo.divide %v292, %v289 : tensor<32x150528xf32>
    %v294 = stablehlo.subtract %v287, %v293 : tensor<32x150528xf32>
    %v295 = stablehlo.multiply %v294, %v294 : tensor<32x150528xf32>
    %v296 = stablehlo.reduce(%v295 init: %v288) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v297 = stablehlo.broadcast_in_dim %v296, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v298 = stablehlo.divide %v297, %v289 : tensor<32x150528xf32>
    %v299 = stablehlo.add %v298, %v290 : tensor<32x150528xf32>
    %v300 = stablehlo.rsqrt %v299 : tensor<32x150528xf32>
    %v301 = stablehlo.multiply %v294, %v300 : tensor<32x150528xf32>
    %v302 = stablehlo.broadcast_in_dim %s1b2ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v303 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v304 = stablehlo.multiply %v301, %v302 : tensor<32x150528xf32>
    %v305 = stablehlo.add %v304, %v303 : tensor<32x150528xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v307 = stablehlo.convolution(%v306, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<32x768x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v311 = stablehlo.multiply %v310, %v310 : tensor<32x602112xf32>
    %v312 = stablehlo.multiply %v311, %v310 : tensor<32x602112xf32>
    %v313 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v314 = stablehlo.multiply %v313, %v312 : tensor<32x602112xf32>
    %v315 = stablehlo.add %v310, %v314 : tensor<32x602112xf32>
    %v316 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v317 = stablehlo.multiply %v316, %v315 : tensor<32x602112xf32>
    %v318 = stablehlo.tanh %v317 : tensor<32x602112xf32>
    %v319 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v320 = stablehlo.add %v319, %v318 : tensor<32x602112xf32>
    %v321 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v322 = stablehlo.multiply %v321, %v310 : tensor<32x602112xf32>
    %v323 = stablehlo.multiply %v322, %v320 : tensor<32x602112xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v325 = stablehlo.convolution(%v324, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v327 = stablehlo.add %v325, %v326 : tensor<32x192x28x28xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v330 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v331 = stablehlo.multiply %v329, %v330 : tensor<32x192x28x28xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v333 = stablehlo.add %v332, %v282 : tensor<32x150528xf32>
    %v334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v335 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v336 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v337 = stablehlo.reduce(%v333 init: %v334) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v338 = stablehlo.broadcast_in_dim %v337, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v339 = stablehlo.divide %v338, %v335 : tensor<32x150528xf32>
    %v340 = stablehlo.subtract %v333, %v339 : tensor<32x150528xf32>
    %v341 = stablehlo.multiply %v340, %v340 : tensor<32x150528xf32>
    %v342 = stablehlo.reduce(%v341 init: %v334) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v343 = stablehlo.broadcast_in_dim %v342, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v344 = stablehlo.divide %v343, %v335 : tensor<32x150528xf32>
    %v345 = stablehlo.add %v344, %v336 : tensor<32x150528xf32>
    %v346 = stablehlo.rsqrt %v345 : tensor<32x150528xf32>
    %v347 = stablehlo.multiply %v340, %v346 : tensor<32x150528xf32>
    %v348 = stablehlo.broadcast_in_dim %d1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v349 = stablehlo.broadcast_in_dim %d1nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v350 = stablehlo.multiply %v347, %v348 : tensor<32x150528xf32>
    %v351 = stablehlo.add %v350, %v349 : tensor<32x150528xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v353 = stablehlo.convolution(%v352, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v354 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v355 = stablehlo.add %v353, %v354 : tensor<32x384x14x14xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v358 = stablehlo.convolution(%v357, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v359 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v360 = stablehlo.add %v358, %v359 : tensor<32x384x14x14xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v363 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v364 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v365 = stablehlo.reduce(%v361 init: %v362) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v366 = stablehlo.broadcast_in_dim %v365, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v367 = stablehlo.divide %v366, %v363 : tensor<32x75264xf32>
    %v368 = stablehlo.subtract %v361, %v367 : tensor<32x75264xf32>
    %v369 = stablehlo.multiply %v368, %v368 : tensor<32x75264xf32>
    %v370 = stablehlo.reduce(%v369 init: %v362) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v371 = stablehlo.broadcast_in_dim %v370, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v372 = stablehlo.divide %v371, %v363 : tensor<32x75264xf32>
    %v373 = stablehlo.add %v372, %v364 : tensor<32x75264xf32>
    %v374 = stablehlo.rsqrt %v373 : tensor<32x75264xf32>
    %v375 = stablehlo.multiply %v368, %v374 : tensor<32x75264xf32>
    %v376 = stablehlo.broadcast_in_dim %s2b0ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v377 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v378 = stablehlo.multiply %v375, %v376 : tensor<32x75264xf32>
    %v379 = stablehlo.add %v378, %v377 : tensor<32x75264xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v381 = stablehlo.convolution(%v380, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v382 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v383 = stablehlo.add %v381, %v382 : tensor<32x1536x14x14xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v385 = stablehlo.multiply %v384, %v384 : tensor<32x301056xf32>
    %v386 = stablehlo.multiply %v385, %v384 : tensor<32x301056xf32>
    %v387 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v388 = stablehlo.multiply %v387, %v386 : tensor<32x301056xf32>
    %v389 = stablehlo.add %v384, %v388 : tensor<32x301056xf32>
    %v390 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v391 = stablehlo.multiply %v390, %v389 : tensor<32x301056xf32>
    %v392 = stablehlo.tanh %v391 : tensor<32x301056xf32>
    %v393 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v394 = stablehlo.add %v393, %v392 : tensor<32x301056xf32>
    %v395 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v396 = stablehlo.multiply %v395, %v384 : tensor<32x301056xf32>
    %v397 = stablehlo.multiply %v396, %v394 : tensor<32x301056xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v399 = stablehlo.convolution(%v398, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v400 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v401 = stablehlo.add %v399, %v400 : tensor<32x384x14x14xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v404 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v405 = stablehlo.multiply %v403, %v404 : tensor<32x384x14x14xf32>
    %v406 = stablehlo.reshape %v405 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v407 = stablehlo.add %v406, %v356 : tensor<32x75264xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v409 = stablehlo.convolution(%v408, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v410 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v411 = stablehlo.add %v409, %v410 : tensor<32x384x14x14xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v414 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v415 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v416 = stablehlo.reduce(%v412 init: %v413) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v417 = stablehlo.broadcast_in_dim %v416, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v418 = stablehlo.divide %v417, %v414 : tensor<32x75264xf32>
    %v419 = stablehlo.subtract %v412, %v418 : tensor<32x75264xf32>
    %v420 = stablehlo.multiply %v419, %v419 : tensor<32x75264xf32>
    %v421 = stablehlo.reduce(%v420 init: %v413) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v422 = stablehlo.broadcast_in_dim %v421, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v423 = stablehlo.divide %v422, %v414 : tensor<32x75264xf32>
    %v424 = stablehlo.add %v423, %v415 : tensor<32x75264xf32>
    %v425 = stablehlo.rsqrt %v424 : tensor<32x75264xf32>
    %v426 = stablehlo.multiply %v419, %v425 : tensor<32x75264xf32>
    %v427 = stablehlo.broadcast_in_dim %s2b1ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v428 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v429 = stablehlo.multiply %v426, %v427 : tensor<32x75264xf32>
    %v430 = stablehlo.add %v429, %v428 : tensor<32x75264xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v432 = stablehlo.convolution(%v431, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v433 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v434 = stablehlo.add %v432, %v433 : tensor<32x1536x14x14xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v436 = stablehlo.multiply %v435, %v435 : tensor<32x301056xf32>
    %v437 = stablehlo.multiply %v436, %v435 : tensor<32x301056xf32>
    %v438 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v439 = stablehlo.multiply %v438, %v437 : tensor<32x301056xf32>
    %v440 = stablehlo.add %v435, %v439 : tensor<32x301056xf32>
    %v441 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v442 = stablehlo.multiply %v441, %v440 : tensor<32x301056xf32>
    %v443 = stablehlo.tanh %v442 : tensor<32x301056xf32>
    %v444 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v445 = stablehlo.add %v444, %v443 : tensor<32x301056xf32>
    %v446 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v447 = stablehlo.multiply %v446, %v435 : tensor<32x301056xf32>
    %v448 = stablehlo.multiply %v447, %v445 : tensor<32x301056xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v450 = stablehlo.convolution(%v449, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v451 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v452 = stablehlo.add %v450, %v451 : tensor<32x384x14x14xf32>
    %v453 = stablehlo.reshape %v452 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v455 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v456 = stablehlo.multiply %v454, %v455 : tensor<32x384x14x14xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v458 = stablehlo.add %v457, %v407 : tensor<32x75264xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v460 = stablehlo.convolution(%v459, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v461 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<32x384x14x14xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v464 = stablehlo.constant dense<0.0> : tensor<f32>
    %v465 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v466 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v467 = stablehlo.reduce(%v463 init: %v464) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v468 = stablehlo.broadcast_in_dim %v467, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v469 = stablehlo.divide %v468, %v465 : tensor<32x75264xf32>
    %v470 = stablehlo.subtract %v463, %v469 : tensor<32x75264xf32>
    %v471 = stablehlo.multiply %v470, %v470 : tensor<32x75264xf32>
    %v472 = stablehlo.reduce(%v471 init: %v464) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v473 = stablehlo.broadcast_in_dim %v472, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v474 = stablehlo.divide %v473, %v465 : tensor<32x75264xf32>
    %v475 = stablehlo.add %v474, %v466 : tensor<32x75264xf32>
    %v476 = stablehlo.rsqrt %v475 : tensor<32x75264xf32>
    %v477 = stablehlo.multiply %v470, %v476 : tensor<32x75264xf32>
    %v478 = stablehlo.broadcast_in_dim %s2b2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v479 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v480 = stablehlo.multiply %v477, %v478 : tensor<32x75264xf32>
    %v481 = stablehlo.add %v480, %v479 : tensor<32x75264xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v483 = stablehlo.convolution(%v482, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v484 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v485 = stablehlo.add %v483, %v484 : tensor<32x1536x14x14xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v487 = stablehlo.multiply %v486, %v486 : tensor<32x301056xf32>
    %v488 = stablehlo.multiply %v487, %v486 : tensor<32x301056xf32>
    %v489 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v490 = stablehlo.multiply %v489, %v488 : tensor<32x301056xf32>
    %v491 = stablehlo.add %v486, %v490 : tensor<32x301056xf32>
    %v492 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v493 = stablehlo.multiply %v492, %v491 : tensor<32x301056xf32>
    %v494 = stablehlo.tanh %v493 : tensor<32x301056xf32>
    %v495 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v496 = stablehlo.add %v495, %v494 : tensor<32x301056xf32>
    %v497 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v498 = stablehlo.multiply %v497, %v486 : tensor<32x301056xf32>
    %v499 = stablehlo.multiply %v498, %v496 : tensor<32x301056xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v501 = stablehlo.convolution(%v500, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v502 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v503 = stablehlo.add %v501, %v502 : tensor<32x384x14x14xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v506 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v507 = stablehlo.multiply %v505, %v506 : tensor<32x384x14x14xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v509 = stablehlo.add %v508, %v458 : tensor<32x75264xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v511 = stablehlo.convolution(%v510, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v513 = stablehlo.add %v511, %v512 : tensor<32x384x14x14xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v516 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v517 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v518 = stablehlo.reduce(%v514 init: %v515) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v519 = stablehlo.broadcast_in_dim %v518, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v520 = stablehlo.divide %v519, %v516 : tensor<32x75264xf32>
    %v521 = stablehlo.subtract %v514, %v520 : tensor<32x75264xf32>
    %v522 = stablehlo.multiply %v521, %v521 : tensor<32x75264xf32>
    %v523 = stablehlo.reduce(%v522 init: %v515) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v524 = stablehlo.broadcast_in_dim %v523, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v525 = stablehlo.divide %v524, %v516 : tensor<32x75264xf32>
    %v526 = stablehlo.add %v525, %v517 : tensor<32x75264xf32>
    %v527 = stablehlo.rsqrt %v526 : tensor<32x75264xf32>
    %v528 = stablehlo.multiply %v521, %v527 : tensor<32x75264xf32>
    %v529 = stablehlo.broadcast_in_dim %s2b3ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v530 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v531 = stablehlo.multiply %v528, %v529 : tensor<32x75264xf32>
    %v532 = stablehlo.add %v531, %v530 : tensor<32x75264xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v534 = stablehlo.convolution(%v533, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v535 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v536 = stablehlo.add %v534, %v535 : tensor<32x1536x14x14xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v538 = stablehlo.multiply %v537, %v537 : tensor<32x301056xf32>
    %v539 = stablehlo.multiply %v538, %v537 : tensor<32x301056xf32>
    %v540 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v541 = stablehlo.multiply %v540, %v539 : tensor<32x301056xf32>
    %v542 = stablehlo.add %v537, %v541 : tensor<32x301056xf32>
    %v543 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v544 = stablehlo.multiply %v543, %v542 : tensor<32x301056xf32>
    %v545 = stablehlo.tanh %v544 : tensor<32x301056xf32>
    %v546 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v547 = stablehlo.add %v546, %v545 : tensor<32x301056xf32>
    %v548 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v549 = stablehlo.multiply %v548, %v537 : tensor<32x301056xf32>
    %v550 = stablehlo.multiply %v549, %v547 : tensor<32x301056xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v552 = stablehlo.convolution(%v551, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v553 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v554 = stablehlo.add %v552, %v553 : tensor<32x384x14x14xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v557 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v558 = stablehlo.multiply %v556, %v557 : tensor<32x384x14x14xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v560 = stablehlo.add %v559, %v509 : tensor<32x75264xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v562 = stablehlo.convolution(%v561, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v563 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v564 = stablehlo.add %v562, %v563 : tensor<32x384x14x14xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v567 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v568 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v569 = stablehlo.reduce(%v565 init: %v566) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v570 = stablehlo.broadcast_in_dim %v569, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v571 = stablehlo.divide %v570, %v567 : tensor<32x75264xf32>
    %v572 = stablehlo.subtract %v565, %v571 : tensor<32x75264xf32>
    %v573 = stablehlo.multiply %v572, %v572 : tensor<32x75264xf32>
    %v574 = stablehlo.reduce(%v573 init: %v566) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v575 = stablehlo.broadcast_in_dim %v574, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v576 = stablehlo.divide %v575, %v567 : tensor<32x75264xf32>
    %v577 = stablehlo.add %v576, %v568 : tensor<32x75264xf32>
    %v578 = stablehlo.rsqrt %v577 : tensor<32x75264xf32>
    %v579 = stablehlo.multiply %v572, %v578 : tensor<32x75264xf32>
    %v580 = stablehlo.broadcast_in_dim %s2b4ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v581 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v582 = stablehlo.multiply %v579, %v580 : tensor<32x75264xf32>
    %v583 = stablehlo.add %v582, %v581 : tensor<32x75264xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v585 = stablehlo.convolution(%v584, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v586 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v587 = stablehlo.add %v585, %v586 : tensor<32x1536x14x14xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v589 = stablehlo.multiply %v588, %v588 : tensor<32x301056xf32>
    %v590 = stablehlo.multiply %v589, %v588 : tensor<32x301056xf32>
    %v591 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v592 = stablehlo.multiply %v591, %v590 : tensor<32x301056xf32>
    %v593 = stablehlo.add %v588, %v592 : tensor<32x301056xf32>
    %v594 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v595 = stablehlo.multiply %v594, %v593 : tensor<32x301056xf32>
    %v596 = stablehlo.tanh %v595 : tensor<32x301056xf32>
    %v597 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v598 = stablehlo.add %v597, %v596 : tensor<32x301056xf32>
    %v599 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v600 = stablehlo.multiply %v599, %v588 : tensor<32x301056xf32>
    %v601 = stablehlo.multiply %v600, %v598 : tensor<32x301056xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v603 = stablehlo.convolution(%v602, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v604 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v605 = stablehlo.add %v603, %v604 : tensor<32x384x14x14xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v608 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v609 = stablehlo.multiply %v607, %v608 : tensor<32x384x14x14xf32>
    %v610 = stablehlo.reshape %v609 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v611 = stablehlo.add %v610, %v560 : tensor<32x75264xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v613 = stablehlo.convolution(%v612, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v615 = stablehlo.add %v613, %v614 : tensor<32x384x14x14xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v618 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v619 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v620 = stablehlo.reduce(%v616 init: %v617) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v621 = stablehlo.broadcast_in_dim %v620, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v622 = stablehlo.divide %v621, %v618 : tensor<32x75264xf32>
    %v623 = stablehlo.subtract %v616, %v622 : tensor<32x75264xf32>
    %v624 = stablehlo.multiply %v623, %v623 : tensor<32x75264xf32>
    %v625 = stablehlo.reduce(%v624 init: %v617) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v626 = stablehlo.broadcast_in_dim %v625, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v627 = stablehlo.divide %v626, %v618 : tensor<32x75264xf32>
    %v628 = stablehlo.add %v627, %v619 : tensor<32x75264xf32>
    %v629 = stablehlo.rsqrt %v628 : tensor<32x75264xf32>
    %v630 = stablehlo.multiply %v623, %v629 : tensor<32x75264xf32>
    %v631 = stablehlo.broadcast_in_dim %s2b5ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v632 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v633 = stablehlo.multiply %v630, %v631 : tensor<32x75264xf32>
    %v634 = stablehlo.add %v633, %v632 : tensor<32x75264xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v636 = stablehlo.convolution(%v635, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v637 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v638 = stablehlo.add %v636, %v637 : tensor<32x1536x14x14xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v640 = stablehlo.multiply %v639, %v639 : tensor<32x301056xf32>
    %v641 = stablehlo.multiply %v640, %v639 : tensor<32x301056xf32>
    %v642 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v643 = stablehlo.multiply %v642, %v641 : tensor<32x301056xf32>
    %v644 = stablehlo.add %v639, %v643 : tensor<32x301056xf32>
    %v645 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v646 = stablehlo.multiply %v645, %v644 : tensor<32x301056xf32>
    %v647 = stablehlo.tanh %v646 : tensor<32x301056xf32>
    %v648 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v649 = stablehlo.add %v648, %v647 : tensor<32x301056xf32>
    %v650 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v651 = stablehlo.multiply %v650, %v639 : tensor<32x301056xf32>
    %v652 = stablehlo.multiply %v651, %v649 : tensor<32x301056xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v654 = stablehlo.convolution(%v653, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v655 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v656 = stablehlo.add %v654, %v655 : tensor<32x384x14x14xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v659 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v660 = stablehlo.multiply %v658, %v659 : tensor<32x384x14x14xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v662 = stablehlo.add %v661, %v611 : tensor<32x75264xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v664 = stablehlo.convolution(%v663, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v665 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v666 = stablehlo.add %v664, %v665 : tensor<32x384x14x14xf32>
    %v667 = stablehlo.reshape %v666 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v668 = stablehlo.constant dense<0.0> : tensor<f32>
    %v669 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v670 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v671 = stablehlo.reduce(%v667 init: %v668) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v672 = stablehlo.broadcast_in_dim %v671, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v673 = stablehlo.divide %v672, %v669 : tensor<32x75264xf32>
    %v674 = stablehlo.subtract %v667, %v673 : tensor<32x75264xf32>
    %v675 = stablehlo.multiply %v674, %v674 : tensor<32x75264xf32>
    %v676 = stablehlo.reduce(%v675 init: %v668) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v677 = stablehlo.broadcast_in_dim %v676, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v678 = stablehlo.divide %v677, %v669 : tensor<32x75264xf32>
    %v679 = stablehlo.add %v678, %v670 : tensor<32x75264xf32>
    %v680 = stablehlo.rsqrt %v679 : tensor<32x75264xf32>
    %v681 = stablehlo.multiply %v674, %v680 : tensor<32x75264xf32>
    %v682 = stablehlo.broadcast_in_dim %s2b6ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v683 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v684 = stablehlo.multiply %v681, %v682 : tensor<32x75264xf32>
    %v685 = stablehlo.add %v684, %v683 : tensor<32x75264xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v687 = stablehlo.convolution(%v686, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<32x1536x14x14xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v691 = stablehlo.multiply %v690, %v690 : tensor<32x301056xf32>
    %v692 = stablehlo.multiply %v691, %v690 : tensor<32x301056xf32>
    %v693 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v694 = stablehlo.multiply %v693, %v692 : tensor<32x301056xf32>
    %v695 = stablehlo.add %v690, %v694 : tensor<32x301056xf32>
    %v696 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v697 = stablehlo.multiply %v696, %v695 : tensor<32x301056xf32>
    %v698 = stablehlo.tanh %v697 : tensor<32x301056xf32>
    %v699 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v700 = stablehlo.add %v699, %v698 : tensor<32x301056xf32>
    %v701 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v702 = stablehlo.multiply %v701, %v690 : tensor<32x301056xf32>
    %v703 = stablehlo.multiply %v702, %v700 : tensor<32x301056xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v705 = stablehlo.convolution(%v704, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v706 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v707 = stablehlo.add %v705, %v706 : tensor<32x384x14x14xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v710 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v711 = stablehlo.multiply %v709, %v710 : tensor<32x384x14x14xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v713 = stablehlo.add %v712, %v662 : tensor<32x75264xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v715 = stablehlo.convolution(%v714, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v716 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v717 = stablehlo.add %v715, %v716 : tensor<32x384x14x14xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v720 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v721 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v722 = stablehlo.reduce(%v718 init: %v719) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v723 = stablehlo.broadcast_in_dim %v722, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v724 = stablehlo.divide %v723, %v720 : tensor<32x75264xf32>
    %v725 = stablehlo.subtract %v718, %v724 : tensor<32x75264xf32>
    %v726 = stablehlo.multiply %v725, %v725 : tensor<32x75264xf32>
    %v727 = stablehlo.reduce(%v726 init: %v719) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v728 = stablehlo.broadcast_in_dim %v727, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v729 = stablehlo.divide %v728, %v720 : tensor<32x75264xf32>
    %v730 = stablehlo.add %v729, %v721 : tensor<32x75264xf32>
    %v731 = stablehlo.rsqrt %v730 : tensor<32x75264xf32>
    %v732 = stablehlo.multiply %v725, %v731 : tensor<32x75264xf32>
    %v733 = stablehlo.broadcast_in_dim %s2b7ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v734 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v735 = stablehlo.multiply %v732, %v733 : tensor<32x75264xf32>
    %v736 = stablehlo.add %v735, %v734 : tensor<32x75264xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v738 = stablehlo.convolution(%v737, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v739 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v740 = stablehlo.add %v738, %v739 : tensor<32x1536x14x14xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v742 = stablehlo.multiply %v741, %v741 : tensor<32x301056xf32>
    %v743 = stablehlo.multiply %v742, %v741 : tensor<32x301056xf32>
    %v744 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v745 = stablehlo.multiply %v744, %v743 : tensor<32x301056xf32>
    %v746 = stablehlo.add %v741, %v745 : tensor<32x301056xf32>
    %v747 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v748 = stablehlo.multiply %v747, %v746 : tensor<32x301056xf32>
    %v749 = stablehlo.tanh %v748 : tensor<32x301056xf32>
    %v750 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v751 = stablehlo.add %v750, %v749 : tensor<32x301056xf32>
    %v752 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v753 = stablehlo.multiply %v752, %v741 : tensor<32x301056xf32>
    %v754 = stablehlo.multiply %v753, %v751 : tensor<32x301056xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v756 = stablehlo.convolution(%v755, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v757 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v758 = stablehlo.add %v756, %v757 : tensor<32x384x14x14xf32>
    %v759 = stablehlo.reshape %v758 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v762 = stablehlo.multiply %v760, %v761 : tensor<32x384x14x14xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v764 = stablehlo.add %v763, %v713 : tensor<32x75264xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v766 = stablehlo.convolution(%v765, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x384x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v771 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v772 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v773 = stablehlo.reduce(%v769 init: %v770) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v774 = stablehlo.broadcast_in_dim %v773, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v775 = stablehlo.divide %v774, %v771 : tensor<32x75264xf32>
    %v776 = stablehlo.subtract %v769, %v775 : tensor<32x75264xf32>
    %v777 = stablehlo.multiply %v776, %v776 : tensor<32x75264xf32>
    %v778 = stablehlo.reduce(%v777 init: %v770) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v779 = stablehlo.broadcast_in_dim %v778, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v780 = stablehlo.divide %v779, %v771 : tensor<32x75264xf32>
    %v781 = stablehlo.add %v780, %v772 : tensor<32x75264xf32>
    %v782 = stablehlo.rsqrt %v781 : tensor<32x75264xf32>
    %v783 = stablehlo.multiply %v776, %v782 : tensor<32x75264xf32>
    %v784 = stablehlo.broadcast_in_dim %s2b8ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v785 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v786 = stablehlo.multiply %v783, %v784 : tensor<32x75264xf32>
    %v787 = stablehlo.add %v786, %v785 : tensor<32x75264xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v789 = stablehlo.convolution(%v788, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v790 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v791 = stablehlo.add %v789, %v790 : tensor<32x1536x14x14xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v793 = stablehlo.multiply %v792, %v792 : tensor<32x301056xf32>
    %v794 = stablehlo.multiply %v793, %v792 : tensor<32x301056xf32>
    %v795 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v796 = stablehlo.multiply %v795, %v794 : tensor<32x301056xf32>
    %v797 = stablehlo.add %v792, %v796 : tensor<32x301056xf32>
    %v798 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v799 = stablehlo.multiply %v798, %v797 : tensor<32x301056xf32>
    %v800 = stablehlo.tanh %v799 : tensor<32x301056xf32>
    %v801 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v802 = stablehlo.add %v801, %v800 : tensor<32x301056xf32>
    %v803 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v804 = stablehlo.multiply %v803, %v792 : tensor<32x301056xf32>
    %v805 = stablehlo.multiply %v804, %v802 : tensor<32x301056xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v807 = stablehlo.convolution(%v806, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v808 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v809 = stablehlo.add %v807, %v808 : tensor<32x384x14x14xf32>
    %v810 = stablehlo.reshape %v809 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v812 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v813 = stablehlo.multiply %v811, %v812 : tensor<32x384x14x14xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v815 = stablehlo.add %v814, %v764 : tensor<32x75264xf32>
    %v816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v817 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v818 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v819 = stablehlo.reduce(%v815 init: %v816) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v820 = stablehlo.broadcast_in_dim %v819, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v821 = stablehlo.divide %v820, %v817 : tensor<32x75264xf32>
    %v822 = stablehlo.subtract %v815, %v821 : tensor<32x75264xf32>
    %v823 = stablehlo.multiply %v822, %v822 : tensor<32x75264xf32>
    %v824 = stablehlo.reduce(%v823 init: %v816) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v825 = stablehlo.broadcast_in_dim %v824, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v826 = stablehlo.divide %v825, %v817 : tensor<32x75264xf32>
    %v827 = stablehlo.add %v826, %v818 : tensor<32x75264xf32>
    %v828 = stablehlo.rsqrt %v827 : tensor<32x75264xf32>
    %v829 = stablehlo.multiply %v822, %v828 : tensor<32x75264xf32>
    %v830 = stablehlo.broadcast_in_dim %d2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v831 = stablehlo.broadcast_in_dim %d2nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v832 = stablehlo.multiply %v829, %v830 : tensor<32x75264xf32>
    %v833 = stablehlo.add %v832, %v831 : tensor<32x75264xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v835 = stablehlo.convolution(%v834, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %v836 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v837 = stablehlo.add %v835, %v836 : tensor<32x768x7x7xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v840 = stablehlo.convolution(%v839, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v841 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v842 = stablehlo.add %v840, %v841 : tensor<32x768x7x7xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v844 = stablehlo.constant dense<0.0> : tensor<f32>
    %v845 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v846 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v847 = stablehlo.reduce(%v843 init: %v844) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v848 = stablehlo.broadcast_in_dim %v847, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v849 = stablehlo.divide %v848, %v845 : tensor<32x37632xf32>
    %v850 = stablehlo.subtract %v843, %v849 : tensor<32x37632xf32>
    %v851 = stablehlo.multiply %v850, %v850 : tensor<32x37632xf32>
    %v852 = stablehlo.reduce(%v851 init: %v844) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v853 = stablehlo.broadcast_in_dim %v852, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v854 = stablehlo.divide %v853, %v845 : tensor<32x37632xf32>
    %v855 = stablehlo.add %v854, %v846 : tensor<32x37632xf32>
    %v856 = stablehlo.rsqrt %v855 : tensor<32x37632xf32>
    %v857 = stablehlo.multiply %v850, %v856 : tensor<32x37632xf32>
    %v858 = stablehlo.broadcast_in_dim %s3b0ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v859 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v860 = stablehlo.multiply %v857, %v858 : tensor<32x37632xf32>
    %v861 = stablehlo.add %v860, %v859 : tensor<32x37632xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v863 = stablehlo.convolution(%v862, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v864 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v865 = stablehlo.add %v863, %v864 : tensor<32x3072x7x7xf32>
    %v866 = stablehlo.reshape %v865 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v867 = stablehlo.multiply %v866, %v866 : tensor<32x150528xf32>
    %v868 = stablehlo.multiply %v867, %v866 : tensor<32x150528xf32>
    %v869 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v870 = stablehlo.multiply %v869, %v868 : tensor<32x150528xf32>
    %v871 = stablehlo.add %v866, %v870 : tensor<32x150528xf32>
    %v872 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v873 = stablehlo.multiply %v872, %v871 : tensor<32x150528xf32>
    %v874 = stablehlo.tanh %v873 : tensor<32x150528xf32>
    %v875 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v876 = stablehlo.add %v875, %v874 : tensor<32x150528xf32>
    %v877 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v878 = stablehlo.multiply %v877, %v866 : tensor<32x150528xf32>
    %v879 = stablehlo.multiply %v878, %v876 : tensor<32x150528xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v881 = stablehlo.convolution(%v880, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v882 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v883 = stablehlo.add %v881, %v882 : tensor<32x768x7x7xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v886 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v887 = stablehlo.multiply %v885, %v886 : tensor<32x768x7x7xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v889 = stablehlo.add %v888, %v838 : tensor<32x37632xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v891 = stablehlo.convolution(%v890, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v892 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v893 = stablehlo.add %v891, %v892 : tensor<32x768x7x7xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v896 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v897 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v898 = stablehlo.reduce(%v894 init: %v895) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v899 = stablehlo.broadcast_in_dim %v898, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v900 = stablehlo.divide %v899, %v896 : tensor<32x37632xf32>
    %v901 = stablehlo.subtract %v894, %v900 : tensor<32x37632xf32>
    %v902 = stablehlo.multiply %v901, %v901 : tensor<32x37632xf32>
    %v903 = stablehlo.reduce(%v902 init: %v895) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v905 = stablehlo.divide %v904, %v896 : tensor<32x37632xf32>
    %v906 = stablehlo.add %v905, %v897 : tensor<32x37632xf32>
    %v907 = stablehlo.rsqrt %v906 : tensor<32x37632xf32>
    %v908 = stablehlo.multiply %v901, %v907 : tensor<32x37632xf32>
    %v909 = stablehlo.broadcast_in_dim %s3b1ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v910 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v911 = stablehlo.multiply %v908, %v909 : tensor<32x37632xf32>
    %v912 = stablehlo.add %v911, %v910 : tensor<32x37632xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v914 = stablehlo.convolution(%v913, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v915 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v916 = stablehlo.add %v914, %v915 : tensor<32x3072x7x7xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v918 = stablehlo.multiply %v917, %v917 : tensor<32x150528xf32>
    %v919 = stablehlo.multiply %v918, %v917 : tensor<32x150528xf32>
    %v920 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v921 = stablehlo.multiply %v920, %v919 : tensor<32x150528xf32>
    %v922 = stablehlo.add %v917, %v921 : tensor<32x150528xf32>
    %v923 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v924 = stablehlo.multiply %v923, %v922 : tensor<32x150528xf32>
    %v925 = stablehlo.tanh %v924 : tensor<32x150528xf32>
    %v926 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v927 = stablehlo.add %v926, %v925 : tensor<32x150528xf32>
    %v928 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v929 = stablehlo.multiply %v928, %v917 : tensor<32x150528xf32>
    %v930 = stablehlo.multiply %v929, %v927 : tensor<32x150528xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v932 = stablehlo.convolution(%v931, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v933 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v934 = stablehlo.add %v932, %v933 : tensor<32x768x7x7xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v937 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v938 = stablehlo.multiply %v936, %v937 : tensor<32x768x7x7xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v940 = stablehlo.add %v939, %v889 : tensor<32x37632xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v942 = stablehlo.convolution(%v941, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v943 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v944 = stablehlo.add %v942, %v943 : tensor<32x768x7x7xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v946 = stablehlo.constant dense<0.0> : tensor<f32>
    %v947 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v948 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v949 = stablehlo.reduce(%v945 init: %v946) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v950 = stablehlo.broadcast_in_dim %v949, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v951 = stablehlo.divide %v950, %v947 : tensor<32x37632xf32>
    %v952 = stablehlo.subtract %v945, %v951 : tensor<32x37632xf32>
    %v953 = stablehlo.multiply %v952, %v952 : tensor<32x37632xf32>
    %v954 = stablehlo.reduce(%v953 init: %v946) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v955 = stablehlo.broadcast_in_dim %v954, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v956 = stablehlo.divide %v955, %v947 : tensor<32x37632xf32>
    %v957 = stablehlo.add %v956, %v948 : tensor<32x37632xf32>
    %v958 = stablehlo.rsqrt %v957 : tensor<32x37632xf32>
    %v959 = stablehlo.multiply %v952, %v958 : tensor<32x37632xf32>
    %v960 = stablehlo.broadcast_in_dim %s3b2ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v961 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v962 = stablehlo.multiply %v959, %v960 : tensor<32x37632xf32>
    %v963 = stablehlo.add %v962, %v961 : tensor<32x37632xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v965 = stablehlo.convolution(%v964, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v966 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v967 = stablehlo.add %v965, %v966 : tensor<32x3072x7x7xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v969 = stablehlo.multiply %v968, %v968 : tensor<32x150528xf32>
    %v970 = stablehlo.multiply %v969, %v968 : tensor<32x150528xf32>
    %v971 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v972 = stablehlo.multiply %v971, %v970 : tensor<32x150528xf32>
    %v973 = stablehlo.add %v968, %v972 : tensor<32x150528xf32>
    %v974 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v975 = stablehlo.multiply %v974, %v973 : tensor<32x150528xf32>
    %v976 = stablehlo.tanh %v975 : tensor<32x150528xf32>
    %v977 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v978 = stablehlo.add %v977, %v976 : tensor<32x150528xf32>
    %v979 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v980 = stablehlo.multiply %v979, %v968 : tensor<32x150528xf32>
    %v981 = stablehlo.multiply %v980, %v978 : tensor<32x150528xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v983 = stablehlo.convolution(%v982, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v984 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v985 = stablehlo.add %v983, %v984 : tensor<32x768x7x7xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v987 = stablehlo.reshape %v986 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v988 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v989 = stablehlo.multiply %v987, %v988 : tensor<32x768x7x7xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v991 = stablehlo.add %v990, %v940 : tensor<32x37632xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v994 = stablehlo.reduce(%v992 init: %v993) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %v995 = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %v996 = stablehlo.divide %v994, %v995 : tensor<32x768xf32>
    %v997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v998 = stablehlo.constant dense<768.0> : tensor<32x768xf32>
    %v999 = stablehlo.constant dense<1.0e-6> : tensor<32x768xf32>
    %v1000 = stablehlo.reduce(%v996 init: %v997) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1001 = stablehlo.broadcast_in_dim %v1000, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1002 = stablehlo.divide %v1001, %v998 : tensor<32x768xf32>
    %v1003 = stablehlo.subtract %v996, %v1002 : tensor<32x768xf32>
    %v1004 = stablehlo.multiply %v1003, %v1003 : tensor<32x768xf32>
    %v1005 = stablehlo.reduce(%v1004 init: %v997) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1006 = stablehlo.broadcast_in_dim %v1005, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1007 = stablehlo.divide %v1006, %v998 : tensor<32x768xf32>
    %v1008 = stablehlo.add %v1007, %v999 : tensor<32x768xf32>
    %v1009 = stablehlo.rsqrt %v1008 : tensor<32x768xf32>
    %v1010 = stablehlo.multiply %v1003, %v1009 : tensor<32x768xf32>
    %v1011 = stablehlo.broadcast_in_dim %hng, dims = [] : (tensor<f32>) -> tensor<32x768xf32>
    %v1012 = stablehlo.broadcast_in_dim %hnbt, dims = [] : (tensor<f32>) -> tensor<32x768xf32>
    %v1013 = stablehlo.multiply %v1010, %v1011 : tensor<32x768xf32>
    %v1014 = stablehlo.add %v1013, %v1012 : tensor<32x768xf32>
    %v1015 = stablehlo.dot_general %v1014, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x10xf32>) -> tensor<32x10xf32>
    %v1016 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1017 = stablehlo.add %v1015, %v1016 : tensor<32x10xf32>
    %v1018 = stablehlo.exponential %v1017 : tensor<32x10xf32>
    %v1019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1020 = stablehlo.reduce(%v1018 init: %v1019) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v1021 = stablehlo.broadcast_in_dim %v1020, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v1022 = stablehlo.divide %v1018, %v1021 : tensor<32x10xf32>
    %v1023 = stablehlo.subtract %v1022, %onehot : tensor<32x10xf32>
    %dy = stablehlo.divide %v1023, %bsc : tensor<32x10xf32>
    %v1024 = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<768x10xf32>) -> tensor<32x768xf32>
    %v1025 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1026 = stablehlo.constant dense<768.0> : tensor<32x768xf32>
    %v1027 = stablehlo.constant dense<1.0e-6> : tensor<32x768xf32>
    %v1028 = stablehlo.reduce(%v996 init: %v1025) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1029 = stablehlo.broadcast_in_dim %v1028, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1030 = stablehlo.divide %v1029, %v1026 : tensor<32x768xf32>
    %v1031 = stablehlo.subtract %v996, %v1030 : tensor<32x768xf32>
    %v1032 = stablehlo.multiply %v1031, %v1031 : tensor<32x768xf32>
    %v1033 = stablehlo.reduce(%v1032 init: %v1025) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1034 = stablehlo.broadcast_in_dim %v1033, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1035 = stablehlo.divide %v1034, %v1026 : tensor<32x768xf32>
    %v1036 = stablehlo.add %v1035, %v1027 : tensor<32x768xf32>
    %v1037 = stablehlo.rsqrt %v1036 : tensor<32x768xf32>
    %v1038 = stablehlo.multiply %v1031, %v1037 : tensor<32x768xf32>
    %v1039 = stablehlo.broadcast_in_dim %hng, dims = [] : (tensor<f32>) -> tensor<32x768xf32>
    %v1040 = stablehlo.multiply %v1039, %v1024 : tensor<32x768xf32>
    %v1041 = stablehlo.reduce(%v1040 init: %v1025) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1042 = stablehlo.broadcast_in_dim %v1041, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1043 = stablehlo.multiply %v1038, %v1040 : tensor<32x768xf32>
    %v1044 = stablehlo.reduce(%v1043 init: %v1025) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1045 = stablehlo.broadcast_in_dim %v1044, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1046 = stablehlo.multiply %v1040, %v1026 : tensor<32x768xf32>
    %v1047 = stablehlo.subtract %v1046, %v1042 : tensor<32x768xf32>
    %v1048 = stablehlo.multiply %v1038, %v1045 : tensor<32x768xf32>
    %v1049 = stablehlo.subtract %v1047, %v1048 : tensor<32x768xf32>
    %v1050 = stablehlo.divide %v1037, %v1026 : tensor<32x768xf32>
    %v1051 = stablehlo.multiply %v1050, %v1049 : tensor<32x768xf32>
    %v1052 = stablehlo.dot_general %v1014, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<32x10xf32>) -> tensor<768x10xf32>
    %v1053 = stablehlo.constant dense<0.1> : tensor<768x10xf32>
    %v1054 = stablehlo.multiply %v1052, %v1053 : tensor<768x10xf32>
    %v1055 = stablehlo.subtract %Wd, %v1054 : tensor<768x10xf32>
    %v1056 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1057 = stablehlo.reduce(%dy init: %v1056) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1058 = stablehlo.constant dense<0.1> : tensor<10xf32>
    %v1059 = stablehlo.multiply %v1057, %v1058 : tensor<10xf32>
    %v1060 = stablehlo.subtract %bd, %v1059 : tensor<10xf32>
    %v1061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1062 = stablehlo.constant dense<768.0> : tensor<32x768xf32>
    %v1063 = stablehlo.constant dense<1.0e-6> : tensor<32x768xf32>
    %v1064 = stablehlo.reduce(%v996 init: %v1061) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1065 = stablehlo.broadcast_in_dim %v1064, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1066 = stablehlo.divide %v1065, %v1062 : tensor<32x768xf32>
    %v1067 = stablehlo.subtract %v996, %v1066 : tensor<32x768xf32>
    %v1068 = stablehlo.multiply %v1067, %v1067 : tensor<32x768xf32>
    %v1069 = stablehlo.reduce(%v1068 init: %v1061) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1070 = stablehlo.broadcast_in_dim %v1069, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1071 = stablehlo.divide %v1070, %v1062 : tensor<32x768xf32>
    %v1072 = stablehlo.add %v1071, %v1063 : tensor<32x768xf32>
    %v1073 = stablehlo.rsqrt %v1072 : tensor<32x768xf32>
    %v1074 = stablehlo.multiply %v1067, %v1073 : tensor<32x768xf32>
    %v1075 = stablehlo.multiply %v1024, %v1074 : tensor<32x768xf32>
    %v1076 = stablehlo.reduce(%v1075 init: %v1061) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<f32>
    %v1077 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1078 = stablehlo.multiply %v1076, %v1077 : tensor<f32>
    %v1079 = stablehlo.subtract %hng, %v1078 : tensor<f32>
    %v1080 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1081 = stablehlo.reduce(%v1024 init: %v1080) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<f32>
    %v1082 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1083 = stablehlo.multiply %v1081, %v1082 : tensor<f32>
    %v1084 = stablehlo.subtract %hnbt, %v1083 : tensor<f32>
    %dgi = stablehlo.reshape %v1051 : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : (tensor<32x768x1x1xf32>) -> tensor<32x768x7x7xf32>
    %dgn = stablehlo.constant dense<49.0> : tensor<32x768x7x7xf32>
    %dgd = stablehlo.divide %dgb, %dgn : tensor<32x768x7x7xf32>
    %dgapf = stablehlo.reshape %dgd : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1085 = stablehlo.reshape %dgapf : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1086 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1087 = stablehlo.multiply %v1085, %v1086 : tensor<32x768x7x7xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1090 = stablehlo.transpose %s3b2pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1091 = stablehlo.reverse %v1090, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1092 = stablehlo.convolution(%v1089, %v1091)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1094 = stablehlo.multiply %v968, %v968 : tensor<32x150528xf32>
    %v1095 = stablehlo.multiply %v1094, %v968 : tensor<32x150528xf32>
    %v1096 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1097 = stablehlo.multiply %v1096, %v1095 : tensor<32x150528xf32>
    %v1098 = stablehlo.add %v968, %v1097 : tensor<32x150528xf32>
    %v1099 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1100 = stablehlo.multiply %v1099, %v1098 : tensor<32x150528xf32>
    %v1101 = stablehlo.tanh %v1100 : tensor<32x150528xf32>
    %v1102 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1103 = stablehlo.add %v1102, %v1101 : tensor<32x150528xf32>
    %v1104 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1105 = stablehlo.multiply %v1104, %v1103 : tensor<32x150528xf32>
    %v1106 = stablehlo.multiply %v1101, %v1101 : tensor<32x150528xf32>
    %v1107 = stablehlo.subtract %v1102, %v1106 : tensor<32x150528xf32>
    %v1108 = stablehlo.multiply %v1104, %v968 : tensor<32x150528xf32>
    %v1109 = stablehlo.multiply %v1108, %v1107 : tensor<32x150528xf32>
    %v1110 = stablehlo.constant dense<0.134145> : tensor<32x150528xf32>
    %v1111 = stablehlo.multiply %v1110, %v1094 : tensor<32x150528xf32>
    %v1112 = stablehlo.add %v1102, %v1111 : tensor<32x150528xf32>
    %v1113 = stablehlo.multiply %v1099, %v1112 : tensor<32x150528xf32>
    %v1114 = stablehlo.multiply %v1109, %v1113 : tensor<32x150528xf32>
    %v1115 = stablehlo.add %v1105, %v1114 : tensor<32x150528xf32>
    %v1116 = stablehlo.multiply %v1093, %v1115 : tensor<32x150528xf32>
    %v1117 = stablehlo.reshape %v1116 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1118 = stablehlo.transpose %s3b2eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1119 = stablehlo.reverse %v1118, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1120 = stablehlo.convolution(%v1117, %v1119)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1123 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v1124 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v1125 = stablehlo.reduce(%v945 init: %v1122) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1126 = stablehlo.broadcast_in_dim %v1125, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1127 = stablehlo.divide %v1126, %v1123 : tensor<32x37632xf32>
    %v1128 = stablehlo.subtract %v945, %v1127 : tensor<32x37632xf32>
    %v1129 = stablehlo.multiply %v1128, %v1128 : tensor<32x37632xf32>
    %v1130 = stablehlo.reduce(%v1129 init: %v1122) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1131 = stablehlo.broadcast_in_dim %v1130, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1132 = stablehlo.divide %v1131, %v1123 : tensor<32x37632xf32>
    %v1133 = stablehlo.add %v1132, %v1124 : tensor<32x37632xf32>
    %v1134 = stablehlo.rsqrt %v1133 : tensor<32x37632xf32>
    %v1135 = stablehlo.multiply %v1128, %v1134 : tensor<32x37632xf32>
    %v1136 = stablehlo.broadcast_in_dim %s3b2ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v1137 = stablehlo.multiply %v1136, %v1121 : tensor<32x37632xf32>
    %v1138 = stablehlo.reduce(%v1137 init: %v1122) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1139 = stablehlo.broadcast_in_dim %v1138, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1140 = stablehlo.multiply %v1135, %v1137 : tensor<32x37632xf32>
    %v1141 = stablehlo.reduce(%v1140 init: %v1122) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1142 = stablehlo.broadcast_in_dim %v1141, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1143 = stablehlo.multiply %v1137, %v1123 : tensor<32x37632xf32>
    %v1144 = stablehlo.subtract %v1143, %v1139 : tensor<32x37632xf32>
    %v1145 = stablehlo.multiply %v1135, %v1142 : tensor<32x37632xf32>
    %v1146 = stablehlo.subtract %v1144, %v1145 : tensor<32x37632xf32>
    %v1147 = stablehlo.divide %v1134, %v1123 : tensor<32x37632xf32>
    %v1148 = stablehlo.multiply %v1147, %v1146 : tensor<32x37632xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1150 = stablehlo.reverse %s3b2dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1151 = stablehlo.convolution(%v1149, %v1150)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1153 = stablehlo.add %v1152, %dgapf : tensor<32x37632xf32>
    %v1154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1155 = stablehlo.reshape %v986 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1156 = stablehlo.reshape %dgapf : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1157 = stablehlo.multiply %v1155, %v1156 : tensor<32x768x7x7xf32>
    %v1158 = stablehlo.reduce(%v1157 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1159 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1160 = stablehlo.multiply %v1158, %v1159 : tensor<768xf32>
    %v1161 = stablehlo.subtract %s3b2lg, %v1160 : tensor<768xf32>
    %v1162 = stablehlo.reshape %v981 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1163 = stablehlo.reshape %v1088 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1164 = stablehlo.transpose %v1162, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1165 = stablehlo.transpose %v1163, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1166 = stablehlo.convolution(%v1164, %v1165)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1167 = stablehlo.transpose %v1166, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1168 = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %v1169 = stablehlo.multiply %v1167, %v1168 : tensor<768x3072x1x1xf32>
    %v1170 = stablehlo.subtract %s3b2pW, %v1169 : tensor<768x3072x1x1xf32>
    %v1171 = stablehlo.reshape %v1088 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1173 = stablehlo.reduce(%v1171 init: %v1172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1174 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1175 = stablehlo.multiply %v1173, %v1174 : tensor<768xf32>
    %v1176 = stablehlo.subtract %s3b2pb, %v1175 : tensor<768xf32>
    %v1177 = stablehlo.reshape %v963 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1178 = stablehlo.reshape %v1116 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1179 = stablehlo.transpose %v1177, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1180 = stablehlo.transpose %v1178, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1181 = stablehlo.convolution(%v1179, %v1180)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1182 = stablehlo.transpose %v1181, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1183 = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %v1184 = stablehlo.multiply %v1182, %v1183 : tensor<3072x768x1x1xf32>
    %v1185 = stablehlo.subtract %s3b2eW, %v1184 : tensor<3072x768x1x1xf32>
    %v1186 = stablehlo.reshape %v1116 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1187 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1188 = stablehlo.reduce(%v1186 init: %v1187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1189 = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %v1190 = stablehlo.multiply %v1188, %v1189 : tensor<3072xf32>
    %v1191 = stablehlo.subtract %s3b2eb, %v1190 : tensor<3072xf32>
    %v1192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1193 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v1194 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v1195 = stablehlo.reduce(%v945 init: %v1192) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1196 = stablehlo.broadcast_in_dim %v1195, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1197 = stablehlo.divide %v1196, %v1193 : tensor<32x37632xf32>
    %v1198 = stablehlo.subtract %v945, %v1197 : tensor<32x37632xf32>
    %v1199 = stablehlo.multiply %v1198, %v1198 : tensor<32x37632xf32>
    %v1200 = stablehlo.reduce(%v1199 init: %v1192) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1201 = stablehlo.broadcast_in_dim %v1200, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1202 = stablehlo.divide %v1201, %v1193 : tensor<32x37632xf32>
    %v1203 = stablehlo.add %v1202, %v1194 : tensor<32x37632xf32>
    %v1204 = stablehlo.rsqrt %v1203 : tensor<32x37632xf32>
    %v1205 = stablehlo.multiply %v1198, %v1204 : tensor<32x37632xf32>
    %v1206 = stablehlo.multiply %v1121, %v1205 : tensor<32x37632xf32>
    %v1207 = stablehlo.reduce(%v1206 init: %v1192) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %v1208 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1209 = stablehlo.multiply %v1207, %v1208 : tensor<f32>
    %v1210 = stablehlo.subtract %s3b2ng, %v1209 : tensor<f32>
    %v1211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1212 = stablehlo.reduce(%v1121 init: %v1211) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %v1213 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1214 = stablehlo.multiply %v1212, %v1213 : tensor<f32>
    %v1215 = stablehlo.subtract %s3b2nbt, %v1214 : tensor<f32>
    %v1216 = stablehlo.reshape %v940 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1217 = stablehlo.reshape %v1148 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1218 = stablehlo.transpose %v1216, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1219 = stablehlo.transpose %v1217, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1220 = stablehlo.convolution(%v1218, %v1219)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1222 = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %v1223 = stablehlo.multiply %v1221, %v1222 : tensor<768x1x7x7xf32>
    %v1224 = stablehlo.subtract %s3b2dW, %v1223 : tensor<768x1x7x7xf32>
    %v1225 = stablehlo.reshape %v1148 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1227 = stablehlo.reduce(%v1225 init: %v1226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1228 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1229 = stablehlo.multiply %v1227, %v1228 : tensor<768xf32>
    %v1230 = stablehlo.subtract %s3b2db, %v1229 : tensor<768xf32>
    %v1231 = stablehlo.reshape %v1153 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1232 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1233 = stablehlo.multiply %v1231, %v1232 : tensor<32x768x7x7xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1236 = stablehlo.transpose %s3b1pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1237 = stablehlo.reverse %v1236, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1238 = stablehlo.convolution(%v1235, %v1237)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1239 = stablehlo.reshape %v1238 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1240 = stablehlo.multiply %v917, %v917 : tensor<32x150528xf32>
    %v1241 = stablehlo.multiply %v1240, %v917 : tensor<32x150528xf32>
    %v1242 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1243 = stablehlo.multiply %v1242, %v1241 : tensor<32x150528xf32>
    %v1244 = stablehlo.add %v917, %v1243 : tensor<32x150528xf32>
    %v1245 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1246 = stablehlo.multiply %v1245, %v1244 : tensor<32x150528xf32>
    %v1247 = stablehlo.tanh %v1246 : tensor<32x150528xf32>
    %v1248 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1249 = stablehlo.add %v1248, %v1247 : tensor<32x150528xf32>
    %v1250 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1251 = stablehlo.multiply %v1250, %v1249 : tensor<32x150528xf32>
    %v1252 = stablehlo.multiply %v1247, %v1247 : tensor<32x150528xf32>
    %v1253 = stablehlo.subtract %v1248, %v1252 : tensor<32x150528xf32>
    %v1254 = stablehlo.multiply %v1250, %v917 : tensor<32x150528xf32>
    %v1255 = stablehlo.multiply %v1254, %v1253 : tensor<32x150528xf32>
    %v1256 = stablehlo.constant dense<0.134145> : tensor<32x150528xf32>
    %v1257 = stablehlo.multiply %v1256, %v1240 : tensor<32x150528xf32>
    %v1258 = stablehlo.add %v1248, %v1257 : tensor<32x150528xf32>
    %v1259 = stablehlo.multiply %v1245, %v1258 : tensor<32x150528xf32>
    %v1260 = stablehlo.multiply %v1255, %v1259 : tensor<32x150528xf32>
    %v1261 = stablehlo.add %v1251, %v1260 : tensor<32x150528xf32>
    %v1262 = stablehlo.multiply %v1239, %v1261 : tensor<32x150528xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1264 = stablehlo.transpose %s3b1eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1265 = stablehlo.reverse %v1264, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1266 = stablehlo.convolution(%v1263, %v1265)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1269 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v1270 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v1271 = stablehlo.reduce(%v894 init: %v1268) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1272 = stablehlo.broadcast_in_dim %v1271, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1273 = stablehlo.divide %v1272, %v1269 : tensor<32x37632xf32>
    %v1274 = stablehlo.subtract %v894, %v1273 : tensor<32x37632xf32>
    %v1275 = stablehlo.multiply %v1274, %v1274 : tensor<32x37632xf32>
    %v1276 = stablehlo.reduce(%v1275 init: %v1268) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1277 = stablehlo.broadcast_in_dim %v1276, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1278 = stablehlo.divide %v1277, %v1269 : tensor<32x37632xf32>
    %v1279 = stablehlo.add %v1278, %v1270 : tensor<32x37632xf32>
    %v1280 = stablehlo.rsqrt %v1279 : tensor<32x37632xf32>
    %v1281 = stablehlo.multiply %v1274, %v1280 : tensor<32x37632xf32>
    %v1282 = stablehlo.broadcast_in_dim %s3b1ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v1283 = stablehlo.multiply %v1282, %v1267 : tensor<32x37632xf32>
    %v1284 = stablehlo.reduce(%v1283 init: %v1268) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1285 = stablehlo.broadcast_in_dim %v1284, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1286 = stablehlo.multiply %v1281, %v1283 : tensor<32x37632xf32>
    %v1287 = stablehlo.reduce(%v1286 init: %v1268) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1288 = stablehlo.broadcast_in_dim %v1287, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1289 = stablehlo.multiply %v1283, %v1269 : tensor<32x37632xf32>
    %v1290 = stablehlo.subtract %v1289, %v1285 : tensor<32x37632xf32>
    %v1291 = stablehlo.multiply %v1281, %v1288 : tensor<32x37632xf32>
    %v1292 = stablehlo.subtract %v1290, %v1291 : tensor<32x37632xf32>
    %v1293 = stablehlo.divide %v1280, %v1269 : tensor<32x37632xf32>
    %v1294 = stablehlo.multiply %v1293, %v1292 : tensor<32x37632xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1296 = stablehlo.reverse %s3b1dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1297 = stablehlo.convolution(%v1295, %v1296)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1299 = stablehlo.add %v1298, %v1153 : tensor<32x37632xf32>
    %v1300 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1301 = stablehlo.reshape %v935 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1302 = stablehlo.reshape %v1153 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1303 = stablehlo.multiply %v1301, %v1302 : tensor<32x768x7x7xf32>
    %v1304 = stablehlo.reduce(%v1303 init: %v1300) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1305 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1306 = stablehlo.multiply %v1304, %v1305 : tensor<768xf32>
    %v1307 = stablehlo.subtract %s3b1lg, %v1306 : tensor<768xf32>
    %v1308 = stablehlo.reshape %v930 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1309 = stablehlo.reshape %v1234 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1310 = stablehlo.transpose %v1308, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1311 = stablehlo.transpose %v1309, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1312 = stablehlo.convolution(%v1310, %v1311)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1313 = stablehlo.transpose %v1312, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1314 = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %v1315 = stablehlo.multiply %v1313, %v1314 : tensor<768x3072x1x1xf32>
    %v1316 = stablehlo.subtract %s3b1pW, %v1315 : tensor<768x3072x1x1xf32>
    %v1317 = stablehlo.reshape %v1234 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1319 = stablehlo.reduce(%v1317 init: %v1318) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1320 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1321 = stablehlo.multiply %v1319, %v1320 : tensor<768xf32>
    %v1322 = stablehlo.subtract %s3b1pb, %v1321 : tensor<768xf32>
    %v1323 = stablehlo.reshape %v912 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1324 = stablehlo.reshape %v1262 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1325 = stablehlo.transpose %v1323, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1326 = stablehlo.transpose %v1324, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1327 = stablehlo.convolution(%v1325, %v1326)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1328 = stablehlo.transpose %v1327, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1329 = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %v1330 = stablehlo.multiply %v1328, %v1329 : tensor<3072x768x1x1xf32>
    %v1331 = stablehlo.subtract %s3b1eW, %v1330 : tensor<3072x768x1x1xf32>
    %v1332 = stablehlo.reshape %v1262 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1334 = stablehlo.reduce(%v1332 init: %v1333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1335 = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %v1336 = stablehlo.multiply %v1334, %v1335 : tensor<3072xf32>
    %v1337 = stablehlo.subtract %s3b1eb, %v1336 : tensor<3072xf32>
    %v1338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1339 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v1340 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v1341 = stablehlo.reduce(%v894 init: %v1338) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1342 = stablehlo.broadcast_in_dim %v1341, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1343 = stablehlo.divide %v1342, %v1339 : tensor<32x37632xf32>
    %v1344 = stablehlo.subtract %v894, %v1343 : tensor<32x37632xf32>
    %v1345 = stablehlo.multiply %v1344, %v1344 : tensor<32x37632xf32>
    %v1346 = stablehlo.reduce(%v1345 init: %v1338) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1347 = stablehlo.broadcast_in_dim %v1346, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1348 = stablehlo.divide %v1347, %v1339 : tensor<32x37632xf32>
    %v1349 = stablehlo.add %v1348, %v1340 : tensor<32x37632xf32>
    %v1350 = stablehlo.rsqrt %v1349 : tensor<32x37632xf32>
    %v1351 = stablehlo.multiply %v1344, %v1350 : tensor<32x37632xf32>
    %v1352 = stablehlo.multiply %v1267, %v1351 : tensor<32x37632xf32>
    %v1353 = stablehlo.reduce(%v1352 init: %v1338) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %v1354 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1355 = stablehlo.multiply %v1353, %v1354 : tensor<f32>
    %v1356 = stablehlo.subtract %s3b1ng, %v1355 : tensor<f32>
    %v1357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1358 = stablehlo.reduce(%v1267 init: %v1357) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %v1359 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1360 = stablehlo.multiply %v1358, %v1359 : tensor<f32>
    %v1361 = stablehlo.subtract %s3b1nbt, %v1360 : tensor<f32>
    %v1362 = stablehlo.reshape %v889 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1363 = stablehlo.reshape %v1294 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1364 = stablehlo.transpose %v1362, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1365 = stablehlo.transpose %v1363, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1366 = stablehlo.convolution(%v1364, %v1365)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1368 = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %v1369 = stablehlo.multiply %v1367, %v1368 : tensor<768x1x7x7xf32>
    %v1370 = stablehlo.subtract %s3b1dW, %v1369 : tensor<768x1x7x7xf32>
    %v1371 = stablehlo.reshape %v1294 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1373 = stablehlo.reduce(%v1371 init: %v1372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1374 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1375 = stablehlo.multiply %v1373, %v1374 : tensor<768xf32>
    %v1376 = stablehlo.subtract %s3b1db, %v1375 : tensor<768xf32>
    %v1377 = stablehlo.reshape %v1299 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1378 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1379 = stablehlo.multiply %v1377, %v1378 : tensor<32x768x7x7xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1382 = stablehlo.transpose %s3b0pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1383 = stablehlo.reverse %v1382, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1384 = stablehlo.convolution(%v1381, %v1383)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1386 = stablehlo.multiply %v866, %v866 : tensor<32x150528xf32>
    %v1387 = stablehlo.multiply %v1386, %v866 : tensor<32x150528xf32>
    %v1388 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1389 = stablehlo.multiply %v1388, %v1387 : tensor<32x150528xf32>
    %v1390 = stablehlo.add %v866, %v1389 : tensor<32x150528xf32>
    %v1391 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1392 = stablehlo.multiply %v1391, %v1390 : tensor<32x150528xf32>
    %v1393 = stablehlo.tanh %v1392 : tensor<32x150528xf32>
    %v1394 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1395 = stablehlo.add %v1394, %v1393 : tensor<32x150528xf32>
    %v1396 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1397 = stablehlo.multiply %v1396, %v1395 : tensor<32x150528xf32>
    %v1398 = stablehlo.multiply %v1393, %v1393 : tensor<32x150528xf32>
    %v1399 = stablehlo.subtract %v1394, %v1398 : tensor<32x150528xf32>
    %v1400 = stablehlo.multiply %v1396, %v866 : tensor<32x150528xf32>
    %v1401 = stablehlo.multiply %v1400, %v1399 : tensor<32x150528xf32>
    %v1402 = stablehlo.constant dense<0.134145> : tensor<32x150528xf32>
    %v1403 = stablehlo.multiply %v1402, %v1386 : tensor<32x150528xf32>
    %v1404 = stablehlo.add %v1394, %v1403 : tensor<32x150528xf32>
    %v1405 = stablehlo.multiply %v1391, %v1404 : tensor<32x150528xf32>
    %v1406 = stablehlo.multiply %v1401, %v1405 : tensor<32x150528xf32>
    %v1407 = stablehlo.add %v1397, %v1406 : tensor<32x150528xf32>
    %v1408 = stablehlo.multiply %v1385, %v1407 : tensor<32x150528xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1410 = stablehlo.transpose %s3b0eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1411 = stablehlo.reverse %v1410, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1412 = stablehlo.convolution(%v1409, %v1411)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1415 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v1416 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v1417 = stablehlo.reduce(%v843 init: %v1414) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1418 = stablehlo.broadcast_in_dim %v1417, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1419 = stablehlo.divide %v1418, %v1415 : tensor<32x37632xf32>
    %v1420 = stablehlo.subtract %v843, %v1419 : tensor<32x37632xf32>
    %v1421 = stablehlo.multiply %v1420, %v1420 : tensor<32x37632xf32>
    %v1422 = stablehlo.reduce(%v1421 init: %v1414) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1423 = stablehlo.broadcast_in_dim %v1422, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1424 = stablehlo.divide %v1423, %v1415 : tensor<32x37632xf32>
    %v1425 = stablehlo.add %v1424, %v1416 : tensor<32x37632xf32>
    %v1426 = stablehlo.rsqrt %v1425 : tensor<32x37632xf32>
    %v1427 = stablehlo.multiply %v1420, %v1426 : tensor<32x37632xf32>
    %v1428 = stablehlo.broadcast_in_dim %s3b0ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v1429 = stablehlo.multiply %v1428, %v1413 : tensor<32x37632xf32>
    %v1430 = stablehlo.reduce(%v1429 init: %v1414) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1431 = stablehlo.broadcast_in_dim %v1430, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1432 = stablehlo.multiply %v1427, %v1429 : tensor<32x37632xf32>
    %v1433 = stablehlo.reduce(%v1432 init: %v1414) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1434 = stablehlo.broadcast_in_dim %v1433, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1435 = stablehlo.multiply %v1429, %v1415 : tensor<32x37632xf32>
    %v1436 = stablehlo.subtract %v1435, %v1431 : tensor<32x37632xf32>
    %v1437 = stablehlo.multiply %v1427, %v1434 : tensor<32x37632xf32>
    %v1438 = stablehlo.subtract %v1436, %v1437 : tensor<32x37632xf32>
    %v1439 = stablehlo.divide %v1426, %v1415 : tensor<32x37632xf32>
    %v1440 = stablehlo.multiply %v1439, %v1438 : tensor<32x37632xf32>
    %v1441 = stablehlo.reshape %v1440 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1442 = stablehlo.reverse %s3b0dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1443 = stablehlo.convolution(%v1441, %v1442)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1445 = stablehlo.add %v1444, %v1299 : tensor<32x37632xf32>
    %v1446 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1447 = stablehlo.reshape %v884 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1448 = stablehlo.reshape %v1299 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1449 = stablehlo.multiply %v1447, %v1448 : tensor<32x768x7x7xf32>
    %v1450 = stablehlo.reduce(%v1449 init: %v1446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1451 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1452 = stablehlo.multiply %v1450, %v1451 : tensor<768xf32>
    %v1453 = stablehlo.subtract %s3b0lg, %v1452 : tensor<768xf32>
    %v1454 = stablehlo.reshape %v879 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1455 = stablehlo.reshape %v1380 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1456 = stablehlo.transpose %v1454, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1457 = stablehlo.transpose %v1455, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1458 = stablehlo.convolution(%v1456, %v1457)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1459 = stablehlo.transpose %v1458, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1460 = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %v1461 = stablehlo.multiply %v1459, %v1460 : tensor<768x3072x1x1xf32>
    %v1462 = stablehlo.subtract %s3b0pW, %v1461 : tensor<768x3072x1x1xf32>
    %v1463 = stablehlo.reshape %v1380 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1464 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1465 = stablehlo.reduce(%v1463 init: %v1464) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1466 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1467 = stablehlo.multiply %v1465, %v1466 : tensor<768xf32>
    %v1468 = stablehlo.subtract %s3b0pb, %v1467 : tensor<768xf32>
    %v1469 = stablehlo.reshape %v861 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1470 = stablehlo.reshape %v1408 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1471 = stablehlo.transpose %v1469, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1472 = stablehlo.transpose %v1470, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1473 = stablehlo.convolution(%v1471, %v1472)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1474 = stablehlo.transpose %v1473, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1475 = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %v1476 = stablehlo.multiply %v1474, %v1475 : tensor<3072x768x1x1xf32>
    %v1477 = stablehlo.subtract %s3b0eW, %v1476 : tensor<3072x768x1x1xf32>
    %v1478 = stablehlo.reshape %v1408 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1479 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1480 = stablehlo.reduce(%v1478 init: %v1479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1481 = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %v1482 = stablehlo.multiply %v1480, %v1481 : tensor<3072xf32>
    %v1483 = stablehlo.subtract %s3b0eb, %v1482 : tensor<3072xf32>
    %v1484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1485 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v1486 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v1487 = stablehlo.reduce(%v843 init: %v1484) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1488 = stablehlo.broadcast_in_dim %v1487, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1489 = stablehlo.divide %v1488, %v1485 : tensor<32x37632xf32>
    %v1490 = stablehlo.subtract %v843, %v1489 : tensor<32x37632xf32>
    %v1491 = stablehlo.multiply %v1490, %v1490 : tensor<32x37632xf32>
    %v1492 = stablehlo.reduce(%v1491 init: %v1484) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1493 = stablehlo.broadcast_in_dim %v1492, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1494 = stablehlo.divide %v1493, %v1485 : tensor<32x37632xf32>
    %v1495 = stablehlo.add %v1494, %v1486 : tensor<32x37632xf32>
    %v1496 = stablehlo.rsqrt %v1495 : tensor<32x37632xf32>
    %v1497 = stablehlo.multiply %v1490, %v1496 : tensor<32x37632xf32>
    %v1498 = stablehlo.multiply %v1413, %v1497 : tensor<32x37632xf32>
    %v1499 = stablehlo.reduce(%v1498 init: %v1484) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %v1500 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1501 = stablehlo.multiply %v1499, %v1500 : tensor<f32>
    %v1502 = stablehlo.subtract %s3b0ng, %v1501 : tensor<f32>
    %v1503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1504 = stablehlo.reduce(%v1413 init: %v1503) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %v1505 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1506 = stablehlo.multiply %v1504, %v1505 : tensor<f32>
    %v1507 = stablehlo.subtract %s3b0nbt, %v1506 : tensor<f32>
    %v1508 = stablehlo.reshape %v838 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1509 = stablehlo.reshape %v1440 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1510 = stablehlo.transpose %v1508, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1511 = stablehlo.transpose %v1509, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1512 = stablehlo.convolution(%v1510, %v1511)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1513 = stablehlo.reshape %v1512 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1514 = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %v1515 = stablehlo.multiply %v1513, %v1514 : tensor<768x1x7x7xf32>
    %v1516 = stablehlo.subtract %s3b0dW, %v1515 : tensor<768x1x7x7xf32>
    %v1517 = stablehlo.reshape %v1440 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1519 = stablehlo.reduce(%v1517 init: %v1518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1520 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1521 = stablehlo.multiply %v1519, %v1520 : tensor<768xf32>
    %v1522 = stablehlo.subtract %s3b0db, %v1521 : tensor<768xf32>
    %v1523 = stablehlo.reshape %v1445 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1525 = stablehlo.pad %v1523, %v1524, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x14x14xf32>
    %v1526 = stablehlo.transpose %d2W, dims = [1, 0, 2, 3] : (tensor<768x384x2x2xf32>) -> tensor<384x768x2x2xf32>
    %v1527 = stablehlo.reverse %v1526, dims = [2, 3] : tensor<384x768x2x2xf32>
    %v1528 = stablehlo.convolution(%v1525, %v1527)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x14x14xf32>, tensor<384x768x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1531 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1532 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1533 = stablehlo.reduce(%v815 init: %v1530) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1534 = stablehlo.broadcast_in_dim %v1533, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1535 = stablehlo.divide %v1534, %v1531 : tensor<32x75264xf32>
    %v1536 = stablehlo.subtract %v815, %v1535 : tensor<32x75264xf32>
    %v1537 = stablehlo.multiply %v1536, %v1536 : tensor<32x75264xf32>
    %v1538 = stablehlo.reduce(%v1537 init: %v1530) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1539 = stablehlo.broadcast_in_dim %v1538, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1540 = stablehlo.divide %v1539, %v1531 : tensor<32x75264xf32>
    %v1541 = stablehlo.add %v1540, %v1532 : tensor<32x75264xf32>
    %v1542 = stablehlo.rsqrt %v1541 : tensor<32x75264xf32>
    %v1543 = stablehlo.multiply %v1536, %v1542 : tensor<32x75264xf32>
    %v1544 = stablehlo.broadcast_in_dim %d2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1545 = stablehlo.multiply %v1544, %v1529 : tensor<32x75264xf32>
    %v1546 = stablehlo.reduce(%v1545 init: %v1530) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1547 = stablehlo.broadcast_in_dim %v1546, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1548 = stablehlo.multiply %v1543, %v1545 : tensor<32x75264xf32>
    %v1549 = stablehlo.reduce(%v1548 init: %v1530) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1550 = stablehlo.broadcast_in_dim %v1549, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1551 = stablehlo.multiply %v1545, %v1531 : tensor<32x75264xf32>
    %v1552 = stablehlo.subtract %v1551, %v1547 : tensor<32x75264xf32>
    %v1553 = stablehlo.multiply %v1543, %v1550 : tensor<32x75264xf32>
    %v1554 = stablehlo.subtract %v1552, %v1553 : tensor<32x75264xf32>
    %v1555 = stablehlo.divide %v1542, %v1531 : tensor<32x75264xf32>
    %v1556 = stablehlo.multiply %v1555, %v1554 : tensor<32x75264xf32>
    %v1557 = stablehlo.reshape %v1445 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1558 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1559 = stablehlo.reduce(%v1557 init: %v1558) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1560 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1561 = stablehlo.multiply %v1559, %v1560 : tensor<768xf32>
    %v1562 = stablehlo.subtract %d2b, %v1561 : tensor<768xf32>
    %v1563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1564 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1565 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1566 = stablehlo.reduce(%v815 init: %v1563) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1567 = stablehlo.broadcast_in_dim %v1566, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1568 = stablehlo.divide %v1567, %v1564 : tensor<32x75264xf32>
    %v1569 = stablehlo.subtract %v815, %v1568 : tensor<32x75264xf32>
    %v1570 = stablehlo.multiply %v1569, %v1569 : tensor<32x75264xf32>
    %v1571 = stablehlo.reduce(%v1570 init: %v1563) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1572 = stablehlo.broadcast_in_dim %v1571, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1573 = stablehlo.divide %v1572, %v1564 : tensor<32x75264xf32>
    %v1574 = stablehlo.add %v1573, %v1565 : tensor<32x75264xf32>
    %v1575 = stablehlo.rsqrt %v1574 : tensor<32x75264xf32>
    %v1576 = stablehlo.multiply %v1569, %v1575 : tensor<32x75264xf32>
    %v1577 = stablehlo.multiply %v1529, %v1576 : tensor<32x75264xf32>
    %v1578 = stablehlo.reduce(%v1577 init: %v1563) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1579 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1580 = stablehlo.multiply %v1578, %v1579 : tensor<f32>
    %v1581 = stablehlo.subtract %d2ng, %v1580 : tensor<f32>
    %v1582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1583 = stablehlo.reduce(%v1529 init: %v1582) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1584 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1585 = stablehlo.multiply %v1583, %v1584 : tensor<f32>
    %v1586 = stablehlo.subtract %d2nbt, %v1585 : tensor<f32>
    %v1587 = stablehlo.reshape %v833 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1588 = stablehlo.reshape %v1445 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1590 = stablehlo.pad %v1588, %v1589, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x13x13xf32>
    %v1591 = stablehlo.transpose %v1587, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1592 = stablehlo.transpose %v1590, dims = [1, 0, 2, 3] : (tensor<32x768x13x13xf32>) -> tensor<768x32x13x13xf32>
    %v1593 = stablehlo.convolution(%v1591, %v1592)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<768x32x13x13xf32>) -> tensor<384x768x2x2xf32>
    %v1594 = stablehlo.transpose %v1593, dims = [1, 0, 2, 3] : (tensor<384x768x2x2xf32>) -> tensor<768x384x2x2xf32>
    %v1595 = stablehlo.constant dense<0.1> : tensor<768x384x2x2xf32>
    %v1596 = stablehlo.multiply %v1594, %v1595 : tensor<768x384x2x2xf32>
    %v1597 = stablehlo.subtract %d2W, %v1596 : tensor<768x384x2x2xf32>
    %v1598 = stablehlo.reshape %v1556 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1599 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1600 = stablehlo.multiply %v1598, %v1599 : tensor<32x384x14x14xf32>
    %v1601 = stablehlo.reshape %v1600 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1603 = stablehlo.transpose %s2b8pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1604 = stablehlo.reverse %v1603, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1605 = stablehlo.convolution(%v1602, %v1604)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1606 = stablehlo.reshape %v1605 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1607 = stablehlo.multiply %v792, %v792 : tensor<32x301056xf32>
    %v1608 = stablehlo.multiply %v1607, %v792 : tensor<32x301056xf32>
    %v1609 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1610 = stablehlo.multiply %v1609, %v1608 : tensor<32x301056xf32>
    %v1611 = stablehlo.add %v792, %v1610 : tensor<32x301056xf32>
    %v1612 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1613 = stablehlo.multiply %v1612, %v1611 : tensor<32x301056xf32>
    %v1614 = stablehlo.tanh %v1613 : tensor<32x301056xf32>
    %v1615 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1616 = stablehlo.add %v1615, %v1614 : tensor<32x301056xf32>
    %v1617 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1618 = stablehlo.multiply %v1617, %v1616 : tensor<32x301056xf32>
    %v1619 = stablehlo.multiply %v1614, %v1614 : tensor<32x301056xf32>
    %v1620 = stablehlo.subtract %v1615, %v1619 : tensor<32x301056xf32>
    %v1621 = stablehlo.multiply %v1617, %v792 : tensor<32x301056xf32>
    %v1622 = stablehlo.multiply %v1621, %v1620 : tensor<32x301056xf32>
    %v1623 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1624 = stablehlo.multiply %v1623, %v1607 : tensor<32x301056xf32>
    %v1625 = stablehlo.add %v1615, %v1624 : tensor<32x301056xf32>
    %v1626 = stablehlo.multiply %v1612, %v1625 : tensor<32x301056xf32>
    %v1627 = stablehlo.multiply %v1622, %v1626 : tensor<32x301056xf32>
    %v1628 = stablehlo.add %v1618, %v1627 : tensor<32x301056xf32>
    %v1629 = stablehlo.multiply %v1606, %v1628 : tensor<32x301056xf32>
    %v1630 = stablehlo.reshape %v1629 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1631 = stablehlo.transpose %s2b8eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1632 = stablehlo.reverse %v1631, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1633 = stablehlo.convolution(%v1630, %v1632)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1635 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1636 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1637 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1638 = stablehlo.reduce(%v769 init: %v1635) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1639 = stablehlo.broadcast_in_dim %v1638, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1640 = stablehlo.divide %v1639, %v1636 : tensor<32x75264xf32>
    %v1641 = stablehlo.subtract %v769, %v1640 : tensor<32x75264xf32>
    %v1642 = stablehlo.multiply %v1641, %v1641 : tensor<32x75264xf32>
    %v1643 = stablehlo.reduce(%v1642 init: %v1635) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1644 = stablehlo.broadcast_in_dim %v1643, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1645 = stablehlo.divide %v1644, %v1636 : tensor<32x75264xf32>
    %v1646 = stablehlo.add %v1645, %v1637 : tensor<32x75264xf32>
    %v1647 = stablehlo.rsqrt %v1646 : tensor<32x75264xf32>
    %v1648 = stablehlo.multiply %v1641, %v1647 : tensor<32x75264xf32>
    %v1649 = stablehlo.broadcast_in_dim %s2b8ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1650 = stablehlo.multiply %v1649, %v1634 : tensor<32x75264xf32>
    %v1651 = stablehlo.reduce(%v1650 init: %v1635) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1652 = stablehlo.broadcast_in_dim %v1651, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1653 = stablehlo.multiply %v1648, %v1650 : tensor<32x75264xf32>
    %v1654 = stablehlo.reduce(%v1653 init: %v1635) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1655 = stablehlo.broadcast_in_dim %v1654, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1656 = stablehlo.multiply %v1650, %v1636 : tensor<32x75264xf32>
    %v1657 = stablehlo.subtract %v1656, %v1652 : tensor<32x75264xf32>
    %v1658 = stablehlo.multiply %v1648, %v1655 : tensor<32x75264xf32>
    %v1659 = stablehlo.subtract %v1657, %v1658 : tensor<32x75264xf32>
    %v1660 = stablehlo.divide %v1647, %v1636 : tensor<32x75264xf32>
    %v1661 = stablehlo.multiply %v1660, %v1659 : tensor<32x75264xf32>
    %v1662 = stablehlo.reshape %v1661 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1663 = stablehlo.reverse %s2b8dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1664 = stablehlo.convolution(%v1662, %v1663)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1666 = stablehlo.add %v1665, %v1556 : tensor<32x75264xf32>
    %v1667 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1668 = stablehlo.reshape %v810 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1669 = stablehlo.reshape %v1556 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1670 = stablehlo.multiply %v1668, %v1669 : tensor<32x384x14x14xf32>
    %v1671 = stablehlo.reduce(%v1670 init: %v1667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1672 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1673 = stablehlo.multiply %v1671, %v1672 : tensor<384xf32>
    %v1674 = stablehlo.subtract %s2b8lg, %v1673 : tensor<384xf32>
    %v1675 = stablehlo.reshape %v805 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1676 = stablehlo.reshape %v1601 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1677 = stablehlo.transpose %v1675, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1678 = stablehlo.transpose %v1676, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1679 = stablehlo.convolution(%v1677, %v1678)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1680 = stablehlo.transpose %v1679, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1681 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v1682 = stablehlo.multiply %v1680, %v1681 : tensor<384x1536x1x1xf32>
    %v1683 = stablehlo.subtract %s2b8pW, %v1682 : tensor<384x1536x1x1xf32>
    %v1684 = stablehlo.reshape %v1601 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1685 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1686 = stablehlo.reduce(%v1684 init: %v1685) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1687 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1688 = stablehlo.multiply %v1686, %v1687 : tensor<384xf32>
    %v1689 = stablehlo.subtract %s2b8pb, %v1688 : tensor<384xf32>
    %v1690 = stablehlo.reshape %v787 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1691 = stablehlo.reshape %v1629 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1692 = stablehlo.transpose %v1690, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1693 = stablehlo.transpose %v1691, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1694 = stablehlo.convolution(%v1692, %v1693)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1695 = stablehlo.transpose %v1694, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1696 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v1697 = stablehlo.multiply %v1695, %v1696 : tensor<1536x384x1x1xf32>
    %v1698 = stablehlo.subtract %s2b8eW, %v1697 : tensor<1536x384x1x1xf32>
    %v1699 = stablehlo.reshape %v1629 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1700 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1701 = stablehlo.reduce(%v1699 init: %v1700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1702 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v1703 = stablehlo.multiply %v1701, %v1702 : tensor<1536xf32>
    %v1704 = stablehlo.subtract %s2b8eb, %v1703 : tensor<1536xf32>
    %v1705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1706 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1707 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1708 = stablehlo.reduce(%v769 init: %v1705) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1709 = stablehlo.broadcast_in_dim %v1708, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1710 = stablehlo.divide %v1709, %v1706 : tensor<32x75264xf32>
    %v1711 = stablehlo.subtract %v769, %v1710 : tensor<32x75264xf32>
    %v1712 = stablehlo.multiply %v1711, %v1711 : tensor<32x75264xf32>
    %v1713 = stablehlo.reduce(%v1712 init: %v1705) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1714 = stablehlo.broadcast_in_dim %v1713, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1715 = stablehlo.divide %v1714, %v1706 : tensor<32x75264xf32>
    %v1716 = stablehlo.add %v1715, %v1707 : tensor<32x75264xf32>
    %v1717 = stablehlo.rsqrt %v1716 : tensor<32x75264xf32>
    %v1718 = stablehlo.multiply %v1711, %v1717 : tensor<32x75264xf32>
    %v1719 = stablehlo.multiply %v1634, %v1718 : tensor<32x75264xf32>
    %v1720 = stablehlo.reduce(%v1719 init: %v1705) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1721 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1722 = stablehlo.multiply %v1720, %v1721 : tensor<f32>
    %v1723 = stablehlo.subtract %s2b8ng, %v1722 : tensor<f32>
    %v1724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1725 = stablehlo.reduce(%v1634 init: %v1724) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1726 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1727 = stablehlo.multiply %v1725, %v1726 : tensor<f32>
    %v1728 = stablehlo.subtract %s2b8nbt, %v1727 : tensor<f32>
    %v1729 = stablehlo.reshape %v764 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1730 = stablehlo.reshape %v1661 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1731 = stablehlo.transpose %v1729, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1732 = stablehlo.transpose %v1730, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1733 = stablehlo.convolution(%v1731, %v1732)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v1734 = stablehlo.reshape %v1733 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v1735 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v1736 = stablehlo.multiply %v1734, %v1735 : tensor<384x1x7x7xf32>
    %v1737 = stablehlo.subtract %s2b8dW, %v1736 : tensor<384x1x7x7xf32>
    %v1738 = stablehlo.reshape %v1661 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1740 = stablehlo.reduce(%v1738 init: %v1739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1741 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1742 = stablehlo.multiply %v1740, %v1741 : tensor<384xf32>
    %v1743 = stablehlo.subtract %s2b8db, %v1742 : tensor<384xf32>
    %v1744 = stablehlo.reshape %v1666 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1745 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1746 = stablehlo.multiply %v1744, %v1745 : tensor<32x384x14x14xf32>
    %v1747 = stablehlo.reshape %v1746 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1748 = stablehlo.reshape %v1747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1749 = stablehlo.transpose %s2b7pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1750 = stablehlo.reverse %v1749, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1751 = stablehlo.convolution(%v1748, %v1750)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1752 = stablehlo.reshape %v1751 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1753 = stablehlo.multiply %v741, %v741 : tensor<32x301056xf32>
    %v1754 = stablehlo.multiply %v1753, %v741 : tensor<32x301056xf32>
    %v1755 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1756 = stablehlo.multiply %v1755, %v1754 : tensor<32x301056xf32>
    %v1757 = stablehlo.add %v741, %v1756 : tensor<32x301056xf32>
    %v1758 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1759 = stablehlo.multiply %v1758, %v1757 : tensor<32x301056xf32>
    %v1760 = stablehlo.tanh %v1759 : tensor<32x301056xf32>
    %v1761 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1762 = stablehlo.add %v1761, %v1760 : tensor<32x301056xf32>
    %v1763 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1764 = stablehlo.multiply %v1763, %v1762 : tensor<32x301056xf32>
    %v1765 = stablehlo.multiply %v1760, %v1760 : tensor<32x301056xf32>
    %v1766 = stablehlo.subtract %v1761, %v1765 : tensor<32x301056xf32>
    %v1767 = stablehlo.multiply %v1763, %v741 : tensor<32x301056xf32>
    %v1768 = stablehlo.multiply %v1767, %v1766 : tensor<32x301056xf32>
    %v1769 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1770 = stablehlo.multiply %v1769, %v1753 : tensor<32x301056xf32>
    %v1771 = stablehlo.add %v1761, %v1770 : tensor<32x301056xf32>
    %v1772 = stablehlo.multiply %v1758, %v1771 : tensor<32x301056xf32>
    %v1773 = stablehlo.multiply %v1768, %v1772 : tensor<32x301056xf32>
    %v1774 = stablehlo.add %v1764, %v1773 : tensor<32x301056xf32>
    %v1775 = stablehlo.multiply %v1752, %v1774 : tensor<32x301056xf32>
    %v1776 = stablehlo.reshape %v1775 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1777 = stablehlo.transpose %s2b7eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1778 = stablehlo.reverse %v1777, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1779 = stablehlo.convolution(%v1776, %v1778)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1782 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1783 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1784 = stablehlo.reduce(%v718 init: %v1781) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1785 = stablehlo.broadcast_in_dim %v1784, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1786 = stablehlo.divide %v1785, %v1782 : tensor<32x75264xf32>
    %v1787 = stablehlo.subtract %v718, %v1786 : tensor<32x75264xf32>
    %v1788 = stablehlo.multiply %v1787, %v1787 : tensor<32x75264xf32>
    %v1789 = stablehlo.reduce(%v1788 init: %v1781) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1790 = stablehlo.broadcast_in_dim %v1789, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1791 = stablehlo.divide %v1790, %v1782 : tensor<32x75264xf32>
    %v1792 = stablehlo.add %v1791, %v1783 : tensor<32x75264xf32>
    %v1793 = stablehlo.rsqrt %v1792 : tensor<32x75264xf32>
    %v1794 = stablehlo.multiply %v1787, %v1793 : tensor<32x75264xf32>
    %v1795 = stablehlo.broadcast_in_dim %s2b7ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1796 = stablehlo.multiply %v1795, %v1780 : tensor<32x75264xf32>
    %v1797 = stablehlo.reduce(%v1796 init: %v1781) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1798 = stablehlo.broadcast_in_dim %v1797, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1799 = stablehlo.multiply %v1794, %v1796 : tensor<32x75264xf32>
    %v1800 = stablehlo.reduce(%v1799 init: %v1781) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1801 = stablehlo.broadcast_in_dim %v1800, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1802 = stablehlo.multiply %v1796, %v1782 : tensor<32x75264xf32>
    %v1803 = stablehlo.subtract %v1802, %v1798 : tensor<32x75264xf32>
    %v1804 = stablehlo.multiply %v1794, %v1801 : tensor<32x75264xf32>
    %v1805 = stablehlo.subtract %v1803, %v1804 : tensor<32x75264xf32>
    %v1806 = stablehlo.divide %v1793, %v1782 : tensor<32x75264xf32>
    %v1807 = stablehlo.multiply %v1806, %v1805 : tensor<32x75264xf32>
    %v1808 = stablehlo.reshape %v1807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1809 = stablehlo.reverse %s2b7dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1810 = stablehlo.convolution(%v1808, %v1809)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1812 = stablehlo.add %v1811, %v1666 : tensor<32x75264xf32>
    %v1813 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1814 = stablehlo.reshape %v759 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1815 = stablehlo.reshape %v1666 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1816 = stablehlo.multiply %v1814, %v1815 : tensor<32x384x14x14xf32>
    %v1817 = stablehlo.reduce(%v1816 init: %v1813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1818 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1819 = stablehlo.multiply %v1817, %v1818 : tensor<384xf32>
    %v1820 = stablehlo.subtract %s2b7lg, %v1819 : tensor<384xf32>
    %v1821 = stablehlo.reshape %v754 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1822 = stablehlo.reshape %v1747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1823 = stablehlo.transpose %v1821, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1824 = stablehlo.transpose %v1822, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1825 = stablehlo.convolution(%v1823, %v1824)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1826 = stablehlo.transpose %v1825, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1827 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v1828 = stablehlo.multiply %v1826, %v1827 : tensor<384x1536x1x1xf32>
    %v1829 = stablehlo.subtract %s2b7pW, %v1828 : tensor<384x1536x1x1xf32>
    %v1830 = stablehlo.reshape %v1747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1832 = stablehlo.reduce(%v1830 init: %v1831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1833 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1834 = stablehlo.multiply %v1832, %v1833 : tensor<384xf32>
    %v1835 = stablehlo.subtract %s2b7pb, %v1834 : tensor<384xf32>
    %v1836 = stablehlo.reshape %v736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1837 = stablehlo.reshape %v1775 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1838 = stablehlo.transpose %v1836, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1839 = stablehlo.transpose %v1837, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1840 = stablehlo.convolution(%v1838, %v1839)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1841 = stablehlo.transpose %v1840, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1842 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v1843 = stablehlo.multiply %v1841, %v1842 : tensor<1536x384x1x1xf32>
    %v1844 = stablehlo.subtract %s2b7eW, %v1843 : tensor<1536x384x1x1xf32>
    %v1845 = stablehlo.reshape %v1775 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1847 = stablehlo.reduce(%v1845 init: %v1846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1848 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v1849 = stablehlo.multiply %v1847, %v1848 : tensor<1536xf32>
    %v1850 = stablehlo.subtract %s2b7eb, %v1849 : tensor<1536xf32>
    %v1851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1852 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1853 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1854 = stablehlo.reduce(%v718 init: %v1851) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1855 = stablehlo.broadcast_in_dim %v1854, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1856 = stablehlo.divide %v1855, %v1852 : tensor<32x75264xf32>
    %v1857 = stablehlo.subtract %v718, %v1856 : tensor<32x75264xf32>
    %v1858 = stablehlo.multiply %v1857, %v1857 : tensor<32x75264xf32>
    %v1859 = stablehlo.reduce(%v1858 init: %v1851) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1860 = stablehlo.broadcast_in_dim %v1859, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1861 = stablehlo.divide %v1860, %v1852 : tensor<32x75264xf32>
    %v1862 = stablehlo.add %v1861, %v1853 : tensor<32x75264xf32>
    %v1863 = stablehlo.rsqrt %v1862 : tensor<32x75264xf32>
    %v1864 = stablehlo.multiply %v1857, %v1863 : tensor<32x75264xf32>
    %v1865 = stablehlo.multiply %v1780, %v1864 : tensor<32x75264xf32>
    %v1866 = stablehlo.reduce(%v1865 init: %v1851) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1867 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1868 = stablehlo.multiply %v1866, %v1867 : tensor<f32>
    %v1869 = stablehlo.subtract %s2b7ng, %v1868 : tensor<f32>
    %v1870 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1871 = stablehlo.reduce(%v1780 init: %v1870) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1872 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1873 = stablehlo.multiply %v1871, %v1872 : tensor<f32>
    %v1874 = stablehlo.subtract %s2b7nbt, %v1873 : tensor<f32>
    %v1875 = stablehlo.reshape %v713 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1876 = stablehlo.reshape %v1807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1877 = stablehlo.transpose %v1875, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1878 = stablehlo.transpose %v1876, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1879 = stablehlo.convolution(%v1877, %v1878)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v1880 = stablehlo.reshape %v1879 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v1881 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v1882 = stablehlo.multiply %v1880, %v1881 : tensor<384x1x7x7xf32>
    %v1883 = stablehlo.subtract %s2b7dW, %v1882 : tensor<384x1x7x7xf32>
    %v1884 = stablehlo.reshape %v1807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1886 = stablehlo.reduce(%v1884 init: %v1885) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1887 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1888 = stablehlo.multiply %v1886, %v1887 : tensor<384xf32>
    %v1889 = stablehlo.subtract %s2b7db, %v1888 : tensor<384xf32>
    %v1890 = stablehlo.reshape %v1812 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1891 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1892 = stablehlo.multiply %v1890, %v1891 : tensor<32x384x14x14xf32>
    %v1893 = stablehlo.reshape %v1892 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1894 = stablehlo.reshape %v1893 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1895 = stablehlo.transpose %s2b6pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1896 = stablehlo.reverse %v1895, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1897 = stablehlo.convolution(%v1894, %v1896)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1898 = stablehlo.reshape %v1897 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1899 = stablehlo.multiply %v690, %v690 : tensor<32x301056xf32>
    %v1900 = stablehlo.multiply %v1899, %v690 : tensor<32x301056xf32>
    %v1901 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1902 = stablehlo.multiply %v1901, %v1900 : tensor<32x301056xf32>
    %v1903 = stablehlo.add %v690, %v1902 : tensor<32x301056xf32>
    %v1904 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1905 = stablehlo.multiply %v1904, %v1903 : tensor<32x301056xf32>
    %v1906 = stablehlo.tanh %v1905 : tensor<32x301056xf32>
    %v1907 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1908 = stablehlo.add %v1907, %v1906 : tensor<32x301056xf32>
    %v1909 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1910 = stablehlo.multiply %v1909, %v1908 : tensor<32x301056xf32>
    %v1911 = stablehlo.multiply %v1906, %v1906 : tensor<32x301056xf32>
    %v1912 = stablehlo.subtract %v1907, %v1911 : tensor<32x301056xf32>
    %v1913 = stablehlo.multiply %v1909, %v690 : tensor<32x301056xf32>
    %v1914 = stablehlo.multiply %v1913, %v1912 : tensor<32x301056xf32>
    %v1915 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1916 = stablehlo.multiply %v1915, %v1899 : tensor<32x301056xf32>
    %v1917 = stablehlo.add %v1907, %v1916 : tensor<32x301056xf32>
    %v1918 = stablehlo.multiply %v1904, %v1917 : tensor<32x301056xf32>
    %v1919 = stablehlo.multiply %v1914, %v1918 : tensor<32x301056xf32>
    %v1920 = stablehlo.add %v1910, %v1919 : tensor<32x301056xf32>
    %v1921 = stablehlo.multiply %v1898, %v1920 : tensor<32x301056xf32>
    %v1922 = stablehlo.reshape %v1921 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1923 = stablehlo.transpose %s2b6eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1924 = stablehlo.reverse %v1923, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1925 = stablehlo.convolution(%v1922, %v1924)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1928 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1929 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1930 = stablehlo.reduce(%v667 init: %v1927) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1931 = stablehlo.broadcast_in_dim %v1930, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1932 = stablehlo.divide %v1931, %v1928 : tensor<32x75264xf32>
    %v1933 = stablehlo.subtract %v667, %v1932 : tensor<32x75264xf32>
    %v1934 = stablehlo.multiply %v1933, %v1933 : tensor<32x75264xf32>
    %v1935 = stablehlo.reduce(%v1934 init: %v1927) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1936 = stablehlo.broadcast_in_dim %v1935, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1937 = stablehlo.divide %v1936, %v1928 : tensor<32x75264xf32>
    %v1938 = stablehlo.add %v1937, %v1929 : tensor<32x75264xf32>
    %v1939 = stablehlo.rsqrt %v1938 : tensor<32x75264xf32>
    %v1940 = stablehlo.multiply %v1933, %v1939 : tensor<32x75264xf32>
    %v1941 = stablehlo.broadcast_in_dim %s2b6ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1942 = stablehlo.multiply %v1941, %v1926 : tensor<32x75264xf32>
    %v1943 = stablehlo.reduce(%v1942 init: %v1927) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1944 = stablehlo.broadcast_in_dim %v1943, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1945 = stablehlo.multiply %v1940, %v1942 : tensor<32x75264xf32>
    %v1946 = stablehlo.reduce(%v1945 init: %v1927) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1947 = stablehlo.broadcast_in_dim %v1946, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1948 = stablehlo.multiply %v1942, %v1928 : tensor<32x75264xf32>
    %v1949 = stablehlo.subtract %v1948, %v1944 : tensor<32x75264xf32>
    %v1950 = stablehlo.multiply %v1940, %v1947 : tensor<32x75264xf32>
    %v1951 = stablehlo.subtract %v1949, %v1950 : tensor<32x75264xf32>
    %v1952 = stablehlo.divide %v1939, %v1928 : tensor<32x75264xf32>
    %v1953 = stablehlo.multiply %v1952, %v1951 : tensor<32x75264xf32>
    %v1954 = stablehlo.reshape %v1953 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1955 = stablehlo.reverse %s2b6dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1956 = stablehlo.convolution(%v1954, %v1955)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1957 = stablehlo.reshape %v1956 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1958 = stablehlo.add %v1957, %v1812 : tensor<32x75264xf32>
    %v1959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1960 = stablehlo.reshape %v708 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1961 = stablehlo.reshape %v1812 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1962 = stablehlo.multiply %v1960, %v1961 : tensor<32x384x14x14xf32>
    %v1963 = stablehlo.reduce(%v1962 init: %v1959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1964 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1965 = stablehlo.multiply %v1963, %v1964 : tensor<384xf32>
    %v1966 = stablehlo.subtract %s2b6lg, %v1965 : tensor<384xf32>
    %v1967 = stablehlo.reshape %v703 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1968 = stablehlo.reshape %v1893 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1969 = stablehlo.transpose %v1967, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1970 = stablehlo.transpose %v1968, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1971 = stablehlo.convolution(%v1969, %v1970)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1972 = stablehlo.transpose %v1971, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1973 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v1974 = stablehlo.multiply %v1972, %v1973 : tensor<384x1536x1x1xf32>
    %v1975 = stablehlo.subtract %s2b6pW, %v1974 : tensor<384x1536x1x1xf32>
    %v1976 = stablehlo.reshape %v1893 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1978 = stablehlo.reduce(%v1976 init: %v1977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1979 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1980 = stablehlo.multiply %v1978, %v1979 : tensor<384xf32>
    %v1981 = stablehlo.subtract %s2b6pb, %v1980 : tensor<384xf32>
    %v1982 = stablehlo.reshape %v685 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1983 = stablehlo.reshape %v1921 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1984 = stablehlo.transpose %v1982, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1985 = stablehlo.transpose %v1983, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1986 = stablehlo.convolution(%v1984, %v1985)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1987 = stablehlo.transpose %v1986, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1988 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v1989 = stablehlo.multiply %v1987, %v1988 : tensor<1536x384x1x1xf32>
    %v1990 = stablehlo.subtract %s2b6eW, %v1989 : tensor<1536x384x1x1xf32>
    %v1991 = stablehlo.reshape %v1921 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1993 = stablehlo.reduce(%v1991 init: %v1992) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1994 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v1995 = stablehlo.multiply %v1993, %v1994 : tensor<1536xf32>
    %v1996 = stablehlo.subtract %s2b6eb, %v1995 : tensor<1536xf32>
    %v1997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1998 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1999 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2000 = stablehlo.reduce(%v667 init: %v1997) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2001 = stablehlo.broadcast_in_dim %v2000, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2002 = stablehlo.divide %v2001, %v1998 : tensor<32x75264xf32>
    %v2003 = stablehlo.subtract %v667, %v2002 : tensor<32x75264xf32>
    %v2004 = stablehlo.multiply %v2003, %v2003 : tensor<32x75264xf32>
    %v2005 = stablehlo.reduce(%v2004 init: %v1997) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2006 = stablehlo.broadcast_in_dim %v2005, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2007 = stablehlo.divide %v2006, %v1998 : tensor<32x75264xf32>
    %v2008 = stablehlo.add %v2007, %v1999 : tensor<32x75264xf32>
    %v2009 = stablehlo.rsqrt %v2008 : tensor<32x75264xf32>
    %v2010 = stablehlo.multiply %v2003, %v2009 : tensor<32x75264xf32>
    %v2011 = stablehlo.multiply %v1926, %v2010 : tensor<32x75264xf32>
    %v2012 = stablehlo.reduce(%v2011 init: %v1997) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2013 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2014 = stablehlo.multiply %v2012, %v2013 : tensor<f32>
    %v2015 = stablehlo.subtract %s2b6ng, %v2014 : tensor<f32>
    %v2016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2017 = stablehlo.reduce(%v1926 init: %v2016) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2018 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2019 = stablehlo.multiply %v2017, %v2018 : tensor<f32>
    %v2020 = stablehlo.subtract %s2b6nbt, %v2019 : tensor<f32>
    %v2021 = stablehlo.reshape %v662 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2022 = stablehlo.reshape %v1953 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2023 = stablehlo.transpose %v2021, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2024 = stablehlo.transpose %v2022, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2025 = stablehlo.convolution(%v2023, %v2024)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2026 = stablehlo.reshape %v2025 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2027 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2028 = stablehlo.multiply %v2026, %v2027 : tensor<384x1x7x7xf32>
    %v2029 = stablehlo.subtract %s2b6dW, %v2028 : tensor<384x1x7x7xf32>
    %v2030 = stablehlo.reshape %v1953 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2031 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2032 = stablehlo.reduce(%v2030 init: %v2031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2033 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2034 = stablehlo.multiply %v2032, %v2033 : tensor<384xf32>
    %v2035 = stablehlo.subtract %s2b6db, %v2034 : tensor<384xf32>
    %v2036 = stablehlo.reshape %v1958 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2037 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2038 = stablehlo.multiply %v2036, %v2037 : tensor<32x384x14x14xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2040 = stablehlo.reshape %v2039 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2041 = stablehlo.transpose %s2b5pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2042 = stablehlo.reverse %v2041, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2043 = stablehlo.convolution(%v2040, %v2042)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2044 = stablehlo.reshape %v2043 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2045 = stablehlo.multiply %v639, %v639 : tensor<32x301056xf32>
    %v2046 = stablehlo.multiply %v2045, %v639 : tensor<32x301056xf32>
    %v2047 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2048 = stablehlo.multiply %v2047, %v2046 : tensor<32x301056xf32>
    %v2049 = stablehlo.add %v639, %v2048 : tensor<32x301056xf32>
    %v2050 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2051 = stablehlo.multiply %v2050, %v2049 : tensor<32x301056xf32>
    %v2052 = stablehlo.tanh %v2051 : tensor<32x301056xf32>
    %v2053 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2054 = stablehlo.add %v2053, %v2052 : tensor<32x301056xf32>
    %v2055 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2056 = stablehlo.multiply %v2055, %v2054 : tensor<32x301056xf32>
    %v2057 = stablehlo.multiply %v2052, %v2052 : tensor<32x301056xf32>
    %v2058 = stablehlo.subtract %v2053, %v2057 : tensor<32x301056xf32>
    %v2059 = stablehlo.multiply %v2055, %v639 : tensor<32x301056xf32>
    %v2060 = stablehlo.multiply %v2059, %v2058 : tensor<32x301056xf32>
    %v2061 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2062 = stablehlo.multiply %v2061, %v2045 : tensor<32x301056xf32>
    %v2063 = stablehlo.add %v2053, %v2062 : tensor<32x301056xf32>
    %v2064 = stablehlo.multiply %v2050, %v2063 : tensor<32x301056xf32>
    %v2065 = stablehlo.multiply %v2060, %v2064 : tensor<32x301056xf32>
    %v2066 = stablehlo.add %v2056, %v2065 : tensor<32x301056xf32>
    %v2067 = stablehlo.multiply %v2044, %v2066 : tensor<32x301056xf32>
    %v2068 = stablehlo.reshape %v2067 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2069 = stablehlo.transpose %s2b5eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2070 = stablehlo.reverse %v2069, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2071 = stablehlo.convolution(%v2068, %v2070)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2072 = stablehlo.reshape %v2071 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2074 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2075 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2076 = stablehlo.reduce(%v616 init: %v2073) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2077 = stablehlo.broadcast_in_dim %v2076, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2078 = stablehlo.divide %v2077, %v2074 : tensor<32x75264xf32>
    %v2079 = stablehlo.subtract %v616, %v2078 : tensor<32x75264xf32>
    %v2080 = stablehlo.multiply %v2079, %v2079 : tensor<32x75264xf32>
    %v2081 = stablehlo.reduce(%v2080 init: %v2073) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2082 = stablehlo.broadcast_in_dim %v2081, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2083 = stablehlo.divide %v2082, %v2074 : tensor<32x75264xf32>
    %v2084 = stablehlo.add %v2083, %v2075 : tensor<32x75264xf32>
    %v2085 = stablehlo.rsqrt %v2084 : tensor<32x75264xf32>
    %v2086 = stablehlo.multiply %v2079, %v2085 : tensor<32x75264xf32>
    %v2087 = stablehlo.broadcast_in_dim %s2b5ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2088 = stablehlo.multiply %v2087, %v2072 : tensor<32x75264xf32>
    %v2089 = stablehlo.reduce(%v2088 init: %v2073) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2090 = stablehlo.broadcast_in_dim %v2089, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2091 = stablehlo.multiply %v2086, %v2088 : tensor<32x75264xf32>
    %v2092 = stablehlo.reduce(%v2091 init: %v2073) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2093 = stablehlo.broadcast_in_dim %v2092, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2094 = stablehlo.multiply %v2088, %v2074 : tensor<32x75264xf32>
    %v2095 = stablehlo.subtract %v2094, %v2090 : tensor<32x75264xf32>
    %v2096 = stablehlo.multiply %v2086, %v2093 : tensor<32x75264xf32>
    %v2097 = stablehlo.subtract %v2095, %v2096 : tensor<32x75264xf32>
    %v2098 = stablehlo.divide %v2085, %v2074 : tensor<32x75264xf32>
    %v2099 = stablehlo.multiply %v2098, %v2097 : tensor<32x75264xf32>
    %v2100 = stablehlo.reshape %v2099 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2101 = stablehlo.reverse %s2b5dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2102 = stablehlo.convolution(%v2100, %v2101)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2103 = stablehlo.reshape %v2102 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2104 = stablehlo.add %v2103, %v1958 : tensor<32x75264xf32>
    %v2105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2106 = stablehlo.reshape %v657 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2107 = stablehlo.reshape %v1958 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2108 = stablehlo.multiply %v2106, %v2107 : tensor<32x384x14x14xf32>
    %v2109 = stablehlo.reduce(%v2108 init: %v2105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2110 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2111 = stablehlo.multiply %v2109, %v2110 : tensor<384xf32>
    %v2112 = stablehlo.subtract %s2b5lg, %v2111 : tensor<384xf32>
    %v2113 = stablehlo.reshape %v652 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2114 = stablehlo.reshape %v2039 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2115 = stablehlo.transpose %v2113, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2116 = stablehlo.transpose %v2114, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2117 = stablehlo.convolution(%v2115, %v2116)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2118 = stablehlo.transpose %v2117, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2119 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2120 = stablehlo.multiply %v2118, %v2119 : tensor<384x1536x1x1xf32>
    %v2121 = stablehlo.subtract %s2b5pW, %v2120 : tensor<384x1536x1x1xf32>
    %v2122 = stablehlo.reshape %v2039 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2124 = stablehlo.reduce(%v2122 init: %v2123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2125 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2126 = stablehlo.multiply %v2124, %v2125 : tensor<384xf32>
    %v2127 = stablehlo.subtract %s2b5pb, %v2126 : tensor<384xf32>
    %v2128 = stablehlo.reshape %v634 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2129 = stablehlo.reshape %v2067 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2130 = stablehlo.transpose %v2128, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2131 = stablehlo.transpose %v2129, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2132 = stablehlo.convolution(%v2130, %v2131)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2133 = stablehlo.transpose %v2132, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2134 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2135 = stablehlo.multiply %v2133, %v2134 : tensor<1536x384x1x1xf32>
    %v2136 = stablehlo.subtract %s2b5eW, %v2135 : tensor<1536x384x1x1xf32>
    %v2137 = stablehlo.reshape %v2067 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2139 = stablehlo.reduce(%v2137 init: %v2138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2140 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2141 = stablehlo.multiply %v2139, %v2140 : tensor<1536xf32>
    %v2142 = stablehlo.subtract %s2b5eb, %v2141 : tensor<1536xf32>
    %v2143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2144 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2145 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2146 = stablehlo.reduce(%v616 init: %v2143) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2147 = stablehlo.broadcast_in_dim %v2146, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2148 = stablehlo.divide %v2147, %v2144 : tensor<32x75264xf32>
    %v2149 = stablehlo.subtract %v616, %v2148 : tensor<32x75264xf32>
    %v2150 = stablehlo.multiply %v2149, %v2149 : tensor<32x75264xf32>
    %v2151 = stablehlo.reduce(%v2150 init: %v2143) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2152 = stablehlo.broadcast_in_dim %v2151, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2153 = stablehlo.divide %v2152, %v2144 : tensor<32x75264xf32>
    %v2154 = stablehlo.add %v2153, %v2145 : tensor<32x75264xf32>
    %v2155 = stablehlo.rsqrt %v2154 : tensor<32x75264xf32>
    %v2156 = stablehlo.multiply %v2149, %v2155 : tensor<32x75264xf32>
    %v2157 = stablehlo.multiply %v2072, %v2156 : tensor<32x75264xf32>
    %v2158 = stablehlo.reduce(%v2157 init: %v2143) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2159 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2160 = stablehlo.multiply %v2158, %v2159 : tensor<f32>
    %v2161 = stablehlo.subtract %s2b5ng, %v2160 : tensor<f32>
    %v2162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2163 = stablehlo.reduce(%v2072 init: %v2162) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2164 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2165 = stablehlo.multiply %v2163, %v2164 : tensor<f32>
    %v2166 = stablehlo.subtract %s2b5nbt, %v2165 : tensor<f32>
    %v2167 = stablehlo.reshape %v611 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2168 = stablehlo.reshape %v2099 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2169 = stablehlo.transpose %v2167, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2170 = stablehlo.transpose %v2168, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2171 = stablehlo.convolution(%v2169, %v2170)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2172 = stablehlo.reshape %v2171 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2173 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2174 = stablehlo.multiply %v2172, %v2173 : tensor<384x1x7x7xf32>
    %v2175 = stablehlo.subtract %s2b5dW, %v2174 : tensor<384x1x7x7xf32>
    %v2176 = stablehlo.reshape %v2099 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2178 = stablehlo.reduce(%v2176 init: %v2177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2179 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2180 = stablehlo.multiply %v2178, %v2179 : tensor<384xf32>
    %v2181 = stablehlo.subtract %s2b5db, %v2180 : tensor<384xf32>
    %v2182 = stablehlo.reshape %v2104 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2183 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2184 = stablehlo.multiply %v2182, %v2183 : tensor<32x384x14x14xf32>
    %v2185 = stablehlo.reshape %v2184 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2186 = stablehlo.reshape %v2185 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2187 = stablehlo.transpose %s2b4pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2188 = stablehlo.reverse %v2187, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2189 = stablehlo.convolution(%v2186, %v2188)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2190 = stablehlo.reshape %v2189 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2191 = stablehlo.multiply %v588, %v588 : tensor<32x301056xf32>
    %v2192 = stablehlo.multiply %v2191, %v588 : tensor<32x301056xf32>
    %v2193 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2194 = stablehlo.multiply %v2193, %v2192 : tensor<32x301056xf32>
    %v2195 = stablehlo.add %v588, %v2194 : tensor<32x301056xf32>
    %v2196 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2197 = stablehlo.multiply %v2196, %v2195 : tensor<32x301056xf32>
    %v2198 = stablehlo.tanh %v2197 : tensor<32x301056xf32>
    %v2199 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2200 = stablehlo.add %v2199, %v2198 : tensor<32x301056xf32>
    %v2201 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2202 = stablehlo.multiply %v2201, %v2200 : tensor<32x301056xf32>
    %v2203 = stablehlo.multiply %v2198, %v2198 : tensor<32x301056xf32>
    %v2204 = stablehlo.subtract %v2199, %v2203 : tensor<32x301056xf32>
    %v2205 = stablehlo.multiply %v2201, %v588 : tensor<32x301056xf32>
    %v2206 = stablehlo.multiply %v2205, %v2204 : tensor<32x301056xf32>
    %v2207 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2208 = stablehlo.multiply %v2207, %v2191 : tensor<32x301056xf32>
    %v2209 = stablehlo.add %v2199, %v2208 : tensor<32x301056xf32>
    %v2210 = stablehlo.multiply %v2196, %v2209 : tensor<32x301056xf32>
    %v2211 = stablehlo.multiply %v2206, %v2210 : tensor<32x301056xf32>
    %v2212 = stablehlo.add %v2202, %v2211 : tensor<32x301056xf32>
    %v2213 = stablehlo.multiply %v2190, %v2212 : tensor<32x301056xf32>
    %v2214 = stablehlo.reshape %v2213 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2215 = stablehlo.transpose %s2b4eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2216 = stablehlo.reverse %v2215, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2217 = stablehlo.convolution(%v2214, %v2216)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2218 = stablehlo.reshape %v2217 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2220 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2221 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2222 = stablehlo.reduce(%v565 init: %v2219) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2223 = stablehlo.broadcast_in_dim %v2222, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2224 = stablehlo.divide %v2223, %v2220 : tensor<32x75264xf32>
    %v2225 = stablehlo.subtract %v565, %v2224 : tensor<32x75264xf32>
    %v2226 = stablehlo.multiply %v2225, %v2225 : tensor<32x75264xf32>
    %v2227 = stablehlo.reduce(%v2226 init: %v2219) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2228 = stablehlo.broadcast_in_dim %v2227, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2229 = stablehlo.divide %v2228, %v2220 : tensor<32x75264xf32>
    %v2230 = stablehlo.add %v2229, %v2221 : tensor<32x75264xf32>
    %v2231 = stablehlo.rsqrt %v2230 : tensor<32x75264xf32>
    %v2232 = stablehlo.multiply %v2225, %v2231 : tensor<32x75264xf32>
    %v2233 = stablehlo.broadcast_in_dim %s2b4ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2234 = stablehlo.multiply %v2233, %v2218 : tensor<32x75264xf32>
    %v2235 = stablehlo.reduce(%v2234 init: %v2219) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2236 = stablehlo.broadcast_in_dim %v2235, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2237 = stablehlo.multiply %v2232, %v2234 : tensor<32x75264xf32>
    %v2238 = stablehlo.reduce(%v2237 init: %v2219) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2239 = stablehlo.broadcast_in_dim %v2238, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2240 = stablehlo.multiply %v2234, %v2220 : tensor<32x75264xf32>
    %v2241 = stablehlo.subtract %v2240, %v2236 : tensor<32x75264xf32>
    %v2242 = stablehlo.multiply %v2232, %v2239 : tensor<32x75264xf32>
    %v2243 = stablehlo.subtract %v2241, %v2242 : tensor<32x75264xf32>
    %v2244 = stablehlo.divide %v2231, %v2220 : tensor<32x75264xf32>
    %v2245 = stablehlo.multiply %v2244, %v2243 : tensor<32x75264xf32>
    %v2246 = stablehlo.reshape %v2245 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2247 = stablehlo.reverse %s2b4dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2248 = stablehlo.convolution(%v2246, %v2247)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2249 = stablehlo.reshape %v2248 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2250 = stablehlo.add %v2249, %v2104 : tensor<32x75264xf32>
    %v2251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2252 = stablehlo.reshape %v606 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2253 = stablehlo.reshape %v2104 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2254 = stablehlo.multiply %v2252, %v2253 : tensor<32x384x14x14xf32>
    %v2255 = stablehlo.reduce(%v2254 init: %v2251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2256 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2257 = stablehlo.multiply %v2255, %v2256 : tensor<384xf32>
    %v2258 = stablehlo.subtract %s2b4lg, %v2257 : tensor<384xf32>
    %v2259 = stablehlo.reshape %v601 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2260 = stablehlo.reshape %v2185 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2261 = stablehlo.transpose %v2259, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2262 = stablehlo.transpose %v2260, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2263 = stablehlo.convolution(%v2261, %v2262)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2264 = stablehlo.transpose %v2263, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2265 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2266 = stablehlo.multiply %v2264, %v2265 : tensor<384x1536x1x1xf32>
    %v2267 = stablehlo.subtract %s2b4pW, %v2266 : tensor<384x1536x1x1xf32>
    %v2268 = stablehlo.reshape %v2185 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2270 = stablehlo.reduce(%v2268 init: %v2269) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2271 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2272 = stablehlo.multiply %v2270, %v2271 : tensor<384xf32>
    %v2273 = stablehlo.subtract %s2b4pb, %v2272 : tensor<384xf32>
    %v2274 = stablehlo.reshape %v583 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2275 = stablehlo.reshape %v2213 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2276 = stablehlo.transpose %v2274, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2277 = stablehlo.transpose %v2275, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2278 = stablehlo.convolution(%v2276, %v2277)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2279 = stablehlo.transpose %v2278, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2280 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2281 = stablehlo.multiply %v2279, %v2280 : tensor<1536x384x1x1xf32>
    %v2282 = stablehlo.subtract %s2b4eW, %v2281 : tensor<1536x384x1x1xf32>
    %v2283 = stablehlo.reshape %v2213 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2285 = stablehlo.reduce(%v2283 init: %v2284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2286 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2287 = stablehlo.multiply %v2285, %v2286 : tensor<1536xf32>
    %v2288 = stablehlo.subtract %s2b4eb, %v2287 : tensor<1536xf32>
    %v2289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2290 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2291 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2292 = stablehlo.reduce(%v565 init: %v2289) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2293 = stablehlo.broadcast_in_dim %v2292, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2294 = stablehlo.divide %v2293, %v2290 : tensor<32x75264xf32>
    %v2295 = stablehlo.subtract %v565, %v2294 : tensor<32x75264xf32>
    %v2296 = stablehlo.multiply %v2295, %v2295 : tensor<32x75264xf32>
    %v2297 = stablehlo.reduce(%v2296 init: %v2289) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2298 = stablehlo.broadcast_in_dim %v2297, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2299 = stablehlo.divide %v2298, %v2290 : tensor<32x75264xf32>
    %v2300 = stablehlo.add %v2299, %v2291 : tensor<32x75264xf32>
    %v2301 = stablehlo.rsqrt %v2300 : tensor<32x75264xf32>
    %v2302 = stablehlo.multiply %v2295, %v2301 : tensor<32x75264xf32>
    %v2303 = stablehlo.multiply %v2218, %v2302 : tensor<32x75264xf32>
    %v2304 = stablehlo.reduce(%v2303 init: %v2289) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2305 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2306 = stablehlo.multiply %v2304, %v2305 : tensor<f32>
    %v2307 = stablehlo.subtract %s2b4ng, %v2306 : tensor<f32>
    %v2308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2309 = stablehlo.reduce(%v2218 init: %v2308) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2310 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2311 = stablehlo.multiply %v2309, %v2310 : tensor<f32>
    %v2312 = stablehlo.subtract %s2b4nbt, %v2311 : tensor<f32>
    %v2313 = stablehlo.reshape %v560 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2314 = stablehlo.reshape %v2245 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2315 = stablehlo.transpose %v2313, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2316 = stablehlo.transpose %v2314, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2317 = stablehlo.convolution(%v2315, %v2316)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2318 = stablehlo.reshape %v2317 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2319 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2320 = stablehlo.multiply %v2318, %v2319 : tensor<384x1x7x7xf32>
    %v2321 = stablehlo.subtract %s2b4dW, %v2320 : tensor<384x1x7x7xf32>
    %v2322 = stablehlo.reshape %v2245 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2324 = stablehlo.reduce(%v2322 init: %v2323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2325 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2326 = stablehlo.multiply %v2324, %v2325 : tensor<384xf32>
    %v2327 = stablehlo.subtract %s2b4db, %v2326 : tensor<384xf32>
    %v2328 = stablehlo.reshape %v2250 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2329 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2330 = stablehlo.multiply %v2328, %v2329 : tensor<32x384x14x14xf32>
    %v2331 = stablehlo.reshape %v2330 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2332 = stablehlo.reshape %v2331 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2333 = stablehlo.transpose %s2b3pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2334 = stablehlo.reverse %v2333, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2335 = stablehlo.convolution(%v2332, %v2334)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2336 = stablehlo.reshape %v2335 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2337 = stablehlo.multiply %v537, %v537 : tensor<32x301056xf32>
    %v2338 = stablehlo.multiply %v2337, %v537 : tensor<32x301056xf32>
    %v2339 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2340 = stablehlo.multiply %v2339, %v2338 : tensor<32x301056xf32>
    %v2341 = stablehlo.add %v537, %v2340 : tensor<32x301056xf32>
    %v2342 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2343 = stablehlo.multiply %v2342, %v2341 : tensor<32x301056xf32>
    %v2344 = stablehlo.tanh %v2343 : tensor<32x301056xf32>
    %v2345 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2346 = stablehlo.add %v2345, %v2344 : tensor<32x301056xf32>
    %v2347 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2348 = stablehlo.multiply %v2347, %v2346 : tensor<32x301056xf32>
    %v2349 = stablehlo.multiply %v2344, %v2344 : tensor<32x301056xf32>
    %v2350 = stablehlo.subtract %v2345, %v2349 : tensor<32x301056xf32>
    %v2351 = stablehlo.multiply %v2347, %v537 : tensor<32x301056xf32>
    %v2352 = stablehlo.multiply %v2351, %v2350 : tensor<32x301056xf32>
    %v2353 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2354 = stablehlo.multiply %v2353, %v2337 : tensor<32x301056xf32>
    %v2355 = stablehlo.add %v2345, %v2354 : tensor<32x301056xf32>
    %v2356 = stablehlo.multiply %v2342, %v2355 : tensor<32x301056xf32>
    %v2357 = stablehlo.multiply %v2352, %v2356 : tensor<32x301056xf32>
    %v2358 = stablehlo.add %v2348, %v2357 : tensor<32x301056xf32>
    %v2359 = stablehlo.multiply %v2336, %v2358 : tensor<32x301056xf32>
    %v2360 = stablehlo.reshape %v2359 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2361 = stablehlo.transpose %s2b3eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2362 = stablehlo.reverse %v2361, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2363 = stablehlo.convolution(%v2360, %v2362)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2364 = stablehlo.reshape %v2363 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2366 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2367 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2368 = stablehlo.reduce(%v514 init: %v2365) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2369 = stablehlo.broadcast_in_dim %v2368, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2370 = stablehlo.divide %v2369, %v2366 : tensor<32x75264xf32>
    %v2371 = stablehlo.subtract %v514, %v2370 : tensor<32x75264xf32>
    %v2372 = stablehlo.multiply %v2371, %v2371 : tensor<32x75264xf32>
    %v2373 = stablehlo.reduce(%v2372 init: %v2365) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2374 = stablehlo.broadcast_in_dim %v2373, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2375 = stablehlo.divide %v2374, %v2366 : tensor<32x75264xf32>
    %v2376 = stablehlo.add %v2375, %v2367 : tensor<32x75264xf32>
    %v2377 = stablehlo.rsqrt %v2376 : tensor<32x75264xf32>
    %v2378 = stablehlo.multiply %v2371, %v2377 : tensor<32x75264xf32>
    %v2379 = stablehlo.broadcast_in_dim %s2b3ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2380 = stablehlo.multiply %v2379, %v2364 : tensor<32x75264xf32>
    %v2381 = stablehlo.reduce(%v2380 init: %v2365) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2382 = stablehlo.broadcast_in_dim %v2381, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2383 = stablehlo.multiply %v2378, %v2380 : tensor<32x75264xf32>
    %v2384 = stablehlo.reduce(%v2383 init: %v2365) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2385 = stablehlo.broadcast_in_dim %v2384, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2386 = stablehlo.multiply %v2380, %v2366 : tensor<32x75264xf32>
    %v2387 = stablehlo.subtract %v2386, %v2382 : tensor<32x75264xf32>
    %v2388 = stablehlo.multiply %v2378, %v2385 : tensor<32x75264xf32>
    %v2389 = stablehlo.subtract %v2387, %v2388 : tensor<32x75264xf32>
    %v2390 = stablehlo.divide %v2377, %v2366 : tensor<32x75264xf32>
    %v2391 = stablehlo.multiply %v2390, %v2389 : tensor<32x75264xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2393 = stablehlo.reverse %s2b3dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2394 = stablehlo.convolution(%v2392, %v2393)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2396 = stablehlo.add %v2395, %v2250 : tensor<32x75264xf32>
    %v2397 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2398 = stablehlo.reshape %v555 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2399 = stablehlo.reshape %v2250 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2400 = stablehlo.multiply %v2398, %v2399 : tensor<32x384x14x14xf32>
    %v2401 = stablehlo.reduce(%v2400 init: %v2397) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2402 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2403 = stablehlo.multiply %v2401, %v2402 : tensor<384xf32>
    %v2404 = stablehlo.subtract %s2b3lg, %v2403 : tensor<384xf32>
    %v2405 = stablehlo.reshape %v550 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2406 = stablehlo.reshape %v2331 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2407 = stablehlo.transpose %v2405, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2408 = stablehlo.transpose %v2406, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2409 = stablehlo.convolution(%v2407, %v2408)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2410 = stablehlo.transpose %v2409, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2411 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2412 = stablehlo.multiply %v2410, %v2411 : tensor<384x1536x1x1xf32>
    %v2413 = stablehlo.subtract %s2b3pW, %v2412 : tensor<384x1536x1x1xf32>
    %v2414 = stablehlo.reshape %v2331 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2416 = stablehlo.reduce(%v2414 init: %v2415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2417 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2418 = stablehlo.multiply %v2416, %v2417 : tensor<384xf32>
    %v2419 = stablehlo.subtract %s2b3pb, %v2418 : tensor<384xf32>
    %v2420 = stablehlo.reshape %v532 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2421 = stablehlo.reshape %v2359 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2422 = stablehlo.transpose %v2420, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2423 = stablehlo.transpose %v2421, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2424 = stablehlo.convolution(%v2422, %v2423)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2425 = stablehlo.transpose %v2424, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2426 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2427 = stablehlo.multiply %v2425, %v2426 : tensor<1536x384x1x1xf32>
    %v2428 = stablehlo.subtract %s2b3eW, %v2427 : tensor<1536x384x1x1xf32>
    %v2429 = stablehlo.reshape %v2359 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2431 = stablehlo.reduce(%v2429 init: %v2430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2432 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2433 = stablehlo.multiply %v2431, %v2432 : tensor<1536xf32>
    %v2434 = stablehlo.subtract %s2b3eb, %v2433 : tensor<1536xf32>
    %v2435 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2436 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2437 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2438 = stablehlo.reduce(%v514 init: %v2435) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2439 = stablehlo.broadcast_in_dim %v2438, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2440 = stablehlo.divide %v2439, %v2436 : tensor<32x75264xf32>
    %v2441 = stablehlo.subtract %v514, %v2440 : tensor<32x75264xf32>
    %v2442 = stablehlo.multiply %v2441, %v2441 : tensor<32x75264xf32>
    %v2443 = stablehlo.reduce(%v2442 init: %v2435) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2444 = stablehlo.broadcast_in_dim %v2443, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2445 = stablehlo.divide %v2444, %v2436 : tensor<32x75264xf32>
    %v2446 = stablehlo.add %v2445, %v2437 : tensor<32x75264xf32>
    %v2447 = stablehlo.rsqrt %v2446 : tensor<32x75264xf32>
    %v2448 = stablehlo.multiply %v2441, %v2447 : tensor<32x75264xf32>
    %v2449 = stablehlo.multiply %v2364, %v2448 : tensor<32x75264xf32>
    %v2450 = stablehlo.reduce(%v2449 init: %v2435) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2451 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2452 = stablehlo.multiply %v2450, %v2451 : tensor<f32>
    %v2453 = stablehlo.subtract %s2b3ng, %v2452 : tensor<f32>
    %v2454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2455 = stablehlo.reduce(%v2364 init: %v2454) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2456 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2457 = stablehlo.multiply %v2455, %v2456 : tensor<f32>
    %v2458 = stablehlo.subtract %s2b3nbt, %v2457 : tensor<f32>
    %v2459 = stablehlo.reshape %v509 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2460 = stablehlo.reshape %v2391 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2461 = stablehlo.transpose %v2459, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2462 = stablehlo.transpose %v2460, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2463 = stablehlo.convolution(%v2461, %v2462)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2464 = stablehlo.reshape %v2463 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2465 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2466 = stablehlo.multiply %v2464, %v2465 : tensor<384x1x7x7xf32>
    %v2467 = stablehlo.subtract %s2b3dW, %v2466 : tensor<384x1x7x7xf32>
    %v2468 = stablehlo.reshape %v2391 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2470 = stablehlo.reduce(%v2468 init: %v2469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2471 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2472 = stablehlo.multiply %v2470, %v2471 : tensor<384xf32>
    %v2473 = stablehlo.subtract %s2b3db, %v2472 : tensor<384xf32>
    %v2474 = stablehlo.reshape %v2396 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2475 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2476 = stablehlo.multiply %v2474, %v2475 : tensor<32x384x14x14xf32>
    %v2477 = stablehlo.reshape %v2476 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2478 = stablehlo.reshape %v2477 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2479 = stablehlo.transpose %s2b2pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2480 = stablehlo.reverse %v2479, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2481 = stablehlo.convolution(%v2478, %v2480)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2482 = stablehlo.reshape %v2481 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2483 = stablehlo.multiply %v486, %v486 : tensor<32x301056xf32>
    %v2484 = stablehlo.multiply %v2483, %v486 : tensor<32x301056xf32>
    %v2485 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2486 = stablehlo.multiply %v2485, %v2484 : tensor<32x301056xf32>
    %v2487 = stablehlo.add %v486, %v2486 : tensor<32x301056xf32>
    %v2488 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2489 = stablehlo.multiply %v2488, %v2487 : tensor<32x301056xf32>
    %v2490 = stablehlo.tanh %v2489 : tensor<32x301056xf32>
    %v2491 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2492 = stablehlo.add %v2491, %v2490 : tensor<32x301056xf32>
    %v2493 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2494 = stablehlo.multiply %v2493, %v2492 : tensor<32x301056xf32>
    %v2495 = stablehlo.multiply %v2490, %v2490 : tensor<32x301056xf32>
    %v2496 = stablehlo.subtract %v2491, %v2495 : tensor<32x301056xf32>
    %v2497 = stablehlo.multiply %v2493, %v486 : tensor<32x301056xf32>
    %v2498 = stablehlo.multiply %v2497, %v2496 : tensor<32x301056xf32>
    %v2499 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2500 = stablehlo.multiply %v2499, %v2483 : tensor<32x301056xf32>
    %v2501 = stablehlo.add %v2491, %v2500 : tensor<32x301056xf32>
    %v2502 = stablehlo.multiply %v2488, %v2501 : tensor<32x301056xf32>
    %v2503 = stablehlo.multiply %v2498, %v2502 : tensor<32x301056xf32>
    %v2504 = stablehlo.add %v2494, %v2503 : tensor<32x301056xf32>
    %v2505 = stablehlo.multiply %v2482, %v2504 : tensor<32x301056xf32>
    %v2506 = stablehlo.reshape %v2505 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2507 = stablehlo.transpose %s2b2eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2508 = stablehlo.reverse %v2507, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2509 = stablehlo.convolution(%v2506, %v2508)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2510 = stablehlo.reshape %v2509 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2512 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2513 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2514 = stablehlo.reduce(%v463 init: %v2511) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2515 = stablehlo.broadcast_in_dim %v2514, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2516 = stablehlo.divide %v2515, %v2512 : tensor<32x75264xf32>
    %v2517 = stablehlo.subtract %v463, %v2516 : tensor<32x75264xf32>
    %v2518 = stablehlo.multiply %v2517, %v2517 : tensor<32x75264xf32>
    %v2519 = stablehlo.reduce(%v2518 init: %v2511) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2520 = stablehlo.broadcast_in_dim %v2519, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2521 = stablehlo.divide %v2520, %v2512 : tensor<32x75264xf32>
    %v2522 = stablehlo.add %v2521, %v2513 : tensor<32x75264xf32>
    %v2523 = stablehlo.rsqrt %v2522 : tensor<32x75264xf32>
    %v2524 = stablehlo.multiply %v2517, %v2523 : tensor<32x75264xf32>
    %v2525 = stablehlo.broadcast_in_dim %s2b2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2526 = stablehlo.multiply %v2525, %v2510 : tensor<32x75264xf32>
    %v2527 = stablehlo.reduce(%v2526 init: %v2511) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2528 = stablehlo.broadcast_in_dim %v2527, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2529 = stablehlo.multiply %v2524, %v2526 : tensor<32x75264xf32>
    %v2530 = stablehlo.reduce(%v2529 init: %v2511) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2531 = stablehlo.broadcast_in_dim %v2530, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2532 = stablehlo.multiply %v2526, %v2512 : tensor<32x75264xf32>
    %v2533 = stablehlo.subtract %v2532, %v2528 : tensor<32x75264xf32>
    %v2534 = stablehlo.multiply %v2524, %v2531 : tensor<32x75264xf32>
    %v2535 = stablehlo.subtract %v2533, %v2534 : tensor<32x75264xf32>
    %v2536 = stablehlo.divide %v2523, %v2512 : tensor<32x75264xf32>
    %v2537 = stablehlo.multiply %v2536, %v2535 : tensor<32x75264xf32>
    %v2538 = stablehlo.reshape %v2537 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2539 = stablehlo.reverse %s2b2dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2540 = stablehlo.convolution(%v2538, %v2539)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2541 = stablehlo.reshape %v2540 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2542 = stablehlo.add %v2541, %v2396 : tensor<32x75264xf32>
    %v2543 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2544 = stablehlo.reshape %v504 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2545 = stablehlo.reshape %v2396 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2546 = stablehlo.multiply %v2544, %v2545 : tensor<32x384x14x14xf32>
    %v2547 = stablehlo.reduce(%v2546 init: %v2543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2548 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2549 = stablehlo.multiply %v2547, %v2548 : tensor<384xf32>
    %v2550 = stablehlo.subtract %s2b2lg, %v2549 : tensor<384xf32>
    %v2551 = stablehlo.reshape %v499 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2552 = stablehlo.reshape %v2477 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2553 = stablehlo.transpose %v2551, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2554 = stablehlo.transpose %v2552, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2555 = stablehlo.convolution(%v2553, %v2554)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2556 = stablehlo.transpose %v2555, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2557 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2558 = stablehlo.multiply %v2556, %v2557 : tensor<384x1536x1x1xf32>
    %v2559 = stablehlo.subtract %s2b2pW, %v2558 : tensor<384x1536x1x1xf32>
    %v2560 = stablehlo.reshape %v2477 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2561 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2562 = stablehlo.reduce(%v2560 init: %v2561) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2563 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2564 = stablehlo.multiply %v2562, %v2563 : tensor<384xf32>
    %v2565 = stablehlo.subtract %s2b2pb, %v2564 : tensor<384xf32>
    %v2566 = stablehlo.reshape %v481 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2567 = stablehlo.reshape %v2505 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2568 = stablehlo.transpose %v2566, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2569 = stablehlo.transpose %v2567, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2570 = stablehlo.convolution(%v2568, %v2569)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2571 = stablehlo.transpose %v2570, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2572 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2573 = stablehlo.multiply %v2571, %v2572 : tensor<1536x384x1x1xf32>
    %v2574 = stablehlo.subtract %s2b2eW, %v2573 : tensor<1536x384x1x1xf32>
    %v2575 = stablehlo.reshape %v2505 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2576 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2577 = stablehlo.reduce(%v2575 init: %v2576) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2578 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2579 = stablehlo.multiply %v2577, %v2578 : tensor<1536xf32>
    %v2580 = stablehlo.subtract %s2b2eb, %v2579 : tensor<1536xf32>
    %v2581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2582 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2583 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2584 = stablehlo.reduce(%v463 init: %v2581) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2585 = stablehlo.broadcast_in_dim %v2584, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2586 = stablehlo.divide %v2585, %v2582 : tensor<32x75264xf32>
    %v2587 = stablehlo.subtract %v463, %v2586 : tensor<32x75264xf32>
    %v2588 = stablehlo.multiply %v2587, %v2587 : tensor<32x75264xf32>
    %v2589 = stablehlo.reduce(%v2588 init: %v2581) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2590 = stablehlo.broadcast_in_dim %v2589, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2591 = stablehlo.divide %v2590, %v2582 : tensor<32x75264xf32>
    %v2592 = stablehlo.add %v2591, %v2583 : tensor<32x75264xf32>
    %v2593 = stablehlo.rsqrt %v2592 : tensor<32x75264xf32>
    %v2594 = stablehlo.multiply %v2587, %v2593 : tensor<32x75264xf32>
    %v2595 = stablehlo.multiply %v2510, %v2594 : tensor<32x75264xf32>
    %v2596 = stablehlo.reduce(%v2595 init: %v2581) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2597 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2598 = stablehlo.multiply %v2596, %v2597 : tensor<f32>
    %v2599 = stablehlo.subtract %s2b2ng, %v2598 : tensor<f32>
    %v2600 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2601 = stablehlo.reduce(%v2510 init: %v2600) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2602 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2603 = stablehlo.multiply %v2601, %v2602 : tensor<f32>
    %v2604 = stablehlo.subtract %s2b2nbt, %v2603 : tensor<f32>
    %v2605 = stablehlo.reshape %v458 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2606 = stablehlo.reshape %v2537 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2607 = stablehlo.transpose %v2605, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2608 = stablehlo.transpose %v2606, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2609 = stablehlo.convolution(%v2607, %v2608)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2611 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2612 = stablehlo.multiply %v2610, %v2611 : tensor<384x1x7x7xf32>
    %v2613 = stablehlo.subtract %s2b2dW, %v2612 : tensor<384x1x7x7xf32>
    %v2614 = stablehlo.reshape %v2537 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2616 = stablehlo.reduce(%v2614 init: %v2615) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2617 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2618 = stablehlo.multiply %v2616, %v2617 : tensor<384xf32>
    %v2619 = stablehlo.subtract %s2b2db, %v2618 : tensor<384xf32>
    %v2620 = stablehlo.reshape %v2542 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2621 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2622 = stablehlo.multiply %v2620, %v2621 : tensor<32x384x14x14xf32>
    %v2623 = stablehlo.reshape %v2622 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2624 = stablehlo.reshape %v2623 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2625 = stablehlo.transpose %s2b1pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2626 = stablehlo.reverse %v2625, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2627 = stablehlo.convolution(%v2624, %v2626)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2628 = stablehlo.reshape %v2627 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2629 = stablehlo.multiply %v435, %v435 : tensor<32x301056xf32>
    %v2630 = stablehlo.multiply %v2629, %v435 : tensor<32x301056xf32>
    %v2631 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2632 = stablehlo.multiply %v2631, %v2630 : tensor<32x301056xf32>
    %v2633 = stablehlo.add %v435, %v2632 : tensor<32x301056xf32>
    %v2634 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2635 = stablehlo.multiply %v2634, %v2633 : tensor<32x301056xf32>
    %v2636 = stablehlo.tanh %v2635 : tensor<32x301056xf32>
    %v2637 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2638 = stablehlo.add %v2637, %v2636 : tensor<32x301056xf32>
    %v2639 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2640 = stablehlo.multiply %v2639, %v2638 : tensor<32x301056xf32>
    %v2641 = stablehlo.multiply %v2636, %v2636 : tensor<32x301056xf32>
    %v2642 = stablehlo.subtract %v2637, %v2641 : tensor<32x301056xf32>
    %v2643 = stablehlo.multiply %v2639, %v435 : tensor<32x301056xf32>
    %v2644 = stablehlo.multiply %v2643, %v2642 : tensor<32x301056xf32>
    %v2645 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2646 = stablehlo.multiply %v2645, %v2629 : tensor<32x301056xf32>
    %v2647 = stablehlo.add %v2637, %v2646 : tensor<32x301056xf32>
    %v2648 = stablehlo.multiply %v2634, %v2647 : tensor<32x301056xf32>
    %v2649 = stablehlo.multiply %v2644, %v2648 : tensor<32x301056xf32>
    %v2650 = stablehlo.add %v2640, %v2649 : tensor<32x301056xf32>
    %v2651 = stablehlo.multiply %v2628, %v2650 : tensor<32x301056xf32>
    %v2652 = stablehlo.reshape %v2651 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2653 = stablehlo.transpose %s2b1eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2654 = stablehlo.reverse %v2653, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2655 = stablehlo.convolution(%v2652, %v2654)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2656 = stablehlo.reshape %v2655 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2658 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2659 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2660 = stablehlo.reduce(%v412 init: %v2657) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2661 = stablehlo.broadcast_in_dim %v2660, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2662 = stablehlo.divide %v2661, %v2658 : tensor<32x75264xf32>
    %v2663 = stablehlo.subtract %v412, %v2662 : tensor<32x75264xf32>
    %v2664 = stablehlo.multiply %v2663, %v2663 : tensor<32x75264xf32>
    %v2665 = stablehlo.reduce(%v2664 init: %v2657) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2666 = stablehlo.broadcast_in_dim %v2665, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2667 = stablehlo.divide %v2666, %v2658 : tensor<32x75264xf32>
    %v2668 = stablehlo.add %v2667, %v2659 : tensor<32x75264xf32>
    %v2669 = stablehlo.rsqrt %v2668 : tensor<32x75264xf32>
    %v2670 = stablehlo.multiply %v2663, %v2669 : tensor<32x75264xf32>
    %v2671 = stablehlo.broadcast_in_dim %s2b1ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2672 = stablehlo.multiply %v2671, %v2656 : tensor<32x75264xf32>
    %v2673 = stablehlo.reduce(%v2672 init: %v2657) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2674 = stablehlo.broadcast_in_dim %v2673, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2675 = stablehlo.multiply %v2670, %v2672 : tensor<32x75264xf32>
    %v2676 = stablehlo.reduce(%v2675 init: %v2657) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2677 = stablehlo.broadcast_in_dim %v2676, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2678 = stablehlo.multiply %v2672, %v2658 : tensor<32x75264xf32>
    %v2679 = stablehlo.subtract %v2678, %v2674 : tensor<32x75264xf32>
    %v2680 = stablehlo.multiply %v2670, %v2677 : tensor<32x75264xf32>
    %v2681 = stablehlo.subtract %v2679, %v2680 : tensor<32x75264xf32>
    %v2682 = stablehlo.divide %v2669, %v2658 : tensor<32x75264xf32>
    %v2683 = stablehlo.multiply %v2682, %v2681 : tensor<32x75264xf32>
    %v2684 = stablehlo.reshape %v2683 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2685 = stablehlo.reverse %s2b1dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2686 = stablehlo.convolution(%v2684, %v2685)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2687 = stablehlo.reshape %v2686 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2688 = stablehlo.add %v2687, %v2542 : tensor<32x75264xf32>
    %v2689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2690 = stablehlo.reshape %v453 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2691 = stablehlo.reshape %v2542 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2692 = stablehlo.multiply %v2690, %v2691 : tensor<32x384x14x14xf32>
    %v2693 = stablehlo.reduce(%v2692 init: %v2689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2694 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2695 = stablehlo.multiply %v2693, %v2694 : tensor<384xf32>
    %v2696 = stablehlo.subtract %s2b1lg, %v2695 : tensor<384xf32>
    %v2697 = stablehlo.reshape %v448 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2698 = stablehlo.reshape %v2623 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2699 = stablehlo.transpose %v2697, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2700 = stablehlo.transpose %v2698, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2701 = stablehlo.convolution(%v2699, %v2700)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2702 = stablehlo.transpose %v2701, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2703 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2704 = stablehlo.multiply %v2702, %v2703 : tensor<384x1536x1x1xf32>
    %v2705 = stablehlo.subtract %s2b1pW, %v2704 : tensor<384x1536x1x1xf32>
    %v2706 = stablehlo.reshape %v2623 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2707 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2708 = stablehlo.reduce(%v2706 init: %v2707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2709 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2710 = stablehlo.multiply %v2708, %v2709 : tensor<384xf32>
    %v2711 = stablehlo.subtract %s2b1pb, %v2710 : tensor<384xf32>
    %v2712 = stablehlo.reshape %v430 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2713 = stablehlo.reshape %v2651 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2714 = stablehlo.transpose %v2712, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2715 = stablehlo.transpose %v2713, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2716 = stablehlo.convolution(%v2714, %v2715)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2717 = stablehlo.transpose %v2716, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2718 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2719 = stablehlo.multiply %v2717, %v2718 : tensor<1536x384x1x1xf32>
    %v2720 = stablehlo.subtract %s2b1eW, %v2719 : tensor<1536x384x1x1xf32>
    %v2721 = stablehlo.reshape %v2651 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2722 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2723 = stablehlo.reduce(%v2721 init: %v2722) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2724 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2725 = stablehlo.multiply %v2723, %v2724 : tensor<1536xf32>
    %v2726 = stablehlo.subtract %s2b1eb, %v2725 : tensor<1536xf32>
    %v2727 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2728 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2729 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2730 = stablehlo.reduce(%v412 init: %v2727) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2731 = stablehlo.broadcast_in_dim %v2730, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2732 = stablehlo.divide %v2731, %v2728 : tensor<32x75264xf32>
    %v2733 = stablehlo.subtract %v412, %v2732 : tensor<32x75264xf32>
    %v2734 = stablehlo.multiply %v2733, %v2733 : tensor<32x75264xf32>
    %v2735 = stablehlo.reduce(%v2734 init: %v2727) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2736 = stablehlo.broadcast_in_dim %v2735, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2737 = stablehlo.divide %v2736, %v2728 : tensor<32x75264xf32>
    %v2738 = stablehlo.add %v2737, %v2729 : tensor<32x75264xf32>
    %v2739 = stablehlo.rsqrt %v2738 : tensor<32x75264xf32>
    %v2740 = stablehlo.multiply %v2733, %v2739 : tensor<32x75264xf32>
    %v2741 = stablehlo.multiply %v2656, %v2740 : tensor<32x75264xf32>
    %v2742 = stablehlo.reduce(%v2741 init: %v2727) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2743 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2744 = stablehlo.multiply %v2742, %v2743 : tensor<f32>
    %v2745 = stablehlo.subtract %s2b1ng, %v2744 : tensor<f32>
    %v2746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2747 = stablehlo.reduce(%v2656 init: %v2746) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2748 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2749 = stablehlo.multiply %v2747, %v2748 : tensor<f32>
    %v2750 = stablehlo.subtract %s2b1nbt, %v2749 : tensor<f32>
    %v2751 = stablehlo.reshape %v407 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2752 = stablehlo.reshape %v2683 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2753 = stablehlo.transpose %v2751, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2754 = stablehlo.transpose %v2752, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2755 = stablehlo.convolution(%v2753, %v2754)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2756 = stablehlo.reshape %v2755 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2757 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2758 = stablehlo.multiply %v2756, %v2757 : tensor<384x1x7x7xf32>
    %v2759 = stablehlo.subtract %s2b1dW, %v2758 : tensor<384x1x7x7xf32>
    %v2760 = stablehlo.reshape %v2683 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2761 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2762 = stablehlo.reduce(%v2760 init: %v2761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2763 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2764 = stablehlo.multiply %v2762, %v2763 : tensor<384xf32>
    %v2765 = stablehlo.subtract %s2b1db, %v2764 : tensor<384xf32>
    %v2766 = stablehlo.reshape %v2688 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2767 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2768 = stablehlo.multiply %v2766, %v2767 : tensor<32x384x14x14xf32>
    %v2769 = stablehlo.reshape %v2768 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2770 = stablehlo.reshape %v2769 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2771 = stablehlo.transpose %s2b0pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2772 = stablehlo.reverse %v2771, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2773 = stablehlo.convolution(%v2770, %v2772)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2774 = stablehlo.reshape %v2773 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2775 = stablehlo.multiply %v384, %v384 : tensor<32x301056xf32>
    %v2776 = stablehlo.multiply %v2775, %v384 : tensor<32x301056xf32>
    %v2777 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2778 = stablehlo.multiply %v2777, %v2776 : tensor<32x301056xf32>
    %v2779 = stablehlo.add %v384, %v2778 : tensor<32x301056xf32>
    %v2780 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2781 = stablehlo.multiply %v2780, %v2779 : tensor<32x301056xf32>
    %v2782 = stablehlo.tanh %v2781 : tensor<32x301056xf32>
    %v2783 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2784 = stablehlo.add %v2783, %v2782 : tensor<32x301056xf32>
    %v2785 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2786 = stablehlo.multiply %v2785, %v2784 : tensor<32x301056xf32>
    %v2787 = stablehlo.multiply %v2782, %v2782 : tensor<32x301056xf32>
    %v2788 = stablehlo.subtract %v2783, %v2787 : tensor<32x301056xf32>
    %v2789 = stablehlo.multiply %v2785, %v384 : tensor<32x301056xf32>
    %v2790 = stablehlo.multiply %v2789, %v2788 : tensor<32x301056xf32>
    %v2791 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2792 = stablehlo.multiply %v2791, %v2775 : tensor<32x301056xf32>
    %v2793 = stablehlo.add %v2783, %v2792 : tensor<32x301056xf32>
    %v2794 = stablehlo.multiply %v2780, %v2793 : tensor<32x301056xf32>
    %v2795 = stablehlo.multiply %v2790, %v2794 : tensor<32x301056xf32>
    %v2796 = stablehlo.add %v2786, %v2795 : tensor<32x301056xf32>
    %v2797 = stablehlo.multiply %v2774, %v2796 : tensor<32x301056xf32>
    %v2798 = stablehlo.reshape %v2797 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2799 = stablehlo.transpose %s2b0eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2800 = stablehlo.reverse %v2799, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2801 = stablehlo.convolution(%v2798, %v2800)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2802 = stablehlo.reshape %v2801 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2804 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2805 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2806 = stablehlo.reduce(%v361 init: %v2803) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2807 = stablehlo.broadcast_in_dim %v2806, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2808 = stablehlo.divide %v2807, %v2804 : tensor<32x75264xf32>
    %v2809 = stablehlo.subtract %v361, %v2808 : tensor<32x75264xf32>
    %v2810 = stablehlo.multiply %v2809, %v2809 : tensor<32x75264xf32>
    %v2811 = stablehlo.reduce(%v2810 init: %v2803) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2812 = stablehlo.broadcast_in_dim %v2811, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2813 = stablehlo.divide %v2812, %v2804 : tensor<32x75264xf32>
    %v2814 = stablehlo.add %v2813, %v2805 : tensor<32x75264xf32>
    %v2815 = stablehlo.rsqrt %v2814 : tensor<32x75264xf32>
    %v2816 = stablehlo.multiply %v2809, %v2815 : tensor<32x75264xf32>
    %v2817 = stablehlo.broadcast_in_dim %s2b0ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2818 = stablehlo.multiply %v2817, %v2802 : tensor<32x75264xf32>
    %v2819 = stablehlo.reduce(%v2818 init: %v2803) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2820 = stablehlo.broadcast_in_dim %v2819, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2821 = stablehlo.multiply %v2816, %v2818 : tensor<32x75264xf32>
    %v2822 = stablehlo.reduce(%v2821 init: %v2803) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2823 = stablehlo.broadcast_in_dim %v2822, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2824 = stablehlo.multiply %v2818, %v2804 : tensor<32x75264xf32>
    %v2825 = stablehlo.subtract %v2824, %v2820 : tensor<32x75264xf32>
    %v2826 = stablehlo.multiply %v2816, %v2823 : tensor<32x75264xf32>
    %v2827 = stablehlo.subtract %v2825, %v2826 : tensor<32x75264xf32>
    %v2828 = stablehlo.divide %v2815, %v2804 : tensor<32x75264xf32>
    %v2829 = stablehlo.multiply %v2828, %v2827 : tensor<32x75264xf32>
    %v2830 = stablehlo.reshape %v2829 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2831 = stablehlo.reverse %s2b0dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2832 = stablehlo.convolution(%v2830, %v2831)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2833 = stablehlo.reshape %v2832 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2834 = stablehlo.add %v2833, %v2688 : tensor<32x75264xf32>
    %v2835 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2836 = stablehlo.reshape %v402 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2837 = stablehlo.reshape %v2688 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2838 = stablehlo.multiply %v2836, %v2837 : tensor<32x384x14x14xf32>
    %v2839 = stablehlo.reduce(%v2838 init: %v2835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2840 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2841 = stablehlo.multiply %v2839, %v2840 : tensor<384xf32>
    %v2842 = stablehlo.subtract %s2b0lg, %v2841 : tensor<384xf32>
    %v2843 = stablehlo.reshape %v397 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2844 = stablehlo.reshape %v2769 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2845 = stablehlo.transpose %v2843, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2846 = stablehlo.transpose %v2844, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2847 = stablehlo.convolution(%v2845, %v2846)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2848 = stablehlo.transpose %v2847, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2849 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2850 = stablehlo.multiply %v2848, %v2849 : tensor<384x1536x1x1xf32>
    %v2851 = stablehlo.subtract %s2b0pW, %v2850 : tensor<384x1536x1x1xf32>
    %v2852 = stablehlo.reshape %v2769 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2853 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2854 = stablehlo.reduce(%v2852 init: %v2853) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2855 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2856 = stablehlo.multiply %v2854, %v2855 : tensor<384xf32>
    %v2857 = stablehlo.subtract %s2b0pb, %v2856 : tensor<384xf32>
    %v2858 = stablehlo.reshape %v379 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2859 = stablehlo.reshape %v2797 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2860 = stablehlo.transpose %v2858, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2861 = stablehlo.transpose %v2859, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2862 = stablehlo.convolution(%v2860, %v2861)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2863 = stablehlo.transpose %v2862, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2864 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2865 = stablehlo.multiply %v2863, %v2864 : tensor<1536x384x1x1xf32>
    %v2866 = stablehlo.subtract %s2b0eW, %v2865 : tensor<1536x384x1x1xf32>
    %v2867 = stablehlo.reshape %v2797 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2869 = stablehlo.reduce(%v2867 init: %v2868) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2870 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2871 = stablehlo.multiply %v2869, %v2870 : tensor<1536xf32>
    %v2872 = stablehlo.subtract %s2b0eb, %v2871 : tensor<1536xf32>
    %v2873 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2874 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2875 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2876 = stablehlo.reduce(%v361 init: %v2873) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2877 = stablehlo.broadcast_in_dim %v2876, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2878 = stablehlo.divide %v2877, %v2874 : tensor<32x75264xf32>
    %v2879 = stablehlo.subtract %v361, %v2878 : tensor<32x75264xf32>
    %v2880 = stablehlo.multiply %v2879, %v2879 : tensor<32x75264xf32>
    %v2881 = stablehlo.reduce(%v2880 init: %v2873) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2882 = stablehlo.broadcast_in_dim %v2881, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2883 = stablehlo.divide %v2882, %v2874 : tensor<32x75264xf32>
    %v2884 = stablehlo.add %v2883, %v2875 : tensor<32x75264xf32>
    %v2885 = stablehlo.rsqrt %v2884 : tensor<32x75264xf32>
    %v2886 = stablehlo.multiply %v2879, %v2885 : tensor<32x75264xf32>
    %v2887 = stablehlo.multiply %v2802, %v2886 : tensor<32x75264xf32>
    %v2888 = stablehlo.reduce(%v2887 init: %v2873) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2889 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2890 = stablehlo.multiply %v2888, %v2889 : tensor<f32>
    %v2891 = stablehlo.subtract %s2b0ng, %v2890 : tensor<f32>
    %v2892 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2893 = stablehlo.reduce(%v2802 init: %v2892) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2894 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2895 = stablehlo.multiply %v2893, %v2894 : tensor<f32>
    %v2896 = stablehlo.subtract %s2b0nbt, %v2895 : tensor<f32>
    %v2897 = stablehlo.reshape %v356 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2898 = stablehlo.reshape %v2829 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2899 = stablehlo.transpose %v2897, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2900 = stablehlo.transpose %v2898, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2901 = stablehlo.convolution(%v2899, %v2900)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2902 = stablehlo.reshape %v2901 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2903 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2904 = stablehlo.multiply %v2902, %v2903 : tensor<384x1x7x7xf32>
    %v2905 = stablehlo.subtract %s2b0dW, %v2904 : tensor<384x1x7x7xf32>
    %v2906 = stablehlo.reshape %v2829 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2907 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2908 = stablehlo.reduce(%v2906 init: %v2907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2909 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2910 = stablehlo.multiply %v2908, %v2909 : tensor<384xf32>
    %v2911 = stablehlo.subtract %s2b0db, %v2910 : tensor<384xf32>
    %v2912 = stablehlo.reshape %v2834 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2913 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2914 = stablehlo.pad %v2912, %v2913, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x28x28xf32>
    %v2915 = stablehlo.transpose %d1W, dims = [1, 0, 2, 3] : (tensor<384x192x2x2xf32>) -> tensor<192x384x2x2xf32>
    %v2916 = stablehlo.reverse %v2915, dims = [2, 3] : tensor<192x384x2x2xf32>
    %v2917 = stablehlo.convolution(%v2914, %v2916)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x28x28xf32>, tensor<192x384x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v2918 = stablehlo.reshape %v2917 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2919 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2920 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2921 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2922 = stablehlo.reduce(%v333 init: %v2919) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2923 = stablehlo.broadcast_in_dim %v2922, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2924 = stablehlo.divide %v2923, %v2920 : tensor<32x150528xf32>
    %v2925 = stablehlo.subtract %v333, %v2924 : tensor<32x150528xf32>
    %v2926 = stablehlo.multiply %v2925, %v2925 : tensor<32x150528xf32>
    %v2927 = stablehlo.reduce(%v2926 init: %v2919) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2928 = stablehlo.broadcast_in_dim %v2927, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2929 = stablehlo.divide %v2928, %v2920 : tensor<32x150528xf32>
    %v2930 = stablehlo.add %v2929, %v2921 : tensor<32x150528xf32>
    %v2931 = stablehlo.rsqrt %v2930 : tensor<32x150528xf32>
    %v2932 = stablehlo.multiply %v2925, %v2931 : tensor<32x150528xf32>
    %v2933 = stablehlo.broadcast_in_dim %d1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v2934 = stablehlo.multiply %v2933, %v2918 : tensor<32x150528xf32>
    %v2935 = stablehlo.reduce(%v2934 init: %v2919) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2936 = stablehlo.broadcast_in_dim %v2935, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2937 = stablehlo.multiply %v2932, %v2934 : tensor<32x150528xf32>
    %v2938 = stablehlo.reduce(%v2937 init: %v2919) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2939 = stablehlo.broadcast_in_dim %v2938, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2940 = stablehlo.multiply %v2934, %v2920 : tensor<32x150528xf32>
    %v2941 = stablehlo.subtract %v2940, %v2936 : tensor<32x150528xf32>
    %v2942 = stablehlo.multiply %v2932, %v2939 : tensor<32x150528xf32>
    %v2943 = stablehlo.subtract %v2941, %v2942 : tensor<32x150528xf32>
    %v2944 = stablehlo.divide %v2931, %v2920 : tensor<32x150528xf32>
    %v2945 = stablehlo.multiply %v2944, %v2943 : tensor<32x150528xf32>
    %v2946 = stablehlo.reshape %v2834 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2948 = stablehlo.reduce(%v2946 init: %v2947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2949 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2950 = stablehlo.multiply %v2948, %v2949 : tensor<384xf32>
    %v2951 = stablehlo.subtract %d1b, %v2950 : tensor<384xf32>
    %v2952 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2953 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2954 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2955 = stablehlo.reduce(%v333 init: %v2952) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2956 = stablehlo.broadcast_in_dim %v2955, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2957 = stablehlo.divide %v2956, %v2953 : tensor<32x150528xf32>
    %v2958 = stablehlo.subtract %v333, %v2957 : tensor<32x150528xf32>
    %v2959 = stablehlo.multiply %v2958, %v2958 : tensor<32x150528xf32>
    %v2960 = stablehlo.reduce(%v2959 init: %v2952) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2961 = stablehlo.broadcast_in_dim %v2960, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2962 = stablehlo.divide %v2961, %v2953 : tensor<32x150528xf32>
    %v2963 = stablehlo.add %v2962, %v2954 : tensor<32x150528xf32>
    %v2964 = stablehlo.rsqrt %v2963 : tensor<32x150528xf32>
    %v2965 = stablehlo.multiply %v2958, %v2964 : tensor<32x150528xf32>
    %v2966 = stablehlo.multiply %v2918, %v2965 : tensor<32x150528xf32>
    %v2967 = stablehlo.reduce(%v2966 init: %v2952) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2968 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2969 = stablehlo.multiply %v2967, %v2968 : tensor<f32>
    %v2970 = stablehlo.subtract %d1ng, %v2969 : tensor<f32>
    %v2971 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2972 = stablehlo.reduce(%v2918 init: %v2971) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2973 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2974 = stablehlo.multiply %v2972, %v2973 : tensor<f32>
    %v2975 = stablehlo.subtract %d1nbt, %v2974 : tensor<f32>
    %v2976 = stablehlo.reshape %v351 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2977 = stablehlo.reshape %v2834 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2978 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2979 = stablehlo.pad %v2977, %v2978, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x27x27xf32>
    %v2980 = stablehlo.transpose %v2976, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2981 = stablehlo.transpose %v2979, dims = [1, 0, 2, 3] : (tensor<32x384x27x27xf32>) -> tensor<384x32x27x27xf32>
    %v2982 = stablehlo.convolution(%v2980, %v2981)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<384x32x27x27xf32>) -> tensor<192x384x2x2xf32>
    %v2983 = stablehlo.transpose %v2982, dims = [1, 0, 2, 3] : (tensor<192x384x2x2xf32>) -> tensor<384x192x2x2xf32>
    %v2984 = stablehlo.constant dense<0.1> : tensor<384x192x2x2xf32>
    %v2985 = stablehlo.multiply %v2983, %v2984 : tensor<384x192x2x2xf32>
    %v2986 = stablehlo.subtract %d1W, %v2985 : tensor<384x192x2x2xf32>
    %v2987 = stablehlo.reshape %v2945 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2988 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v2989 = stablehlo.multiply %v2987, %v2988 : tensor<32x192x28x28xf32>
    %v2990 = stablehlo.reshape %v2989 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2991 = stablehlo.reshape %v2990 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2992 = stablehlo.transpose %s1b2pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2993 = stablehlo.reverse %v2992, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v2994 = stablehlo.convolution(%v2991, %v2993)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v2995 = stablehlo.reshape %v2994 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v2996 = stablehlo.multiply %v310, %v310 : tensor<32x602112xf32>
    %v2997 = stablehlo.multiply %v2996, %v310 : tensor<32x602112xf32>
    %v2998 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v2999 = stablehlo.multiply %v2998, %v2997 : tensor<32x602112xf32>
    %v3000 = stablehlo.add %v310, %v2999 : tensor<32x602112xf32>
    %v3001 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v3002 = stablehlo.multiply %v3001, %v3000 : tensor<32x602112xf32>
    %v3003 = stablehlo.tanh %v3002 : tensor<32x602112xf32>
    %v3004 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v3005 = stablehlo.add %v3004, %v3003 : tensor<32x602112xf32>
    %v3006 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v3007 = stablehlo.multiply %v3006, %v3005 : tensor<32x602112xf32>
    %v3008 = stablehlo.multiply %v3003, %v3003 : tensor<32x602112xf32>
    %v3009 = stablehlo.subtract %v3004, %v3008 : tensor<32x602112xf32>
    %v3010 = stablehlo.multiply %v3006, %v310 : tensor<32x602112xf32>
    %v3011 = stablehlo.multiply %v3010, %v3009 : tensor<32x602112xf32>
    %v3012 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v3013 = stablehlo.multiply %v3012, %v2996 : tensor<32x602112xf32>
    %v3014 = stablehlo.add %v3004, %v3013 : tensor<32x602112xf32>
    %v3015 = stablehlo.multiply %v3001, %v3014 : tensor<32x602112xf32>
    %v3016 = stablehlo.multiply %v3011, %v3015 : tensor<32x602112xf32>
    %v3017 = stablehlo.add %v3007, %v3016 : tensor<32x602112xf32>
    %v3018 = stablehlo.multiply %v2995, %v3017 : tensor<32x602112xf32>
    %v3019 = stablehlo.reshape %v3018 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3020 = stablehlo.transpose %s1b2eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3021 = stablehlo.reverse %v3020, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3022 = stablehlo.convolution(%v3019, %v3021)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3023 = stablehlo.reshape %v3022 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3024 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3025 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v3026 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v3027 = stablehlo.reduce(%v287 init: %v3024) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3028 = stablehlo.broadcast_in_dim %v3027, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3029 = stablehlo.divide %v3028, %v3025 : tensor<32x150528xf32>
    %v3030 = stablehlo.subtract %v287, %v3029 : tensor<32x150528xf32>
    %v3031 = stablehlo.multiply %v3030, %v3030 : tensor<32x150528xf32>
    %v3032 = stablehlo.reduce(%v3031 init: %v3024) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3033 = stablehlo.broadcast_in_dim %v3032, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3034 = stablehlo.divide %v3033, %v3025 : tensor<32x150528xf32>
    %v3035 = stablehlo.add %v3034, %v3026 : tensor<32x150528xf32>
    %v3036 = stablehlo.rsqrt %v3035 : tensor<32x150528xf32>
    %v3037 = stablehlo.multiply %v3030, %v3036 : tensor<32x150528xf32>
    %v3038 = stablehlo.broadcast_in_dim %s1b2ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v3039 = stablehlo.multiply %v3038, %v3023 : tensor<32x150528xf32>
    %v3040 = stablehlo.reduce(%v3039 init: %v3024) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3041 = stablehlo.broadcast_in_dim %v3040, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3042 = stablehlo.multiply %v3037, %v3039 : tensor<32x150528xf32>
    %v3043 = stablehlo.reduce(%v3042 init: %v3024) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3044 = stablehlo.broadcast_in_dim %v3043, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3045 = stablehlo.multiply %v3039, %v3025 : tensor<32x150528xf32>
    %v3046 = stablehlo.subtract %v3045, %v3041 : tensor<32x150528xf32>
    %v3047 = stablehlo.multiply %v3037, %v3044 : tensor<32x150528xf32>
    %v3048 = stablehlo.subtract %v3046, %v3047 : tensor<32x150528xf32>
    %v3049 = stablehlo.divide %v3036, %v3025 : tensor<32x150528xf32>
    %v3050 = stablehlo.multiply %v3049, %v3048 : tensor<32x150528xf32>
    %v3051 = stablehlo.reshape %v3050 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3052 = stablehlo.reverse %s1b2dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v3053 = stablehlo.convolution(%v3051, %v3052)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v3054 = stablehlo.reshape %v3053 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3055 = stablehlo.add %v3054, %v2945 : tensor<32x150528xf32>
    %v3056 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3057 = stablehlo.reshape %v328 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3058 = stablehlo.reshape %v2945 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3059 = stablehlo.multiply %v3057, %v3058 : tensor<32x192x28x28xf32>
    %v3060 = stablehlo.reduce(%v3059 init: %v3056) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3061 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3062 = stablehlo.multiply %v3060, %v3061 : tensor<192xf32>
    %v3063 = stablehlo.subtract %s1b2lg, %v3062 : tensor<192xf32>
    %v3064 = stablehlo.reshape %v323 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3065 = stablehlo.reshape %v2990 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3066 = stablehlo.transpose %v3064, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3067 = stablehlo.transpose %v3065, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3068 = stablehlo.convolution(%v3066, %v3067)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v3069 = stablehlo.transpose %v3068, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3070 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v3071 = stablehlo.multiply %v3069, %v3070 : tensor<192x768x1x1xf32>
    %v3072 = stablehlo.subtract %s1b2pW, %v3071 : tensor<192x768x1x1xf32>
    %v3073 = stablehlo.reshape %v2990 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3074 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3075 = stablehlo.reduce(%v3073 init: %v3074) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3076 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3077 = stablehlo.multiply %v3075, %v3076 : tensor<192xf32>
    %v3078 = stablehlo.subtract %s1b2pb, %v3077 : tensor<192xf32>
    %v3079 = stablehlo.reshape %v305 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3080 = stablehlo.reshape %v3018 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3081 = stablehlo.transpose %v3079, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3082 = stablehlo.transpose %v3080, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3083 = stablehlo.convolution(%v3081, %v3082)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v3084 = stablehlo.transpose %v3083, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3085 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v3086 = stablehlo.multiply %v3084, %v3085 : tensor<768x192x1x1xf32>
    %v3087 = stablehlo.subtract %s1b2eW, %v3086 : tensor<768x192x1x1xf32>
    %v3088 = stablehlo.reshape %v3018 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3090 = stablehlo.reduce(%v3088 init: %v3089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v3091 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v3092 = stablehlo.multiply %v3090, %v3091 : tensor<768xf32>
    %v3093 = stablehlo.subtract %s1b2eb, %v3092 : tensor<768xf32>
    %v3094 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3095 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v3096 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v3097 = stablehlo.reduce(%v287 init: %v3094) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3098 = stablehlo.broadcast_in_dim %v3097, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3099 = stablehlo.divide %v3098, %v3095 : tensor<32x150528xf32>
    %v3100 = stablehlo.subtract %v287, %v3099 : tensor<32x150528xf32>
    %v3101 = stablehlo.multiply %v3100, %v3100 : tensor<32x150528xf32>
    %v3102 = stablehlo.reduce(%v3101 init: %v3094) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3103 = stablehlo.broadcast_in_dim %v3102, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3104 = stablehlo.divide %v3103, %v3095 : tensor<32x150528xf32>
    %v3105 = stablehlo.add %v3104, %v3096 : tensor<32x150528xf32>
    %v3106 = stablehlo.rsqrt %v3105 : tensor<32x150528xf32>
    %v3107 = stablehlo.multiply %v3100, %v3106 : tensor<32x150528xf32>
    %v3108 = stablehlo.multiply %v3023, %v3107 : tensor<32x150528xf32>
    %v3109 = stablehlo.reduce(%v3108 init: %v3094) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v3110 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3111 = stablehlo.multiply %v3109, %v3110 : tensor<f32>
    %v3112 = stablehlo.subtract %s1b2ng, %v3111 : tensor<f32>
    %v3113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3114 = stablehlo.reduce(%v3023 init: %v3113) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v3115 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3116 = stablehlo.multiply %v3114, %v3115 : tensor<f32>
    %v3117 = stablehlo.subtract %s1b2nbt, %v3116 : tensor<f32>
    %v3118 = stablehlo.reshape %v282 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3119 = stablehlo.reshape %v3050 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3120 = stablehlo.transpose %v3118, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3121 = stablehlo.transpose %v3119, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3122 = stablehlo.convolution(%v3120, %v3121)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v3123 = stablehlo.reshape %v3122 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v3124 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v3125 = stablehlo.multiply %v3123, %v3124 : tensor<192x1x7x7xf32>
    %v3126 = stablehlo.subtract %s1b2dW, %v3125 : tensor<192x1x7x7xf32>
    %v3127 = stablehlo.reshape %v3050 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3129 = stablehlo.reduce(%v3127 init: %v3128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3130 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3131 = stablehlo.multiply %v3129, %v3130 : tensor<192xf32>
    %v3132 = stablehlo.subtract %s1b2db, %v3131 : tensor<192xf32>
    %v3133 = stablehlo.reshape %v3055 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3134 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3135 = stablehlo.multiply %v3133, %v3134 : tensor<32x192x28x28xf32>
    %v3136 = stablehlo.reshape %v3135 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3137 = stablehlo.reshape %v3136 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3138 = stablehlo.transpose %s1b1pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3139 = stablehlo.reverse %v3138, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v3140 = stablehlo.convolution(%v3137, %v3139)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v3141 = stablehlo.reshape %v3140 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3142 = stablehlo.multiply %v259, %v259 : tensor<32x602112xf32>
    %v3143 = stablehlo.multiply %v3142, %v259 : tensor<32x602112xf32>
    %v3144 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v3145 = stablehlo.multiply %v3144, %v3143 : tensor<32x602112xf32>
    %v3146 = stablehlo.add %v259, %v3145 : tensor<32x602112xf32>
    %v3147 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v3148 = stablehlo.multiply %v3147, %v3146 : tensor<32x602112xf32>
    %v3149 = stablehlo.tanh %v3148 : tensor<32x602112xf32>
    %v3150 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v3151 = stablehlo.add %v3150, %v3149 : tensor<32x602112xf32>
    %v3152 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v3153 = stablehlo.multiply %v3152, %v3151 : tensor<32x602112xf32>
    %v3154 = stablehlo.multiply %v3149, %v3149 : tensor<32x602112xf32>
    %v3155 = stablehlo.subtract %v3150, %v3154 : tensor<32x602112xf32>
    %v3156 = stablehlo.multiply %v3152, %v259 : tensor<32x602112xf32>
    %v3157 = stablehlo.multiply %v3156, %v3155 : tensor<32x602112xf32>
    %v3158 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v3159 = stablehlo.multiply %v3158, %v3142 : tensor<32x602112xf32>
    %v3160 = stablehlo.add %v3150, %v3159 : tensor<32x602112xf32>
    %v3161 = stablehlo.multiply %v3147, %v3160 : tensor<32x602112xf32>
    %v3162 = stablehlo.multiply %v3157, %v3161 : tensor<32x602112xf32>
    %v3163 = stablehlo.add %v3153, %v3162 : tensor<32x602112xf32>
    %v3164 = stablehlo.multiply %v3141, %v3163 : tensor<32x602112xf32>
    %v3165 = stablehlo.reshape %v3164 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3166 = stablehlo.transpose %s1b1eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3167 = stablehlo.reverse %v3166, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3168 = stablehlo.convolution(%v3165, %v3167)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3169 = stablehlo.reshape %v3168 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3171 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v3172 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v3173 = stablehlo.reduce(%v236 init: %v3170) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3174 = stablehlo.broadcast_in_dim %v3173, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3175 = stablehlo.divide %v3174, %v3171 : tensor<32x150528xf32>
    %v3176 = stablehlo.subtract %v236, %v3175 : tensor<32x150528xf32>
    %v3177 = stablehlo.multiply %v3176, %v3176 : tensor<32x150528xf32>
    %v3178 = stablehlo.reduce(%v3177 init: %v3170) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3179 = stablehlo.broadcast_in_dim %v3178, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3180 = stablehlo.divide %v3179, %v3171 : tensor<32x150528xf32>
    %v3181 = stablehlo.add %v3180, %v3172 : tensor<32x150528xf32>
    %v3182 = stablehlo.rsqrt %v3181 : tensor<32x150528xf32>
    %v3183 = stablehlo.multiply %v3176, %v3182 : tensor<32x150528xf32>
    %v3184 = stablehlo.broadcast_in_dim %s1b1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v3185 = stablehlo.multiply %v3184, %v3169 : tensor<32x150528xf32>
    %v3186 = stablehlo.reduce(%v3185 init: %v3170) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3187 = stablehlo.broadcast_in_dim %v3186, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3188 = stablehlo.multiply %v3183, %v3185 : tensor<32x150528xf32>
    %v3189 = stablehlo.reduce(%v3188 init: %v3170) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3190 = stablehlo.broadcast_in_dim %v3189, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3191 = stablehlo.multiply %v3185, %v3171 : tensor<32x150528xf32>
    %v3192 = stablehlo.subtract %v3191, %v3187 : tensor<32x150528xf32>
    %v3193 = stablehlo.multiply %v3183, %v3190 : tensor<32x150528xf32>
    %v3194 = stablehlo.subtract %v3192, %v3193 : tensor<32x150528xf32>
    %v3195 = stablehlo.divide %v3182, %v3171 : tensor<32x150528xf32>
    %v3196 = stablehlo.multiply %v3195, %v3194 : tensor<32x150528xf32>
    %v3197 = stablehlo.reshape %v3196 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3198 = stablehlo.reverse %s1b1dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v3199 = stablehlo.convolution(%v3197, %v3198)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v3200 = stablehlo.reshape %v3199 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3201 = stablehlo.add %v3200, %v3055 : tensor<32x150528xf32>
    %v3202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3203 = stablehlo.reshape %v277 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3204 = stablehlo.reshape %v3055 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3205 = stablehlo.multiply %v3203, %v3204 : tensor<32x192x28x28xf32>
    %v3206 = stablehlo.reduce(%v3205 init: %v3202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3207 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3208 = stablehlo.multiply %v3206, %v3207 : tensor<192xf32>
    %v3209 = stablehlo.subtract %s1b1lg, %v3208 : tensor<192xf32>
    %v3210 = stablehlo.reshape %v272 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3211 = stablehlo.reshape %v3136 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3212 = stablehlo.transpose %v3210, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3213 = stablehlo.transpose %v3211, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3214 = stablehlo.convolution(%v3212, %v3213)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v3215 = stablehlo.transpose %v3214, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3216 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v3217 = stablehlo.multiply %v3215, %v3216 : tensor<192x768x1x1xf32>
    %v3218 = stablehlo.subtract %s1b1pW, %v3217 : tensor<192x768x1x1xf32>
    %v3219 = stablehlo.reshape %v3136 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3221 = stablehlo.reduce(%v3219 init: %v3220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3222 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3223 = stablehlo.multiply %v3221, %v3222 : tensor<192xf32>
    %v3224 = stablehlo.subtract %s1b1pb, %v3223 : tensor<192xf32>
    %v3225 = stablehlo.reshape %v254 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3226 = stablehlo.reshape %v3164 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3227 = stablehlo.transpose %v3225, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3228 = stablehlo.transpose %v3226, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3229 = stablehlo.convolution(%v3227, %v3228)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v3230 = stablehlo.transpose %v3229, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3231 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v3232 = stablehlo.multiply %v3230, %v3231 : tensor<768x192x1x1xf32>
    %v3233 = stablehlo.subtract %s1b1eW, %v3232 : tensor<768x192x1x1xf32>
    %v3234 = stablehlo.reshape %v3164 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3236 = stablehlo.reduce(%v3234 init: %v3235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v3237 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v3238 = stablehlo.multiply %v3236, %v3237 : tensor<768xf32>
    %v3239 = stablehlo.subtract %s1b1eb, %v3238 : tensor<768xf32>
    %v3240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3241 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v3242 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v3243 = stablehlo.reduce(%v236 init: %v3240) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3244 = stablehlo.broadcast_in_dim %v3243, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3245 = stablehlo.divide %v3244, %v3241 : tensor<32x150528xf32>
    %v3246 = stablehlo.subtract %v236, %v3245 : tensor<32x150528xf32>
    %v3247 = stablehlo.multiply %v3246, %v3246 : tensor<32x150528xf32>
    %v3248 = stablehlo.reduce(%v3247 init: %v3240) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3249 = stablehlo.broadcast_in_dim %v3248, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3250 = stablehlo.divide %v3249, %v3241 : tensor<32x150528xf32>
    %v3251 = stablehlo.add %v3250, %v3242 : tensor<32x150528xf32>
    %v3252 = stablehlo.rsqrt %v3251 : tensor<32x150528xf32>
    %v3253 = stablehlo.multiply %v3246, %v3252 : tensor<32x150528xf32>
    %v3254 = stablehlo.multiply %v3169, %v3253 : tensor<32x150528xf32>
    %v3255 = stablehlo.reduce(%v3254 init: %v3240) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v3256 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3257 = stablehlo.multiply %v3255, %v3256 : tensor<f32>
    %v3258 = stablehlo.subtract %s1b1ng, %v3257 : tensor<f32>
    %v3259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3260 = stablehlo.reduce(%v3169 init: %v3259) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v3261 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3262 = stablehlo.multiply %v3260, %v3261 : tensor<f32>
    %v3263 = stablehlo.subtract %s1b1nbt, %v3262 : tensor<f32>
    %v3264 = stablehlo.reshape %v231 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3265 = stablehlo.reshape %v3196 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3266 = stablehlo.transpose %v3264, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3267 = stablehlo.transpose %v3265, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3268 = stablehlo.convolution(%v3266, %v3267)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v3269 = stablehlo.reshape %v3268 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v3270 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v3271 = stablehlo.multiply %v3269, %v3270 : tensor<192x1x7x7xf32>
    %v3272 = stablehlo.subtract %s1b1dW, %v3271 : tensor<192x1x7x7xf32>
    %v3273 = stablehlo.reshape %v3196 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3275 = stablehlo.reduce(%v3273 init: %v3274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3276 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3277 = stablehlo.multiply %v3275, %v3276 : tensor<192xf32>
    %v3278 = stablehlo.subtract %s1b1db, %v3277 : tensor<192xf32>
    %v3279 = stablehlo.reshape %v3201 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3280 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3281 = stablehlo.multiply %v3279, %v3280 : tensor<32x192x28x28xf32>
    %v3282 = stablehlo.reshape %v3281 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3283 = stablehlo.reshape %v3282 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3284 = stablehlo.transpose %s1b0pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3285 = stablehlo.reverse %v3284, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v3286 = stablehlo.convolution(%v3283, %v3285)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v3287 = stablehlo.reshape %v3286 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3288 = stablehlo.multiply %v208, %v208 : tensor<32x602112xf32>
    %v3289 = stablehlo.multiply %v3288, %v208 : tensor<32x602112xf32>
    %v3290 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v3291 = stablehlo.multiply %v3290, %v3289 : tensor<32x602112xf32>
    %v3292 = stablehlo.add %v208, %v3291 : tensor<32x602112xf32>
    %v3293 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v3294 = stablehlo.multiply %v3293, %v3292 : tensor<32x602112xf32>
    %v3295 = stablehlo.tanh %v3294 : tensor<32x602112xf32>
    %v3296 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v3297 = stablehlo.add %v3296, %v3295 : tensor<32x602112xf32>
    %v3298 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v3299 = stablehlo.multiply %v3298, %v3297 : tensor<32x602112xf32>
    %v3300 = stablehlo.multiply %v3295, %v3295 : tensor<32x602112xf32>
    %v3301 = stablehlo.subtract %v3296, %v3300 : tensor<32x602112xf32>
    %v3302 = stablehlo.multiply %v3298, %v208 : tensor<32x602112xf32>
    %v3303 = stablehlo.multiply %v3302, %v3301 : tensor<32x602112xf32>
    %v3304 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v3305 = stablehlo.multiply %v3304, %v3288 : tensor<32x602112xf32>
    %v3306 = stablehlo.add %v3296, %v3305 : tensor<32x602112xf32>
    %v3307 = stablehlo.multiply %v3293, %v3306 : tensor<32x602112xf32>
    %v3308 = stablehlo.multiply %v3303, %v3307 : tensor<32x602112xf32>
    %v3309 = stablehlo.add %v3299, %v3308 : tensor<32x602112xf32>
    %v3310 = stablehlo.multiply %v3287, %v3309 : tensor<32x602112xf32>
    %v3311 = stablehlo.reshape %v3310 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3312 = stablehlo.transpose %s1b0eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3313 = stablehlo.reverse %v3312, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3314 = stablehlo.convolution(%v3311, %v3313)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3315 = stablehlo.reshape %v3314 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3317 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v3318 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v3319 = stablehlo.reduce(%v185 init: %v3316) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3320 = stablehlo.broadcast_in_dim %v3319, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3321 = stablehlo.divide %v3320, %v3317 : tensor<32x150528xf32>
    %v3322 = stablehlo.subtract %v185, %v3321 : tensor<32x150528xf32>
    %v3323 = stablehlo.multiply %v3322, %v3322 : tensor<32x150528xf32>
    %v3324 = stablehlo.reduce(%v3323 init: %v3316) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3325 = stablehlo.broadcast_in_dim %v3324, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3326 = stablehlo.divide %v3325, %v3317 : tensor<32x150528xf32>
    %v3327 = stablehlo.add %v3326, %v3318 : tensor<32x150528xf32>
    %v3328 = stablehlo.rsqrt %v3327 : tensor<32x150528xf32>
    %v3329 = stablehlo.multiply %v3322, %v3328 : tensor<32x150528xf32>
    %v3330 = stablehlo.broadcast_in_dim %s1b0ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v3331 = stablehlo.multiply %v3330, %v3315 : tensor<32x150528xf32>
    %v3332 = stablehlo.reduce(%v3331 init: %v3316) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3333 = stablehlo.broadcast_in_dim %v3332, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3334 = stablehlo.multiply %v3329, %v3331 : tensor<32x150528xf32>
    %v3335 = stablehlo.reduce(%v3334 init: %v3316) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3336 = stablehlo.broadcast_in_dim %v3335, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3337 = stablehlo.multiply %v3331, %v3317 : tensor<32x150528xf32>
    %v3338 = stablehlo.subtract %v3337, %v3333 : tensor<32x150528xf32>
    %v3339 = stablehlo.multiply %v3329, %v3336 : tensor<32x150528xf32>
    %v3340 = stablehlo.subtract %v3338, %v3339 : tensor<32x150528xf32>
    %v3341 = stablehlo.divide %v3328, %v3317 : tensor<32x150528xf32>
    %v3342 = stablehlo.multiply %v3341, %v3340 : tensor<32x150528xf32>
    %v3343 = stablehlo.reshape %v3342 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3344 = stablehlo.reverse %s1b0dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v3345 = stablehlo.convolution(%v3343, %v3344)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v3346 = stablehlo.reshape %v3345 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3347 = stablehlo.add %v3346, %v3201 : tensor<32x150528xf32>
    %v3348 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3349 = stablehlo.reshape %v226 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3350 = stablehlo.reshape %v3201 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3351 = stablehlo.multiply %v3349, %v3350 : tensor<32x192x28x28xf32>
    %v3352 = stablehlo.reduce(%v3351 init: %v3348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3353 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3354 = stablehlo.multiply %v3352, %v3353 : tensor<192xf32>
    %v3355 = stablehlo.subtract %s1b0lg, %v3354 : tensor<192xf32>
    %v3356 = stablehlo.reshape %v221 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3357 = stablehlo.reshape %v3282 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3358 = stablehlo.transpose %v3356, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3359 = stablehlo.transpose %v3357, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3360 = stablehlo.convolution(%v3358, %v3359)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v3361 = stablehlo.transpose %v3360, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3362 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v3363 = stablehlo.multiply %v3361, %v3362 : tensor<192x768x1x1xf32>
    %v3364 = stablehlo.subtract %s1b0pW, %v3363 : tensor<192x768x1x1xf32>
    %v3365 = stablehlo.reshape %v3282 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3367 = stablehlo.reduce(%v3365 init: %v3366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3368 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3369 = stablehlo.multiply %v3367, %v3368 : tensor<192xf32>
    %v3370 = stablehlo.subtract %s1b0pb, %v3369 : tensor<192xf32>
    %v3371 = stablehlo.reshape %v203 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3372 = stablehlo.reshape %v3310 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3373 = stablehlo.transpose %v3371, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3374 = stablehlo.transpose %v3372, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3375 = stablehlo.convolution(%v3373, %v3374)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v3376 = stablehlo.transpose %v3375, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3377 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v3378 = stablehlo.multiply %v3376, %v3377 : tensor<768x192x1x1xf32>
    %v3379 = stablehlo.subtract %s1b0eW, %v3378 : tensor<768x192x1x1xf32>
    %v3380 = stablehlo.reshape %v3310 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3382 = stablehlo.reduce(%v3380 init: %v3381) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v3383 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v3384 = stablehlo.multiply %v3382, %v3383 : tensor<768xf32>
    %v3385 = stablehlo.subtract %s1b0eb, %v3384 : tensor<768xf32>
    %v3386 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3387 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v3388 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v3389 = stablehlo.reduce(%v185 init: %v3386) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3390 = stablehlo.broadcast_in_dim %v3389, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3391 = stablehlo.divide %v3390, %v3387 : tensor<32x150528xf32>
    %v3392 = stablehlo.subtract %v185, %v3391 : tensor<32x150528xf32>
    %v3393 = stablehlo.multiply %v3392, %v3392 : tensor<32x150528xf32>
    %v3394 = stablehlo.reduce(%v3393 init: %v3386) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3395 = stablehlo.broadcast_in_dim %v3394, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3396 = stablehlo.divide %v3395, %v3387 : tensor<32x150528xf32>
    %v3397 = stablehlo.add %v3396, %v3388 : tensor<32x150528xf32>
    %v3398 = stablehlo.rsqrt %v3397 : tensor<32x150528xf32>
    %v3399 = stablehlo.multiply %v3392, %v3398 : tensor<32x150528xf32>
    %v3400 = stablehlo.multiply %v3315, %v3399 : tensor<32x150528xf32>
    %v3401 = stablehlo.reduce(%v3400 init: %v3386) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v3402 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3403 = stablehlo.multiply %v3401, %v3402 : tensor<f32>
    %v3404 = stablehlo.subtract %s1b0ng, %v3403 : tensor<f32>
    %v3405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3406 = stablehlo.reduce(%v3315 init: %v3405) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v3407 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3408 = stablehlo.multiply %v3406, %v3407 : tensor<f32>
    %v3409 = stablehlo.subtract %s1b0nbt, %v3408 : tensor<f32>
    %v3410 = stablehlo.reshape %v180 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3411 = stablehlo.reshape %v3342 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3412 = stablehlo.transpose %v3410, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3413 = stablehlo.transpose %v3411, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3414 = stablehlo.convolution(%v3412, %v3413)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v3415 = stablehlo.reshape %v3414 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v3416 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v3417 = stablehlo.multiply %v3415, %v3416 : tensor<192x1x7x7xf32>
    %v3418 = stablehlo.subtract %s1b0dW, %v3417 : tensor<192x1x7x7xf32>
    %v3419 = stablehlo.reshape %v3342 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3421 = stablehlo.reduce(%v3419 init: %v3420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3422 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3423 = stablehlo.multiply %v3421, %v3422 : tensor<192xf32>
    %v3424 = stablehlo.subtract %s1b0db, %v3423 : tensor<192xf32>
    %v3425 = stablehlo.reshape %v3347 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3426 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3427 = stablehlo.pad %v3425, %v3426, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x56x56xf32>
    %v3428 = stablehlo.transpose %d0W, dims = [1, 0, 2, 3] : (tensor<192x96x2x2xf32>) -> tensor<96x192x2x2xf32>
    %v3429 = stablehlo.reverse %v3428, dims = [2, 3] : tensor<96x192x2x2xf32>
    %v3430 = stablehlo.convolution(%v3427, %v3429)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x56x56xf32>, tensor<96x192x2x2xf32>) -> tensor<32x96x56x56xf32>
    %v3431 = stablehlo.reshape %v3430 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3433 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3434 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3435 = stablehlo.reduce(%v157 init: %v3432) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3436 = stablehlo.broadcast_in_dim %v3435, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3437 = stablehlo.divide %v3436, %v3433 : tensor<32x301056xf32>
    %v3438 = stablehlo.subtract %v157, %v3437 : tensor<32x301056xf32>
    %v3439 = stablehlo.multiply %v3438, %v3438 : tensor<32x301056xf32>
    %v3440 = stablehlo.reduce(%v3439 init: %v3432) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3441 = stablehlo.broadcast_in_dim %v3440, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3442 = stablehlo.divide %v3441, %v3433 : tensor<32x301056xf32>
    %v3443 = stablehlo.add %v3442, %v3434 : tensor<32x301056xf32>
    %v3444 = stablehlo.rsqrt %v3443 : tensor<32x301056xf32>
    %v3445 = stablehlo.multiply %v3438, %v3444 : tensor<32x301056xf32>
    %v3446 = stablehlo.broadcast_in_dim %d0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3447 = stablehlo.multiply %v3446, %v3431 : tensor<32x301056xf32>
    %v3448 = stablehlo.reduce(%v3447 init: %v3432) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3449 = stablehlo.broadcast_in_dim %v3448, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3450 = stablehlo.multiply %v3445, %v3447 : tensor<32x301056xf32>
    %v3451 = stablehlo.reduce(%v3450 init: %v3432) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3452 = stablehlo.broadcast_in_dim %v3451, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3453 = stablehlo.multiply %v3447, %v3433 : tensor<32x301056xf32>
    %v3454 = stablehlo.subtract %v3453, %v3449 : tensor<32x301056xf32>
    %v3455 = stablehlo.multiply %v3445, %v3452 : tensor<32x301056xf32>
    %v3456 = stablehlo.subtract %v3454, %v3455 : tensor<32x301056xf32>
    %v3457 = stablehlo.divide %v3444, %v3433 : tensor<32x301056xf32>
    %v3458 = stablehlo.multiply %v3457, %v3456 : tensor<32x301056xf32>
    %v3459 = stablehlo.reshape %v3347 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3460 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3461 = stablehlo.reduce(%v3459 init: %v3460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3462 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3463 = stablehlo.multiply %v3461, %v3462 : tensor<192xf32>
    %v3464 = stablehlo.subtract %d0b, %v3463 : tensor<192xf32>
    %v3465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3466 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3467 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3468 = stablehlo.reduce(%v157 init: %v3465) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3469 = stablehlo.broadcast_in_dim %v3468, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3470 = stablehlo.divide %v3469, %v3466 : tensor<32x301056xf32>
    %v3471 = stablehlo.subtract %v157, %v3470 : tensor<32x301056xf32>
    %v3472 = stablehlo.multiply %v3471, %v3471 : tensor<32x301056xf32>
    %v3473 = stablehlo.reduce(%v3472 init: %v3465) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3474 = stablehlo.broadcast_in_dim %v3473, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3475 = stablehlo.divide %v3474, %v3466 : tensor<32x301056xf32>
    %v3476 = stablehlo.add %v3475, %v3467 : tensor<32x301056xf32>
    %v3477 = stablehlo.rsqrt %v3476 : tensor<32x301056xf32>
    %v3478 = stablehlo.multiply %v3471, %v3477 : tensor<32x301056xf32>
    %v3479 = stablehlo.multiply %v3431, %v3478 : tensor<32x301056xf32>
    %v3480 = stablehlo.reduce(%v3479 init: %v3465) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3481 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3482 = stablehlo.multiply %v3480, %v3481 : tensor<f32>
    %v3483 = stablehlo.subtract %d0ng, %v3482 : tensor<f32>
    %v3484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3485 = stablehlo.reduce(%v3431 init: %v3484) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3486 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3487 = stablehlo.multiply %v3485, %v3486 : tensor<f32>
    %v3488 = stablehlo.subtract %d0nbt, %v3487 : tensor<f32>
    %v3489 = stablehlo.reshape %v175 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3490 = stablehlo.reshape %v3347 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3492 = stablehlo.pad %v3490, %v3491, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x55x55xf32>
    %v3493 = stablehlo.transpose %v3489, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3494 = stablehlo.transpose %v3492, dims = [1, 0, 2, 3] : (tensor<32x192x55x55xf32>) -> tensor<192x32x55x55xf32>
    %v3495 = stablehlo.convolution(%v3493, %v3494)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<192x32x55x55xf32>) -> tensor<96x192x2x2xf32>
    %v3496 = stablehlo.transpose %v3495, dims = [1, 0, 2, 3] : (tensor<96x192x2x2xf32>) -> tensor<192x96x2x2xf32>
    %v3497 = stablehlo.constant dense<0.1> : tensor<192x96x2x2xf32>
    %v3498 = stablehlo.multiply %v3496, %v3497 : tensor<192x96x2x2xf32>
    %v3499 = stablehlo.subtract %d0W, %v3498 : tensor<192x96x2x2xf32>
    %v3500 = stablehlo.reshape %v3458 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3501 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3502 = stablehlo.multiply %v3500, %v3501 : tensor<32x96x56x56xf32>
    %v3503 = stablehlo.reshape %v3502 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3504 = stablehlo.reshape %v3503 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3505 = stablehlo.transpose %s0b2pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3506 = stablehlo.reverse %v3505, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3507 = stablehlo.convolution(%v3504, %v3506)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3508 = stablehlo.reshape %v3507 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3509 = stablehlo.multiply %v134, %v134 : tensor<32x1204224xf32>
    %v3510 = stablehlo.multiply %v3509, %v134 : tensor<32x1204224xf32>
    %v3511 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3512 = stablehlo.multiply %v3511, %v3510 : tensor<32x1204224xf32>
    %v3513 = stablehlo.add %v134, %v3512 : tensor<32x1204224xf32>
    %v3514 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3515 = stablehlo.multiply %v3514, %v3513 : tensor<32x1204224xf32>
    %v3516 = stablehlo.tanh %v3515 : tensor<32x1204224xf32>
    %v3517 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3518 = stablehlo.add %v3517, %v3516 : tensor<32x1204224xf32>
    %v3519 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3520 = stablehlo.multiply %v3519, %v3518 : tensor<32x1204224xf32>
    %v3521 = stablehlo.multiply %v3516, %v3516 : tensor<32x1204224xf32>
    %v3522 = stablehlo.subtract %v3517, %v3521 : tensor<32x1204224xf32>
    %v3523 = stablehlo.multiply %v3519, %v134 : tensor<32x1204224xf32>
    %v3524 = stablehlo.multiply %v3523, %v3522 : tensor<32x1204224xf32>
    %v3525 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3526 = stablehlo.multiply %v3525, %v3509 : tensor<32x1204224xf32>
    %v3527 = stablehlo.add %v3517, %v3526 : tensor<32x1204224xf32>
    %v3528 = stablehlo.multiply %v3514, %v3527 : tensor<32x1204224xf32>
    %v3529 = stablehlo.multiply %v3524, %v3528 : tensor<32x1204224xf32>
    %v3530 = stablehlo.add %v3520, %v3529 : tensor<32x1204224xf32>
    %v3531 = stablehlo.multiply %v3508, %v3530 : tensor<32x1204224xf32>
    %v3532 = stablehlo.reshape %v3531 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3533 = stablehlo.transpose %s0b2eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3534 = stablehlo.reverse %v3533, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3535 = stablehlo.convolution(%v3532, %v3534)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3536 = stablehlo.reshape %v3535 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3538 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3539 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3540 = stablehlo.reduce(%v111 init: %v3537) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3541 = stablehlo.broadcast_in_dim %v3540, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3542 = stablehlo.divide %v3541, %v3538 : tensor<32x301056xf32>
    %v3543 = stablehlo.subtract %v111, %v3542 : tensor<32x301056xf32>
    %v3544 = stablehlo.multiply %v3543, %v3543 : tensor<32x301056xf32>
    %v3545 = stablehlo.reduce(%v3544 init: %v3537) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3546 = stablehlo.broadcast_in_dim %v3545, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3547 = stablehlo.divide %v3546, %v3538 : tensor<32x301056xf32>
    %v3548 = stablehlo.add %v3547, %v3539 : tensor<32x301056xf32>
    %v3549 = stablehlo.rsqrt %v3548 : tensor<32x301056xf32>
    %v3550 = stablehlo.multiply %v3543, %v3549 : tensor<32x301056xf32>
    %v3551 = stablehlo.broadcast_in_dim %s0b2ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3552 = stablehlo.multiply %v3551, %v3536 : tensor<32x301056xf32>
    %v3553 = stablehlo.reduce(%v3552 init: %v3537) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3554 = stablehlo.broadcast_in_dim %v3553, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3555 = stablehlo.multiply %v3550, %v3552 : tensor<32x301056xf32>
    %v3556 = stablehlo.reduce(%v3555 init: %v3537) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3557 = stablehlo.broadcast_in_dim %v3556, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3558 = stablehlo.multiply %v3552, %v3538 : tensor<32x301056xf32>
    %v3559 = stablehlo.subtract %v3558, %v3554 : tensor<32x301056xf32>
    %v3560 = stablehlo.multiply %v3550, %v3557 : tensor<32x301056xf32>
    %v3561 = stablehlo.subtract %v3559, %v3560 : tensor<32x301056xf32>
    %v3562 = stablehlo.divide %v3549, %v3538 : tensor<32x301056xf32>
    %v3563 = stablehlo.multiply %v3562, %v3561 : tensor<32x301056xf32>
    %v3564 = stablehlo.reshape %v3563 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3565 = stablehlo.reverse %s0b2dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v3566 = stablehlo.convolution(%v3564, %v3565)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v3567 = stablehlo.reshape %v3566 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3568 = stablehlo.add %v3567, %v3458 : tensor<32x301056xf32>
    %v3569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3570 = stablehlo.reshape %v152 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3571 = stablehlo.reshape %v3458 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3572 = stablehlo.multiply %v3570, %v3571 : tensor<32x96x56x56xf32>
    %v3573 = stablehlo.reduce(%v3572 init: %v3569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3574 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3575 = stablehlo.multiply %v3573, %v3574 : tensor<96xf32>
    %v3576 = stablehlo.subtract %s0b2lg, %v3575 : tensor<96xf32>
    %v3577 = stablehlo.reshape %v147 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3578 = stablehlo.reshape %v3503 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3579 = stablehlo.transpose %v3577, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3580 = stablehlo.transpose %v3578, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3581 = stablehlo.convolution(%v3579, %v3580)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v3582 = stablehlo.transpose %v3581, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3583 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v3584 = stablehlo.multiply %v3582, %v3583 : tensor<96x384x1x1xf32>
    %v3585 = stablehlo.subtract %s0b2pW, %v3584 : tensor<96x384x1x1xf32>
    %v3586 = stablehlo.reshape %v3503 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3587 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3588 = stablehlo.reduce(%v3586 init: %v3587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3589 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3590 = stablehlo.multiply %v3588, %v3589 : tensor<96xf32>
    %v3591 = stablehlo.subtract %s0b2pb, %v3590 : tensor<96xf32>
    %v3592 = stablehlo.reshape %v129 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3593 = stablehlo.reshape %v3531 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3594 = stablehlo.transpose %v3592, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3595 = stablehlo.transpose %v3593, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3596 = stablehlo.convolution(%v3594, %v3595)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v3597 = stablehlo.transpose %v3596, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3598 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v3599 = stablehlo.multiply %v3597, %v3598 : tensor<384x96x1x1xf32>
    %v3600 = stablehlo.subtract %s0b2eW, %v3599 : tensor<384x96x1x1xf32>
    %v3601 = stablehlo.reshape %v3531 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3603 = stablehlo.reduce(%v3601 init: %v3602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v3604 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3605 = stablehlo.multiply %v3603, %v3604 : tensor<384xf32>
    %v3606 = stablehlo.subtract %s0b2eb, %v3605 : tensor<384xf32>
    %v3607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3608 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3609 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3610 = stablehlo.reduce(%v111 init: %v3607) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3611 = stablehlo.broadcast_in_dim %v3610, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3612 = stablehlo.divide %v3611, %v3608 : tensor<32x301056xf32>
    %v3613 = stablehlo.subtract %v111, %v3612 : tensor<32x301056xf32>
    %v3614 = stablehlo.multiply %v3613, %v3613 : tensor<32x301056xf32>
    %v3615 = stablehlo.reduce(%v3614 init: %v3607) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3616 = stablehlo.broadcast_in_dim %v3615, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3617 = stablehlo.divide %v3616, %v3608 : tensor<32x301056xf32>
    %v3618 = stablehlo.add %v3617, %v3609 : tensor<32x301056xf32>
    %v3619 = stablehlo.rsqrt %v3618 : tensor<32x301056xf32>
    %v3620 = stablehlo.multiply %v3613, %v3619 : tensor<32x301056xf32>
    %v3621 = stablehlo.multiply %v3536, %v3620 : tensor<32x301056xf32>
    %v3622 = stablehlo.reduce(%v3621 init: %v3607) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3623 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3624 = stablehlo.multiply %v3622, %v3623 : tensor<f32>
    %v3625 = stablehlo.subtract %s0b2ng, %v3624 : tensor<f32>
    %v3626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3627 = stablehlo.reduce(%v3536 init: %v3626) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3628 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3629 = stablehlo.multiply %v3627, %v3628 : tensor<f32>
    %v3630 = stablehlo.subtract %s0b2nbt, %v3629 : tensor<f32>
    %v3631 = stablehlo.reshape %v106 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3632 = stablehlo.reshape %v3563 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3633 = stablehlo.transpose %v3631, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3634 = stablehlo.transpose %v3632, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3635 = stablehlo.convolution(%v3633, %v3634)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v3636 = stablehlo.reshape %v3635 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v3637 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v3638 = stablehlo.multiply %v3636, %v3637 : tensor<96x1x7x7xf32>
    %v3639 = stablehlo.subtract %s0b2dW, %v3638 : tensor<96x1x7x7xf32>
    %v3640 = stablehlo.reshape %v3563 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3642 = stablehlo.reduce(%v3640 init: %v3641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3643 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3644 = stablehlo.multiply %v3642, %v3643 : tensor<96xf32>
    %v3645 = stablehlo.subtract %s0b2db, %v3644 : tensor<96xf32>
    %v3646 = stablehlo.reshape %v3568 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3647 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3648 = stablehlo.multiply %v3646, %v3647 : tensor<32x96x56x56xf32>
    %v3649 = stablehlo.reshape %v3648 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3650 = stablehlo.reshape %v3649 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3651 = stablehlo.transpose %s0b1pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3652 = stablehlo.reverse %v3651, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3653 = stablehlo.convolution(%v3650, %v3652)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3654 = stablehlo.reshape %v3653 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3655 = stablehlo.multiply %v83, %v83 : tensor<32x1204224xf32>
    %v3656 = stablehlo.multiply %v3655, %v83 : tensor<32x1204224xf32>
    %v3657 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3658 = stablehlo.multiply %v3657, %v3656 : tensor<32x1204224xf32>
    %v3659 = stablehlo.add %v83, %v3658 : tensor<32x1204224xf32>
    %v3660 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3661 = stablehlo.multiply %v3660, %v3659 : tensor<32x1204224xf32>
    %v3662 = stablehlo.tanh %v3661 : tensor<32x1204224xf32>
    %v3663 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3664 = stablehlo.add %v3663, %v3662 : tensor<32x1204224xf32>
    %v3665 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3666 = stablehlo.multiply %v3665, %v3664 : tensor<32x1204224xf32>
    %v3667 = stablehlo.multiply %v3662, %v3662 : tensor<32x1204224xf32>
    %v3668 = stablehlo.subtract %v3663, %v3667 : tensor<32x1204224xf32>
    %v3669 = stablehlo.multiply %v3665, %v83 : tensor<32x1204224xf32>
    %v3670 = stablehlo.multiply %v3669, %v3668 : tensor<32x1204224xf32>
    %v3671 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3672 = stablehlo.multiply %v3671, %v3655 : tensor<32x1204224xf32>
    %v3673 = stablehlo.add %v3663, %v3672 : tensor<32x1204224xf32>
    %v3674 = stablehlo.multiply %v3660, %v3673 : tensor<32x1204224xf32>
    %v3675 = stablehlo.multiply %v3670, %v3674 : tensor<32x1204224xf32>
    %v3676 = stablehlo.add %v3666, %v3675 : tensor<32x1204224xf32>
    %v3677 = stablehlo.multiply %v3654, %v3676 : tensor<32x1204224xf32>
    %v3678 = stablehlo.reshape %v3677 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3679 = stablehlo.transpose %s0b1eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3680 = stablehlo.reverse %v3679, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3681 = stablehlo.convolution(%v3678, %v3680)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3682 = stablehlo.reshape %v3681 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3683 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3684 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3685 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3686 = stablehlo.reduce(%v60 init: %v3683) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3687 = stablehlo.broadcast_in_dim %v3686, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3688 = stablehlo.divide %v3687, %v3684 : tensor<32x301056xf32>
    %v3689 = stablehlo.subtract %v60, %v3688 : tensor<32x301056xf32>
    %v3690 = stablehlo.multiply %v3689, %v3689 : tensor<32x301056xf32>
    %v3691 = stablehlo.reduce(%v3690 init: %v3683) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3692 = stablehlo.broadcast_in_dim %v3691, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3693 = stablehlo.divide %v3692, %v3684 : tensor<32x301056xf32>
    %v3694 = stablehlo.add %v3693, %v3685 : tensor<32x301056xf32>
    %v3695 = stablehlo.rsqrt %v3694 : tensor<32x301056xf32>
    %v3696 = stablehlo.multiply %v3689, %v3695 : tensor<32x301056xf32>
    %v3697 = stablehlo.broadcast_in_dim %s0b1ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3698 = stablehlo.multiply %v3697, %v3682 : tensor<32x301056xf32>
    %v3699 = stablehlo.reduce(%v3698 init: %v3683) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3700 = stablehlo.broadcast_in_dim %v3699, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3701 = stablehlo.multiply %v3696, %v3698 : tensor<32x301056xf32>
    %v3702 = stablehlo.reduce(%v3701 init: %v3683) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3703 = stablehlo.broadcast_in_dim %v3702, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3704 = stablehlo.multiply %v3698, %v3684 : tensor<32x301056xf32>
    %v3705 = stablehlo.subtract %v3704, %v3700 : tensor<32x301056xf32>
    %v3706 = stablehlo.multiply %v3696, %v3703 : tensor<32x301056xf32>
    %v3707 = stablehlo.subtract %v3705, %v3706 : tensor<32x301056xf32>
    %v3708 = stablehlo.divide %v3695, %v3684 : tensor<32x301056xf32>
    %v3709 = stablehlo.multiply %v3708, %v3707 : tensor<32x301056xf32>
    %v3710 = stablehlo.reshape %v3709 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3711 = stablehlo.reverse %s0b1dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v3712 = stablehlo.convolution(%v3710, %v3711)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v3713 = stablehlo.reshape %v3712 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3714 = stablehlo.add %v3713, %v3568 : tensor<32x301056xf32>
    %v3715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3716 = stablehlo.reshape %v101 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3717 = stablehlo.reshape %v3568 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3718 = stablehlo.multiply %v3716, %v3717 : tensor<32x96x56x56xf32>
    %v3719 = stablehlo.reduce(%v3718 init: %v3715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3720 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3721 = stablehlo.multiply %v3719, %v3720 : tensor<96xf32>
    %v3722 = stablehlo.subtract %s0b1lg, %v3721 : tensor<96xf32>
    %v3723 = stablehlo.reshape %v96 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3724 = stablehlo.reshape %v3649 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3725 = stablehlo.transpose %v3723, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3726 = stablehlo.transpose %v3724, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3727 = stablehlo.convolution(%v3725, %v3726)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v3728 = stablehlo.transpose %v3727, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3729 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v3730 = stablehlo.multiply %v3728, %v3729 : tensor<96x384x1x1xf32>
    %v3731 = stablehlo.subtract %s0b1pW, %v3730 : tensor<96x384x1x1xf32>
    %v3732 = stablehlo.reshape %v3649 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3734 = stablehlo.reduce(%v3732 init: %v3733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3735 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3736 = stablehlo.multiply %v3734, %v3735 : tensor<96xf32>
    %v3737 = stablehlo.subtract %s0b1pb, %v3736 : tensor<96xf32>
    %v3738 = stablehlo.reshape %v78 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3739 = stablehlo.reshape %v3677 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3740 = stablehlo.transpose %v3738, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3741 = stablehlo.transpose %v3739, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3742 = stablehlo.convolution(%v3740, %v3741)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v3743 = stablehlo.transpose %v3742, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3744 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v3745 = stablehlo.multiply %v3743, %v3744 : tensor<384x96x1x1xf32>
    %v3746 = stablehlo.subtract %s0b1eW, %v3745 : tensor<384x96x1x1xf32>
    %v3747 = stablehlo.reshape %v3677 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3748 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3749 = stablehlo.reduce(%v3747 init: %v3748) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v3750 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3751 = stablehlo.multiply %v3749, %v3750 : tensor<384xf32>
    %v3752 = stablehlo.subtract %s0b1eb, %v3751 : tensor<384xf32>
    %v3753 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3754 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3755 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3756 = stablehlo.reduce(%v60 init: %v3753) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3757 = stablehlo.broadcast_in_dim %v3756, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3758 = stablehlo.divide %v3757, %v3754 : tensor<32x301056xf32>
    %v3759 = stablehlo.subtract %v60, %v3758 : tensor<32x301056xf32>
    %v3760 = stablehlo.multiply %v3759, %v3759 : tensor<32x301056xf32>
    %v3761 = stablehlo.reduce(%v3760 init: %v3753) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3762 = stablehlo.broadcast_in_dim %v3761, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3763 = stablehlo.divide %v3762, %v3754 : tensor<32x301056xf32>
    %v3764 = stablehlo.add %v3763, %v3755 : tensor<32x301056xf32>
    %v3765 = stablehlo.rsqrt %v3764 : tensor<32x301056xf32>
    %v3766 = stablehlo.multiply %v3759, %v3765 : tensor<32x301056xf32>
    %v3767 = stablehlo.multiply %v3682, %v3766 : tensor<32x301056xf32>
    %v3768 = stablehlo.reduce(%v3767 init: %v3753) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3769 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3770 = stablehlo.multiply %v3768, %v3769 : tensor<f32>
    %v3771 = stablehlo.subtract %s0b1ng, %v3770 : tensor<f32>
    %v3772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3773 = stablehlo.reduce(%v3682 init: %v3772) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3774 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3775 = stablehlo.multiply %v3773, %v3774 : tensor<f32>
    %v3776 = stablehlo.subtract %s0b1nbt, %v3775 : tensor<f32>
    %v3777 = stablehlo.reshape %v55 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3778 = stablehlo.reshape %v3709 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3779 = stablehlo.transpose %v3777, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3780 = stablehlo.transpose %v3778, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3781 = stablehlo.convolution(%v3779, %v3780)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v3782 = stablehlo.reshape %v3781 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v3783 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v3784 = stablehlo.multiply %v3782, %v3783 : tensor<96x1x7x7xf32>
    %v3785 = stablehlo.subtract %s0b1dW, %v3784 : tensor<96x1x7x7xf32>
    %v3786 = stablehlo.reshape %v3709 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3787 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3788 = stablehlo.reduce(%v3786 init: %v3787) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3789 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3790 = stablehlo.multiply %v3788, %v3789 : tensor<96xf32>
    %v3791 = stablehlo.subtract %s0b1db, %v3790 : tensor<96xf32>
    %v3792 = stablehlo.reshape %v3714 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3793 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3794 = stablehlo.multiply %v3792, %v3793 : tensor<32x96x56x56xf32>
    %v3795 = stablehlo.reshape %v3794 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3796 = stablehlo.reshape %v3795 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3797 = stablehlo.transpose %s0b0pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3798 = stablehlo.reverse %v3797, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3799 = stablehlo.convolution(%v3796, %v3798)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3800 = stablehlo.reshape %v3799 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3801 = stablehlo.multiply %v32, %v32 : tensor<32x1204224xf32>
    %v3802 = stablehlo.multiply %v3801, %v32 : tensor<32x1204224xf32>
    %v3803 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3804 = stablehlo.multiply %v3803, %v3802 : tensor<32x1204224xf32>
    %v3805 = stablehlo.add %v32, %v3804 : tensor<32x1204224xf32>
    %v3806 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3807 = stablehlo.multiply %v3806, %v3805 : tensor<32x1204224xf32>
    %v3808 = stablehlo.tanh %v3807 : tensor<32x1204224xf32>
    %v3809 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3810 = stablehlo.add %v3809, %v3808 : tensor<32x1204224xf32>
    %v3811 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3812 = stablehlo.multiply %v3811, %v3810 : tensor<32x1204224xf32>
    %v3813 = stablehlo.multiply %v3808, %v3808 : tensor<32x1204224xf32>
    %v3814 = stablehlo.subtract %v3809, %v3813 : tensor<32x1204224xf32>
    %v3815 = stablehlo.multiply %v3811, %v32 : tensor<32x1204224xf32>
    %v3816 = stablehlo.multiply %v3815, %v3814 : tensor<32x1204224xf32>
    %v3817 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3818 = stablehlo.multiply %v3817, %v3801 : tensor<32x1204224xf32>
    %v3819 = stablehlo.add %v3809, %v3818 : tensor<32x1204224xf32>
    %v3820 = stablehlo.multiply %v3806, %v3819 : tensor<32x1204224xf32>
    %v3821 = stablehlo.multiply %v3816, %v3820 : tensor<32x1204224xf32>
    %v3822 = stablehlo.add %v3812, %v3821 : tensor<32x1204224xf32>
    %v3823 = stablehlo.multiply %v3800, %v3822 : tensor<32x1204224xf32>
    %v3824 = stablehlo.reshape %v3823 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3825 = stablehlo.transpose %s0b0eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3826 = stablehlo.reverse %v3825, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3827 = stablehlo.convolution(%v3824, %v3826)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3828 = stablehlo.reshape %v3827 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3830 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3831 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3832 = stablehlo.reduce(%v9 init: %v3829) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3833 = stablehlo.broadcast_in_dim %v3832, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3834 = stablehlo.divide %v3833, %v3830 : tensor<32x301056xf32>
    %v3835 = stablehlo.subtract %v9, %v3834 : tensor<32x301056xf32>
    %v3836 = stablehlo.multiply %v3835, %v3835 : tensor<32x301056xf32>
    %v3837 = stablehlo.reduce(%v3836 init: %v3829) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3838 = stablehlo.broadcast_in_dim %v3837, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3839 = stablehlo.divide %v3838, %v3830 : tensor<32x301056xf32>
    %v3840 = stablehlo.add %v3839, %v3831 : tensor<32x301056xf32>
    %v3841 = stablehlo.rsqrt %v3840 : tensor<32x301056xf32>
    %v3842 = stablehlo.multiply %v3835, %v3841 : tensor<32x301056xf32>
    %v3843 = stablehlo.broadcast_in_dim %s0b0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3844 = stablehlo.multiply %v3843, %v3828 : tensor<32x301056xf32>
    %v3845 = stablehlo.reduce(%v3844 init: %v3829) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3846 = stablehlo.broadcast_in_dim %v3845, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3847 = stablehlo.multiply %v3842, %v3844 : tensor<32x301056xf32>
    %v3848 = stablehlo.reduce(%v3847 init: %v3829) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3849 = stablehlo.broadcast_in_dim %v3848, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3850 = stablehlo.multiply %v3844, %v3830 : tensor<32x301056xf32>
    %v3851 = stablehlo.subtract %v3850, %v3846 : tensor<32x301056xf32>
    %v3852 = stablehlo.multiply %v3842, %v3849 : tensor<32x301056xf32>
    %v3853 = stablehlo.subtract %v3851, %v3852 : tensor<32x301056xf32>
    %v3854 = stablehlo.divide %v3841, %v3830 : tensor<32x301056xf32>
    %v3855 = stablehlo.multiply %v3854, %v3853 : tensor<32x301056xf32>
    %v3856 = stablehlo.reshape %v3855 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3857 = stablehlo.reverse %s0b0dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v3858 = stablehlo.convolution(%v3856, %v3857)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v3859 = stablehlo.reshape %v3858 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3860 = stablehlo.add %v3859, %v3714 : tensor<32x301056xf32>
    %v3861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3862 = stablehlo.reshape %v50 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3863 = stablehlo.reshape %v3714 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3864 = stablehlo.multiply %v3862, %v3863 : tensor<32x96x56x56xf32>
    %v3865 = stablehlo.reduce(%v3864 init: %v3861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3866 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3867 = stablehlo.multiply %v3865, %v3866 : tensor<96xf32>
    %v3868 = stablehlo.subtract %s0b0lg, %v3867 : tensor<96xf32>
    %v3869 = stablehlo.reshape %v45 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3870 = stablehlo.reshape %v3795 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3871 = stablehlo.transpose %v3869, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3872 = stablehlo.transpose %v3870, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3873 = stablehlo.convolution(%v3871, %v3872)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v3874 = stablehlo.transpose %v3873, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3875 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v3876 = stablehlo.multiply %v3874, %v3875 : tensor<96x384x1x1xf32>
    %v3877 = stablehlo.subtract %s0b0pW, %v3876 : tensor<96x384x1x1xf32>
    %v3878 = stablehlo.reshape %v3795 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3880 = stablehlo.reduce(%v3878 init: %v3879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3881 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3882 = stablehlo.multiply %v3880, %v3881 : tensor<96xf32>
    %v3883 = stablehlo.subtract %s0b0pb, %v3882 : tensor<96xf32>
    %v3884 = stablehlo.reshape %v27 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3885 = stablehlo.reshape %v3823 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3886 = stablehlo.transpose %v3884, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3887 = stablehlo.transpose %v3885, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3888 = stablehlo.convolution(%v3886, %v3887)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v3889 = stablehlo.transpose %v3888, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3890 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v3891 = stablehlo.multiply %v3889, %v3890 : tensor<384x96x1x1xf32>
    %v3892 = stablehlo.subtract %s0b0eW, %v3891 : tensor<384x96x1x1xf32>
    %v3893 = stablehlo.reshape %v3823 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3895 = stablehlo.reduce(%v3893 init: %v3894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v3896 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3897 = stablehlo.multiply %v3895, %v3896 : tensor<384xf32>
    %v3898 = stablehlo.subtract %s0b0eb, %v3897 : tensor<384xf32>
    %v3899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3900 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3901 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3902 = stablehlo.reduce(%v9 init: %v3899) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3903 = stablehlo.broadcast_in_dim %v3902, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3904 = stablehlo.divide %v3903, %v3900 : tensor<32x301056xf32>
    %v3905 = stablehlo.subtract %v9, %v3904 : tensor<32x301056xf32>
    %v3906 = stablehlo.multiply %v3905, %v3905 : tensor<32x301056xf32>
    %v3907 = stablehlo.reduce(%v3906 init: %v3899) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3908 = stablehlo.broadcast_in_dim %v3907, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3909 = stablehlo.divide %v3908, %v3900 : tensor<32x301056xf32>
    %v3910 = stablehlo.add %v3909, %v3901 : tensor<32x301056xf32>
    %v3911 = stablehlo.rsqrt %v3910 : tensor<32x301056xf32>
    %v3912 = stablehlo.multiply %v3905, %v3911 : tensor<32x301056xf32>
    %v3913 = stablehlo.multiply %v3828, %v3912 : tensor<32x301056xf32>
    %v3914 = stablehlo.reduce(%v3913 init: %v3899) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3915 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3916 = stablehlo.multiply %v3914, %v3915 : tensor<f32>
    %v3917 = stablehlo.subtract %s0b0ng, %v3916 : tensor<f32>
    %v3918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3919 = stablehlo.reduce(%v3828 init: %v3918) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3920 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3921 = stablehlo.multiply %v3919, %v3920 : tensor<f32>
    %v3922 = stablehlo.subtract %s0b0nbt, %v3921 : tensor<f32>
    %v3923 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3924 = stablehlo.reshape %v3855 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3925 = stablehlo.transpose %v3923, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3926 = stablehlo.transpose %v3924, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3927 = stablehlo.convolution(%v3925, %v3926)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v3928 = stablehlo.reshape %v3927 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v3929 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v3930 = stablehlo.multiply %v3928, %v3929 : tensor<96x1x7x7xf32>
    %v3931 = stablehlo.subtract %s0b0dW, %v3930 : tensor<96x1x7x7xf32>
    %v3932 = stablehlo.reshape %v3855 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3934 = stablehlo.reduce(%v3932 init: %v3933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3935 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3936 = stablehlo.multiply %v3934, %v3935 : tensor<96xf32>
    %v3937 = stablehlo.subtract %s0b0db, %v3936 : tensor<96xf32>
    %v3944 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v3945 = stablehlo.reshape %v3860 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3946 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3947 = stablehlo.pad %v3945, %v3946, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x221x221xf32>
    %v3948 = stablehlo.transpose %v3944, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v3949 = stablehlo.transpose %v3947, dims = [1, 0, 2, 3] : (tensor<32x96x221x221xf32>) -> tensor<96x32x221x221xf32>
    %v3950 = stablehlo.convolution(%v3948, %v3949)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<96x32x221x221xf32>) -> tensor<3x96x4x4xf32>
    %v3951 = stablehlo.transpose %v3950, dims = [1, 0, 2, 3] : (tensor<3x96x4x4xf32>) -> tensor<96x3x4x4xf32>
    %psWl = stablehlo.constant dense<0.1> : tensor<96x3x4x4xf32>
    %psWs = stablehlo.multiply %v3951, %psWl : tensor<96x3x4x4xf32>
    %psWn = stablehlo.subtract %psW, %psWs : tensor<96x3x4x4xf32>
    %v3938 = stablehlo.reshape %v3860 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3940 = stablehlo.reduce(%v3938 init: %v3939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3941 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3942 = stablehlo.multiply %v3940, %v3941 : tensor<96xf32>
    %v3943 = stablehlo.subtract %psb, %v3942 : tensor<96xf32>
    return %psWn, %v3943, %v3931, %v3937, %v3917, %v3922, %v3892, %v3898, %v3877, %v3883, %v3868, %v3785, %v3791, %v3771, %v3776, %v3746, %v3752, %v3731, %v3737, %v3722, %v3639, %v3645, %v3625, %v3630, %v3600, %v3606, %v3585, %v3591, %v3576, %v3483, %v3488, %v3499, %v3464, %v3418, %v3424, %v3404, %v3409, %v3379, %v3385, %v3364, %v3370, %v3355, %v3272, %v3278, %v3258, %v3263, %v3233, %v3239, %v3218, %v3224, %v3209, %v3126, %v3132, %v3112, %v3117, %v3087, %v3093, %v3072, %v3078, %v3063, %v2970, %v2975, %v2986, %v2951, %v2905, %v2911, %v2891, %v2896, %v2866, %v2872, %v2851, %v2857, %v2842, %v2759, %v2765, %v2745, %v2750, %v2720, %v2726, %v2705, %v2711, %v2696, %v2613, %v2619, %v2599, %v2604, %v2574, %v2580, %v2559, %v2565, %v2550, %v2467, %v2473, %v2453, %v2458, %v2428, %v2434, %v2413, %v2419, %v2404, %v2321, %v2327, %v2307, %v2312, %v2282, %v2288, %v2267, %v2273, %v2258, %v2175, %v2181, %v2161, %v2166, %v2136, %v2142, %v2121, %v2127, %v2112, %v2029, %v2035, %v2015, %v2020, %v1990, %v1996, %v1975, %v1981, %v1966, %v1883, %v1889, %v1869, %v1874, %v1844, %v1850, %v1829, %v1835, %v1820, %v1737, %v1743, %v1723, %v1728, %v1698, %v1704, %v1683, %v1689, %v1674, %v1581, %v1586, %v1597, %v1562, %v1516, %v1522, %v1502, %v1507, %v1477, %v1483, %v1462, %v1468, %v1453, %v1370, %v1376, %v1356, %v1361, %v1331, %v1337, %v1316, %v1322, %v1307, %v1224, %v1230, %v1210, %v1215, %v1185, %v1191, %v1170, %v1176, %v1161, %v1079, %v1084, %v1055, %v1060 : tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>
  }
}
