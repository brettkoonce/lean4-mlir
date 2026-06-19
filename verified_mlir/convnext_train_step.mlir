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
    %dd2Wxi = stablehlo.reshape %v833 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %dd2Wdi = stablehlo.reshape %v1445 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %dd2Wu = stablehlo.pad %dd2Wdi, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x13x13xf32>
    %dd2Wxt = stablehlo.transpose %dd2Wxi, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %dd2Wdt = stablehlo.transpose %dd2Wu, dims = [1, 0, 2, 3] : (tensor<32x768x13x13xf32>) -> tensor<768x32x13x13xf32>
    %dd2Wraw = stablehlo.convolution(%dd2Wxt, %dd2Wdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<768x32x13x13xf32>) -> tensor<384x768x2x2xf32>
    %dd2W = stablehlo.transpose %dd2Wraw, dims = [1, 0, 2, 3] : (tensor<384x768x2x2xf32>) -> tensor<768x384x2x2xf32>
    %d2Wl = stablehlo.constant dense<0.1> : tensor<768x384x2x2xf32>
    %d2Ws = stablehlo.multiply %dd2W, %d2Wl : tensor<768x384x2x2xf32>
    %d2Wn = stablehlo.subtract %d2W, %d2Ws : tensor<768x384x2x2xf32>
    %v1587 = stablehlo.reshape %v1556 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1588 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1589 = stablehlo.multiply %v1587, %v1588 : tensor<32x384x14x14xf32>
    %v1590 = stablehlo.reshape %v1589 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1591 = stablehlo.reshape %v1590 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1592 = stablehlo.transpose %s2b8pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1593 = stablehlo.reverse %v1592, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1594 = stablehlo.convolution(%v1591, %v1593)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1595 = stablehlo.reshape %v1594 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1596 = stablehlo.multiply %v792, %v792 : tensor<32x301056xf32>
    %v1597 = stablehlo.multiply %v1596, %v792 : tensor<32x301056xf32>
    %v1598 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1599 = stablehlo.multiply %v1598, %v1597 : tensor<32x301056xf32>
    %v1600 = stablehlo.add %v792, %v1599 : tensor<32x301056xf32>
    %v1601 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1602 = stablehlo.multiply %v1601, %v1600 : tensor<32x301056xf32>
    %v1603 = stablehlo.tanh %v1602 : tensor<32x301056xf32>
    %v1604 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1605 = stablehlo.add %v1604, %v1603 : tensor<32x301056xf32>
    %v1606 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1607 = stablehlo.multiply %v1606, %v1605 : tensor<32x301056xf32>
    %v1608 = stablehlo.multiply %v1603, %v1603 : tensor<32x301056xf32>
    %v1609 = stablehlo.subtract %v1604, %v1608 : tensor<32x301056xf32>
    %v1610 = stablehlo.multiply %v1606, %v792 : tensor<32x301056xf32>
    %v1611 = stablehlo.multiply %v1610, %v1609 : tensor<32x301056xf32>
    %v1612 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1613 = stablehlo.multiply %v1612, %v1596 : tensor<32x301056xf32>
    %v1614 = stablehlo.add %v1604, %v1613 : tensor<32x301056xf32>
    %v1615 = stablehlo.multiply %v1601, %v1614 : tensor<32x301056xf32>
    %v1616 = stablehlo.multiply %v1611, %v1615 : tensor<32x301056xf32>
    %v1617 = stablehlo.add %v1607, %v1616 : tensor<32x301056xf32>
    %v1618 = stablehlo.multiply %v1595, %v1617 : tensor<32x301056xf32>
    %v1619 = stablehlo.reshape %v1618 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1620 = stablehlo.transpose %s2b8eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1621 = stablehlo.reverse %v1620, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1622 = stablehlo.convolution(%v1619, %v1621)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1623 = stablehlo.reshape %v1622 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1625 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1626 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1627 = stablehlo.reduce(%v769 init: %v1624) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1628 = stablehlo.broadcast_in_dim %v1627, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1629 = stablehlo.divide %v1628, %v1625 : tensor<32x75264xf32>
    %v1630 = stablehlo.subtract %v769, %v1629 : tensor<32x75264xf32>
    %v1631 = stablehlo.multiply %v1630, %v1630 : tensor<32x75264xf32>
    %v1632 = stablehlo.reduce(%v1631 init: %v1624) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1633 = stablehlo.broadcast_in_dim %v1632, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1634 = stablehlo.divide %v1633, %v1625 : tensor<32x75264xf32>
    %v1635 = stablehlo.add %v1634, %v1626 : tensor<32x75264xf32>
    %v1636 = stablehlo.rsqrt %v1635 : tensor<32x75264xf32>
    %v1637 = stablehlo.multiply %v1630, %v1636 : tensor<32x75264xf32>
    %v1638 = stablehlo.broadcast_in_dim %s2b8ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1639 = stablehlo.multiply %v1638, %v1623 : tensor<32x75264xf32>
    %v1640 = stablehlo.reduce(%v1639 init: %v1624) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1641 = stablehlo.broadcast_in_dim %v1640, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1642 = stablehlo.multiply %v1637, %v1639 : tensor<32x75264xf32>
    %v1643 = stablehlo.reduce(%v1642 init: %v1624) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1644 = stablehlo.broadcast_in_dim %v1643, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1645 = stablehlo.multiply %v1639, %v1625 : tensor<32x75264xf32>
    %v1646 = stablehlo.subtract %v1645, %v1641 : tensor<32x75264xf32>
    %v1647 = stablehlo.multiply %v1637, %v1644 : tensor<32x75264xf32>
    %v1648 = stablehlo.subtract %v1646, %v1647 : tensor<32x75264xf32>
    %v1649 = stablehlo.divide %v1636, %v1625 : tensor<32x75264xf32>
    %v1650 = stablehlo.multiply %v1649, %v1648 : tensor<32x75264xf32>
    %v1651 = stablehlo.reshape %v1650 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1652 = stablehlo.reverse %s2b8dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1653 = stablehlo.convolution(%v1651, %v1652)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1654 = stablehlo.reshape %v1653 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1655 = stablehlo.add %v1654, %v1556 : tensor<32x75264xf32>
    %v1656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1657 = stablehlo.reshape %v810 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1658 = stablehlo.reshape %v1556 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1659 = stablehlo.multiply %v1657, %v1658 : tensor<32x384x14x14xf32>
    %v1660 = stablehlo.reduce(%v1659 init: %v1656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1661 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1662 = stablehlo.multiply %v1660, %v1661 : tensor<384xf32>
    %v1663 = stablehlo.subtract %s2b8lg, %v1662 : tensor<384xf32>
    %v1664 = stablehlo.reshape %v805 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1665 = stablehlo.reshape %v1590 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1666 = stablehlo.transpose %v1664, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1667 = stablehlo.transpose %v1665, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1668 = stablehlo.convolution(%v1666, %v1667)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1669 = stablehlo.transpose %v1668, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1670 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v1671 = stablehlo.multiply %v1669, %v1670 : tensor<384x1536x1x1xf32>
    %v1672 = stablehlo.subtract %s2b8pW, %v1671 : tensor<384x1536x1x1xf32>
    %v1673 = stablehlo.reshape %v1590 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1675 = stablehlo.reduce(%v1673 init: %v1674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1676 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1677 = stablehlo.multiply %v1675, %v1676 : tensor<384xf32>
    %v1678 = stablehlo.subtract %s2b8pb, %v1677 : tensor<384xf32>
    %v1679 = stablehlo.reshape %v787 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1680 = stablehlo.reshape %v1618 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1681 = stablehlo.transpose %v1679, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1682 = stablehlo.transpose %v1680, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1683 = stablehlo.convolution(%v1681, %v1682)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1684 = stablehlo.transpose %v1683, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1685 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v1686 = stablehlo.multiply %v1684, %v1685 : tensor<1536x384x1x1xf32>
    %v1687 = stablehlo.subtract %s2b8eW, %v1686 : tensor<1536x384x1x1xf32>
    %v1688 = stablehlo.reshape %v1618 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1690 = stablehlo.reduce(%v1688 init: %v1689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1691 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v1692 = stablehlo.multiply %v1690, %v1691 : tensor<1536xf32>
    %v1693 = stablehlo.subtract %s2b8eb, %v1692 : tensor<1536xf32>
    %v1694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1695 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1696 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1697 = stablehlo.reduce(%v769 init: %v1694) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1698 = stablehlo.broadcast_in_dim %v1697, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1699 = stablehlo.divide %v1698, %v1695 : tensor<32x75264xf32>
    %v1700 = stablehlo.subtract %v769, %v1699 : tensor<32x75264xf32>
    %v1701 = stablehlo.multiply %v1700, %v1700 : tensor<32x75264xf32>
    %v1702 = stablehlo.reduce(%v1701 init: %v1694) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1703 = stablehlo.broadcast_in_dim %v1702, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1704 = stablehlo.divide %v1703, %v1695 : tensor<32x75264xf32>
    %v1705 = stablehlo.add %v1704, %v1696 : tensor<32x75264xf32>
    %v1706 = stablehlo.rsqrt %v1705 : tensor<32x75264xf32>
    %v1707 = stablehlo.multiply %v1700, %v1706 : tensor<32x75264xf32>
    %v1708 = stablehlo.multiply %v1623, %v1707 : tensor<32x75264xf32>
    %v1709 = stablehlo.reduce(%v1708 init: %v1694) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1710 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1711 = stablehlo.multiply %v1709, %v1710 : tensor<f32>
    %v1712 = stablehlo.subtract %s2b8ng, %v1711 : tensor<f32>
    %v1713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1714 = stablehlo.reduce(%v1623 init: %v1713) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1715 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1716 = stablehlo.multiply %v1714, %v1715 : tensor<f32>
    %v1717 = stablehlo.subtract %s2b8nbt, %v1716 : tensor<f32>
    %v1718 = stablehlo.reshape %v764 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1719 = stablehlo.reshape %v1650 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1720 = stablehlo.transpose %v1718, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1721 = stablehlo.transpose %v1719, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1722 = stablehlo.convolution(%v1720, %v1721)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v1723 = stablehlo.reshape %v1722 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v1724 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v1725 = stablehlo.multiply %v1723, %v1724 : tensor<384x1x7x7xf32>
    %v1726 = stablehlo.subtract %s2b8dW, %v1725 : tensor<384x1x7x7xf32>
    %v1727 = stablehlo.reshape %v1650 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1729 = stablehlo.reduce(%v1727 init: %v1728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1730 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1731 = stablehlo.multiply %v1729, %v1730 : tensor<384xf32>
    %v1732 = stablehlo.subtract %s2b8db, %v1731 : tensor<384xf32>
    %v1733 = stablehlo.reshape %v1655 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1734 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1735 = stablehlo.multiply %v1733, %v1734 : tensor<32x384x14x14xf32>
    %v1736 = stablehlo.reshape %v1735 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1737 = stablehlo.reshape %v1736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1738 = stablehlo.transpose %s2b7pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1739 = stablehlo.reverse %v1738, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1740 = stablehlo.convolution(%v1737, %v1739)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1741 = stablehlo.reshape %v1740 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1742 = stablehlo.multiply %v741, %v741 : tensor<32x301056xf32>
    %v1743 = stablehlo.multiply %v1742, %v741 : tensor<32x301056xf32>
    %v1744 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1745 = stablehlo.multiply %v1744, %v1743 : tensor<32x301056xf32>
    %v1746 = stablehlo.add %v741, %v1745 : tensor<32x301056xf32>
    %v1747 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1748 = stablehlo.multiply %v1747, %v1746 : tensor<32x301056xf32>
    %v1749 = stablehlo.tanh %v1748 : tensor<32x301056xf32>
    %v1750 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1751 = stablehlo.add %v1750, %v1749 : tensor<32x301056xf32>
    %v1752 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1753 = stablehlo.multiply %v1752, %v1751 : tensor<32x301056xf32>
    %v1754 = stablehlo.multiply %v1749, %v1749 : tensor<32x301056xf32>
    %v1755 = stablehlo.subtract %v1750, %v1754 : tensor<32x301056xf32>
    %v1756 = stablehlo.multiply %v1752, %v741 : tensor<32x301056xf32>
    %v1757 = stablehlo.multiply %v1756, %v1755 : tensor<32x301056xf32>
    %v1758 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1759 = stablehlo.multiply %v1758, %v1742 : tensor<32x301056xf32>
    %v1760 = stablehlo.add %v1750, %v1759 : tensor<32x301056xf32>
    %v1761 = stablehlo.multiply %v1747, %v1760 : tensor<32x301056xf32>
    %v1762 = stablehlo.multiply %v1757, %v1761 : tensor<32x301056xf32>
    %v1763 = stablehlo.add %v1753, %v1762 : tensor<32x301056xf32>
    %v1764 = stablehlo.multiply %v1741, %v1763 : tensor<32x301056xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1766 = stablehlo.transpose %s2b7eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1767 = stablehlo.reverse %v1766, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1768 = stablehlo.convolution(%v1765, %v1767)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1769 = stablehlo.reshape %v1768 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1771 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1772 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1773 = stablehlo.reduce(%v718 init: %v1770) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1774 = stablehlo.broadcast_in_dim %v1773, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1775 = stablehlo.divide %v1774, %v1771 : tensor<32x75264xf32>
    %v1776 = stablehlo.subtract %v718, %v1775 : tensor<32x75264xf32>
    %v1777 = stablehlo.multiply %v1776, %v1776 : tensor<32x75264xf32>
    %v1778 = stablehlo.reduce(%v1777 init: %v1770) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1779 = stablehlo.broadcast_in_dim %v1778, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1780 = stablehlo.divide %v1779, %v1771 : tensor<32x75264xf32>
    %v1781 = stablehlo.add %v1780, %v1772 : tensor<32x75264xf32>
    %v1782 = stablehlo.rsqrt %v1781 : tensor<32x75264xf32>
    %v1783 = stablehlo.multiply %v1776, %v1782 : tensor<32x75264xf32>
    %v1784 = stablehlo.broadcast_in_dim %s2b7ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1785 = stablehlo.multiply %v1784, %v1769 : tensor<32x75264xf32>
    %v1786 = stablehlo.reduce(%v1785 init: %v1770) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1787 = stablehlo.broadcast_in_dim %v1786, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1788 = stablehlo.multiply %v1783, %v1785 : tensor<32x75264xf32>
    %v1789 = stablehlo.reduce(%v1788 init: %v1770) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1790 = stablehlo.broadcast_in_dim %v1789, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1791 = stablehlo.multiply %v1785, %v1771 : tensor<32x75264xf32>
    %v1792 = stablehlo.subtract %v1791, %v1787 : tensor<32x75264xf32>
    %v1793 = stablehlo.multiply %v1783, %v1790 : tensor<32x75264xf32>
    %v1794 = stablehlo.subtract %v1792, %v1793 : tensor<32x75264xf32>
    %v1795 = stablehlo.divide %v1782, %v1771 : tensor<32x75264xf32>
    %v1796 = stablehlo.multiply %v1795, %v1794 : tensor<32x75264xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1798 = stablehlo.reverse %s2b7dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1799 = stablehlo.convolution(%v1797, %v1798)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1800 = stablehlo.reshape %v1799 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1801 = stablehlo.add %v1800, %v1655 : tensor<32x75264xf32>
    %v1802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1803 = stablehlo.reshape %v759 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1804 = stablehlo.reshape %v1655 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1805 = stablehlo.multiply %v1803, %v1804 : tensor<32x384x14x14xf32>
    %v1806 = stablehlo.reduce(%v1805 init: %v1802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1807 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1808 = stablehlo.multiply %v1806, %v1807 : tensor<384xf32>
    %v1809 = stablehlo.subtract %s2b7lg, %v1808 : tensor<384xf32>
    %v1810 = stablehlo.reshape %v754 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1811 = stablehlo.reshape %v1736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1812 = stablehlo.transpose %v1810, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1813 = stablehlo.transpose %v1811, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1814 = stablehlo.convolution(%v1812, %v1813)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1815 = stablehlo.transpose %v1814, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1816 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v1817 = stablehlo.multiply %v1815, %v1816 : tensor<384x1536x1x1xf32>
    %v1818 = stablehlo.subtract %s2b7pW, %v1817 : tensor<384x1536x1x1xf32>
    %v1819 = stablehlo.reshape %v1736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1821 = stablehlo.reduce(%v1819 init: %v1820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1822 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1823 = stablehlo.multiply %v1821, %v1822 : tensor<384xf32>
    %v1824 = stablehlo.subtract %s2b7pb, %v1823 : tensor<384xf32>
    %v1825 = stablehlo.reshape %v736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1826 = stablehlo.reshape %v1764 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1827 = stablehlo.transpose %v1825, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1828 = stablehlo.transpose %v1826, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1829 = stablehlo.convolution(%v1827, %v1828)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1830 = stablehlo.transpose %v1829, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1831 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v1832 = stablehlo.multiply %v1830, %v1831 : tensor<1536x384x1x1xf32>
    %v1833 = stablehlo.subtract %s2b7eW, %v1832 : tensor<1536x384x1x1xf32>
    %v1834 = stablehlo.reshape %v1764 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1835 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1836 = stablehlo.reduce(%v1834 init: %v1835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1837 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v1838 = stablehlo.multiply %v1836, %v1837 : tensor<1536xf32>
    %v1839 = stablehlo.subtract %s2b7eb, %v1838 : tensor<1536xf32>
    %v1840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1841 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1842 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1843 = stablehlo.reduce(%v718 init: %v1840) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1844 = stablehlo.broadcast_in_dim %v1843, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1845 = stablehlo.divide %v1844, %v1841 : tensor<32x75264xf32>
    %v1846 = stablehlo.subtract %v718, %v1845 : tensor<32x75264xf32>
    %v1847 = stablehlo.multiply %v1846, %v1846 : tensor<32x75264xf32>
    %v1848 = stablehlo.reduce(%v1847 init: %v1840) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1849 = stablehlo.broadcast_in_dim %v1848, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1850 = stablehlo.divide %v1849, %v1841 : tensor<32x75264xf32>
    %v1851 = stablehlo.add %v1850, %v1842 : tensor<32x75264xf32>
    %v1852 = stablehlo.rsqrt %v1851 : tensor<32x75264xf32>
    %v1853 = stablehlo.multiply %v1846, %v1852 : tensor<32x75264xf32>
    %v1854 = stablehlo.multiply %v1769, %v1853 : tensor<32x75264xf32>
    %v1855 = stablehlo.reduce(%v1854 init: %v1840) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1856 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1857 = stablehlo.multiply %v1855, %v1856 : tensor<f32>
    %v1858 = stablehlo.subtract %s2b7ng, %v1857 : tensor<f32>
    %v1859 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1860 = stablehlo.reduce(%v1769 init: %v1859) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1861 = stablehlo.constant dense<0.1> : tensor<f32>
    %v1862 = stablehlo.multiply %v1860, %v1861 : tensor<f32>
    %v1863 = stablehlo.subtract %s2b7nbt, %v1862 : tensor<f32>
    %v1864 = stablehlo.reshape %v713 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1865 = stablehlo.reshape %v1796 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1866 = stablehlo.transpose %v1864, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1867 = stablehlo.transpose %v1865, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1868 = stablehlo.convolution(%v1866, %v1867)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v1869 = stablehlo.reshape %v1868 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v1870 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v1871 = stablehlo.multiply %v1869, %v1870 : tensor<384x1x7x7xf32>
    %v1872 = stablehlo.subtract %s2b7dW, %v1871 : tensor<384x1x7x7xf32>
    %v1873 = stablehlo.reshape %v1796 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1875 = stablehlo.reduce(%v1873 init: %v1874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1876 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1877 = stablehlo.multiply %v1875, %v1876 : tensor<384xf32>
    %v1878 = stablehlo.subtract %s2b7db, %v1877 : tensor<384xf32>
    %v1879 = stablehlo.reshape %v1801 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1880 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1881 = stablehlo.multiply %v1879, %v1880 : tensor<32x384x14x14xf32>
    %v1882 = stablehlo.reshape %v1881 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1883 = stablehlo.reshape %v1882 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1884 = stablehlo.transpose %s2b6pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1885 = stablehlo.reverse %v1884, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1886 = stablehlo.convolution(%v1883, %v1885)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1887 = stablehlo.reshape %v1886 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1888 = stablehlo.multiply %v690, %v690 : tensor<32x301056xf32>
    %v1889 = stablehlo.multiply %v1888, %v690 : tensor<32x301056xf32>
    %v1890 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1891 = stablehlo.multiply %v1890, %v1889 : tensor<32x301056xf32>
    %v1892 = stablehlo.add %v690, %v1891 : tensor<32x301056xf32>
    %v1893 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1894 = stablehlo.multiply %v1893, %v1892 : tensor<32x301056xf32>
    %v1895 = stablehlo.tanh %v1894 : tensor<32x301056xf32>
    %v1896 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1897 = stablehlo.add %v1896, %v1895 : tensor<32x301056xf32>
    %v1898 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1899 = stablehlo.multiply %v1898, %v1897 : tensor<32x301056xf32>
    %v1900 = stablehlo.multiply %v1895, %v1895 : tensor<32x301056xf32>
    %v1901 = stablehlo.subtract %v1896, %v1900 : tensor<32x301056xf32>
    %v1902 = stablehlo.multiply %v1898, %v690 : tensor<32x301056xf32>
    %v1903 = stablehlo.multiply %v1902, %v1901 : tensor<32x301056xf32>
    %v1904 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1905 = stablehlo.multiply %v1904, %v1888 : tensor<32x301056xf32>
    %v1906 = stablehlo.add %v1896, %v1905 : tensor<32x301056xf32>
    %v1907 = stablehlo.multiply %v1893, %v1906 : tensor<32x301056xf32>
    %v1908 = stablehlo.multiply %v1903, %v1907 : tensor<32x301056xf32>
    %v1909 = stablehlo.add %v1899, %v1908 : tensor<32x301056xf32>
    %v1910 = stablehlo.multiply %v1887, %v1909 : tensor<32x301056xf32>
    %v1911 = stablehlo.reshape %v1910 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1912 = stablehlo.transpose %s2b6eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1913 = stablehlo.reverse %v1912, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1914 = stablehlo.convolution(%v1911, %v1913)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1915 = stablehlo.reshape %v1914 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1917 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1918 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1919 = stablehlo.reduce(%v667 init: %v1916) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1920 = stablehlo.broadcast_in_dim %v1919, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1921 = stablehlo.divide %v1920, %v1917 : tensor<32x75264xf32>
    %v1922 = stablehlo.subtract %v667, %v1921 : tensor<32x75264xf32>
    %v1923 = stablehlo.multiply %v1922, %v1922 : tensor<32x75264xf32>
    %v1924 = stablehlo.reduce(%v1923 init: %v1916) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1925 = stablehlo.broadcast_in_dim %v1924, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1926 = stablehlo.divide %v1925, %v1917 : tensor<32x75264xf32>
    %v1927 = stablehlo.add %v1926, %v1918 : tensor<32x75264xf32>
    %v1928 = stablehlo.rsqrt %v1927 : tensor<32x75264xf32>
    %v1929 = stablehlo.multiply %v1922, %v1928 : tensor<32x75264xf32>
    %v1930 = stablehlo.broadcast_in_dim %s2b6ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1931 = stablehlo.multiply %v1930, %v1915 : tensor<32x75264xf32>
    %v1932 = stablehlo.reduce(%v1931 init: %v1916) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1933 = stablehlo.broadcast_in_dim %v1932, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1934 = stablehlo.multiply %v1929, %v1931 : tensor<32x75264xf32>
    %v1935 = stablehlo.reduce(%v1934 init: %v1916) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1936 = stablehlo.broadcast_in_dim %v1935, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1937 = stablehlo.multiply %v1931, %v1917 : tensor<32x75264xf32>
    %v1938 = stablehlo.subtract %v1937, %v1933 : tensor<32x75264xf32>
    %v1939 = stablehlo.multiply %v1929, %v1936 : tensor<32x75264xf32>
    %v1940 = stablehlo.subtract %v1938, %v1939 : tensor<32x75264xf32>
    %v1941 = stablehlo.divide %v1928, %v1917 : tensor<32x75264xf32>
    %v1942 = stablehlo.multiply %v1941, %v1940 : tensor<32x75264xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1944 = stablehlo.reverse %s2b6dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1945 = stablehlo.convolution(%v1943, %v1944)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1947 = stablehlo.add %v1946, %v1801 : tensor<32x75264xf32>
    %v1948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1949 = stablehlo.reshape %v708 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1950 = stablehlo.reshape %v1801 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1951 = stablehlo.multiply %v1949, %v1950 : tensor<32x384x14x14xf32>
    %v1952 = stablehlo.reduce(%v1951 init: %v1948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1953 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1954 = stablehlo.multiply %v1952, %v1953 : tensor<384xf32>
    %v1955 = stablehlo.subtract %s2b6lg, %v1954 : tensor<384xf32>
    %v1956 = stablehlo.reshape %v703 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1957 = stablehlo.reshape %v1882 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1958 = stablehlo.transpose %v1956, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1959 = stablehlo.transpose %v1957, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1960 = stablehlo.convolution(%v1958, %v1959)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1961 = stablehlo.transpose %v1960, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1962 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v1963 = stablehlo.multiply %v1961, %v1962 : tensor<384x1536x1x1xf32>
    %v1964 = stablehlo.subtract %s2b6pW, %v1963 : tensor<384x1536x1x1xf32>
    %v1965 = stablehlo.reshape %v1882 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1967 = stablehlo.reduce(%v1965 init: %v1966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1968 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1969 = stablehlo.multiply %v1967, %v1968 : tensor<384xf32>
    %v1970 = stablehlo.subtract %s2b6pb, %v1969 : tensor<384xf32>
    %v1971 = stablehlo.reshape %v685 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1972 = stablehlo.reshape %v1910 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1973 = stablehlo.transpose %v1971, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1974 = stablehlo.transpose %v1972, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1975 = stablehlo.convolution(%v1973, %v1974)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1976 = stablehlo.transpose %v1975, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1977 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v1978 = stablehlo.multiply %v1976, %v1977 : tensor<1536x384x1x1xf32>
    %v1979 = stablehlo.subtract %s2b6eW, %v1978 : tensor<1536x384x1x1xf32>
    %v1980 = stablehlo.reshape %v1910 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1981 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1982 = stablehlo.reduce(%v1980 init: %v1981) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1983 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v1984 = stablehlo.multiply %v1982, %v1983 : tensor<1536xf32>
    %v1985 = stablehlo.subtract %s2b6eb, %v1984 : tensor<1536xf32>
    %v1986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1987 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1988 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1989 = stablehlo.reduce(%v667 init: %v1986) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1990 = stablehlo.broadcast_in_dim %v1989, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1991 = stablehlo.divide %v1990, %v1987 : tensor<32x75264xf32>
    %v1992 = stablehlo.subtract %v667, %v1991 : tensor<32x75264xf32>
    %v1993 = stablehlo.multiply %v1992, %v1992 : tensor<32x75264xf32>
    %v1994 = stablehlo.reduce(%v1993 init: %v1986) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1995 = stablehlo.broadcast_in_dim %v1994, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1996 = stablehlo.divide %v1995, %v1987 : tensor<32x75264xf32>
    %v1997 = stablehlo.add %v1996, %v1988 : tensor<32x75264xf32>
    %v1998 = stablehlo.rsqrt %v1997 : tensor<32x75264xf32>
    %v1999 = stablehlo.multiply %v1992, %v1998 : tensor<32x75264xf32>
    %v2000 = stablehlo.multiply %v1915, %v1999 : tensor<32x75264xf32>
    %v2001 = stablehlo.reduce(%v2000 init: %v1986) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2002 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2003 = stablehlo.multiply %v2001, %v2002 : tensor<f32>
    %v2004 = stablehlo.subtract %s2b6ng, %v2003 : tensor<f32>
    %v2005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2006 = stablehlo.reduce(%v1915 init: %v2005) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2007 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2008 = stablehlo.multiply %v2006, %v2007 : tensor<f32>
    %v2009 = stablehlo.subtract %s2b6nbt, %v2008 : tensor<f32>
    %v2010 = stablehlo.reshape %v662 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2011 = stablehlo.reshape %v1942 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2012 = stablehlo.transpose %v2010, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2013 = stablehlo.transpose %v2011, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2014 = stablehlo.convolution(%v2012, %v2013)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2015 = stablehlo.reshape %v2014 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2016 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2017 = stablehlo.multiply %v2015, %v2016 : tensor<384x1x7x7xf32>
    %v2018 = stablehlo.subtract %s2b6dW, %v2017 : tensor<384x1x7x7xf32>
    %v2019 = stablehlo.reshape %v1942 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2020 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2021 = stablehlo.reduce(%v2019 init: %v2020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2022 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2023 = stablehlo.multiply %v2021, %v2022 : tensor<384xf32>
    %v2024 = stablehlo.subtract %s2b6db, %v2023 : tensor<384xf32>
    %v2025 = stablehlo.reshape %v1947 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2026 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2027 = stablehlo.multiply %v2025, %v2026 : tensor<32x384x14x14xf32>
    %v2028 = stablehlo.reshape %v2027 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2029 = stablehlo.reshape %v2028 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2030 = stablehlo.transpose %s2b5pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2031 = stablehlo.reverse %v2030, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2032 = stablehlo.convolution(%v2029, %v2031)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2033 = stablehlo.reshape %v2032 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2034 = stablehlo.multiply %v639, %v639 : tensor<32x301056xf32>
    %v2035 = stablehlo.multiply %v2034, %v639 : tensor<32x301056xf32>
    %v2036 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2037 = stablehlo.multiply %v2036, %v2035 : tensor<32x301056xf32>
    %v2038 = stablehlo.add %v639, %v2037 : tensor<32x301056xf32>
    %v2039 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2040 = stablehlo.multiply %v2039, %v2038 : tensor<32x301056xf32>
    %v2041 = stablehlo.tanh %v2040 : tensor<32x301056xf32>
    %v2042 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2043 = stablehlo.add %v2042, %v2041 : tensor<32x301056xf32>
    %v2044 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2045 = stablehlo.multiply %v2044, %v2043 : tensor<32x301056xf32>
    %v2046 = stablehlo.multiply %v2041, %v2041 : tensor<32x301056xf32>
    %v2047 = stablehlo.subtract %v2042, %v2046 : tensor<32x301056xf32>
    %v2048 = stablehlo.multiply %v2044, %v639 : tensor<32x301056xf32>
    %v2049 = stablehlo.multiply %v2048, %v2047 : tensor<32x301056xf32>
    %v2050 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2051 = stablehlo.multiply %v2050, %v2034 : tensor<32x301056xf32>
    %v2052 = stablehlo.add %v2042, %v2051 : tensor<32x301056xf32>
    %v2053 = stablehlo.multiply %v2039, %v2052 : tensor<32x301056xf32>
    %v2054 = stablehlo.multiply %v2049, %v2053 : tensor<32x301056xf32>
    %v2055 = stablehlo.add %v2045, %v2054 : tensor<32x301056xf32>
    %v2056 = stablehlo.multiply %v2033, %v2055 : tensor<32x301056xf32>
    %v2057 = stablehlo.reshape %v2056 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2058 = stablehlo.transpose %s2b5eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2059 = stablehlo.reverse %v2058, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2060 = stablehlo.convolution(%v2057, %v2059)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2061 = stablehlo.reshape %v2060 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2063 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2064 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2065 = stablehlo.reduce(%v616 init: %v2062) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2066 = stablehlo.broadcast_in_dim %v2065, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2067 = stablehlo.divide %v2066, %v2063 : tensor<32x75264xf32>
    %v2068 = stablehlo.subtract %v616, %v2067 : tensor<32x75264xf32>
    %v2069 = stablehlo.multiply %v2068, %v2068 : tensor<32x75264xf32>
    %v2070 = stablehlo.reduce(%v2069 init: %v2062) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2071 = stablehlo.broadcast_in_dim %v2070, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2072 = stablehlo.divide %v2071, %v2063 : tensor<32x75264xf32>
    %v2073 = stablehlo.add %v2072, %v2064 : tensor<32x75264xf32>
    %v2074 = stablehlo.rsqrt %v2073 : tensor<32x75264xf32>
    %v2075 = stablehlo.multiply %v2068, %v2074 : tensor<32x75264xf32>
    %v2076 = stablehlo.broadcast_in_dim %s2b5ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2077 = stablehlo.multiply %v2076, %v2061 : tensor<32x75264xf32>
    %v2078 = stablehlo.reduce(%v2077 init: %v2062) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2079 = stablehlo.broadcast_in_dim %v2078, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2080 = stablehlo.multiply %v2075, %v2077 : tensor<32x75264xf32>
    %v2081 = stablehlo.reduce(%v2080 init: %v2062) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2082 = stablehlo.broadcast_in_dim %v2081, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2083 = stablehlo.multiply %v2077, %v2063 : tensor<32x75264xf32>
    %v2084 = stablehlo.subtract %v2083, %v2079 : tensor<32x75264xf32>
    %v2085 = stablehlo.multiply %v2075, %v2082 : tensor<32x75264xf32>
    %v2086 = stablehlo.subtract %v2084, %v2085 : tensor<32x75264xf32>
    %v2087 = stablehlo.divide %v2074, %v2063 : tensor<32x75264xf32>
    %v2088 = stablehlo.multiply %v2087, %v2086 : tensor<32x75264xf32>
    %v2089 = stablehlo.reshape %v2088 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2090 = stablehlo.reverse %s2b5dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2091 = stablehlo.convolution(%v2089, %v2090)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2092 = stablehlo.reshape %v2091 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2093 = stablehlo.add %v2092, %v1947 : tensor<32x75264xf32>
    %v2094 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2095 = stablehlo.reshape %v657 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2096 = stablehlo.reshape %v1947 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2097 = stablehlo.multiply %v2095, %v2096 : tensor<32x384x14x14xf32>
    %v2098 = stablehlo.reduce(%v2097 init: %v2094) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2099 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2100 = stablehlo.multiply %v2098, %v2099 : tensor<384xf32>
    %v2101 = stablehlo.subtract %s2b5lg, %v2100 : tensor<384xf32>
    %v2102 = stablehlo.reshape %v652 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2103 = stablehlo.reshape %v2028 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2104 = stablehlo.transpose %v2102, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2105 = stablehlo.transpose %v2103, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2106 = stablehlo.convolution(%v2104, %v2105)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2107 = stablehlo.transpose %v2106, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2108 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2109 = stablehlo.multiply %v2107, %v2108 : tensor<384x1536x1x1xf32>
    %v2110 = stablehlo.subtract %s2b5pW, %v2109 : tensor<384x1536x1x1xf32>
    %v2111 = stablehlo.reshape %v2028 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2113 = stablehlo.reduce(%v2111 init: %v2112) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2114 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2115 = stablehlo.multiply %v2113, %v2114 : tensor<384xf32>
    %v2116 = stablehlo.subtract %s2b5pb, %v2115 : tensor<384xf32>
    %v2117 = stablehlo.reshape %v634 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2118 = stablehlo.reshape %v2056 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2119 = stablehlo.transpose %v2117, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2120 = stablehlo.transpose %v2118, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2121 = stablehlo.convolution(%v2119, %v2120)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2122 = stablehlo.transpose %v2121, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2123 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2124 = stablehlo.multiply %v2122, %v2123 : tensor<1536x384x1x1xf32>
    %v2125 = stablehlo.subtract %s2b5eW, %v2124 : tensor<1536x384x1x1xf32>
    %v2126 = stablehlo.reshape %v2056 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2128 = stablehlo.reduce(%v2126 init: %v2127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2129 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2130 = stablehlo.multiply %v2128, %v2129 : tensor<1536xf32>
    %v2131 = stablehlo.subtract %s2b5eb, %v2130 : tensor<1536xf32>
    %v2132 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2133 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2134 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2135 = stablehlo.reduce(%v616 init: %v2132) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2136 = stablehlo.broadcast_in_dim %v2135, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2137 = stablehlo.divide %v2136, %v2133 : tensor<32x75264xf32>
    %v2138 = stablehlo.subtract %v616, %v2137 : tensor<32x75264xf32>
    %v2139 = stablehlo.multiply %v2138, %v2138 : tensor<32x75264xf32>
    %v2140 = stablehlo.reduce(%v2139 init: %v2132) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2141 = stablehlo.broadcast_in_dim %v2140, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2142 = stablehlo.divide %v2141, %v2133 : tensor<32x75264xf32>
    %v2143 = stablehlo.add %v2142, %v2134 : tensor<32x75264xf32>
    %v2144 = stablehlo.rsqrt %v2143 : tensor<32x75264xf32>
    %v2145 = stablehlo.multiply %v2138, %v2144 : tensor<32x75264xf32>
    %v2146 = stablehlo.multiply %v2061, %v2145 : tensor<32x75264xf32>
    %v2147 = stablehlo.reduce(%v2146 init: %v2132) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2148 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2149 = stablehlo.multiply %v2147, %v2148 : tensor<f32>
    %v2150 = stablehlo.subtract %s2b5ng, %v2149 : tensor<f32>
    %v2151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2152 = stablehlo.reduce(%v2061 init: %v2151) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2153 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2154 = stablehlo.multiply %v2152, %v2153 : tensor<f32>
    %v2155 = stablehlo.subtract %s2b5nbt, %v2154 : tensor<f32>
    %v2156 = stablehlo.reshape %v611 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2157 = stablehlo.reshape %v2088 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2158 = stablehlo.transpose %v2156, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2159 = stablehlo.transpose %v2157, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2160 = stablehlo.convolution(%v2158, %v2159)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2161 = stablehlo.reshape %v2160 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2162 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2163 = stablehlo.multiply %v2161, %v2162 : tensor<384x1x7x7xf32>
    %v2164 = stablehlo.subtract %s2b5dW, %v2163 : tensor<384x1x7x7xf32>
    %v2165 = stablehlo.reshape %v2088 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2166 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2167 = stablehlo.reduce(%v2165 init: %v2166) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2168 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2169 = stablehlo.multiply %v2167, %v2168 : tensor<384xf32>
    %v2170 = stablehlo.subtract %s2b5db, %v2169 : tensor<384xf32>
    %v2171 = stablehlo.reshape %v2093 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2172 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2173 = stablehlo.multiply %v2171, %v2172 : tensor<32x384x14x14xf32>
    %v2174 = stablehlo.reshape %v2173 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2175 = stablehlo.reshape %v2174 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2176 = stablehlo.transpose %s2b4pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2177 = stablehlo.reverse %v2176, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2178 = stablehlo.convolution(%v2175, %v2177)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2179 = stablehlo.reshape %v2178 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2180 = stablehlo.multiply %v588, %v588 : tensor<32x301056xf32>
    %v2181 = stablehlo.multiply %v2180, %v588 : tensor<32x301056xf32>
    %v2182 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2183 = stablehlo.multiply %v2182, %v2181 : tensor<32x301056xf32>
    %v2184 = stablehlo.add %v588, %v2183 : tensor<32x301056xf32>
    %v2185 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2186 = stablehlo.multiply %v2185, %v2184 : tensor<32x301056xf32>
    %v2187 = stablehlo.tanh %v2186 : tensor<32x301056xf32>
    %v2188 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2189 = stablehlo.add %v2188, %v2187 : tensor<32x301056xf32>
    %v2190 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2191 = stablehlo.multiply %v2190, %v2189 : tensor<32x301056xf32>
    %v2192 = stablehlo.multiply %v2187, %v2187 : tensor<32x301056xf32>
    %v2193 = stablehlo.subtract %v2188, %v2192 : tensor<32x301056xf32>
    %v2194 = stablehlo.multiply %v2190, %v588 : tensor<32x301056xf32>
    %v2195 = stablehlo.multiply %v2194, %v2193 : tensor<32x301056xf32>
    %v2196 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2197 = stablehlo.multiply %v2196, %v2180 : tensor<32x301056xf32>
    %v2198 = stablehlo.add %v2188, %v2197 : tensor<32x301056xf32>
    %v2199 = stablehlo.multiply %v2185, %v2198 : tensor<32x301056xf32>
    %v2200 = stablehlo.multiply %v2195, %v2199 : tensor<32x301056xf32>
    %v2201 = stablehlo.add %v2191, %v2200 : tensor<32x301056xf32>
    %v2202 = stablehlo.multiply %v2179, %v2201 : tensor<32x301056xf32>
    %v2203 = stablehlo.reshape %v2202 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2204 = stablehlo.transpose %s2b4eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2205 = stablehlo.reverse %v2204, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2206 = stablehlo.convolution(%v2203, %v2205)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2207 = stablehlo.reshape %v2206 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2208 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2209 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2210 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2211 = stablehlo.reduce(%v565 init: %v2208) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2212 = stablehlo.broadcast_in_dim %v2211, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2213 = stablehlo.divide %v2212, %v2209 : tensor<32x75264xf32>
    %v2214 = stablehlo.subtract %v565, %v2213 : tensor<32x75264xf32>
    %v2215 = stablehlo.multiply %v2214, %v2214 : tensor<32x75264xf32>
    %v2216 = stablehlo.reduce(%v2215 init: %v2208) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2217 = stablehlo.broadcast_in_dim %v2216, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2218 = stablehlo.divide %v2217, %v2209 : tensor<32x75264xf32>
    %v2219 = stablehlo.add %v2218, %v2210 : tensor<32x75264xf32>
    %v2220 = stablehlo.rsqrt %v2219 : tensor<32x75264xf32>
    %v2221 = stablehlo.multiply %v2214, %v2220 : tensor<32x75264xf32>
    %v2222 = stablehlo.broadcast_in_dim %s2b4ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2223 = stablehlo.multiply %v2222, %v2207 : tensor<32x75264xf32>
    %v2224 = stablehlo.reduce(%v2223 init: %v2208) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2225 = stablehlo.broadcast_in_dim %v2224, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2226 = stablehlo.multiply %v2221, %v2223 : tensor<32x75264xf32>
    %v2227 = stablehlo.reduce(%v2226 init: %v2208) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2228 = stablehlo.broadcast_in_dim %v2227, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2229 = stablehlo.multiply %v2223, %v2209 : tensor<32x75264xf32>
    %v2230 = stablehlo.subtract %v2229, %v2225 : tensor<32x75264xf32>
    %v2231 = stablehlo.multiply %v2221, %v2228 : tensor<32x75264xf32>
    %v2232 = stablehlo.subtract %v2230, %v2231 : tensor<32x75264xf32>
    %v2233 = stablehlo.divide %v2220, %v2209 : tensor<32x75264xf32>
    %v2234 = stablehlo.multiply %v2233, %v2232 : tensor<32x75264xf32>
    %v2235 = stablehlo.reshape %v2234 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2236 = stablehlo.reverse %s2b4dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2237 = stablehlo.convolution(%v2235, %v2236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2238 = stablehlo.reshape %v2237 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2239 = stablehlo.add %v2238, %v2093 : tensor<32x75264xf32>
    %v2240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2241 = stablehlo.reshape %v606 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2242 = stablehlo.reshape %v2093 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2243 = stablehlo.multiply %v2241, %v2242 : tensor<32x384x14x14xf32>
    %v2244 = stablehlo.reduce(%v2243 init: %v2240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2245 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2246 = stablehlo.multiply %v2244, %v2245 : tensor<384xf32>
    %v2247 = stablehlo.subtract %s2b4lg, %v2246 : tensor<384xf32>
    %v2248 = stablehlo.reshape %v601 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2249 = stablehlo.reshape %v2174 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2250 = stablehlo.transpose %v2248, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2251 = stablehlo.transpose %v2249, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2252 = stablehlo.convolution(%v2250, %v2251)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2253 = stablehlo.transpose %v2252, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2254 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2255 = stablehlo.multiply %v2253, %v2254 : tensor<384x1536x1x1xf32>
    %v2256 = stablehlo.subtract %s2b4pW, %v2255 : tensor<384x1536x1x1xf32>
    %v2257 = stablehlo.reshape %v2174 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2259 = stablehlo.reduce(%v2257 init: %v2258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2260 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2261 = stablehlo.multiply %v2259, %v2260 : tensor<384xf32>
    %v2262 = stablehlo.subtract %s2b4pb, %v2261 : tensor<384xf32>
    %v2263 = stablehlo.reshape %v583 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2264 = stablehlo.reshape %v2202 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2265 = stablehlo.transpose %v2263, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2266 = stablehlo.transpose %v2264, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2267 = stablehlo.convolution(%v2265, %v2266)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2268 = stablehlo.transpose %v2267, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2269 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2270 = stablehlo.multiply %v2268, %v2269 : tensor<1536x384x1x1xf32>
    %v2271 = stablehlo.subtract %s2b4eW, %v2270 : tensor<1536x384x1x1xf32>
    %v2272 = stablehlo.reshape %v2202 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2274 = stablehlo.reduce(%v2272 init: %v2273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2275 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2276 = stablehlo.multiply %v2274, %v2275 : tensor<1536xf32>
    %v2277 = stablehlo.subtract %s2b4eb, %v2276 : tensor<1536xf32>
    %v2278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2279 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2280 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2281 = stablehlo.reduce(%v565 init: %v2278) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2282 = stablehlo.broadcast_in_dim %v2281, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2283 = stablehlo.divide %v2282, %v2279 : tensor<32x75264xf32>
    %v2284 = stablehlo.subtract %v565, %v2283 : tensor<32x75264xf32>
    %v2285 = stablehlo.multiply %v2284, %v2284 : tensor<32x75264xf32>
    %v2286 = stablehlo.reduce(%v2285 init: %v2278) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2287 = stablehlo.broadcast_in_dim %v2286, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2288 = stablehlo.divide %v2287, %v2279 : tensor<32x75264xf32>
    %v2289 = stablehlo.add %v2288, %v2280 : tensor<32x75264xf32>
    %v2290 = stablehlo.rsqrt %v2289 : tensor<32x75264xf32>
    %v2291 = stablehlo.multiply %v2284, %v2290 : tensor<32x75264xf32>
    %v2292 = stablehlo.multiply %v2207, %v2291 : tensor<32x75264xf32>
    %v2293 = stablehlo.reduce(%v2292 init: %v2278) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2294 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2295 = stablehlo.multiply %v2293, %v2294 : tensor<f32>
    %v2296 = stablehlo.subtract %s2b4ng, %v2295 : tensor<f32>
    %v2297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2298 = stablehlo.reduce(%v2207 init: %v2297) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2299 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2300 = stablehlo.multiply %v2298, %v2299 : tensor<f32>
    %v2301 = stablehlo.subtract %s2b4nbt, %v2300 : tensor<f32>
    %v2302 = stablehlo.reshape %v560 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2303 = stablehlo.reshape %v2234 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2304 = stablehlo.transpose %v2302, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2305 = stablehlo.transpose %v2303, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2306 = stablehlo.convolution(%v2304, %v2305)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2307 = stablehlo.reshape %v2306 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2308 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2309 = stablehlo.multiply %v2307, %v2308 : tensor<384x1x7x7xf32>
    %v2310 = stablehlo.subtract %s2b4dW, %v2309 : tensor<384x1x7x7xf32>
    %v2311 = stablehlo.reshape %v2234 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2313 = stablehlo.reduce(%v2311 init: %v2312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2314 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2315 = stablehlo.multiply %v2313, %v2314 : tensor<384xf32>
    %v2316 = stablehlo.subtract %s2b4db, %v2315 : tensor<384xf32>
    %v2317 = stablehlo.reshape %v2239 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2318 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2319 = stablehlo.multiply %v2317, %v2318 : tensor<32x384x14x14xf32>
    %v2320 = stablehlo.reshape %v2319 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2321 = stablehlo.reshape %v2320 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2322 = stablehlo.transpose %s2b3pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2323 = stablehlo.reverse %v2322, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2324 = stablehlo.convolution(%v2321, %v2323)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2325 = stablehlo.reshape %v2324 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2326 = stablehlo.multiply %v537, %v537 : tensor<32x301056xf32>
    %v2327 = stablehlo.multiply %v2326, %v537 : tensor<32x301056xf32>
    %v2328 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2329 = stablehlo.multiply %v2328, %v2327 : tensor<32x301056xf32>
    %v2330 = stablehlo.add %v537, %v2329 : tensor<32x301056xf32>
    %v2331 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2332 = stablehlo.multiply %v2331, %v2330 : tensor<32x301056xf32>
    %v2333 = stablehlo.tanh %v2332 : tensor<32x301056xf32>
    %v2334 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2335 = stablehlo.add %v2334, %v2333 : tensor<32x301056xf32>
    %v2336 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2337 = stablehlo.multiply %v2336, %v2335 : tensor<32x301056xf32>
    %v2338 = stablehlo.multiply %v2333, %v2333 : tensor<32x301056xf32>
    %v2339 = stablehlo.subtract %v2334, %v2338 : tensor<32x301056xf32>
    %v2340 = stablehlo.multiply %v2336, %v537 : tensor<32x301056xf32>
    %v2341 = stablehlo.multiply %v2340, %v2339 : tensor<32x301056xf32>
    %v2342 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2343 = stablehlo.multiply %v2342, %v2326 : tensor<32x301056xf32>
    %v2344 = stablehlo.add %v2334, %v2343 : tensor<32x301056xf32>
    %v2345 = stablehlo.multiply %v2331, %v2344 : tensor<32x301056xf32>
    %v2346 = stablehlo.multiply %v2341, %v2345 : tensor<32x301056xf32>
    %v2347 = stablehlo.add %v2337, %v2346 : tensor<32x301056xf32>
    %v2348 = stablehlo.multiply %v2325, %v2347 : tensor<32x301056xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2350 = stablehlo.transpose %s2b3eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2351 = stablehlo.reverse %v2350, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2352 = stablehlo.convolution(%v2349, %v2351)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2353 = stablehlo.reshape %v2352 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2354 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2355 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2356 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2357 = stablehlo.reduce(%v514 init: %v2354) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2358 = stablehlo.broadcast_in_dim %v2357, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2359 = stablehlo.divide %v2358, %v2355 : tensor<32x75264xf32>
    %v2360 = stablehlo.subtract %v514, %v2359 : tensor<32x75264xf32>
    %v2361 = stablehlo.multiply %v2360, %v2360 : tensor<32x75264xf32>
    %v2362 = stablehlo.reduce(%v2361 init: %v2354) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2363 = stablehlo.broadcast_in_dim %v2362, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2364 = stablehlo.divide %v2363, %v2355 : tensor<32x75264xf32>
    %v2365 = stablehlo.add %v2364, %v2356 : tensor<32x75264xf32>
    %v2366 = stablehlo.rsqrt %v2365 : tensor<32x75264xf32>
    %v2367 = stablehlo.multiply %v2360, %v2366 : tensor<32x75264xf32>
    %v2368 = stablehlo.broadcast_in_dim %s2b3ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2369 = stablehlo.multiply %v2368, %v2353 : tensor<32x75264xf32>
    %v2370 = stablehlo.reduce(%v2369 init: %v2354) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2371 = stablehlo.broadcast_in_dim %v2370, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2372 = stablehlo.multiply %v2367, %v2369 : tensor<32x75264xf32>
    %v2373 = stablehlo.reduce(%v2372 init: %v2354) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2374 = stablehlo.broadcast_in_dim %v2373, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2375 = stablehlo.multiply %v2369, %v2355 : tensor<32x75264xf32>
    %v2376 = stablehlo.subtract %v2375, %v2371 : tensor<32x75264xf32>
    %v2377 = stablehlo.multiply %v2367, %v2374 : tensor<32x75264xf32>
    %v2378 = stablehlo.subtract %v2376, %v2377 : tensor<32x75264xf32>
    %v2379 = stablehlo.divide %v2366, %v2355 : tensor<32x75264xf32>
    %v2380 = stablehlo.multiply %v2379, %v2378 : tensor<32x75264xf32>
    %v2381 = stablehlo.reshape %v2380 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2382 = stablehlo.reverse %s2b3dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2383 = stablehlo.convolution(%v2381, %v2382)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2384 = stablehlo.reshape %v2383 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2385 = stablehlo.add %v2384, %v2239 : tensor<32x75264xf32>
    %v2386 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2387 = stablehlo.reshape %v555 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2388 = stablehlo.reshape %v2239 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2389 = stablehlo.multiply %v2387, %v2388 : tensor<32x384x14x14xf32>
    %v2390 = stablehlo.reduce(%v2389 init: %v2386) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2391 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2392 = stablehlo.multiply %v2390, %v2391 : tensor<384xf32>
    %v2393 = stablehlo.subtract %s2b3lg, %v2392 : tensor<384xf32>
    %v2394 = stablehlo.reshape %v550 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2395 = stablehlo.reshape %v2320 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2396 = stablehlo.transpose %v2394, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2397 = stablehlo.transpose %v2395, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2398 = stablehlo.convolution(%v2396, %v2397)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2399 = stablehlo.transpose %v2398, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2400 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2401 = stablehlo.multiply %v2399, %v2400 : tensor<384x1536x1x1xf32>
    %v2402 = stablehlo.subtract %s2b3pW, %v2401 : tensor<384x1536x1x1xf32>
    %v2403 = stablehlo.reshape %v2320 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2405 = stablehlo.reduce(%v2403 init: %v2404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2406 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2407 = stablehlo.multiply %v2405, %v2406 : tensor<384xf32>
    %v2408 = stablehlo.subtract %s2b3pb, %v2407 : tensor<384xf32>
    %v2409 = stablehlo.reshape %v532 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2410 = stablehlo.reshape %v2348 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2411 = stablehlo.transpose %v2409, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2412 = stablehlo.transpose %v2410, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2413 = stablehlo.convolution(%v2411, %v2412)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2414 = stablehlo.transpose %v2413, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2415 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2416 = stablehlo.multiply %v2414, %v2415 : tensor<1536x384x1x1xf32>
    %v2417 = stablehlo.subtract %s2b3eW, %v2416 : tensor<1536x384x1x1xf32>
    %v2418 = stablehlo.reshape %v2348 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2420 = stablehlo.reduce(%v2418 init: %v2419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2421 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2422 = stablehlo.multiply %v2420, %v2421 : tensor<1536xf32>
    %v2423 = stablehlo.subtract %s2b3eb, %v2422 : tensor<1536xf32>
    %v2424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2425 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2426 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2427 = stablehlo.reduce(%v514 init: %v2424) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2428 = stablehlo.broadcast_in_dim %v2427, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2429 = stablehlo.divide %v2428, %v2425 : tensor<32x75264xf32>
    %v2430 = stablehlo.subtract %v514, %v2429 : tensor<32x75264xf32>
    %v2431 = stablehlo.multiply %v2430, %v2430 : tensor<32x75264xf32>
    %v2432 = stablehlo.reduce(%v2431 init: %v2424) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2433 = stablehlo.broadcast_in_dim %v2432, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2434 = stablehlo.divide %v2433, %v2425 : tensor<32x75264xf32>
    %v2435 = stablehlo.add %v2434, %v2426 : tensor<32x75264xf32>
    %v2436 = stablehlo.rsqrt %v2435 : tensor<32x75264xf32>
    %v2437 = stablehlo.multiply %v2430, %v2436 : tensor<32x75264xf32>
    %v2438 = stablehlo.multiply %v2353, %v2437 : tensor<32x75264xf32>
    %v2439 = stablehlo.reduce(%v2438 init: %v2424) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2440 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2441 = stablehlo.multiply %v2439, %v2440 : tensor<f32>
    %v2442 = stablehlo.subtract %s2b3ng, %v2441 : tensor<f32>
    %v2443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2444 = stablehlo.reduce(%v2353 init: %v2443) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2445 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2446 = stablehlo.multiply %v2444, %v2445 : tensor<f32>
    %v2447 = stablehlo.subtract %s2b3nbt, %v2446 : tensor<f32>
    %v2448 = stablehlo.reshape %v509 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2449 = stablehlo.reshape %v2380 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2450 = stablehlo.transpose %v2448, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2451 = stablehlo.transpose %v2449, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2452 = stablehlo.convolution(%v2450, %v2451)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2453 = stablehlo.reshape %v2452 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2454 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2455 = stablehlo.multiply %v2453, %v2454 : tensor<384x1x7x7xf32>
    %v2456 = stablehlo.subtract %s2b3dW, %v2455 : tensor<384x1x7x7xf32>
    %v2457 = stablehlo.reshape %v2380 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2459 = stablehlo.reduce(%v2457 init: %v2458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2460 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2461 = stablehlo.multiply %v2459, %v2460 : tensor<384xf32>
    %v2462 = stablehlo.subtract %s2b3db, %v2461 : tensor<384xf32>
    %v2463 = stablehlo.reshape %v2385 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2464 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2465 = stablehlo.multiply %v2463, %v2464 : tensor<32x384x14x14xf32>
    %v2466 = stablehlo.reshape %v2465 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2467 = stablehlo.reshape %v2466 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2468 = stablehlo.transpose %s2b2pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2469 = stablehlo.reverse %v2468, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2470 = stablehlo.convolution(%v2467, %v2469)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2471 = stablehlo.reshape %v2470 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2472 = stablehlo.multiply %v486, %v486 : tensor<32x301056xf32>
    %v2473 = stablehlo.multiply %v2472, %v486 : tensor<32x301056xf32>
    %v2474 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2475 = stablehlo.multiply %v2474, %v2473 : tensor<32x301056xf32>
    %v2476 = stablehlo.add %v486, %v2475 : tensor<32x301056xf32>
    %v2477 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2478 = stablehlo.multiply %v2477, %v2476 : tensor<32x301056xf32>
    %v2479 = stablehlo.tanh %v2478 : tensor<32x301056xf32>
    %v2480 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2481 = stablehlo.add %v2480, %v2479 : tensor<32x301056xf32>
    %v2482 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2483 = stablehlo.multiply %v2482, %v2481 : tensor<32x301056xf32>
    %v2484 = stablehlo.multiply %v2479, %v2479 : tensor<32x301056xf32>
    %v2485 = stablehlo.subtract %v2480, %v2484 : tensor<32x301056xf32>
    %v2486 = stablehlo.multiply %v2482, %v486 : tensor<32x301056xf32>
    %v2487 = stablehlo.multiply %v2486, %v2485 : tensor<32x301056xf32>
    %v2488 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2489 = stablehlo.multiply %v2488, %v2472 : tensor<32x301056xf32>
    %v2490 = stablehlo.add %v2480, %v2489 : tensor<32x301056xf32>
    %v2491 = stablehlo.multiply %v2477, %v2490 : tensor<32x301056xf32>
    %v2492 = stablehlo.multiply %v2487, %v2491 : tensor<32x301056xf32>
    %v2493 = stablehlo.add %v2483, %v2492 : tensor<32x301056xf32>
    %v2494 = stablehlo.multiply %v2471, %v2493 : tensor<32x301056xf32>
    %v2495 = stablehlo.reshape %v2494 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2496 = stablehlo.transpose %s2b2eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2497 = stablehlo.reverse %v2496, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2498 = stablehlo.convolution(%v2495, %v2497)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2499 = stablehlo.reshape %v2498 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2500 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2501 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2502 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2503 = stablehlo.reduce(%v463 init: %v2500) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2504 = stablehlo.broadcast_in_dim %v2503, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2505 = stablehlo.divide %v2504, %v2501 : tensor<32x75264xf32>
    %v2506 = stablehlo.subtract %v463, %v2505 : tensor<32x75264xf32>
    %v2507 = stablehlo.multiply %v2506, %v2506 : tensor<32x75264xf32>
    %v2508 = stablehlo.reduce(%v2507 init: %v2500) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2509 = stablehlo.broadcast_in_dim %v2508, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2510 = stablehlo.divide %v2509, %v2501 : tensor<32x75264xf32>
    %v2511 = stablehlo.add %v2510, %v2502 : tensor<32x75264xf32>
    %v2512 = stablehlo.rsqrt %v2511 : tensor<32x75264xf32>
    %v2513 = stablehlo.multiply %v2506, %v2512 : tensor<32x75264xf32>
    %v2514 = stablehlo.broadcast_in_dim %s2b2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2515 = stablehlo.multiply %v2514, %v2499 : tensor<32x75264xf32>
    %v2516 = stablehlo.reduce(%v2515 init: %v2500) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2517 = stablehlo.broadcast_in_dim %v2516, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2518 = stablehlo.multiply %v2513, %v2515 : tensor<32x75264xf32>
    %v2519 = stablehlo.reduce(%v2518 init: %v2500) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2520 = stablehlo.broadcast_in_dim %v2519, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2521 = stablehlo.multiply %v2515, %v2501 : tensor<32x75264xf32>
    %v2522 = stablehlo.subtract %v2521, %v2517 : tensor<32x75264xf32>
    %v2523 = stablehlo.multiply %v2513, %v2520 : tensor<32x75264xf32>
    %v2524 = stablehlo.subtract %v2522, %v2523 : tensor<32x75264xf32>
    %v2525 = stablehlo.divide %v2512, %v2501 : tensor<32x75264xf32>
    %v2526 = stablehlo.multiply %v2525, %v2524 : tensor<32x75264xf32>
    %v2527 = stablehlo.reshape %v2526 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2528 = stablehlo.reverse %s2b2dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2529 = stablehlo.convolution(%v2527, %v2528)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2530 = stablehlo.reshape %v2529 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2531 = stablehlo.add %v2530, %v2385 : tensor<32x75264xf32>
    %v2532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2533 = stablehlo.reshape %v504 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2534 = stablehlo.reshape %v2385 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2535 = stablehlo.multiply %v2533, %v2534 : tensor<32x384x14x14xf32>
    %v2536 = stablehlo.reduce(%v2535 init: %v2532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2537 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2538 = stablehlo.multiply %v2536, %v2537 : tensor<384xf32>
    %v2539 = stablehlo.subtract %s2b2lg, %v2538 : tensor<384xf32>
    %v2540 = stablehlo.reshape %v499 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2541 = stablehlo.reshape %v2466 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2542 = stablehlo.transpose %v2540, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2543 = stablehlo.transpose %v2541, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2544 = stablehlo.convolution(%v2542, %v2543)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2545 = stablehlo.transpose %v2544, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2546 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2547 = stablehlo.multiply %v2545, %v2546 : tensor<384x1536x1x1xf32>
    %v2548 = stablehlo.subtract %s2b2pW, %v2547 : tensor<384x1536x1x1xf32>
    %v2549 = stablehlo.reshape %v2466 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2551 = stablehlo.reduce(%v2549 init: %v2550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2552 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2553 = stablehlo.multiply %v2551, %v2552 : tensor<384xf32>
    %v2554 = stablehlo.subtract %s2b2pb, %v2553 : tensor<384xf32>
    %v2555 = stablehlo.reshape %v481 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2556 = stablehlo.reshape %v2494 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2557 = stablehlo.transpose %v2555, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2558 = stablehlo.transpose %v2556, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2559 = stablehlo.convolution(%v2557, %v2558)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2560 = stablehlo.transpose %v2559, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2561 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2562 = stablehlo.multiply %v2560, %v2561 : tensor<1536x384x1x1xf32>
    %v2563 = stablehlo.subtract %s2b2eW, %v2562 : tensor<1536x384x1x1xf32>
    %v2564 = stablehlo.reshape %v2494 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2566 = stablehlo.reduce(%v2564 init: %v2565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2567 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2568 = stablehlo.multiply %v2566, %v2567 : tensor<1536xf32>
    %v2569 = stablehlo.subtract %s2b2eb, %v2568 : tensor<1536xf32>
    %v2570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2571 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2572 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2573 = stablehlo.reduce(%v463 init: %v2570) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2574 = stablehlo.broadcast_in_dim %v2573, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2575 = stablehlo.divide %v2574, %v2571 : tensor<32x75264xf32>
    %v2576 = stablehlo.subtract %v463, %v2575 : tensor<32x75264xf32>
    %v2577 = stablehlo.multiply %v2576, %v2576 : tensor<32x75264xf32>
    %v2578 = stablehlo.reduce(%v2577 init: %v2570) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2579 = stablehlo.broadcast_in_dim %v2578, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2580 = stablehlo.divide %v2579, %v2571 : tensor<32x75264xf32>
    %v2581 = stablehlo.add %v2580, %v2572 : tensor<32x75264xf32>
    %v2582 = stablehlo.rsqrt %v2581 : tensor<32x75264xf32>
    %v2583 = stablehlo.multiply %v2576, %v2582 : tensor<32x75264xf32>
    %v2584 = stablehlo.multiply %v2499, %v2583 : tensor<32x75264xf32>
    %v2585 = stablehlo.reduce(%v2584 init: %v2570) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2586 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2587 = stablehlo.multiply %v2585, %v2586 : tensor<f32>
    %v2588 = stablehlo.subtract %s2b2ng, %v2587 : tensor<f32>
    %v2589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2590 = stablehlo.reduce(%v2499 init: %v2589) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2591 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2592 = stablehlo.multiply %v2590, %v2591 : tensor<f32>
    %v2593 = stablehlo.subtract %s2b2nbt, %v2592 : tensor<f32>
    %v2594 = stablehlo.reshape %v458 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2595 = stablehlo.reshape %v2526 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2596 = stablehlo.transpose %v2594, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2597 = stablehlo.transpose %v2595, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2598 = stablehlo.convolution(%v2596, %v2597)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2599 = stablehlo.reshape %v2598 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2600 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2601 = stablehlo.multiply %v2599, %v2600 : tensor<384x1x7x7xf32>
    %v2602 = stablehlo.subtract %s2b2dW, %v2601 : tensor<384x1x7x7xf32>
    %v2603 = stablehlo.reshape %v2526 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2604 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2605 = stablehlo.reduce(%v2603 init: %v2604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2606 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2607 = stablehlo.multiply %v2605, %v2606 : tensor<384xf32>
    %v2608 = stablehlo.subtract %s2b2db, %v2607 : tensor<384xf32>
    %v2609 = stablehlo.reshape %v2531 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2610 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2611 = stablehlo.multiply %v2609, %v2610 : tensor<32x384x14x14xf32>
    %v2612 = stablehlo.reshape %v2611 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2613 = stablehlo.reshape %v2612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2614 = stablehlo.transpose %s2b1pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2615 = stablehlo.reverse %v2614, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2616 = stablehlo.convolution(%v2613, %v2615)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2617 = stablehlo.reshape %v2616 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2618 = stablehlo.multiply %v435, %v435 : tensor<32x301056xf32>
    %v2619 = stablehlo.multiply %v2618, %v435 : tensor<32x301056xf32>
    %v2620 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2621 = stablehlo.multiply %v2620, %v2619 : tensor<32x301056xf32>
    %v2622 = stablehlo.add %v435, %v2621 : tensor<32x301056xf32>
    %v2623 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2624 = stablehlo.multiply %v2623, %v2622 : tensor<32x301056xf32>
    %v2625 = stablehlo.tanh %v2624 : tensor<32x301056xf32>
    %v2626 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2627 = stablehlo.add %v2626, %v2625 : tensor<32x301056xf32>
    %v2628 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2629 = stablehlo.multiply %v2628, %v2627 : tensor<32x301056xf32>
    %v2630 = stablehlo.multiply %v2625, %v2625 : tensor<32x301056xf32>
    %v2631 = stablehlo.subtract %v2626, %v2630 : tensor<32x301056xf32>
    %v2632 = stablehlo.multiply %v2628, %v435 : tensor<32x301056xf32>
    %v2633 = stablehlo.multiply %v2632, %v2631 : tensor<32x301056xf32>
    %v2634 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2635 = stablehlo.multiply %v2634, %v2618 : tensor<32x301056xf32>
    %v2636 = stablehlo.add %v2626, %v2635 : tensor<32x301056xf32>
    %v2637 = stablehlo.multiply %v2623, %v2636 : tensor<32x301056xf32>
    %v2638 = stablehlo.multiply %v2633, %v2637 : tensor<32x301056xf32>
    %v2639 = stablehlo.add %v2629, %v2638 : tensor<32x301056xf32>
    %v2640 = stablehlo.multiply %v2617, %v2639 : tensor<32x301056xf32>
    %v2641 = stablehlo.reshape %v2640 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2642 = stablehlo.transpose %s2b1eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2643 = stablehlo.reverse %v2642, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2644 = stablehlo.convolution(%v2641, %v2643)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2645 = stablehlo.reshape %v2644 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2646 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2647 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2648 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2649 = stablehlo.reduce(%v412 init: %v2646) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2650 = stablehlo.broadcast_in_dim %v2649, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2651 = stablehlo.divide %v2650, %v2647 : tensor<32x75264xf32>
    %v2652 = stablehlo.subtract %v412, %v2651 : tensor<32x75264xf32>
    %v2653 = stablehlo.multiply %v2652, %v2652 : tensor<32x75264xf32>
    %v2654 = stablehlo.reduce(%v2653 init: %v2646) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2655 = stablehlo.broadcast_in_dim %v2654, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2656 = stablehlo.divide %v2655, %v2647 : tensor<32x75264xf32>
    %v2657 = stablehlo.add %v2656, %v2648 : tensor<32x75264xf32>
    %v2658 = stablehlo.rsqrt %v2657 : tensor<32x75264xf32>
    %v2659 = stablehlo.multiply %v2652, %v2658 : tensor<32x75264xf32>
    %v2660 = stablehlo.broadcast_in_dim %s2b1ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2661 = stablehlo.multiply %v2660, %v2645 : tensor<32x75264xf32>
    %v2662 = stablehlo.reduce(%v2661 init: %v2646) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2663 = stablehlo.broadcast_in_dim %v2662, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2664 = stablehlo.multiply %v2659, %v2661 : tensor<32x75264xf32>
    %v2665 = stablehlo.reduce(%v2664 init: %v2646) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2666 = stablehlo.broadcast_in_dim %v2665, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2667 = stablehlo.multiply %v2661, %v2647 : tensor<32x75264xf32>
    %v2668 = stablehlo.subtract %v2667, %v2663 : tensor<32x75264xf32>
    %v2669 = stablehlo.multiply %v2659, %v2666 : tensor<32x75264xf32>
    %v2670 = stablehlo.subtract %v2668, %v2669 : tensor<32x75264xf32>
    %v2671 = stablehlo.divide %v2658, %v2647 : tensor<32x75264xf32>
    %v2672 = stablehlo.multiply %v2671, %v2670 : tensor<32x75264xf32>
    %v2673 = stablehlo.reshape %v2672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2674 = stablehlo.reverse %s2b1dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2675 = stablehlo.convolution(%v2673, %v2674)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2676 = stablehlo.reshape %v2675 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2677 = stablehlo.add %v2676, %v2531 : tensor<32x75264xf32>
    %v2678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2679 = stablehlo.reshape %v453 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2680 = stablehlo.reshape %v2531 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2681 = stablehlo.multiply %v2679, %v2680 : tensor<32x384x14x14xf32>
    %v2682 = stablehlo.reduce(%v2681 init: %v2678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2683 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2684 = stablehlo.multiply %v2682, %v2683 : tensor<384xf32>
    %v2685 = stablehlo.subtract %s2b1lg, %v2684 : tensor<384xf32>
    %v2686 = stablehlo.reshape %v448 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2687 = stablehlo.reshape %v2612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2688 = stablehlo.transpose %v2686, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2689 = stablehlo.transpose %v2687, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2690 = stablehlo.convolution(%v2688, %v2689)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2691 = stablehlo.transpose %v2690, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2692 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2693 = stablehlo.multiply %v2691, %v2692 : tensor<384x1536x1x1xf32>
    %v2694 = stablehlo.subtract %s2b1pW, %v2693 : tensor<384x1536x1x1xf32>
    %v2695 = stablehlo.reshape %v2612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2696 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2697 = stablehlo.reduce(%v2695 init: %v2696) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2698 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2699 = stablehlo.multiply %v2697, %v2698 : tensor<384xf32>
    %v2700 = stablehlo.subtract %s2b1pb, %v2699 : tensor<384xf32>
    %v2701 = stablehlo.reshape %v430 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2702 = stablehlo.reshape %v2640 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2703 = stablehlo.transpose %v2701, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2704 = stablehlo.transpose %v2702, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2705 = stablehlo.convolution(%v2703, %v2704)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2706 = stablehlo.transpose %v2705, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2707 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2708 = stablehlo.multiply %v2706, %v2707 : tensor<1536x384x1x1xf32>
    %v2709 = stablehlo.subtract %s2b1eW, %v2708 : tensor<1536x384x1x1xf32>
    %v2710 = stablehlo.reshape %v2640 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2712 = stablehlo.reduce(%v2710 init: %v2711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2713 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2714 = stablehlo.multiply %v2712, %v2713 : tensor<1536xf32>
    %v2715 = stablehlo.subtract %s2b1eb, %v2714 : tensor<1536xf32>
    %v2716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2717 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2718 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2719 = stablehlo.reduce(%v412 init: %v2716) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2720 = stablehlo.broadcast_in_dim %v2719, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2721 = stablehlo.divide %v2720, %v2717 : tensor<32x75264xf32>
    %v2722 = stablehlo.subtract %v412, %v2721 : tensor<32x75264xf32>
    %v2723 = stablehlo.multiply %v2722, %v2722 : tensor<32x75264xf32>
    %v2724 = stablehlo.reduce(%v2723 init: %v2716) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2725 = stablehlo.broadcast_in_dim %v2724, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2726 = stablehlo.divide %v2725, %v2717 : tensor<32x75264xf32>
    %v2727 = stablehlo.add %v2726, %v2718 : tensor<32x75264xf32>
    %v2728 = stablehlo.rsqrt %v2727 : tensor<32x75264xf32>
    %v2729 = stablehlo.multiply %v2722, %v2728 : tensor<32x75264xf32>
    %v2730 = stablehlo.multiply %v2645, %v2729 : tensor<32x75264xf32>
    %v2731 = stablehlo.reduce(%v2730 init: %v2716) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2732 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2733 = stablehlo.multiply %v2731, %v2732 : tensor<f32>
    %v2734 = stablehlo.subtract %s2b1ng, %v2733 : tensor<f32>
    %v2735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2736 = stablehlo.reduce(%v2645 init: %v2735) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2737 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2738 = stablehlo.multiply %v2736, %v2737 : tensor<f32>
    %v2739 = stablehlo.subtract %s2b1nbt, %v2738 : tensor<f32>
    %v2740 = stablehlo.reshape %v407 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2741 = stablehlo.reshape %v2672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2742 = stablehlo.transpose %v2740, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2743 = stablehlo.transpose %v2741, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2744 = stablehlo.convolution(%v2742, %v2743)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2745 = stablehlo.reshape %v2744 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2746 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2747 = stablehlo.multiply %v2745, %v2746 : tensor<384x1x7x7xf32>
    %v2748 = stablehlo.subtract %s2b1dW, %v2747 : tensor<384x1x7x7xf32>
    %v2749 = stablehlo.reshape %v2672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2751 = stablehlo.reduce(%v2749 init: %v2750) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2752 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2753 = stablehlo.multiply %v2751, %v2752 : tensor<384xf32>
    %v2754 = stablehlo.subtract %s2b1db, %v2753 : tensor<384xf32>
    %v2755 = stablehlo.reshape %v2677 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2756 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2757 = stablehlo.multiply %v2755, %v2756 : tensor<32x384x14x14xf32>
    %v2758 = stablehlo.reshape %v2757 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2759 = stablehlo.reshape %v2758 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2760 = stablehlo.transpose %s2b0pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2761 = stablehlo.reverse %v2760, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2762 = stablehlo.convolution(%v2759, %v2761)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2763 = stablehlo.reshape %v2762 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2764 = stablehlo.multiply %v384, %v384 : tensor<32x301056xf32>
    %v2765 = stablehlo.multiply %v2764, %v384 : tensor<32x301056xf32>
    %v2766 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2767 = stablehlo.multiply %v2766, %v2765 : tensor<32x301056xf32>
    %v2768 = stablehlo.add %v384, %v2767 : tensor<32x301056xf32>
    %v2769 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2770 = stablehlo.multiply %v2769, %v2768 : tensor<32x301056xf32>
    %v2771 = stablehlo.tanh %v2770 : tensor<32x301056xf32>
    %v2772 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2773 = stablehlo.add %v2772, %v2771 : tensor<32x301056xf32>
    %v2774 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2775 = stablehlo.multiply %v2774, %v2773 : tensor<32x301056xf32>
    %v2776 = stablehlo.multiply %v2771, %v2771 : tensor<32x301056xf32>
    %v2777 = stablehlo.subtract %v2772, %v2776 : tensor<32x301056xf32>
    %v2778 = stablehlo.multiply %v2774, %v384 : tensor<32x301056xf32>
    %v2779 = stablehlo.multiply %v2778, %v2777 : tensor<32x301056xf32>
    %v2780 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2781 = stablehlo.multiply %v2780, %v2764 : tensor<32x301056xf32>
    %v2782 = stablehlo.add %v2772, %v2781 : tensor<32x301056xf32>
    %v2783 = stablehlo.multiply %v2769, %v2782 : tensor<32x301056xf32>
    %v2784 = stablehlo.multiply %v2779, %v2783 : tensor<32x301056xf32>
    %v2785 = stablehlo.add %v2775, %v2784 : tensor<32x301056xf32>
    %v2786 = stablehlo.multiply %v2763, %v2785 : tensor<32x301056xf32>
    %v2787 = stablehlo.reshape %v2786 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2788 = stablehlo.transpose %s2b0eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2789 = stablehlo.reverse %v2788, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2790 = stablehlo.convolution(%v2787, %v2789)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2791 = stablehlo.reshape %v2790 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2792 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2793 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2794 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2795 = stablehlo.reduce(%v361 init: %v2792) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2796 = stablehlo.broadcast_in_dim %v2795, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2797 = stablehlo.divide %v2796, %v2793 : tensor<32x75264xf32>
    %v2798 = stablehlo.subtract %v361, %v2797 : tensor<32x75264xf32>
    %v2799 = stablehlo.multiply %v2798, %v2798 : tensor<32x75264xf32>
    %v2800 = stablehlo.reduce(%v2799 init: %v2792) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2801 = stablehlo.broadcast_in_dim %v2800, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2802 = stablehlo.divide %v2801, %v2793 : tensor<32x75264xf32>
    %v2803 = stablehlo.add %v2802, %v2794 : tensor<32x75264xf32>
    %v2804 = stablehlo.rsqrt %v2803 : tensor<32x75264xf32>
    %v2805 = stablehlo.multiply %v2798, %v2804 : tensor<32x75264xf32>
    %v2806 = stablehlo.broadcast_in_dim %s2b0ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2807 = stablehlo.multiply %v2806, %v2791 : tensor<32x75264xf32>
    %v2808 = stablehlo.reduce(%v2807 init: %v2792) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2809 = stablehlo.broadcast_in_dim %v2808, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2810 = stablehlo.multiply %v2805, %v2807 : tensor<32x75264xf32>
    %v2811 = stablehlo.reduce(%v2810 init: %v2792) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2812 = stablehlo.broadcast_in_dim %v2811, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2813 = stablehlo.multiply %v2807, %v2793 : tensor<32x75264xf32>
    %v2814 = stablehlo.subtract %v2813, %v2809 : tensor<32x75264xf32>
    %v2815 = stablehlo.multiply %v2805, %v2812 : tensor<32x75264xf32>
    %v2816 = stablehlo.subtract %v2814, %v2815 : tensor<32x75264xf32>
    %v2817 = stablehlo.divide %v2804, %v2793 : tensor<32x75264xf32>
    %v2818 = stablehlo.multiply %v2817, %v2816 : tensor<32x75264xf32>
    %v2819 = stablehlo.reshape %v2818 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2820 = stablehlo.reverse %s2b0dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2821 = stablehlo.convolution(%v2819, %v2820)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2822 = stablehlo.reshape %v2821 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2823 = stablehlo.add %v2822, %v2677 : tensor<32x75264xf32>
    %v2824 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2825 = stablehlo.reshape %v402 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2826 = stablehlo.reshape %v2677 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2827 = stablehlo.multiply %v2825, %v2826 : tensor<32x384x14x14xf32>
    %v2828 = stablehlo.reduce(%v2827 init: %v2824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2829 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2830 = stablehlo.multiply %v2828, %v2829 : tensor<384xf32>
    %v2831 = stablehlo.subtract %s2b0lg, %v2830 : tensor<384xf32>
    %v2832 = stablehlo.reshape %v397 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2833 = stablehlo.reshape %v2758 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2834 = stablehlo.transpose %v2832, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2835 = stablehlo.transpose %v2833, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2836 = stablehlo.convolution(%v2834, %v2835)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2837 = stablehlo.transpose %v2836, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2838 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2839 = stablehlo.multiply %v2837, %v2838 : tensor<384x1536x1x1xf32>
    %v2840 = stablehlo.subtract %s2b0pW, %v2839 : tensor<384x1536x1x1xf32>
    %v2841 = stablehlo.reshape %v2758 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2843 = stablehlo.reduce(%v2841 init: %v2842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2844 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2845 = stablehlo.multiply %v2843, %v2844 : tensor<384xf32>
    %v2846 = stablehlo.subtract %s2b0pb, %v2845 : tensor<384xf32>
    %v2847 = stablehlo.reshape %v379 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2848 = stablehlo.reshape %v2786 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2849 = stablehlo.transpose %v2847, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2850 = stablehlo.transpose %v2848, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2851 = stablehlo.convolution(%v2849, %v2850)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2852 = stablehlo.transpose %v2851, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2853 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2854 = stablehlo.multiply %v2852, %v2853 : tensor<1536x384x1x1xf32>
    %v2855 = stablehlo.subtract %s2b0eW, %v2854 : tensor<1536x384x1x1xf32>
    %v2856 = stablehlo.reshape %v2786 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2857 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2858 = stablehlo.reduce(%v2856 init: %v2857) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2859 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2860 = stablehlo.multiply %v2858, %v2859 : tensor<1536xf32>
    %v2861 = stablehlo.subtract %s2b0eb, %v2860 : tensor<1536xf32>
    %v2862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2863 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2864 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2865 = stablehlo.reduce(%v361 init: %v2862) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2866 = stablehlo.broadcast_in_dim %v2865, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2867 = stablehlo.divide %v2866, %v2863 : tensor<32x75264xf32>
    %v2868 = stablehlo.subtract %v361, %v2867 : tensor<32x75264xf32>
    %v2869 = stablehlo.multiply %v2868, %v2868 : tensor<32x75264xf32>
    %v2870 = stablehlo.reduce(%v2869 init: %v2862) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2871 = stablehlo.broadcast_in_dim %v2870, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2872 = stablehlo.divide %v2871, %v2863 : tensor<32x75264xf32>
    %v2873 = stablehlo.add %v2872, %v2864 : tensor<32x75264xf32>
    %v2874 = stablehlo.rsqrt %v2873 : tensor<32x75264xf32>
    %v2875 = stablehlo.multiply %v2868, %v2874 : tensor<32x75264xf32>
    %v2876 = stablehlo.multiply %v2791, %v2875 : tensor<32x75264xf32>
    %v2877 = stablehlo.reduce(%v2876 init: %v2862) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2878 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2879 = stablehlo.multiply %v2877, %v2878 : tensor<f32>
    %v2880 = stablehlo.subtract %s2b0ng, %v2879 : tensor<f32>
    %v2881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2882 = stablehlo.reduce(%v2791 init: %v2881) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2883 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2884 = stablehlo.multiply %v2882, %v2883 : tensor<f32>
    %v2885 = stablehlo.subtract %s2b0nbt, %v2884 : tensor<f32>
    %v2886 = stablehlo.reshape %v356 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2887 = stablehlo.reshape %v2818 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2888 = stablehlo.transpose %v2886, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2889 = stablehlo.transpose %v2887, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2890 = stablehlo.convolution(%v2888, %v2889)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2891 = stablehlo.reshape %v2890 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2892 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2893 = stablehlo.multiply %v2891, %v2892 : tensor<384x1x7x7xf32>
    %v2894 = stablehlo.subtract %s2b0dW, %v2893 : tensor<384x1x7x7xf32>
    %v2895 = stablehlo.reshape %v2818 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2896 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2897 = stablehlo.reduce(%v2895 init: %v2896) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2898 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2899 = stablehlo.multiply %v2897, %v2898 : tensor<384xf32>
    %v2900 = stablehlo.subtract %s2b0db, %v2899 : tensor<384xf32>
    %v2901 = stablehlo.reshape %v2823 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2902 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2903 = stablehlo.pad %v2901, %v2902, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x28x28xf32>
    %v2904 = stablehlo.transpose %d1W, dims = [1, 0, 2, 3] : (tensor<384x192x2x2xf32>) -> tensor<192x384x2x2xf32>
    %v2905 = stablehlo.reverse %v2904, dims = [2, 3] : tensor<192x384x2x2xf32>
    %v2906 = stablehlo.convolution(%v2903, %v2905)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x28x28xf32>, tensor<192x384x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v2907 = stablehlo.reshape %v2906 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2908 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2909 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2910 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2911 = stablehlo.reduce(%v333 init: %v2908) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2912 = stablehlo.broadcast_in_dim %v2911, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2913 = stablehlo.divide %v2912, %v2909 : tensor<32x150528xf32>
    %v2914 = stablehlo.subtract %v333, %v2913 : tensor<32x150528xf32>
    %v2915 = stablehlo.multiply %v2914, %v2914 : tensor<32x150528xf32>
    %v2916 = stablehlo.reduce(%v2915 init: %v2908) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2917 = stablehlo.broadcast_in_dim %v2916, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2918 = stablehlo.divide %v2917, %v2909 : tensor<32x150528xf32>
    %v2919 = stablehlo.add %v2918, %v2910 : tensor<32x150528xf32>
    %v2920 = stablehlo.rsqrt %v2919 : tensor<32x150528xf32>
    %v2921 = stablehlo.multiply %v2914, %v2920 : tensor<32x150528xf32>
    %v2922 = stablehlo.broadcast_in_dim %d1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v2923 = stablehlo.multiply %v2922, %v2907 : tensor<32x150528xf32>
    %v2924 = stablehlo.reduce(%v2923 init: %v2908) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2925 = stablehlo.broadcast_in_dim %v2924, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2926 = stablehlo.multiply %v2921, %v2923 : tensor<32x150528xf32>
    %v2927 = stablehlo.reduce(%v2926 init: %v2908) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2928 = stablehlo.broadcast_in_dim %v2927, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2929 = stablehlo.multiply %v2923, %v2909 : tensor<32x150528xf32>
    %v2930 = stablehlo.subtract %v2929, %v2925 : tensor<32x150528xf32>
    %v2931 = stablehlo.multiply %v2921, %v2928 : tensor<32x150528xf32>
    %v2932 = stablehlo.subtract %v2930, %v2931 : tensor<32x150528xf32>
    %v2933 = stablehlo.divide %v2920, %v2909 : tensor<32x150528xf32>
    %v2934 = stablehlo.multiply %v2933, %v2932 : tensor<32x150528xf32>
    %v2935 = stablehlo.reshape %v2823 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2937 = stablehlo.reduce(%v2935 init: %v2936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2938 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2939 = stablehlo.multiply %v2937, %v2938 : tensor<384xf32>
    %v2940 = stablehlo.subtract %d1b, %v2939 : tensor<384xf32>
    %v2941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2942 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2943 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2944 = stablehlo.reduce(%v333 init: %v2941) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2945 = stablehlo.broadcast_in_dim %v2944, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2946 = stablehlo.divide %v2945, %v2942 : tensor<32x150528xf32>
    %v2947 = stablehlo.subtract %v333, %v2946 : tensor<32x150528xf32>
    %v2948 = stablehlo.multiply %v2947, %v2947 : tensor<32x150528xf32>
    %v2949 = stablehlo.reduce(%v2948 init: %v2941) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2950 = stablehlo.broadcast_in_dim %v2949, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2951 = stablehlo.divide %v2950, %v2942 : tensor<32x150528xf32>
    %v2952 = stablehlo.add %v2951, %v2943 : tensor<32x150528xf32>
    %v2953 = stablehlo.rsqrt %v2952 : tensor<32x150528xf32>
    %v2954 = stablehlo.multiply %v2947, %v2953 : tensor<32x150528xf32>
    %v2955 = stablehlo.multiply %v2907, %v2954 : tensor<32x150528xf32>
    %v2956 = stablehlo.reduce(%v2955 init: %v2941) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2957 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2958 = stablehlo.multiply %v2956, %v2957 : tensor<f32>
    %v2959 = stablehlo.subtract %d1ng, %v2958 : tensor<f32>
    %v2960 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2961 = stablehlo.reduce(%v2907 init: %v2960) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2962 = stablehlo.constant dense<0.1> : tensor<f32>
    %v2963 = stablehlo.multiply %v2961, %v2962 : tensor<f32>
    %v2964 = stablehlo.subtract %d1nbt, %v2963 : tensor<f32>
    %dd1Wxi = stablehlo.reshape %v351 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %dd1Wdi = stablehlo.reshape %v2823 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %dd1Wu = stablehlo.pad %dd1Wdi, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x27x27xf32>
    %dd1Wxt = stablehlo.transpose %dd1Wxi, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %dd1Wdt = stablehlo.transpose %dd1Wu, dims = [1, 0, 2, 3] : (tensor<32x384x27x27xf32>) -> tensor<384x32x27x27xf32>
    %dd1Wraw = stablehlo.convolution(%dd1Wxt, %dd1Wdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<384x32x27x27xf32>) -> tensor<192x384x2x2xf32>
    %dd1W = stablehlo.transpose %dd1Wraw, dims = [1, 0, 2, 3] : (tensor<192x384x2x2xf32>) -> tensor<384x192x2x2xf32>
    %d1Wl = stablehlo.constant dense<0.1> : tensor<384x192x2x2xf32>
    %d1Ws = stablehlo.multiply %dd1W, %d1Wl : tensor<384x192x2x2xf32>
    %d1Wn = stablehlo.subtract %d1W, %d1Ws : tensor<384x192x2x2xf32>
    %v2965 = stablehlo.reshape %v2934 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2966 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v2967 = stablehlo.multiply %v2965, %v2966 : tensor<32x192x28x28xf32>
    %v2968 = stablehlo.reshape %v2967 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2969 = stablehlo.reshape %v2968 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2970 = stablehlo.transpose %s1b2pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2971 = stablehlo.reverse %v2970, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v2972 = stablehlo.convolution(%v2969, %v2971)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v2973 = stablehlo.reshape %v2972 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v2974 = stablehlo.multiply %v310, %v310 : tensor<32x602112xf32>
    %v2975 = stablehlo.multiply %v2974, %v310 : tensor<32x602112xf32>
    %v2976 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v2977 = stablehlo.multiply %v2976, %v2975 : tensor<32x602112xf32>
    %v2978 = stablehlo.add %v310, %v2977 : tensor<32x602112xf32>
    %v2979 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v2980 = stablehlo.multiply %v2979, %v2978 : tensor<32x602112xf32>
    %v2981 = stablehlo.tanh %v2980 : tensor<32x602112xf32>
    %v2982 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v2983 = stablehlo.add %v2982, %v2981 : tensor<32x602112xf32>
    %v2984 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v2985 = stablehlo.multiply %v2984, %v2983 : tensor<32x602112xf32>
    %v2986 = stablehlo.multiply %v2981, %v2981 : tensor<32x602112xf32>
    %v2987 = stablehlo.subtract %v2982, %v2986 : tensor<32x602112xf32>
    %v2988 = stablehlo.multiply %v2984, %v310 : tensor<32x602112xf32>
    %v2989 = stablehlo.multiply %v2988, %v2987 : tensor<32x602112xf32>
    %v2990 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v2991 = stablehlo.multiply %v2990, %v2974 : tensor<32x602112xf32>
    %v2992 = stablehlo.add %v2982, %v2991 : tensor<32x602112xf32>
    %v2993 = stablehlo.multiply %v2979, %v2992 : tensor<32x602112xf32>
    %v2994 = stablehlo.multiply %v2989, %v2993 : tensor<32x602112xf32>
    %v2995 = stablehlo.add %v2985, %v2994 : tensor<32x602112xf32>
    %v2996 = stablehlo.multiply %v2973, %v2995 : tensor<32x602112xf32>
    %v2997 = stablehlo.reshape %v2996 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2998 = stablehlo.transpose %s1b2eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2999 = stablehlo.reverse %v2998, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3000 = stablehlo.convolution(%v2997, %v2999)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3001 = stablehlo.reshape %v3000 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3003 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v3004 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v3005 = stablehlo.reduce(%v287 init: %v3002) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3006 = stablehlo.broadcast_in_dim %v3005, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3007 = stablehlo.divide %v3006, %v3003 : tensor<32x150528xf32>
    %v3008 = stablehlo.subtract %v287, %v3007 : tensor<32x150528xf32>
    %v3009 = stablehlo.multiply %v3008, %v3008 : tensor<32x150528xf32>
    %v3010 = stablehlo.reduce(%v3009 init: %v3002) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3011 = stablehlo.broadcast_in_dim %v3010, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3012 = stablehlo.divide %v3011, %v3003 : tensor<32x150528xf32>
    %v3013 = stablehlo.add %v3012, %v3004 : tensor<32x150528xf32>
    %v3014 = stablehlo.rsqrt %v3013 : tensor<32x150528xf32>
    %v3015 = stablehlo.multiply %v3008, %v3014 : tensor<32x150528xf32>
    %v3016 = stablehlo.broadcast_in_dim %s1b2ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v3017 = stablehlo.multiply %v3016, %v3001 : tensor<32x150528xf32>
    %v3018 = stablehlo.reduce(%v3017 init: %v3002) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3019 = stablehlo.broadcast_in_dim %v3018, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3020 = stablehlo.multiply %v3015, %v3017 : tensor<32x150528xf32>
    %v3021 = stablehlo.reduce(%v3020 init: %v3002) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3022 = stablehlo.broadcast_in_dim %v3021, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3023 = stablehlo.multiply %v3017, %v3003 : tensor<32x150528xf32>
    %v3024 = stablehlo.subtract %v3023, %v3019 : tensor<32x150528xf32>
    %v3025 = stablehlo.multiply %v3015, %v3022 : tensor<32x150528xf32>
    %v3026 = stablehlo.subtract %v3024, %v3025 : tensor<32x150528xf32>
    %v3027 = stablehlo.divide %v3014, %v3003 : tensor<32x150528xf32>
    %v3028 = stablehlo.multiply %v3027, %v3026 : tensor<32x150528xf32>
    %v3029 = stablehlo.reshape %v3028 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3030 = stablehlo.reverse %s1b2dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v3031 = stablehlo.convolution(%v3029, %v3030)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v3032 = stablehlo.reshape %v3031 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3033 = stablehlo.add %v3032, %v2934 : tensor<32x150528xf32>
    %v3034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3035 = stablehlo.reshape %v328 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3036 = stablehlo.reshape %v2934 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3037 = stablehlo.multiply %v3035, %v3036 : tensor<32x192x28x28xf32>
    %v3038 = stablehlo.reduce(%v3037 init: %v3034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3039 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3040 = stablehlo.multiply %v3038, %v3039 : tensor<192xf32>
    %v3041 = stablehlo.subtract %s1b2lg, %v3040 : tensor<192xf32>
    %v3042 = stablehlo.reshape %v323 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3043 = stablehlo.reshape %v2968 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3044 = stablehlo.transpose %v3042, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3045 = stablehlo.transpose %v3043, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3046 = stablehlo.convolution(%v3044, %v3045)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v3047 = stablehlo.transpose %v3046, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3048 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v3049 = stablehlo.multiply %v3047, %v3048 : tensor<192x768x1x1xf32>
    %v3050 = stablehlo.subtract %s1b2pW, %v3049 : tensor<192x768x1x1xf32>
    %v3051 = stablehlo.reshape %v2968 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3052 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3053 = stablehlo.reduce(%v3051 init: %v3052) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3054 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3055 = stablehlo.multiply %v3053, %v3054 : tensor<192xf32>
    %v3056 = stablehlo.subtract %s1b2pb, %v3055 : tensor<192xf32>
    %v3057 = stablehlo.reshape %v305 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3058 = stablehlo.reshape %v2996 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3059 = stablehlo.transpose %v3057, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3060 = stablehlo.transpose %v3058, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3061 = stablehlo.convolution(%v3059, %v3060)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v3062 = stablehlo.transpose %v3061, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3063 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v3064 = stablehlo.multiply %v3062, %v3063 : tensor<768x192x1x1xf32>
    %v3065 = stablehlo.subtract %s1b2eW, %v3064 : tensor<768x192x1x1xf32>
    %v3066 = stablehlo.reshape %v2996 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3068 = stablehlo.reduce(%v3066 init: %v3067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v3069 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v3070 = stablehlo.multiply %v3068, %v3069 : tensor<768xf32>
    %v3071 = stablehlo.subtract %s1b2eb, %v3070 : tensor<768xf32>
    %v3072 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3073 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v3074 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v3075 = stablehlo.reduce(%v287 init: %v3072) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3076 = stablehlo.broadcast_in_dim %v3075, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3077 = stablehlo.divide %v3076, %v3073 : tensor<32x150528xf32>
    %v3078 = stablehlo.subtract %v287, %v3077 : tensor<32x150528xf32>
    %v3079 = stablehlo.multiply %v3078, %v3078 : tensor<32x150528xf32>
    %v3080 = stablehlo.reduce(%v3079 init: %v3072) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3081 = stablehlo.broadcast_in_dim %v3080, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3082 = stablehlo.divide %v3081, %v3073 : tensor<32x150528xf32>
    %v3083 = stablehlo.add %v3082, %v3074 : tensor<32x150528xf32>
    %v3084 = stablehlo.rsqrt %v3083 : tensor<32x150528xf32>
    %v3085 = stablehlo.multiply %v3078, %v3084 : tensor<32x150528xf32>
    %v3086 = stablehlo.multiply %v3001, %v3085 : tensor<32x150528xf32>
    %v3087 = stablehlo.reduce(%v3086 init: %v3072) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v3088 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3089 = stablehlo.multiply %v3087, %v3088 : tensor<f32>
    %v3090 = stablehlo.subtract %s1b2ng, %v3089 : tensor<f32>
    %v3091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3092 = stablehlo.reduce(%v3001 init: %v3091) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v3093 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3094 = stablehlo.multiply %v3092, %v3093 : tensor<f32>
    %v3095 = stablehlo.subtract %s1b2nbt, %v3094 : tensor<f32>
    %v3096 = stablehlo.reshape %v282 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3097 = stablehlo.reshape %v3028 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3098 = stablehlo.transpose %v3096, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3099 = stablehlo.transpose %v3097, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3100 = stablehlo.convolution(%v3098, %v3099)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v3102 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v3103 = stablehlo.multiply %v3101, %v3102 : tensor<192x1x7x7xf32>
    %v3104 = stablehlo.subtract %s1b2dW, %v3103 : tensor<192x1x7x7xf32>
    %v3105 = stablehlo.reshape %v3028 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3107 = stablehlo.reduce(%v3105 init: %v3106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3108 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3109 = stablehlo.multiply %v3107, %v3108 : tensor<192xf32>
    %v3110 = stablehlo.subtract %s1b2db, %v3109 : tensor<192xf32>
    %v3111 = stablehlo.reshape %v3033 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3112 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3113 = stablehlo.multiply %v3111, %v3112 : tensor<32x192x28x28xf32>
    %v3114 = stablehlo.reshape %v3113 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3115 = stablehlo.reshape %v3114 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3116 = stablehlo.transpose %s1b1pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3117 = stablehlo.reverse %v3116, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v3118 = stablehlo.convolution(%v3115, %v3117)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v3119 = stablehlo.reshape %v3118 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3120 = stablehlo.multiply %v259, %v259 : tensor<32x602112xf32>
    %v3121 = stablehlo.multiply %v3120, %v259 : tensor<32x602112xf32>
    %v3122 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v3123 = stablehlo.multiply %v3122, %v3121 : tensor<32x602112xf32>
    %v3124 = stablehlo.add %v259, %v3123 : tensor<32x602112xf32>
    %v3125 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v3126 = stablehlo.multiply %v3125, %v3124 : tensor<32x602112xf32>
    %v3127 = stablehlo.tanh %v3126 : tensor<32x602112xf32>
    %v3128 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v3129 = stablehlo.add %v3128, %v3127 : tensor<32x602112xf32>
    %v3130 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v3131 = stablehlo.multiply %v3130, %v3129 : tensor<32x602112xf32>
    %v3132 = stablehlo.multiply %v3127, %v3127 : tensor<32x602112xf32>
    %v3133 = stablehlo.subtract %v3128, %v3132 : tensor<32x602112xf32>
    %v3134 = stablehlo.multiply %v3130, %v259 : tensor<32x602112xf32>
    %v3135 = stablehlo.multiply %v3134, %v3133 : tensor<32x602112xf32>
    %v3136 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v3137 = stablehlo.multiply %v3136, %v3120 : tensor<32x602112xf32>
    %v3138 = stablehlo.add %v3128, %v3137 : tensor<32x602112xf32>
    %v3139 = stablehlo.multiply %v3125, %v3138 : tensor<32x602112xf32>
    %v3140 = stablehlo.multiply %v3135, %v3139 : tensor<32x602112xf32>
    %v3141 = stablehlo.add %v3131, %v3140 : tensor<32x602112xf32>
    %v3142 = stablehlo.multiply %v3119, %v3141 : tensor<32x602112xf32>
    %v3143 = stablehlo.reshape %v3142 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3144 = stablehlo.transpose %s1b1eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3145 = stablehlo.reverse %v3144, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3146 = stablehlo.convolution(%v3143, %v3145)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3147 = stablehlo.reshape %v3146 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3149 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v3150 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v3151 = stablehlo.reduce(%v236 init: %v3148) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3152 = stablehlo.broadcast_in_dim %v3151, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3153 = stablehlo.divide %v3152, %v3149 : tensor<32x150528xf32>
    %v3154 = stablehlo.subtract %v236, %v3153 : tensor<32x150528xf32>
    %v3155 = stablehlo.multiply %v3154, %v3154 : tensor<32x150528xf32>
    %v3156 = stablehlo.reduce(%v3155 init: %v3148) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3157 = stablehlo.broadcast_in_dim %v3156, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3158 = stablehlo.divide %v3157, %v3149 : tensor<32x150528xf32>
    %v3159 = stablehlo.add %v3158, %v3150 : tensor<32x150528xf32>
    %v3160 = stablehlo.rsqrt %v3159 : tensor<32x150528xf32>
    %v3161 = stablehlo.multiply %v3154, %v3160 : tensor<32x150528xf32>
    %v3162 = stablehlo.broadcast_in_dim %s1b1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v3163 = stablehlo.multiply %v3162, %v3147 : tensor<32x150528xf32>
    %v3164 = stablehlo.reduce(%v3163 init: %v3148) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3165 = stablehlo.broadcast_in_dim %v3164, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3166 = stablehlo.multiply %v3161, %v3163 : tensor<32x150528xf32>
    %v3167 = stablehlo.reduce(%v3166 init: %v3148) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3168 = stablehlo.broadcast_in_dim %v3167, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3169 = stablehlo.multiply %v3163, %v3149 : tensor<32x150528xf32>
    %v3170 = stablehlo.subtract %v3169, %v3165 : tensor<32x150528xf32>
    %v3171 = stablehlo.multiply %v3161, %v3168 : tensor<32x150528xf32>
    %v3172 = stablehlo.subtract %v3170, %v3171 : tensor<32x150528xf32>
    %v3173 = stablehlo.divide %v3160, %v3149 : tensor<32x150528xf32>
    %v3174 = stablehlo.multiply %v3173, %v3172 : tensor<32x150528xf32>
    %v3175 = stablehlo.reshape %v3174 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3176 = stablehlo.reverse %s1b1dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v3177 = stablehlo.convolution(%v3175, %v3176)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v3178 = stablehlo.reshape %v3177 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3179 = stablehlo.add %v3178, %v3033 : tensor<32x150528xf32>
    %v3180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3181 = stablehlo.reshape %v277 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3182 = stablehlo.reshape %v3033 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3183 = stablehlo.multiply %v3181, %v3182 : tensor<32x192x28x28xf32>
    %v3184 = stablehlo.reduce(%v3183 init: %v3180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3185 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3186 = stablehlo.multiply %v3184, %v3185 : tensor<192xf32>
    %v3187 = stablehlo.subtract %s1b1lg, %v3186 : tensor<192xf32>
    %v3188 = stablehlo.reshape %v272 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3189 = stablehlo.reshape %v3114 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3190 = stablehlo.transpose %v3188, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3191 = stablehlo.transpose %v3189, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3192 = stablehlo.convolution(%v3190, %v3191)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v3193 = stablehlo.transpose %v3192, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3194 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v3195 = stablehlo.multiply %v3193, %v3194 : tensor<192x768x1x1xf32>
    %v3196 = stablehlo.subtract %s1b1pW, %v3195 : tensor<192x768x1x1xf32>
    %v3197 = stablehlo.reshape %v3114 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3199 = stablehlo.reduce(%v3197 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3200 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3201 = stablehlo.multiply %v3199, %v3200 : tensor<192xf32>
    %v3202 = stablehlo.subtract %s1b1pb, %v3201 : tensor<192xf32>
    %v3203 = stablehlo.reshape %v254 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3204 = stablehlo.reshape %v3142 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3205 = stablehlo.transpose %v3203, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3206 = stablehlo.transpose %v3204, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3207 = stablehlo.convolution(%v3205, %v3206)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v3208 = stablehlo.transpose %v3207, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3209 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v3210 = stablehlo.multiply %v3208, %v3209 : tensor<768x192x1x1xf32>
    %v3211 = stablehlo.subtract %s1b1eW, %v3210 : tensor<768x192x1x1xf32>
    %v3212 = stablehlo.reshape %v3142 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3214 = stablehlo.reduce(%v3212 init: %v3213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v3215 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v3216 = stablehlo.multiply %v3214, %v3215 : tensor<768xf32>
    %v3217 = stablehlo.subtract %s1b1eb, %v3216 : tensor<768xf32>
    %v3218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3219 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v3220 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v3221 = stablehlo.reduce(%v236 init: %v3218) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3222 = stablehlo.broadcast_in_dim %v3221, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3223 = stablehlo.divide %v3222, %v3219 : tensor<32x150528xf32>
    %v3224 = stablehlo.subtract %v236, %v3223 : tensor<32x150528xf32>
    %v3225 = stablehlo.multiply %v3224, %v3224 : tensor<32x150528xf32>
    %v3226 = stablehlo.reduce(%v3225 init: %v3218) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3227 = stablehlo.broadcast_in_dim %v3226, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3228 = stablehlo.divide %v3227, %v3219 : tensor<32x150528xf32>
    %v3229 = stablehlo.add %v3228, %v3220 : tensor<32x150528xf32>
    %v3230 = stablehlo.rsqrt %v3229 : tensor<32x150528xf32>
    %v3231 = stablehlo.multiply %v3224, %v3230 : tensor<32x150528xf32>
    %v3232 = stablehlo.multiply %v3147, %v3231 : tensor<32x150528xf32>
    %v3233 = stablehlo.reduce(%v3232 init: %v3218) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v3234 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3235 = stablehlo.multiply %v3233, %v3234 : tensor<f32>
    %v3236 = stablehlo.subtract %s1b1ng, %v3235 : tensor<f32>
    %v3237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3238 = stablehlo.reduce(%v3147 init: %v3237) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v3239 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3240 = stablehlo.multiply %v3238, %v3239 : tensor<f32>
    %v3241 = stablehlo.subtract %s1b1nbt, %v3240 : tensor<f32>
    %v3242 = stablehlo.reshape %v231 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3243 = stablehlo.reshape %v3174 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3244 = stablehlo.transpose %v3242, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3245 = stablehlo.transpose %v3243, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3246 = stablehlo.convolution(%v3244, %v3245)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v3247 = stablehlo.reshape %v3246 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v3248 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v3249 = stablehlo.multiply %v3247, %v3248 : tensor<192x1x7x7xf32>
    %v3250 = stablehlo.subtract %s1b1dW, %v3249 : tensor<192x1x7x7xf32>
    %v3251 = stablehlo.reshape %v3174 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3252 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3253 = stablehlo.reduce(%v3251 init: %v3252) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3254 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3255 = stablehlo.multiply %v3253, %v3254 : tensor<192xf32>
    %v3256 = stablehlo.subtract %s1b1db, %v3255 : tensor<192xf32>
    %v3257 = stablehlo.reshape %v3179 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3258 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3259 = stablehlo.multiply %v3257, %v3258 : tensor<32x192x28x28xf32>
    %v3260 = stablehlo.reshape %v3259 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3261 = stablehlo.reshape %v3260 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3262 = stablehlo.transpose %s1b0pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3263 = stablehlo.reverse %v3262, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v3264 = stablehlo.convolution(%v3261, %v3263)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v3265 = stablehlo.reshape %v3264 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3266 = stablehlo.multiply %v208, %v208 : tensor<32x602112xf32>
    %v3267 = stablehlo.multiply %v3266, %v208 : tensor<32x602112xf32>
    %v3268 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v3269 = stablehlo.multiply %v3268, %v3267 : tensor<32x602112xf32>
    %v3270 = stablehlo.add %v208, %v3269 : tensor<32x602112xf32>
    %v3271 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v3272 = stablehlo.multiply %v3271, %v3270 : tensor<32x602112xf32>
    %v3273 = stablehlo.tanh %v3272 : tensor<32x602112xf32>
    %v3274 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v3275 = stablehlo.add %v3274, %v3273 : tensor<32x602112xf32>
    %v3276 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v3277 = stablehlo.multiply %v3276, %v3275 : tensor<32x602112xf32>
    %v3278 = stablehlo.multiply %v3273, %v3273 : tensor<32x602112xf32>
    %v3279 = stablehlo.subtract %v3274, %v3278 : tensor<32x602112xf32>
    %v3280 = stablehlo.multiply %v3276, %v208 : tensor<32x602112xf32>
    %v3281 = stablehlo.multiply %v3280, %v3279 : tensor<32x602112xf32>
    %v3282 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v3283 = stablehlo.multiply %v3282, %v3266 : tensor<32x602112xf32>
    %v3284 = stablehlo.add %v3274, %v3283 : tensor<32x602112xf32>
    %v3285 = stablehlo.multiply %v3271, %v3284 : tensor<32x602112xf32>
    %v3286 = stablehlo.multiply %v3281, %v3285 : tensor<32x602112xf32>
    %v3287 = stablehlo.add %v3277, %v3286 : tensor<32x602112xf32>
    %v3288 = stablehlo.multiply %v3265, %v3287 : tensor<32x602112xf32>
    %v3289 = stablehlo.reshape %v3288 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3290 = stablehlo.transpose %s1b0eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3291 = stablehlo.reverse %v3290, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3292 = stablehlo.convolution(%v3289, %v3291)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3293 = stablehlo.reshape %v3292 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3295 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v3296 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v3297 = stablehlo.reduce(%v185 init: %v3294) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3298 = stablehlo.broadcast_in_dim %v3297, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3299 = stablehlo.divide %v3298, %v3295 : tensor<32x150528xf32>
    %v3300 = stablehlo.subtract %v185, %v3299 : tensor<32x150528xf32>
    %v3301 = stablehlo.multiply %v3300, %v3300 : tensor<32x150528xf32>
    %v3302 = stablehlo.reduce(%v3301 init: %v3294) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3303 = stablehlo.broadcast_in_dim %v3302, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3304 = stablehlo.divide %v3303, %v3295 : tensor<32x150528xf32>
    %v3305 = stablehlo.add %v3304, %v3296 : tensor<32x150528xf32>
    %v3306 = stablehlo.rsqrt %v3305 : tensor<32x150528xf32>
    %v3307 = stablehlo.multiply %v3300, %v3306 : tensor<32x150528xf32>
    %v3308 = stablehlo.broadcast_in_dim %s1b0ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v3309 = stablehlo.multiply %v3308, %v3293 : tensor<32x150528xf32>
    %v3310 = stablehlo.reduce(%v3309 init: %v3294) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3311 = stablehlo.broadcast_in_dim %v3310, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3312 = stablehlo.multiply %v3307, %v3309 : tensor<32x150528xf32>
    %v3313 = stablehlo.reduce(%v3312 init: %v3294) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3314 = stablehlo.broadcast_in_dim %v3313, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3315 = stablehlo.multiply %v3309, %v3295 : tensor<32x150528xf32>
    %v3316 = stablehlo.subtract %v3315, %v3311 : tensor<32x150528xf32>
    %v3317 = stablehlo.multiply %v3307, %v3314 : tensor<32x150528xf32>
    %v3318 = stablehlo.subtract %v3316, %v3317 : tensor<32x150528xf32>
    %v3319 = stablehlo.divide %v3306, %v3295 : tensor<32x150528xf32>
    %v3320 = stablehlo.multiply %v3319, %v3318 : tensor<32x150528xf32>
    %v3321 = stablehlo.reshape %v3320 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3322 = stablehlo.reverse %s1b0dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v3323 = stablehlo.convolution(%v3321, %v3322)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v3324 = stablehlo.reshape %v3323 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3325 = stablehlo.add %v3324, %v3179 : tensor<32x150528xf32>
    %v3326 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3327 = stablehlo.reshape %v226 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3328 = stablehlo.reshape %v3179 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3329 = stablehlo.multiply %v3327, %v3328 : tensor<32x192x28x28xf32>
    %v3330 = stablehlo.reduce(%v3329 init: %v3326) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3331 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3332 = stablehlo.multiply %v3330, %v3331 : tensor<192xf32>
    %v3333 = stablehlo.subtract %s1b0lg, %v3332 : tensor<192xf32>
    %v3334 = stablehlo.reshape %v221 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3335 = stablehlo.reshape %v3260 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3336 = stablehlo.transpose %v3334, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3337 = stablehlo.transpose %v3335, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3338 = stablehlo.convolution(%v3336, %v3337)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v3339 = stablehlo.transpose %v3338, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3340 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v3341 = stablehlo.multiply %v3339, %v3340 : tensor<192x768x1x1xf32>
    %v3342 = stablehlo.subtract %s1b0pW, %v3341 : tensor<192x768x1x1xf32>
    %v3343 = stablehlo.reshape %v3260 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3344 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3345 = stablehlo.reduce(%v3343 init: %v3344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3346 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3347 = stablehlo.multiply %v3345, %v3346 : tensor<192xf32>
    %v3348 = stablehlo.subtract %s1b0pb, %v3347 : tensor<192xf32>
    %v3349 = stablehlo.reshape %v203 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3350 = stablehlo.reshape %v3288 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3351 = stablehlo.transpose %v3349, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3352 = stablehlo.transpose %v3350, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3353 = stablehlo.convolution(%v3351, %v3352)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v3354 = stablehlo.transpose %v3353, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3355 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v3356 = stablehlo.multiply %v3354, %v3355 : tensor<768x192x1x1xf32>
    %v3357 = stablehlo.subtract %s1b0eW, %v3356 : tensor<768x192x1x1xf32>
    %v3358 = stablehlo.reshape %v3288 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3360 = stablehlo.reduce(%v3358 init: %v3359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v3361 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v3362 = stablehlo.multiply %v3360, %v3361 : tensor<768xf32>
    %v3363 = stablehlo.subtract %s1b0eb, %v3362 : tensor<768xf32>
    %v3364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3365 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v3366 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v3367 = stablehlo.reduce(%v185 init: %v3364) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3368 = stablehlo.broadcast_in_dim %v3367, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3369 = stablehlo.divide %v3368, %v3365 : tensor<32x150528xf32>
    %v3370 = stablehlo.subtract %v185, %v3369 : tensor<32x150528xf32>
    %v3371 = stablehlo.multiply %v3370, %v3370 : tensor<32x150528xf32>
    %v3372 = stablehlo.reduce(%v3371 init: %v3364) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v3373 = stablehlo.broadcast_in_dim %v3372, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3374 = stablehlo.divide %v3373, %v3365 : tensor<32x150528xf32>
    %v3375 = stablehlo.add %v3374, %v3366 : tensor<32x150528xf32>
    %v3376 = stablehlo.rsqrt %v3375 : tensor<32x150528xf32>
    %v3377 = stablehlo.multiply %v3370, %v3376 : tensor<32x150528xf32>
    %v3378 = stablehlo.multiply %v3293, %v3377 : tensor<32x150528xf32>
    %v3379 = stablehlo.reduce(%v3378 init: %v3364) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v3380 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3381 = stablehlo.multiply %v3379, %v3380 : tensor<f32>
    %v3382 = stablehlo.subtract %s1b0ng, %v3381 : tensor<f32>
    %v3383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3384 = stablehlo.reduce(%v3293 init: %v3383) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v3385 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3386 = stablehlo.multiply %v3384, %v3385 : tensor<f32>
    %v3387 = stablehlo.subtract %s1b0nbt, %v3386 : tensor<f32>
    %v3388 = stablehlo.reshape %v180 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3389 = stablehlo.reshape %v3320 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3390 = stablehlo.transpose %v3388, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3391 = stablehlo.transpose %v3389, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3392 = stablehlo.convolution(%v3390, %v3391)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v3393 = stablehlo.reshape %v3392 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v3394 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v3395 = stablehlo.multiply %v3393, %v3394 : tensor<192x1x7x7xf32>
    %v3396 = stablehlo.subtract %s1b0dW, %v3395 : tensor<192x1x7x7xf32>
    %v3397 = stablehlo.reshape %v3320 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3399 = stablehlo.reduce(%v3397 init: %v3398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3400 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3401 = stablehlo.multiply %v3399, %v3400 : tensor<192xf32>
    %v3402 = stablehlo.subtract %s1b0db, %v3401 : tensor<192xf32>
    %v3403 = stablehlo.reshape %v3325 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3405 = stablehlo.pad %v3403, %v3404, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x56x56xf32>
    %v3406 = stablehlo.transpose %d0W, dims = [1, 0, 2, 3] : (tensor<192x96x2x2xf32>) -> tensor<96x192x2x2xf32>
    %v3407 = stablehlo.reverse %v3406, dims = [2, 3] : tensor<96x192x2x2xf32>
    %v3408 = stablehlo.convolution(%v3405, %v3407)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x56x56xf32>, tensor<96x192x2x2xf32>) -> tensor<32x96x56x56xf32>
    %v3409 = stablehlo.reshape %v3408 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3411 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3412 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3413 = stablehlo.reduce(%v157 init: %v3410) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3414 = stablehlo.broadcast_in_dim %v3413, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3415 = stablehlo.divide %v3414, %v3411 : tensor<32x301056xf32>
    %v3416 = stablehlo.subtract %v157, %v3415 : tensor<32x301056xf32>
    %v3417 = stablehlo.multiply %v3416, %v3416 : tensor<32x301056xf32>
    %v3418 = stablehlo.reduce(%v3417 init: %v3410) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3419 = stablehlo.broadcast_in_dim %v3418, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3420 = stablehlo.divide %v3419, %v3411 : tensor<32x301056xf32>
    %v3421 = stablehlo.add %v3420, %v3412 : tensor<32x301056xf32>
    %v3422 = stablehlo.rsqrt %v3421 : tensor<32x301056xf32>
    %v3423 = stablehlo.multiply %v3416, %v3422 : tensor<32x301056xf32>
    %v3424 = stablehlo.broadcast_in_dim %d0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3425 = stablehlo.multiply %v3424, %v3409 : tensor<32x301056xf32>
    %v3426 = stablehlo.reduce(%v3425 init: %v3410) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3427 = stablehlo.broadcast_in_dim %v3426, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3428 = stablehlo.multiply %v3423, %v3425 : tensor<32x301056xf32>
    %v3429 = stablehlo.reduce(%v3428 init: %v3410) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3430 = stablehlo.broadcast_in_dim %v3429, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3431 = stablehlo.multiply %v3425, %v3411 : tensor<32x301056xf32>
    %v3432 = stablehlo.subtract %v3431, %v3427 : tensor<32x301056xf32>
    %v3433 = stablehlo.multiply %v3423, %v3430 : tensor<32x301056xf32>
    %v3434 = stablehlo.subtract %v3432, %v3433 : tensor<32x301056xf32>
    %v3435 = stablehlo.divide %v3422, %v3411 : tensor<32x301056xf32>
    %v3436 = stablehlo.multiply %v3435, %v3434 : tensor<32x301056xf32>
    %v3437 = stablehlo.reshape %v3325 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3439 = stablehlo.reduce(%v3437 init: %v3438) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3440 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3441 = stablehlo.multiply %v3439, %v3440 : tensor<192xf32>
    %v3442 = stablehlo.subtract %d0b, %v3441 : tensor<192xf32>
    %v3443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3444 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3445 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3446 = stablehlo.reduce(%v157 init: %v3443) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3447 = stablehlo.broadcast_in_dim %v3446, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3448 = stablehlo.divide %v3447, %v3444 : tensor<32x301056xf32>
    %v3449 = stablehlo.subtract %v157, %v3448 : tensor<32x301056xf32>
    %v3450 = stablehlo.multiply %v3449, %v3449 : tensor<32x301056xf32>
    %v3451 = stablehlo.reduce(%v3450 init: %v3443) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3452 = stablehlo.broadcast_in_dim %v3451, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3453 = stablehlo.divide %v3452, %v3444 : tensor<32x301056xf32>
    %v3454 = stablehlo.add %v3453, %v3445 : tensor<32x301056xf32>
    %v3455 = stablehlo.rsqrt %v3454 : tensor<32x301056xf32>
    %v3456 = stablehlo.multiply %v3449, %v3455 : tensor<32x301056xf32>
    %v3457 = stablehlo.multiply %v3409, %v3456 : tensor<32x301056xf32>
    %v3458 = stablehlo.reduce(%v3457 init: %v3443) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3459 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3460 = stablehlo.multiply %v3458, %v3459 : tensor<f32>
    %v3461 = stablehlo.subtract %d0ng, %v3460 : tensor<f32>
    %v3462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3463 = stablehlo.reduce(%v3409 init: %v3462) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3464 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3465 = stablehlo.multiply %v3463, %v3464 : tensor<f32>
    %v3466 = stablehlo.subtract %d0nbt, %v3465 : tensor<f32>
    %dd0Wxi = stablehlo.reshape %v175 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %dd0Wdi = stablehlo.reshape %v3325 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %dd0Wu = stablehlo.pad %dd0Wdi, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x55x55xf32>
    %dd0Wxt = stablehlo.transpose %dd0Wxi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %dd0Wdt = stablehlo.transpose %dd0Wu, dims = [1, 0, 2, 3] : (tensor<32x192x55x55xf32>) -> tensor<192x32x55x55xf32>
    %dd0Wraw = stablehlo.convolution(%dd0Wxt, %dd0Wdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<192x32x55x55xf32>) -> tensor<96x192x2x2xf32>
    %dd0W = stablehlo.transpose %dd0Wraw, dims = [1, 0, 2, 3] : (tensor<96x192x2x2xf32>) -> tensor<192x96x2x2xf32>
    %d0Wl = stablehlo.constant dense<0.1> : tensor<192x96x2x2xf32>
    %d0Ws = stablehlo.multiply %dd0W, %d0Wl : tensor<192x96x2x2xf32>
    %d0Wn = stablehlo.subtract %d0W, %d0Ws : tensor<192x96x2x2xf32>
    %v3467 = stablehlo.reshape %v3436 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3468 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3469 = stablehlo.multiply %v3467, %v3468 : tensor<32x96x56x56xf32>
    %v3470 = stablehlo.reshape %v3469 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3471 = stablehlo.reshape %v3470 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3472 = stablehlo.transpose %s0b2pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3473 = stablehlo.reverse %v3472, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3474 = stablehlo.convolution(%v3471, %v3473)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3475 = stablehlo.reshape %v3474 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3476 = stablehlo.multiply %v134, %v134 : tensor<32x1204224xf32>
    %v3477 = stablehlo.multiply %v3476, %v134 : tensor<32x1204224xf32>
    %v3478 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3479 = stablehlo.multiply %v3478, %v3477 : tensor<32x1204224xf32>
    %v3480 = stablehlo.add %v134, %v3479 : tensor<32x1204224xf32>
    %v3481 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3482 = stablehlo.multiply %v3481, %v3480 : tensor<32x1204224xf32>
    %v3483 = stablehlo.tanh %v3482 : tensor<32x1204224xf32>
    %v3484 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3485 = stablehlo.add %v3484, %v3483 : tensor<32x1204224xf32>
    %v3486 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3487 = stablehlo.multiply %v3486, %v3485 : tensor<32x1204224xf32>
    %v3488 = stablehlo.multiply %v3483, %v3483 : tensor<32x1204224xf32>
    %v3489 = stablehlo.subtract %v3484, %v3488 : tensor<32x1204224xf32>
    %v3490 = stablehlo.multiply %v3486, %v134 : tensor<32x1204224xf32>
    %v3491 = stablehlo.multiply %v3490, %v3489 : tensor<32x1204224xf32>
    %v3492 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3493 = stablehlo.multiply %v3492, %v3476 : tensor<32x1204224xf32>
    %v3494 = stablehlo.add %v3484, %v3493 : tensor<32x1204224xf32>
    %v3495 = stablehlo.multiply %v3481, %v3494 : tensor<32x1204224xf32>
    %v3496 = stablehlo.multiply %v3491, %v3495 : tensor<32x1204224xf32>
    %v3497 = stablehlo.add %v3487, %v3496 : tensor<32x1204224xf32>
    %v3498 = stablehlo.multiply %v3475, %v3497 : tensor<32x1204224xf32>
    %v3499 = stablehlo.reshape %v3498 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3500 = stablehlo.transpose %s0b2eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3501 = stablehlo.reverse %v3500, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3502 = stablehlo.convolution(%v3499, %v3501)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3503 = stablehlo.reshape %v3502 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3505 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3506 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3507 = stablehlo.reduce(%v111 init: %v3504) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3508 = stablehlo.broadcast_in_dim %v3507, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3509 = stablehlo.divide %v3508, %v3505 : tensor<32x301056xf32>
    %v3510 = stablehlo.subtract %v111, %v3509 : tensor<32x301056xf32>
    %v3511 = stablehlo.multiply %v3510, %v3510 : tensor<32x301056xf32>
    %v3512 = stablehlo.reduce(%v3511 init: %v3504) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3513 = stablehlo.broadcast_in_dim %v3512, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3514 = stablehlo.divide %v3513, %v3505 : tensor<32x301056xf32>
    %v3515 = stablehlo.add %v3514, %v3506 : tensor<32x301056xf32>
    %v3516 = stablehlo.rsqrt %v3515 : tensor<32x301056xf32>
    %v3517 = stablehlo.multiply %v3510, %v3516 : tensor<32x301056xf32>
    %v3518 = stablehlo.broadcast_in_dim %s0b2ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3519 = stablehlo.multiply %v3518, %v3503 : tensor<32x301056xf32>
    %v3520 = stablehlo.reduce(%v3519 init: %v3504) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3521 = stablehlo.broadcast_in_dim %v3520, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3522 = stablehlo.multiply %v3517, %v3519 : tensor<32x301056xf32>
    %v3523 = stablehlo.reduce(%v3522 init: %v3504) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3524 = stablehlo.broadcast_in_dim %v3523, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3525 = stablehlo.multiply %v3519, %v3505 : tensor<32x301056xf32>
    %v3526 = stablehlo.subtract %v3525, %v3521 : tensor<32x301056xf32>
    %v3527 = stablehlo.multiply %v3517, %v3524 : tensor<32x301056xf32>
    %v3528 = stablehlo.subtract %v3526, %v3527 : tensor<32x301056xf32>
    %v3529 = stablehlo.divide %v3516, %v3505 : tensor<32x301056xf32>
    %v3530 = stablehlo.multiply %v3529, %v3528 : tensor<32x301056xf32>
    %v3531 = stablehlo.reshape %v3530 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3532 = stablehlo.reverse %s0b2dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v3533 = stablehlo.convolution(%v3531, %v3532)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v3534 = stablehlo.reshape %v3533 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3535 = stablehlo.add %v3534, %v3436 : tensor<32x301056xf32>
    %v3536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3537 = stablehlo.reshape %v152 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3538 = stablehlo.reshape %v3436 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3539 = stablehlo.multiply %v3537, %v3538 : tensor<32x96x56x56xf32>
    %v3540 = stablehlo.reduce(%v3539 init: %v3536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3541 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3542 = stablehlo.multiply %v3540, %v3541 : tensor<96xf32>
    %v3543 = stablehlo.subtract %s0b2lg, %v3542 : tensor<96xf32>
    %v3544 = stablehlo.reshape %v147 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3545 = stablehlo.reshape %v3470 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3546 = stablehlo.transpose %v3544, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3547 = stablehlo.transpose %v3545, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3548 = stablehlo.convolution(%v3546, %v3547)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v3549 = stablehlo.transpose %v3548, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3550 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v3551 = stablehlo.multiply %v3549, %v3550 : tensor<96x384x1x1xf32>
    %v3552 = stablehlo.subtract %s0b2pW, %v3551 : tensor<96x384x1x1xf32>
    %v3553 = stablehlo.reshape %v3470 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3555 = stablehlo.reduce(%v3553 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3556 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3557 = stablehlo.multiply %v3555, %v3556 : tensor<96xf32>
    %v3558 = stablehlo.subtract %s0b2pb, %v3557 : tensor<96xf32>
    %v3559 = stablehlo.reshape %v129 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3560 = stablehlo.reshape %v3498 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3561 = stablehlo.transpose %v3559, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3562 = stablehlo.transpose %v3560, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3563 = stablehlo.convolution(%v3561, %v3562)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v3564 = stablehlo.transpose %v3563, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3565 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v3566 = stablehlo.multiply %v3564, %v3565 : tensor<384x96x1x1xf32>
    %v3567 = stablehlo.subtract %s0b2eW, %v3566 : tensor<384x96x1x1xf32>
    %v3568 = stablehlo.reshape %v3498 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3570 = stablehlo.reduce(%v3568 init: %v3569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v3571 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3572 = stablehlo.multiply %v3570, %v3571 : tensor<384xf32>
    %v3573 = stablehlo.subtract %s0b2eb, %v3572 : tensor<384xf32>
    %v3574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3575 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3576 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3577 = stablehlo.reduce(%v111 init: %v3574) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3578 = stablehlo.broadcast_in_dim %v3577, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3579 = stablehlo.divide %v3578, %v3575 : tensor<32x301056xf32>
    %v3580 = stablehlo.subtract %v111, %v3579 : tensor<32x301056xf32>
    %v3581 = stablehlo.multiply %v3580, %v3580 : tensor<32x301056xf32>
    %v3582 = stablehlo.reduce(%v3581 init: %v3574) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3583 = stablehlo.broadcast_in_dim %v3582, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3584 = stablehlo.divide %v3583, %v3575 : tensor<32x301056xf32>
    %v3585 = stablehlo.add %v3584, %v3576 : tensor<32x301056xf32>
    %v3586 = stablehlo.rsqrt %v3585 : tensor<32x301056xf32>
    %v3587 = stablehlo.multiply %v3580, %v3586 : tensor<32x301056xf32>
    %v3588 = stablehlo.multiply %v3503, %v3587 : tensor<32x301056xf32>
    %v3589 = stablehlo.reduce(%v3588 init: %v3574) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3590 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3591 = stablehlo.multiply %v3589, %v3590 : tensor<f32>
    %v3592 = stablehlo.subtract %s0b2ng, %v3591 : tensor<f32>
    %v3593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3594 = stablehlo.reduce(%v3503 init: %v3593) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3595 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3596 = stablehlo.multiply %v3594, %v3595 : tensor<f32>
    %v3597 = stablehlo.subtract %s0b2nbt, %v3596 : tensor<f32>
    %v3598 = stablehlo.reshape %v106 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3599 = stablehlo.reshape %v3530 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3600 = stablehlo.transpose %v3598, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3601 = stablehlo.transpose %v3599, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3602 = stablehlo.convolution(%v3600, %v3601)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v3603 = stablehlo.reshape %v3602 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v3604 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v3605 = stablehlo.multiply %v3603, %v3604 : tensor<96x1x7x7xf32>
    %v3606 = stablehlo.subtract %s0b2dW, %v3605 : tensor<96x1x7x7xf32>
    %v3607 = stablehlo.reshape %v3530 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3609 = stablehlo.reduce(%v3607 init: %v3608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3610 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3611 = stablehlo.multiply %v3609, %v3610 : tensor<96xf32>
    %v3612 = stablehlo.subtract %s0b2db, %v3611 : tensor<96xf32>
    %v3613 = stablehlo.reshape %v3535 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3614 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3615 = stablehlo.multiply %v3613, %v3614 : tensor<32x96x56x56xf32>
    %v3616 = stablehlo.reshape %v3615 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3617 = stablehlo.reshape %v3616 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3618 = stablehlo.transpose %s0b1pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3619 = stablehlo.reverse %v3618, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3620 = stablehlo.convolution(%v3617, %v3619)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3621 = stablehlo.reshape %v3620 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3622 = stablehlo.multiply %v83, %v83 : tensor<32x1204224xf32>
    %v3623 = stablehlo.multiply %v3622, %v83 : tensor<32x1204224xf32>
    %v3624 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3625 = stablehlo.multiply %v3624, %v3623 : tensor<32x1204224xf32>
    %v3626 = stablehlo.add %v83, %v3625 : tensor<32x1204224xf32>
    %v3627 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3628 = stablehlo.multiply %v3627, %v3626 : tensor<32x1204224xf32>
    %v3629 = stablehlo.tanh %v3628 : tensor<32x1204224xf32>
    %v3630 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3631 = stablehlo.add %v3630, %v3629 : tensor<32x1204224xf32>
    %v3632 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3633 = stablehlo.multiply %v3632, %v3631 : tensor<32x1204224xf32>
    %v3634 = stablehlo.multiply %v3629, %v3629 : tensor<32x1204224xf32>
    %v3635 = stablehlo.subtract %v3630, %v3634 : tensor<32x1204224xf32>
    %v3636 = stablehlo.multiply %v3632, %v83 : tensor<32x1204224xf32>
    %v3637 = stablehlo.multiply %v3636, %v3635 : tensor<32x1204224xf32>
    %v3638 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3639 = stablehlo.multiply %v3638, %v3622 : tensor<32x1204224xf32>
    %v3640 = stablehlo.add %v3630, %v3639 : tensor<32x1204224xf32>
    %v3641 = stablehlo.multiply %v3627, %v3640 : tensor<32x1204224xf32>
    %v3642 = stablehlo.multiply %v3637, %v3641 : tensor<32x1204224xf32>
    %v3643 = stablehlo.add %v3633, %v3642 : tensor<32x1204224xf32>
    %v3644 = stablehlo.multiply %v3621, %v3643 : tensor<32x1204224xf32>
    %v3645 = stablehlo.reshape %v3644 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3646 = stablehlo.transpose %s0b1eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3647 = stablehlo.reverse %v3646, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3648 = stablehlo.convolution(%v3645, %v3647)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3649 = stablehlo.reshape %v3648 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3651 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3652 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3653 = stablehlo.reduce(%v60 init: %v3650) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3654 = stablehlo.broadcast_in_dim %v3653, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3655 = stablehlo.divide %v3654, %v3651 : tensor<32x301056xf32>
    %v3656 = stablehlo.subtract %v60, %v3655 : tensor<32x301056xf32>
    %v3657 = stablehlo.multiply %v3656, %v3656 : tensor<32x301056xf32>
    %v3658 = stablehlo.reduce(%v3657 init: %v3650) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3659 = stablehlo.broadcast_in_dim %v3658, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3660 = stablehlo.divide %v3659, %v3651 : tensor<32x301056xf32>
    %v3661 = stablehlo.add %v3660, %v3652 : tensor<32x301056xf32>
    %v3662 = stablehlo.rsqrt %v3661 : tensor<32x301056xf32>
    %v3663 = stablehlo.multiply %v3656, %v3662 : tensor<32x301056xf32>
    %v3664 = stablehlo.broadcast_in_dim %s0b1ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3665 = stablehlo.multiply %v3664, %v3649 : tensor<32x301056xf32>
    %v3666 = stablehlo.reduce(%v3665 init: %v3650) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3667 = stablehlo.broadcast_in_dim %v3666, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3668 = stablehlo.multiply %v3663, %v3665 : tensor<32x301056xf32>
    %v3669 = stablehlo.reduce(%v3668 init: %v3650) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3670 = stablehlo.broadcast_in_dim %v3669, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3671 = stablehlo.multiply %v3665, %v3651 : tensor<32x301056xf32>
    %v3672 = stablehlo.subtract %v3671, %v3667 : tensor<32x301056xf32>
    %v3673 = stablehlo.multiply %v3663, %v3670 : tensor<32x301056xf32>
    %v3674 = stablehlo.subtract %v3672, %v3673 : tensor<32x301056xf32>
    %v3675 = stablehlo.divide %v3662, %v3651 : tensor<32x301056xf32>
    %v3676 = stablehlo.multiply %v3675, %v3674 : tensor<32x301056xf32>
    %v3677 = stablehlo.reshape %v3676 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3678 = stablehlo.reverse %s0b1dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v3679 = stablehlo.convolution(%v3677, %v3678)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v3680 = stablehlo.reshape %v3679 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3681 = stablehlo.add %v3680, %v3535 : tensor<32x301056xf32>
    %v3682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3683 = stablehlo.reshape %v101 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3684 = stablehlo.reshape %v3535 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3685 = stablehlo.multiply %v3683, %v3684 : tensor<32x96x56x56xf32>
    %v3686 = stablehlo.reduce(%v3685 init: %v3682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3687 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3688 = stablehlo.multiply %v3686, %v3687 : tensor<96xf32>
    %v3689 = stablehlo.subtract %s0b1lg, %v3688 : tensor<96xf32>
    %v3690 = stablehlo.reshape %v96 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3691 = stablehlo.reshape %v3616 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3692 = stablehlo.transpose %v3690, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3693 = stablehlo.transpose %v3691, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3694 = stablehlo.convolution(%v3692, %v3693)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v3695 = stablehlo.transpose %v3694, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3696 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v3697 = stablehlo.multiply %v3695, %v3696 : tensor<96x384x1x1xf32>
    %v3698 = stablehlo.subtract %s0b1pW, %v3697 : tensor<96x384x1x1xf32>
    %v3699 = stablehlo.reshape %v3616 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3700 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3701 = stablehlo.reduce(%v3699 init: %v3700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3702 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3703 = stablehlo.multiply %v3701, %v3702 : tensor<96xf32>
    %v3704 = stablehlo.subtract %s0b1pb, %v3703 : tensor<96xf32>
    %v3705 = stablehlo.reshape %v78 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3706 = stablehlo.reshape %v3644 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3707 = stablehlo.transpose %v3705, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3708 = stablehlo.transpose %v3706, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3709 = stablehlo.convolution(%v3707, %v3708)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v3710 = stablehlo.transpose %v3709, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3711 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v3712 = stablehlo.multiply %v3710, %v3711 : tensor<384x96x1x1xf32>
    %v3713 = stablehlo.subtract %s0b1eW, %v3712 : tensor<384x96x1x1xf32>
    %v3714 = stablehlo.reshape %v3644 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3716 = stablehlo.reduce(%v3714 init: %v3715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v3717 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3718 = stablehlo.multiply %v3716, %v3717 : tensor<384xf32>
    %v3719 = stablehlo.subtract %s0b1eb, %v3718 : tensor<384xf32>
    %v3720 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3721 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3722 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3723 = stablehlo.reduce(%v60 init: %v3720) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3724 = stablehlo.broadcast_in_dim %v3723, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3725 = stablehlo.divide %v3724, %v3721 : tensor<32x301056xf32>
    %v3726 = stablehlo.subtract %v60, %v3725 : tensor<32x301056xf32>
    %v3727 = stablehlo.multiply %v3726, %v3726 : tensor<32x301056xf32>
    %v3728 = stablehlo.reduce(%v3727 init: %v3720) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3729 = stablehlo.broadcast_in_dim %v3728, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3730 = stablehlo.divide %v3729, %v3721 : tensor<32x301056xf32>
    %v3731 = stablehlo.add %v3730, %v3722 : tensor<32x301056xf32>
    %v3732 = stablehlo.rsqrt %v3731 : tensor<32x301056xf32>
    %v3733 = stablehlo.multiply %v3726, %v3732 : tensor<32x301056xf32>
    %v3734 = stablehlo.multiply %v3649, %v3733 : tensor<32x301056xf32>
    %v3735 = stablehlo.reduce(%v3734 init: %v3720) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3736 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3737 = stablehlo.multiply %v3735, %v3736 : tensor<f32>
    %v3738 = stablehlo.subtract %s0b1ng, %v3737 : tensor<f32>
    %v3739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3740 = stablehlo.reduce(%v3649 init: %v3739) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3741 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3742 = stablehlo.multiply %v3740, %v3741 : tensor<f32>
    %v3743 = stablehlo.subtract %s0b1nbt, %v3742 : tensor<f32>
    %v3744 = stablehlo.reshape %v55 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3745 = stablehlo.reshape %v3676 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3746 = stablehlo.transpose %v3744, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3747 = stablehlo.transpose %v3745, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3748 = stablehlo.convolution(%v3746, %v3747)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v3749 = stablehlo.reshape %v3748 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v3750 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v3751 = stablehlo.multiply %v3749, %v3750 : tensor<96x1x7x7xf32>
    %v3752 = stablehlo.subtract %s0b1dW, %v3751 : tensor<96x1x7x7xf32>
    %v3753 = stablehlo.reshape %v3676 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3754 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3755 = stablehlo.reduce(%v3753 init: %v3754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3756 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3757 = stablehlo.multiply %v3755, %v3756 : tensor<96xf32>
    %v3758 = stablehlo.subtract %s0b1db, %v3757 : tensor<96xf32>
    %v3759 = stablehlo.reshape %v3681 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3760 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3761 = stablehlo.multiply %v3759, %v3760 : tensor<32x96x56x56xf32>
    %v3762 = stablehlo.reshape %v3761 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3763 = stablehlo.reshape %v3762 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3764 = stablehlo.transpose %s0b0pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3765 = stablehlo.reverse %v3764, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3766 = stablehlo.convolution(%v3763, %v3765)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3767 = stablehlo.reshape %v3766 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3768 = stablehlo.multiply %v32, %v32 : tensor<32x1204224xf32>
    %v3769 = stablehlo.multiply %v3768, %v32 : tensor<32x1204224xf32>
    %v3770 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3771 = stablehlo.multiply %v3770, %v3769 : tensor<32x1204224xf32>
    %v3772 = stablehlo.add %v32, %v3771 : tensor<32x1204224xf32>
    %v3773 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3774 = stablehlo.multiply %v3773, %v3772 : tensor<32x1204224xf32>
    %v3775 = stablehlo.tanh %v3774 : tensor<32x1204224xf32>
    %v3776 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3777 = stablehlo.add %v3776, %v3775 : tensor<32x1204224xf32>
    %v3778 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3779 = stablehlo.multiply %v3778, %v3777 : tensor<32x1204224xf32>
    %v3780 = stablehlo.multiply %v3775, %v3775 : tensor<32x1204224xf32>
    %v3781 = stablehlo.subtract %v3776, %v3780 : tensor<32x1204224xf32>
    %v3782 = stablehlo.multiply %v3778, %v32 : tensor<32x1204224xf32>
    %v3783 = stablehlo.multiply %v3782, %v3781 : tensor<32x1204224xf32>
    %v3784 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3785 = stablehlo.multiply %v3784, %v3768 : tensor<32x1204224xf32>
    %v3786 = stablehlo.add %v3776, %v3785 : tensor<32x1204224xf32>
    %v3787 = stablehlo.multiply %v3773, %v3786 : tensor<32x1204224xf32>
    %v3788 = stablehlo.multiply %v3783, %v3787 : tensor<32x1204224xf32>
    %v3789 = stablehlo.add %v3779, %v3788 : tensor<32x1204224xf32>
    %v3790 = stablehlo.multiply %v3767, %v3789 : tensor<32x1204224xf32>
    %v3791 = stablehlo.reshape %v3790 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3792 = stablehlo.transpose %s0b0eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3793 = stablehlo.reverse %v3792, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3794 = stablehlo.convolution(%v3791, %v3793)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3795 = stablehlo.reshape %v3794 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3797 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3798 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3799 = stablehlo.reduce(%v9 init: %v3796) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3800 = stablehlo.broadcast_in_dim %v3799, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3801 = stablehlo.divide %v3800, %v3797 : tensor<32x301056xf32>
    %v3802 = stablehlo.subtract %v9, %v3801 : tensor<32x301056xf32>
    %v3803 = stablehlo.multiply %v3802, %v3802 : tensor<32x301056xf32>
    %v3804 = stablehlo.reduce(%v3803 init: %v3796) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3805 = stablehlo.broadcast_in_dim %v3804, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3806 = stablehlo.divide %v3805, %v3797 : tensor<32x301056xf32>
    %v3807 = stablehlo.add %v3806, %v3798 : tensor<32x301056xf32>
    %v3808 = stablehlo.rsqrt %v3807 : tensor<32x301056xf32>
    %v3809 = stablehlo.multiply %v3802, %v3808 : tensor<32x301056xf32>
    %v3810 = stablehlo.broadcast_in_dim %s0b0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3811 = stablehlo.multiply %v3810, %v3795 : tensor<32x301056xf32>
    %v3812 = stablehlo.reduce(%v3811 init: %v3796) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3813 = stablehlo.broadcast_in_dim %v3812, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3814 = stablehlo.multiply %v3809, %v3811 : tensor<32x301056xf32>
    %v3815 = stablehlo.reduce(%v3814 init: %v3796) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3816 = stablehlo.broadcast_in_dim %v3815, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3817 = stablehlo.multiply %v3811, %v3797 : tensor<32x301056xf32>
    %v3818 = stablehlo.subtract %v3817, %v3813 : tensor<32x301056xf32>
    %v3819 = stablehlo.multiply %v3809, %v3816 : tensor<32x301056xf32>
    %v3820 = stablehlo.subtract %v3818, %v3819 : tensor<32x301056xf32>
    %v3821 = stablehlo.divide %v3808, %v3797 : tensor<32x301056xf32>
    %v3822 = stablehlo.multiply %v3821, %v3820 : tensor<32x301056xf32>
    %v3823 = stablehlo.reshape %v3822 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3824 = stablehlo.reverse %s0b0dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v3825 = stablehlo.convolution(%v3823, %v3824)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v3826 = stablehlo.reshape %v3825 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3827 = stablehlo.add %v3826, %v3681 : tensor<32x301056xf32>
    %v3828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3829 = stablehlo.reshape %v50 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3830 = stablehlo.reshape %v3681 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3831 = stablehlo.multiply %v3829, %v3830 : tensor<32x96x56x56xf32>
    %v3832 = stablehlo.reduce(%v3831 init: %v3828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3833 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3834 = stablehlo.multiply %v3832, %v3833 : tensor<96xf32>
    %v3835 = stablehlo.subtract %s0b0lg, %v3834 : tensor<96xf32>
    %v3836 = stablehlo.reshape %v45 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3837 = stablehlo.reshape %v3762 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3838 = stablehlo.transpose %v3836, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3839 = stablehlo.transpose %v3837, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3840 = stablehlo.convolution(%v3838, %v3839)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v3841 = stablehlo.transpose %v3840, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3842 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v3843 = stablehlo.multiply %v3841, %v3842 : tensor<96x384x1x1xf32>
    %v3844 = stablehlo.subtract %s0b0pW, %v3843 : tensor<96x384x1x1xf32>
    %v3845 = stablehlo.reshape %v3762 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3847 = stablehlo.reduce(%v3845 init: %v3846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3848 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3849 = stablehlo.multiply %v3847, %v3848 : tensor<96xf32>
    %v3850 = stablehlo.subtract %s0b0pb, %v3849 : tensor<96xf32>
    %v3851 = stablehlo.reshape %v27 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3852 = stablehlo.reshape %v3790 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3853 = stablehlo.transpose %v3851, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3854 = stablehlo.transpose %v3852, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3855 = stablehlo.convolution(%v3853, %v3854)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v3856 = stablehlo.transpose %v3855, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3857 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v3858 = stablehlo.multiply %v3856, %v3857 : tensor<384x96x1x1xf32>
    %v3859 = stablehlo.subtract %s0b0eW, %v3858 : tensor<384x96x1x1xf32>
    %v3860 = stablehlo.reshape %v3790 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3862 = stablehlo.reduce(%v3860 init: %v3861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v3863 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3864 = stablehlo.multiply %v3862, %v3863 : tensor<384xf32>
    %v3865 = stablehlo.subtract %s0b0eb, %v3864 : tensor<384xf32>
    %v3866 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3867 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3868 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3869 = stablehlo.reduce(%v9 init: %v3866) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3870 = stablehlo.broadcast_in_dim %v3869, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3871 = stablehlo.divide %v3870, %v3867 : tensor<32x301056xf32>
    %v3872 = stablehlo.subtract %v9, %v3871 : tensor<32x301056xf32>
    %v3873 = stablehlo.multiply %v3872, %v3872 : tensor<32x301056xf32>
    %v3874 = stablehlo.reduce(%v3873 init: %v3866) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3875 = stablehlo.broadcast_in_dim %v3874, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3876 = stablehlo.divide %v3875, %v3867 : tensor<32x301056xf32>
    %v3877 = stablehlo.add %v3876, %v3868 : tensor<32x301056xf32>
    %v3878 = stablehlo.rsqrt %v3877 : tensor<32x301056xf32>
    %v3879 = stablehlo.multiply %v3872, %v3878 : tensor<32x301056xf32>
    %v3880 = stablehlo.multiply %v3795, %v3879 : tensor<32x301056xf32>
    %v3881 = stablehlo.reduce(%v3880 init: %v3866) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3882 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3883 = stablehlo.multiply %v3881, %v3882 : tensor<f32>
    %v3884 = stablehlo.subtract %s0b0ng, %v3883 : tensor<f32>
    %v3885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3886 = stablehlo.reduce(%v3795 init: %v3885) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3887 = stablehlo.constant dense<0.1> : tensor<f32>
    %v3888 = stablehlo.multiply %v3886, %v3887 : tensor<f32>
    %v3889 = stablehlo.subtract %s0b0nbt, %v3888 : tensor<f32>
    %v3890 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3891 = stablehlo.reshape %v3822 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3892 = stablehlo.transpose %v3890, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3893 = stablehlo.transpose %v3891, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3894 = stablehlo.convolution(%v3892, %v3893)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v3895 = stablehlo.reshape %v3894 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v3896 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v3897 = stablehlo.multiply %v3895, %v3896 : tensor<96x1x7x7xf32>
    %v3898 = stablehlo.subtract %s0b0dW, %v3897 : tensor<96x1x7x7xf32>
    %v3899 = stablehlo.reshape %v3822 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3901 = stablehlo.reduce(%v3899 init: %v3900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3902 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3903 = stablehlo.multiply %v3901, %v3902 : tensor<96xf32>
    %v3904 = stablehlo.subtract %s0b0db, %v3903 : tensor<96xf32>
    %dpsWxi = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %dpsWdi = stablehlo.reshape %v3827 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %dpsWu = stablehlo.pad %dpsWdi, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x221x221xf32>
    %dpsWxt = stablehlo.transpose %dpsWxi, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %dpsWdt = stablehlo.transpose %dpsWu, dims = [1, 0, 2, 3] : (tensor<32x96x221x221xf32>) -> tensor<96x32x221x221xf32>
    %dpsWraw = stablehlo.convolution(%dpsWxt, %dpsWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<96x32x221x221xf32>) -> tensor<3x96x4x4xf32>
    %dpsW = stablehlo.transpose %dpsWraw, dims = [1, 0, 2, 3] : (tensor<3x96x4x4xf32>) -> tensor<96x3x4x4xf32>
    %psWl = stablehlo.constant dense<0.1> : tensor<96x3x4x4xf32>
    %psWs = stablehlo.multiply %dpsW, %psWl : tensor<96x3x4x4xf32>
    %psWn = stablehlo.subtract %psW, %psWs : tensor<96x3x4x4xf32>
    %v3905 = stablehlo.reshape %v3827 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3907 = stablehlo.reduce(%v3905 init: %v3906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3908 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v3909 = stablehlo.multiply %v3907, %v3908 : tensor<96xf32>
    %v3910 = stablehlo.subtract %psb, %v3909 : tensor<96xf32>
    return %psWn, %v3910, %v3898, %v3904, %v3884, %v3889, %v3859, %v3865, %v3844, %v3850, %v3835, %v3752, %v3758, %v3738, %v3743, %v3713, %v3719, %v3698, %v3704, %v3689, %v3606, %v3612, %v3592, %v3597, %v3567, %v3573, %v3552, %v3558, %v3543, %v3461, %v3466, %d0Wn, %v3442, %v3396, %v3402, %v3382, %v3387, %v3357, %v3363, %v3342, %v3348, %v3333, %v3250, %v3256, %v3236, %v3241, %v3211, %v3217, %v3196, %v3202, %v3187, %v3104, %v3110, %v3090, %v3095, %v3065, %v3071, %v3050, %v3056, %v3041, %v2959, %v2964, %d1Wn, %v2940, %v2894, %v2900, %v2880, %v2885, %v2855, %v2861, %v2840, %v2846, %v2831, %v2748, %v2754, %v2734, %v2739, %v2709, %v2715, %v2694, %v2700, %v2685, %v2602, %v2608, %v2588, %v2593, %v2563, %v2569, %v2548, %v2554, %v2539, %v2456, %v2462, %v2442, %v2447, %v2417, %v2423, %v2402, %v2408, %v2393, %v2310, %v2316, %v2296, %v2301, %v2271, %v2277, %v2256, %v2262, %v2247, %v2164, %v2170, %v2150, %v2155, %v2125, %v2131, %v2110, %v2116, %v2101, %v2018, %v2024, %v2004, %v2009, %v1979, %v1985, %v1964, %v1970, %v1955, %v1872, %v1878, %v1858, %v1863, %v1833, %v1839, %v1818, %v1824, %v1809, %v1726, %v1732, %v1712, %v1717, %v1687, %v1693, %v1672, %v1678, %v1663, %v1581, %v1586, %d2Wn, %v1562, %v1516, %v1522, %v1502, %v1507, %v1477, %v1483, %v1462, %v1468, %v1453, %v1370, %v1376, %v1356, %v1361, %v1331, %v1337, %v1316, %v1322, %v1307, %v1224, %v1230, %v1210, %v1215, %v1185, %v1191, %v1170, %v1176, %v1161, %v1079, %v1084, %v1055, %v1060 : tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>
  }
}
