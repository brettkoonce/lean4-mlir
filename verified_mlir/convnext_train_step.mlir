module @m {
  func.func @convnext_train_step(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %psng: tensor<96xf32>, %psnbt: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<96xf32>, %s0b0nbt: tensor<96xf32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<96xf32>, %s0b1nbt: tensor<96xf32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<96xf32>, %s0b2nbt: tensor<96xf32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<96xf32>, %d0nbt: tensor<96xf32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<192xf32>, %s1b0nbt: tensor<192xf32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<192xf32>, %s1b1nbt: tensor<192xf32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<192xf32>, %s1b2nbt: tensor<192xf32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<192xf32>, %d1nbt: tensor<192xf32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<384xf32>, %s2b0nbt: tensor<384xf32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<384xf32>, %s2b1nbt: tensor<384xf32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<384xf32>, %s2b2nbt: tensor<384xf32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<384xf32>, %s2b3nbt: tensor<384xf32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<384xf32>, %s2b4nbt: tensor<384xf32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<384xf32>, %s2b5nbt: tensor<384xf32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<384xf32>, %s2b6nbt: tensor<384xf32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<384xf32>, %s2b7nbt: tensor<384xf32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<384xf32>, %s2b8nbt: tensor<384xf32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<384xf32>, %d2nbt: tensor<384xf32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<768xf32>, %s3b0nbt: tensor<768xf32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<768xf32>, %s3b1nbt: tensor<768xf32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<768xf32>, %s3b2nbt: tensor<768xf32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<768xf32>, %hnbt: tensor<768xf32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x10xf32>, tensor<10xf32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %bsc = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    // §2m: the channel-LN chain normalises with lnRowF at γ=1/β=0 and applies the REAL
    // per-channel affine with rowScaleF/rowBiasF, so these two are its scalar identities.
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %psW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<96x3x4x4xf32>) -> tensor<32x96x56x56xf32>
    %v2 = stablehlo.broadcast_in_dim %psb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x96x56x56xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v6 = stablehlo.transpose %v5, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v9 = stablehlo.constant dense<0.0> : tensor<f32>
    %v10 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v11 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v12 = stablehlo.reduce(%v8 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v13 = stablehlo.broadcast_in_dim %v12, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v14 = stablehlo.divide %v13, %v10 : tensor<32x3136x96xf32>
    %v15 = stablehlo.subtract %v8, %v14 : tensor<32x3136x96xf32>
    %v16 = stablehlo.multiply %v15, %v15 : tensor<32x3136x96xf32>
    %v17 = stablehlo.reduce(%v16 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v18 = stablehlo.broadcast_in_dim %v17, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v19 = stablehlo.divide %v18, %v10 : tensor<32x3136x96xf32>
    %v20 = stablehlo.add %v19, %v11 : tensor<32x3136x96xf32>
    %v21 = stablehlo.rsqrt %v20 : tensor<32x3136x96xf32>
    %v22 = stablehlo.multiply %v15, %v21 : tensor<32x3136x96xf32>
    %v23 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v24 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v25 = stablehlo.multiply %v22, %v23 : tensor<32x3136x96xf32>
    %v26 = stablehlo.add %v25, %v24 : tensor<32x3136x96xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v29 = stablehlo.broadcast_in_dim %psng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v30 = stablehlo.multiply %v28, %v29 : tensor<32x3136x96xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v33 = stablehlo.broadcast_in_dim %psnbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<32x3136x96xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v37 = stablehlo.transpose %v36, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v40 = stablehlo.convolution(%v39, %s0b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v41 = stablehlo.broadcast_in_dim %s0b0db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<32x96x56x56xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v45 = stablehlo.transpose %v44, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v46 = stablehlo.reshape %v45 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v48 = stablehlo.constant dense<0.0> : tensor<f32>
    %v49 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v50 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v51 = stablehlo.reduce(%v47 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v52 = stablehlo.broadcast_in_dim %v51, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v53 = stablehlo.divide %v52, %v49 : tensor<32x3136x96xf32>
    %v54 = stablehlo.subtract %v47, %v53 : tensor<32x3136x96xf32>
    %v55 = stablehlo.multiply %v54, %v54 : tensor<32x3136x96xf32>
    %v56 = stablehlo.reduce(%v55 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v57 = stablehlo.broadcast_in_dim %v56, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v58 = stablehlo.divide %v57, %v49 : tensor<32x3136x96xf32>
    %v59 = stablehlo.add %v58, %v50 : tensor<32x3136x96xf32>
    %v60 = stablehlo.rsqrt %v59 : tensor<32x3136x96xf32>
    %v61 = stablehlo.multiply %v54, %v60 : tensor<32x3136x96xf32>
    %v62 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v63 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v64 = stablehlo.multiply %v61, %v62 : tensor<32x3136x96xf32>
    %v65 = stablehlo.add %v64, %v63 : tensor<32x3136x96xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v68 = stablehlo.broadcast_in_dim %s0b0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v69 = stablehlo.multiply %v67, %v68 : tensor<32x3136x96xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v72 = stablehlo.broadcast_in_dim %s0b0nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v73 = stablehlo.add %v71, %v72 : tensor<32x3136x96xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v76 = stablehlo.transpose %v75, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v77 = stablehlo.reshape %v76 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v79 = stablehlo.convolution(%v78, %s0b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v80 = stablehlo.broadcast_in_dim %s0b0eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v81 = stablehlo.add %v79, %v80 : tensor<32x384x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v84 = stablehlo.multiply %v83, %v83 : tensor<32x384x56x56xf32>
    %v85 = stablehlo.multiply %v84, %v83 : tensor<32x384x56x56xf32>
    %v86 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v87 = stablehlo.multiply %v86, %v85 : tensor<32x384x56x56xf32>
    %v88 = stablehlo.add %v83, %v87 : tensor<32x384x56x56xf32>
    %v89 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v90 = stablehlo.multiply %v89, %v88 : tensor<32x384x56x56xf32>
    %v91 = stablehlo.tanh %v90 : tensor<32x384x56x56xf32>
    %v92 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v93 = stablehlo.add %v92, %v91 : tensor<32x384x56x56xf32>
    %v94 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v95 = stablehlo.multiply %v94, %v83 : tensor<32x384x56x56xf32>
    %v96 = stablehlo.multiply %v95, %v93 : tensor<32x384x56x56xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v98 = stablehlo.reshape %v97 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v99 = stablehlo.convolution(%v98, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v100 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v101 = stablehlo.add %v99, %v100 : tensor<32x96x56x56xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v104 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v105 = stablehlo.multiply %v103, %v104 : tensor<32x96x56x56xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v108 = stablehlo.reshape %v38 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v109 = stablehlo.add %v107, %v108 : tensor<32x96x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v112 = stablehlo.convolution(%v111, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v113 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v114 = stablehlo.add %v112, %v113 : tensor<32x96x56x56xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v117 = stablehlo.transpose %v116, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v121 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v122 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v123 = stablehlo.reduce(%v119 init: %v120) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v124 = stablehlo.broadcast_in_dim %v123, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v125 = stablehlo.divide %v124, %v121 : tensor<32x3136x96xf32>
    %v126 = stablehlo.subtract %v119, %v125 : tensor<32x3136x96xf32>
    %v127 = stablehlo.multiply %v126, %v126 : tensor<32x3136x96xf32>
    %v128 = stablehlo.reduce(%v127 init: %v120) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v129 = stablehlo.broadcast_in_dim %v128, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v130 = stablehlo.divide %v129, %v121 : tensor<32x3136x96xf32>
    %v131 = stablehlo.add %v130, %v122 : tensor<32x3136x96xf32>
    %v132 = stablehlo.rsqrt %v131 : tensor<32x3136x96xf32>
    %v133 = stablehlo.multiply %v126, %v132 : tensor<32x3136x96xf32>
    %v134 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v135 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v136 = stablehlo.multiply %v133, %v134 : tensor<32x3136x96xf32>
    %v137 = stablehlo.add %v136, %v135 : tensor<32x3136x96xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v140 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v141 = stablehlo.multiply %v139, %v140 : tensor<32x3136x96xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v144 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v145 = stablehlo.add %v143, %v144 : tensor<32x3136x96xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v148 = stablehlo.transpose %v147, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v151 = stablehlo.convolution(%v150, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v152 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v153 = stablehlo.add %v151, %v152 : tensor<32x384x56x56xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v155 = stablehlo.reshape %v154 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v156 = stablehlo.multiply %v155, %v155 : tensor<32x384x56x56xf32>
    %v157 = stablehlo.multiply %v156, %v155 : tensor<32x384x56x56xf32>
    %v158 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v159 = stablehlo.multiply %v158, %v157 : tensor<32x384x56x56xf32>
    %v160 = stablehlo.add %v155, %v159 : tensor<32x384x56x56xf32>
    %v161 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v162 = stablehlo.multiply %v161, %v160 : tensor<32x384x56x56xf32>
    %v163 = stablehlo.tanh %v162 : tensor<32x384x56x56xf32>
    %v164 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v165 = stablehlo.add %v164, %v163 : tensor<32x384x56x56xf32>
    %v166 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v167 = stablehlo.multiply %v166, %v155 : tensor<32x384x56x56xf32>
    %v168 = stablehlo.multiply %v167, %v165 : tensor<32x384x56x56xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v171 = stablehlo.convolution(%v170, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v173 = stablehlo.add %v171, %v172 : tensor<32x96x56x56xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v176 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v177 = stablehlo.multiply %v175, %v176 : tensor<32x96x56x56xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v180 = stablehlo.reshape %v110 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v181 = stablehlo.add %v179, %v180 : tensor<32x96x56x56xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v184 = stablehlo.convolution(%v183, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v185 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v186 = stablehlo.add %v184, %v185 : tensor<32x96x56x56xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v189 = stablehlo.transpose %v188, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v194 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v195 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v197 = stablehlo.divide %v196, %v193 : tensor<32x3136x96xf32>
    %v198 = stablehlo.subtract %v191, %v197 : tensor<32x3136x96xf32>
    %v199 = stablehlo.multiply %v198, %v198 : tensor<32x3136x96xf32>
    %v200 = stablehlo.reduce(%v199 init: %v192) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v202 = stablehlo.divide %v201, %v193 : tensor<32x3136x96xf32>
    %v203 = stablehlo.add %v202, %v194 : tensor<32x3136x96xf32>
    %v204 = stablehlo.rsqrt %v203 : tensor<32x3136x96xf32>
    %v205 = stablehlo.multiply %v198, %v204 : tensor<32x3136x96xf32>
    %v206 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v207 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v208 = stablehlo.multiply %v205, %v206 : tensor<32x3136x96xf32>
    %v209 = stablehlo.add %v208, %v207 : tensor<32x3136x96xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v212 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v213 = stablehlo.multiply %v211, %v212 : tensor<32x3136x96xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v216 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v217 = stablehlo.add %v215, %v216 : tensor<32x3136x96xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v220 = stablehlo.transpose %v219, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v223 = stablehlo.convolution(%v222, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v224 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v225 = stablehlo.add %v223, %v224 : tensor<32x384x56x56xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v228 = stablehlo.multiply %v227, %v227 : tensor<32x384x56x56xf32>
    %v229 = stablehlo.multiply %v228, %v227 : tensor<32x384x56x56xf32>
    %v230 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v231 = stablehlo.multiply %v230, %v229 : tensor<32x384x56x56xf32>
    %v232 = stablehlo.add %v227, %v231 : tensor<32x384x56x56xf32>
    %v233 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v234 = stablehlo.multiply %v233, %v232 : tensor<32x384x56x56xf32>
    %v235 = stablehlo.tanh %v234 : tensor<32x384x56x56xf32>
    %v236 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v237 = stablehlo.add %v236, %v235 : tensor<32x384x56x56xf32>
    %v238 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v239 = stablehlo.multiply %v238, %v227 : tensor<32x384x56x56xf32>
    %v240 = stablehlo.multiply %v239, %v237 : tensor<32x384x56x56xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v243 = stablehlo.convolution(%v242, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v244 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v245 = stablehlo.add %v243, %v244 : tensor<32x96x56x56xf32>
    %v246 = stablehlo.reshape %v245 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v248 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v249 = stablehlo.multiply %v247, %v248 : tensor<32x96x56x56xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v252 = stablehlo.reshape %v182 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v253 = stablehlo.add %v251, %v252 : tensor<32x96x56x56xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v256 = stablehlo.transpose %v255, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v257 = stablehlo.reshape %v256 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v260 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v261 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v262 = stablehlo.reduce(%v258 init: %v259) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v263 = stablehlo.broadcast_in_dim %v262, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v264 = stablehlo.divide %v263, %v260 : tensor<32x3136x96xf32>
    %v265 = stablehlo.subtract %v258, %v264 : tensor<32x3136x96xf32>
    %v266 = stablehlo.multiply %v265, %v265 : tensor<32x3136x96xf32>
    %v267 = stablehlo.reduce(%v266 init: %v259) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v268 = stablehlo.broadcast_in_dim %v267, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v269 = stablehlo.divide %v268, %v260 : tensor<32x3136x96xf32>
    %v270 = stablehlo.add %v269, %v261 : tensor<32x3136x96xf32>
    %v271 = stablehlo.rsqrt %v270 : tensor<32x3136x96xf32>
    %v272 = stablehlo.multiply %v265, %v271 : tensor<32x3136x96xf32>
    %v273 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v274 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v275 = stablehlo.multiply %v272, %v273 : tensor<32x3136x96xf32>
    %v276 = stablehlo.add %v275, %v274 : tensor<32x3136x96xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v279 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v280 = stablehlo.multiply %v278, %v279 : tensor<32x3136x96xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v283 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v284 = stablehlo.add %v282, %v283 : tensor<32x3136x96xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v287 = stablehlo.transpose %v286, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v290 = stablehlo.convolution(%v289, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v291 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v292 = stablehlo.add %v290, %v291 : tensor<32x192x28x28xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v294 = stablehlo.reshape %v293 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v295 = stablehlo.convolution(%v294, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v296 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v297 = stablehlo.add %v295, %v296 : tensor<32x192x28x28xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v299 = stablehlo.reshape %v298 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v300 = stablehlo.transpose %v299, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v304 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v305 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v306 = stablehlo.reduce(%v302 init: %v303) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v307 = stablehlo.broadcast_in_dim %v306, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v308 = stablehlo.divide %v307, %v304 : tensor<32x784x192xf32>
    %v309 = stablehlo.subtract %v302, %v308 : tensor<32x784x192xf32>
    %v310 = stablehlo.multiply %v309, %v309 : tensor<32x784x192xf32>
    %v311 = stablehlo.reduce(%v310 init: %v303) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v312 = stablehlo.broadcast_in_dim %v311, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v313 = stablehlo.divide %v312, %v304 : tensor<32x784x192xf32>
    %v314 = stablehlo.add %v313, %v305 : tensor<32x784x192xf32>
    %v315 = stablehlo.rsqrt %v314 : tensor<32x784x192xf32>
    %v316 = stablehlo.multiply %v309, %v315 : tensor<32x784x192xf32>
    %v317 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v318 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v319 = stablehlo.multiply %v316, %v317 : tensor<32x784x192xf32>
    %v320 = stablehlo.add %v319, %v318 : tensor<32x784x192xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v323 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v324 = stablehlo.multiply %v322, %v323 : tensor<32x784x192xf32>
    %v325 = stablehlo.reshape %v324 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v327 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v328 = stablehlo.add %v326, %v327 : tensor<32x784x192xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v331 = stablehlo.transpose %v330, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v334 = stablehlo.convolution(%v333, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v335 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v336 = stablehlo.add %v334, %v335 : tensor<32x768x28x28xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v339 = stablehlo.multiply %v338, %v338 : tensor<32x768x28x28xf32>
    %v340 = stablehlo.multiply %v339, %v338 : tensor<32x768x28x28xf32>
    %v341 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v342 = stablehlo.multiply %v341, %v340 : tensor<32x768x28x28xf32>
    %v343 = stablehlo.add %v338, %v342 : tensor<32x768x28x28xf32>
    %v344 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v345 = stablehlo.multiply %v344, %v343 : tensor<32x768x28x28xf32>
    %v346 = stablehlo.tanh %v345 : tensor<32x768x28x28xf32>
    %v347 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v348 = stablehlo.add %v347, %v346 : tensor<32x768x28x28xf32>
    %v349 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v350 = stablehlo.multiply %v349, %v338 : tensor<32x768x28x28xf32>
    %v351 = stablehlo.multiply %v350, %v348 : tensor<32x768x28x28xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v354 = stablehlo.convolution(%v353, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v355 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v356 = stablehlo.add %v354, %v355 : tensor<32x192x28x28xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v359 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v360 = stablehlo.multiply %v358, %v359 : tensor<32x192x28x28xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v363 = stablehlo.reshape %v293 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v364 = stablehlo.add %v362, %v363 : tensor<32x192x28x28xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v366 = stablehlo.reshape %v365 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v367 = stablehlo.convolution(%v366, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v368 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v369 = stablehlo.add %v367, %v368 : tensor<32x192x28x28xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v371 = stablehlo.reshape %v370 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v372 = stablehlo.transpose %v371, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v376 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v377 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v378 = stablehlo.reduce(%v374 init: %v375) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v379 = stablehlo.broadcast_in_dim %v378, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v380 = stablehlo.divide %v379, %v376 : tensor<32x784x192xf32>
    %v381 = stablehlo.subtract %v374, %v380 : tensor<32x784x192xf32>
    %v382 = stablehlo.multiply %v381, %v381 : tensor<32x784x192xf32>
    %v383 = stablehlo.reduce(%v382 init: %v375) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v384 = stablehlo.broadcast_in_dim %v383, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v385 = stablehlo.divide %v384, %v376 : tensor<32x784x192xf32>
    %v386 = stablehlo.add %v385, %v377 : tensor<32x784x192xf32>
    %v387 = stablehlo.rsqrt %v386 : tensor<32x784x192xf32>
    %v388 = stablehlo.multiply %v381, %v387 : tensor<32x784x192xf32>
    %v389 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v390 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v391 = stablehlo.multiply %v388, %v389 : tensor<32x784x192xf32>
    %v392 = stablehlo.add %v391, %v390 : tensor<32x784x192xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v395 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v396 = stablehlo.multiply %v394, %v395 : tensor<32x784x192xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v399 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<32x784x192xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v403 = stablehlo.transpose %v402, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v406 = stablehlo.convolution(%v405, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v407 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v408 = stablehlo.add %v406, %v407 : tensor<32x768x28x28xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v411 = stablehlo.multiply %v410, %v410 : tensor<32x768x28x28xf32>
    %v412 = stablehlo.multiply %v411, %v410 : tensor<32x768x28x28xf32>
    %v413 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v414 = stablehlo.multiply %v413, %v412 : tensor<32x768x28x28xf32>
    %v415 = stablehlo.add %v410, %v414 : tensor<32x768x28x28xf32>
    %v416 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v417 = stablehlo.multiply %v416, %v415 : tensor<32x768x28x28xf32>
    %v418 = stablehlo.tanh %v417 : tensor<32x768x28x28xf32>
    %v419 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v420 = stablehlo.add %v419, %v418 : tensor<32x768x28x28xf32>
    %v421 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v422 = stablehlo.multiply %v421, %v410 : tensor<32x768x28x28xf32>
    %v423 = stablehlo.multiply %v422, %v420 : tensor<32x768x28x28xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v426 = stablehlo.convolution(%v425, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v427 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v428 = stablehlo.add %v426, %v427 : tensor<32x192x28x28xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v431 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v432 = stablehlo.multiply %v430, %v431 : tensor<32x192x28x28xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v435 = stablehlo.reshape %v365 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v436 = stablehlo.add %v434, %v435 : tensor<32x192x28x28xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v439 = stablehlo.convolution(%v438, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v440 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v441 = stablehlo.add %v439, %v440 : tensor<32x192x28x28xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v444 = stablehlo.transpose %v443, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v448 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v449 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v450 = stablehlo.reduce(%v446 init: %v447) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v452 = stablehlo.divide %v451, %v448 : tensor<32x784x192xf32>
    %v453 = stablehlo.subtract %v446, %v452 : tensor<32x784x192xf32>
    %v454 = stablehlo.multiply %v453, %v453 : tensor<32x784x192xf32>
    %v455 = stablehlo.reduce(%v454 init: %v447) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v456 = stablehlo.broadcast_in_dim %v455, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v457 = stablehlo.divide %v456, %v448 : tensor<32x784x192xf32>
    %v458 = stablehlo.add %v457, %v449 : tensor<32x784x192xf32>
    %v459 = stablehlo.rsqrt %v458 : tensor<32x784x192xf32>
    %v460 = stablehlo.multiply %v453, %v459 : tensor<32x784x192xf32>
    %v461 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v462 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v463 = stablehlo.multiply %v460, %v461 : tensor<32x784x192xf32>
    %v464 = stablehlo.add %v463, %v462 : tensor<32x784x192xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v467 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v468 = stablehlo.multiply %v466, %v467 : tensor<32x784x192xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v471 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v472 = stablehlo.add %v470, %v471 : tensor<32x784x192xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v475 = stablehlo.transpose %v474, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v478 = stablehlo.convolution(%v477, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v479 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v480 = stablehlo.add %v478, %v479 : tensor<32x768x28x28xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v483 = stablehlo.multiply %v482, %v482 : tensor<32x768x28x28xf32>
    %v484 = stablehlo.multiply %v483, %v482 : tensor<32x768x28x28xf32>
    %v485 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v486 = stablehlo.multiply %v485, %v484 : tensor<32x768x28x28xf32>
    %v487 = stablehlo.add %v482, %v486 : tensor<32x768x28x28xf32>
    %v488 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v489 = stablehlo.multiply %v488, %v487 : tensor<32x768x28x28xf32>
    %v490 = stablehlo.tanh %v489 : tensor<32x768x28x28xf32>
    %v491 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v492 = stablehlo.add %v491, %v490 : tensor<32x768x28x28xf32>
    %v493 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v494 = stablehlo.multiply %v493, %v482 : tensor<32x768x28x28xf32>
    %v495 = stablehlo.multiply %v494, %v492 : tensor<32x768x28x28xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v498 = stablehlo.convolution(%v497, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v499 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v500 = stablehlo.add %v498, %v499 : tensor<32x192x28x28xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v503 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v504 = stablehlo.multiply %v502, %v503 : tensor<32x192x28x28xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v507 = stablehlo.reshape %v437 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<32x192x28x28xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v511 = stablehlo.transpose %v510, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v515 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v516 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v517 = stablehlo.reduce(%v513 init: %v514) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v518 = stablehlo.broadcast_in_dim %v517, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v519 = stablehlo.divide %v518, %v515 : tensor<32x784x192xf32>
    %v520 = stablehlo.subtract %v513, %v519 : tensor<32x784x192xf32>
    %v521 = stablehlo.multiply %v520, %v520 : tensor<32x784x192xf32>
    %v522 = stablehlo.reduce(%v521 init: %v514) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v523 = stablehlo.broadcast_in_dim %v522, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v524 = stablehlo.divide %v523, %v515 : tensor<32x784x192xf32>
    %v525 = stablehlo.add %v524, %v516 : tensor<32x784x192xf32>
    %v526 = stablehlo.rsqrt %v525 : tensor<32x784x192xf32>
    %v527 = stablehlo.multiply %v520, %v526 : tensor<32x784x192xf32>
    %v528 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v529 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v530 = stablehlo.multiply %v527, %v528 : tensor<32x784x192xf32>
    %v531 = stablehlo.add %v530, %v529 : tensor<32x784x192xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v534 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v535 = stablehlo.multiply %v533, %v534 : tensor<32x784x192xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v538 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v539 = stablehlo.add %v537, %v538 : tensor<32x784x192xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v542 = stablehlo.transpose %v541, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v545 = stablehlo.convolution(%v544, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v546 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v547 = stablehlo.add %v545, %v546 : tensor<32x384x14x14xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v550 = stablehlo.convolution(%v549, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v551 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v552 = stablehlo.add %v550, %v551 : tensor<32x384x14x14xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v555 = stablehlo.transpose %v554, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v558 = stablehlo.constant dense<0.0> : tensor<f32>
    %v559 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v560 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v561 = stablehlo.reduce(%v557 init: %v558) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v562 = stablehlo.broadcast_in_dim %v561, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v563 = stablehlo.divide %v562, %v559 : tensor<32x196x384xf32>
    %v564 = stablehlo.subtract %v557, %v563 : tensor<32x196x384xf32>
    %v565 = stablehlo.multiply %v564, %v564 : tensor<32x196x384xf32>
    %v566 = stablehlo.reduce(%v565 init: %v558) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v567 = stablehlo.broadcast_in_dim %v566, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v568 = stablehlo.divide %v567, %v559 : tensor<32x196x384xf32>
    %v569 = stablehlo.add %v568, %v560 : tensor<32x196x384xf32>
    %v570 = stablehlo.rsqrt %v569 : tensor<32x196x384xf32>
    %v571 = stablehlo.multiply %v564, %v570 : tensor<32x196x384xf32>
    %v572 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v573 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v574 = stablehlo.multiply %v571, %v572 : tensor<32x196x384xf32>
    %v575 = stablehlo.add %v574, %v573 : tensor<32x196x384xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v578 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v579 = stablehlo.multiply %v577, %v578 : tensor<32x196x384xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v582 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v583 = stablehlo.add %v581, %v582 : tensor<32x196x384xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v586 = stablehlo.transpose %v585, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v589 = stablehlo.convolution(%v588, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v590 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v591 = stablehlo.add %v589, %v590 : tensor<32x1536x14x14xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v594 = stablehlo.multiply %v593, %v593 : tensor<32x1536x14x14xf32>
    %v595 = stablehlo.multiply %v594, %v593 : tensor<32x1536x14x14xf32>
    %v596 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v597 = stablehlo.multiply %v596, %v595 : tensor<32x1536x14x14xf32>
    %v598 = stablehlo.add %v593, %v597 : tensor<32x1536x14x14xf32>
    %v599 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v600 = stablehlo.multiply %v599, %v598 : tensor<32x1536x14x14xf32>
    %v601 = stablehlo.tanh %v600 : tensor<32x1536x14x14xf32>
    %v602 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v603 = stablehlo.add %v602, %v601 : tensor<32x1536x14x14xf32>
    %v604 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v605 = stablehlo.multiply %v604, %v593 : tensor<32x1536x14x14xf32>
    %v606 = stablehlo.multiply %v605, %v603 : tensor<32x1536x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v609 = stablehlo.convolution(%v608, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<32x384x14x14xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v615 = stablehlo.multiply %v613, %v614 : tensor<32x384x14x14xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v618 = stablehlo.reshape %v548 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v619 = stablehlo.add %v617, %v618 : tensor<32x384x14x14xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v622 = stablehlo.convolution(%v621, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v624 = stablehlo.add %v622, %v623 : tensor<32x384x14x14xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v627 = stablehlo.transpose %v626, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v631 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v632 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v633 = stablehlo.reduce(%v629 init: %v630) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v634 = stablehlo.broadcast_in_dim %v633, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v635 = stablehlo.divide %v634, %v631 : tensor<32x196x384xf32>
    %v636 = stablehlo.subtract %v629, %v635 : tensor<32x196x384xf32>
    %v637 = stablehlo.multiply %v636, %v636 : tensor<32x196x384xf32>
    %v638 = stablehlo.reduce(%v637 init: %v630) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v639 = stablehlo.broadcast_in_dim %v638, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v640 = stablehlo.divide %v639, %v631 : tensor<32x196x384xf32>
    %v641 = stablehlo.add %v640, %v632 : tensor<32x196x384xf32>
    %v642 = stablehlo.rsqrt %v641 : tensor<32x196x384xf32>
    %v643 = stablehlo.multiply %v636, %v642 : tensor<32x196x384xf32>
    %v644 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v645 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v646 = stablehlo.multiply %v643, %v644 : tensor<32x196x384xf32>
    %v647 = stablehlo.add %v646, %v645 : tensor<32x196x384xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v650 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v651 = stablehlo.multiply %v649, %v650 : tensor<32x196x384xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v654 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v655 = stablehlo.add %v653, %v654 : tensor<32x196x384xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v658 = stablehlo.transpose %v657, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v661 = stablehlo.convolution(%v660, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v663 = stablehlo.add %v661, %v662 : tensor<32x1536x14x14xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v666 = stablehlo.multiply %v665, %v665 : tensor<32x1536x14x14xf32>
    %v667 = stablehlo.multiply %v666, %v665 : tensor<32x1536x14x14xf32>
    %v668 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v669 = stablehlo.multiply %v668, %v667 : tensor<32x1536x14x14xf32>
    %v670 = stablehlo.add %v665, %v669 : tensor<32x1536x14x14xf32>
    %v671 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v672 = stablehlo.multiply %v671, %v670 : tensor<32x1536x14x14xf32>
    %v673 = stablehlo.tanh %v672 : tensor<32x1536x14x14xf32>
    %v674 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v675 = stablehlo.add %v674, %v673 : tensor<32x1536x14x14xf32>
    %v676 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v677 = stablehlo.multiply %v676, %v665 : tensor<32x1536x14x14xf32>
    %v678 = stablehlo.multiply %v677, %v675 : tensor<32x1536x14x14xf32>
    %v679 = stablehlo.reshape %v678 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v681 = stablehlo.convolution(%v680, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v682 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v683 = stablehlo.add %v681, %v682 : tensor<32x384x14x14xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v686 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v687 = stablehlo.multiply %v685, %v686 : tensor<32x384x14x14xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v690 = stablehlo.reshape %v620 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v691 = stablehlo.add %v689, %v690 : tensor<32x384x14x14xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v694 = stablehlo.convolution(%v693, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v695 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v696 = stablehlo.add %v694, %v695 : tensor<32x384x14x14xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v699 = stablehlo.transpose %v698, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v703 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v704 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v705 = stablehlo.reduce(%v701 init: %v702) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v706 = stablehlo.broadcast_in_dim %v705, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v707 = stablehlo.divide %v706, %v703 : tensor<32x196x384xf32>
    %v708 = stablehlo.subtract %v701, %v707 : tensor<32x196x384xf32>
    %v709 = stablehlo.multiply %v708, %v708 : tensor<32x196x384xf32>
    %v710 = stablehlo.reduce(%v709 init: %v702) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v711 = stablehlo.broadcast_in_dim %v710, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v712 = stablehlo.divide %v711, %v703 : tensor<32x196x384xf32>
    %v713 = stablehlo.add %v712, %v704 : tensor<32x196x384xf32>
    %v714 = stablehlo.rsqrt %v713 : tensor<32x196x384xf32>
    %v715 = stablehlo.multiply %v708, %v714 : tensor<32x196x384xf32>
    %v716 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v717 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v718 = stablehlo.multiply %v715, %v716 : tensor<32x196x384xf32>
    %v719 = stablehlo.add %v718, %v717 : tensor<32x196x384xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v722 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v723 = stablehlo.multiply %v721, %v722 : tensor<32x196x384xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v726 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v727 = stablehlo.add %v725, %v726 : tensor<32x196x384xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v730 = stablehlo.transpose %v729, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v733 = stablehlo.convolution(%v732, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<32x1536x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v738 = stablehlo.multiply %v737, %v737 : tensor<32x1536x14x14xf32>
    %v739 = stablehlo.multiply %v738, %v737 : tensor<32x1536x14x14xf32>
    %v740 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v741 = stablehlo.multiply %v740, %v739 : tensor<32x1536x14x14xf32>
    %v742 = stablehlo.add %v737, %v741 : tensor<32x1536x14x14xf32>
    %v743 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v744 = stablehlo.multiply %v743, %v742 : tensor<32x1536x14x14xf32>
    %v745 = stablehlo.tanh %v744 : tensor<32x1536x14x14xf32>
    %v746 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v747 = stablehlo.add %v746, %v745 : tensor<32x1536x14x14xf32>
    %v748 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v749 = stablehlo.multiply %v748, %v737 : tensor<32x1536x14x14xf32>
    %v750 = stablehlo.multiply %v749, %v747 : tensor<32x1536x14x14xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v753 = stablehlo.convolution(%v752, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v754 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v755 = stablehlo.add %v753, %v754 : tensor<32x384x14x14xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v758 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v759 = stablehlo.multiply %v757, %v758 : tensor<32x384x14x14xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v762 = stablehlo.reshape %v692 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v763 = stablehlo.add %v761, %v762 : tensor<32x384x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v766 = stablehlo.convolution(%v765, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x384x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v771 = stablehlo.transpose %v770, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v775 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v776 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v777 = stablehlo.reduce(%v773 init: %v774) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v778 = stablehlo.broadcast_in_dim %v777, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v779 = stablehlo.divide %v778, %v775 : tensor<32x196x384xf32>
    %v780 = stablehlo.subtract %v773, %v779 : tensor<32x196x384xf32>
    %v781 = stablehlo.multiply %v780, %v780 : tensor<32x196x384xf32>
    %v782 = stablehlo.reduce(%v781 init: %v774) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v783 = stablehlo.broadcast_in_dim %v782, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v784 = stablehlo.divide %v783, %v775 : tensor<32x196x384xf32>
    %v785 = stablehlo.add %v784, %v776 : tensor<32x196x384xf32>
    %v786 = stablehlo.rsqrt %v785 : tensor<32x196x384xf32>
    %v787 = stablehlo.multiply %v780, %v786 : tensor<32x196x384xf32>
    %v788 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v789 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v790 = stablehlo.multiply %v787, %v788 : tensor<32x196x384xf32>
    %v791 = stablehlo.add %v790, %v789 : tensor<32x196x384xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v794 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v795 = stablehlo.multiply %v793, %v794 : tensor<32x196x384xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v798 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<32x196x384xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v802 = stablehlo.transpose %v801, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v805 = stablehlo.convolution(%v804, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v806 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v807 = stablehlo.add %v805, %v806 : tensor<32x1536x14x14xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v810 = stablehlo.multiply %v809, %v809 : tensor<32x1536x14x14xf32>
    %v811 = stablehlo.multiply %v810, %v809 : tensor<32x1536x14x14xf32>
    %v812 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v813 = stablehlo.multiply %v812, %v811 : tensor<32x1536x14x14xf32>
    %v814 = stablehlo.add %v809, %v813 : tensor<32x1536x14x14xf32>
    %v815 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v816 = stablehlo.multiply %v815, %v814 : tensor<32x1536x14x14xf32>
    %v817 = stablehlo.tanh %v816 : tensor<32x1536x14x14xf32>
    %v818 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v819 = stablehlo.add %v818, %v817 : tensor<32x1536x14x14xf32>
    %v820 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v821 = stablehlo.multiply %v820, %v809 : tensor<32x1536x14x14xf32>
    %v822 = stablehlo.multiply %v821, %v819 : tensor<32x1536x14x14xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v825 = stablehlo.convolution(%v824, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v826 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v827 = stablehlo.add %v825, %v826 : tensor<32x384x14x14xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v830 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v831 = stablehlo.multiply %v829, %v830 : tensor<32x384x14x14xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v834 = stablehlo.reshape %v764 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v835 = stablehlo.add %v833, %v834 : tensor<32x384x14x14xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v838 = stablehlo.convolution(%v837, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v839 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v840 = stablehlo.add %v838, %v839 : tensor<32x384x14x14xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v843 = stablehlo.transpose %v842, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v847 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v848 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v849 = stablehlo.reduce(%v845 init: %v846) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v850 = stablehlo.broadcast_in_dim %v849, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v851 = stablehlo.divide %v850, %v847 : tensor<32x196x384xf32>
    %v852 = stablehlo.subtract %v845, %v851 : tensor<32x196x384xf32>
    %v853 = stablehlo.multiply %v852, %v852 : tensor<32x196x384xf32>
    %v854 = stablehlo.reduce(%v853 init: %v846) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v855 = stablehlo.broadcast_in_dim %v854, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v856 = stablehlo.divide %v855, %v847 : tensor<32x196x384xf32>
    %v857 = stablehlo.add %v856, %v848 : tensor<32x196x384xf32>
    %v858 = stablehlo.rsqrt %v857 : tensor<32x196x384xf32>
    %v859 = stablehlo.multiply %v852, %v858 : tensor<32x196x384xf32>
    %v860 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v861 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v862 = stablehlo.multiply %v859, %v860 : tensor<32x196x384xf32>
    %v863 = stablehlo.add %v862, %v861 : tensor<32x196x384xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v866 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v867 = stablehlo.multiply %v865, %v866 : tensor<32x196x384xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v870 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v871 = stablehlo.add %v869, %v870 : tensor<32x196x384xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v874 = stablehlo.transpose %v873, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v877 = stablehlo.convolution(%v876, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v878 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<32x1536x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v882 = stablehlo.multiply %v881, %v881 : tensor<32x1536x14x14xf32>
    %v883 = stablehlo.multiply %v882, %v881 : tensor<32x1536x14x14xf32>
    %v884 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v885 = stablehlo.multiply %v884, %v883 : tensor<32x1536x14x14xf32>
    %v886 = stablehlo.add %v881, %v885 : tensor<32x1536x14x14xf32>
    %v887 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v888 = stablehlo.multiply %v887, %v886 : tensor<32x1536x14x14xf32>
    %v889 = stablehlo.tanh %v888 : tensor<32x1536x14x14xf32>
    %v890 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v891 = stablehlo.add %v890, %v889 : tensor<32x1536x14x14xf32>
    %v892 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v893 = stablehlo.multiply %v892, %v881 : tensor<32x1536x14x14xf32>
    %v894 = stablehlo.multiply %v893, %v891 : tensor<32x1536x14x14xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v897 = stablehlo.convolution(%v896, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v898 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v899 = stablehlo.add %v897, %v898 : tensor<32x384x14x14xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v901 = stablehlo.reshape %v900 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v902 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v903 = stablehlo.multiply %v901, %v902 : tensor<32x384x14x14xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v906 = stablehlo.reshape %v836 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v907 = stablehlo.add %v905, %v906 : tensor<32x384x14x14xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v910 = stablehlo.convolution(%v909, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v911 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v912 = stablehlo.add %v910, %v911 : tensor<32x384x14x14xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v915 = stablehlo.transpose %v914, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v919 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v920 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v921 = stablehlo.reduce(%v917 init: %v918) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v922 = stablehlo.broadcast_in_dim %v921, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v923 = stablehlo.divide %v922, %v919 : tensor<32x196x384xf32>
    %v924 = stablehlo.subtract %v917, %v923 : tensor<32x196x384xf32>
    %v925 = stablehlo.multiply %v924, %v924 : tensor<32x196x384xf32>
    %v926 = stablehlo.reduce(%v925 init: %v918) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v927 = stablehlo.broadcast_in_dim %v926, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v928 = stablehlo.divide %v927, %v919 : tensor<32x196x384xf32>
    %v929 = stablehlo.add %v928, %v920 : tensor<32x196x384xf32>
    %v930 = stablehlo.rsqrt %v929 : tensor<32x196x384xf32>
    %v931 = stablehlo.multiply %v924, %v930 : tensor<32x196x384xf32>
    %v932 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v933 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v934 = stablehlo.multiply %v931, %v932 : tensor<32x196x384xf32>
    %v935 = stablehlo.add %v934, %v933 : tensor<32x196x384xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v938 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v939 = stablehlo.multiply %v937, %v938 : tensor<32x196x384xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v942 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v943 = stablehlo.add %v941, %v942 : tensor<32x196x384xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v946 = stablehlo.transpose %v945, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v949 = stablehlo.convolution(%v948, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v950 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v951 = stablehlo.add %v949, %v950 : tensor<32x1536x14x14xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v954 = stablehlo.multiply %v953, %v953 : tensor<32x1536x14x14xf32>
    %v955 = stablehlo.multiply %v954, %v953 : tensor<32x1536x14x14xf32>
    %v956 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v957 = stablehlo.multiply %v956, %v955 : tensor<32x1536x14x14xf32>
    %v958 = stablehlo.add %v953, %v957 : tensor<32x1536x14x14xf32>
    %v959 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v960 = stablehlo.multiply %v959, %v958 : tensor<32x1536x14x14xf32>
    %v961 = stablehlo.tanh %v960 : tensor<32x1536x14x14xf32>
    %v962 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v963 = stablehlo.add %v962, %v961 : tensor<32x1536x14x14xf32>
    %v964 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v965 = stablehlo.multiply %v964, %v953 : tensor<32x1536x14x14xf32>
    %v966 = stablehlo.multiply %v965, %v963 : tensor<32x1536x14x14xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v969 = stablehlo.convolution(%v968, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v970 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v971 = stablehlo.add %v969, %v970 : tensor<32x384x14x14xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v974 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v975 = stablehlo.multiply %v973, %v974 : tensor<32x384x14x14xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v978 = stablehlo.reshape %v908 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v979 = stablehlo.add %v977, %v978 : tensor<32x384x14x14xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v982 = stablehlo.convolution(%v981, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v983 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v984 = stablehlo.add %v982, %v983 : tensor<32x384x14x14xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v987 = stablehlo.transpose %v986, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v991 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v992 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v993 = stablehlo.reduce(%v989 init: %v990) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v994 = stablehlo.broadcast_in_dim %v993, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v995 = stablehlo.divide %v994, %v991 : tensor<32x196x384xf32>
    %v996 = stablehlo.subtract %v989, %v995 : tensor<32x196x384xf32>
    %v997 = stablehlo.multiply %v996, %v996 : tensor<32x196x384xf32>
    %v998 = stablehlo.reduce(%v997 init: %v990) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v999 = stablehlo.broadcast_in_dim %v998, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1000 = stablehlo.divide %v999, %v991 : tensor<32x196x384xf32>
    %v1001 = stablehlo.add %v1000, %v992 : tensor<32x196x384xf32>
    %v1002 = stablehlo.rsqrt %v1001 : tensor<32x196x384xf32>
    %v1003 = stablehlo.multiply %v996, %v1002 : tensor<32x196x384xf32>
    %v1004 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1005 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1006 = stablehlo.multiply %v1003, %v1004 : tensor<32x196x384xf32>
    %v1007 = stablehlo.add %v1006, %v1005 : tensor<32x196x384xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1010 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1011 = stablehlo.multiply %v1009, %v1010 : tensor<32x196x384xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1014 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1015 = stablehlo.add %v1013, %v1014 : tensor<32x196x384xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1018 = stablehlo.transpose %v1017, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1021 = stablehlo.convolution(%v1020, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1022 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1023 = stablehlo.add %v1021, %v1022 : tensor<32x1536x14x14xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1025 = stablehlo.reshape %v1024 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1026 = stablehlo.multiply %v1025, %v1025 : tensor<32x1536x14x14xf32>
    %v1027 = stablehlo.multiply %v1026, %v1025 : tensor<32x1536x14x14xf32>
    %v1028 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v1029 = stablehlo.multiply %v1028, %v1027 : tensor<32x1536x14x14xf32>
    %v1030 = stablehlo.add %v1025, %v1029 : tensor<32x1536x14x14xf32>
    %v1031 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v1032 = stablehlo.multiply %v1031, %v1030 : tensor<32x1536x14x14xf32>
    %v1033 = stablehlo.tanh %v1032 : tensor<32x1536x14x14xf32>
    %v1034 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v1035 = stablehlo.add %v1034, %v1033 : tensor<32x1536x14x14xf32>
    %v1036 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1037 = stablehlo.multiply %v1036, %v1025 : tensor<32x1536x14x14xf32>
    %v1038 = stablehlo.multiply %v1037, %v1035 : tensor<32x1536x14x14xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1041 = stablehlo.convolution(%v1040, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1042 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1043 = stablehlo.add %v1041, %v1042 : tensor<32x384x14x14xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1046 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1047 = stablehlo.multiply %v1045, %v1046 : tensor<32x384x14x14xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1050 = stablehlo.reshape %v980 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1051 = stablehlo.add %v1049, %v1050 : tensor<32x384x14x14xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1054 = stablehlo.convolution(%v1053, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1055 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1056 = stablehlo.add %v1054, %v1055 : tensor<32x384x14x14xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1059 = stablehlo.transpose %v1058, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1063 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1064 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1065 = stablehlo.reduce(%v1061 init: %v1062) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1066 = stablehlo.broadcast_in_dim %v1065, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1067 = stablehlo.divide %v1066, %v1063 : tensor<32x196x384xf32>
    %v1068 = stablehlo.subtract %v1061, %v1067 : tensor<32x196x384xf32>
    %v1069 = stablehlo.multiply %v1068, %v1068 : tensor<32x196x384xf32>
    %v1070 = stablehlo.reduce(%v1069 init: %v1062) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1071 = stablehlo.broadcast_in_dim %v1070, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1072 = stablehlo.divide %v1071, %v1063 : tensor<32x196x384xf32>
    %v1073 = stablehlo.add %v1072, %v1064 : tensor<32x196x384xf32>
    %v1074 = stablehlo.rsqrt %v1073 : tensor<32x196x384xf32>
    %v1075 = stablehlo.multiply %v1068, %v1074 : tensor<32x196x384xf32>
    %v1076 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1077 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1078 = stablehlo.multiply %v1075, %v1076 : tensor<32x196x384xf32>
    %v1079 = stablehlo.add %v1078, %v1077 : tensor<32x196x384xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1082 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1083 = stablehlo.multiply %v1081, %v1082 : tensor<32x196x384xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1086 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1087 = stablehlo.add %v1085, %v1086 : tensor<32x196x384xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1090 = stablehlo.transpose %v1089, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1093 = stablehlo.convolution(%v1092, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1094 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1095 = stablehlo.add %v1093, %v1094 : tensor<32x1536x14x14xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1098 = stablehlo.multiply %v1097, %v1097 : tensor<32x1536x14x14xf32>
    %v1099 = stablehlo.multiply %v1098, %v1097 : tensor<32x1536x14x14xf32>
    %v1100 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v1101 = stablehlo.multiply %v1100, %v1099 : tensor<32x1536x14x14xf32>
    %v1102 = stablehlo.add %v1097, %v1101 : tensor<32x1536x14x14xf32>
    %v1103 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v1104 = stablehlo.multiply %v1103, %v1102 : tensor<32x1536x14x14xf32>
    %v1105 = stablehlo.tanh %v1104 : tensor<32x1536x14x14xf32>
    %v1106 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v1107 = stablehlo.add %v1106, %v1105 : tensor<32x1536x14x14xf32>
    %v1108 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1109 = stablehlo.multiply %v1108, %v1097 : tensor<32x1536x14x14xf32>
    %v1110 = stablehlo.multiply %v1109, %v1107 : tensor<32x1536x14x14xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1113 = stablehlo.convolution(%v1112, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1114 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1115 = stablehlo.add %v1113, %v1114 : tensor<32x384x14x14xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1117 = stablehlo.reshape %v1116 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1118 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1119 = stablehlo.multiply %v1117, %v1118 : tensor<32x384x14x14xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1122 = stablehlo.reshape %v1052 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1123 = stablehlo.add %v1121, %v1122 : tensor<32x384x14x14xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1126 = stablehlo.convolution(%v1125, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1127 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1128 = stablehlo.add %v1126, %v1127 : tensor<32x384x14x14xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1130 = stablehlo.reshape %v1129 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1131 = stablehlo.transpose %v1130, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1134 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1135 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1136 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1137 = stablehlo.reduce(%v1133 init: %v1134) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1138 = stablehlo.broadcast_in_dim %v1137, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1139 = stablehlo.divide %v1138, %v1135 : tensor<32x196x384xf32>
    %v1140 = stablehlo.subtract %v1133, %v1139 : tensor<32x196x384xf32>
    %v1141 = stablehlo.multiply %v1140, %v1140 : tensor<32x196x384xf32>
    %v1142 = stablehlo.reduce(%v1141 init: %v1134) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1143 = stablehlo.broadcast_in_dim %v1142, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1144 = stablehlo.divide %v1143, %v1135 : tensor<32x196x384xf32>
    %v1145 = stablehlo.add %v1144, %v1136 : tensor<32x196x384xf32>
    %v1146 = stablehlo.rsqrt %v1145 : tensor<32x196x384xf32>
    %v1147 = stablehlo.multiply %v1140, %v1146 : tensor<32x196x384xf32>
    %v1148 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1149 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1150 = stablehlo.multiply %v1147, %v1148 : tensor<32x196x384xf32>
    %v1151 = stablehlo.add %v1150, %v1149 : tensor<32x196x384xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1154 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1155 = stablehlo.multiply %v1153, %v1154 : tensor<32x196x384xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1157 = stablehlo.reshape %v1156 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1158 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1159 = stablehlo.add %v1157, %v1158 : tensor<32x196x384xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1162 = stablehlo.transpose %v1161, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1165 = stablehlo.convolution(%v1164, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1166 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1167 = stablehlo.add %v1165, %v1166 : tensor<32x1536x14x14xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1170 = stablehlo.multiply %v1169, %v1169 : tensor<32x1536x14x14xf32>
    %v1171 = stablehlo.multiply %v1170, %v1169 : tensor<32x1536x14x14xf32>
    %v1172 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v1173 = stablehlo.multiply %v1172, %v1171 : tensor<32x1536x14x14xf32>
    %v1174 = stablehlo.add %v1169, %v1173 : tensor<32x1536x14x14xf32>
    %v1175 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v1176 = stablehlo.multiply %v1175, %v1174 : tensor<32x1536x14x14xf32>
    %v1177 = stablehlo.tanh %v1176 : tensor<32x1536x14x14xf32>
    %v1178 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v1179 = stablehlo.add %v1178, %v1177 : tensor<32x1536x14x14xf32>
    %v1180 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1181 = stablehlo.multiply %v1180, %v1169 : tensor<32x1536x14x14xf32>
    %v1182 = stablehlo.multiply %v1181, %v1179 : tensor<32x1536x14x14xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1185 = stablehlo.convolution(%v1184, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1186 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1187 = stablehlo.add %v1185, %v1186 : tensor<32x384x14x14xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1190 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1191 = stablehlo.multiply %v1189, %v1190 : tensor<32x384x14x14xf32>
    %v1192 = stablehlo.reshape %v1191 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1194 = stablehlo.reshape %v1124 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1195 = stablehlo.add %v1193, %v1194 : tensor<32x384x14x14xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1197 = stablehlo.reshape %v1196 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1198 = stablehlo.transpose %v1197, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1202 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1203 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1204 = stablehlo.reduce(%v1200 init: %v1201) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1205 = stablehlo.broadcast_in_dim %v1204, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1206 = stablehlo.divide %v1205, %v1202 : tensor<32x196x384xf32>
    %v1207 = stablehlo.subtract %v1200, %v1206 : tensor<32x196x384xf32>
    %v1208 = stablehlo.multiply %v1207, %v1207 : tensor<32x196x384xf32>
    %v1209 = stablehlo.reduce(%v1208 init: %v1201) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1210 = stablehlo.broadcast_in_dim %v1209, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1211 = stablehlo.divide %v1210, %v1202 : tensor<32x196x384xf32>
    %v1212 = stablehlo.add %v1211, %v1203 : tensor<32x196x384xf32>
    %v1213 = stablehlo.rsqrt %v1212 : tensor<32x196x384xf32>
    %v1214 = stablehlo.multiply %v1207, %v1213 : tensor<32x196x384xf32>
    %v1215 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1216 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1217 = stablehlo.multiply %v1214, %v1215 : tensor<32x196x384xf32>
    %v1218 = stablehlo.add %v1217, %v1216 : tensor<32x196x384xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1221 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1222 = stablehlo.multiply %v1220, %v1221 : tensor<32x196x384xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1225 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1226 = stablehlo.add %v1224, %v1225 : tensor<32x196x384xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1229 = stablehlo.transpose %v1228, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1232 = stablehlo.convolution(%v1231, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %v1233 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1234 = stablehlo.add %v1232, %v1233 : tensor<32x768x7x7xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1236 = stablehlo.reshape %v1235 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1237 = stablehlo.convolution(%v1236, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1238 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1239 = stablehlo.add %v1237, %v1238 : tensor<32x768x7x7xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1242 = stablehlo.transpose %v1241, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1246 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1247 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1248 = stablehlo.reduce(%v1244 init: %v1245) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1249 = stablehlo.broadcast_in_dim %v1248, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1250 = stablehlo.divide %v1249, %v1246 : tensor<32x49x768xf32>
    %v1251 = stablehlo.subtract %v1244, %v1250 : tensor<32x49x768xf32>
    %v1252 = stablehlo.multiply %v1251, %v1251 : tensor<32x49x768xf32>
    %v1253 = stablehlo.reduce(%v1252 init: %v1245) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1254 = stablehlo.broadcast_in_dim %v1253, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1255 = stablehlo.divide %v1254, %v1246 : tensor<32x49x768xf32>
    %v1256 = stablehlo.add %v1255, %v1247 : tensor<32x49x768xf32>
    %v1257 = stablehlo.rsqrt %v1256 : tensor<32x49x768xf32>
    %v1258 = stablehlo.multiply %v1251, %v1257 : tensor<32x49x768xf32>
    %v1259 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1260 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1261 = stablehlo.multiply %v1258, %v1259 : tensor<32x49x768xf32>
    %v1262 = stablehlo.add %v1261, %v1260 : tensor<32x49x768xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1265 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1266 = stablehlo.multiply %v1264, %v1265 : tensor<32x49x768xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1268 = stablehlo.reshape %v1267 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1269 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1270 = stablehlo.add %v1268, %v1269 : tensor<32x49x768xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1273 = stablehlo.transpose %v1272, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1276 = stablehlo.convolution(%v1275, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1277 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1278 = stablehlo.add %v1276, %v1277 : tensor<32x3072x7x7xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1281 = stablehlo.multiply %v1280, %v1280 : tensor<32x3072x7x7xf32>
    %v1282 = stablehlo.multiply %v1281, %v1280 : tensor<32x3072x7x7xf32>
    %v1283 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1284 = stablehlo.multiply %v1283, %v1282 : tensor<32x3072x7x7xf32>
    %v1285 = stablehlo.add %v1280, %v1284 : tensor<32x3072x7x7xf32>
    %v1286 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1287 = stablehlo.multiply %v1286, %v1285 : tensor<32x3072x7x7xf32>
    %v1288 = stablehlo.tanh %v1287 : tensor<32x3072x7x7xf32>
    %v1289 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1290 = stablehlo.add %v1289, %v1288 : tensor<32x3072x7x7xf32>
    %v1291 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1292 = stablehlo.multiply %v1291, %v1280 : tensor<32x3072x7x7xf32>
    %v1293 = stablehlo.multiply %v1292, %v1290 : tensor<32x3072x7x7xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1296 = stablehlo.convolution(%v1295, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1297 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1298 = stablehlo.add %v1296, %v1297 : tensor<32x768x7x7xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1301 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1302 = stablehlo.multiply %v1300, %v1301 : tensor<32x768x7x7xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1305 = stablehlo.reshape %v1235 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1306 = stablehlo.add %v1304, %v1305 : tensor<32x768x7x7xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1309 = stablehlo.convolution(%v1308, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1310 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1311 = stablehlo.add %v1309, %v1310 : tensor<32x768x7x7xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1314 = stablehlo.transpose %v1313, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1318 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1319 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1320 = stablehlo.reduce(%v1316 init: %v1317) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1321 = stablehlo.broadcast_in_dim %v1320, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1322 = stablehlo.divide %v1321, %v1318 : tensor<32x49x768xf32>
    %v1323 = stablehlo.subtract %v1316, %v1322 : tensor<32x49x768xf32>
    %v1324 = stablehlo.multiply %v1323, %v1323 : tensor<32x49x768xf32>
    %v1325 = stablehlo.reduce(%v1324 init: %v1317) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1326 = stablehlo.broadcast_in_dim %v1325, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1327 = stablehlo.divide %v1326, %v1318 : tensor<32x49x768xf32>
    %v1328 = stablehlo.add %v1327, %v1319 : tensor<32x49x768xf32>
    %v1329 = stablehlo.rsqrt %v1328 : tensor<32x49x768xf32>
    %v1330 = stablehlo.multiply %v1323, %v1329 : tensor<32x49x768xf32>
    %v1331 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1332 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1333 = stablehlo.multiply %v1330, %v1331 : tensor<32x49x768xf32>
    %v1334 = stablehlo.add %v1333, %v1332 : tensor<32x49x768xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1337 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1338 = stablehlo.multiply %v1336, %v1337 : tensor<32x49x768xf32>
    %v1339 = stablehlo.reshape %v1338 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1341 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1342 = stablehlo.add %v1340, %v1341 : tensor<32x49x768xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1345 = stablehlo.transpose %v1344, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1348 = stablehlo.convolution(%v1347, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1349 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1350 = stablehlo.add %v1348, %v1349 : tensor<32x3072x7x7xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1353 = stablehlo.multiply %v1352, %v1352 : tensor<32x3072x7x7xf32>
    %v1354 = stablehlo.multiply %v1353, %v1352 : tensor<32x3072x7x7xf32>
    %v1355 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1356 = stablehlo.multiply %v1355, %v1354 : tensor<32x3072x7x7xf32>
    %v1357 = stablehlo.add %v1352, %v1356 : tensor<32x3072x7x7xf32>
    %v1358 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1359 = stablehlo.multiply %v1358, %v1357 : tensor<32x3072x7x7xf32>
    %v1360 = stablehlo.tanh %v1359 : tensor<32x3072x7x7xf32>
    %v1361 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1362 = stablehlo.add %v1361, %v1360 : tensor<32x3072x7x7xf32>
    %v1363 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1364 = stablehlo.multiply %v1363, %v1352 : tensor<32x3072x7x7xf32>
    %v1365 = stablehlo.multiply %v1364, %v1362 : tensor<32x3072x7x7xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1368 = stablehlo.convolution(%v1367, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1369 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1370 = stablehlo.add %v1368, %v1369 : tensor<32x768x7x7xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1373 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1374 = stablehlo.multiply %v1372, %v1373 : tensor<32x768x7x7xf32>
    %v1375 = stablehlo.reshape %v1374 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1377 = stablehlo.reshape %v1307 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1378 = stablehlo.add %v1376, %v1377 : tensor<32x768x7x7xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1381 = stablehlo.convolution(%v1380, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1382 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1383 = stablehlo.add %v1381, %v1382 : tensor<32x768x7x7xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1386 = stablehlo.transpose %v1385, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1387 = stablehlo.reshape %v1386 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1388 = stablehlo.reshape %v1387 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1390 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1391 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1392 = stablehlo.reduce(%v1388 init: %v1389) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1393 = stablehlo.broadcast_in_dim %v1392, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1394 = stablehlo.divide %v1393, %v1390 : tensor<32x49x768xf32>
    %v1395 = stablehlo.subtract %v1388, %v1394 : tensor<32x49x768xf32>
    %v1396 = stablehlo.multiply %v1395, %v1395 : tensor<32x49x768xf32>
    %v1397 = stablehlo.reduce(%v1396 init: %v1389) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1398 = stablehlo.broadcast_in_dim %v1397, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1399 = stablehlo.divide %v1398, %v1390 : tensor<32x49x768xf32>
    %v1400 = stablehlo.add %v1399, %v1391 : tensor<32x49x768xf32>
    %v1401 = stablehlo.rsqrt %v1400 : tensor<32x49x768xf32>
    %v1402 = stablehlo.multiply %v1395, %v1401 : tensor<32x49x768xf32>
    %v1403 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1404 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1405 = stablehlo.multiply %v1402, %v1403 : tensor<32x49x768xf32>
    %v1406 = stablehlo.add %v1405, %v1404 : tensor<32x49x768xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1409 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1410 = stablehlo.multiply %v1408, %v1409 : tensor<32x49x768xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1413 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1414 = stablehlo.add %v1412, %v1413 : tensor<32x49x768xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1417 = stablehlo.transpose %v1416, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1418 = stablehlo.reshape %v1417 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1420 = stablehlo.convolution(%v1419, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1421 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1422 = stablehlo.add %v1420, %v1421 : tensor<32x3072x7x7xf32>
    %v1423 = stablehlo.reshape %v1422 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1424 = stablehlo.reshape %v1423 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1425 = stablehlo.multiply %v1424, %v1424 : tensor<32x3072x7x7xf32>
    %v1426 = stablehlo.multiply %v1425, %v1424 : tensor<32x3072x7x7xf32>
    %v1427 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1428 = stablehlo.multiply %v1427, %v1426 : tensor<32x3072x7x7xf32>
    %v1429 = stablehlo.add %v1424, %v1428 : tensor<32x3072x7x7xf32>
    %v1430 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1431 = stablehlo.multiply %v1430, %v1429 : tensor<32x3072x7x7xf32>
    %v1432 = stablehlo.tanh %v1431 : tensor<32x3072x7x7xf32>
    %v1433 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1434 = stablehlo.add %v1433, %v1432 : tensor<32x3072x7x7xf32>
    %v1435 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1436 = stablehlo.multiply %v1435, %v1424 : tensor<32x3072x7x7xf32>
    %v1437 = stablehlo.multiply %v1436, %v1434 : tensor<32x3072x7x7xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1440 = stablehlo.convolution(%v1439, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1441 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1442 = stablehlo.add %v1440, %v1441 : tensor<32x768x7x7xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1445 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1446 = stablehlo.multiply %v1444, %v1445 : tensor<32x768x7x7xf32>
    %v1447 = stablehlo.reshape %v1446 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1449 = stablehlo.reshape %v1379 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1450 = stablehlo.add %v1448, %v1449 : tensor<32x768x7x7xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1454 = stablehlo.reduce(%v1452 init: %v1453) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %v1455 = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %v1456 = stablehlo.divide %v1454, %v1455 : tensor<32x768xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1459 = stablehlo.constant dense<768.0> : tensor<32x1x768xf32>
    %v1460 = stablehlo.constant dense<1.0e-6> : tensor<32x1x768xf32>
    %v1461 = stablehlo.reduce(%v1457 init: %v1458) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1462 = stablehlo.broadcast_in_dim %v1461, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v1463 = stablehlo.divide %v1462, %v1459 : tensor<32x1x768xf32>
    %v1464 = stablehlo.subtract %v1457, %v1463 : tensor<32x1x768xf32>
    %v1465 = stablehlo.multiply %v1464, %v1464 : tensor<32x1x768xf32>
    %v1466 = stablehlo.reduce(%v1465 init: %v1458) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1467 = stablehlo.broadcast_in_dim %v1466, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v1468 = stablehlo.divide %v1467, %v1459 : tensor<32x1x768xf32>
    %v1469 = stablehlo.add %v1468, %v1460 : tensor<32x1x768xf32>
    %v1470 = stablehlo.rsqrt %v1469 : tensor<32x1x768xf32>
    %v1471 = stablehlo.multiply %v1464, %v1470 : tensor<32x1x768xf32>
    %v1472 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x1x768xf32>
    %v1473 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x1x768xf32>
    %v1474 = stablehlo.multiply %v1471, %v1472 : tensor<32x1x768xf32>
    %v1475 = stablehlo.add %v1474, %v1473 : tensor<32x1x768xf32>
    %v1476 = stablehlo.reshape %v1475 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v1477 = stablehlo.reshape %v1476 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1478 = stablehlo.broadcast_in_dim %hng, dims = [2] : (tensor<768xf32>) -> tensor<32x1x768xf32>
    %v1479 = stablehlo.multiply %v1477, %v1478 : tensor<32x1x768xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v1481 = stablehlo.reshape %v1480 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1482 = stablehlo.broadcast_in_dim %hnbt, dims = [2] : (tensor<768xf32>) -> tensor<32x1x768xf32>
    %v1483 = stablehlo.add %v1481, %v1482 : tensor<32x1x768xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v1485 = stablehlo.dot_general %v1484, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x10xf32>) -> tensor<32x10xf32>
    %v1486 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1487 = stablehlo.add %v1485, %v1486 : tensor<32x10xf32>
    %v1488 = stablehlo.exponential %v1487 : tensor<32x10xf32>
    %v1489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1490 = stablehlo.reduce(%v1488 init: %v1489) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v1491 = stablehlo.broadcast_in_dim %v1490, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v1492 = stablehlo.divide %v1488, %v1491 : tensor<32x10xf32>
    %v1493 = stablehlo.subtract %v1492, %onehot : tensor<32x10xf32>
    %dy = stablehlo.divide %v1493, %bsc : tensor<32x10xf32>
    %v1494 = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<768x10xf32>) -> tensor<32x768xf32>
    %v1495 = stablehlo.reshape %v1494 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1496 = stablehlo.broadcast_in_dim %hng, dims = [2] : (tensor<768xf32>) -> tensor<32x1x768xf32>
    %v1497 = stablehlo.multiply %v1495, %v1496 : tensor<32x1x768xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v1499 = stablehlo.reshape %v1498 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1500 = stablehlo.reshape %v1456 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1502 = stablehlo.constant dense<768.0> : tensor<32x1x768xf32>
    %v1503 = stablehlo.constant dense<1.0e-6> : tensor<32x1x768xf32>
    %v1504 = stablehlo.reduce(%v1500 init: %v1501) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1505 = stablehlo.broadcast_in_dim %v1504, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v1506 = stablehlo.divide %v1505, %v1502 : tensor<32x1x768xf32>
    %v1507 = stablehlo.subtract %v1500, %v1506 : tensor<32x1x768xf32>
    %v1508 = stablehlo.multiply %v1507, %v1507 : tensor<32x1x768xf32>
    %v1509 = stablehlo.reduce(%v1508 init: %v1501) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1510 = stablehlo.broadcast_in_dim %v1509, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v1511 = stablehlo.divide %v1510, %v1502 : tensor<32x1x768xf32>
    %v1512 = stablehlo.add %v1511, %v1503 : tensor<32x1x768xf32>
    %v1513 = stablehlo.rsqrt %v1512 : tensor<32x1x768xf32>
    %v1514 = stablehlo.multiply %v1507, %v1513 : tensor<32x1x768xf32>
    %v1515 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x1x768xf32>
    %v1516 = stablehlo.multiply %v1515, %v1499 : tensor<32x1x768xf32>
    %v1517 = stablehlo.reduce(%v1516 init: %v1501) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1518 = stablehlo.broadcast_in_dim %v1517, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v1519 = stablehlo.multiply %v1514, %v1516 : tensor<32x1x768xf32>
    %v1520 = stablehlo.reduce(%v1519 init: %v1501) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1521 = stablehlo.broadcast_in_dim %v1520, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v1522 = stablehlo.multiply %v1516, %v1502 : tensor<32x1x768xf32>
    %v1523 = stablehlo.subtract %v1522, %v1518 : tensor<32x1x768xf32>
    %v1524 = stablehlo.multiply %v1514, %v1521 : tensor<32x1x768xf32>
    %v1525 = stablehlo.subtract %v1523, %v1524 : tensor<32x1x768xf32>
    %v1526 = stablehlo.divide %v1513, %v1502 : tensor<32x1x768xf32>
    %v1527 = stablehlo.multiply %v1526, %v1525 : tensor<32x1x768xf32>
    %v1528 = stablehlo.reshape %v1527 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v1529 = stablehlo.dot_general %v1484, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<32x10xf32>) -> tensor<768x10xf32>
    %v1530 = stablehlo.constant dense<0.1> : tensor<768x10xf32>
    %v1531 = stablehlo.multiply %v1529, %v1530 : tensor<768x10xf32>
    %v1532 = stablehlo.subtract %Wd, %v1531 : tensor<768x10xf32>
    %v1533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1534 = stablehlo.reduce(%dy init: %v1533) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1535 = stablehlo.constant dense<0.1> : tensor<10xf32>
    %v1536 = stablehlo.multiply %v1534, %v1535 : tensor<10xf32>
    %v1537 = stablehlo.subtract %bd, %v1536 : tensor<10xf32>
    %v1538 = stablehlo.reshape %v1456 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1539 = stablehlo.reshape %v1494 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1541 = stablehlo.constant dense<768.0> : tensor<32x1x768xf32>
    %v1542 = stablehlo.constant dense<1.0e-6> : tensor<32x1x768xf32>
    %v1543 = stablehlo.reduce(%v1538 init: %v1540) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1544 = stablehlo.broadcast_in_dim %v1543, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v1545 = stablehlo.divide %v1544, %v1541 : tensor<32x1x768xf32>
    %v1546 = stablehlo.subtract %v1538, %v1545 : tensor<32x1x768xf32>
    %v1547 = stablehlo.multiply %v1546, %v1546 : tensor<32x1x768xf32>
    %v1548 = stablehlo.reduce(%v1547 init: %v1540) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1549 = stablehlo.broadcast_in_dim %v1548, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v1550 = stablehlo.divide %v1549, %v1541 : tensor<32x1x768xf32>
    %v1551 = stablehlo.add %v1550, %v1542 : tensor<32x1x768xf32>
    %v1552 = stablehlo.rsqrt %v1551 : tensor<32x1x768xf32>
    %v1553 = stablehlo.multiply %v1546, %v1552 : tensor<32x1x768xf32>
    %v1554 = stablehlo.multiply %v1539, %v1553 : tensor<32x1x768xf32>
    %v1555 = stablehlo.reduce(%v1554 init: %v1540) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1556 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1557 = stablehlo.multiply %v1555, %v1556 : tensor<768xf32>
    %v1558 = stablehlo.subtract %hng, %v1557 : tensor<768xf32>
    %v1559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1560 = stablehlo.reshape %v1494 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1561 = stablehlo.reduce(%v1560 init: %v1559) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1562 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1563 = stablehlo.multiply %v1561, %v1562 : tensor<768xf32>
    %v1564 = stablehlo.subtract %hnbt, %v1563 : tensor<768xf32>
    %dgi = stablehlo.reshape %v1528 : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : (tensor<32x768x1x1xf32>) -> tensor<32x768x7x7xf32>
    %dgn = stablehlo.constant dense<49.0> : tensor<32x768x7x7xf32>
    %dgd = stablehlo.divide %dgb, %dgn : tensor<32x768x7x7xf32>
    %dgapf = stablehlo.reshape %dgd : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1565 = stablehlo.reshape %dgapf : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1566 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1567 = stablehlo.multiply %v1565, %v1566 : tensor<32x768x7x7xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1570 = stablehlo.transpose %s3b2pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1571 = stablehlo.reverse %v1570, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1572 = stablehlo.convolution(%v1569, %v1571)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1574 = stablehlo.reshape %v1573 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1575 = stablehlo.reshape %v1423 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1576 = stablehlo.multiply %v1575, %v1575 : tensor<32x3072x7x7xf32>
    %v1577 = stablehlo.multiply %v1576, %v1575 : tensor<32x3072x7x7xf32>
    %v1578 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1579 = stablehlo.multiply %v1578, %v1577 : tensor<32x3072x7x7xf32>
    %v1580 = stablehlo.add %v1575, %v1579 : tensor<32x3072x7x7xf32>
    %v1581 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1582 = stablehlo.multiply %v1581, %v1580 : tensor<32x3072x7x7xf32>
    %v1583 = stablehlo.tanh %v1582 : tensor<32x3072x7x7xf32>
    %v1584 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1585 = stablehlo.add %v1584, %v1583 : tensor<32x3072x7x7xf32>
    %v1586 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1587 = stablehlo.multiply %v1586, %v1585 : tensor<32x3072x7x7xf32>
    %v1588 = stablehlo.multiply %v1583, %v1583 : tensor<32x3072x7x7xf32>
    %v1589 = stablehlo.subtract %v1584, %v1588 : tensor<32x3072x7x7xf32>
    %v1590 = stablehlo.multiply %v1586, %v1575 : tensor<32x3072x7x7xf32>
    %v1591 = stablehlo.multiply %v1590, %v1589 : tensor<32x3072x7x7xf32>
    %v1592 = stablehlo.constant dense<0.134145> : tensor<32x3072x7x7xf32>
    %v1593 = stablehlo.multiply %v1592, %v1576 : tensor<32x3072x7x7xf32>
    %v1594 = stablehlo.add %v1584, %v1593 : tensor<32x3072x7x7xf32>
    %v1595 = stablehlo.multiply %v1581, %v1594 : tensor<32x3072x7x7xf32>
    %v1596 = stablehlo.multiply %v1591, %v1595 : tensor<32x3072x7x7xf32>
    %v1597 = stablehlo.add %v1587, %v1596 : tensor<32x3072x7x7xf32>
    %v1598 = stablehlo.multiply %v1574, %v1597 : tensor<32x3072x7x7xf32>
    %v1599 = stablehlo.reshape %v1598 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1600 = stablehlo.reshape %v1599 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1601 = stablehlo.transpose %s3b2eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1602 = stablehlo.reverse %v1601, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1603 = stablehlo.convolution(%v1600, %v1602)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1604 = stablehlo.reshape %v1603 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1605 = stablehlo.reshape %v1384 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1606 = stablehlo.transpose %v1605, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1608 = stablehlo.reshape %v1604 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1609 = stablehlo.transpose %v1608, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1610 = stablehlo.reshape %v1609 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1611 = stablehlo.reshape %v1610 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1612 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1613 = stablehlo.multiply %v1611, %v1612 : tensor<32x49x768xf32>
    %v1614 = stablehlo.reshape %v1613 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1615 = stablehlo.reshape %v1614 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1616 = stablehlo.reshape %v1607 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1618 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1619 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1620 = stablehlo.reduce(%v1616 init: %v1617) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1621 = stablehlo.broadcast_in_dim %v1620, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1622 = stablehlo.divide %v1621, %v1618 : tensor<32x49x768xf32>
    %v1623 = stablehlo.subtract %v1616, %v1622 : tensor<32x49x768xf32>
    %v1624 = stablehlo.multiply %v1623, %v1623 : tensor<32x49x768xf32>
    %v1625 = stablehlo.reduce(%v1624 init: %v1617) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1626 = stablehlo.broadcast_in_dim %v1625, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1627 = stablehlo.divide %v1626, %v1618 : tensor<32x49x768xf32>
    %v1628 = stablehlo.add %v1627, %v1619 : tensor<32x49x768xf32>
    %v1629 = stablehlo.rsqrt %v1628 : tensor<32x49x768xf32>
    %v1630 = stablehlo.multiply %v1623, %v1629 : tensor<32x49x768xf32>
    %v1631 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1632 = stablehlo.multiply %v1631, %v1615 : tensor<32x49x768xf32>
    %v1633 = stablehlo.reduce(%v1632 init: %v1617) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1634 = stablehlo.broadcast_in_dim %v1633, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1635 = stablehlo.multiply %v1630, %v1632 : tensor<32x49x768xf32>
    %v1636 = stablehlo.reduce(%v1635 init: %v1617) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1637 = stablehlo.broadcast_in_dim %v1636, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1638 = stablehlo.multiply %v1632, %v1618 : tensor<32x49x768xf32>
    %v1639 = stablehlo.subtract %v1638, %v1634 : tensor<32x49x768xf32>
    %v1640 = stablehlo.multiply %v1630, %v1637 : tensor<32x49x768xf32>
    %v1641 = stablehlo.subtract %v1639, %v1640 : tensor<32x49x768xf32>
    %v1642 = stablehlo.divide %v1629, %v1618 : tensor<32x49x768xf32>
    %v1643 = stablehlo.multiply %v1642, %v1641 : tensor<32x49x768xf32>
    %v1644 = stablehlo.reshape %v1643 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1645 = stablehlo.reshape %v1644 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1646 = stablehlo.transpose %v1645, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1647 = stablehlo.reshape %v1646 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1649 = stablehlo.reverse %s3b2dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1650 = stablehlo.convolution(%v1648, %v1649)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1651 = stablehlo.reshape %v1650 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1652 = stablehlo.reshape %v1651 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1653 = stablehlo.reshape %dgapf : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1654 = stablehlo.add %v1652, %v1653 : tensor<32x768x7x7xf32>
    %v1655 = stablehlo.reshape %v1654 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1657 = stablehlo.reshape %v1443 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1658 = stablehlo.reshape %dgapf : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1659 = stablehlo.multiply %v1657, %v1658 : tensor<32x768x7x7xf32>
    %v1660 = stablehlo.reduce(%v1659 init: %v1656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1661 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1662 = stablehlo.multiply %v1660, %v1661 : tensor<768xf32>
    %v1663 = stablehlo.subtract %s3b2lg, %v1662 : tensor<768xf32>
    %v1664 = stablehlo.reshape %v1438 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1665 = stablehlo.reshape %v1568 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1666 = stablehlo.transpose %v1664, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1667 = stablehlo.transpose %v1665, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1668 = stablehlo.convolution(%v1666, %v1667)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1669 = stablehlo.transpose %v1668, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1670 = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %v1671 = stablehlo.multiply %v1669, %v1670 : tensor<768x3072x1x1xf32>
    %v1672 = stablehlo.subtract %s3b2pW, %v1671 : tensor<768x3072x1x1xf32>
    %v1673 = stablehlo.reshape %v1568 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1675 = stablehlo.reduce(%v1673 init: %v1674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1676 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1677 = stablehlo.multiply %v1675, %v1676 : tensor<768xf32>
    %v1678 = stablehlo.subtract %s3b2pb, %v1677 : tensor<768xf32>
    %v1679 = stablehlo.reshape %v1418 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1680 = stablehlo.reshape %v1599 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1681 = stablehlo.transpose %v1679, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1682 = stablehlo.transpose %v1680, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1683 = stablehlo.convolution(%v1681, %v1682)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1684 = stablehlo.transpose %v1683, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1685 = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %v1686 = stablehlo.multiply %v1684, %v1685 : tensor<3072x768x1x1xf32>
    %v1687 = stablehlo.subtract %s3b2eW, %v1686 : tensor<3072x768x1x1xf32>
    %v1688 = stablehlo.reshape %v1599 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1690 = stablehlo.reduce(%v1688 init: %v1689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1691 = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %v1692 = stablehlo.multiply %v1690, %v1691 : tensor<3072xf32>
    %v1693 = stablehlo.subtract %s3b2eb, %v1692 : tensor<3072xf32>
    %v1694 = stablehlo.reshape %v1384 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1695 = stablehlo.transpose %v1694, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1696 = stablehlo.reshape %v1695 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1697 = stablehlo.reshape %v1604 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1698 = stablehlo.transpose %v1697, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1699 = stablehlo.reshape %v1698 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1700 = stablehlo.reshape %v1696 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1701 = stablehlo.reshape %v1699 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1703 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1704 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1705 = stablehlo.reduce(%v1700 init: %v1702) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1706 = stablehlo.broadcast_in_dim %v1705, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1707 = stablehlo.divide %v1706, %v1703 : tensor<32x49x768xf32>
    %v1708 = stablehlo.subtract %v1700, %v1707 : tensor<32x49x768xf32>
    %v1709 = stablehlo.multiply %v1708, %v1708 : tensor<32x49x768xf32>
    %v1710 = stablehlo.reduce(%v1709 init: %v1702) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1711 = stablehlo.broadcast_in_dim %v1710, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1712 = stablehlo.divide %v1711, %v1703 : tensor<32x49x768xf32>
    %v1713 = stablehlo.add %v1712, %v1704 : tensor<32x49x768xf32>
    %v1714 = stablehlo.rsqrt %v1713 : tensor<32x49x768xf32>
    %v1715 = stablehlo.multiply %v1708, %v1714 : tensor<32x49x768xf32>
    %v1716 = stablehlo.multiply %v1701, %v1715 : tensor<32x49x768xf32>
    %v1717 = stablehlo.reduce(%v1716 init: %v1702) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1718 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1719 = stablehlo.multiply %v1717, %v1718 : tensor<768xf32>
    %v1720 = stablehlo.subtract %s3b2ng, %v1719 : tensor<768xf32>
    %v1721 = stablehlo.reshape %v1604 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1722 = stablehlo.transpose %v1721, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1723 = stablehlo.reshape %v1722 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1725 = stablehlo.reshape %v1723 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1726 = stablehlo.reduce(%v1725 init: %v1724) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1727 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1728 = stablehlo.multiply %v1726, %v1727 : tensor<768xf32>
    %v1729 = stablehlo.subtract %s3b2nbt, %v1728 : tensor<768xf32>
    %v1730 = stablehlo.reshape %v1379 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1731 = stablehlo.reshape %v1647 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1732 = stablehlo.transpose %v1730, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1733 = stablehlo.transpose %v1731, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1734 = stablehlo.convolution(%v1732, %v1733)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1735 = stablehlo.reshape %v1734 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1736 = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %v1737 = stablehlo.multiply %v1735, %v1736 : tensor<768x1x7x7xf32>
    %v1738 = stablehlo.subtract %s3b2dW, %v1737 : tensor<768x1x7x7xf32>
    %v1739 = stablehlo.reshape %v1647 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1740 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1741 = stablehlo.reduce(%v1739 init: %v1740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1742 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1743 = stablehlo.multiply %v1741, %v1742 : tensor<768xf32>
    %v1744 = stablehlo.subtract %s3b2db, %v1743 : tensor<768xf32>
    %v1745 = stablehlo.reshape %v1655 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1746 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1747 = stablehlo.multiply %v1745, %v1746 : tensor<32x768x7x7xf32>
    %v1748 = stablehlo.reshape %v1747 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1749 = stablehlo.reshape %v1748 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1750 = stablehlo.transpose %s3b1pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1751 = stablehlo.reverse %v1750, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1752 = stablehlo.convolution(%v1749, %v1751)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1754 = stablehlo.reshape %v1753 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1755 = stablehlo.reshape %v1351 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1756 = stablehlo.multiply %v1755, %v1755 : tensor<32x3072x7x7xf32>
    %v1757 = stablehlo.multiply %v1756, %v1755 : tensor<32x3072x7x7xf32>
    %v1758 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1759 = stablehlo.multiply %v1758, %v1757 : tensor<32x3072x7x7xf32>
    %v1760 = stablehlo.add %v1755, %v1759 : tensor<32x3072x7x7xf32>
    %v1761 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1762 = stablehlo.multiply %v1761, %v1760 : tensor<32x3072x7x7xf32>
    %v1763 = stablehlo.tanh %v1762 : tensor<32x3072x7x7xf32>
    %v1764 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1765 = stablehlo.add %v1764, %v1763 : tensor<32x3072x7x7xf32>
    %v1766 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1767 = stablehlo.multiply %v1766, %v1765 : tensor<32x3072x7x7xf32>
    %v1768 = stablehlo.multiply %v1763, %v1763 : tensor<32x3072x7x7xf32>
    %v1769 = stablehlo.subtract %v1764, %v1768 : tensor<32x3072x7x7xf32>
    %v1770 = stablehlo.multiply %v1766, %v1755 : tensor<32x3072x7x7xf32>
    %v1771 = stablehlo.multiply %v1770, %v1769 : tensor<32x3072x7x7xf32>
    %v1772 = stablehlo.constant dense<0.134145> : tensor<32x3072x7x7xf32>
    %v1773 = stablehlo.multiply %v1772, %v1756 : tensor<32x3072x7x7xf32>
    %v1774 = stablehlo.add %v1764, %v1773 : tensor<32x3072x7x7xf32>
    %v1775 = stablehlo.multiply %v1761, %v1774 : tensor<32x3072x7x7xf32>
    %v1776 = stablehlo.multiply %v1771, %v1775 : tensor<32x3072x7x7xf32>
    %v1777 = stablehlo.add %v1767, %v1776 : tensor<32x3072x7x7xf32>
    %v1778 = stablehlo.multiply %v1754, %v1777 : tensor<32x3072x7x7xf32>
    %v1779 = stablehlo.reshape %v1778 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1781 = stablehlo.transpose %s3b1eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1782 = stablehlo.reverse %v1781, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1783 = stablehlo.convolution(%v1780, %v1782)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1784 = stablehlo.reshape %v1783 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1785 = stablehlo.reshape %v1312 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1786 = stablehlo.transpose %v1785, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1787 = stablehlo.reshape %v1786 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1788 = stablehlo.reshape %v1784 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1789 = stablehlo.transpose %v1788, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1790 = stablehlo.reshape %v1789 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1791 = stablehlo.reshape %v1790 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1792 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1793 = stablehlo.multiply %v1791, %v1792 : tensor<32x49x768xf32>
    %v1794 = stablehlo.reshape %v1793 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1795 = stablehlo.reshape %v1794 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1796 = stablehlo.reshape %v1787 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1798 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1799 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1800 = stablehlo.reduce(%v1796 init: %v1797) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1801 = stablehlo.broadcast_in_dim %v1800, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1802 = stablehlo.divide %v1801, %v1798 : tensor<32x49x768xf32>
    %v1803 = stablehlo.subtract %v1796, %v1802 : tensor<32x49x768xf32>
    %v1804 = stablehlo.multiply %v1803, %v1803 : tensor<32x49x768xf32>
    %v1805 = stablehlo.reduce(%v1804 init: %v1797) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1806 = stablehlo.broadcast_in_dim %v1805, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1807 = stablehlo.divide %v1806, %v1798 : tensor<32x49x768xf32>
    %v1808 = stablehlo.add %v1807, %v1799 : tensor<32x49x768xf32>
    %v1809 = stablehlo.rsqrt %v1808 : tensor<32x49x768xf32>
    %v1810 = stablehlo.multiply %v1803, %v1809 : tensor<32x49x768xf32>
    %v1811 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1812 = stablehlo.multiply %v1811, %v1795 : tensor<32x49x768xf32>
    %v1813 = stablehlo.reduce(%v1812 init: %v1797) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1814 = stablehlo.broadcast_in_dim %v1813, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1815 = stablehlo.multiply %v1810, %v1812 : tensor<32x49x768xf32>
    %v1816 = stablehlo.reduce(%v1815 init: %v1797) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1817 = stablehlo.broadcast_in_dim %v1816, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1818 = stablehlo.multiply %v1812, %v1798 : tensor<32x49x768xf32>
    %v1819 = stablehlo.subtract %v1818, %v1814 : tensor<32x49x768xf32>
    %v1820 = stablehlo.multiply %v1810, %v1817 : tensor<32x49x768xf32>
    %v1821 = stablehlo.subtract %v1819, %v1820 : tensor<32x49x768xf32>
    %v1822 = stablehlo.divide %v1809, %v1798 : tensor<32x49x768xf32>
    %v1823 = stablehlo.multiply %v1822, %v1821 : tensor<32x49x768xf32>
    %v1824 = stablehlo.reshape %v1823 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1825 = stablehlo.reshape %v1824 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1826 = stablehlo.transpose %v1825, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1827 = stablehlo.reshape %v1826 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1828 = stablehlo.reshape %v1827 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1829 = stablehlo.reverse %s3b1dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1830 = stablehlo.convolution(%v1828, %v1829)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1831 = stablehlo.reshape %v1830 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1832 = stablehlo.reshape %v1831 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1833 = stablehlo.reshape %v1655 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1834 = stablehlo.add %v1832, %v1833 : tensor<32x768x7x7xf32>
    %v1835 = stablehlo.reshape %v1834 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1836 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1837 = stablehlo.reshape %v1371 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1838 = stablehlo.reshape %v1655 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1839 = stablehlo.multiply %v1837, %v1838 : tensor<32x768x7x7xf32>
    %v1840 = stablehlo.reduce(%v1839 init: %v1836) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1841 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1842 = stablehlo.multiply %v1840, %v1841 : tensor<768xf32>
    %v1843 = stablehlo.subtract %s3b1lg, %v1842 : tensor<768xf32>
    %v1844 = stablehlo.reshape %v1366 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1845 = stablehlo.reshape %v1748 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1846 = stablehlo.transpose %v1844, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1847 = stablehlo.transpose %v1845, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1848 = stablehlo.convolution(%v1846, %v1847)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1849 = stablehlo.transpose %v1848, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1850 = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %v1851 = stablehlo.multiply %v1849, %v1850 : tensor<768x3072x1x1xf32>
    %v1852 = stablehlo.subtract %s3b1pW, %v1851 : tensor<768x3072x1x1xf32>
    %v1853 = stablehlo.reshape %v1748 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1855 = stablehlo.reduce(%v1853 init: %v1854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1856 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1857 = stablehlo.multiply %v1855, %v1856 : tensor<768xf32>
    %v1858 = stablehlo.subtract %s3b1pb, %v1857 : tensor<768xf32>
    %v1859 = stablehlo.reshape %v1346 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1860 = stablehlo.reshape %v1779 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1861 = stablehlo.transpose %v1859, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1862 = stablehlo.transpose %v1860, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1863 = stablehlo.convolution(%v1861, %v1862)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1864 = stablehlo.transpose %v1863, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1865 = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %v1866 = stablehlo.multiply %v1864, %v1865 : tensor<3072x768x1x1xf32>
    %v1867 = stablehlo.subtract %s3b1eW, %v1866 : tensor<3072x768x1x1xf32>
    %v1868 = stablehlo.reshape %v1779 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1870 = stablehlo.reduce(%v1868 init: %v1869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1871 = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %v1872 = stablehlo.multiply %v1870, %v1871 : tensor<3072xf32>
    %v1873 = stablehlo.subtract %s3b1eb, %v1872 : tensor<3072xf32>
    %v1874 = stablehlo.reshape %v1312 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1875 = stablehlo.transpose %v1874, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1876 = stablehlo.reshape %v1875 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1877 = stablehlo.reshape %v1784 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1878 = stablehlo.transpose %v1877, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1879 = stablehlo.reshape %v1878 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1880 = stablehlo.reshape %v1876 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1881 = stablehlo.reshape %v1879 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1883 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1884 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1885 = stablehlo.reduce(%v1880 init: %v1882) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1886 = stablehlo.broadcast_in_dim %v1885, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1887 = stablehlo.divide %v1886, %v1883 : tensor<32x49x768xf32>
    %v1888 = stablehlo.subtract %v1880, %v1887 : tensor<32x49x768xf32>
    %v1889 = stablehlo.multiply %v1888, %v1888 : tensor<32x49x768xf32>
    %v1890 = stablehlo.reduce(%v1889 init: %v1882) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1891 = stablehlo.broadcast_in_dim %v1890, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1892 = stablehlo.divide %v1891, %v1883 : tensor<32x49x768xf32>
    %v1893 = stablehlo.add %v1892, %v1884 : tensor<32x49x768xf32>
    %v1894 = stablehlo.rsqrt %v1893 : tensor<32x49x768xf32>
    %v1895 = stablehlo.multiply %v1888, %v1894 : tensor<32x49x768xf32>
    %v1896 = stablehlo.multiply %v1881, %v1895 : tensor<32x49x768xf32>
    %v1897 = stablehlo.reduce(%v1896 init: %v1882) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1898 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1899 = stablehlo.multiply %v1897, %v1898 : tensor<768xf32>
    %v1900 = stablehlo.subtract %s3b1ng, %v1899 : tensor<768xf32>
    %v1901 = stablehlo.reshape %v1784 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1902 = stablehlo.transpose %v1901, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1903 = stablehlo.reshape %v1902 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1904 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1905 = stablehlo.reshape %v1903 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1906 = stablehlo.reduce(%v1905 init: %v1904) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1907 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1908 = stablehlo.multiply %v1906, %v1907 : tensor<768xf32>
    %v1909 = stablehlo.subtract %s3b1nbt, %v1908 : tensor<768xf32>
    %v1910 = stablehlo.reshape %v1307 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1911 = stablehlo.reshape %v1827 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1912 = stablehlo.transpose %v1910, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1913 = stablehlo.transpose %v1911, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1914 = stablehlo.convolution(%v1912, %v1913)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1915 = stablehlo.reshape %v1914 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1916 = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %v1917 = stablehlo.multiply %v1915, %v1916 : tensor<768x1x7x7xf32>
    %v1918 = stablehlo.subtract %s3b1dW, %v1917 : tensor<768x1x7x7xf32>
    %v1919 = stablehlo.reshape %v1827 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1920 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1921 = stablehlo.reduce(%v1919 init: %v1920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1922 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1923 = stablehlo.multiply %v1921, %v1922 : tensor<768xf32>
    %v1924 = stablehlo.subtract %s3b1db, %v1923 : tensor<768xf32>
    %v1925 = stablehlo.reshape %v1835 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1926 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1927 = stablehlo.multiply %v1925, %v1926 : tensor<32x768x7x7xf32>
    %v1928 = stablehlo.reshape %v1927 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1929 = stablehlo.reshape %v1928 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1930 = stablehlo.transpose %s3b0pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1931 = stablehlo.reverse %v1930, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1932 = stablehlo.convolution(%v1929, %v1931)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1933 = stablehlo.reshape %v1932 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1934 = stablehlo.reshape %v1933 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1935 = stablehlo.reshape %v1279 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1936 = stablehlo.multiply %v1935, %v1935 : tensor<32x3072x7x7xf32>
    %v1937 = stablehlo.multiply %v1936, %v1935 : tensor<32x3072x7x7xf32>
    %v1938 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1939 = stablehlo.multiply %v1938, %v1937 : tensor<32x3072x7x7xf32>
    %v1940 = stablehlo.add %v1935, %v1939 : tensor<32x3072x7x7xf32>
    %v1941 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1942 = stablehlo.multiply %v1941, %v1940 : tensor<32x3072x7x7xf32>
    %v1943 = stablehlo.tanh %v1942 : tensor<32x3072x7x7xf32>
    %v1944 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1945 = stablehlo.add %v1944, %v1943 : tensor<32x3072x7x7xf32>
    %v1946 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1947 = stablehlo.multiply %v1946, %v1945 : tensor<32x3072x7x7xf32>
    %v1948 = stablehlo.multiply %v1943, %v1943 : tensor<32x3072x7x7xf32>
    %v1949 = stablehlo.subtract %v1944, %v1948 : tensor<32x3072x7x7xf32>
    %v1950 = stablehlo.multiply %v1946, %v1935 : tensor<32x3072x7x7xf32>
    %v1951 = stablehlo.multiply %v1950, %v1949 : tensor<32x3072x7x7xf32>
    %v1952 = stablehlo.constant dense<0.134145> : tensor<32x3072x7x7xf32>
    %v1953 = stablehlo.multiply %v1952, %v1936 : tensor<32x3072x7x7xf32>
    %v1954 = stablehlo.add %v1944, %v1953 : tensor<32x3072x7x7xf32>
    %v1955 = stablehlo.multiply %v1941, %v1954 : tensor<32x3072x7x7xf32>
    %v1956 = stablehlo.multiply %v1951, %v1955 : tensor<32x3072x7x7xf32>
    %v1957 = stablehlo.add %v1947, %v1956 : tensor<32x3072x7x7xf32>
    %v1958 = stablehlo.multiply %v1934, %v1957 : tensor<32x3072x7x7xf32>
    %v1959 = stablehlo.reshape %v1958 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1960 = stablehlo.reshape %v1959 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1961 = stablehlo.transpose %s3b0eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1962 = stablehlo.reverse %v1961, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1963 = stablehlo.convolution(%v1960, %v1962)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1964 = stablehlo.reshape %v1963 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1965 = stablehlo.reshape %v1240 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1966 = stablehlo.transpose %v1965, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1967 = stablehlo.reshape %v1966 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1968 = stablehlo.reshape %v1964 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1969 = stablehlo.transpose %v1968, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1970 = stablehlo.reshape %v1969 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1971 = stablehlo.reshape %v1970 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1972 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1973 = stablehlo.multiply %v1971, %v1972 : tensor<32x49x768xf32>
    %v1974 = stablehlo.reshape %v1973 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1975 = stablehlo.reshape %v1974 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1976 = stablehlo.reshape %v1967 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1978 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1979 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1980 = stablehlo.reduce(%v1976 init: %v1977) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1981 = stablehlo.broadcast_in_dim %v1980, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1982 = stablehlo.divide %v1981, %v1978 : tensor<32x49x768xf32>
    %v1983 = stablehlo.subtract %v1976, %v1982 : tensor<32x49x768xf32>
    %v1984 = stablehlo.multiply %v1983, %v1983 : tensor<32x49x768xf32>
    %v1985 = stablehlo.reduce(%v1984 init: %v1977) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1986 = stablehlo.broadcast_in_dim %v1985, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1987 = stablehlo.divide %v1986, %v1978 : tensor<32x49x768xf32>
    %v1988 = stablehlo.add %v1987, %v1979 : tensor<32x49x768xf32>
    %v1989 = stablehlo.rsqrt %v1988 : tensor<32x49x768xf32>
    %v1990 = stablehlo.multiply %v1983, %v1989 : tensor<32x49x768xf32>
    %v1991 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1992 = stablehlo.multiply %v1991, %v1975 : tensor<32x49x768xf32>
    %v1993 = stablehlo.reduce(%v1992 init: %v1977) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1994 = stablehlo.broadcast_in_dim %v1993, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1995 = stablehlo.multiply %v1990, %v1992 : tensor<32x49x768xf32>
    %v1996 = stablehlo.reduce(%v1995 init: %v1977) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1997 = stablehlo.broadcast_in_dim %v1996, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1998 = stablehlo.multiply %v1992, %v1978 : tensor<32x49x768xf32>
    %v1999 = stablehlo.subtract %v1998, %v1994 : tensor<32x49x768xf32>
    %v2000 = stablehlo.multiply %v1990, %v1997 : tensor<32x49x768xf32>
    %v2001 = stablehlo.subtract %v1999, %v2000 : tensor<32x49x768xf32>
    %v2002 = stablehlo.divide %v1989, %v1978 : tensor<32x49x768xf32>
    %v2003 = stablehlo.multiply %v2002, %v2001 : tensor<32x49x768xf32>
    %v2004 = stablehlo.reshape %v2003 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2005 = stablehlo.reshape %v2004 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2006 = stablehlo.transpose %v2005, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v2007 = stablehlo.reshape %v2006 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v2008 = stablehlo.reshape %v2007 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2009 = stablehlo.reverse %s3b0dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v2010 = stablehlo.convolution(%v2008, %v2009)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v2011 = stablehlo.reshape %v2010 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2012 = stablehlo.reshape %v2011 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2013 = stablehlo.reshape %v1835 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2014 = stablehlo.add %v2012, %v2013 : tensor<32x768x7x7xf32>
    %v2015 = stablehlo.reshape %v2014 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v2016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2017 = stablehlo.reshape %v1299 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2018 = stablehlo.reshape %v1835 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2019 = stablehlo.multiply %v2017, %v2018 : tensor<32x768x7x7xf32>
    %v2020 = stablehlo.reduce(%v2019 init: %v2016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v2021 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v2022 = stablehlo.multiply %v2020, %v2021 : tensor<768xf32>
    %v2023 = stablehlo.subtract %s3b0lg, %v2022 : tensor<768xf32>
    %v2024 = stablehlo.reshape %v1294 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v2025 = stablehlo.reshape %v1928 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2026 = stablehlo.transpose %v2024, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v2027 = stablehlo.transpose %v2025, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v2028 = stablehlo.convolution(%v2026, %v2027)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v2029 = stablehlo.transpose %v2028, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v2030 = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %v2031 = stablehlo.multiply %v2029, %v2030 : tensor<768x3072x1x1xf32>
    %v2032 = stablehlo.subtract %s3b0pW, %v2031 : tensor<768x3072x1x1xf32>
    %v2033 = stablehlo.reshape %v1928 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2035 = stablehlo.reduce(%v2033 init: %v2034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v2036 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v2037 = stablehlo.multiply %v2035, %v2036 : tensor<768xf32>
    %v2038 = stablehlo.subtract %s3b0pb, %v2037 : tensor<768xf32>
    %v2039 = stablehlo.reshape %v1274 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2040 = stablehlo.reshape %v1959 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v2041 = stablehlo.transpose %v2039, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v2042 = stablehlo.transpose %v2040, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v2043 = stablehlo.convolution(%v2041, %v2042)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v2044 = stablehlo.transpose %v2043, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v2045 = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %v2046 = stablehlo.multiply %v2044, %v2045 : tensor<3072x768x1x1xf32>
    %v2047 = stablehlo.subtract %s3b0eW, %v2046 : tensor<3072x768x1x1xf32>
    %v2048 = stablehlo.reshape %v1959 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v2049 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2050 = stablehlo.reduce(%v2048 init: %v2049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v2051 = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %v2052 = stablehlo.multiply %v2050, %v2051 : tensor<3072xf32>
    %v2053 = stablehlo.subtract %s3b0eb, %v2052 : tensor<3072xf32>
    %v2054 = stablehlo.reshape %v1240 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v2055 = stablehlo.transpose %v2054, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v2056 = stablehlo.reshape %v2055 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2057 = stablehlo.reshape %v1964 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v2058 = stablehlo.transpose %v2057, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v2059 = stablehlo.reshape %v2058 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2060 = stablehlo.reshape %v2056 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2061 = stablehlo.reshape %v2059 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2063 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v2064 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v2065 = stablehlo.reduce(%v2060 init: %v2062) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2066 = stablehlo.broadcast_in_dim %v2065, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2067 = stablehlo.divide %v2066, %v2063 : tensor<32x49x768xf32>
    %v2068 = stablehlo.subtract %v2060, %v2067 : tensor<32x49x768xf32>
    %v2069 = stablehlo.multiply %v2068, %v2068 : tensor<32x49x768xf32>
    %v2070 = stablehlo.reduce(%v2069 init: %v2062) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2071 = stablehlo.broadcast_in_dim %v2070, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v2072 = stablehlo.divide %v2071, %v2063 : tensor<32x49x768xf32>
    %v2073 = stablehlo.add %v2072, %v2064 : tensor<32x49x768xf32>
    %v2074 = stablehlo.rsqrt %v2073 : tensor<32x49x768xf32>
    %v2075 = stablehlo.multiply %v2068, %v2074 : tensor<32x49x768xf32>
    %v2076 = stablehlo.multiply %v2061, %v2075 : tensor<32x49x768xf32>
    %v2077 = stablehlo.reduce(%v2076 init: %v2062) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v2078 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v2079 = stablehlo.multiply %v2077, %v2078 : tensor<768xf32>
    %v2080 = stablehlo.subtract %s3b0ng, %v2079 : tensor<768xf32>
    %v2081 = stablehlo.reshape %v1964 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v2082 = stablehlo.transpose %v2081, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v2083 = stablehlo.reshape %v2082 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v2084 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2085 = stablehlo.reshape %v2083 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v2086 = stablehlo.reduce(%v2085 init: %v2084) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v2087 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v2088 = stablehlo.multiply %v2086, %v2087 : tensor<768xf32>
    %v2089 = stablehlo.subtract %s3b0nbt, %v2088 : tensor<768xf32>
    %v2090 = stablehlo.reshape %v1235 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2091 = stablehlo.reshape %v2007 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2092 = stablehlo.transpose %v2090, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v2093 = stablehlo.transpose %v2091, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v2094 = stablehlo.convolution(%v2092, %v2093)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v2095 = stablehlo.reshape %v2094 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v2096 = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %v2097 = stablehlo.multiply %v2095, %v2096 : tensor<768x1x7x7xf32>
    %v2098 = stablehlo.subtract %s3b0dW, %v2097 : tensor<768x1x7x7xf32>
    %v2099 = stablehlo.reshape %v2007 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2100 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2101 = stablehlo.reduce(%v2099 init: %v2100) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v2102 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v2103 = stablehlo.multiply %v2101, %v2102 : tensor<768xf32>
    %v2104 = stablehlo.subtract %s3b0db, %v2103 : tensor<768xf32>
    %v2105 = stablehlo.reshape %v2015 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2107 = stablehlo.pad %v2105, %v2106, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x14x14xf32>
    %v2108 = stablehlo.transpose %d2W, dims = [1, 0, 2, 3] : (tensor<768x384x2x2xf32>) -> tensor<384x768x2x2xf32>
    %v2109 = stablehlo.reverse %v2108, dims = [2, 3] : tensor<384x768x2x2xf32>
    %v2110 = stablehlo.convolution(%v2107, %v2109)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x14x14xf32>, tensor<384x768x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v2111 = stablehlo.reshape %v2110 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2112 = stablehlo.reshape %v1196 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2113 = stablehlo.transpose %v2112, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2114 = stablehlo.reshape %v2113 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2115 = stablehlo.reshape %v2111 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2116 = stablehlo.transpose %v2115, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2117 = stablehlo.reshape %v2116 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2118 = stablehlo.reshape %v2117 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2119 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2120 = stablehlo.multiply %v2118, %v2119 : tensor<32x196x384xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2122 = stablehlo.reshape %v2121 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2123 = stablehlo.reshape %v2114 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2125 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2126 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2127 = stablehlo.reduce(%v2123 init: %v2124) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2128 = stablehlo.broadcast_in_dim %v2127, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2129 = stablehlo.divide %v2128, %v2125 : tensor<32x196x384xf32>
    %v2130 = stablehlo.subtract %v2123, %v2129 : tensor<32x196x384xf32>
    %v2131 = stablehlo.multiply %v2130, %v2130 : tensor<32x196x384xf32>
    %v2132 = stablehlo.reduce(%v2131 init: %v2124) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2133 = stablehlo.broadcast_in_dim %v2132, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2134 = stablehlo.divide %v2133, %v2125 : tensor<32x196x384xf32>
    %v2135 = stablehlo.add %v2134, %v2126 : tensor<32x196x384xf32>
    %v2136 = stablehlo.rsqrt %v2135 : tensor<32x196x384xf32>
    %v2137 = stablehlo.multiply %v2130, %v2136 : tensor<32x196x384xf32>
    %v2138 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2139 = stablehlo.multiply %v2138, %v2122 : tensor<32x196x384xf32>
    %v2140 = stablehlo.reduce(%v2139 init: %v2124) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2141 = stablehlo.broadcast_in_dim %v2140, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2142 = stablehlo.multiply %v2137, %v2139 : tensor<32x196x384xf32>
    %v2143 = stablehlo.reduce(%v2142 init: %v2124) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2144 = stablehlo.broadcast_in_dim %v2143, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2145 = stablehlo.multiply %v2139, %v2125 : tensor<32x196x384xf32>
    %v2146 = stablehlo.subtract %v2145, %v2141 : tensor<32x196x384xf32>
    %v2147 = stablehlo.multiply %v2137, %v2144 : tensor<32x196x384xf32>
    %v2148 = stablehlo.subtract %v2146, %v2147 : tensor<32x196x384xf32>
    %v2149 = stablehlo.divide %v2136, %v2125 : tensor<32x196x384xf32>
    %v2150 = stablehlo.multiply %v2149, %v2148 : tensor<32x196x384xf32>
    %v2151 = stablehlo.reshape %v2150 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2152 = stablehlo.reshape %v2151 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2153 = stablehlo.transpose %v2152, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2155 = stablehlo.reshape %v2015 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2157 = stablehlo.reduce(%v2155 init: %v2156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v2158 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v2159 = stablehlo.multiply %v2157, %v2158 : tensor<768xf32>
    %v2160 = stablehlo.subtract %d2b, %v2159 : tensor<768xf32>
    %v2161 = stablehlo.reshape %v1196 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2162 = stablehlo.transpose %v2161, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2163 = stablehlo.reshape %v2162 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2164 = stablehlo.reshape %v2111 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2165 = stablehlo.transpose %v2164, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2166 = stablehlo.reshape %v2165 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2167 = stablehlo.reshape %v2163 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2168 = stablehlo.reshape %v2166 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2169 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2170 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2171 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2172 = stablehlo.reduce(%v2167 init: %v2169) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2173 = stablehlo.broadcast_in_dim %v2172, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2174 = stablehlo.divide %v2173, %v2170 : tensor<32x196x384xf32>
    %v2175 = stablehlo.subtract %v2167, %v2174 : tensor<32x196x384xf32>
    %v2176 = stablehlo.multiply %v2175, %v2175 : tensor<32x196x384xf32>
    %v2177 = stablehlo.reduce(%v2176 init: %v2169) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2178 = stablehlo.broadcast_in_dim %v2177, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2179 = stablehlo.divide %v2178, %v2170 : tensor<32x196x384xf32>
    %v2180 = stablehlo.add %v2179, %v2171 : tensor<32x196x384xf32>
    %v2181 = stablehlo.rsqrt %v2180 : tensor<32x196x384xf32>
    %v2182 = stablehlo.multiply %v2175, %v2181 : tensor<32x196x384xf32>
    %v2183 = stablehlo.multiply %v2168, %v2182 : tensor<32x196x384xf32>
    %v2184 = stablehlo.reduce(%v2183 init: %v2169) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2185 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2186 = stablehlo.multiply %v2184, %v2185 : tensor<384xf32>
    %v2187 = stablehlo.subtract %d2ng, %v2186 : tensor<384xf32>
    %v2188 = stablehlo.reshape %v2111 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2189 = stablehlo.transpose %v2188, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2190 = stablehlo.reshape %v2189 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2191 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2192 = stablehlo.reshape %v2190 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2193 = stablehlo.reduce(%v2192 init: %v2191) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2194 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2195 = stablehlo.multiply %v2193, %v2194 : tensor<384xf32>
    %v2196 = stablehlo.subtract %d2nbt, %v2195 : tensor<384xf32>
    %v2197 = stablehlo.reshape %v1230 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2198 = stablehlo.reshape %v2015 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2200 = stablehlo.pad %v2198, %v2199, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x13x13xf32>
    %v2201 = stablehlo.transpose %v2197, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2202 = stablehlo.transpose %v2200, dims = [1, 0, 2, 3] : (tensor<32x768x13x13xf32>) -> tensor<768x32x13x13xf32>
    %v2203 = stablehlo.convolution(%v2201, %v2202)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<768x32x13x13xf32>) -> tensor<384x768x2x2xf32>
    %v2204 = stablehlo.transpose %v2203, dims = [1, 0, 2, 3] : (tensor<384x768x2x2xf32>) -> tensor<768x384x2x2xf32>
    %v2205 = stablehlo.constant dense<0.1> : tensor<768x384x2x2xf32>
    %v2206 = stablehlo.multiply %v2204, %v2205 : tensor<768x384x2x2xf32>
    %v2207 = stablehlo.subtract %d2W, %v2206 : tensor<768x384x2x2xf32>
    %v2208 = stablehlo.reshape %v2154 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2209 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2210 = stablehlo.multiply %v2208, %v2209 : tensor<32x384x14x14xf32>
    %v2211 = stablehlo.reshape %v2210 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2212 = stablehlo.reshape %v2211 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2213 = stablehlo.transpose %s2b8pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2214 = stablehlo.reverse %v2213, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2215 = stablehlo.convolution(%v2212, %v2214)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2216 = stablehlo.reshape %v2215 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2217 = stablehlo.reshape %v2216 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2218 = stablehlo.reshape %v1168 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2219 = stablehlo.multiply %v2218, %v2218 : tensor<32x1536x14x14xf32>
    %v2220 = stablehlo.multiply %v2219, %v2218 : tensor<32x1536x14x14xf32>
    %v2221 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v2222 = stablehlo.multiply %v2221, %v2220 : tensor<32x1536x14x14xf32>
    %v2223 = stablehlo.add %v2218, %v2222 : tensor<32x1536x14x14xf32>
    %v2224 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v2225 = stablehlo.multiply %v2224, %v2223 : tensor<32x1536x14x14xf32>
    %v2226 = stablehlo.tanh %v2225 : tensor<32x1536x14x14xf32>
    %v2227 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v2228 = stablehlo.add %v2227, %v2226 : tensor<32x1536x14x14xf32>
    %v2229 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2230 = stablehlo.multiply %v2229, %v2228 : tensor<32x1536x14x14xf32>
    %v2231 = stablehlo.multiply %v2226, %v2226 : tensor<32x1536x14x14xf32>
    %v2232 = stablehlo.subtract %v2227, %v2231 : tensor<32x1536x14x14xf32>
    %v2233 = stablehlo.multiply %v2229, %v2218 : tensor<32x1536x14x14xf32>
    %v2234 = stablehlo.multiply %v2233, %v2232 : tensor<32x1536x14x14xf32>
    %v2235 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v2236 = stablehlo.multiply %v2235, %v2219 : tensor<32x1536x14x14xf32>
    %v2237 = stablehlo.add %v2227, %v2236 : tensor<32x1536x14x14xf32>
    %v2238 = stablehlo.multiply %v2224, %v2237 : tensor<32x1536x14x14xf32>
    %v2239 = stablehlo.multiply %v2234, %v2238 : tensor<32x1536x14x14xf32>
    %v2240 = stablehlo.add %v2230, %v2239 : tensor<32x1536x14x14xf32>
    %v2241 = stablehlo.multiply %v2217, %v2240 : tensor<32x1536x14x14xf32>
    %v2242 = stablehlo.reshape %v2241 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2243 = stablehlo.reshape %v2242 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2244 = stablehlo.transpose %s2b8eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2245 = stablehlo.reverse %v2244, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2246 = stablehlo.convolution(%v2243, %v2245)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2247 = stablehlo.reshape %v2246 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2248 = stablehlo.reshape %v1129 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2249 = stablehlo.transpose %v2248, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2250 = stablehlo.reshape %v2249 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2251 = stablehlo.reshape %v2247 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2252 = stablehlo.transpose %v2251, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2253 = stablehlo.reshape %v2252 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2254 = stablehlo.reshape %v2253 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2255 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2256 = stablehlo.multiply %v2254, %v2255 : tensor<32x196x384xf32>
    %v2257 = stablehlo.reshape %v2256 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2258 = stablehlo.reshape %v2257 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2259 = stablehlo.reshape %v2250 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2261 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2262 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2263 = stablehlo.reduce(%v2259 init: %v2260) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2264 = stablehlo.broadcast_in_dim %v2263, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2265 = stablehlo.divide %v2264, %v2261 : tensor<32x196x384xf32>
    %v2266 = stablehlo.subtract %v2259, %v2265 : tensor<32x196x384xf32>
    %v2267 = stablehlo.multiply %v2266, %v2266 : tensor<32x196x384xf32>
    %v2268 = stablehlo.reduce(%v2267 init: %v2260) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2269 = stablehlo.broadcast_in_dim %v2268, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2270 = stablehlo.divide %v2269, %v2261 : tensor<32x196x384xf32>
    %v2271 = stablehlo.add %v2270, %v2262 : tensor<32x196x384xf32>
    %v2272 = stablehlo.rsqrt %v2271 : tensor<32x196x384xf32>
    %v2273 = stablehlo.multiply %v2266, %v2272 : tensor<32x196x384xf32>
    %v2274 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2275 = stablehlo.multiply %v2274, %v2258 : tensor<32x196x384xf32>
    %v2276 = stablehlo.reduce(%v2275 init: %v2260) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2277 = stablehlo.broadcast_in_dim %v2276, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2278 = stablehlo.multiply %v2273, %v2275 : tensor<32x196x384xf32>
    %v2279 = stablehlo.reduce(%v2278 init: %v2260) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2280 = stablehlo.broadcast_in_dim %v2279, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2281 = stablehlo.multiply %v2275, %v2261 : tensor<32x196x384xf32>
    %v2282 = stablehlo.subtract %v2281, %v2277 : tensor<32x196x384xf32>
    %v2283 = stablehlo.multiply %v2273, %v2280 : tensor<32x196x384xf32>
    %v2284 = stablehlo.subtract %v2282, %v2283 : tensor<32x196x384xf32>
    %v2285 = stablehlo.divide %v2272, %v2261 : tensor<32x196x384xf32>
    %v2286 = stablehlo.multiply %v2285, %v2284 : tensor<32x196x384xf32>
    %v2287 = stablehlo.reshape %v2286 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2288 = stablehlo.reshape %v2287 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2289 = stablehlo.transpose %v2288, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2290 = stablehlo.reshape %v2289 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2291 = stablehlo.reshape %v2290 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2292 = stablehlo.reverse %s2b8dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2293 = stablehlo.convolution(%v2291, %v2292)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2294 = stablehlo.reshape %v2293 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2295 = stablehlo.reshape %v2294 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2296 = stablehlo.reshape %v2154 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2297 = stablehlo.add %v2295, %v2296 : tensor<32x384x14x14xf32>
    %v2298 = stablehlo.reshape %v2297 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2299 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2300 = stablehlo.reshape %v1188 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2301 = stablehlo.reshape %v2154 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2302 = stablehlo.multiply %v2300, %v2301 : tensor<32x384x14x14xf32>
    %v2303 = stablehlo.reduce(%v2302 init: %v2299) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2304 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2305 = stablehlo.multiply %v2303, %v2304 : tensor<384xf32>
    %v2306 = stablehlo.subtract %s2b8lg, %v2305 : tensor<384xf32>
    %v2307 = stablehlo.reshape %v1183 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2308 = stablehlo.reshape %v2211 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2309 = stablehlo.transpose %v2307, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2310 = stablehlo.transpose %v2308, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2311 = stablehlo.convolution(%v2309, %v2310)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2312 = stablehlo.transpose %v2311, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2313 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2314 = stablehlo.multiply %v2312, %v2313 : tensor<384x1536x1x1xf32>
    %v2315 = stablehlo.subtract %s2b8pW, %v2314 : tensor<384x1536x1x1xf32>
    %v2316 = stablehlo.reshape %v2211 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2318 = stablehlo.reduce(%v2316 init: %v2317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2319 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2320 = stablehlo.multiply %v2318, %v2319 : tensor<384xf32>
    %v2321 = stablehlo.subtract %s2b8pb, %v2320 : tensor<384xf32>
    %v2322 = stablehlo.reshape %v1163 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2323 = stablehlo.reshape %v2242 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2324 = stablehlo.transpose %v2322, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2325 = stablehlo.transpose %v2323, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2326 = stablehlo.convolution(%v2324, %v2325)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2327 = stablehlo.transpose %v2326, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2328 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2329 = stablehlo.multiply %v2327, %v2328 : tensor<1536x384x1x1xf32>
    %v2330 = stablehlo.subtract %s2b8eW, %v2329 : tensor<1536x384x1x1xf32>
    %v2331 = stablehlo.reshape %v2242 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2333 = stablehlo.reduce(%v2331 init: %v2332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2334 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2335 = stablehlo.multiply %v2333, %v2334 : tensor<1536xf32>
    %v2336 = stablehlo.subtract %s2b8eb, %v2335 : tensor<1536xf32>
    %v2337 = stablehlo.reshape %v1129 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2338 = stablehlo.transpose %v2337, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2339 = stablehlo.reshape %v2338 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2340 = stablehlo.reshape %v2247 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2341 = stablehlo.transpose %v2340, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2342 = stablehlo.reshape %v2341 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2343 = stablehlo.reshape %v2339 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2344 = stablehlo.reshape %v2342 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2346 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2347 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2348 = stablehlo.reduce(%v2343 init: %v2345) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2349 = stablehlo.broadcast_in_dim %v2348, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2350 = stablehlo.divide %v2349, %v2346 : tensor<32x196x384xf32>
    %v2351 = stablehlo.subtract %v2343, %v2350 : tensor<32x196x384xf32>
    %v2352 = stablehlo.multiply %v2351, %v2351 : tensor<32x196x384xf32>
    %v2353 = stablehlo.reduce(%v2352 init: %v2345) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2354 = stablehlo.broadcast_in_dim %v2353, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2355 = stablehlo.divide %v2354, %v2346 : tensor<32x196x384xf32>
    %v2356 = stablehlo.add %v2355, %v2347 : tensor<32x196x384xf32>
    %v2357 = stablehlo.rsqrt %v2356 : tensor<32x196x384xf32>
    %v2358 = stablehlo.multiply %v2351, %v2357 : tensor<32x196x384xf32>
    %v2359 = stablehlo.multiply %v2344, %v2358 : tensor<32x196x384xf32>
    %v2360 = stablehlo.reduce(%v2359 init: %v2345) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2361 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2362 = stablehlo.multiply %v2360, %v2361 : tensor<384xf32>
    %v2363 = stablehlo.subtract %s2b8ng, %v2362 : tensor<384xf32>
    %v2364 = stablehlo.reshape %v2247 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2365 = stablehlo.transpose %v2364, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2366 = stablehlo.reshape %v2365 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2368 = stablehlo.reshape %v2366 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2369 = stablehlo.reduce(%v2368 init: %v2367) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2370 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2371 = stablehlo.multiply %v2369, %v2370 : tensor<384xf32>
    %v2372 = stablehlo.subtract %s2b8nbt, %v2371 : tensor<384xf32>
    %v2373 = stablehlo.reshape %v1124 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2374 = stablehlo.reshape %v2290 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2375 = stablehlo.transpose %v2373, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2376 = stablehlo.transpose %v2374, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2377 = stablehlo.convolution(%v2375, %v2376)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2378 = stablehlo.reshape %v2377 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2379 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2380 = stablehlo.multiply %v2378, %v2379 : tensor<384x1x7x7xf32>
    %v2381 = stablehlo.subtract %s2b8dW, %v2380 : tensor<384x1x7x7xf32>
    %v2382 = stablehlo.reshape %v2290 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2384 = stablehlo.reduce(%v2382 init: %v2383) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2385 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2386 = stablehlo.multiply %v2384, %v2385 : tensor<384xf32>
    %v2387 = stablehlo.subtract %s2b8db, %v2386 : tensor<384xf32>
    %v2388 = stablehlo.reshape %v2298 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2389 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2390 = stablehlo.multiply %v2388, %v2389 : tensor<32x384x14x14xf32>
    %v2391 = stablehlo.reshape %v2390 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2393 = stablehlo.transpose %s2b7pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2394 = stablehlo.reverse %v2393, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2395 = stablehlo.convolution(%v2392, %v2394)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2396 = stablehlo.reshape %v2395 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2397 = stablehlo.reshape %v2396 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2398 = stablehlo.reshape %v1096 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2399 = stablehlo.multiply %v2398, %v2398 : tensor<32x1536x14x14xf32>
    %v2400 = stablehlo.multiply %v2399, %v2398 : tensor<32x1536x14x14xf32>
    %v2401 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v2402 = stablehlo.multiply %v2401, %v2400 : tensor<32x1536x14x14xf32>
    %v2403 = stablehlo.add %v2398, %v2402 : tensor<32x1536x14x14xf32>
    %v2404 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v2405 = stablehlo.multiply %v2404, %v2403 : tensor<32x1536x14x14xf32>
    %v2406 = stablehlo.tanh %v2405 : tensor<32x1536x14x14xf32>
    %v2407 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v2408 = stablehlo.add %v2407, %v2406 : tensor<32x1536x14x14xf32>
    %v2409 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2410 = stablehlo.multiply %v2409, %v2408 : tensor<32x1536x14x14xf32>
    %v2411 = stablehlo.multiply %v2406, %v2406 : tensor<32x1536x14x14xf32>
    %v2412 = stablehlo.subtract %v2407, %v2411 : tensor<32x1536x14x14xf32>
    %v2413 = stablehlo.multiply %v2409, %v2398 : tensor<32x1536x14x14xf32>
    %v2414 = stablehlo.multiply %v2413, %v2412 : tensor<32x1536x14x14xf32>
    %v2415 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v2416 = stablehlo.multiply %v2415, %v2399 : tensor<32x1536x14x14xf32>
    %v2417 = stablehlo.add %v2407, %v2416 : tensor<32x1536x14x14xf32>
    %v2418 = stablehlo.multiply %v2404, %v2417 : tensor<32x1536x14x14xf32>
    %v2419 = stablehlo.multiply %v2414, %v2418 : tensor<32x1536x14x14xf32>
    %v2420 = stablehlo.add %v2410, %v2419 : tensor<32x1536x14x14xf32>
    %v2421 = stablehlo.multiply %v2397, %v2420 : tensor<32x1536x14x14xf32>
    %v2422 = stablehlo.reshape %v2421 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2423 = stablehlo.reshape %v2422 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2424 = stablehlo.transpose %s2b7eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2425 = stablehlo.reverse %v2424, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2426 = stablehlo.convolution(%v2423, %v2425)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2427 = stablehlo.reshape %v2426 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2428 = stablehlo.reshape %v1057 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2429 = stablehlo.transpose %v2428, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2430 = stablehlo.reshape %v2429 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2431 = stablehlo.reshape %v2427 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2432 = stablehlo.transpose %v2431, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2433 = stablehlo.reshape %v2432 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2434 = stablehlo.reshape %v2433 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2435 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2436 = stablehlo.multiply %v2434, %v2435 : tensor<32x196x384xf32>
    %v2437 = stablehlo.reshape %v2436 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2438 = stablehlo.reshape %v2437 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2439 = stablehlo.reshape %v2430 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2441 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2442 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2443 = stablehlo.reduce(%v2439 init: %v2440) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2444 = stablehlo.broadcast_in_dim %v2443, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2445 = stablehlo.divide %v2444, %v2441 : tensor<32x196x384xf32>
    %v2446 = stablehlo.subtract %v2439, %v2445 : tensor<32x196x384xf32>
    %v2447 = stablehlo.multiply %v2446, %v2446 : tensor<32x196x384xf32>
    %v2448 = stablehlo.reduce(%v2447 init: %v2440) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2449 = stablehlo.broadcast_in_dim %v2448, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2450 = stablehlo.divide %v2449, %v2441 : tensor<32x196x384xf32>
    %v2451 = stablehlo.add %v2450, %v2442 : tensor<32x196x384xf32>
    %v2452 = stablehlo.rsqrt %v2451 : tensor<32x196x384xf32>
    %v2453 = stablehlo.multiply %v2446, %v2452 : tensor<32x196x384xf32>
    %v2454 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2455 = stablehlo.multiply %v2454, %v2438 : tensor<32x196x384xf32>
    %v2456 = stablehlo.reduce(%v2455 init: %v2440) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2457 = stablehlo.broadcast_in_dim %v2456, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2458 = stablehlo.multiply %v2453, %v2455 : tensor<32x196x384xf32>
    %v2459 = stablehlo.reduce(%v2458 init: %v2440) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2460 = stablehlo.broadcast_in_dim %v2459, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2461 = stablehlo.multiply %v2455, %v2441 : tensor<32x196x384xf32>
    %v2462 = stablehlo.subtract %v2461, %v2457 : tensor<32x196x384xf32>
    %v2463 = stablehlo.multiply %v2453, %v2460 : tensor<32x196x384xf32>
    %v2464 = stablehlo.subtract %v2462, %v2463 : tensor<32x196x384xf32>
    %v2465 = stablehlo.divide %v2452, %v2441 : tensor<32x196x384xf32>
    %v2466 = stablehlo.multiply %v2465, %v2464 : tensor<32x196x384xf32>
    %v2467 = stablehlo.reshape %v2466 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2468 = stablehlo.reshape %v2467 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2469 = stablehlo.transpose %v2468, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2470 = stablehlo.reshape %v2469 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2471 = stablehlo.reshape %v2470 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2472 = stablehlo.reverse %s2b7dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2473 = stablehlo.convolution(%v2471, %v2472)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2474 = stablehlo.reshape %v2473 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2475 = stablehlo.reshape %v2474 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2476 = stablehlo.reshape %v2298 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2477 = stablehlo.add %v2475, %v2476 : tensor<32x384x14x14xf32>
    %v2478 = stablehlo.reshape %v2477 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2479 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2480 = stablehlo.reshape %v1116 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2481 = stablehlo.reshape %v2298 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2482 = stablehlo.multiply %v2480, %v2481 : tensor<32x384x14x14xf32>
    %v2483 = stablehlo.reduce(%v2482 init: %v2479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2484 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2485 = stablehlo.multiply %v2483, %v2484 : tensor<384xf32>
    %v2486 = stablehlo.subtract %s2b7lg, %v2485 : tensor<384xf32>
    %v2487 = stablehlo.reshape %v1111 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2488 = stablehlo.reshape %v2391 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2489 = stablehlo.transpose %v2487, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2490 = stablehlo.transpose %v2488, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2491 = stablehlo.convolution(%v2489, %v2490)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2492 = stablehlo.transpose %v2491, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2493 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2494 = stablehlo.multiply %v2492, %v2493 : tensor<384x1536x1x1xf32>
    %v2495 = stablehlo.subtract %s2b7pW, %v2494 : tensor<384x1536x1x1xf32>
    %v2496 = stablehlo.reshape %v2391 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2497 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2498 = stablehlo.reduce(%v2496 init: %v2497) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2499 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2500 = stablehlo.multiply %v2498, %v2499 : tensor<384xf32>
    %v2501 = stablehlo.subtract %s2b7pb, %v2500 : tensor<384xf32>
    %v2502 = stablehlo.reshape %v1091 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2503 = stablehlo.reshape %v2422 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2504 = stablehlo.transpose %v2502, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2505 = stablehlo.transpose %v2503, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2506 = stablehlo.convolution(%v2504, %v2505)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2507 = stablehlo.transpose %v2506, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2508 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2509 = stablehlo.multiply %v2507, %v2508 : tensor<1536x384x1x1xf32>
    %v2510 = stablehlo.subtract %s2b7eW, %v2509 : tensor<1536x384x1x1xf32>
    %v2511 = stablehlo.reshape %v2422 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2513 = stablehlo.reduce(%v2511 init: %v2512) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2514 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2515 = stablehlo.multiply %v2513, %v2514 : tensor<1536xf32>
    %v2516 = stablehlo.subtract %s2b7eb, %v2515 : tensor<1536xf32>
    %v2517 = stablehlo.reshape %v1057 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2518 = stablehlo.transpose %v2517, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2519 = stablehlo.reshape %v2518 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2520 = stablehlo.reshape %v2427 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2521 = stablehlo.transpose %v2520, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2522 = stablehlo.reshape %v2521 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2523 = stablehlo.reshape %v2519 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2524 = stablehlo.reshape %v2522 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2526 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2527 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2528 = stablehlo.reduce(%v2523 init: %v2525) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2529 = stablehlo.broadcast_in_dim %v2528, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2530 = stablehlo.divide %v2529, %v2526 : tensor<32x196x384xf32>
    %v2531 = stablehlo.subtract %v2523, %v2530 : tensor<32x196x384xf32>
    %v2532 = stablehlo.multiply %v2531, %v2531 : tensor<32x196x384xf32>
    %v2533 = stablehlo.reduce(%v2532 init: %v2525) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2534 = stablehlo.broadcast_in_dim %v2533, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2535 = stablehlo.divide %v2534, %v2526 : tensor<32x196x384xf32>
    %v2536 = stablehlo.add %v2535, %v2527 : tensor<32x196x384xf32>
    %v2537 = stablehlo.rsqrt %v2536 : tensor<32x196x384xf32>
    %v2538 = stablehlo.multiply %v2531, %v2537 : tensor<32x196x384xf32>
    %v2539 = stablehlo.multiply %v2524, %v2538 : tensor<32x196x384xf32>
    %v2540 = stablehlo.reduce(%v2539 init: %v2525) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2541 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2542 = stablehlo.multiply %v2540, %v2541 : tensor<384xf32>
    %v2543 = stablehlo.subtract %s2b7ng, %v2542 : tensor<384xf32>
    %v2544 = stablehlo.reshape %v2427 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2545 = stablehlo.transpose %v2544, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2546 = stablehlo.reshape %v2545 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2547 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2548 = stablehlo.reshape %v2546 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2549 = stablehlo.reduce(%v2548 init: %v2547) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2550 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2551 = stablehlo.multiply %v2549, %v2550 : tensor<384xf32>
    %v2552 = stablehlo.subtract %s2b7nbt, %v2551 : tensor<384xf32>
    %v2553 = stablehlo.reshape %v1052 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2554 = stablehlo.reshape %v2470 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2555 = stablehlo.transpose %v2553, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2556 = stablehlo.transpose %v2554, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2557 = stablehlo.convolution(%v2555, %v2556)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2558 = stablehlo.reshape %v2557 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2559 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2560 = stablehlo.multiply %v2558, %v2559 : tensor<384x1x7x7xf32>
    %v2561 = stablehlo.subtract %s2b7dW, %v2560 : tensor<384x1x7x7xf32>
    %v2562 = stablehlo.reshape %v2470 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2564 = stablehlo.reduce(%v2562 init: %v2563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2565 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2566 = stablehlo.multiply %v2564, %v2565 : tensor<384xf32>
    %v2567 = stablehlo.subtract %s2b7db, %v2566 : tensor<384xf32>
    %v2568 = stablehlo.reshape %v2478 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2569 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2570 = stablehlo.multiply %v2568, %v2569 : tensor<32x384x14x14xf32>
    %v2571 = stablehlo.reshape %v2570 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2572 = stablehlo.reshape %v2571 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2573 = stablehlo.transpose %s2b6pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2574 = stablehlo.reverse %v2573, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2575 = stablehlo.convolution(%v2572, %v2574)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2576 = stablehlo.reshape %v2575 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2577 = stablehlo.reshape %v2576 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2578 = stablehlo.reshape %v1024 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2579 = stablehlo.multiply %v2578, %v2578 : tensor<32x1536x14x14xf32>
    %v2580 = stablehlo.multiply %v2579, %v2578 : tensor<32x1536x14x14xf32>
    %v2581 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v2582 = stablehlo.multiply %v2581, %v2580 : tensor<32x1536x14x14xf32>
    %v2583 = stablehlo.add %v2578, %v2582 : tensor<32x1536x14x14xf32>
    %v2584 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v2585 = stablehlo.multiply %v2584, %v2583 : tensor<32x1536x14x14xf32>
    %v2586 = stablehlo.tanh %v2585 : tensor<32x1536x14x14xf32>
    %v2587 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v2588 = stablehlo.add %v2587, %v2586 : tensor<32x1536x14x14xf32>
    %v2589 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2590 = stablehlo.multiply %v2589, %v2588 : tensor<32x1536x14x14xf32>
    %v2591 = stablehlo.multiply %v2586, %v2586 : tensor<32x1536x14x14xf32>
    %v2592 = stablehlo.subtract %v2587, %v2591 : tensor<32x1536x14x14xf32>
    %v2593 = stablehlo.multiply %v2589, %v2578 : tensor<32x1536x14x14xf32>
    %v2594 = stablehlo.multiply %v2593, %v2592 : tensor<32x1536x14x14xf32>
    %v2595 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v2596 = stablehlo.multiply %v2595, %v2579 : tensor<32x1536x14x14xf32>
    %v2597 = stablehlo.add %v2587, %v2596 : tensor<32x1536x14x14xf32>
    %v2598 = stablehlo.multiply %v2584, %v2597 : tensor<32x1536x14x14xf32>
    %v2599 = stablehlo.multiply %v2594, %v2598 : tensor<32x1536x14x14xf32>
    %v2600 = stablehlo.add %v2590, %v2599 : tensor<32x1536x14x14xf32>
    %v2601 = stablehlo.multiply %v2577, %v2600 : tensor<32x1536x14x14xf32>
    %v2602 = stablehlo.reshape %v2601 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2603 = stablehlo.reshape %v2602 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2604 = stablehlo.transpose %s2b6eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2605 = stablehlo.reverse %v2604, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2606 = stablehlo.convolution(%v2603, %v2605)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2607 = stablehlo.reshape %v2606 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2608 = stablehlo.reshape %v985 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2609 = stablehlo.transpose %v2608, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2611 = stablehlo.reshape %v2607 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2612 = stablehlo.transpose %v2611, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2613 = stablehlo.reshape %v2612 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2614 = stablehlo.reshape %v2613 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2615 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2616 = stablehlo.multiply %v2614, %v2615 : tensor<32x196x384xf32>
    %v2617 = stablehlo.reshape %v2616 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2618 = stablehlo.reshape %v2617 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2619 = stablehlo.reshape %v2610 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2620 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2621 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2622 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2623 = stablehlo.reduce(%v2619 init: %v2620) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2624 = stablehlo.broadcast_in_dim %v2623, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2625 = stablehlo.divide %v2624, %v2621 : tensor<32x196x384xf32>
    %v2626 = stablehlo.subtract %v2619, %v2625 : tensor<32x196x384xf32>
    %v2627 = stablehlo.multiply %v2626, %v2626 : tensor<32x196x384xf32>
    %v2628 = stablehlo.reduce(%v2627 init: %v2620) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2629 = stablehlo.broadcast_in_dim %v2628, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2630 = stablehlo.divide %v2629, %v2621 : tensor<32x196x384xf32>
    %v2631 = stablehlo.add %v2630, %v2622 : tensor<32x196x384xf32>
    %v2632 = stablehlo.rsqrt %v2631 : tensor<32x196x384xf32>
    %v2633 = stablehlo.multiply %v2626, %v2632 : tensor<32x196x384xf32>
    %v2634 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2635 = stablehlo.multiply %v2634, %v2618 : tensor<32x196x384xf32>
    %v2636 = stablehlo.reduce(%v2635 init: %v2620) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2637 = stablehlo.broadcast_in_dim %v2636, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2638 = stablehlo.multiply %v2633, %v2635 : tensor<32x196x384xf32>
    %v2639 = stablehlo.reduce(%v2638 init: %v2620) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2640 = stablehlo.broadcast_in_dim %v2639, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2641 = stablehlo.multiply %v2635, %v2621 : tensor<32x196x384xf32>
    %v2642 = stablehlo.subtract %v2641, %v2637 : tensor<32x196x384xf32>
    %v2643 = stablehlo.multiply %v2633, %v2640 : tensor<32x196x384xf32>
    %v2644 = stablehlo.subtract %v2642, %v2643 : tensor<32x196x384xf32>
    %v2645 = stablehlo.divide %v2632, %v2621 : tensor<32x196x384xf32>
    %v2646 = stablehlo.multiply %v2645, %v2644 : tensor<32x196x384xf32>
    %v2647 = stablehlo.reshape %v2646 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2648 = stablehlo.reshape %v2647 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2649 = stablehlo.transpose %v2648, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2650 = stablehlo.reshape %v2649 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2651 = stablehlo.reshape %v2650 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2652 = stablehlo.reverse %s2b6dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2653 = stablehlo.convolution(%v2651, %v2652)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2654 = stablehlo.reshape %v2653 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2655 = stablehlo.reshape %v2654 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2656 = stablehlo.reshape %v2478 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2657 = stablehlo.add %v2655, %v2656 : tensor<32x384x14x14xf32>
    %v2658 = stablehlo.reshape %v2657 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2660 = stablehlo.reshape %v1044 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2661 = stablehlo.reshape %v2478 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2662 = stablehlo.multiply %v2660, %v2661 : tensor<32x384x14x14xf32>
    %v2663 = stablehlo.reduce(%v2662 init: %v2659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2664 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2665 = stablehlo.multiply %v2663, %v2664 : tensor<384xf32>
    %v2666 = stablehlo.subtract %s2b6lg, %v2665 : tensor<384xf32>
    %v2667 = stablehlo.reshape %v1039 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2668 = stablehlo.reshape %v2571 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2669 = stablehlo.transpose %v2667, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2670 = stablehlo.transpose %v2668, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2671 = stablehlo.convolution(%v2669, %v2670)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2672 = stablehlo.transpose %v2671, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2673 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2674 = stablehlo.multiply %v2672, %v2673 : tensor<384x1536x1x1xf32>
    %v2675 = stablehlo.subtract %s2b6pW, %v2674 : tensor<384x1536x1x1xf32>
    %v2676 = stablehlo.reshape %v2571 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2678 = stablehlo.reduce(%v2676 init: %v2677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2679 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2680 = stablehlo.multiply %v2678, %v2679 : tensor<384xf32>
    %v2681 = stablehlo.subtract %s2b6pb, %v2680 : tensor<384xf32>
    %v2682 = stablehlo.reshape %v1019 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2683 = stablehlo.reshape %v2602 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2684 = stablehlo.transpose %v2682, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2685 = stablehlo.transpose %v2683, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2686 = stablehlo.convolution(%v2684, %v2685)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2687 = stablehlo.transpose %v2686, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2688 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2689 = stablehlo.multiply %v2687, %v2688 : tensor<1536x384x1x1xf32>
    %v2690 = stablehlo.subtract %s2b6eW, %v2689 : tensor<1536x384x1x1xf32>
    %v2691 = stablehlo.reshape %v2602 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2693 = stablehlo.reduce(%v2691 init: %v2692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2694 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2695 = stablehlo.multiply %v2693, %v2694 : tensor<1536xf32>
    %v2696 = stablehlo.subtract %s2b6eb, %v2695 : tensor<1536xf32>
    %v2697 = stablehlo.reshape %v985 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2698 = stablehlo.transpose %v2697, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2699 = stablehlo.reshape %v2698 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2700 = stablehlo.reshape %v2607 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2701 = stablehlo.transpose %v2700, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2702 = stablehlo.reshape %v2701 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2703 = stablehlo.reshape %v2699 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2704 = stablehlo.reshape %v2702 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2706 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2707 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2708 = stablehlo.reduce(%v2703 init: %v2705) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2709 = stablehlo.broadcast_in_dim %v2708, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2710 = stablehlo.divide %v2709, %v2706 : tensor<32x196x384xf32>
    %v2711 = stablehlo.subtract %v2703, %v2710 : tensor<32x196x384xf32>
    %v2712 = stablehlo.multiply %v2711, %v2711 : tensor<32x196x384xf32>
    %v2713 = stablehlo.reduce(%v2712 init: %v2705) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2714 = stablehlo.broadcast_in_dim %v2713, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2715 = stablehlo.divide %v2714, %v2706 : tensor<32x196x384xf32>
    %v2716 = stablehlo.add %v2715, %v2707 : tensor<32x196x384xf32>
    %v2717 = stablehlo.rsqrt %v2716 : tensor<32x196x384xf32>
    %v2718 = stablehlo.multiply %v2711, %v2717 : tensor<32x196x384xf32>
    %v2719 = stablehlo.multiply %v2704, %v2718 : tensor<32x196x384xf32>
    %v2720 = stablehlo.reduce(%v2719 init: %v2705) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2721 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2722 = stablehlo.multiply %v2720, %v2721 : tensor<384xf32>
    %v2723 = stablehlo.subtract %s2b6ng, %v2722 : tensor<384xf32>
    %v2724 = stablehlo.reshape %v2607 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2725 = stablehlo.transpose %v2724, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2726 = stablehlo.reshape %v2725 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2727 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2728 = stablehlo.reshape %v2726 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2729 = stablehlo.reduce(%v2728 init: %v2727) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2730 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2731 = stablehlo.multiply %v2729, %v2730 : tensor<384xf32>
    %v2732 = stablehlo.subtract %s2b6nbt, %v2731 : tensor<384xf32>
    %v2733 = stablehlo.reshape %v980 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2734 = stablehlo.reshape %v2650 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2735 = stablehlo.transpose %v2733, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2736 = stablehlo.transpose %v2734, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2737 = stablehlo.convolution(%v2735, %v2736)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2738 = stablehlo.reshape %v2737 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2739 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2740 = stablehlo.multiply %v2738, %v2739 : tensor<384x1x7x7xf32>
    %v2741 = stablehlo.subtract %s2b6dW, %v2740 : tensor<384x1x7x7xf32>
    %v2742 = stablehlo.reshape %v2650 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2744 = stablehlo.reduce(%v2742 init: %v2743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2745 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2746 = stablehlo.multiply %v2744, %v2745 : tensor<384xf32>
    %v2747 = stablehlo.subtract %s2b6db, %v2746 : tensor<384xf32>
    %v2748 = stablehlo.reshape %v2658 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2749 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2750 = stablehlo.multiply %v2748, %v2749 : tensor<32x384x14x14xf32>
    %v2751 = stablehlo.reshape %v2750 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2752 = stablehlo.reshape %v2751 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2753 = stablehlo.transpose %s2b5pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2754 = stablehlo.reverse %v2753, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2755 = stablehlo.convolution(%v2752, %v2754)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2756 = stablehlo.reshape %v2755 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2757 = stablehlo.reshape %v2756 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2758 = stablehlo.reshape %v952 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2759 = stablehlo.multiply %v2758, %v2758 : tensor<32x1536x14x14xf32>
    %v2760 = stablehlo.multiply %v2759, %v2758 : tensor<32x1536x14x14xf32>
    %v2761 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v2762 = stablehlo.multiply %v2761, %v2760 : tensor<32x1536x14x14xf32>
    %v2763 = stablehlo.add %v2758, %v2762 : tensor<32x1536x14x14xf32>
    %v2764 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v2765 = stablehlo.multiply %v2764, %v2763 : tensor<32x1536x14x14xf32>
    %v2766 = stablehlo.tanh %v2765 : tensor<32x1536x14x14xf32>
    %v2767 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v2768 = stablehlo.add %v2767, %v2766 : tensor<32x1536x14x14xf32>
    %v2769 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2770 = stablehlo.multiply %v2769, %v2768 : tensor<32x1536x14x14xf32>
    %v2771 = stablehlo.multiply %v2766, %v2766 : tensor<32x1536x14x14xf32>
    %v2772 = stablehlo.subtract %v2767, %v2771 : tensor<32x1536x14x14xf32>
    %v2773 = stablehlo.multiply %v2769, %v2758 : tensor<32x1536x14x14xf32>
    %v2774 = stablehlo.multiply %v2773, %v2772 : tensor<32x1536x14x14xf32>
    %v2775 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v2776 = stablehlo.multiply %v2775, %v2759 : tensor<32x1536x14x14xf32>
    %v2777 = stablehlo.add %v2767, %v2776 : tensor<32x1536x14x14xf32>
    %v2778 = stablehlo.multiply %v2764, %v2777 : tensor<32x1536x14x14xf32>
    %v2779 = stablehlo.multiply %v2774, %v2778 : tensor<32x1536x14x14xf32>
    %v2780 = stablehlo.add %v2770, %v2779 : tensor<32x1536x14x14xf32>
    %v2781 = stablehlo.multiply %v2757, %v2780 : tensor<32x1536x14x14xf32>
    %v2782 = stablehlo.reshape %v2781 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2783 = stablehlo.reshape %v2782 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2784 = stablehlo.transpose %s2b5eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2785 = stablehlo.reverse %v2784, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2786 = stablehlo.convolution(%v2783, %v2785)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2787 = stablehlo.reshape %v2786 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2788 = stablehlo.reshape %v913 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2789 = stablehlo.transpose %v2788, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2790 = stablehlo.reshape %v2789 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2791 = stablehlo.reshape %v2787 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2792 = stablehlo.transpose %v2791, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2793 = stablehlo.reshape %v2792 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2794 = stablehlo.reshape %v2793 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2795 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2796 = stablehlo.multiply %v2794, %v2795 : tensor<32x196x384xf32>
    %v2797 = stablehlo.reshape %v2796 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2798 = stablehlo.reshape %v2797 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2799 = stablehlo.reshape %v2790 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2801 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2802 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2803 = stablehlo.reduce(%v2799 init: %v2800) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2804 = stablehlo.broadcast_in_dim %v2803, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2805 = stablehlo.divide %v2804, %v2801 : tensor<32x196x384xf32>
    %v2806 = stablehlo.subtract %v2799, %v2805 : tensor<32x196x384xf32>
    %v2807 = stablehlo.multiply %v2806, %v2806 : tensor<32x196x384xf32>
    %v2808 = stablehlo.reduce(%v2807 init: %v2800) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2809 = stablehlo.broadcast_in_dim %v2808, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2810 = stablehlo.divide %v2809, %v2801 : tensor<32x196x384xf32>
    %v2811 = stablehlo.add %v2810, %v2802 : tensor<32x196x384xf32>
    %v2812 = stablehlo.rsqrt %v2811 : tensor<32x196x384xf32>
    %v2813 = stablehlo.multiply %v2806, %v2812 : tensor<32x196x384xf32>
    %v2814 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2815 = stablehlo.multiply %v2814, %v2798 : tensor<32x196x384xf32>
    %v2816 = stablehlo.reduce(%v2815 init: %v2800) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2817 = stablehlo.broadcast_in_dim %v2816, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2818 = stablehlo.multiply %v2813, %v2815 : tensor<32x196x384xf32>
    %v2819 = stablehlo.reduce(%v2818 init: %v2800) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2820 = stablehlo.broadcast_in_dim %v2819, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2821 = stablehlo.multiply %v2815, %v2801 : tensor<32x196x384xf32>
    %v2822 = stablehlo.subtract %v2821, %v2817 : tensor<32x196x384xf32>
    %v2823 = stablehlo.multiply %v2813, %v2820 : tensor<32x196x384xf32>
    %v2824 = stablehlo.subtract %v2822, %v2823 : tensor<32x196x384xf32>
    %v2825 = stablehlo.divide %v2812, %v2801 : tensor<32x196x384xf32>
    %v2826 = stablehlo.multiply %v2825, %v2824 : tensor<32x196x384xf32>
    %v2827 = stablehlo.reshape %v2826 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2828 = stablehlo.reshape %v2827 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2829 = stablehlo.transpose %v2828, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2830 = stablehlo.reshape %v2829 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2831 = stablehlo.reshape %v2830 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2832 = stablehlo.reverse %s2b5dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2833 = stablehlo.convolution(%v2831, %v2832)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2834 = stablehlo.reshape %v2833 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2835 = stablehlo.reshape %v2834 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2836 = stablehlo.reshape %v2658 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2837 = stablehlo.add %v2835, %v2836 : tensor<32x384x14x14xf32>
    %v2838 = stablehlo.reshape %v2837 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2839 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2840 = stablehlo.reshape %v972 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2841 = stablehlo.reshape %v2658 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2842 = stablehlo.multiply %v2840, %v2841 : tensor<32x384x14x14xf32>
    %v2843 = stablehlo.reduce(%v2842 init: %v2839) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2844 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2845 = stablehlo.multiply %v2843, %v2844 : tensor<384xf32>
    %v2846 = stablehlo.subtract %s2b5lg, %v2845 : tensor<384xf32>
    %v2847 = stablehlo.reshape %v967 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2848 = stablehlo.reshape %v2751 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2849 = stablehlo.transpose %v2847, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2850 = stablehlo.transpose %v2848, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2851 = stablehlo.convolution(%v2849, %v2850)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2852 = stablehlo.transpose %v2851, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2853 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2854 = stablehlo.multiply %v2852, %v2853 : tensor<384x1536x1x1xf32>
    %v2855 = stablehlo.subtract %s2b5pW, %v2854 : tensor<384x1536x1x1xf32>
    %v2856 = stablehlo.reshape %v2751 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2857 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2858 = stablehlo.reduce(%v2856 init: %v2857) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2859 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2860 = stablehlo.multiply %v2858, %v2859 : tensor<384xf32>
    %v2861 = stablehlo.subtract %s2b5pb, %v2860 : tensor<384xf32>
    %v2862 = stablehlo.reshape %v947 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2863 = stablehlo.reshape %v2782 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2864 = stablehlo.transpose %v2862, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2865 = stablehlo.transpose %v2863, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2866 = stablehlo.convolution(%v2864, %v2865)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2867 = stablehlo.transpose %v2866, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2868 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2869 = stablehlo.multiply %v2867, %v2868 : tensor<1536x384x1x1xf32>
    %v2870 = stablehlo.subtract %s2b5eW, %v2869 : tensor<1536x384x1x1xf32>
    %v2871 = stablehlo.reshape %v2782 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2872 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2873 = stablehlo.reduce(%v2871 init: %v2872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2874 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2875 = stablehlo.multiply %v2873, %v2874 : tensor<1536xf32>
    %v2876 = stablehlo.subtract %s2b5eb, %v2875 : tensor<1536xf32>
    %v2877 = stablehlo.reshape %v913 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2878 = stablehlo.transpose %v2877, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2879 = stablehlo.reshape %v2878 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2880 = stablehlo.reshape %v2787 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2881 = stablehlo.transpose %v2880, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2882 = stablehlo.reshape %v2881 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2883 = stablehlo.reshape %v2879 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2884 = stablehlo.reshape %v2882 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2886 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2887 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2888 = stablehlo.reduce(%v2883 init: %v2885) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2889 = stablehlo.broadcast_in_dim %v2888, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2890 = stablehlo.divide %v2889, %v2886 : tensor<32x196x384xf32>
    %v2891 = stablehlo.subtract %v2883, %v2890 : tensor<32x196x384xf32>
    %v2892 = stablehlo.multiply %v2891, %v2891 : tensor<32x196x384xf32>
    %v2893 = stablehlo.reduce(%v2892 init: %v2885) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2894 = stablehlo.broadcast_in_dim %v2893, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2895 = stablehlo.divide %v2894, %v2886 : tensor<32x196x384xf32>
    %v2896 = stablehlo.add %v2895, %v2887 : tensor<32x196x384xf32>
    %v2897 = stablehlo.rsqrt %v2896 : tensor<32x196x384xf32>
    %v2898 = stablehlo.multiply %v2891, %v2897 : tensor<32x196x384xf32>
    %v2899 = stablehlo.multiply %v2884, %v2898 : tensor<32x196x384xf32>
    %v2900 = stablehlo.reduce(%v2899 init: %v2885) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2901 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2902 = stablehlo.multiply %v2900, %v2901 : tensor<384xf32>
    %v2903 = stablehlo.subtract %s2b5ng, %v2902 : tensor<384xf32>
    %v2904 = stablehlo.reshape %v2787 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2905 = stablehlo.transpose %v2904, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2906 = stablehlo.reshape %v2905 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2907 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2908 = stablehlo.reshape %v2906 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2909 = stablehlo.reduce(%v2908 init: %v2907) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2910 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2911 = stablehlo.multiply %v2909, %v2910 : tensor<384xf32>
    %v2912 = stablehlo.subtract %s2b5nbt, %v2911 : tensor<384xf32>
    %v2913 = stablehlo.reshape %v908 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2914 = stablehlo.reshape %v2830 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2915 = stablehlo.transpose %v2913, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2916 = stablehlo.transpose %v2914, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2917 = stablehlo.convolution(%v2915, %v2916)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2918 = stablehlo.reshape %v2917 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2919 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2920 = stablehlo.multiply %v2918, %v2919 : tensor<384x1x7x7xf32>
    %v2921 = stablehlo.subtract %s2b5dW, %v2920 : tensor<384x1x7x7xf32>
    %v2922 = stablehlo.reshape %v2830 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2923 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2924 = stablehlo.reduce(%v2922 init: %v2923) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2925 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2926 = stablehlo.multiply %v2924, %v2925 : tensor<384xf32>
    %v2927 = stablehlo.subtract %s2b5db, %v2926 : tensor<384xf32>
    %v2928 = stablehlo.reshape %v2838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2929 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2930 = stablehlo.multiply %v2928, %v2929 : tensor<32x384x14x14xf32>
    %v2931 = stablehlo.reshape %v2930 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2932 = stablehlo.reshape %v2931 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2933 = stablehlo.transpose %s2b4pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2934 = stablehlo.reverse %v2933, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2935 = stablehlo.convolution(%v2932, %v2934)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2936 = stablehlo.reshape %v2935 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2937 = stablehlo.reshape %v2936 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2938 = stablehlo.reshape %v880 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2939 = stablehlo.multiply %v2938, %v2938 : tensor<32x1536x14x14xf32>
    %v2940 = stablehlo.multiply %v2939, %v2938 : tensor<32x1536x14x14xf32>
    %v2941 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v2942 = stablehlo.multiply %v2941, %v2940 : tensor<32x1536x14x14xf32>
    %v2943 = stablehlo.add %v2938, %v2942 : tensor<32x1536x14x14xf32>
    %v2944 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v2945 = stablehlo.multiply %v2944, %v2943 : tensor<32x1536x14x14xf32>
    %v2946 = stablehlo.tanh %v2945 : tensor<32x1536x14x14xf32>
    %v2947 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v2948 = stablehlo.add %v2947, %v2946 : tensor<32x1536x14x14xf32>
    %v2949 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2950 = stablehlo.multiply %v2949, %v2948 : tensor<32x1536x14x14xf32>
    %v2951 = stablehlo.multiply %v2946, %v2946 : tensor<32x1536x14x14xf32>
    %v2952 = stablehlo.subtract %v2947, %v2951 : tensor<32x1536x14x14xf32>
    %v2953 = stablehlo.multiply %v2949, %v2938 : tensor<32x1536x14x14xf32>
    %v2954 = stablehlo.multiply %v2953, %v2952 : tensor<32x1536x14x14xf32>
    %v2955 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v2956 = stablehlo.multiply %v2955, %v2939 : tensor<32x1536x14x14xf32>
    %v2957 = stablehlo.add %v2947, %v2956 : tensor<32x1536x14x14xf32>
    %v2958 = stablehlo.multiply %v2944, %v2957 : tensor<32x1536x14x14xf32>
    %v2959 = stablehlo.multiply %v2954, %v2958 : tensor<32x1536x14x14xf32>
    %v2960 = stablehlo.add %v2950, %v2959 : tensor<32x1536x14x14xf32>
    %v2961 = stablehlo.multiply %v2937, %v2960 : tensor<32x1536x14x14xf32>
    %v2962 = stablehlo.reshape %v2961 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2963 = stablehlo.reshape %v2962 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2964 = stablehlo.transpose %s2b4eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2965 = stablehlo.reverse %v2964, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2966 = stablehlo.convolution(%v2963, %v2965)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2967 = stablehlo.reshape %v2966 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2968 = stablehlo.reshape %v841 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2969 = stablehlo.transpose %v2968, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2970 = stablehlo.reshape %v2969 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2971 = stablehlo.reshape %v2967 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2972 = stablehlo.transpose %v2971, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2973 = stablehlo.reshape %v2972 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2974 = stablehlo.reshape %v2973 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2975 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2976 = stablehlo.multiply %v2974, %v2975 : tensor<32x196x384xf32>
    %v2977 = stablehlo.reshape %v2976 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2978 = stablehlo.reshape %v2977 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2979 = stablehlo.reshape %v2970 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2981 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2982 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2983 = stablehlo.reduce(%v2979 init: %v2980) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2984 = stablehlo.broadcast_in_dim %v2983, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2985 = stablehlo.divide %v2984, %v2981 : tensor<32x196x384xf32>
    %v2986 = stablehlo.subtract %v2979, %v2985 : tensor<32x196x384xf32>
    %v2987 = stablehlo.multiply %v2986, %v2986 : tensor<32x196x384xf32>
    %v2988 = stablehlo.reduce(%v2987 init: %v2980) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2989 = stablehlo.broadcast_in_dim %v2988, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2990 = stablehlo.divide %v2989, %v2981 : tensor<32x196x384xf32>
    %v2991 = stablehlo.add %v2990, %v2982 : tensor<32x196x384xf32>
    %v2992 = stablehlo.rsqrt %v2991 : tensor<32x196x384xf32>
    %v2993 = stablehlo.multiply %v2986, %v2992 : tensor<32x196x384xf32>
    %v2994 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2995 = stablehlo.multiply %v2994, %v2978 : tensor<32x196x384xf32>
    %v2996 = stablehlo.reduce(%v2995 init: %v2980) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2997 = stablehlo.broadcast_in_dim %v2996, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2998 = stablehlo.multiply %v2993, %v2995 : tensor<32x196x384xf32>
    %v2999 = stablehlo.reduce(%v2998 init: %v2980) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3000 = stablehlo.broadcast_in_dim %v2999, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3001 = stablehlo.multiply %v2995, %v2981 : tensor<32x196x384xf32>
    %v3002 = stablehlo.subtract %v3001, %v2997 : tensor<32x196x384xf32>
    %v3003 = stablehlo.multiply %v2993, %v3000 : tensor<32x196x384xf32>
    %v3004 = stablehlo.subtract %v3002, %v3003 : tensor<32x196x384xf32>
    %v3005 = stablehlo.divide %v2992, %v2981 : tensor<32x196x384xf32>
    %v3006 = stablehlo.multiply %v3005, %v3004 : tensor<32x196x384xf32>
    %v3007 = stablehlo.reshape %v3006 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3008 = stablehlo.reshape %v3007 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3009 = stablehlo.transpose %v3008, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3010 = stablehlo.reshape %v3009 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3011 = stablehlo.reshape %v3010 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3012 = stablehlo.reverse %s2b4dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3013 = stablehlo.convolution(%v3011, %v3012)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3014 = stablehlo.reshape %v3013 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3015 = stablehlo.reshape %v3014 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3016 = stablehlo.reshape %v2838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3017 = stablehlo.add %v3015, %v3016 : tensor<32x384x14x14xf32>
    %v3018 = stablehlo.reshape %v3017 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3020 = stablehlo.reshape %v900 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3021 = stablehlo.reshape %v2838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3022 = stablehlo.multiply %v3020, %v3021 : tensor<32x384x14x14xf32>
    %v3023 = stablehlo.reduce(%v3022 init: %v3019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3024 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3025 = stablehlo.multiply %v3023, %v3024 : tensor<384xf32>
    %v3026 = stablehlo.subtract %s2b4lg, %v3025 : tensor<384xf32>
    %v3027 = stablehlo.reshape %v895 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3028 = stablehlo.reshape %v2931 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3029 = stablehlo.transpose %v3027, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3030 = stablehlo.transpose %v3028, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3031 = stablehlo.convolution(%v3029, %v3030)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3032 = stablehlo.transpose %v3031, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3033 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v3034 = stablehlo.multiply %v3032, %v3033 : tensor<384x1536x1x1xf32>
    %v3035 = stablehlo.subtract %s2b4pW, %v3034 : tensor<384x1536x1x1xf32>
    %v3036 = stablehlo.reshape %v2931 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3037 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3038 = stablehlo.reduce(%v3036 init: %v3037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3039 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3040 = stablehlo.multiply %v3038, %v3039 : tensor<384xf32>
    %v3041 = stablehlo.subtract %s2b4pb, %v3040 : tensor<384xf32>
    %v3042 = stablehlo.reshape %v875 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3043 = stablehlo.reshape %v2962 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3044 = stablehlo.transpose %v3042, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3045 = stablehlo.transpose %v3043, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3046 = stablehlo.convolution(%v3044, %v3045)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3047 = stablehlo.transpose %v3046, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3048 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v3049 = stablehlo.multiply %v3047, %v3048 : tensor<1536x384x1x1xf32>
    %v3050 = stablehlo.subtract %s2b4eW, %v3049 : tensor<1536x384x1x1xf32>
    %v3051 = stablehlo.reshape %v2962 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3052 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3053 = stablehlo.reduce(%v3051 init: %v3052) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3054 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3055 = stablehlo.multiply %v3053, %v3054 : tensor<1536xf32>
    %v3056 = stablehlo.subtract %s2b4eb, %v3055 : tensor<1536xf32>
    %v3057 = stablehlo.reshape %v841 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3058 = stablehlo.transpose %v3057, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3059 = stablehlo.reshape %v3058 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3060 = stablehlo.reshape %v2967 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3061 = stablehlo.transpose %v3060, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3062 = stablehlo.reshape %v3061 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3063 = stablehlo.reshape %v3059 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3064 = stablehlo.reshape %v3062 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3065 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3066 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3067 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3068 = stablehlo.reduce(%v3063 init: %v3065) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3069 = stablehlo.broadcast_in_dim %v3068, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3070 = stablehlo.divide %v3069, %v3066 : tensor<32x196x384xf32>
    %v3071 = stablehlo.subtract %v3063, %v3070 : tensor<32x196x384xf32>
    %v3072 = stablehlo.multiply %v3071, %v3071 : tensor<32x196x384xf32>
    %v3073 = stablehlo.reduce(%v3072 init: %v3065) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3074 = stablehlo.broadcast_in_dim %v3073, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3075 = stablehlo.divide %v3074, %v3066 : tensor<32x196x384xf32>
    %v3076 = stablehlo.add %v3075, %v3067 : tensor<32x196x384xf32>
    %v3077 = stablehlo.rsqrt %v3076 : tensor<32x196x384xf32>
    %v3078 = stablehlo.multiply %v3071, %v3077 : tensor<32x196x384xf32>
    %v3079 = stablehlo.multiply %v3064, %v3078 : tensor<32x196x384xf32>
    %v3080 = stablehlo.reduce(%v3079 init: %v3065) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3081 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3082 = stablehlo.multiply %v3080, %v3081 : tensor<384xf32>
    %v3083 = stablehlo.subtract %s2b4ng, %v3082 : tensor<384xf32>
    %v3084 = stablehlo.reshape %v2967 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3085 = stablehlo.transpose %v3084, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3086 = stablehlo.reshape %v3085 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3088 = stablehlo.reshape %v3086 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3089 = stablehlo.reduce(%v3088 init: %v3087) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3090 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3091 = stablehlo.multiply %v3089, %v3090 : tensor<384xf32>
    %v3092 = stablehlo.subtract %s2b4nbt, %v3091 : tensor<384xf32>
    %v3093 = stablehlo.reshape %v836 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3094 = stablehlo.reshape %v3010 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3095 = stablehlo.transpose %v3093, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3096 = stablehlo.transpose %v3094, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3097 = stablehlo.convolution(%v3095, %v3096)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3098 = stablehlo.reshape %v3097 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3099 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3100 = stablehlo.multiply %v3098, %v3099 : tensor<384x1x7x7xf32>
    %v3101 = stablehlo.subtract %s2b4dW, %v3100 : tensor<384x1x7x7xf32>
    %v3102 = stablehlo.reshape %v3010 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3104 = stablehlo.reduce(%v3102 init: %v3103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3105 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3106 = stablehlo.multiply %v3104, %v3105 : tensor<384xf32>
    %v3107 = stablehlo.subtract %s2b4db, %v3106 : tensor<384xf32>
    %v3108 = stablehlo.reshape %v3018 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3109 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3110 = stablehlo.multiply %v3108, %v3109 : tensor<32x384x14x14xf32>
    %v3111 = stablehlo.reshape %v3110 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3112 = stablehlo.reshape %v3111 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3113 = stablehlo.transpose %s2b3pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3114 = stablehlo.reverse %v3113, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3115 = stablehlo.convolution(%v3112, %v3114)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3116 = stablehlo.reshape %v3115 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3117 = stablehlo.reshape %v3116 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3118 = stablehlo.reshape %v808 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3119 = stablehlo.multiply %v3118, %v3118 : tensor<32x1536x14x14xf32>
    %v3120 = stablehlo.multiply %v3119, %v3118 : tensor<32x1536x14x14xf32>
    %v3121 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v3122 = stablehlo.multiply %v3121, %v3120 : tensor<32x1536x14x14xf32>
    %v3123 = stablehlo.add %v3118, %v3122 : tensor<32x1536x14x14xf32>
    %v3124 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v3125 = stablehlo.multiply %v3124, %v3123 : tensor<32x1536x14x14xf32>
    %v3126 = stablehlo.tanh %v3125 : tensor<32x1536x14x14xf32>
    %v3127 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v3128 = stablehlo.add %v3127, %v3126 : tensor<32x1536x14x14xf32>
    %v3129 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v3130 = stablehlo.multiply %v3129, %v3128 : tensor<32x1536x14x14xf32>
    %v3131 = stablehlo.multiply %v3126, %v3126 : tensor<32x1536x14x14xf32>
    %v3132 = stablehlo.subtract %v3127, %v3131 : tensor<32x1536x14x14xf32>
    %v3133 = stablehlo.multiply %v3129, %v3118 : tensor<32x1536x14x14xf32>
    %v3134 = stablehlo.multiply %v3133, %v3132 : tensor<32x1536x14x14xf32>
    %v3135 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v3136 = stablehlo.multiply %v3135, %v3119 : tensor<32x1536x14x14xf32>
    %v3137 = stablehlo.add %v3127, %v3136 : tensor<32x1536x14x14xf32>
    %v3138 = stablehlo.multiply %v3124, %v3137 : tensor<32x1536x14x14xf32>
    %v3139 = stablehlo.multiply %v3134, %v3138 : tensor<32x1536x14x14xf32>
    %v3140 = stablehlo.add %v3130, %v3139 : tensor<32x1536x14x14xf32>
    %v3141 = stablehlo.multiply %v3117, %v3140 : tensor<32x1536x14x14xf32>
    %v3142 = stablehlo.reshape %v3141 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3143 = stablehlo.reshape %v3142 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3144 = stablehlo.transpose %s2b3eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3145 = stablehlo.reverse %v3144, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3146 = stablehlo.convolution(%v3143, %v3145)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3147 = stablehlo.reshape %v3146 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3148 = stablehlo.reshape %v769 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3149 = stablehlo.transpose %v3148, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3150 = stablehlo.reshape %v3149 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3151 = stablehlo.reshape %v3147 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3152 = stablehlo.transpose %v3151, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3153 = stablehlo.reshape %v3152 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3154 = stablehlo.reshape %v3153 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3155 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3156 = stablehlo.multiply %v3154, %v3155 : tensor<32x196x384xf32>
    %v3157 = stablehlo.reshape %v3156 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3158 = stablehlo.reshape %v3157 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3159 = stablehlo.reshape %v3150 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3161 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3162 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3163 = stablehlo.reduce(%v3159 init: %v3160) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3164 = stablehlo.broadcast_in_dim %v3163, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3165 = stablehlo.divide %v3164, %v3161 : tensor<32x196x384xf32>
    %v3166 = stablehlo.subtract %v3159, %v3165 : tensor<32x196x384xf32>
    %v3167 = stablehlo.multiply %v3166, %v3166 : tensor<32x196x384xf32>
    %v3168 = stablehlo.reduce(%v3167 init: %v3160) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3169 = stablehlo.broadcast_in_dim %v3168, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3170 = stablehlo.divide %v3169, %v3161 : tensor<32x196x384xf32>
    %v3171 = stablehlo.add %v3170, %v3162 : tensor<32x196x384xf32>
    %v3172 = stablehlo.rsqrt %v3171 : tensor<32x196x384xf32>
    %v3173 = stablehlo.multiply %v3166, %v3172 : tensor<32x196x384xf32>
    %v3174 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3175 = stablehlo.multiply %v3174, %v3158 : tensor<32x196x384xf32>
    %v3176 = stablehlo.reduce(%v3175 init: %v3160) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3177 = stablehlo.broadcast_in_dim %v3176, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3178 = stablehlo.multiply %v3173, %v3175 : tensor<32x196x384xf32>
    %v3179 = stablehlo.reduce(%v3178 init: %v3160) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3180 = stablehlo.broadcast_in_dim %v3179, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3181 = stablehlo.multiply %v3175, %v3161 : tensor<32x196x384xf32>
    %v3182 = stablehlo.subtract %v3181, %v3177 : tensor<32x196x384xf32>
    %v3183 = stablehlo.multiply %v3173, %v3180 : tensor<32x196x384xf32>
    %v3184 = stablehlo.subtract %v3182, %v3183 : tensor<32x196x384xf32>
    %v3185 = stablehlo.divide %v3172, %v3161 : tensor<32x196x384xf32>
    %v3186 = stablehlo.multiply %v3185, %v3184 : tensor<32x196x384xf32>
    %v3187 = stablehlo.reshape %v3186 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3188 = stablehlo.reshape %v3187 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3189 = stablehlo.transpose %v3188, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3190 = stablehlo.reshape %v3189 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3191 = stablehlo.reshape %v3190 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3192 = stablehlo.reverse %s2b3dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3193 = stablehlo.convolution(%v3191, %v3192)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3194 = stablehlo.reshape %v3193 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3195 = stablehlo.reshape %v3194 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3196 = stablehlo.reshape %v3018 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3197 = stablehlo.add %v3195, %v3196 : tensor<32x384x14x14xf32>
    %v3198 = stablehlo.reshape %v3197 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3200 = stablehlo.reshape %v828 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3201 = stablehlo.reshape %v3018 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3202 = stablehlo.multiply %v3200, %v3201 : tensor<32x384x14x14xf32>
    %v3203 = stablehlo.reduce(%v3202 init: %v3199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3204 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3205 = stablehlo.multiply %v3203, %v3204 : tensor<384xf32>
    %v3206 = stablehlo.subtract %s2b3lg, %v3205 : tensor<384xf32>
    %v3207 = stablehlo.reshape %v823 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3208 = stablehlo.reshape %v3111 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3209 = stablehlo.transpose %v3207, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3210 = stablehlo.transpose %v3208, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3211 = stablehlo.convolution(%v3209, %v3210)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3212 = stablehlo.transpose %v3211, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3213 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v3214 = stablehlo.multiply %v3212, %v3213 : tensor<384x1536x1x1xf32>
    %v3215 = stablehlo.subtract %s2b3pW, %v3214 : tensor<384x1536x1x1xf32>
    %v3216 = stablehlo.reshape %v3111 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3218 = stablehlo.reduce(%v3216 init: %v3217) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3219 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3220 = stablehlo.multiply %v3218, %v3219 : tensor<384xf32>
    %v3221 = stablehlo.subtract %s2b3pb, %v3220 : tensor<384xf32>
    %v3222 = stablehlo.reshape %v803 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3223 = stablehlo.reshape %v3142 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3224 = stablehlo.transpose %v3222, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3225 = stablehlo.transpose %v3223, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3226 = stablehlo.convolution(%v3224, %v3225)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3227 = stablehlo.transpose %v3226, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3228 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v3229 = stablehlo.multiply %v3227, %v3228 : tensor<1536x384x1x1xf32>
    %v3230 = stablehlo.subtract %s2b3eW, %v3229 : tensor<1536x384x1x1xf32>
    %v3231 = stablehlo.reshape %v3142 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3233 = stablehlo.reduce(%v3231 init: %v3232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3234 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3235 = stablehlo.multiply %v3233, %v3234 : tensor<1536xf32>
    %v3236 = stablehlo.subtract %s2b3eb, %v3235 : tensor<1536xf32>
    %v3237 = stablehlo.reshape %v769 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3238 = stablehlo.transpose %v3237, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3239 = stablehlo.reshape %v3238 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3240 = stablehlo.reshape %v3147 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3241 = stablehlo.transpose %v3240, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3242 = stablehlo.reshape %v3241 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3243 = stablehlo.reshape %v3239 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3244 = stablehlo.reshape %v3242 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3246 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3247 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3248 = stablehlo.reduce(%v3243 init: %v3245) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3249 = stablehlo.broadcast_in_dim %v3248, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3250 = stablehlo.divide %v3249, %v3246 : tensor<32x196x384xf32>
    %v3251 = stablehlo.subtract %v3243, %v3250 : tensor<32x196x384xf32>
    %v3252 = stablehlo.multiply %v3251, %v3251 : tensor<32x196x384xf32>
    %v3253 = stablehlo.reduce(%v3252 init: %v3245) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3254 = stablehlo.broadcast_in_dim %v3253, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3255 = stablehlo.divide %v3254, %v3246 : tensor<32x196x384xf32>
    %v3256 = stablehlo.add %v3255, %v3247 : tensor<32x196x384xf32>
    %v3257 = stablehlo.rsqrt %v3256 : tensor<32x196x384xf32>
    %v3258 = stablehlo.multiply %v3251, %v3257 : tensor<32x196x384xf32>
    %v3259 = stablehlo.multiply %v3244, %v3258 : tensor<32x196x384xf32>
    %v3260 = stablehlo.reduce(%v3259 init: %v3245) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3261 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3262 = stablehlo.multiply %v3260, %v3261 : tensor<384xf32>
    %v3263 = stablehlo.subtract %s2b3ng, %v3262 : tensor<384xf32>
    %v3264 = stablehlo.reshape %v3147 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3265 = stablehlo.transpose %v3264, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3266 = stablehlo.reshape %v3265 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3267 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3268 = stablehlo.reshape %v3266 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3269 = stablehlo.reduce(%v3268 init: %v3267) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3270 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3271 = stablehlo.multiply %v3269, %v3270 : tensor<384xf32>
    %v3272 = stablehlo.subtract %s2b3nbt, %v3271 : tensor<384xf32>
    %v3273 = stablehlo.reshape %v764 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3274 = stablehlo.reshape %v3190 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3275 = stablehlo.transpose %v3273, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3276 = stablehlo.transpose %v3274, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3277 = stablehlo.convolution(%v3275, %v3276)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3278 = stablehlo.reshape %v3277 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3279 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3280 = stablehlo.multiply %v3278, %v3279 : tensor<384x1x7x7xf32>
    %v3281 = stablehlo.subtract %s2b3dW, %v3280 : tensor<384x1x7x7xf32>
    %v3282 = stablehlo.reshape %v3190 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3284 = stablehlo.reduce(%v3282 init: %v3283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3285 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3286 = stablehlo.multiply %v3284, %v3285 : tensor<384xf32>
    %v3287 = stablehlo.subtract %s2b3db, %v3286 : tensor<384xf32>
    %v3288 = stablehlo.reshape %v3198 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3289 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3290 = stablehlo.multiply %v3288, %v3289 : tensor<32x384x14x14xf32>
    %v3291 = stablehlo.reshape %v3290 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3292 = stablehlo.reshape %v3291 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3293 = stablehlo.transpose %s2b2pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3294 = stablehlo.reverse %v3293, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3295 = stablehlo.convolution(%v3292, %v3294)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3296 = stablehlo.reshape %v3295 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3297 = stablehlo.reshape %v3296 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3298 = stablehlo.reshape %v736 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3299 = stablehlo.multiply %v3298, %v3298 : tensor<32x1536x14x14xf32>
    %v3300 = stablehlo.multiply %v3299, %v3298 : tensor<32x1536x14x14xf32>
    %v3301 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v3302 = stablehlo.multiply %v3301, %v3300 : tensor<32x1536x14x14xf32>
    %v3303 = stablehlo.add %v3298, %v3302 : tensor<32x1536x14x14xf32>
    %v3304 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v3305 = stablehlo.multiply %v3304, %v3303 : tensor<32x1536x14x14xf32>
    %v3306 = stablehlo.tanh %v3305 : tensor<32x1536x14x14xf32>
    %v3307 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v3308 = stablehlo.add %v3307, %v3306 : tensor<32x1536x14x14xf32>
    %v3309 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v3310 = stablehlo.multiply %v3309, %v3308 : tensor<32x1536x14x14xf32>
    %v3311 = stablehlo.multiply %v3306, %v3306 : tensor<32x1536x14x14xf32>
    %v3312 = stablehlo.subtract %v3307, %v3311 : tensor<32x1536x14x14xf32>
    %v3313 = stablehlo.multiply %v3309, %v3298 : tensor<32x1536x14x14xf32>
    %v3314 = stablehlo.multiply %v3313, %v3312 : tensor<32x1536x14x14xf32>
    %v3315 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v3316 = stablehlo.multiply %v3315, %v3299 : tensor<32x1536x14x14xf32>
    %v3317 = stablehlo.add %v3307, %v3316 : tensor<32x1536x14x14xf32>
    %v3318 = stablehlo.multiply %v3304, %v3317 : tensor<32x1536x14x14xf32>
    %v3319 = stablehlo.multiply %v3314, %v3318 : tensor<32x1536x14x14xf32>
    %v3320 = stablehlo.add %v3310, %v3319 : tensor<32x1536x14x14xf32>
    %v3321 = stablehlo.multiply %v3297, %v3320 : tensor<32x1536x14x14xf32>
    %v3322 = stablehlo.reshape %v3321 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3323 = stablehlo.reshape %v3322 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3324 = stablehlo.transpose %s2b2eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3325 = stablehlo.reverse %v3324, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3326 = stablehlo.convolution(%v3323, %v3325)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3327 = stablehlo.reshape %v3326 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3328 = stablehlo.reshape %v697 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3329 = stablehlo.transpose %v3328, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3330 = stablehlo.reshape %v3329 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3331 = stablehlo.reshape %v3327 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3332 = stablehlo.transpose %v3331, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3333 = stablehlo.reshape %v3332 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3334 = stablehlo.reshape %v3333 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3335 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3336 = stablehlo.multiply %v3334, %v3335 : tensor<32x196x384xf32>
    %v3337 = stablehlo.reshape %v3336 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3338 = stablehlo.reshape %v3337 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3339 = stablehlo.reshape %v3330 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3340 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3341 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3342 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3343 = stablehlo.reduce(%v3339 init: %v3340) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3344 = stablehlo.broadcast_in_dim %v3343, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3345 = stablehlo.divide %v3344, %v3341 : tensor<32x196x384xf32>
    %v3346 = stablehlo.subtract %v3339, %v3345 : tensor<32x196x384xf32>
    %v3347 = stablehlo.multiply %v3346, %v3346 : tensor<32x196x384xf32>
    %v3348 = stablehlo.reduce(%v3347 init: %v3340) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3349 = stablehlo.broadcast_in_dim %v3348, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3350 = stablehlo.divide %v3349, %v3341 : tensor<32x196x384xf32>
    %v3351 = stablehlo.add %v3350, %v3342 : tensor<32x196x384xf32>
    %v3352 = stablehlo.rsqrt %v3351 : tensor<32x196x384xf32>
    %v3353 = stablehlo.multiply %v3346, %v3352 : tensor<32x196x384xf32>
    %v3354 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3355 = stablehlo.multiply %v3354, %v3338 : tensor<32x196x384xf32>
    %v3356 = stablehlo.reduce(%v3355 init: %v3340) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3357 = stablehlo.broadcast_in_dim %v3356, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3358 = stablehlo.multiply %v3353, %v3355 : tensor<32x196x384xf32>
    %v3359 = stablehlo.reduce(%v3358 init: %v3340) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3360 = stablehlo.broadcast_in_dim %v3359, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3361 = stablehlo.multiply %v3355, %v3341 : tensor<32x196x384xf32>
    %v3362 = stablehlo.subtract %v3361, %v3357 : tensor<32x196x384xf32>
    %v3363 = stablehlo.multiply %v3353, %v3360 : tensor<32x196x384xf32>
    %v3364 = stablehlo.subtract %v3362, %v3363 : tensor<32x196x384xf32>
    %v3365 = stablehlo.divide %v3352, %v3341 : tensor<32x196x384xf32>
    %v3366 = stablehlo.multiply %v3365, %v3364 : tensor<32x196x384xf32>
    %v3367 = stablehlo.reshape %v3366 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3368 = stablehlo.reshape %v3367 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3369 = stablehlo.transpose %v3368, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3370 = stablehlo.reshape %v3369 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3371 = stablehlo.reshape %v3370 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3372 = stablehlo.reverse %s2b2dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3373 = stablehlo.convolution(%v3371, %v3372)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3374 = stablehlo.reshape %v3373 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3375 = stablehlo.reshape %v3374 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3376 = stablehlo.reshape %v3198 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3377 = stablehlo.add %v3375, %v3376 : tensor<32x384x14x14xf32>
    %v3378 = stablehlo.reshape %v3377 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3379 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3380 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3381 = stablehlo.reshape %v3198 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3382 = stablehlo.multiply %v3380, %v3381 : tensor<32x384x14x14xf32>
    %v3383 = stablehlo.reduce(%v3382 init: %v3379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3384 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3385 = stablehlo.multiply %v3383, %v3384 : tensor<384xf32>
    %v3386 = stablehlo.subtract %s2b2lg, %v3385 : tensor<384xf32>
    %v3387 = stablehlo.reshape %v751 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3388 = stablehlo.reshape %v3291 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3389 = stablehlo.transpose %v3387, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3390 = stablehlo.transpose %v3388, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3391 = stablehlo.convolution(%v3389, %v3390)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3392 = stablehlo.transpose %v3391, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3393 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v3394 = stablehlo.multiply %v3392, %v3393 : tensor<384x1536x1x1xf32>
    %v3395 = stablehlo.subtract %s2b2pW, %v3394 : tensor<384x1536x1x1xf32>
    %v3396 = stablehlo.reshape %v3291 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3397 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3398 = stablehlo.reduce(%v3396 init: %v3397) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3399 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3400 = stablehlo.multiply %v3398, %v3399 : tensor<384xf32>
    %v3401 = stablehlo.subtract %s2b2pb, %v3400 : tensor<384xf32>
    %v3402 = stablehlo.reshape %v731 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3403 = stablehlo.reshape %v3322 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3404 = stablehlo.transpose %v3402, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3405 = stablehlo.transpose %v3403, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3406 = stablehlo.convolution(%v3404, %v3405)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3407 = stablehlo.transpose %v3406, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3408 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v3409 = stablehlo.multiply %v3407, %v3408 : tensor<1536x384x1x1xf32>
    %v3410 = stablehlo.subtract %s2b2eW, %v3409 : tensor<1536x384x1x1xf32>
    %v3411 = stablehlo.reshape %v3322 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3413 = stablehlo.reduce(%v3411 init: %v3412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3414 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3415 = stablehlo.multiply %v3413, %v3414 : tensor<1536xf32>
    %v3416 = stablehlo.subtract %s2b2eb, %v3415 : tensor<1536xf32>
    %v3417 = stablehlo.reshape %v697 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3418 = stablehlo.transpose %v3417, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3419 = stablehlo.reshape %v3418 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3420 = stablehlo.reshape %v3327 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3421 = stablehlo.transpose %v3420, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3422 = stablehlo.reshape %v3421 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3423 = stablehlo.reshape %v3419 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3424 = stablehlo.reshape %v3422 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3426 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3427 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3428 = stablehlo.reduce(%v3423 init: %v3425) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3429 = stablehlo.broadcast_in_dim %v3428, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3430 = stablehlo.divide %v3429, %v3426 : tensor<32x196x384xf32>
    %v3431 = stablehlo.subtract %v3423, %v3430 : tensor<32x196x384xf32>
    %v3432 = stablehlo.multiply %v3431, %v3431 : tensor<32x196x384xf32>
    %v3433 = stablehlo.reduce(%v3432 init: %v3425) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3434 = stablehlo.broadcast_in_dim %v3433, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3435 = stablehlo.divide %v3434, %v3426 : tensor<32x196x384xf32>
    %v3436 = stablehlo.add %v3435, %v3427 : tensor<32x196x384xf32>
    %v3437 = stablehlo.rsqrt %v3436 : tensor<32x196x384xf32>
    %v3438 = stablehlo.multiply %v3431, %v3437 : tensor<32x196x384xf32>
    %v3439 = stablehlo.multiply %v3424, %v3438 : tensor<32x196x384xf32>
    %v3440 = stablehlo.reduce(%v3439 init: %v3425) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3441 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3442 = stablehlo.multiply %v3440, %v3441 : tensor<384xf32>
    %v3443 = stablehlo.subtract %s2b2ng, %v3442 : tensor<384xf32>
    %v3444 = stablehlo.reshape %v3327 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3445 = stablehlo.transpose %v3444, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3446 = stablehlo.reshape %v3445 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3448 = stablehlo.reshape %v3446 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3449 = stablehlo.reduce(%v3448 init: %v3447) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3450 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3451 = stablehlo.multiply %v3449, %v3450 : tensor<384xf32>
    %v3452 = stablehlo.subtract %s2b2nbt, %v3451 : tensor<384xf32>
    %v3453 = stablehlo.reshape %v692 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3454 = stablehlo.reshape %v3370 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3455 = stablehlo.transpose %v3453, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3456 = stablehlo.transpose %v3454, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3457 = stablehlo.convolution(%v3455, %v3456)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3458 = stablehlo.reshape %v3457 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3459 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3460 = stablehlo.multiply %v3458, %v3459 : tensor<384x1x7x7xf32>
    %v3461 = stablehlo.subtract %s2b2dW, %v3460 : tensor<384x1x7x7xf32>
    %v3462 = stablehlo.reshape %v3370 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3464 = stablehlo.reduce(%v3462 init: %v3463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3465 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3466 = stablehlo.multiply %v3464, %v3465 : tensor<384xf32>
    %v3467 = stablehlo.subtract %s2b2db, %v3466 : tensor<384xf32>
    %v3468 = stablehlo.reshape %v3378 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3469 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3470 = stablehlo.multiply %v3468, %v3469 : tensor<32x384x14x14xf32>
    %v3471 = stablehlo.reshape %v3470 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3472 = stablehlo.reshape %v3471 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3473 = stablehlo.transpose %s2b1pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3474 = stablehlo.reverse %v3473, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3475 = stablehlo.convolution(%v3472, %v3474)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3476 = stablehlo.reshape %v3475 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3477 = stablehlo.reshape %v3476 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3478 = stablehlo.reshape %v664 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3479 = stablehlo.multiply %v3478, %v3478 : tensor<32x1536x14x14xf32>
    %v3480 = stablehlo.multiply %v3479, %v3478 : tensor<32x1536x14x14xf32>
    %v3481 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v3482 = stablehlo.multiply %v3481, %v3480 : tensor<32x1536x14x14xf32>
    %v3483 = stablehlo.add %v3478, %v3482 : tensor<32x1536x14x14xf32>
    %v3484 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v3485 = stablehlo.multiply %v3484, %v3483 : tensor<32x1536x14x14xf32>
    %v3486 = stablehlo.tanh %v3485 : tensor<32x1536x14x14xf32>
    %v3487 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v3488 = stablehlo.add %v3487, %v3486 : tensor<32x1536x14x14xf32>
    %v3489 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v3490 = stablehlo.multiply %v3489, %v3488 : tensor<32x1536x14x14xf32>
    %v3491 = stablehlo.multiply %v3486, %v3486 : tensor<32x1536x14x14xf32>
    %v3492 = stablehlo.subtract %v3487, %v3491 : tensor<32x1536x14x14xf32>
    %v3493 = stablehlo.multiply %v3489, %v3478 : tensor<32x1536x14x14xf32>
    %v3494 = stablehlo.multiply %v3493, %v3492 : tensor<32x1536x14x14xf32>
    %v3495 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v3496 = stablehlo.multiply %v3495, %v3479 : tensor<32x1536x14x14xf32>
    %v3497 = stablehlo.add %v3487, %v3496 : tensor<32x1536x14x14xf32>
    %v3498 = stablehlo.multiply %v3484, %v3497 : tensor<32x1536x14x14xf32>
    %v3499 = stablehlo.multiply %v3494, %v3498 : tensor<32x1536x14x14xf32>
    %v3500 = stablehlo.add %v3490, %v3499 : tensor<32x1536x14x14xf32>
    %v3501 = stablehlo.multiply %v3477, %v3500 : tensor<32x1536x14x14xf32>
    %v3502 = stablehlo.reshape %v3501 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3503 = stablehlo.reshape %v3502 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3504 = stablehlo.transpose %s2b1eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3505 = stablehlo.reverse %v3504, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3506 = stablehlo.convolution(%v3503, %v3505)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3507 = stablehlo.reshape %v3506 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3508 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3509 = stablehlo.transpose %v3508, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3510 = stablehlo.reshape %v3509 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3511 = stablehlo.reshape %v3507 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3512 = stablehlo.transpose %v3511, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3513 = stablehlo.reshape %v3512 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3514 = stablehlo.reshape %v3513 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3515 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3516 = stablehlo.multiply %v3514, %v3515 : tensor<32x196x384xf32>
    %v3517 = stablehlo.reshape %v3516 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3518 = stablehlo.reshape %v3517 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3519 = stablehlo.reshape %v3510 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3520 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3521 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3522 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3523 = stablehlo.reduce(%v3519 init: %v3520) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3524 = stablehlo.broadcast_in_dim %v3523, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3525 = stablehlo.divide %v3524, %v3521 : tensor<32x196x384xf32>
    %v3526 = stablehlo.subtract %v3519, %v3525 : tensor<32x196x384xf32>
    %v3527 = stablehlo.multiply %v3526, %v3526 : tensor<32x196x384xf32>
    %v3528 = stablehlo.reduce(%v3527 init: %v3520) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3529 = stablehlo.broadcast_in_dim %v3528, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3530 = stablehlo.divide %v3529, %v3521 : tensor<32x196x384xf32>
    %v3531 = stablehlo.add %v3530, %v3522 : tensor<32x196x384xf32>
    %v3532 = stablehlo.rsqrt %v3531 : tensor<32x196x384xf32>
    %v3533 = stablehlo.multiply %v3526, %v3532 : tensor<32x196x384xf32>
    %v3534 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3535 = stablehlo.multiply %v3534, %v3518 : tensor<32x196x384xf32>
    %v3536 = stablehlo.reduce(%v3535 init: %v3520) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3537 = stablehlo.broadcast_in_dim %v3536, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3538 = stablehlo.multiply %v3533, %v3535 : tensor<32x196x384xf32>
    %v3539 = stablehlo.reduce(%v3538 init: %v3520) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3540 = stablehlo.broadcast_in_dim %v3539, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3541 = stablehlo.multiply %v3535, %v3521 : tensor<32x196x384xf32>
    %v3542 = stablehlo.subtract %v3541, %v3537 : tensor<32x196x384xf32>
    %v3543 = stablehlo.multiply %v3533, %v3540 : tensor<32x196x384xf32>
    %v3544 = stablehlo.subtract %v3542, %v3543 : tensor<32x196x384xf32>
    %v3545 = stablehlo.divide %v3532, %v3521 : tensor<32x196x384xf32>
    %v3546 = stablehlo.multiply %v3545, %v3544 : tensor<32x196x384xf32>
    %v3547 = stablehlo.reshape %v3546 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3548 = stablehlo.reshape %v3547 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3549 = stablehlo.transpose %v3548, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3550 = stablehlo.reshape %v3549 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3551 = stablehlo.reshape %v3550 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3552 = stablehlo.reverse %s2b1dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3553 = stablehlo.convolution(%v3551, %v3552)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3554 = stablehlo.reshape %v3553 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3555 = stablehlo.reshape %v3554 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3556 = stablehlo.reshape %v3378 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3557 = stablehlo.add %v3555, %v3556 : tensor<32x384x14x14xf32>
    %v3558 = stablehlo.reshape %v3557 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3560 = stablehlo.reshape %v684 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3561 = stablehlo.reshape %v3378 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3562 = stablehlo.multiply %v3560, %v3561 : tensor<32x384x14x14xf32>
    %v3563 = stablehlo.reduce(%v3562 init: %v3559) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3564 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3565 = stablehlo.multiply %v3563, %v3564 : tensor<384xf32>
    %v3566 = stablehlo.subtract %s2b1lg, %v3565 : tensor<384xf32>
    %v3567 = stablehlo.reshape %v679 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3568 = stablehlo.reshape %v3471 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3569 = stablehlo.transpose %v3567, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3570 = stablehlo.transpose %v3568, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3571 = stablehlo.convolution(%v3569, %v3570)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3572 = stablehlo.transpose %v3571, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3573 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v3574 = stablehlo.multiply %v3572, %v3573 : tensor<384x1536x1x1xf32>
    %v3575 = stablehlo.subtract %s2b1pW, %v3574 : tensor<384x1536x1x1xf32>
    %v3576 = stablehlo.reshape %v3471 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3577 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3578 = stablehlo.reduce(%v3576 init: %v3577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3579 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3580 = stablehlo.multiply %v3578, %v3579 : tensor<384xf32>
    %v3581 = stablehlo.subtract %s2b1pb, %v3580 : tensor<384xf32>
    %v3582 = stablehlo.reshape %v659 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3583 = stablehlo.reshape %v3502 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3584 = stablehlo.transpose %v3582, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3585 = stablehlo.transpose %v3583, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3586 = stablehlo.convolution(%v3584, %v3585)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3587 = stablehlo.transpose %v3586, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3588 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v3589 = stablehlo.multiply %v3587, %v3588 : tensor<1536x384x1x1xf32>
    %v3590 = stablehlo.subtract %s2b1eW, %v3589 : tensor<1536x384x1x1xf32>
    %v3591 = stablehlo.reshape %v3502 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3593 = stablehlo.reduce(%v3591 init: %v3592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3594 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3595 = stablehlo.multiply %v3593, %v3594 : tensor<1536xf32>
    %v3596 = stablehlo.subtract %s2b1eb, %v3595 : tensor<1536xf32>
    %v3597 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3598 = stablehlo.transpose %v3597, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3599 = stablehlo.reshape %v3598 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3600 = stablehlo.reshape %v3507 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3601 = stablehlo.transpose %v3600, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3602 = stablehlo.reshape %v3601 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3603 = stablehlo.reshape %v3599 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3604 = stablehlo.reshape %v3602 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3605 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3606 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3607 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3608 = stablehlo.reduce(%v3603 init: %v3605) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3609 = stablehlo.broadcast_in_dim %v3608, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3610 = stablehlo.divide %v3609, %v3606 : tensor<32x196x384xf32>
    %v3611 = stablehlo.subtract %v3603, %v3610 : tensor<32x196x384xf32>
    %v3612 = stablehlo.multiply %v3611, %v3611 : tensor<32x196x384xf32>
    %v3613 = stablehlo.reduce(%v3612 init: %v3605) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3614 = stablehlo.broadcast_in_dim %v3613, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3615 = stablehlo.divide %v3614, %v3606 : tensor<32x196x384xf32>
    %v3616 = stablehlo.add %v3615, %v3607 : tensor<32x196x384xf32>
    %v3617 = stablehlo.rsqrt %v3616 : tensor<32x196x384xf32>
    %v3618 = stablehlo.multiply %v3611, %v3617 : tensor<32x196x384xf32>
    %v3619 = stablehlo.multiply %v3604, %v3618 : tensor<32x196x384xf32>
    %v3620 = stablehlo.reduce(%v3619 init: %v3605) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3621 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3622 = stablehlo.multiply %v3620, %v3621 : tensor<384xf32>
    %v3623 = stablehlo.subtract %s2b1ng, %v3622 : tensor<384xf32>
    %v3624 = stablehlo.reshape %v3507 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3625 = stablehlo.transpose %v3624, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3626 = stablehlo.reshape %v3625 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3627 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3628 = stablehlo.reshape %v3626 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3629 = stablehlo.reduce(%v3628 init: %v3627) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3630 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3631 = stablehlo.multiply %v3629, %v3630 : tensor<384xf32>
    %v3632 = stablehlo.subtract %s2b1nbt, %v3631 : tensor<384xf32>
    %v3633 = stablehlo.reshape %v620 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3634 = stablehlo.reshape %v3550 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3635 = stablehlo.transpose %v3633, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3636 = stablehlo.transpose %v3634, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3637 = stablehlo.convolution(%v3635, %v3636)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3638 = stablehlo.reshape %v3637 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3639 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3640 = stablehlo.multiply %v3638, %v3639 : tensor<384x1x7x7xf32>
    %v3641 = stablehlo.subtract %s2b1dW, %v3640 : tensor<384x1x7x7xf32>
    %v3642 = stablehlo.reshape %v3550 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3643 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3644 = stablehlo.reduce(%v3642 init: %v3643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3645 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3646 = stablehlo.multiply %v3644, %v3645 : tensor<384xf32>
    %v3647 = stablehlo.subtract %s2b1db, %v3646 : tensor<384xf32>
    %v3648 = stablehlo.reshape %v3558 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3649 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3650 = stablehlo.multiply %v3648, %v3649 : tensor<32x384x14x14xf32>
    %v3651 = stablehlo.reshape %v3650 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3652 = stablehlo.reshape %v3651 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3653 = stablehlo.transpose %s2b0pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3654 = stablehlo.reverse %v3653, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3655 = stablehlo.convolution(%v3652, %v3654)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3656 = stablehlo.reshape %v3655 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3657 = stablehlo.reshape %v3656 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3658 = stablehlo.reshape %v592 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3659 = stablehlo.multiply %v3658, %v3658 : tensor<32x1536x14x14xf32>
    %v3660 = stablehlo.multiply %v3659, %v3658 : tensor<32x1536x14x14xf32>
    %v3661 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v3662 = stablehlo.multiply %v3661, %v3660 : tensor<32x1536x14x14xf32>
    %v3663 = stablehlo.add %v3658, %v3662 : tensor<32x1536x14x14xf32>
    %v3664 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v3665 = stablehlo.multiply %v3664, %v3663 : tensor<32x1536x14x14xf32>
    %v3666 = stablehlo.tanh %v3665 : tensor<32x1536x14x14xf32>
    %v3667 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v3668 = stablehlo.add %v3667, %v3666 : tensor<32x1536x14x14xf32>
    %v3669 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v3670 = stablehlo.multiply %v3669, %v3668 : tensor<32x1536x14x14xf32>
    %v3671 = stablehlo.multiply %v3666, %v3666 : tensor<32x1536x14x14xf32>
    %v3672 = stablehlo.subtract %v3667, %v3671 : tensor<32x1536x14x14xf32>
    %v3673 = stablehlo.multiply %v3669, %v3658 : tensor<32x1536x14x14xf32>
    %v3674 = stablehlo.multiply %v3673, %v3672 : tensor<32x1536x14x14xf32>
    %v3675 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v3676 = stablehlo.multiply %v3675, %v3659 : tensor<32x1536x14x14xf32>
    %v3677 = stablehlo.add %v3667, %v3676 : tensor<32x1536x14x14xf32>
    %v3678 = stablehlo.multiply %v3664, %v3677 : tensor<32x1536x14x14xf32>
    %v3679 = stablehlo.multiply %v3674, %v3678 : tensor<32x1536x14x14xf32>
    %v3680 = stablehlo.add %v3670, %v3679 : tensor<32x1536x14x14xf32>
    %v3681 = stablehlo.multiply %v3657, %v3680 : tensor<32x1536x14x14xf32>
    %v3682 = stablehlo.reshape %v3681 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3683 = stablehlo.reshape %v3682 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3684 = stablehlo.transpose %s2b0eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3685 = stablehlo.reverse %v3684, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3686 = stablehlo.convolution(%v3683, %v3685)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3687 = stablehlo.reshape %v3686 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3688 = stablehlo.reshape %v553 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3689 = stablehlo.transpose %v3688, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3690 = stablehlo.reshape %v3689 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3691 = stablehlo.reshape %v3687 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3692 = stablehlo.transpose %v3691, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3693 = stablehlo.reshape %v3692 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3694 = stablehlo.reshape %v3693 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3695 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3696 = stablehlo.multiply %v3694, %v3695 : tensor<32x196x384xf32>
    %v3697 = stablehlo.reshape %v3696 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3698 = stablehlo.reshape %v3697 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3699 = stablehlo.reshape %v3690 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3700 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3701 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3702 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3703 = stablehlo.reduce(%v3699 init: %v3700) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3704 = stablehlo.broadcast_in_dim %v3703, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3705 = stablehlo.divide %v3704, %v3701 : tensor<32x196x384xf32>
    %v3706 = stablehlo.subtract %v3699, %v3705 : tensor<32x196x384xf32>
    %v3707 = stablehlo.multiply %v3706, %v3706 : tensor<32x196x384xf32>
    %v3708 = stablehlo.reduce(%v3707 init: %v3700) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3709 = stablehlo.broadcast_in_dim %v3708, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3710 = stablehlo.divide %v3709, %v3701 : tensor<32x196x384xf32>
    %v3711 = stablehlo.add %v3710, %v3702 : tensor<32x196x384xf32>
    %v3712 = stablehlo.rsqrt %v3711 : tensor<32x196x384xf32>
    %v3713 = stablehlo.multiply %v3706, %v3712 : tensor<32x196x384xf32>
    %v3714 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3715 = stablehlo.multiply %v3714, %v3698 : tensor<32x196x384xf32>
    %v3716 = stablehlo.reduce(%v3715 init: %v3700) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3717 = stablehlo.broadcast_in_dim %v3716, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3718 = stablehlo.multiply %v3713, %v3715 : tensor<32x196x384xf32>
    %v3719 = stablehlo.reduce(%v3718 init: %v3700) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3720 = stablehlo.broadcast_in_dim %v3719, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3721 = stablehlo.multiply %v3715, %v3701 : tensor<32x196x384xf32>
    %v3722 = stablehlo.subtract %v3721, %v3717 : tensor<32x196x384xf32>
    %v3723 = stablehlo.multiply %v3713, %v3720 : tensor<32x196x384xf32>
    %v3724 = stablehlo.subtract %v3722, %v3723 : tensor<32x196x384xf32>
    %v3725 = stablehlo.divide %v3712, %v3701 : tensor<32x196x384xf32>
    %v3726 = stablehlo.multiply %v3725, %v3724 : tensor<32x196x384xf32>
    %v3727 = stablehlo.reshape %v3726 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3728 = stablehlo.reshape %v3727 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3729 = stablehlo.transpose %v3728, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3730 = stablehlo.reshape %v3729 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3731 = stablehlo.reshape %v3730 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3732 = stablehlo.reverse %s2b0dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3733 = stablehlo.convolution(%v3731, %v3732)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3734 = stablehlo.reshape %v3733 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3735 = stablehlo.reshape %v3734 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3736 = stablehlo.reshape %v3558 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3737 = stablehlo.add %v3735, %v3736 : tensor<32x384x14x14xf32>
    %v3738 = stablehlo.reshape %v3737 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3740 = stablehlo.reshape %v612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3741 = stablehlo.reshape %v3558 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3742 = stablehlo.multiply %v3740, %v3741 : tensor<32x384x14x14xf32>
    %v3743 = stablehlo.reduce(%v3742 init: %v3739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3744 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3745 = stablehlo.multiply %v3743, %v3744 : tensor<384xf32>
    %v3746 = stablehlo.subtract %s2b0lg, %v3745 : tensor<384xf32>
    %v3747 = stablehlo.reshape %v607 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3748 = stablehlo.reshape %v3651 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3749 = stablehlo.transpose %v3747, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3750 = stablehlo.transpose %v3748, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3751 = stablehlo.convolution(%v3749, %v3750)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3752 = stablehlo.transpose %v3751, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3753 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v3754 = stablehlo.multiply %v3752, %v3753 : tensor<384x1536x1x1xf32>
    %v3755 = stablehlo.subtract %s2b0pW, %v3754 : tensor<384x1536x1x1xf32>
    %v3756 = stablehlo.reshape %v3651 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3758 = stablehlo.reduce(%v3756 init: %v3757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3759 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3760 = stablehlo.multiply %v3758, %v3759 : tensor<384xf32>
    %v3761 = stablehlo.subtract %s2b0pb, %v3760 : tensor<384xf32>
    %v3762 = stablehlo.reshape %v587 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3763 = stablehlo.reshape %v3682 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3764 = stablehlo.transpose %v3762, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3765 = stablehlo.transpose %v3763, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3766 = stablehlo.convolution(%v3764, %v3765)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3767 = stablehlo.transpose %v3766, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3768 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v3769 = stablehlo.multiply %v3767, %v3768 : tensor<1536x384x1x1xf32>
    %v3770 = stablehlo.subtract %s2b0eW, %v3769 : tensor<1536x384x1x1xf32>
    %v3771 = stablehlo.reshape %v3682 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3773 = stablehlo.reduce(%v3771 init: %v3772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3774 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3775 = stablehlo.multiply %v3773, %v3774 : tensor<1536xf32>
    %v3776 = stablehlo.subtract %s2b0eb, %v3775 : tensor<1536xf32>
    %v3777 = stablehlo.reshape %v553 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3778 = stablehlo.transpose %v3777, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3779 = stablehlo.reshape %v3778 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3780 = stablehlo.reshape %v3687 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3781 = stablehlo.transpose %v3780, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3782 = stablehlo.reshape %v3781 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3783 = stablehlo.reshape %v3779 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3784 = stablehlo.reshape %v3782 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3785 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3786 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3787 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3788 = stablehlo.reduce(%v3783 init: %v3785) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3789 = stablehlo.broadcast_in_dim %v3788, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3790 = stablehlo.divide %v3789, %v3786 : tensor<32x196x384xf32>
    %v3791 = stablehlo.subtract %v3783, %v3790 : tensor<32x196x384xf32>
    %v3792 = stablehlo.multiply %v3791, %v3791 : tensor<32x196x384xf32>
    %v3793 = stablehlo.reduce(%v3792 init: %v3785) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3794 = stablehlo.broadcast_in_dim %v3793, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3795 = stablehlo.divide %v3794, %v3786 : tensor<32x196x384xf32>
    %v3796 = stablehlo.add %v3795, %v3787 : tensor<32x196x384xf32>
    %v3797 = stablehlo.rsqrt %v3796 : tensor<32x196x384xf32>
    %v3798 = stablehlo.multiply %v3791, %v3797 : tensor<32x196x384xf32>
    %v3799 = stablehlo.multiply %v3784, %v3798 : tensor<32x196x384xf32>
    %v3800 = stablehlo.reduce(%v3799 init: %v3785) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3801 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3802 = stablehlo.multiply %v3800, %v3801 : tensor<384xf32>
    %v3803 = stablehlo.subtract %s2b0ng, %v3802 : tensor<384xf32>
    %v3804 = stablehlo.reshape %v3687 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3805 = stablehlo.transpose %v3804, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3806 = stablehlo.reshape %v3805 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3807 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3808 = stablehlo.reshape %v3806 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3809 = stablehlo.reduce(%v3808 init: %v3807) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3810 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3811 = stablehlo.multiply %v3809, %v3810 : tensor<384xf32>
    %v3812 = stablehlo.subtract %s2b0nbt, %v3811 : tensor<384xf32>
    %v3813 = stablehlo.reshape %v548 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3814 = stablehlo.reshape %v3730 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3815 = stablehlo.transpose %v3813, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3816 = stablehlo.transpose %v3814, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3817 = stablehlo.convolution(%v3815, %v3816)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3818 = stablehlo.reshape %v3817 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3819 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3820 = stablehlo.multiply %v3818, %v3819 : tensor<384x1x7x7xf32>
    %v3821 = stablehlo.subtract %s2b0dW, %v3820 : tensor<384x1x7x7xf32>
    %v3822 = stablehlo.reshape %v3730 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3823 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3824 = stablehlo.reduce(%v3822 init: %v3823) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3825 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3826 = stablehlo.multiply %v3824, %v3825 : tensor<384xf32>
    %v3827 = stablehlo.subtract %s2b0db, %v3826 : tensor<384xf32>
    %v3828 = stablehlo.reshape %v3738 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3830 = stablehlo.pad %v3828, %v3829, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x28x28xf32>
    %v3831 = stablehlo.transpose %d1W, dims = [1, 0, 2, 3] : (tensor<384x192x2x2xf32>) -> tensor<192x384x2x2xf32>
    %v3832 = stablehlo.reverse %v3831, dims = [2, 3] : tensor<192x384x2x2xf32>
    %v3833 = stablehlo.convolution(%v3830, %v3832)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x28x28xf32>, tensor<192x384x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v3834 = stablehlo.reshape %v3833 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3835 = stablehlo.reshape %v509 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3836 = stablehlo.transpose %v3835, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3837 = stablehlo.reshape %v3836 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3838 = stablehlo.reshape %v3834 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3839 = stablehlo.transpose %v3838, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3840 = stablehlo.reshape %v3839 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3841 = stablehlo.reshape %v3840 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3842 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v3843 = stablehlo.multiply %v3841, %v3842 : tensor<32x784x192xf32>
    %v3844 = stablehlo.reshape %v3843 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3845 = stablehlo.reshape %v3844 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3846 = stablehlo.reshape %v3837 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3848 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3849 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3850 = stablehlo.reduce(%v3846 init: %v3847) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3851 = stablehlo.broadcast_in_dim %v3850, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3852 = stablehlo.divide %v3851, %v3848 : tensor<32x784x192xf32>
    %v3853 = stablehlo.subtract %v3846, %v3852 : tensor<32x784x192xf32>
    %v3854 = stablehlo.multiply %v3853, %v3853 : tensor<32x784x192xf32>
    %v3855 = stablehlo.reduce(%v3854 init: %v3847) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3856 = stablehlo.broadcast_in_dim %v3855, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3857 = stablehlo.divide %v3856, %v3848 : tensor<32x784x192xf32>
    %v3858 = stablehlo.add %v3857, %v3849 : tensor<32x784x192xf32>
    %v3859 = stablehlo.rsqrt %v3858 : tensor<32x784x192xf32>
    %v3860 = stablehlo.multiply %v3853, %v3859 : tensor<32x784x192xf32>
    %v3861 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v3862 = stablehlo.multiply %v3861, %v3845 : tensor<32x784x192xf32>
    %v3863 = stablehlo.reduce(%v3862 init: %v3847) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3864 = stablehlo.broadcast_in_dim %v3863, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3865 = stablehlo.multiply %v3860, %v3862 : tensor<32x784x192xf32>
    %v3866 = stablehlo.reduce(%v3865 init: %v3847) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3867 = stablehlo.broadcast_in_dim %v3866, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3868 = stablehlo.multiply %v3862, %v3848 : tensor<32x784x192xf32>
    %v3869 = stablehlo.subtract %v3868, %v3864 : tensor<32x784x192xf32>
    %v3870 = stablehlo.multiply %v3860, %v3867 : tensor<32x784x192xf32>
    %v3871 = stablehlo.subtract %v3869, %v3870 : tensor<32x784x192xf32>
    %v3872 = stablehlo.divide %v3859, %v3848 : tensor<32x784x192xf32>
    %v3873 = stablehlo.multiply %v3872, %v3871 : tensor<32x784x192xf32>
    %v3874 = stablehlo.reshape %v3873 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3875 = stablehlo.reshape %v3874 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3876 = stablehlo.transpose %v3875, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v3877 = stablehlo.reshape %v3876 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v3878 = stablehlo.reshape %v3738 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3880 = stablehlo.reduce(%v3878 init: %v3879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3881 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3882 = stablehlo.multiply %v3880, %v3881 : tensor<384xf32>
    %v3883 = stablehlo.subtract %d1b, %v3882 : tensor<384xf32>
    %v3884 = stablehlo.reshape %v509 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3885 = stablehlo.transpose %v3884, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3886 = stablehlo.reshape %v3885 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3887 = stablehlo.reshape %v3834 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3888 = stablehlo.transpose %v3887, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3889 = stablehlo.reshape %v3888 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3890 = stablehlo.reshape %v3886 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3891 = stablehlo.reshape %v3889 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3892 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3893 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3894 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3895 = stablehlo.reduce(%v3890 init: %v3892) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3896 = stablehlo.broadcast_in_dim %v3895, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3897 = stablehlo.divide %v3896, %v3893 : tensor<32x784x192xf32>
    %v3898 = stablehlo.subtract %v3890, %v3897 : tensor<32x784x192xf32>
    %v3899 = stablehlo.multiply %v3898, %v3898 : tensor<32x784x192xf32>
    %v3900 = stablehlo.reduce(%v3899 init: %v3892) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3901 = stablehlo.broadcast_in_dim %v3900, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3902 = stablehlo.divide %v3901, %v3893 : tensor<32x784x192xf32>
    %v3903 = stablehlo.add %v3902, %v3894 : tensor<32x784x192xf32>
    %v3904 = stablehlo.rsqrt %v3903 : tensor<32x784x192xf32>
    %v3905 = stablehlo.multiply %v3898, %v3904 : tensor<32x784x192xf32>
    %v3906 = stablehlo.multiply %v3891, %v3905 : tensor<32x784x192xf32>
    %v3907 = stablehlo.reduce(%v3906 init: %v3892) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3908 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3909 = stablehlo.multiply %v3907, %v3908 : tensor<192xf32>
    %v3910 = stablehlo.subtract %d1ng, %v3909 : tensor<192xf32>
    %v3911 = stablehlo.reshape %v3834 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3912 = stablehlo.transpose %v3911, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3913 = stablehlo.reshape %v3912 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3914 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3915 = stablehlo.reshape %v3913 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3916 = stablehlo.reduce(%v3915 init: %v3914) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3917 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3918 = stablehlo.multiply %v3916, %v3917 : tensor<192xf32>
    %v3919 = stablehlo.subtract %d1nbt, %v3918 : tensor<192xf32>
    %v3920 = stablehlo.reshape %v543 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3921 = stablehlo.reshape %v3738 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3923 = stablehlo.pad %v3921, %v3922, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x27x27xf32>
    %v3924 = stablehlo.transpose %v3920, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3925 = stablehlo.transpose %v3923, dims = [1, 0, 2, 3] : (tensor<32x384x27x27xf32>) -> tensor<384x32x27x27xf32>
    %v3926 = stablehlo.convolution(%v3924, %v3925)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<384x32x27x27xf32>) -> tensor<192x384x2x2xf32>
    %v3927 = stablehlo.transpose %v3926, dims = [1, 0, 2, 3] : (tensor<192x384x2x2xf32>) -> tensor<384x192x2x2xf32>
    %v3928 = stablehlo.constant dense<0.1> : tensor<384x192x2x2xf32>
    %v3929 = stablehlo.multiply %v3927, %v3928 : tensor<384x192x2x2xf32>
    %v3930 = stablehlo.subtract %d1W, %v3929 : tensor<384x192x2x2xf32>
    %v3931 = stablehlo.reshape %v3877 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3932 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3933 = stablehlo.multiply %v3931, %v3932 : tensor<32x192x28x28xf32>
    %v3934 = stablehlo.reshape %v3933 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3935 = stablehlo.reshape %v3934 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3936 = stablehlo.transpose %s1b2pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3937 = stablehlo.reverse %v3936, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v3938 = stablehlo.convolution(%v3935, %v3937)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v3939 = stablehlo.reshape %v3938 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3940 = stablehlo.reshape %v3939 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3941 = stablehlo.reshape %v481 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3942 = stablehlo.multiply %v3941, %v3941 : tensor<32x768x28x28xf32>
    %v3943 = stablehlo.multiply %v3942, %v3941 : tensor<32x768x28x28xf32>
    %v3944 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v3945 = stablehlo.multiply %v3944, %v3943 : tensor<32x768x28x28xf32>
    %v3946 = stablehlo.add %v3941, %v3945 : tensor<32x768x28x28xf32>
    %v3947 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v3948 = stablehlo.multiply %v3947, %v3946 : tensor<32x768x28x28xf32>
    %v3949 = stablehlo.tanh %v3948 : tensor<32x768x28x28xf32>
    %v3950 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v3951 = stablehlo.add %v3950, %v3949 : tensor<32x768x28x28xf32>
    %v3952 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v3953 = stablehlo.multiply %v3952, %v3951 : tensor<32x768x28x28xf32>
    %v3954 = stablehlo.multiply %v3949, %v3949 : tensor<32x768x28x28xf32>
    %v3955 = stablehlo.subtract %v3950, %v3954 : tensor<32x768x28x28xf32>
    %v3956 = stablehlo.multiply %v3952, %v3941 : tensor<32x768x28x28xf32>
    %v3957 = stablehlo.multiply %v3956, %v3955 : tensor<32x768x28x28xf32>
    %v3958 = stablehlo.constant dense<0.134145> : tensor<32x768x28x28xf32>
    %v3959 = stablehlo.multiply %v3958, %v3942 : tensor<32x768x28x28xf32>
    %v3960 = stablehlo.add %v3950, %v3959 : tensor<32x768x28x28xf32>
    %v3961 = stablehlo.multiply %v3947, %v3960 : tensor<32x768x28x28xf32>
    %v3962 = stablehlo.multiply %v3957, %v3961 : tensor<32x768x28x28xf32>
    %v3963 = stablehlo.add %v3953, %v3962 : tensor<32x768x28x28xf32>
    %v3964 = stablehlo.multiply %v3940, %v3963 : tensor<32x768x28x28xf32>
    %v3965 = stablehlo.reshape %v3964 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3966 = stablehlo.reshape %v3965 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3967 = stablehlo.transpose %s1b2eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3968 = stablehlo.reverse %v3967, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3969 = stablehlo.convolution(%v3966, %v3968)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3970 = stablehlo.reshape %v3969 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3971 = stablehlo.reshape %v442 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3972 = stablehlo.transpose %v3971, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3973 = stablehlo.reshape %v3972 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3974 = stablehlo.reshape %v3970 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3975 = stablehlo.transpose %v3974, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3976 = stablehlo.reshape %v3975 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3977 = stablehlo.reshape %v3976 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3978 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v3979 = stablehlo.multiply %v3977, %v3978 : tensor<32x784x192xf32>
    %v3980 = stablehlo.reshape %v3979 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3981 = stablehlo.reshape %v3980 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3982 = stablehlo.reshape %v3973 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3983 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3984 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3985 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3986 = stablehlo.reduce(%v3982 init: %v3983) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3987 = stablehlo.broadcast_in_dim %v3986, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3988 = stablehlo.divide %v3987, %v3984 : tensor<32x784x192xf32>
    %v3989 = stablehlo.subtract %v3982, %v3988 : tensor<32x784x192xf32>
    %v3990 = stablehlo.multiply %v3989, %v3989 : tensor<32x784x192xf32>
    %v3991 = stablehlo.reduce(%v3990 init: %v3983) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3992 = stablehlo.broadcast_in_dim %v3991, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3993 = stablehlo.divide %v3992, %v3984 : tensor<32x784x192xf32>
    %v3994 = stablehlo.add %v3993, %v3985 : tensor<32x784x192xf32>
    %v3995 = stablehlo.rsqrt %v3994 : tensor<32x784x192xf32>
    %v3996 = stablehlo.multiply %v3989, %v3995 : tensor<32x784x192xf32>
    %v3997 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v3998 = stablehlo.multiply %v3997, %v3981 : tensor<32x784x192xf32>
    %v3999 = stablehlo.reduce(%v3998 init: %v3983) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4000 = stablehlo.broadcast_in_dim %v3999, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4001 = stablehlo.multiply %v3996, %v3998 : tensor<32x784x192xf32>
    %v4002 = stablehlo.reduce(%v4001 init: %v3983) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4003 = stablehlo.broadcast_in_dim %v4002, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4004 = stablehlo.multiply %v3998, %v3984 : tensor<32x784x192xf32>
    %v4005 = stablehlo.subtract %v4004, %v4000 : tensor<32x784x192xf32>
    %v4006 = stablehlo.multiply %v3996, %v4003 : tensor<32x784x192xf32>
    %v4007 = stablehlo.subtract %v4005, %v4006 : tensor<32x784x192xf32>
    %v4008 = stablehlo.divide %v3995, %v3984 : tensor<32x784x192xf32>
    %v4009 = stablehlo.multiply %v4008, %v4007 : tensor<32x784x192xf32>
    %v4010 = stablehlo.reshape %v4009 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4011 = stablehlo.reshape %v4010 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4012 = stablehlo.transpose %v4011, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v4013 = stablehlo.reshape %v4012 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v4014 = stablehlo.reshape %v4013 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4015 = stablehlo.reverse %s1b2dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v4016 = stablehlo.convolution(%v4014, %v4015)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v4017 = stablehlo.reshape %v4016 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4018 = stablehlo.reshape %v4017 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4019 = stablehlo.reshape %v3877 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4020 = stablehlo.add %v4018, %v4019 : tensor<32x192x28x28xf32>
    %v4021 = stablehlo.reshape %v4020 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4023 = stablehlo.reshape %v501 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4024 = stablehlo.reshape %v3877 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4025 = stablehlo.multiply %v4023, %v4024 : tensor<32x192x28x28xf32>
    %v4026 = stablehlo.reduce(%v4025 init: %v4022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4027 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4028 = stablehlo.multiply %v4026, %v4027 : tensor<192xf32>
    %v4029 = stablehlo.subtract %s1b2lg, %v4028 : tensor<192xf32>
    %v4030 = stablehlo.reshape %v496 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4031 = stablehlo.reshape %v3934 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4032 = stablehlo.transpose %v4030, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v4033 = stablehlo.transpose %v4031, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4034 = stablehlo.convolution(%v4032, %v4033)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v4035 = stablehlo.transpose %v4034, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v4036 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v4037 = stablehlo.multiply %v4035, %v4036 : tensor<192x768x1x1xf32>
    %v4038 = stablehlo.subtract %s1b2pW, %v4037 : tensor<192x768x1x1xf32>
    %v4039 = stablehlo.reshape %v3934 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4041 = stablehlo.reduce(%v4039 init: %v4040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4042 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4043 = stablehlo.multiply %v4041, %v4042 : tensor<192xf32>
    %v4044 = stablehlo.subtract %s1b2pb, %v4043 : tensor<192xf32>
    %v4045 = stablehlo.reshape %v476 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4046 = stablehlo.reshape %v3965 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4047 = stablehlo.transpose %v4045, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4048 = stablehlo.transpose %v4046, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v4049 = stablehlo.convolution(%v4047, %v4048)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v4050 = stablehlo.transpose %v4049, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v4051 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v4052 = stablehlo.multiply %v4050, %v4051 : tensor<768x192x1x1xf32>
    %v4053 = stablehlo.subtract %s1b2eW, %v4052 : tensor<768x192x1x1xf32>
    %v4054 = stablehlo.reshape %v3965 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4056 = stablehlo.reduce(%v4054 init: %v4055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v4057 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v4058 = stablehlo.multiply %v4056, %v4057 : tensor<768xf32>
    %v4059 = stablehlo.subtract %s1b2eb, %v4058 : tensor<768xf32>
    %v4060 = stablehlo.reshape %v442 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4061 = stablehlo.transpose %v4060, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4062 = stablehlo.reshape %v4061 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4063 = stablehlo.reshape %v3970 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4064 = stablehlo.transpose %v4063, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4065 = stablehlo.reshape %v4064 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4066 = stablehlo.reshape %v4062 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4067 = stablehlo.reshape %v4065 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4069 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v4070 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v4071 = stablehlo.reduce(%v4066 init: %v4068) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4072 = stablehlo.broadcast_in_dim %v4071, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4073 = stablehlo.divide %v4072, %v4069 : tensor<32x784x192xf32>
    %v4074 = stablehlo.subtract %v4066, %v4073 : tensor<32x784x192xf32>
    %v4075 = stablehlo.multiply %v4074, %v4074 : tensor<32x784x192xf32>
    %v4076 = stablehlo.reduce(%v4075 init: %v4068) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4077 = stablehlo.broadcast_in_dim %v4076, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4078 = stablehlo.divide %v4077, %v4069 : tensor<32x784x192xf32>
    %v4079 = stablehlo.add %v4078, %v4070 : tensor<32x784x192xf32>
    %v4080 = stablehlo.rsqrt %v4079 : tensor<32x784x192xf32>
    %v4081 = stablehlo.multiply %v4074, %v4080 : tensor<32x784x192xf32>
    %v4082 = stablehlo.multiply %v4067, %v4081 : tensor<32x784x192xf32>
    %v4083 = stablehlo.reduce(%v4082 init: %v4068) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4084 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4085 = stablehlo.multiply %v4083, %v4084 : tensor<192xf32>
    %v4086 = stablehlo.subtract %s1b2ng, %v4085 : tensor<192xf32>
    %v4087 = stablehlo.reshape %v3970 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4088 = stablehlo.transpose %v4087, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4089 = stablehlo.reshape %v4088 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4090 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4091 = stablehlo.reshape %v4089 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4092 = stablehlo.reduce(%v4091 init: %v4090) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4093 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4094 = stablehlo.multiply %v4092, %v4093 : tensor<192xf32>
    %v4095 = stablehlo.subtract %s1b2nbt, %v4094 : tensor<192xf32>
    %v4096 = stablehlo.reshape %v437 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4097 = stablehlo.reshape %v4013 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4098 = stablehlo.transpose %v4096, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4099 = stablehlo.transpose %v4097, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4100 = stablehlo.convolution(%v4098, %v4099)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v4101 = stablehlo.reshape %v4100 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v4102 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v4103 = stablehlo.multiply %v4101, %v4102 : tensor<192x1x7x7xf32>
    %v4104 = stablehlo.subtract %s1b2dW, %v4103 : tensor<192x1x7x7xf32>
    %v4105 = stablehlo.reshape %v4013 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4107 = stablehlo.reduce(%v4105 init: %v4106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4108 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4109 = stablehlo.multiply %v4107, %v4108 : tensor<192xf32>
    %v4110 = stablehlo.subtract %s1b2db, %v4109 : tensor<192xf32>
    %v4111 = stablehlo.reshape %v4021 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4112 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4113 = stablehlo.multiply %v4111, %v4112 : tensor<32x192x28x28xf32>
    %v4114 = stablehlo.reshape %v4113 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4115 = stablehlo.reshape %v4114 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4116 = stablehlo.transpose %s1b1pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v4117 = stablehlo.reverse %v4116, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v4118 = stablehlo.convolution(%v4115, %v4117)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v4119 = stablehlo.reshape %v4118 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v4120 = stablehlo.reshape %v4119 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4121 = stablehlo.reshape %v409 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4122 = stablehlo.multiply %v4121, %v4121 : tensor<32x768x28x28xf32>
    %v4123 = stablehlo.multiply %v4122, %v4121 : tensor<32x768x28x28xf32>
    %v4124 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v4125 = stablehlo.multiply %v4124, %v4123 : tensor<32x768x28x28xf32>
    %v4126 = stablehlo.add %v4121, %v4125 : tensor<32x768x28x28xf32>
    %v4127 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v4128 = stablehlo.multiply %v4127, %v4126 : tensor<32x768x28x28xf32>
    %v4129 = stablehlo.tanh %v4128 : tensor<32x768x28x28xf32>
    %v4130 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v4131 = stablehlo.add %v4130, %v4129 : tensor<32x768x28x28xf32>
    %v4132 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v4133 = stablehlo.multiply %v4132, %v4131 : tensor<32x768x28x28xf32>
    %v4134 = stablehlo.multiply %v4129, %v4129 : tensor<32x768x28x28xf32>
    %v4135 = stablehlo.subtract %v4130, %v4134 : tensor<32x768x28x28xf32>
    %v4136 = stablehlo.multiply %v4132, %v4121 : tensor<32x768x28x28xf32>
    %v4137 = stablehlo.multiply %v4136, %v4135 : tensor<32x768x28x28xf32>
    %v4138 = stablehlo.constant dense<0.134145> : tensor<32x768x28x28xf32>
    %v4139 = stablehlo.multiply %v4138, %v4122 : tensor<32x768x28x28xf32>
    %v4140 = stablehlo.add %v4130, %v4139 : tensor<32x768x28x28xf32>
    %v4141 = stablehlo.multiply %v4127, %v4140 : tensor<32x768x28x28xf32>
    %v4142 = stablehlo.multiply %v4137, %v4141 : tensor<32x768x28x28xf32>
    %v4143 = stablehlo.add %v4133, %v4142 : tensor<32x768x28x28xf32>
    %v4144 = stablehlo.multiply %v4120, %v4143 : tensor<32x768x28x28xf32>
    %v4145 = stablehlo.reshape %v4144 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v4146 = stablehlo.reshape %v4145 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4147 = stablehlo.transpose %s1b1eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v4148 = stablehlo.reverse %v4147, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v4149 = stablehlo.convolution(%v4146, %v4148)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4150 = stablehlo.reshape %v4149 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4151 = stablehlo.reshape %v370 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4152 = stablehlo.transpose %v4151, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4153 = stablehlo.reshape %v4152 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4154 = stablehlo.reshape %v4150 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4155 = stablehlo.transpose %v4154, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4156 = stablehlo.reshape %v4155 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4157 = stablehlo.reshape %v4156 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4158 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v4159 = stablehlo.multiply %v4157, %v4158 : tensor<32x784x192xf32>
    %v4160 = stablehlo.reshape %v4159 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4161 = stablehlo.reshape %v4160 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4162 = stablehlo.reshape %v4153 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4164 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v4165 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v4166 = stablehlo.reduce(%v4162 init: %v4163) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4167 = stablehlo.broadcast_in_dim %v4166, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4168 = stablehlo.divide %v4167, %v4164 : tensor<32x784x192xf32>
    %v4169 = stablehlo.subtract %v4162, %v4168 : tensor<32x784x192xf32>
    %v4170 = stablehlo.multiply %v4169, %v4169 : tensor<32x784x192xf32>
    %v4171 = stablehlo.reduce(%v4170 init: %v4163) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4172 = stablehlo.broadcast_in_dim %v4171, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4173 = stablehlo.divide %v4172, %v4164 : tensor<32x784x192xf32>
    %v4174 = stablehlo.add %v4173, %v4165 : tensor<32x784x192xf32>
    %v4175 = stablehlo.rsqrt %v4174 : tensor<32x784x192xf32>
    %v4176 = stablehlo.multiply %v4169, %v4175 : tensor<32x784x192xf32>
    %v4177 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v4178 = stablehlo.multiply %v4177, %v4161 : tensor<32x784x192xf32>
    %v4179 = stablehlo.reduce(%v4178 init: %v4163) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4180 = stablehlo.broadcast_in_dim %v4179, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4181 = stablehlo.multiply %v4176, %v4178 : tensor<32x784x192xf32>
    %v4182 = stablehlo.reduce(%v4181 init: %v4163) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4183 = stablehlo.broadcast_in_dim %v4182, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4184 = stablehlo.multiply %v4178, %v4164 : tensor<32x784x192xf32>
    %v4185 = stablehlo.subtract %v4184, %v4180 : tensor<32x784x192xf32>
    %v4186 = stablehlo.multiply %v4176, %v4183 : tensor<32x784x192xf32>
    %v4187 = stablehlo.subtract %v4185, %v4186 : tensor<32x784x192xf32>
    %v4188 = stablehlo.divide %v4175, %v4164 : tensor<32x784x192xf32>
    %v4189 = stablehlo.multiply %v4188, %v4187 : tensor<32x784x192xf32>
    %v4190 = stablehlo.reshape %v4189 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4191 = stablehlo.reshape %v4190 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4192 = stablehlo.transpose %v4191, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v4193 = stablehlo.reshape %v4192 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v4194 = stablehlo.reshape %v4193 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4195 = stablehlo.reverse %s1b1dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v4196 = stablehlo.convolution(%v4194, %v4195)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v4197 = stablehlo.reshape %v4196 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4198 = stablehlo.reshape %v4197 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4199 = stablehlo.reshape %v4021 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4200 = stablehlo.add %v4198, %v4199 : tensor<32x192x28x28xf32>
    %v4201 = stablehlo.reshape %v4200 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4203 = stablehlo.reshape %v429 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4204 = stablehlo.reshape %v4021 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4205 = stablehlo.multiply %v4203, %v4204 : tensor<32x192x28x28xf32>
    %v4206 = stablehlo.reduce(%v4205 init: %v4202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4207 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4208 = stablehlo.multiply %v4206, %v4207 : tensor<192xf32>
    %v4209 = stablehlo.subtract %s1b1lg, %v4208 : tensor<192xf32>
    %v4210 = stablehlo.reshape %v424 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4211 = stablehlo.reshape %v4114 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4212 = stablehlo.transpose %v4210, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v4213 = stablehlo.transpose %v4211, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4214 = stablehlo.convolution(%v4212, %v4213)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v4215 = stablehlo.transpose %v4214, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v4216 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v4217 = stablehlo.multiply %v4215, %v4216 : tensor<192x768x1x1xf32>
    %v4218 = stablehlo.subtract %s1b1pW, %v4217 : tensor<192x768x1x1xf32>
    %v4219 = stablehlo.reshape %v4114 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4221 = stablehlo.reduce(%v4219 init: %v4220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4222 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4223 = stablehlo.multiply %v4221, %v4222 : tensor<192xf32>
    %v4224 = stablehlo.subtract %s1b1pb, %v4223 : tensor<192xf32>
    %v4225 = stablehlo.reshape %v404 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4226 = stablehlo.reshape %v4145 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4227 = stablehlo.transpose %v4225, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4228 = stablehlo.transpose %v4226, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v4229 = stablehlo.convolution(%v4227, %v4228)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v4230 = stablehlo.transpose %v4229, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v4231 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v4232 = stablehlo.multiply %v4230, %v4231 : tensor<768x192x1x1xf32>
    %v4233 = stablehlo.subtract %s1b1eW, %v4232 : tensor<768x192x1x1xf32>
    %v4234 = stablehlo.reshape %v4145 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4236 = stablehlo.reduce(%v4234 init: %v4235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v4237 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v4238 = stablehlo.multiply %v4236, %v4237 : tensor<768xf32>
    %v4239 = stablehlo.subtract %s1b1eb, %v4238 : tensor<768xf32>
    %v4240 = stablehlo.reshape %v370 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4241 = stablehlo.transpose %v4240, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4242 = stablehlo.reshape %v4241 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4243 = stablehlo.reshape %v4150 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4244 = stablehlo.transpose %v4243, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4245 = stablehlo.reshape %v4244 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4246 = stablehlo.reshape %v4242 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4247 = stablehlo.reshape %v4245 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4248 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4249 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v4250 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v4251 = stablehlo.reduce(%v4246 init: %v4248) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4252 = stablehlo.broadcast_in_dim %v4251, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4253 = stablehlo.divide %v4252, %v4249 : tensor<32x784x192xf32>
    %v4254 = stablehlo.subtract %v4246, %v4253 : tensor<32x784x192xf32>
    %v4255 = stablehlo.multiply %v4254, %v4254 : tensor<32x784x192xf32>
    %v4256 = stablehlo.reduce(%v4255 init: %v4248) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4257 = stablehlo.broadcast_in_dim %v4256, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4258 = stablehlo.divide %v4257, %v4249 : tensor<32x784x192xf32>
    %v4259 = stablehlo.add %v4258, %v4250 : tensor<32x784x192xf32>
    %v4260 = stablehlo.rsqrt %v4259 : tensor<32x784x192xf32>
    %v4261 = stablehlo.multiply %v4254, %v4260 : tensor<32x784x192xf32>
    %v4262 = stablehlo.multiply %v4247, %v4261 : tensor<32x784x192xf32>
    %v4263 = stablehlo.reduce(%v4262 init: %v4248) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4264 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4265 = stablehlo.multiply %v4263, %v4264 : tensor<192xf32>
    %v4266 = stablehlo.subtract %s1b1ng, %v4265 : tensor<192xf32>
    %v4267 = stablehlo.reshape %v4150 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4268 = stablehlo.transpose %v4267, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4269 = stablehlo.reshape %v4268 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4271 = stablehlo.reshape %v4269 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4272 = stablehlo.reduce(%v4271 init: %v4270) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4273 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4274 = stablehlo.multiply %v4272, %v4273 : tensor<192xf32>
    %v4275 = stablehlo.subtract %s1b1nbt, %v4274 : tensor<192xf32>
    %v4276 = stablehlo.reshape %v365 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4277 = stablehlo.reshape %v4193 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4278 = stablehlo.transpose %v4276, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4279 = stablehlo.transpose %v4277, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4280 = stablehlo.convolution(%v4278, %v4279)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v4281 = stablehlo.reshape %v4280 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v4282 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v4283 = stablehlo.multiply %v4281, %v4282 : tensor<192x1x7x7xf32>
    %v4284 = stablehlo.subtract %s1b1dW, %v4283 : tensor<192x1x7x7xf32>
    %v4285 = stablehlo.reshape %v4193 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4287 = stablehlo.reduce(%v4285 init: %v4286) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4288 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4289 = stablehlo.multiply %v4287, %v4288 : tensor<192xf32>
    %v4290 = stablehlo.subtract %s1b1db, %v4289 : tensor<192xf32>
    %v4291 = stablehlo.reshape %v4201 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4292 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4293 = stablehlo.multiply %v4291, %v4292 : tensor<32x192x28x28xf32>
    %v4294 = stablehlo.reshape %v4293 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4295 = stablehlo.reshape %v4294 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4296 = stablehlo.transpose %s1b0pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v4297 = stablehlo.reverse %v4296, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v4298 = stablehlo.convolution(%v4295, %v4297)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v4299 = stablehlo.reshape %v4298 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v4300 = stablehlo.reshape %v4299 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4301 = stablehlo.reshape %v337 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4302 = stablehlo.multiply %v4301, %v4301 : tensor<32x768x28x28xf32>
    %v4303 = stablehlo.multiply %v4302, %v4301 : tensor<32x768x28x28xf32>
    %v4304 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v4305 = stablehlo.multiply %v4304, %v4303 : tensor<32x768x28x28xf32>
    %v4306 = stablehlo.add %v4301, %v4305 : tensor<32x768x28x28xf32>
    %v4307 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v4308 = stablehlo.multiply %v4307, %v4306 : tensor<32x768x28x28xf32>
    %v4309 = stablehlo.tanh %v4308 : tensor<32x768x28x28xf32>
    %v4310 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v4311 = stablehlo.add %v4310, %v4309 : tensor<32x768x28x28xf32>
    %v4312 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v4313 = stablehlo.multiply %v4312, %v4311 : tensor<32x768x28x28xf32>
    %v4314 = stablehlo.multiply %v4309, %v4309 : tensor<32x768x28x28xf32>
    %v4315 = stablehlo.subtract %v4310, %v4314 : tensor<32x768x28x28xf32>
    %v4316 = stablehlo.multiply %v4312, %v4301 : tensor<32x768x28x28xf32>
    %v4317 = stablehlo.multiply %v4316, %v4315 : tensor<32x768x28x28xf32>
    %v4318 = stablehlo.constant dense<0.134145> : tensor<32x768x28x28xf32>
    %v4319 = stablehlo.multiply %v4318, %v4302 : tensor<32x768x28x28xf32>
    %v4320 = stablehlo.add %v4310, %v4319 : tensor<32x768x28x28xf32>
    %v4321 = stablehlo.multiply %v4307, %v4320 : tensor<32x768x28x28xf32>
    %v4322 = stablehlo.multiply %v4317, %v4321 : tensor<32x768x28x28xf32>
    %v4323 = stablehlo.add %v4313, %v4322 : tensor<32x768x28x28xf32>
    %v4324 = stablehlo.multiply %v4300, %v4323 : tensor<32x768x28x28xf32>
    %v4325 = stablehlo.reshape %v4324 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v4326 = stablehlo.reshape %v4325 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4327 = stablehlo.transpose %s1b0eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v4328 = stablehlo.reverse %v4327, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v4329 = stablehlo.convolution(%v4326, %v4328)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4330 = stablehlo.reshape %v4329 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4331 = stablehlo.reshape %v298 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4332 = stablehlo.transpose %v4331, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4333 = stablehlo.reshape %v4332 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4334 = stablehlo.reshape %v4330 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4335 = stablehlo.transpose %v4334, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4336 = stablehlo.reshape %v4335 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4337 = stablehlo.reshape %v4336 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4338 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v4339 = stablehlo.multiply %v4337, %v4338 : tensor<32x784x192xf32>
    %v4340 = stablehlo.reshape %v4339 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4341 = stablehlo.reshape %v4340 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4342 = stablehlo.reshape %v4333 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4344 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v4345 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v4346 = stablehlo.reduce(%v4342 init: %v4343) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4347 = stablehlo.broadcast_in_dim %v4346, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4348 = stablehlo.divide %v4347, %v4344 : tensor<32x784x192xf32>
    %v4349 = stablehlo.subtract %v4342, %v4348 : tensor<32x784x192xf32>
    %v4350 = stablehlo.multiply %v4349, %v4349 : tensor<32x784x192xf32>
    %v4351 = stablehlo.reduce(%v4350 init: %v4343) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4352 = stablehlo.broadcast_in_dim %v4351, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4353 = stablehlo.divide %v4352, %v4344 : tensor<32x784x192xf32>
    %v4354 = stablehlo.add %v4353, %v4345 : tensor<32x784x192xf32>
    %v4355 = stablehlo.rsqrt %v4354 : tensor<32x784x192xf32>
    %v4356 = stablehlo.multiply %v4349, %v4355 : tensor<32x784x192xf32>
    %v4357 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v4358 = stablehlo.multiply %v4357, %v4341 : tensor<32x784x192xf32>
    %v4359 = stablehlo.reduce(%v4358 init: %v4343) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4360 = stablehlo.broadcast_in_dim %v4359, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4361 = stablehlo.multiply %v4356, %v4358 : tensor<32x784x192xf32>
    %v4362 = stablehlo.reduce(%v4361 init: %v4343) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4363 = stablehlo.broadcast_in_dim %v4362, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4364 = stablehlo.multiply %v4358, %v4344 : tensor<32x784x192xf32>
    %v4365 = stablehlo.subtract %v4364, %v4360 : tensor<32x784x192xf32>
    %v4366 = stablehlo.multiply %v4356, %v4363 : tensor<32x784x192xf32>
    %v4367 = stablehlo.subtract %v4365, %v4366 : tensor<32x784x192xf32>
    %v4368 = stablehlo.divide %v4355, %v4344 : tensor<32x784x192xf32>
    %v4369 = stablehlo.multiply %v4368, %v4367 : tensor<32x784x192xf32>
    %v4370 = stablehlo.reshape %v4369 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4371 = stablehlo.reshape %v4370 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4372 = stablehlo.transpose %v4371, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v4373 = stablehlo.reshape %v4372 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v4374 = stablehlo.reshape %v4373 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4375 = stablehlo.reverse %s1b0dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v4376 = stablehlo.convolution(%v4374, %v4375)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v4377 = stablehlo.reshape %v4376 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4378 = stablehlo.reshape %v4377 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4379 = stablehlo.reshape %v4201 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4380 = stablehlo.add %v4378, %v4379 : tensor<32x192x28x28xf32>
    %v4381 = stablehlo.reshape %v4380 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4383 = stablehlo.reshape %v357 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4384 = stablehlo.reshape %v4201 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4385 = stablehlo.multiply %v4383, %v4384 : tensor<32x192x28x28xf32>
    %v4386 = stablehlo.reduce(%v4385 init: %v4382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4387 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4388 = stablehlo.multiply %v4386, %v4387 : tensor<192xf32>
    %v4389 = stablehlo.subtract %s1b0lg, %v4388 : tensor<192xf32>
    %v4390 = stablehlo.reshape %v352 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4391 = stablehlo.reshape %v4294 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4392 = stablehlo.transpose %v4390, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v4393 = stablehlo.transpose %v4391, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4394 = stablehlo.convolution(%v4392, %v4393)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v4395 = stablehlo.transpose %v4394, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v4396 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v4397 = stablehlo.multiply %v4395, %v4396 : tensor<192x768x1x1xf32>
    %v4398 = stablehlo.subtract %s1b0pW, %v4397 : tensor<192x768x1x1xf32>
    %v4399 = stablehlo.reshape %v4294 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4401 = stablehlo.reduce(%v4399 init: %v4400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4402 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4403 = stablehlo.multiply %v4401, %v4402 : tensor<192xf32>
    %v4404 = stablehlo.subtract %s1b0pb, %v4403 : tensor<192xf32>
    %v4405 = stablehlo.reshape %v332 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4406 = stablehlo.reshape %v4325 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4407 = stablehlo.transpose %v4405, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4408 = stablehlo.transpose %v4406, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v4409 = stablehlo.convolution(%v4407, %v4408)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v4410 = stablehlo.transpose %v4409, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v4411 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v4412 = stablehlo.multiply %v4410, %v4411 : tensor<768x192x1x1xf32>
    %v4413 = stablehlo.subtract %s1b0eW, %v4412 : tensor<768x192x1x1xf32>
    %v4414 = stablehlo.reshape %v4325 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4416 = stablehlo.reduce(%v4414 init: %v4415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v4417 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v4418 = stablehlo.multiply %v4416, %v4417 : tensor<768xf32>
    %v4419 = stablehlo.subtract %s1b0eb, %v4418 : tensor<768xf32>
    %v4420 = stablehlo.reshape %v298 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4421 = stablehlo.transpose %v4420, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4422 = stablehlo.reshape %v4421 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4423 = stablehlo.reshape %v4330 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4424 = stablehlo.transpose %v4423, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4425 = stablehlo.reshape %v4424 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4426 = stablehlo.reshape %v4422 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4427 = stablehlo.reshape %v4425 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4428 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4429 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v4430 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v4431 = stablehlo.reduce(%v4426 init: %v4428) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4432 = stablehlo.broadcast_in_dim %v4431, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4433 = stablehlo.divide %v4432, %v4429 : tensor<32x784x192xf32>
    %v4434 = stablehlo.subtract %v4426, %v4433 : tensor<32x784x192xf32>
    %v4435 = stablehlo.multiply %v4434, %v4434 : tensor<32x784x192xf32>
    %v4436 = stablehlo.reduce(%v4435 init: %v4428) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4437 = stablehlo.broadcast_in_dim %v4436, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4438 = stablehlo.divide %v4437, %v4429 : tensor<32x784x192xf32>
    %v4439 = stablehlo.add %v4438, %v4430 : tensor<32x784x192xf32>
    %v4440 = stablehlo.rsqrt %v4439 : tensor<32x784x192xf32>
    %v4441 = stablehlo.multiply %v4434, %v4440 : tensor<32x784x192xf32>
    %v4442 = stablehlo.multiply %v4427, %v4441 : tensor<32x784x192xf32>
    %v4443 = stablehlo.reduce(%v4442 init: %v4428) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4444 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4445 = stablehlo.multiply %v4443, %v4444 : tensor<192xf32>
    %v4446 = stablehlo.subtract %s1b0ng, %v4445 : tensor<192xf32>
    %v4447 = stablehlo.reshape %v4330 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4448 = stablehlo.transpose %v4447, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4449 = stablehlo.reshape %v4448 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4451 = stablehlo.reshape %v4449 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4452 = stablehlo.reduce(%v4451 init: %v4450) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4453 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4454 = stablehlo.multiply %v4452, %v4453 : tensor<192xf32>
    %v4455 = stablehlo.subtract %s1b0nbt, %v4454 : tensor<192xf32>
    %v4456 = stablehlo.reshape %v293 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4457 = stablehlo.reshape %v4373 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4458 = stablehlo.transpose %v4456, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4459 = stablehlo.transpose %v4457, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4460 = stablehlo.convolution(%v4458, %v4459)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v4461 = stablehlo.reshape %v4460 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v4462 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v4463 = stablehlo.multiply %v4461, %v4462 : tensor<192x1x7x7xf32>
    %v4464 = stablehlo.subtract %s1b0dW, %v4463 : tensor<192x1x7x7xf32>
    %v4465 = stablehlo.reshape %v4373 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4466 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4467 = stablehlo.reduce(%v4465 init: %v4466) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4468 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4469 = stablehlo.multiply %v4467, %v4468 : tensor<192xf32>
    %v4470 = stablehlo.subtract %s1b0db, %v4469 : tensor<192xf32>
    %v4471 = stablehlo.reshape %v4381 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4472 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4473 = stablehlo.pad %v4471, %v4472, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x56x56xf32>
    %v4474 = stablehlo.transpose %d0W, dims = [1, 0, 2, 3] : (tensor<192x96x2x2xf32>) -> tensor<96x192x2x2xf32>
    %v4475 = stablehlo.reverse %v4474, dims = [2, 3] : tensor<96x192x2x2xf32>
    %v4476 = stablehlo.convolution(%v4473, %v4475)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x56x56xf32>, tensor<96x192x2x2xf32>) -> tensor<32x96x56x56xf32>
    %v4477 = stablehlo.reshape %v4476 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4478 = stablehlo.reshape %v254 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4479 = stablehlo.transpose %v4478, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4480 = stablehlo.reshape %v4479 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4481 = stablehlo.reshape %v4477 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4482 = stablehlo.transpose %v4481, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4483 = stablehlo.reshape %v4482 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4484 = stablehlo.reshape %v4483 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4485 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4486 = stablehlo.multiply %v4484, %v4485 : tensor<32x3136x96xf32>
    %v4487 = stablehlo.reshape %v4486 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4488 = stablehlo.reshape %v4487 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4489 = stablehlo.reshape %v4480 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4491 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4492 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4493 = stablehlo.reduce(%v4489 init: %v4490) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4494 = stablehlo.broadcast_in_dim %v4493, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4495 = stablehlo.divide %v4494, %v4491 : tensor<32x3136x96xf32>
    %v4496 = stablehlo.subtract %v4489, %v4495 : tensor<32x3136x96xf32>
    %v4497 = stablehlo.multiply %v4496, %v4496 : tensor<32x3136x96xf32>
    %v4498 = stablehlo.reduce(%v4497 init: %v4490) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4499 = stablehlo.broadcast_in_dim %v4498, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4500 = stablehlo.divide %v4499, %v4491 : tensor<32x3136x96xf32>
    %v4501 = stablehlo.add %v4500, %v4492 : tensor<32x3136x96xf32>
    %v4502 = stablehlo.rsqrt %v4501 : tensor<32x3136x96xf32>
    %v4503 = stablehlo.multiply %v4496, %v4502 : tensor<32x3136x96xf32>
    %v4504 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4505 = stablehlo.multiply %v4504, %v4488 : tensor<32x3136x96xf32>
    %v4506 = stablehlo.reduce(%v4505 init: %v4490) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4507 = stablehlo.broadcast_in_dim %v4506, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4508 = stablehlo.multiply %v4503, %v4505 : tensor<32x3136x96xf32>
    %v4509 = stablehlo.reduce(%v4508 init: %v4490) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4510 = stablehlo.broadcast_in_dim %v4509, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4511 = stablehlo.multiply %v4505, %v4491 : tensor<32x3136x96xf32>
    %v4512 = stablehlo.subtract %v4511, %v4507 : tensor<32x3136x96xf32>
    %v4513 = stablehlo.multiply %v4503, %v4510 : tensor<32x3136x96xf32>
    %v4514 = stablehlo.subtract %v4512, %v4513 : tensor<32x3136x96xf32>
    %v4515 = stablehlo.divide %v4502, %v4491 : tensor<32x3136x96xf32>
    %v4516 = stablehlo.multiply %v4515, %v4514 : tensor<32x3136x96xf32>
    %v4517 = stablehlo.reshape %v4516 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4518 = stablehlo.reshape %v4517 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4519 = stablehlo.transpose %v4518, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4520 = stablehlo.reshape %v4519 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4521 = stablehlo.reshape %v4381 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4523 = stablehlo.reduce(%v4521 init: %v4522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4524 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4525 = stablehlo.multiply %v4523, %v4524 : tensor<192xf32>
    %v4526 = stablehlo.subtract %d0b, %v4525 : tensor<192xf32>
    %v4527 = stablehlo.reshape %v254 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4528 = stablehlo.transpose %v4527, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4529 = stablehlo.reshape %v4528 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4530 = stablehlo.reshape %v4477 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4531 = stablehlo.transpose %v4530, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4532 = stablehlo.reshape %v4531 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4533 = stablehlo.reshape %v4529 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4534 = stablehlo.reshape %v4532 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4535 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4536 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4537 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4538 = stablehlo.reduce(%v4533 init: %v4535) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4539 = stablehlo.broadcast_in_dim %v4538, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4540 = stablehlo.divide %v4539, %v4536 : tensor<32x3136x96xf32>
    %v4541 = stablehlo.subtract %v4533, %v4540 : tensor<32x3136x96xf32>
    %v4542 = stablehlo.multiply %v4541, %v4541 : tensor<32x3136x96xf32>
    %v4543 = stablehlo.reduce(%v4542 init: %v4535) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4544 = stablehlo.broadcast_in_dim %v4543, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4545 = stablehlo.divide %v4544, %v4536 : tensor<32x3136x96xf32>
    %v4546 = stablehlo.add %v4545, %v4537 : tensor<32x3136x96xf32>
    %v4547 = stablehlo.rsqrt %v4546 : tensor<32x3136x96xf32>
    %v4548 = stablehlo.multiply %v4541, %v4547 : tensor<32x3136x96xf32>
    %v4549 = stablehlo.multiply %v4534, %v4548 : tensor<32x3136x96xf32>
    %v4550 = stablehlo.reduce(%v4549 init: %v4535) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4551 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4552 = stablehlo.multiply %v4550, %v4551 : tensor<96xf32>
    %v4553 = stablehlo.subtract %d0ng, %v4552 : tensor<96xf32>
    %v4554 = stablehlo.reshape %v4477 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4555 = stablehlo.transpose %v4554, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4556 = stablehlo.reshape %v4555 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4557 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4558 = stablehlo.reshape %v4556 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4559 = stablehlo.reduce(%v4558 init: %v4557) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4560 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4561 = stablehlo.multiply %v4559, %v4560 : tensor<96xf32>
    %v4562 = stablehlo.subtract %d0nbt, %v4561 : tensor<96xf32>
    %v4563 = stablehlo.reshape %v288 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4564 = stablehlo.reshape %v4381 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4566 = stablehlo.pad %v4564, %v4565, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x55x55xf32>
    %v4567 = stablehlo.transpose %v4563, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4568 = stablehlo.transpose %v4566, dims = [1, 0, 2, 3] : (tensor<32x192x55x55xf32>) -> tensor<192x32x55x55xf32>
    %v4569 = stablehlo.convolution(%v4567, %v4568)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<192x32x55x55xf32>) -> tensor<96x192x2x2xf32>
    %v4570 = stablehlo.transpose %v4569, dims = [1, 0, 2, 3] : (tensor<96x192x2x2xf32>) -> tensor<192x96x2x2xf32>
    %v4571 = stablehlo.constant dense<0.1> : tensor<192x96x2x2xf32>
    %v4572 = stablehlo.multiply %v4570, %v4571 : tensor<192x96x2x2xf32>
    %v4573 = stablehlo.subtract %d0W, %v4572 : tensor<192x96x2x2xf32>
    %v4574 = stablehlo.reshape %v4520 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4575 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4576 = stablehlo.multiply %v4574, %v4575 : tensor<32x96x56x56xf32>
    %v4577 = stablehlo.reshape %v4576 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4578 = stablehlo.reshape %v4577 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4579 = stablehlo.transpose %s0b2pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4580 = stablehlo.reverse %v4579, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v4581 = stablehlo.convolution(%v4578, %v4580)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v4582 = stablehlo.reshape %v4581 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4583 = stablehlo.reshape %v4582 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4584 = stablehlo.reshape %v226 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4585 = stablehlo.multiply %v4584, %v4584 : tensor<32x384x56x56xf32>
    %v4586 = stablehlo.multiply %v4585, %v4584 : tensor<32x384x56x56xf32>
    %v4587 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v4588 = stablehlo.multiply %v4587, %v4586 : tensor<32x384x56x56xf32>
    %v4589 = stablehlo.add %v4584, %v4588 : tensor<32x384x56x56xf32>
    %v4590 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v4591 = stablehlo.multiply %v4590, %v4589 : tensor<32x384x56x56xf32>
    %v4592 = stablehlo.tanh %v4591 : tensor<32x384x56x56xf32>
    %v4593 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v4594 = stablehlo.add %v4593, %v4592 : tensor<32x384x56x56xf32>
    %v4595 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v4596 = stablehlo.multiply %v4595, %v4594 : tensor<32x384x56x56xf32>
    %v4597 = stablehlo.multiply %v4592, %v4592 : tensor<32x384x56x56xf32>
    %v4598 = stablehlo.subtract %v4593, %v4597 : tensor<32x384x56x56xf32>
    %v4599 = stablehlo.multiply %v4595, %v4584 : tensor<32x384x56x56xf32>
    %v4600 = stablehlo.multiply %v4599, %v4598 : tensor<32x384x56x56xf32>
    %v4601 = stablehlo.constant dense<0.134145> : tensor<32x384x56x56xf32>
    %v4602 = stablehlo.multiply %v4601, %v4585 : tensor<32x384x56x56xf32>
    %v4603 = stablehlo.add %v4593, %v4602 : tensor<32x384x56x56xf32>
    %v4604 = stablehlo.multiply %v4590, %v4603 : tensor<32x384x56x56xf32>
    %v4605 = stablehlo.multiply %v4600, %v4604 : tensor<32x384x56x56xf32>
    %v4606 = stablehlo.add %v4596, %v4605 : tensor<32x384x56x56xf32>
    %v4607 = stablehlo.multiply %v4583, %v4606 : tensor<32x384x56x56xf32>
    %v4608 = stablehlo.reshape %v4607 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4609 = stablehlo.reshape %v4608 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4610 = stablehlo.transpose %s0b2eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4611 = stablehlo.reverse %v4610, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v4612 = stablehlo.convolution(%v4609, %v4611)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4613 = stablehlo.reshape %v4612 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4614 = stablehlo.reshape %v187 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4615 = stablehlo.transpose %v4614, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4616 = stablehlo.reshape %v4615 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4617 = stablehlo.reshape %v4613 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4618 = stablehlo.transpose %v4617, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4619 = stablehlo.reshape %v4618 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4620 = stablehlo.reshape %v4619 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4621 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4622 = stablehlo.multiply %v4620, %v4621 : tensor<32x3136x96xf32>
    %v4623 = stablehlo.reshape %v4622 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4624 = stablehlo.reshape %v4623 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4625 = stablehlo.reshape %v4616 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4627 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4628 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4629 = stablehlo.reduce(%v4625 init: %v4626) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4630 = stablehlo.broadcast_in_dim %v4629, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4631 = stablehlo.divide %v4630, %v4627 : tensor<32x3136x96xf32>
    %v4632 = stablehlo.subtract %v4625, %v4631 : tensor<32x3136x96xf32>
    %v4633 = stablehlo.multiply %v4632, %v4632 : tensor<32x3136x96xf32>
    %v4634 = stablehlo.reduce(%v4633 init: %v4626) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4635 = stablehlo.broadcast_in_dim %v4634, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4636 = stablehlo.divide %v4635, %v4627 : tensor<32x3136x96xf32>
    %v4637 = stablehlo.add %v4636, %v4628 : tensor<32x3136x96xf32>
    %v4638 = stablehlo.rsqrt %v4637 : tensor<32x3136x96xf32>
    %v4639 = stablehlo.multiply %v4632, %v4638 : tensor<32x3136x96xf32>
    %v4640 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4641 = stablehlo.multiply %v4640, %v4624 : tensor<32x3136x96xf32>
    %v4642 = stablehlo.reduce(%v4641 init: %v4626) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4643 = stablehlo.broadcast_in_dim %v4642, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4644 = stablehlo.multiply %v4639, %v4641 : tensor<32x3136x96xf32>
    %v4645 = stablehlo.reduce(%v4644 init: %v4626) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4646 = stablehlo.broadcast_in_dim %v4645, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4647 = stablehlo.multiply %v4641, %v4627 : tensor<32x3136x96xf32>
    %v4648 = stablehlo.subtract %v4647, %v4643 : tensor<32x3136x96xf32>
    %v4649 = stablehlo.multiply %v4639, %v4646 : tensor<32x3136x96xf32>
    %v4650 = stablehlo.subtract %v4648, %v4649 : tensor<32x3136x96xf32>
    %v4651 = stablehlo.divide %v4638, %v4627 : tensor<32x3136x96xf32>
    %v4652 = stablehlo.multiply %v4651, %v4650 : tensor<32x3136x96xf32>
    %v4653 = stablehlo.reshape %v4652 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4654 = stablehlo.reshape %v4653 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4655 = stablehlo.transpose %v4654, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4656 = stablehlo.reshape %v4655 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4657 = stablehlo.reshape %v4656 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4658 = stablehlo.reverse %s0b2dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v4659 = stablehlo.convolution(%v4657, %v4658)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v4660 = stablehlo.reshape %v4659 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4661 = stablehlo.reshape %v4660 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4662 = stablehlo.reshape %v4520 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4663 = stablehlo.add %v4661, %v4662 : tensor<32x96x56x56xf32>
    %v4664 = stablehlo.reshape %v4663 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4666 = stablehlo.reshape %v246 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4667 = stablehlo.reshape %v4520 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4668 = stablehlo.multiply %v4666, %v4667 : tensor<32x96x56x56xf32>
    %v4669 = stablehlo.reduce(%v4668 init: %v4665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4670 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4671 = stablehlo.multiply %v4669, %v4670 : tensor<96xf32>
    %v4672 = stablehlo.subtract %s0b2lg, %v4671 : tensor<96xf32>
    %v4673 = stablehlo.reshape %v241 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4674 = stablehlo.reshape %v4577 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4675 = stablehlo.transpose %v4673, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4676 = stablehlo.transpose %v4674, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4677 = stablehlo.convolution(%v4675, %v4676)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v4678 = stablehlo.transpose %v4677, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4679 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v4680 = stablehlo.multiply %v4678, %v4679 : tensor<96x384x1x1xf32>
    %v4681 = stablehlo.subtract %s0b2pW, %v4680 : tensor<96x384x1x1xf32>
    %v4682 = stablehlo.reshape %v4577 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4683 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4684 = stablehlo.reduce(%v4682 init: %v4683) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4685 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4686 = stablehlo.multiply %v4684, %v4685 : tensor<96xf32>
    %v4687 = stablehlo.subtract %s0b2pb, %v4686 : tensor<96xf32>
    %v4688 = stablehlo.reshape %v221 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4689 = stablehlo.reshape %v4608 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4690 = stablehlo.transpose %v4688, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4691 = stablehlo.transpose %v4689, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4692 = stablehlo.convolution(%v4690, %v4691)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v4693 = stablehlo.transpose %v4692, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4694 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v4695 = stablehlo.multiply %v4693, %v4694 : tensor<384x96x1x1xf32>
    %v4696 = stablehlo.subtract %s0b2eW, %v4695 : tensor<384x96x1x1xf32>
    %v4697 = stablehlo.reshape %v4608 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4699 = stablehlo.reduce(%v4697 init: %v4698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v4700 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v4701 = stablehlo.multiply %v4699, %v4700 : tensor<384xf32>
    %v4702 = stablehlo.subtract %s0b2eb, %v4701 : tensor<384xf32>
    %v4703 = stablehlo.reshape %v187 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4704 = stablehlo.transpose %v4703, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4705 = stablehlo.reshape %v4704 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4706 = stablehlo.reshape %v4613 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4707 = stablehlo.transpose %v4706, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4708 = stablehlo.reshape %v4707 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4709 = stablehlo.reshape %v4705 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4710 = stablehlo.reshape %v4708 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4712 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4713 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4714 = stablehlo.reduce(%v4709 init: %v4711) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4715 = stablehlo.broadcast_in_dim %v4714, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4716 = stablehlo.divide %v4715, %v4712 : tensor<32x3136x96xf32>
    %v4717 = stablehlo.subtract %v4709, %v4716 : tensor<32x3136x96xf32>
    %v4718 = stablehlo.multiply %v4717, %v4717 : tensor<32x3136x96xf32>
    %v4719 = stablehlo.reduce(%v4718 init: %v4711) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4720 = stablehlo.broadcast_in_dim %v4719, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4721 = stablehlo.divide %v4720, %v4712 : tensor<32x3136x96xf32>
    %v4722 = stablehlo.add %v4721, %v4713 : tensor<32x3136x96xf32>
    %v4723 = stablehlo.rsqrt %v4722 : tensor<32x3136x96xf32>
    %v4724 = stablehlo.multiply %v4717, %v4723 : tensor<32x3136x96xf32>
    %v4725 = stablehlo.multiply %v4710, %v4724 : tensor<32x3136x96xf32>
    %v4726 = stablehlo.reduce(%v4725 init: %v4711) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4727 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4728 = stablehlo.multiply %v4726, %v4727 : tensor<96xf32>
    %v4729 = stablehlo.subtract %s0b2ng, %v4728 : tensor<96xf32>
    %v4730 = stablehlo.reshape %v4613 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4731 = stablehlo.transpose %v4730, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4732 = stablehlo.reshape %v4731 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4734 = stablehlo.reshape %v4732 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4735 = stablehlo.reduce(%v4734 init: %v4733) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4736 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4737 = stablehlo.multiply %v4735, %v4736 : tensor<96xf32>
    %v4738 = stablehlo.subtract %s0b2nbt, %v4737 : tensor<96xf32>
    %v4739 = stablehlo.reshape %v182 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4740 = stablehlo.reshape %v4656 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4741 = stablehlo.transpose %v4739, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4742 = stablehlo.transpose %v4740, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4743 = stablehlo.convolution(%v4741, %v4742)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v4744 = stablehlo.reshape %v4743 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v4745 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v4746 = stablehlo.multiply %v4744, %v4745 : tensor<96x1x7x7xf32>
    %v4747 = stablehlo.subtract %s0b2dW, %v4746 : tensor<96x1x7x7xf32>
    %v4748 = stablehlo.reshape %v4656 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4750 = stablehlo.reduce(%v4748 init: %v4749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4751 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4752 = stablehlo.multiply %v4750, %v4751 : tensor<96xf32>
    %v4753 = stablehlo.subtract %s0b2db, %v4752 : tensor<96xf32>
    %v4754 = stablehlo.reshape %v4664 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4755 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4756 = stablehlo.multiply %v4754, %v4755 : tensor<32x96x56x56xf32>
    %v4757 = stablehlo.reshape %v4756 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4758 = stablehlo.reshape %v4757 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4759 = stablehlo.transpose %s0b1pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4760 = stablehlo.reverse %v4759, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v4761 = stablehlo.convolution(%v4758, %v4760)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v4762 = stablehlo.reshape %v4761 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4763 = stablehlo.reshape %v4762 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4764 = stablehlo.reshape %v154 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4765 = stablehlo.multiply %v4764, %v4764 : tensor<32x384x56x56xf32>
    %v4766 = stablehlo.multiply %v4765, %v4764 : tensor<32x384x56x56xf32>
    %v4767 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v4768 = stablehlo.multiply %v4767, %v4766 : tensor<32x384x56x56xf32>
    %v4769 = stablehlo.add %v4764, %v4768 : tensor<32x384x56x56xf32>
    %v4770 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v4771 = stablehlo.multiply %v4770, %v4769 : tensor<32x384x56x56xf32>
    %v4772 = stablehlo.tanh %v4771 : tensor<32x384x56x56xf32>
    %v4773 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v4774 = stablehlo.add %v4773, %v4772 : tensor<32x384x56x56xf32>
    %v4775 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v4776 = stablehlo.multiply %v4775, %v4774 : tensor<32x384x56x56xf32>
    %v4777 = stablehlo.multiply %v4772, %v4772 : tensor<32x384x56x56xf32>
    %v4778 = stablehlo.subtract %v4773, %v4777 : tensor<32x384x56x56xf32>
    %v4779 = stablehlo.multiply %v4775, %v4764 : tensor<32x384x56x56xf32>
    %v4780 = stablehlo.multiply %v4779, %v4778 : tensor<32x384x56x56xf32>
    %v4781 = stablehlo.constant dense<0.134145> : tensor<32x384x56x56xf32>
    %v4782 = stablehlo.multiply %v4781, %v4765 : tensor<32x384x56x56xf32>
    %v4783 = stablehlo.add %v4773, %v4782 : tensor<32x384x56x56xf32>
    %v4784 = stablehlo.multiply %v4770, %v4783 : tensor<32x384x56x56xf32>
    %v4785 = stablehlo.multiply %v4780, %v4784 : tensor<32x384x56x56xf32>
    %v4786 = stablehlo.add %v4776, %v4785 : tensor<32x384x56x56xf32>
    %v4787 = stablehlo.multiply %v4763, %v4786 : tensor<32x384x56x56xf32>
    %v4788 = stablehlo.reshape %v4787 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4789 = stablehlo.reshape %v4788 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4790 = stablehlo.transpose %s0b1eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4791 = stablehlo.reverse %v4790, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v4792 = stablehlo.convolution(%v4789, %v4791)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4793 = stablehlo.reshape %v4792 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4794 = stablehlo.reshape %v115 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4795 = stablehlo.transpose %v4794, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4796 = stablehlo.reshape %v4795 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4797 = stablehlo.reshape %v4793 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4798 = stablehlo.transpose %v4797, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4799 = stablehlo.reshape %v4798 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4800 = stablehlo.reshape %v4799 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4801 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4802 = stablehlo.multiply %v4800, %v4801 : tensor<32x3136x96xf32>
    %v4803 = stablehlo.reshape %v4802 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4804 = stablehlo.reshape %v4803 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4805 = stablehlo.reshape %v4796 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4807 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4808 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4809 = stablehlo.reduce(%v4805 init: %v4806) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4810 = stablehlo.broadcast_in_dim %v4809, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4811 = stablehlo.divide %v4810, %v4807 : tensor<32x3136x96xf32>
    %v4812 = stablehlo.subtract %v4805, %v4811 : tensor<32x3136x96xf32>
    %v4813 = stablehlo.multiply %v4812, %v4812 : tensor<32x3136x96xf32>
    %v4814 = stablehlo.reduce(%v4813 init: %v4806) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4815 = stablehlo.broadcast_in_dim %v4814, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4816 = stablehlo.divide %v4815, %v4807 : tensor<32x3136x96xf32>
    %v4817 = stablehlo.add %v4816, %v4808 : tensor<32x3136x96xf32>
    %v4818 = stablehlo.rsqrt %v4817 : tensor<32x3136x96xf32>
    %v4819 = stablehlo.multiply %v4812, %v4818 : tensor<32x3136x96xf32>
    %v4820 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4821 = stablehlo.multiply %v4820, %v4804 : tensor<32x3136x96xf32>
    %v4822 = stablehlo.reduce(%v4821 init: %v4806) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4823 = stablehlo.broadcast_in_dim %v4822, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4824 = stablehlo.multiply %v4819, %v4821 : tensor<32x3136x96xf32>
    %v4825 = stablehlo.reduce(%v4824 init: %v4806) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4826 = stablehlo.broadcast_in_dim %v4825, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4827 = stablehlo.multiply %v4821, %v4807 : tensor<32x3136x96xf32>
    %v4828 = stablehlo.subtract %v4827, %v4823 : tensor<32x3136x96xf32>
    %v4829 = stablehlo.multiply %v4819, %v4826 : tensor<32x3136x96xf32>
    %v4830 = stablehlo.subtract %v4828, %v4829 : tensor<32x3136x96xf32>
    %v4831 = stablehlo.divide %v4818, %v4807 : tensor<32x3136x96xf32>
    %v4832 = stablehlo.multiply %v4831, %v4830 : tensor<32x3136x96xf32>
    %v4833 = stablehlo.reshape %v4832 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4834 = stablehlo.reshape %v4833 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4835 = stablehlo.transpose %v4834, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4836 = stablehlo.reshape %v4835 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4837 = stablehlo.reshape %v4836 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4838 = stablehlo.reverse %s0b1dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v4839 = stablehlo.convolution(%v4837, %v4838)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v4840 = stablehlo.reshape %v4839 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4841 = stablehlo.reshape %v4840 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4842 = stablehlo.reshape %v4664 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4843 = stablehlo.add %v4841, %v4842 : tensor<32x96x56x56xf32>
    %v4844 = stablehlo.reshape %v4843 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4845 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4846 = stablehlo.reshape %v174 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4847 = stablehlo.reshape %v4664 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4848 = stablehlo.multiply %v4846, %v4847 : tensor<32x96x56x56xf32>
    %v4849 = stablehlo.reduce(%v4848 init: %v4845) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4850 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4851 = stablehlo.multiply %v4849, %v4850 : tensor<96xf32>
    %v4852 = stablehlo.subtract %s0b1lg, %v4851 : tensor<96xf32>
    %v4853 = stablehlo.reshape %v169 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4854 = stablehlo.reshape %v4757 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4855 = stablehlo.transpose %v4853, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4856 = stablehlo.transpose %v4854, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4857 = stablehlo.convolution(%v4855, %v4856)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v4858 = stablehlo.transpose %v4857, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4859 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v4860 = stablehlo.multiply %v4858, %v4859 : tensor<96x384x1x1xf32>
    %v4861 = stablehlo.subtract %s0b1pW, %v4860 : tensor<96x384x1x1xf32>
    %v4862 = stablehlo.reshape %v4757 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4864 = stablehlo.reduce(%v4862 init: %v4863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4865 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4866 = stablehlo.multiply %v4864, %v4865 : tensor<96xf32>
    %v4867 = stablehlo.subtract %s0b1pb, %v4866 : tensor<96xf32>
    %v4868 = stablehlo.reshape %v149 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4869 = stablehlo.reshape %v4788 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4870 = stablehlo.transpose %v4868, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4871 = stablehlo.transpose %v4869, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4872 = stablehlo.convolution(%v4870, %v4871)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v4873 = stablehlo.transpose %v4872, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4874 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v4875 = stablehlo.multiply %v4873, %v4874 : tensor<384x96x1x1xf32>
    %v4876 = stablehlo.subtract %s0b1eW, %v4875 : tensor<384x96x1x1xf32>
    %v4877 = stablehlo.reshape %v4788 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4879 = stablehlo.reduce(%v4877 init: %v4878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v4880 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v4881 = stablehlo.multiply %v4879, %v4880 : tensor<384xf32>
    %v4882 = stablehlo.subtract %s0b1eb, %v4881 : tensor<384xf32>
    %v4883 = stablehlo.reshape %v115 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4884 = stablehlo.transpose %v4883, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4885 = stablehlo.reshape %v4884 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4886 = stablehlo.reshape %v4793 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4887 = stablehlo.transpose %v4886, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4888 = stablehlo.reshape %v4887 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4889 = stablehlo.reshape %v4885 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4890 = stablehlo.reshape %v4888 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4892 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4893 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4894 = stablehlo.reduce(%v4889 init: %v4891) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4895 = stablehlo.broadcast_in_dim %v4894, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4896 = stablehlo.divide %v4895, %v4892 : tensor<32x3136x96xf32>
    %v4897 = stablehlo.subtract %v4889, %v4896 : tensor<32x3136x96xf32>
    %v4898 = stablehlo.multiply %v4897, %v4897 : tensor<32x3136x96xf32>
    %v4899 = stablehlo.reduce(%v4898 init: %v4891) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4900 = stablehlo.broadcast_in_dim %v4899, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4901 = stablehlo.divide %v4900, %v4892 : tensor<32x3136x96xf32>
    %v4902 = stablehlo.add %v4901, %v4893 : tensor<32x3136x96xf32>
    %v4903 = stablehlo.rsqrt %v4902 : tensor<32x3136x96xf32>
    %v4904 = stablehlo.multiply %v4897, %v4903 : tensor<32x3136x96xf32>
    %v4905 = stablehlo.multiply %v4890, %v4904 : tensor<32x3136x96xf32>
    %v4906 = stablehlo.reduce(%v4905 init: %v4891) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4907 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4908 = stablehlo.multiply %v4906, %v4907 : tensor<96xf32>
    %v4909 = stablehlo.subtract %s0b1ng, %v4908 : tensor<96xf32>
    %v4910 = stablehlo.reshape %v4793 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4911 = stablehlo.transpose %v4910, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4912 = stablehlo.reshape %v4911 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4913 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4914 = stablehlo.reshape %v4912 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4915 = stablehlo.reduce(%v4914 init: %v4913) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4916 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4917 = stablehlo.multiply %v4915, %v4916 : tensor<96xf32>
    %v4918 = stablehlo.subtract %s0b1nbt, %v4917 : tensor<96xf32>
    %v4919 = stablehlo.reshape %v110 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4920 = stablehlo.reshape %v4836 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4921 = stablehlo.transpose %v4919, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4922 = stablehlo.transpose %v4920, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4923 = stablehlo.convolution(%v4921, %v4922)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v4924 = stablehlo.reshape %v4923 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v4925 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v4926 = stablehlo.multiply %v4924, %v4925 : tensor<96x1x7x7xf32>
    %v4927 = stablehlo.subtract %s0b1dW, %v4926 : tensor<96x1x7x7xf32>
    %v4928 = stablehlo.reshape %v4836 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4929 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4930 = stablehlo.reduce(%v4928 init: %v4929) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4931 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4932 = stablehlo.multiply %v4930, %v4931 : tensor<96xf32>
    %v4933 = stablehlo.subtract %s0b1db, %v4932 : tensor<96xf32>
    %v4934 = stablehlo.reshape %v4844 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4935 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4936 = stablehlo.multiply %v4934, %v4935 : tensor<32x96x56x56xf32>
    %v4937 = stablehlo.reshape %v4936 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4938 = stablehlo.reshape %v4937 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4939 = stablehlo.transpose %s0b0pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4940 = stablehlo.reverse %v4939, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v4941 = stablehlo.convolution(%v4938, %v4940)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v4942 = stablehlo.reshape %v4941 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4943 = stablehlo.reshape %v4942 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4944 = stablehlo.reshape %v82 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4945 = stablehlo.multiply %v4944, %v4944 : tensor<32x384x56x56xf32>
    %v4946 = stablehlo.multiply %v4945, %v4944 : tensor<32x384x56x56xf32>
    %v4947 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v4948 = stablehlo.multiply %v4947, %v4946 : tensor<32x384x56x56xf32>
    %v4949 = stablehlo.add %v4944, %v4948 : tensor<32x384x56x56xf32>
    %v4950 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v4951 = stablehlo.multiply %v4950, %v4949 : tensor<32x384x56x56xf32>
    %v4952 = stablehlo.tanh %v4951 : tensor<32x384x56x56xf32>
    %v4953 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v4954 = stablehlo.add %v4953, %v4952 : tensor<32x384x56x56xf32>
    %v4955 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v4956 = stablehlo.multiply %v4955, %v4954 : tensor<32x384x56x56xf32>
    %v4957 = stablehlo.multiply %v4952, %v4952 : tensor<32x384x56x56xf32>
    %v4958 = stablehlo.subtract %v4953, %v4957 : tensor<32x384x56x56xf32>
    %v4959 = stablehlo.multiply %v4955, %v4944 : tensor<32x384x56x56xf32>
    %v4960 = stablehlo.multiply %v4959, %v4958 : tensor<32x384x56x56xf32>
    %v4961 = stablehlo.constant dense<0.134145> : tensor<32x384x56x56xf32>
    %v4962 = stablehlo.multiply %v4961, %v4945 : tensor<32x384x56x56xf32>
    %v4963 = stablehlo.add %v4953, %v4962 : tensor<32x384x56x56xf32>
    %v4964 = stablehlo.multiply %v4950, %v4963 : tensor<32x384x56x56xf32>
    %v4965 = stablehlo.multiply %v4960, %v4964 : tensor<32x384x56x56xf32>
    %v4966 = stablehlo.add %v4956, %v4965 : tensor<32x384x56x56xf32>
    %v4967 = stablehlo.multiply %v4943, %v4966 : tensor<32x384x56x56xf32>
    %v4968 = stablehlo.reshape %v4967 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4969 = stablehlo.reshape %v4968 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4970 = stablehlo.transpose %s0b0eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4971 = stablehlo.reverse %v4970, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v4972 = stablehlo.convolution(%v4969, %v4971)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4973 = stablehlo.reshape %v4972 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4974 = stablehlo.reshape %v43 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4975 = stablehlo.transpose %v4974, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4976 = stablehlo.reshape %v4975 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4977 = stablehlo.reshape %v4973 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4978 = stablehlo.transpose %v4977, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4979 = stablehlo.reshape %v4978 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4980 = stablehlo.reshape %v4979 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4981 = stablehlo.broadcast_in_dim %s0b0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4982 = stablehlo.multiply %v4980, %v4981 : tensor<32x3136x96xf32>
    %v4983 = stablehlo.reshape %v4982 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4984 = stablehlo.reshape %v4983 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4985 = stablehlo.reshape %v4976 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4987 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4988 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4989 = stablehlo.reduce(%v4985 init: %v4986) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4990 = stablehlo.broadcast_in_dim %v4989, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4991 = stablehlo.divide %v4990, %v4987 : tensor<32x3136x96xf32>
    %v4992 = stablehlo.subtract %v4985, %v4991 : tensor<32x3136x96xf32>
    %v4993 = stablehlo.multiply %v4992, %v4992 : tensor<32x3136x96xf32>
    %v4994 = stablehlo.reduce(%v4993 init: %v4986) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4995 = stablehlo.broadcast_in_dim %v4994, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4996 = stablehlo.divide %v4995, %v4987 : tensor<32x3136x96xf32>
    %v4997 = stablehlo.add %v4996, %v4988 : tensor<32x3136x96xf32>
    %v4998 = stablehlo.rsqrt %v4997 : tensor<32x3136x96xf32>
    %v4999 = stablehlo.multiply %v4992, %v4998 : tensor<32x3136x96xf32>
    %v5000 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v5001 = stablehlo.multiply %v5000, %v4984 : tensor<32x3136x96xf32>
    %v5002 = stablehlo.reduce(%v5001 init: %v4986) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5003 = stablehlo.broadcast_in_dim %v5002, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5004 = stablehlo.multiply %v4999, %v5001 : tensor<32x3136x96xf32>
    %v5005 = stablehlo.reduce(%v5004 init: %v4986) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5006 = stablehlo.broadcast_in_dim %v5005, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5007 = stablehlo.multiply %v5001, %v4987 : tensor<32x3136x96xf32>
    %v5008 = stablehlo.subtract %v5007, %v5003 : tensor<32x3136x96xf32>
    %v5009 = stablehlo.multiply %v4999, %v5006 : tensor<32x3136x96xf32>
    %v5010 = stablehlo.subtract %v5008, %v5009 : tensor<32x3136x96xf32>
    %v5011 = stablehlo.divide %v4998, %v4987 : tensor<32x3136x96xf32>
    %v5012 = stablehlo.multiply %v5011, %v5010 : tensor<32x3136x96xf32>
    %v5013 = stablehlo.reshape %v5012 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5014 = stablehlo.reshape %v5013 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5015 = stablehlo.transpose %v5014, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v5016 = stablehlo.reshape %v5015 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v5017 = stablehlo.reshape %v5016 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5018 = stablehlo.reverse %s0b0dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v5019 = stablehlo.convolution(%v5017, %v5018)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v5020 = stablehlo.reshape %v5019 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v5021 = stablehlo.reshape %v5020 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5022 = stablehlo.reshape %v4844 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5023 = stablehlo.add %v5021, %v5022 : tensor<32x96x56x56xf32>
    %v5024 = stablehlo.reshape %v5023 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v5025 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5026 = stablehlo.reshape %v102 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5027 = stablehlo.reshape %v4844 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5028 = stablehlo.multiply %v5026, %v5027 : tensor<32x96x56x56xf32>
    %v5029 = stablehlo.reduce(%v5028 init: %v5025) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5030 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5031 = stablehlo.multiply %v5029, %v5030 : tensor<96xf32>
    %v5032 = stablehlo.subtract %s0b0lg, %v5031 : tensor<96xf32>
    %v5033 = stablehlo.reshape %v97 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v5034 = stablehlo.reshape %v4937 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5035 = stablehlo.transpose %v5033, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v5036 = stablehlo.transpose %v5034, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v5037 = stablehlo.convolution(%v5035, %v5036)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v5038 = stablehlo.transpose %v5037, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v5039 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v5040 = stablehlo.multiply %v5038, %v5039 : tensor<96x384x1x1xf32>
    %v5041 = stablehlo.subtract %s0b0pW, %v5040 : tensor<96x384x1x1xf32>
    %v5042 = stablehlo.reshape %v4937 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5044 = stablehlo.reduce(%v5042 init: %v5043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5045 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5046 = stablehlo.multiply %v5044, %v5045 : tensor<96xf32>
    %v5047 = stablehlo.subtract %s0b0pb, %v5046 : tensor<96xf32>
    %v5048 = stablehlo.reshape %v77 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5049 = stablehlo.reshape %v4968 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v5050 = stablehlo.transpose %v5048, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v5051 = stablehlo.transpose %v5049, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v5052 = stablehlo.convolution(%v5050, %v5051)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v5053 = stablehlo.transpose %v5052, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v5054 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v5055 = stablehlo.multiply %v5053, %v5054 : tensor<384x96x1x1xf32>
    %v5056 = stablehlo.subtract %s0b0eW, %v5055 : tensor<384x96x1x1xf32>
    %v5057 = stablehlo.reshape %v4968 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v5058 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5059 = stablehlo.reduce(%v5057 init: %v5058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v5060 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v5061 = stablehlo.multiply %v5059, %v5060 : tensor<384xf32>
    %v5062 = stablehlo.subtract %s0b0eb, %v5061 : tensor<384xf32>
    %v5063 = stablehlo.reshape %v43 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5064 = stablehlo.transpose %v5063, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5065 = stablehlo.reshape %v5064 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5066 = stablehlo.reshape %v4973 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5067 = stablehlo.transpose %v5066, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5068 = stablehlo.reshape %v5067 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5069 = stablehlo.reshape %v5065 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5070 = stablehlo.reshape %v5068 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5071 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5072 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v5073 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v5074 = stablehlo.reduce(%v5069 init: %v5071) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5075 = stablehlo.broadcast_in_dim %v5074, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5076 = stablehlo.divide %v5075, %v5072 : tensor<32x3136x96xf32>
    %v5077 = stablehlo.subtract %v5069, %v5076 : tensor<32x3136x96xf32>
    %v5078 = stablehlo.multiply %v5077, %v5077 : tensor<32x3136x96xf32>
    %v5079 = stablehlo.reduce(%v5078 init: %v5071) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5080 = stablehlo.broadcast_in_dim %v5079, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5081 = stablehlo.divide %v5080, %v5072 : tensor<32x3136x96xf32>
    %v5082 = stablehlo.add %v5081, %v5073 : tensor<32x3136x96xf32>
    %v5083 = stablehlo.rsqrt %v5082 : tensor<32x3136x96xf32>
    %v5084 = stablehlo.multiply %v5077, %v5083 : tensor<32x3136x96xf32>
    %v5085 = stablehlo.multiply %v5070, %v5084 : tensor<32x3136x96xf32>
    %v5086 = stablehlo.reduce(%v5085 init: %v5071) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v5087 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5088 = stablehlo.multiply %v5086, %v5087 : tensor<96xf32>
    %v5089 = stablehlo.subtract %s0b0ng, %v5088 : tensor<96xf32>
    %v5090 = stablehlo.reshape %v4973 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5091 = stablehlo.transpose %v5090, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5092 = stablehlo.reshape %v5091 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5094 = stablehlo.reshape %v5092 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5095 = stablehlo.reduce(%v5094 init: %v5093) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v5096 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5097 = stablehlo.multiply %v5095, %v5096 : tensor<96xf32>
    %v5098 = stablehlo.subtract %s0b0nbt, %v5097 : tensor<96xf32>
    %v5099 = stablehlo.reshape %v38 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5100 = stablehlo.reshape %v5016 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5101 = stablehlo.transpose %v5099, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v5102 = stablehlo.transpose %v5100, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v5103 = stablehlo.convolution(%v5101, %v5102)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v5104 = stablehlo.reshape %v5103 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v5105 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v5106 = stablehlo.multiply %v5104, %v5105 : tensor<96x1x7x7xf32>
    %v5107 = stablehlo.subtract %s0b0dW, %v5106 : tensor<96x1x7x7xf32>
    %v5108 = stablehlo.reshape %v5016 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5110 = stablehlo.reduce(%v5108 init: %v5109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5111 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5112 = stablehlo.multiply %v5110, %v5111 : tensor<96xf32>
    %v5113 = stablehlo.subtract %s0b0db, %v5112 : tensor<96xf32>
    %v5114 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5115 = stablehlo.transpose %v5114, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5116 = stablehlo.reshape %v5115 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5117 = stablehlo.reshape %v5024 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5118 = stablehlo.transpose %v5117, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5119 = stablehlo.reshape %v5118 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5120 = stablehlo.reshape %v5116 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5121 = stablehlo.reshape %v5119 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5123 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v5124 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v5125 = stablehlo.reduce(%v5120 init: %v5122) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5126 = stablehlo.broadcast_in_dim %v5125, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5127 = stablehlo.divide %v5126, %v5123 : tensor<32x3136x96xf32>
    %v5128 = stablehlo.subtract %v5120, %v5127 : tensor<32x3136x96xf32>
    %v5129 = stablehlo.multiply %v5128, %v5128 : tensor<32x3136x96xf32>
    %v5130 = stablehlo.reduce(%v5129 init: %v5122) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5131 = stablehlo.broadcast_in_dim %v5130, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5132 = stablehlo.divide %v5131, %v5123 : tensor<32x3136x96xf32>
    %v5133 = stablehlo.add %v5132, %v5124 : tensor<32x3136x96xf32>
    %v5134 = stablehlo.rsqrt %v5133 : tensor<32x3136x96xf32>
    %v5135 = stablehlo.multiply %v5128, %v5134 : tensor<32x3136x96xf32>
    %v5136 = stablehlo.multiply %v5121, %v5135 : tensor<32x3136x96xf32>
    %v5137 = stablehlo.reduce(%v5136 init: %v5122) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v5138 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5139 = stablehlo.multiply %v5137, %v5138 : tensor<96xf32>
    %v5140 = stablehlo.subtract %psng, %v5139 : tensor<96xf32>
    %v5141 = stablehlo.reshape %v5024 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5142 = stablehlo.transpose %v5141, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5143 = stablehlo.reshape %v5142 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5145 = stablehlo.reshape %v5143 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5146 = stablehlo.reduce(%v5145 init: %v5144) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v5147 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5148 = stablehlo.multiply %v5146, %v5147 : tensor<96xf32>
    %v5149 = stablehlo.subtract %psnbt, %v5148 : tensor<96xf32>
    %v5150 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5151 = stablehlo.transpose %v5150, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5152 = stablehlo.reshape %v5151 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5153 = stablehlo.reshape %v5024 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5154 = stablehlo.transpose %v5153, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5155 = stablehlo.reshape %v5154 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5156 = stablehlo.reshape %v5155 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5157 = stablehlo.broadcast_in_dim %psng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v5158 = stablehlo.multiply %v5156, %v5157 : tensor<32x3136x96xf32>
    %v5159 = stablehlo.reshape %v5158 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5160 = stablehlo.reshape %v5159 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5161 = stablehlo.reshape %v5152 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5163 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v5164 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v5165 = stablehlo.reduce(%v5161 init: %v5162) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5166 = stablehlo.broadcast_in_dim %v5165, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5167 = stablehlo.divide %v5166, %v5163 : tensor<32x3136x96xf32>
    %v5168 = stablehlo.subtract %v5161, %v5167 : tensor<32x3136x96xf32>
    %v5169 = stablehlo.multiply %v5168, %v5168 : tensor<32x3136x96xf32>
    %v5170 = stablehlo.reduce(%v5169 init: %v5162) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5171 = stablehlo.broadcast_in_dim %v5170, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5172 = stablehlo.divide %v5171, %v5163 : tensor<32x3136x96xf32>
    %v5173 = stablehlo.add %v5172, %v5164 : tensor<32x3136x96xf32>
    %v5174 = stablehlo.rsqrt %v5173 : tensor<32x3136x96xf32>
    %v5175 = stablehlo.multiply %v5168, %v5174 : tensor<32x3136x96xf32>
    %v5176 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v5177 = stablehlo.multiply %v5176, %v5160 : tensor<32x3136x96xf32>
    %v5178 = stablehlo.reduce(%v5177 init: %v5162) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5179 = stablehlo.broadcast_in_dim %v5178, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5180 = stablehlo.multiply %v5175, %v5177 : tensor<32x3136x96xf32>
    %v5181 = stablehlo.reduce(%v5180 init: %v5162) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5182 = stablehlo.broadcast_in_dim %v5181, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5183 = stablehlo.multiply %v5177, %v5163 : tensor<32x3136x96xf32>
    %v5184 = stablehlo.subtract %v5183, %v5179 : tensor<32x3136x96xf32>
    %v5185 = stablehlo.multiply %v5175, %v5182 : tensor<32x3136x96xf32>
    %v5186 = stablehlo.subtract %v5184, %v5185 : tensor<32x3136x96xf32>
    %v5187 = stablehlo.divide %v5174, %v5163 : tensor<32x3136x96xf32>
    %v5188 = stablehlo.multiply %v5187, %v5186 : tensor<32x3136x96xf32>
    %v5189 = stablehlo.reshape %v5188 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5190 = stablehlo.reshape %v5189 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5191 = stablehlo.transpose %v5190, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v5192 = stablehlo.reshape %v5191 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v5199 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v5200 = stablehlo.reshape %v5192 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5202 = stablehlo.pad %v5200, %v5201, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x221x221xf32>
    %v5203 = stablehlo.transpose %v5199, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v5204 = stablehlo.transpose %v5202, dims = [1, 0, 2, 3] : (tensor<32x96x221x221xf32>) -> tensor<96x32x221x221xf32>
    %v5205 = stablehlo.convolution(%v5203, %v5204)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<96x32x221x221xf32>) -> tensor<3x96x4x4xf32>
    %v5206 = stablehlo.transpose %v5205, dims = [1, 0, 2, 3] : (tensor<3x96x4x4xf32>) -> tensor<96x3x4x4xf32>
    %psWl = stablehlo.constant dense<0.1> : tensor<96x3x4x4xf32>
    %psWs = stablehlo.multiply %v5206, %psWl : tensor<96x3x4x4xf32>
    %psWn = stablehlo.subtract %psW, %psWs : tensor<96x3x4x4xf32>
    %v5193 = stablehlo.reshape %v5192 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5195 = stablehlo.reduce(%v5193 init: %v5194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5196 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5197 = stablehlo.multiply %v5195, %v5196 : tensor<96xf32>
    %v5198 = stablehlo.subtract %psb, %v5197 : tensor<96xf32>
    return %psWn, %v5198, %v5140, %v5149, %v5107, %v5113, %v5089, %v5098, %v5056, %v5062, %v5041, %v5047, %v5032, %v4927, %v4933, %v4909, %v4918, %v4876, %v4882, %v4861, %v4867, %v4852, %v4747, %v4753, %v4729, %v4738, %v4696, %v4702, %v4681, %v4687, %v4672, %v4553, %v4562, %v4573, %v4526, %v4464, %v4470, %v4446, %v4455, %v4413, %v4419, %v4398, %v4404, %v4389, %v4284, %v4290, %v4266, %v4275, %v4233, %v4239, %v4218, %v4224, %v4209, %v4104, %v4110, %v4086, %v4095, %v4053, %v4059, %v4038, %v4044, %v4029, %v3910, %v3919, %v3930, %v3883, %v3821, %v3827, %v3803, %v3812, %v3770, %v3776, %v3755, %v3761, %v3746, %v3641, %v3647, %v3623, %v3632, %v3590, %v3596, %v3575, %v3581, %v3566, %v3461, %v3467, %v3443, %v3452, %v3410, %v3416, %v3395, %v3401, %v3386, %v3281, %v3287, %v3263, %v3272, %v3230, %v3236, %v3215, %v3221, %v3206, %v3101, %v3107, %v3083, %v3092, %v3050, %v3056, %v3035, %v3041, %v3026, %v2921, %v2927, %v2903, %v2912, %v2870, %v2876, %v2855, %v2861, %v2846, %v2741, %v2747, %v2723, %v2732, %v2690, %v2696, %v2675, %v2681, %v2666, %v2561, %v2567, %v2543, %v2552, %v2510, %v2516, %v2495, %v2501, %v2486, %v2381, %v2387, %v2363, %v2372, %v2330, %v2336, %v2315, %v2321, %v2306, %v2187, %v2196, %v2207, %v2160, %v2098, %v2104, %v2080, %v2089, %v2047, %v2053, %v2032, %v2038, %v2023, %v1918, %v1924, %v1900, %v1909, %v1867, %v1873, %v1852, %v1858, %v1843, %v1738, %v1744, %v1720, %v1729, %v1687, %v1693, %v1672, %v1678, %v1663, %v1558, %v1564, %v1532, %v1537 : tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x10xf32>, tensor<10xf32>
  }
}
