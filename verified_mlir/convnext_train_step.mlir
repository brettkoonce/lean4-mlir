module @m {
  func.func @convnext_train_step(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %psng: tensor<96xf32>, %psnbt: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<96xf32>, %s0b0nbt: tensor<96xf32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<96xf32>, %s0b1nbt: tensor<96xf32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<96xf32>, %s0b2nbt: tensor<96xf32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<96xf32>, %d0nbt: tensor<96xf32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<192xf32>, %s1b0nbt: tensor<192xf32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<192xf32>, %s1b1nbt: tensor<192xf32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<192xf32>, %s1b2nbt: tensor<192xf32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<192xf32>, %d1nbt: tensor<192xf32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<384xf32>, %s2b0nbt: tensor<384xf32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<384xf32>, %s2b1nbt: tensor<384xf32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<384xf32>, %s2b2nbt: tensor<384xf32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<384xf32>, %s2b3nbt: tensor<384xf32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<384xf32>, %s2b4nbt: tensor<384xf32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<384xf32>, %s2b5nbt: tensor<384xf32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<384xf32>, %s2b6nbt: tensor<384xf32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<384xf32>, %s2b7nbt: tensor<384xf32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<384xf32>, %s2b8nbt: tensor<384xf32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<384xf32>, %d2nbt: tensor<384xf32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<768xf32>, %s3b0nbt: tensor<768xf32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<768xf32>, %s3b1nbt: tensor<768xf32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<768xf32>, %s3b2nbt: tensor<768xf32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x10xf32>, tensor<10xf32>) {
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
    %v83 = stablehlo.multiply %v82, %v82 : tensor<32x1204224xf32>
    %v84 = stablehlo.multiply %v83, %v82 : tensor<32x1204224xf32>
    %v85 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v86 = stablehlo.multiply %v85, %v84 : tensor<32x1204224xf32>
    %v87 = stablehlo.add %v82, %v86 : tensor<32x1204224xf32>
    %v88 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v89 = stablehlo.multiply %v88, %v87 : tensor<32x1204224xf32>
    %v90 = stablehlo.tanh %v89 : tensor<32x1204224xf32>
    %v91 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v92 = stablehlo.add %v91, %v90 : tensor<32x1204224xf32>
    %v93 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v94 = stablehlo.multiply %v93, %v82 : tensor<32x1204224xf32>
    %v95 = stablehlo.multiply %v94, %v92 : tensor<32x1204224xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v97 = stablehlo.convolution(%v96, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v98 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v99 = stablehlo.add %v97, %v98 : tensor<32x96x56x56xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v102 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v103 = stablehlo.multiply %v101, %v102 : tensor<32x96x56x56xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v105 = stablehlo.add %v104, %v38 : tensor<32x301056xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v107 = stablehlo.convolution(%v106, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v108 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v109 = stablehlo.add %v107, %v108 : tensor<32x96x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v112 = stablehlo.transpose %v111, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v116 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v117 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v118 = stablehlo.reduce(%v114 init: %v115) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v119 = stablehlo.broadcast_in_dim %v118, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v120 = stablehlo.divide %v119, %v116 : tensor<32x3136x96xf32>
    %v121 = stablehlo.subtract %v114, %v120 : tensor<32x3136x96xf32>
    %v122 = stablehlo.multiply %v121, %v121 : tensor<32x3136x96xf32>
    %v123 = stablehlo.reduce(%v122 init: %v115) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v124 = stablehlo.broadcast_in_dim %v123, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v125 = stablehlo.divide %v124, %v116 : tensor<32x3136x96xf32>
    %v126 = stablehlo.add %v125, %v117 : tensor<32x3136x96xf32>
    %v127 = stablehlo.rsqrt %v126 : tensor<32x3136x96xf32>
    %v128 = stablehlo.multiply %v121, %v127 : tensor<32x3136x96xf32>
    %v129 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v130 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v131 = stablehlo.multiply %v128, %v129 : tensor<32x3136x96xf32>
    %v132 = stablehlo.add %v131, %v130 : tensor<32x3136x96xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v134 = stablehlo.reshape %v133 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v135 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v136 = stablehlo.multiply %v134, %v135 : tensor<32x3136x96xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v139 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v140 = stablehlo.add %v138, %v139 : tensor<32x3136x96xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v143 = stablehlo.transpose %v142, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v146 = stablehlo.convolution(%v145, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v147 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v148 = stablehlo.add %v146, %v147 : tensor<32x384x56x56xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v150 = stablehlo.multiply %v149, %v149 : tensor<32x1204224xf32>
    %v151 = stablehlo.multiply %v150, %v149 : tensor<32x1204224xf32>
    %v152 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v153 = stablehlo.multiply %v152, %v151 : tensor<32x1204224xf32>
    %v154 = stablehlo.add %v149, %v153 : tensor<32x1204224xf32>
    %v155 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v156 = stablehlo.multiply %v155, %v154 : tensor<32x1204224xf32>
    %v157 = stablehlo.tanh %v156 : tensor<32x1204224xf32>
    %v158 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v159 = stablehlo.add %v158, %v157 : tensor<32x1204224xf32>
    %v160 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v161 = stablehlo.multiply %v160, %v149 : tensor<32x1204224xf32>
    %v162 = stablehlo.multiply %v161, %v159 : tensor<32x1204224xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v164 = stablehlo.convolution(%v163, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v165 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v166 = stablehlo.add %v164, %v165 : tensor<32x96x56x56xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v169 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v170 = stablehlo.multiply %v168, %v169 : tensor<32x96x56x56xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v172 = stablehlo.add %v171, %v105 : tensor<32x301056xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v174 = stablehlo.convolution(%v173, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v175 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v176 = stablehlo.add %v174, %v175 : tensor<32x96x56x56xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v179 = stablehlo.transpose %v178, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v183 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v184 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v185 = stablehlo.reduce(%v181 init: %v182) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v186 = stablehlo.broadcast_in_dim %v185, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v187 = stablehlo.divide %v186, %v183 : tensor<32x3136x96xf32>
    %v188 = stablehlo.subtract %v181, %v187 : tensor<32x3136x96xf32>
    %v189 = stablehlo.multiply %v188, %v188 : tensor<32x3136x96xf32>
    %v190 = stablehlo.reduce(%v189 init: %v182) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v191 = stablehlo.broadcast_in_dim %v190, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v192 = stablehlo.divide %v191, %v183 : tensor<32x3136x96xf32>
    %v193 = stablehlo.add %v192, %v184 : tensor<32x3136x96xf32>
    %v194 = stablehlo.rsqrt %v193 : tensor<32x3136x96xf32>
    %v195 = stablehlo.multiply %v188, %v194 : tensor<32x3136x96xf32>
    %v196 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v197 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v198 = stablehlo.multiply %v195, %v196 : tensor<32x3136x96xf32>
    %v199 = stablehlo.add %v198, %v197 : tensor<32x3136x96xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v202 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v203 = stablehlo.multiply %v201, %v202 : tensor<32x3136x96xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v206 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v207 = stablehlo.add %v205, %v206 : tensor<32x3136x96xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v210 = stablehlo.transpose %v209, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v213 = stablehlo.convolution(%v212, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v214 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v215 = stablehlo.add %v213, %v214 : tensor<32x384x56x56xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v217 = stablehlo.multiply %v216, %v216 : tensor<32x1204224xf32>
    %v218 = stablehlo.multiply %v217, %v216 : tensor<32x1204224xf32>
    %v219 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v220 = stablehlo.multiply %v219, %v218 : tensor<32x1204224xf32>
    %v221 = stablehlo.add %v216, %v220 : tensor<32x1204224xf32>
    %v222 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v223 = stablehlo.multiply %v222, %v221 : tensor<32x1204224xf32>
    %v224 = stablehlo.tanh %v223 : tensor<32x1204224xf32>
    %v225 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v226 = stablehlo.add %v225, %v224 : tensor<32x1204224xf32>
    %v227 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v228 = stablehlo.multiply %v227, %v216 : tensor<32x1204224xf32>
    %v229 = stablehlo.multiply %v228, %v226 : tensor<32x1204224xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v231 = stablehlo.convolution(%v230, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v232 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v233 = stablehlo.add %v231, %v232 : tensor<32x96x56x56xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v236 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v237 = stablehlo.multiply %v235, %v236 : tensor<32x96x56x56xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v239 = stablehlo.add %v238, %v172 : tensor<32x301056xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v241 = stablehlo.transpose %v240, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v245 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v246 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v247 = stablehlo.reduce(%v243 init: %v244) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v248 = stablehlo.broadcast_in_dim %v247, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v249 = stablehlo.divide %v248, %v245 : tensor<32x3136x96xf32>
    %v250 = stablehlo.subtract %v243, %v249 : tensor<32x3136x96xf32>
    %v251 = stablehlo.multiply %v250, %v250 : tensor<32x3136x96xf32>
    %v252 = stablehlo.reduce(%v251 init: %v244) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v253 = stablehlo.broadcast_in_dim %v252, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v254 = stablehlo.divide %v253, %v245 : tensor<32x3136x96xf32>
    %v255 = stablehlo.add %v254, %v246 : tensor<32x3136x96xf32>
    %v256 = stablehlo.rsqrt %v255 : tensor<32x3136x96xf32>
    %v257 = stablehlo.multiply %v250, %v256 : tensor<32x3136x96xf32>
    %v258 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v259 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v260 = stablehlo.multiply %v257, %v258 : tensor<32x3136x96xf32>
    %v261 = stablehlo.add %v260, %v259 : tensor<32x3136x96xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v264 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v265 = stablehlo.multiply %v263, %v264 : tensor<32x3136x96xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v268 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v269 = stablehlo.add %v267, %v268 : tensor<32x3136x96xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v272 = stablehlo.transpose %v271, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v275 = stablehlo.convolution(%v274, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v276 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v277 = stablehlo.add %v275, %v276 : tensor<32x192x28x28xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v280 = stablehlo.convolution(%v279, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v281 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v282 = stablehlo.add %v280, %v281 : tensor<32x192x28x28xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v285 = stablehlo.transpose %v284, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v289 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v290 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v291 = stablehlo.reduce(%v287 init: %v288) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v292 = stablehlo.broadcast_in_dim %v291, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v293 = stablehlo.divide %v292, %v289 : tensor<32x784x192xf32>
    %v294 = stablehlo.subtract %v287, %v293 : tensor<32x784x192xf32>
    %v295 = stablehlo.multiply %v294, %v294 : tensor<32x784x192xf32>
    %v296 = stablehlo.reduce(%v295 init: %v288) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v297 = stablehlo.broadcast_in_dim %v296, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v298 = stablehlo.divide %v297, %v289 : tensor<32x784x192xf32>
    %v299 = stablehlo.add %v298, %v290 : tensor<32x784x192xf32>
    %v300 = stablehlo.rsqrt %v299 : tensor<32x784x192xf32>
    %v301 = stablehlo.multiply %v294, %v300 : tensor<32x784x192xf32>
    %v302 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v303 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v304 = stablehlo.multiply %v301, %v302 : tensor<32x784x192xf32>
    %v305 = stablehlo.add %v304, %v303 : tensor<32x784x192xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v308 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v309 = stablehlo.multiply %v307, %v308 : tensor<32x784x192xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v312 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v313 = stablehlo.add %v311, %v312 : tensor<32x784x192xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v316 = stablehlo.transpose %v315, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v319 = stablehlo.convolution(%v318, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v320 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v321 = stablehlo.add %v319, %v320 : tensor<32x768x28x28xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v323 = stablehlo.multiply %v322, %v322 : tensor<32x602112xf32>
    %v324 = stablehlo.multiply %v323, %v322 : tensor<32x602112xf32>
    %v325 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v326 = stablehlo.multiply %v325, %v324 : tensor<32x602112xf32>
    %v327 = stablehlo.add %v322, %v326 : tensor<32x602112xf32>
    %v328 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v329 = stablehlo.multiply %v328, %v327 : tensor<32x602112xf32>
    %v330 = stablehlo.tanh %v329 : tensor<32x602112xf32>
    %v331 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v332 = stablehlo.add %v331, %v330 : tensor<32x602112xf32>
    %v333 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v334 = stablehlo.multiply %v333, %v322 : tensor<32x602112xf32>
    %v335 = stablehlo.multiply %v334, %v332 : tensor<32x602112xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v337 = stablehlo.convolution(%v336, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v338 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v339 = stablehlo.add %v337, %v338 : tensor<32x192x28x28xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v342 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v343 = stablehlo.multiply %v341, %v342 : tensor<32x192x28x28xf32>
    %v344 = stablehlo.reshape %v343 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v345 = stablehlo.add %v344, %v278 : tensor<32x150528xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v347 = stablehlo.convolution(%v346, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v349 = stablehlo.add %v347, %v348 : tensor<32x192x28x28xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v352 = stablehlo.transpose %v351, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v356 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v357 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v358 = stablehlo.reduce(%v354 init: %v355) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v359 = stablehlo.broadcast_in_dim %v358, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v360 = stablehlo.divide %v359, %v356 : tensor<32x784x192xf32>
    %v361 = stablehlo.subtract %v354, %v360 : tensor<32x784x192xf32>
    %v362 = stablehlo.multiply %v361, %v361 : tensor<32x784x192xf32>
    %v363 = stablehlo.reduce(%v362 init: %v355) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v364 = stablehlo.broadcast_in_dim %v363, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v365 = stablehlo.divide %v364, %v356 : tensor<32x784x192xf32>
    %v366 = stablehlo.add %v365, %v357 : tensor<32x784x192xf32>
    %v367 = stablehlo.rsqrt %v366 : tensor<32x784x192xf32>
    %v368 = stablehlo.multiply %v361, %v367 : tensor<32x784x192xf32>
    %v369 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v370 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v371 = stablehlo.multiply %v368, %v369 : tensor<32x784x192xf32>
    %v372 = stablehlo.add %v371, %v370 : tensor<32x784x192xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v375 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v376 = stablehlo.multiply %v374, %v375 : tensor<32x784x192xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v379 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v380 = stablehlo.add %v378, %v379 : tensor<32x784x192xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v383 = stablehlo.transpose %v382, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v386 = stablehlo.convolution(%v385, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v387 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v388 = stablehlo.add %v386, %v387 : tensor<32x768x28x28xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v390 = stablehlo.multiply %v389, %v389 : tensor<32x602112xf32>
    %v391 = stablehlo.multiply %v390, %v389 : tensor<32x602112xf32>
    %v392 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v393 = stablehlo.multiply %v392, %v391 : tensor<32x602112xf32>
    %v394 = stablehlo.add %v389, %v393 : tensor<32x602112xf32>
    %v395 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v396 = stablehlo.multiply %v395, %v394 : tensor<32x602112xf32>
    %v397 = stablehlo.tanh %v396 : tensor<32x602112xf32>
    %v398 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v399 = stablehlo.add %v398, %v397 : tensor<32x602112xf32>
    %v400 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v401 = stablehlo.multiply %v400, %v389 : tensor<32x602112xf32>
    %v402 = stablehlo.multiply %v401, %v399 : tensor<32x602112xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v404 = stablehlo.convolution(%v403, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v405 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v406 = stablehlo.add %v404, %v405 : tensor<32x192x28x28xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v409 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v410 = stablehlo.multiply %v408, %v409 : tensor<32x192x28x28xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v412 = stablehlo.add %v411, %v345 : tensor<32x150528xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v414 = stablehlo.convolution(%v413, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v415 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v416 = stablehlo.add %v414, %v415 : tensor<32x192x28x28xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v419 = stablehlo.transpose %v418, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v420 = stablehlo.reshape %v419 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v423 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v424 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v425 = stablehlo.reduce(%v421 init: %v422) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v426 = stablehlo.broadcast_in_dim %v425, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v427 = stablehlo.divide %v426, %v423 : tensor<32x784x192xf32>
    %v428 = stablehlo.subtract %v421, %v427 : tensor<32x784x192xf32>
    %v429 = stablehlo.multiply %v428, %v428 : tensor<32x784x192xf32>
    %v430 = stablehlo.reduce(%v429 init: %v422) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v431 = stablehlo.broadcast_in_dim %v430, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v432 = stablehlo.divide %v431, %v423 : tensor<32x784x192xf32>
    %v433 = stablehlo.add %v432, %v424 : tensor<32x784x192xf32>
    %v434 = stablehlo.rsqrt %v433 : tensor<32x784x192xf32>
    %v435 = stablehlo.multiply %v428, %v434 : tensor<32x784x192xf32>
    %v436 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v437 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v438 = stablehlo.multiply %v435, %v436 : tensor<32x784x192xf32>
    %v439 = stablehlo.add %v438, %v437 : tensor<32x784x192xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v442 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v443 = stablehlo.multiply %v441, %v442 : tensor<32x784x192xf32>
    %v444 = stablehlo.reshape %v443 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v446 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v447 = stablehlo.add %v445, %v446 : tensor<32x784x192xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v450 = stablehlo.transpose %v449, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v453 = stablehlo.convolution(%v452, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v454 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v455 = stablehlo.add %v453, %v454 : tensor<32x768x28x28xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v457 = stablehlo.multiply %v456, %v456 : tensor<32x602112xf32>
    %v458 = stablehlo.multiply %v457, %v456 : tensor<32x602112xf32>
    %v459 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v460 = stablehlo.multiply %v459, %v458 : tensor<32x602112xf32>
    %v461 = stablehlo.add %v456, %v460 : tensor<32x602112xf32>
    %v462 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v463 = stablehlo.multiply %v462, %v461 : tensor<32x602112xf32>
    %v464 = stablehlo.tanh %v463 : tensor<32x602112xf32>
    %v465 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v466 = stablehlo.add %v465, %v464 : tensor<32x602112xf32>
    %v467 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v468 = stablehlo.multiply %v467, %v456 : tensor<32x602112xf32>
    %v469 = stablehlo.multiply %v468, %v466 : tensor<32x602112xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v471 = stablehlo.convolution(%v470, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v472 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v473 = stablehlo.add %v471, %v472 : tensor<32x192x28x28xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v476 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v477 = stablehlo.multiply %v475, %v476 : tensor<32x192x28x28xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v479 = stablehlo.add %v478, %v412 : tensor<32x150528xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v481 = stablehlo.transpose %v480, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v485 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v486 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v487 = stablehlo.reduce(%v483 init: %v484) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v488 = stablehlo.broadcast_in_dim %v487, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v489 = stablehlo.divide %v488, %v485 : tensor<32x784x192xf32>
    %v490 = stablehlo.subtract %v483, %v489 : tensor<32x784x192xf32>
    %v491 = stablehlo.multiply %v490, %v490 : tensor<32x784x192xf32>
    %v492 = stablehlo.reduce(%v491 init: %v484) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v493 = stablehlo.broadcast_in_dim %v492, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v494 = stablehlo.divide %v493, %v485 : tensor<32x784x192xf32>
    %v495 = stablehlo.add %v494, %v486 : tensor<32x784x192xf32>
    %v496 = stablehlo.rsqrt %v495 : tensor<32x784x192xf32>
    %v497 = stablehlo.multiply %v490, %v496 : tensor<32x784x192xf32>
    %v498 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v499 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v500 = stablehlo.multiply %v497, %v498 : tensor<32x784x192xf32>
    %v501 = stablehlo.add %v500, %v499 : tensor<32x784x192xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v504 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v505 = stablehlo.multiply %v503, %v504 : tensor<32x784x192xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v508 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v509 = stablehlo.add %v507, %v508 : tensor<32x784x192xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v512 = stablehlo.transpose %v511, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v515 = stablehlo.convolution(%v514, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v516 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v517 = stablehlo.add %v515, %v516 : tensor<32x384x14x14xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v520 = stablehlo.convolution(%v519, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v521 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v522 = stablehlo.add %v520, %v521 : tensor<32x384x14x14xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v524 = stablehlo.reshape %v523 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v525 = stablehlo.transpose %v524, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v527 = stablehlo.reshape %v526 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v529 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v530 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v531 = stablehlo.reduce(%v527 init: %v528) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v532 = stablehlo.broadcast_in_dim %v531, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v533 = stablehlo.divide %v532, %v529 : tensor<32x196x384xf32>
    %v534 = stablehlo.subtract %v527, %v533 : tensor<32x196x384xf32>
    %v535 = stablehlo.multiply %v534, %v534 : tensor<32x196x384xf32>
    %v536 = stablehlo.reduce(%v535 init: %v528) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v537 = stablehlo.broadcast_in_dim %v536, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v538 = stablehlo.divide %v537, %v529 : tensor<32x196x384xf32>
    %v539 = stablehlo.add %v538, %v530 : tensor<32x196x384xf32>
    %v540 = stablehlo.rsqrt %v539 : tensor<32x196x384xf32>
    %v541 = stablehlo.multiply %v534, %v540 : tensor<32x196x384xf32>
    %v542 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v543 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v544 = stablehlo.multiply %v541, %v542 : tensor<32x196x384xf32>
    %v545 = stablehlo.add %v544, %v543 : tensor<32x196x384xf32>
    %v546 = stablehlo.reshape %v545 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v548 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v549 = stablehlo.multiply %v547, %v548 : tensor<32x196x384xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v552 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v553 = stablehlo.add %v551, %v552 : tensor<32x196x384xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v556 = stablehlo.transpose %v555, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v559 = stablehlo.convolution(%v558, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v560 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v561 = stablehlo.add %v559, %v560 : tensor<32x1536x14x14xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v563 = stablehlo.multiply %v562, %v562 : tensor<32x301056xf32>
    %v564 = stablehlo.multiply %v563, %v562 : tensor<32x301056xf32>
    %v565 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v566 = stablehlo.multiply %v565, %v564 : tensor<32x301056xf32>
    %v567 = stablehlo.add %v562, %v566 : tensor<32x301056xf32>
    %v568 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v569 = stablehlo.multiply %v568, %v567 : tensor<32x301056xf32>
    %v570 = stablehlo.tanh %v569 : tensor<32x301056xf32>
    %v571 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v572 = stablehlo.add %v571, %v570 : tensor<32x301056xf32>
    %v573 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v574 = stablehlo.multiply %v573, %v562 : tensor<32x301056xf32>
    %v575 = stablehlo.multiply %v574, %v572 : tensor<32x301056xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v577 = stablehlo.convolution(%v576, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<32x384x14x14xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v582 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v583 = stablehlo.multiply %v581, %v582 : tensor<32x384x14x14xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v585 = stablehlo.add %v584, %v518 : tensor<32x75264xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v587 = stablehlo.convolution(%v586, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v588 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v589 = stablehlo.add %v587, %v588 : tensor<32x384x14x14xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v592 = stablehlo.transpose %v591, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v595 = stablehlo.constant dense<0.0> : tensor<f32>
    %v596 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v597 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v598 = stablehlo.reduce(%v594 init: %v595) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v599 = stablehlo.broadcast_in_dim %v598, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v600 = stablehlo.divide %v599, %v596 : tensor<32x196x384xf32>
    %v601 = stablehlo.subtract %v594, %v600 : tensor<32x196x384xf32>
    %v602 = stablehlo.multiply %v601, %v601 : tensor<32x196x384xf32>
    %v603 = stablehlo.reduce(%v602 init: %v595) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v604 = stablehlo.broadcast_in_dim %v603, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v605 = stablehlo.divide %v604, %v596 : tensor<32x196x384xf32>
    %v606 = stablehlo.add %v605, %v597 : tensor<32x196x384xf32>
    %v607 = stablehlo.rsqrt %v606 : tensor<32x196x384xf32>
    %v608 = stablehlo.multiply %v601, %v607 : tensor<32x196x384xf32>
    %v609 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v610 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v611 = stablehlo.multiply %v608, %v609 : tensor<32x196x384xf32>
    %v612 = stablehlo.add %v611, %v610 : tensor<32x196x384xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v614 = stablehlo.reshape %v613 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v615 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v616 = stablehlo.multiply %v614, %v615 : tensor<32x196x384xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v619 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v620 = stablehlo.add %v618, %v619 : tensor<32x196x384xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v622 = stablehlo.reshape %v621 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v623 = stablehlo.transpose %v622, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v624 = stablehlo.reshape %v623 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v626 = stablehlo.convolution(%v625, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v627 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v628 = stablehlo.add %v626, %v627 : tensor<32x1536x14x14xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v630 = stablehlo.multiply %v629, %v629 : tensor<32x301056xf32>
    %v631 = stablehlo.multiply %v630, %v629 : tensor<32x301056xf32>
    %v632 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v633 = stablehlo.multiply %v632, %v631 : tensor<32x301056xf32>
    %v634 = stablehlo.add %v629, %v633 : tensor<32x301056xf32>
    %v635 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v636 = stablehlo.multiply %v635, %v634 : tensor<32x301056xf32>
    %v637 = stablehlo.tanh %v636 : tensor<32x301056xf32>
    %v638 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v639 = stablehlo.add %v638, %v637 : tensor<32x301056xf32>
    %v640 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v641 = stablehlo.multiply %v640, %v629 : tensor<32x301056xf32>
    %v642 = stablehlo.multiply %v641, %v639 : tensor<32x301056xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v644 = stablehlo.convolution(%v643, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v645 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v646 = stablehlo.add %v644, %v645 : tensor<32x384x14x14xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v649 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v650 = stablehlo.multiply %v648, %v649 : tensor<32x384x14x14xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v652 = stablehlo.add %v651, %v585 : tensor<32x75264xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v654 = stablehlo.convolution(%v653, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v655 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v656 = stablehlo.add %v654, %v655 : tensor<32x384x14x14xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v659 = stablehlo.transpose %v658, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v663 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v664 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v665 = stablehlo.reduce(%v661 init: %v662) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v666 = stablehlo.broadcast_in_dim %v665, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v667 = stablehlo.divide %v666, %v663 : tensor<32x196x384xf32>
    %v668 = stablehlo.subtract %v661, %v667 : tensor<32x196x384xf32>
    %v669 = stablehlo.multiply %v668, %v668 : tensor<32x196x384xf32>
    %v670 = stablehlo.reduce(%v669 init: %v662) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v671 = stablehlo.broadcast_in_dim %v670, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v672 = stablehlo.divide %v671, %v663 : tensor<32x196x384xf32>
    %v673 = stablehlo.add %v672, %v664 : tensor<32x196x384xf32>
    %v674 = stablehlo.rsqrt %v673 : tensor<32x196x384xf32>
    %v675 = stablehlo.multiply %v668, %v674 : tensor<32x196x384xf32>
    %v676 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v677 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v678 = stablehlo.multiply %v675, %v676 : tensor<32x196x384xf32>
    %v679 = stablehlo.add %v678, %v677 : tensor<32x196x384xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v682 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v683 = stablehlo.multiply %v681, %v682 : tensor<32x196x384xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v686 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v687 = stablehlo.add %v685, %v686 : tensor<32x196x384xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v690 = stablehlo.transpose %v689, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v693 = stablehlo.convolution(%v692, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v694 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v695 = stablehlo.add %v693, %v694 : tensor<32x1536x14x14xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v697 = stablehlo.multiply %v696, %v696 : tensor<32x301056xf32>
    %v698 = stablehlo.multiply %v697, %v696 : tensor<32x301056xf32>
    %v699 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v700 = stablehlo.multiply %v699, %v698 : tensor<32x301056xf32>
    %v701 = stablehlo.add %v696, %v700 : tensor<32x301056xf32>
    %v702 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v703 = stablehlo.multiply %v702, %v701 : tensor<32x301056xf32>
    %v704 = stablehlo.tanh %v703 : tensor<32x301056xf32>
    %v705 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v706 = stablehlo.add %v705, %v704 : tensor<32x301056xf32>
    %v707 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v708 = stablehlo.multiply %v707, %v696 : tensor<32x301056xf32>
    %v709 = stablehlo.multiply %v708, %v706 : tensor<32x301056xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v711 = stablehlo.convolution(%v710, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v712 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v713 = stablehlo.add %v711, %v712 : tensor<32x384x14x14xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v716 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v717 = stablehlo.multiply %v715, %v716 : tensor<32x384x14x14xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v719 = stablehlo.add %v718, %v652 : tensor<32x75264xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v721 = stablehlo.convolution(%v720, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v722 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v723 = stablehlo.add %v721, %v722 : tensor<32x384x14x14xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v726 = stablehlo.transpose %v725, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v730 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v731 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v732 = stablehlo.reduce(%v728 init: %v729) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v733 = stablehlo.broadcast_in_dim %v732, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v734 = stablehlo.divide %v733, %v730 : tensor<32x196x384xf32>
    %v735 = stablehlo.subtract %v728, %v734 : tensor<32x196x384xf32>
    %v736 = stablehlo.multiply %v735, %v735 : tensor<32x196x384xf32>
    %v737 = stablehlo.reduce(%v736 init: %v729) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v738 = stablehlo.broadcast_in_dim %v737, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v739 = stablehlo.divide %v738, %v730 : tensor<32x196x384xf32>
    %v740 = stablehlo.add %v739, %v731 : tensor<32x196x384xf32>
    %v741 = stablehlo.rsqrt %v740 : tensor<32x196x384xf32>
    %v742 = stablehlo.multiply %v735, %v741 : tensor<32x196x384xf32>
    %v743 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v744 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v745 = stablehlo.multiply %v742, %v743 : tensor<32x196x384xf32>
    %v746 = stablehlo.add %v745, %v744 : tensor<32x196x384xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v749 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v750 = stablehlo.multiply %v748, %v749 : tensor<32x196x384xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v753 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v754 = stablehlo.add %v752, %v753 : tensor<32x196x384xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v757 = stablehlo.transpose %v756, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v758 = stablehlo.reshape %v757 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v759 = stablehlo.reshape %v758 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v760 = stablehlo.convolution(%v759, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v762 = stablehlo.add %v760, %v761 : tensor<32x1536x14x14xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v764 = stablehlo.multiply %v763, %v763 : tensor<32x301056xf32>
    %v765 = stablehlo.multiply %v764, %v763 : tensor<32x301056xf32>
    %v766 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v767 = stablehlo.multiply %v766, %v765 : tensor<32x301056xf32>
    %v768 = stablehlo.add %v763, %v767 : tensor<32x301056xf32>
    %v769 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v770 = stablehlo.multiply %v769, %v768 : tensor<32x301056xf32>
    %v771 = stablehlo.tanh %v770 : tensor<32x301056xf32>
    %v772 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v773 = stablehlo.add %v772, %v771 : tensor<32x301056xf32>
    %v774 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v775 = stablehlo.multiply %v774, %v763 : tensor<32x301056xf32>
    %v776 = stablehlo.multiply %v775, %v773 : tensor<32x301056xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v778 = stablehlo.convolution(%v777, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v779 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v780 = stablehlo.add %v778, %v779 : tensor<32x384x14x14xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v783 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v784 = stablehlo.multiply %v782, %v783 : tensor<32x384x14x14xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v786 = stablehlo.add %v785, %v719 : tensor<32x75264xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v788 = stablehlo.convolution(%v787, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v789 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v790 = stablehlo.add %v788, %v789 : tensor<32x384x14x14xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v793 = stablehlo.transpose %v792, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v797 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v798 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v799 = stablehlo.reduce(%v795 init: %v796) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v800 = stablehlo.broadcast_in_dim %v799, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v801 = stablehlo.divide %v800, %v797 : tensor<32x196x384xf32>
    %v802 = stablehlo.subtract %v795, %v801 : tensor<32x196x384xf32>
    %v803 = stablehlo.multiply %v802, %v802 : tensor<32x196x384xf32>
    %v804 = stablehlo.reduce(%v803 init: %v796) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v805 = stablehlo.broadcast_in_dim %v804, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v806 = stablehlo.divide %v805, %v797 : tensor<32x196x384xf32>
    %v807 = stablehlo.add %v806, %v798 : tensor<32x196x384xf32>
    %v808 = stablehlo.rsqrt %v807 : tensor<32x196x384xf32>
    %v809 = stablehlo.multiply %v802, %v808 : tensor<32x196x384xf32>
    %v810 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v811 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v812 = stablehlo.multiply %v809, %v810 : tensor<32x196x384xf32>
    %v813 = stablehlo.add %v812, %v811 : tensor<32x196x384xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v816 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v817 = stablehlo.multiply %v815, %v816 : tensor<32x196x384xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v820 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v821 = stablehlo.add %v819, %v820 : tensor<32x196x384xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v824 = stablehlo.transpose %v823, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v827 = stablehlo.convolution(%v826, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v828 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v829 = stablehlo.add %v827, %v828 : tensor<32x1536x14x14xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v831 = stablehlo.multiply %v830, %v830 : tensor<32x301056xf32>
    %v832 = stablehlo.multiply %v831, %v830 : tensor<32x301056xf32>
    %v833 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v834 = stablehlo.multiply %v833, %v832 : tensor<32x301056xf32>
    %v835 = stablehlo.add %v830, %v834 : tensor<32x301056xf32>
    %v836 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v837 = stablehlo.multiply %v836, %v835 : tensor<32x301056xf32>
    %v838 = stablehlo.tanh %v837 : tensor<32x301056xf32>
    %v839 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v840 = stablehlo.add %v839, %v838 : tensor<32x301056xf32>
    %v841 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v842 = stablehlo.multiply %v841, %v830 : tensor<32x301056xf32>
    %v843 = stablehlo.multiply %v842, %v840 : tensor<32x301056xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v845 = stablehlo.convolution(%v844, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v846 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<32x384x14x14xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v850 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v851 = stablehlo.multiply %v849, %v850 : tensor<32x384x14x14xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v853 = stablehlo.add %v852, %v786 : tensor<32x75264xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v855 = stablehlo.convolution(%v854, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v856 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v857 = stablehlo.add %v855, %v856 : tensor<32x384x14x14xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v860 = stablehlo.transpose %v859, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v864 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v865 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v866 = stablehlo.reduce(%v862 init: %v863) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v867 = stablehlo.broadcast_in_dim %v866, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v868 = stablehlo.divide %v867, %v864 : tensor<32x196x384xf32>
    %v869 = stablehlo.subtract %v862, %v868 : tensor<32x196x384xf32>
    %v870 = stablehlo.multiply %v869, %v869 : tensor<32x196x384xf32>
    %v871 = stablehlo.reduce(%v870 init: %v863) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v872 = stablehlo.broadcast_in_dim %v871, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v873 = stablehlo.divide %v872, %v864 : tensor<32x196x384xf32>
    %v874 = stablehlo.add %v873, %v865 : tensor<32x196x384xf32>
    %v875 = stablehlo.rsqrt %v874 : tensor<32x196x384xf32>
    %v876 = stablehlo.multiply %v869, %v875 : tensor<32x196x384xf32>
    %v877 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v878 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v879 = stablehlo.multiply %v876, %v877 : tensor<32x196x384xf32>
    %v880 = stablehlo.add %v879, %v878 : tensor<32x196x384xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v883 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v884 = stablehlo.multiply %v882, %v883 : tensor<32x196x384xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v887 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v888 = stablehlo.add %v886, %v887 : tensor<32x196x384xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v891 = stablehlo.transpose %v890, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v892 = stablehlo.reshape %v891 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v894 = stablehlo.convolution(%v893, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v895 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v896 = stablehlo.add %v894, %v895 : tensor<32x1536x14x14xf32>
    %v897 = stablehlo.reshape %v896 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v898 = stablehlo.multiply %v897, %v897 : tensor<32x301056xf32>
    %v899 = stablehlo.multiply %v898, %v897 : tensor<32x301056xf32>
    %v900 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v901 = stablehlo.multiply %v900, %v899 : tensor<32x301056xf32>
    %v902 = stablehlo.add %v897, %v901 : tensor<32x301056xf32>
    %v903 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v904 = stablehlo.multiply %v903, %v902 : tensor<32x301056xf32>
    %v905 = stablehlo.tanh %v904 : tensor<32x301056xf32>
    %v906 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v907 = stablehlo.add %v906, %v905 : tensor<32x301056xf32>
    %v908 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v909 = stablehlo.multiply %v908, %v897 : tensor<32x301056xf32>
    %v910 = stablehlo.multiply %v909, %v907 : tensor<32x301056xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v912 = stablehlo.convolution(%v911, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v913 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v914 = stablehlo.add %v912, %v913 : tensor<32x384x14x14xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v917 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v918 = stablehlo.multiply %v916, %v917 : tensor<32x384x14x14xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v920 = stablehlo.add %v919, %v853 : tensor<32x75264xf32>
    %v921 = stablehlo.reshape %v920 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v922 = stablehlo.convolution(%v921, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v923 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v924 = stablehlo.add %v922, %v923 : tensor<32x384x14x14xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v927 = stablehlo.transpose %v926, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v928 = stablehlo.reshape %v927 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v931 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v932 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v933 = stablehlo.reduce(%v929 init: %v930) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v934 = stablehlo.broadcast_in_dim %v933, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v935 = stablehlo.divide %v934, %v931 : tensor<32x196x384xf32>
    %v936 = stablehlo.subtract %v929, %v935 : tensor<32x196x384xf32>
    %v937 = stablehlo.multiply %v936, %v936 : tensor<32x196x384xf32>
    %v938 = stablehlo.reduce(%v937 init: %v930) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v939 = stablehlo.broadcast_in_dim %v938, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v940 = stablehlo.divide %v939, %v931 : tensor<32x196x384xf32>
    %v941 = stablehlo.add %v940, %v932 : tensor<32x196x384xf32>
    %v942 = stablehlo.rsqrt %v941 : tensor<32x196x384xf32>
    %v943 = stablehlo.multiply %v936, %v942 : tensor<32x196x384xf32>
    %v944 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v945 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v946 = stablehlo.multiply %v943, %v944 : tensor<32x196x384xf32>
    %v947 = stablehlo.add %v946, %v945 : tensor<32x196x384xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v950 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v951 = stablehlo.multiply %v949, %v950 : tensor<32x196x384xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v954 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v955 = stablehlo.add %v953, %v954 : tensor<32x196x384xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v958 = stablehlo.transpose %v957, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v961 = stablehlo.convolution(%v960, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v962 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v963 = stablehlo.add %v961, %v962 : tensor<32x1536x14x14xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v965 = stablehlo.multiply %v964, %v964 : tensor<32x301056xf32>
    %v966 = stablehlo.multiply %v965, %v964 : tensor<32x301056xf32>
    %v967 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v968 = stablehlo.multiply %v967, %v966 : tensor<32x301056xf32>
    %v969 = stablehlo.add %v964, %v968 : tensor<32x301056xf32>
    %v970 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v971 = stablehlo.multiply %v970, %v969 : tensor<32x301056xf32>
    %v972 = stablehlo.tanh %v971 : tensor<32x301056xf32>
    %v973 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v974 = stablehlo.add %v973, %v972 : tensor<32x301056xf32>
    %v975 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v976 = stablehlo.multiply %v975, %v964 : tensor<32x301056xf32>
    %v977 = stablehlo.multiply %v976, %v974 : tensor<32x301056xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v979 = stablehlo.convolution(%v978, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v980 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v981 = stablehlo.add %v979, %v980 : tensor<32x384x14x14xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v984 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v985 = stablehlo.multiply %v983, %v984 : tensor<32x384x14x14xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v987 = stablehlo.add %v986, %v920 : tensor<32x75264xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v989 = stablehlo.convolution(%v988, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v990 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v991 = stablehlo.add %v989, %v990 : tensor<32x384x14x14xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v994 = stablehlo.transpose %v993, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v998 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v999 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1000 = stablehlo.reduce(%v996 init: %v997) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1001 = stablehlo.broadcast_in_dim %v1000, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1002 = stablehlo.divide %v1001, %v998 : tensor<32x196x384xf32>
    %v1003 = stablehlo.subtract %v996, %v1002 : tensor<32x196x384xf32>
    %v1004 = stablehlo.multiply %v1003, %v1003 : tensor<32x196x384xf32>
    %v1005 = stablehlo.reduce(%v1004 init: %v997) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1006 = stablehlo.broadcast_in_dim %v1005, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1007 = stablehlo.divide %v1006, %v998 : tensor<32x196x384xf32>
    %v1008 = stablehlo.add %v1007, %v999 : tensor<32x196x384xf32>
    %v1009 = stablehlo.rsqrt %v1008 : tensor<32x196x384xf32>
    %v1010 = stablehlo.multiply %v1003, %v1009 : tensor<32x196x384xf32>
    %v1011 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1012 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1013 = stablehlo.multiply %v1010, %v1011 : tensor<32x196x384xf32>
    %v1014 = stablehlo.add %v1013, %v1012 : tensor<32x196x384xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1017 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1018 = stablehlo.multiply %v1016, %v1017 : tensor<32x196x384xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1021 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1022 = stablehlo.add %v1020, %v1021 : tensor<32x196x384xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1025 = stablehlo.transpose %v1024, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1026 = stablehlo.reshape %v1025 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1028 = stablehlo.convolution(%v1027, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1029 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1030 = stablehlo.add %v1028, %v1029 : tensor<32x1536x14x14xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1032 = stablehlo.multiply %v1031, %v1031 : tensor<32x301056xf32>
    %v1033 = stablehlo.multiply %v1032, %v1031 : tensor<32x301056xf32>
    %v1034 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1035 = stablehlo.multiply %v1034, %v1033 : tensor<32x301056xf32>
    %v1036 = stablehlo.add %v1031, %v1035 : tensor<32x301056xf32>
    %v1037 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1038 = stablehlo.multiply %v1037, %v1036 : tensor<32x301056xf32>
    %v1039 = stablehlo.tanh %v1038 : tensor<32x301056xf32>
    %v1040 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1041 = stablehlo.add %v1040, %v1039 : tensor<32x301056xf32>
    %v1042 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1043 = stablehlo.multiply %v1042, %v1031 : tensor<32x301056xf32>
    %v1044 = stablehlo.multiply %v1043, %v1041 : tensor<32x301056xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1046 = stablehlo.convolution(%v1045, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1047 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1048 = stablehlo.add %v1046, %v1047 : tensor<32x384x14x14xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1051 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1052 = stablehlo.multiply %v1050, %v1051 : tensor<32x384x14x14xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1054 = stablehlo.add %v1053, %v987 : tensor<32x75264xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1056 = stablehlo.convolution(%v1055, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1057 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1058 = stablehlo.add %v1056, %v1057 : tensor<32x384x14x14xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1061 = stablehlo.transpose %v1060, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1065 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1066 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1067 = stablehlo.reduce(%v1063 init: %v1064) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1068 = stablehlo.broadcast_in_dim %v1067, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1069 = stablehlo.divide %v1068, %v1065 : tensor<32x196x384xf32>
    %v1070 = stablehlo.subtract %v1063, %v1069 : tensor<32x196x384xf32>
    %v1071 = stablehlo.multiply %v1070, %v1070 : tensor<32x196x384xf32>
    %v1072 = stablehlo.reduce(%v1071 init: %v1064) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1073 = stablehlo.broadcast_in_dim %v1072, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1074 = stablehlo.divide %v1073, %v1065 : tensor<32x196x384xf32>
    %v1075 = stablehlo.add %v1074, %v1066 : tensor<32x196x384xf32>
    %v1076 = stablehlo.rsqrt %v1075 : tensor<32x196x384xf32>
    %v1077 = stablehlo.multiply %v1070, %v1076 : tensor<32x196x384xf32>
    %v1078 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1079 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1080 = stablehlo.multiply %v1077, %v1078 : tensor<32x196x384xf32>
    %v1081 = stablehlo.add %v1080, %v1079 : tensor<32x196x384xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1084 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1085 = stablehlo.multiply %v1083, %v1084 : tensor<32x196x384xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1088 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1089 = stablehlo.add %v1087, %v1088 : tensor<32x196x384xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1092 = stablehlo.transpose %v1091, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1095 = stablehlo.convolution(%v1094, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1096 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1097 = stablehlo.add %v1095, %v1096 : tensor<32x1536x14x14xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1099 = stablehlo.multiply %v1098, %v1098 : tensor<32x301056xf32>
    %v1100 = stablehlo.multiply %v1099, %v1098 : tensor<32x301056xf32>
    %v1101 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1102 = stablehlo.multiply %v1101, %v1100 : tensor<32x301056xf32>
    %v1103 = stablehlo.add %v1098, %v1102 : tensor<32x301056xf32>
    %v1104 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1105 = stablehlo.multiply %v1104, %v1103 : tensor<32x301056xf32>
    %v1106 = stablehlo.tanh %v1105 : tensor<32x301056xf32>
    %v1107 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1108 = stablehlo.add %v1107, %v1106 : tensor<32x301056xf32>
    %v1109 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1110 = stablehlo.multiply %v1109, %v1098 : tensor<32x301056xf32>
    %v1111 = stablehlo.multiply %v1110, %v1108 : tensor<32x301056xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1113 = stablehlo.convolution(%v1112, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1114 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1115 = stablehlo.add %v1113, %v1114 : tensor<32x384x14x14xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1117 = stablehlo.reshape %v1116 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1118 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1119 = stablehlo.multiply %v1117, %v1118 : tensor<32x384x14x14xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1121 = stablehlo.add %v1120, %v1054 : tensor<32x75264xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1123 = stablehlo.transpose %v1122, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1127 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1128 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1129 = stablehlo.reduce(%v1125 init: %v1126) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1130 = stablehlo.broadcast_in_dim %v1129, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1131 = stablehlo.divide %v1130, %v1127 : tensor<32x196x384xf32>
    %v1132 = stablehlo.subtract %v1125, %v1131 : tensor<32x196x384xf32>
    %v1133 = stablehlo.multiply %v1132, %v1132 : tensor<32x196x384xf32>
    %v1134 = stablehlo.reduce(%v1133 init: %v1126) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1135 = stablehlo.broadcast_in_dim %v1134, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1136 = stablehlo.divide %v1135, %v1127 : tensor<32x196x384xf32>
    %v1137 = stablehlo.add %v1136, %v1128 : tensor<32x196x384xf32>
    %v1138 = stablehlo.rsqrt %v1137 : tensor<32x196x384xf32>
    %v1139 = stablehlo.multiply %v1132, %v1138 : tensor<32x196x384xf32>
    %v1140 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1141 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1142 = stablehlo.multiply %v1139, %v1140 : tensor<32x196x384xf32>
    %v1143 = stablehlo.add %v1142, %v1141 : tensor<32x196x384xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1146 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1147 = stablehlo.multiply %v1145, %v1146 : tensor<32x196x384xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1150 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1151 = stablehlo.add %v1149, %v1150 : tensor<32x196x384xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1154 = stablehlo.transpose %v1153, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1157 = stablehlo.convolution(%v1156, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %v1158 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1159 = stablehlo.add %v1157, %v1158 : tensor<32x768x7x7xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1162 = stablehlo.convolution(%v1161, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1163 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1164 = stablehlo.add %v1162, %v1163 : tensor<32x768x7x7xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1167 = stablehlo.transpose %v1166, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1171 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1172 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1173 = stablehlo.reduce(%v1169 init: %v1170) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1174 = stablehlo.broadcast_in_dim %v1173, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1175 = stablehlo.divide %v1174, %v1171 : tensor<32x49x768xf32>
    %v1176 = stablehlo.subtract %v1169, %v1175 : tensor<32x49x768xf32>
    %v1177 = stablehlo.multiply %v1176, %v1176 : tensor<32x49x768xf32>
    %v1178 = stablehlo.reduce(%v1177 init: %v1170) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1179 = stablehlo.broadcast_in_dim %v1178, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1180 = stablehlo.divide %v1179, %v1171 : tensor<32x49x768xf32>
    %v1181 = stablehlo.add %v1180, %v1172 : tensor<32x49x768xf32>
    %v1182 = stablehlo.rsqrt %v1181 : tensor<32x49x768xf32>
    %v1183 = stablehlo.multiply %v1176, %v1182 : tensor<32x49x768xf32>
    %v1184 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1185 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1186 = stablehlo.multiply %v1183, %v1184 : tensor<32x49x768xf32>
    %v1187 = stablehlo.add %v1186, %v1185 : tensor<32x49x768xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1190 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1191 = stablehlo.multiply %v1189, %v1190 : tensor<32x49x768xf32>
    %v1192 = stablehlo.reshape %v1191 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1194 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1195 = stablehlo.add %v1193, %v1194 : tensor<32x49x768xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1197 = stablehlo.reshape %v1196 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1198 = stablehlo.transpose %v1197, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1201 = stablehlo.convolution(%v1200, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1202 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1203 = stablehlo.add %v1201, %v1202 : tensor<32x3072x7x7xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1205 = stablehlo.multiply %v1204, %v1204 : tensor<32x150528xf32>
    %v1206 = stablehlo.multiply %v1205, %v1204 : tensor<32x150528xf32>
    %v1207 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1208 = stablehlo.multiply %v1207, %v1206 : tensor<32x150528xf32>
    %v1209 = stablehlo.add %v1204, %v1208 : tensor<32x150528xf32>
    %v1210 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1211 = stablehlo.multiply %v1210, %v1209 : tensor<32x150528xf32>
    %v1212 = stablehlo.tanh %v1211 : tensor<32x150528xf32>
    %v1213 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1214 = stablehlo.add %v1213, %v1212 : tensor<32x150528xf32>
    %v1215 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1216 = stablehlo.multiply %v1215, %v1204 : tensor<32x150528xf32>
    %v1217 = stablehlo.multiply %v1216, %v1214 : tensor<32x150528xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1219 = stablehlo.convolution(%v1218, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1220 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1221 = stablehlo.add %v1219, %v1220 : tensor<32x768x7x7xf32>
    %v1222 = stablehlo.reshape %v1221 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1224 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1225 = stablehlo.multiply %v1223, %v1224 : tensor<32x768x7x7xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1227 = stablehlo.add %v1226, %v1160 : tensor<32x37632xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1229 = stablehlo.convolution(%v1228, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1230 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1231 = stablehlo.add %v1229, %v1230 : tensor<32x768x7x7xf32>
    %v1232 = stablehlo.reshape %v1231 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1234 = stablehlo.transpose %v1233, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1236 = stablehlo.reshape %v1235 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1238 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1239 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1240 = stablehlo.reduce(%v1236 init: %v1237) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1241 = stablehlo.broadcast_in_dim %v1240, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1242 = stablehlo.divide %v1241, %v1238 : tensor<32x49x768xf32>
    %v1243 = stablehlo.subtract %v1236, %v1242 : tensor<32x49x768xf32>
    %v1244 = stablehlo.multiply %v1243, %v1243 : tensor<32x49x768xf32>
    %v1245 = stablehlo.reduce(%v1244 init: %v1237) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1246 = stablehlo.broadcast_in_dim %v1245, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1247 = stablehlo.divide %v1246, %v1238 : tensor<32x49x768xf32>
    %v1248 = stablehlo.add %v1247, %v1239 : tensor<32x49x768xf32>
    %v1249 = stablehlo.rsqrt %v1248 : tensor<32x49x768xf32>
    %v1250 = stablehlo.multiply %v1243, %v1249 : tensor<32x49x768xf32>
    %v1251 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1252 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1253 = stablehlo.multiply %v1250, %v1251 : tensor<32x49x768xf32>
    %v1254 = stablehlo.add %v1253, %v1252 : tensor<32x49x768xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1257 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1258 = stablehlo.multiply %v1256, %v1257 : tensor<32x49x768xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1261 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1262 = stablehlo.add %v1260, %v1261 : tensor<32x49x768xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1265 = stablehlo.transpose %v1264, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1266 = stablehlo.reshape %v1265 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1268 = stablehlo.convolution(%v1267, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1269 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1270 = stablehlo.add %v1268, %v1269 : tensor<32x3072x7x7xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1272 = stablehlo.multiply %v1271, %v1271 : tensor<32x150528xf32>
    %v1273 = stablehlo.multiply %v1272, %v1271 : tensor<32x150528xf32>
    %v1274 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1275 = stablehlo.multiply %v1274, %v1273 : tensor<32x150528xf32>
    %v1276 = stablehlo.add %v1271, %v1275 : tensor<32x150528xf32>
    %v1277 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1278 = stablehlo.multiply %v1277, %v1276 : tensor<32x150528xf32>
    %v1279 = stablehlo.tanh %v1278 : tensor<32x150528xf32>
    %v1280 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1281 = stablehlo.add %v1280, %v1279 : tensor<32x150528xf32>
    %v1282 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1283 = stablehlo.multiply %v1282, %v1271 : tensor<32x150528xf32>
    %v1284 = stablehlo.multiply %v1283, %v1281 : tensor<32x150528xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1286 = stablehlo.convolution(%v1285, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1287 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1288 = stablehlo.add %v1286, %v1287 : tensor<32x768x7x7xf32>
    %v1289 = stablehlo.reshape %v1288 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1291 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1292 = stablehlo.multiply %v1290, %v1291 : tensor<32x768x7x7xf32>
    %v1293 = stablehlo.reshape %v1292 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1294 = stablehlo.add %v1293, %v1227 : tensor<32x37632xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1296 = stablehlo.convolution(%v1295, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1297 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1298 = stablehlo.add %v1296, %v1297 : tensor<32x768x7x7xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1301 = stablehlo.transpose %v1300, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1305 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1306 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1307 = stablehlo.reduce(%v1303 init: %v1304) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1308 = stablehlo.broadcast_in_dim %v1307, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1309 = stablehlo.divide %v1308, %v1305 : tensor<32x49x768xf32>
    %v1310 = stablehlo.subtract %v1303, %v1309 : tensor<32x49x768xf32>
    %v1311 = stablehlo.multiply %v1310, %v1310 : tensor<32x49x768xf32>
    %v1312 = stablehlo.reduce(%v1311 init: %v1304) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1313 = stablehlo.broadcast_in_dim %v1312, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1314 = stablehlo.divide %v1313, %v1305 : tensor<32x49x768xf32>
    %v1315 = stablehlo.add %v1314, %v1306 : tensor<32x49x768xf32>
    %v1316 = stablehlo.rsqrt %v1315 : tensor<32x49x768xf32>
    %v1317 = stablehlo.multiply %v1310, %v1316 : tensor<32x49x768xf32>
    %v1318 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1319 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1320 = stablehlo.multiply %v1317, %v1318 : tensor<32x49x768xf32>
    %v1321 = stablehlo.add %v1320, %v1319 : tensor<32x49x768xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1324 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1325 = stablehlo.multiply %v1323, %v1324 : tensor<32x49x768xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1328 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1329 = stablehlo.add %v1327, %v1328 : tensor<32x49x768xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1332 = stablehlo.transpose %v1331, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1334 = stablehlo.reshape %v1333 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1335 = stablehlo.convolution(%v1334, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1336 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1337 = stablehlo.add %v1335, %v1336 : tensor<32x3072x7x7xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1339 = stablehlo.multiply %v1338, %v1338 : tensor<32x150528xf32>
    %v1340 = stablehlo.multiply %v1339, %v1338 : tensor<32x150528xf32>
    %v1341 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1342 = stablehlo.multiply %v1341, %v1340 : tensor<32x150528xf32>
    %v1343 = stablehlo.add %v1338, %v1342 : tensor<32x150528xf32>
    %v1344 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1345 = stablehlo.multiply %v1344, %v1343 : tensor<32x150528xf32>
    %v1346 = stablehlo.tanh %v1345 : tensor<32x150528xf32>
    %v1347 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1348 = stablehlo.add %v1347, %v1346 : tensor<32x150528xf32>
    %v1349 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1350 = stablehlo.multiply %v1349, %v1338 : tensor<32x150528xf32>
    %v1351 = stablehlo.multiply %v1350, %v1348 : tensor<32x150528xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1353 = stablehlo.convolution(%v1352, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1354 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1355 = stablehlo.add %v1353, %v1354 : tensor<32x768x7x7xf32>
    %v1356 = stablehlo.reshape %v1355 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1358 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1359 = stablehlo.multiply %v1357, %v1358 : tensor<32x768x7x7xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1361 = stablehlo.add %v1360, %v1294 : tensor<32x37632xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1364 = stablehlo.reduce(%v1362 init: %v1363) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %v1365 = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %v1366 = stablehlo.divide %v1364, %v1365 : tensor<32x768xf32>
    %v1367 = stablehlo.dot_general %v1366, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x10xf32>) -> tensor<32x10xf32>
    %v1368 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1369 = stablehlo.add %v1367, %v1368 : tensor<32x10xf32>
    %v1370 = stablehlo.exponential %v1369 : tensor<32x10xf32>
    %v1371 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1372 = stablehlo.reduce(%v1370 init: %v1371) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v1373 = stablehlo.broadcast_in_dim %v1372, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v1374 = stablehlo.divide %v1370, %v1373 : tensor<32x10xf32>
    %v1375 = stablehlo.subtract %v1374, %onehot : tensor<32x10xf32>
    %dy = stablehlo.divide %v1375, %bsc : tensor<32x10xf32>
    %v1376 = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<768x10xf32>) -> tensor<32x768xf32>
    %v1377 = stablehlo.dot_general %v1366, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<32x10xf32>) -> tensor<768x10xf32>
    %v1378 = stablehlo.constant dense<0.1> : tensor<768x10xf32>
    %v1379 = stablehlo.multiply %v1377, %v1378 : tensor<768x10xf32>
    %v1380 = stablehlo.subtract %Wd, %v1379 : tensor<768x10xf32>
    %v1381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1382 = stablehlo.reduce(%dy init: %v1381) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1383 = stablehlo.constant dense<0.1> : tensor<10xf32>
    %v1384 = stablehlo.multiply %v1382, %v1383 : tensor<10xf32>
    %v1385 = stablehlo.subtract %bd, %v1384 : tensor<10xf32>
    %dgi = stablehlo.reshape %v1376 : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : (tensor<32x768x1x1xf32>) -> tensor<32x768x7x7xf32>
    %dgn = stablehlo.constant dense<49.0> : tensor<32x768x7x7xf32>
    %dgd = stablehlo.divide %dgb, %dgn : tensor<32x768x7x7xf32>
    %dgapf = stablehlo.reshape %dgd : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1386 = stablehlo.reshape %dgapf : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1387 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1388 = stablehlo.multiply %v1386, %v1387 : tensor<32x768x7x7xf32>
    %v1389 = stablehlo.reshape %v1388 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1391 = stablehlo.transpose %s3b2pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1392 = stablehlo.reverse %v1391, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1393 = stablehlo.convolution(%v1390, %v1392)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1394 = stablehlo.reshape %v1393 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1395 = stablehlo.multiply %v1338, %v1338 : tensor<32x150528xf32>
    %v1396 = stablehlo.multiply %v1395, %v1338 : tensor<32x150528xf32>
    %v1397 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1398 = stablehlo.multiply %v1397, %v1396 : tensor<32x150528xf32>
    %v1399 = stablehlo.add %v1338, %v1398 : tensor<32x150528xf32>
    %v1400 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1401 = stablehlo.multiply %v1400, %v1399 : tensor<32x150528xf32>
    %v1402 = stablehlo.tanh %v1401 : tensor<32x150528xf32>
    %v1403 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1404 = stablehlo.add %v1403, %v1402 : tensor<32x150528xf32>
    %v1405 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1406 = stablehlo.multiply %v1405, %v1404 : tensor<32x150528xf32>
    %v1407 = stablehlo.multiply %v1402, %v1402 : tensor<32x150528xf32>
    %v1408 = stablehlo.subtract %v1403, %v1407 : tensor<32x150528xf32>
    %v1409 = stablehlo.multiply %v1405, %v1338 : tensor<32x150528xf32>
    %v1410 = stablehlo.multiply %v1409, %v1408 : tensor<32x150528xf32>
    %v1411 = stablehlo.constant dense<0.134145> : tensor<32x150528xf32>
    %v1412 = stablehlo.multiply %v1411, %v1395 : tensor<32x150528xf32>
    %v1413 = stablehlo.add %v1403, %v1412 : tensor<32x150528xf32>
    %v1414 = stablehlo.multiply %v1400, %v1413 : tensor<32x150528xf32>
    %v1415 = stablehlo.multiply %v1410, %v1414 : tensor<32x150528xf32>
    %v1416 = stablehlo.add %v1406, %v1415 : tensor<32x150528xf32>
    %v1417 = stablehlo.multiply %v1394, %v1416 : tensor<32x150528xf32>
    %v1418 = stablehlo.reshape %v1417 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1419 = stablehlo.transpose %s3b2eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1420 = stablehlo.reverse %v1419, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1421 = stablehlo.convolution(%v1418, %v1420)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1422 = stablehlo.reshape %v1421 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1423 = stablehlo.reshape %v1299 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1424 = stablehlo.transpose %v1423, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1425 = stablehlo.reshape %v1424 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1426 = stablehlo.reshape %v1422 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1427 = stablehlo.transpose %v1426, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1430 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1431 = stablehlo.multiply %v1429, %v1430 : tensor<32x49x768xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1434 = stablehlo.reshape %v1425 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1435 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1436 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1437 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1438 = stablehlo.reduce(%v1434 init: %v1435) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1439 = stablehlo.broadcast_in_dim %v1438, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1440 = stablehlo.divide %v1439, %v1436 : tensor<32x49x768xf32>
    %v1441 = stablehlo.subtract %v1434, %v1440 : tensor<32x49x768xf32>
    %v1442 = stablehlo.multiply %v1441, %v1441 : tensor<32x49x768xf32>
    %v1443 = stablehlo.reduce(%v1442 init: %v1435) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1444 = stablehlo.broadcast_in_dim %v1443, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1445 = stablehlo.divide %v1444, %v1436 : tensor<32x49x768xf32>
    %v1446 = stablehlo.add %v1445, %v1437 : tensor<32x49x768xf32>
    %v1447 = stablehlo.rsqrt %v1446 : tensor<32x49x768xf32>
    %v1448 = stablehlo.multiply %v1441, %v1447 : tensor<32x49x768xf32>
    %v1449 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1450 = stablehlo.multiply %v1449, %v1433 : tensor<32x49x768xf32>
    %v1451 = stablehlo.reduce(%v1450 init: %v1435) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1452 = stablehlo.broadcast_in_dim %v1451, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1453 = stablehlo.multiply %v1448, %v1450 : tensor<32x49x768xf32>
    %v1454 = stablehlo.reduce(%v1453 init: %v1435) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1455 = stablehlo.broadcast_in_dim %v1454, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1456 = stablehlo.multiply %v1450, %v1436 : tensor<32x49x768xf32>
    %v1457 = stablehlo.subtract %v1456, %v1452 : tensor<32x49x768xf32>
    %v1458 = stablehlo.multiply %v1448, %v1455 : tensor<32x49x768xf32>
    %v1459 = stablehlo.subtract %v1457, %v1458 : tensor<32x49x768xf32>
    %v1460 = stablehlo.divide %v1447, %v1436 : tensor<32x49x768xf32>
    %v1461 = stablehlo.multiply %v1460, %v1459 : tensor<32x49x768xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1464 = stablehlo.transpose %v1463, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1465 = stablehlo.reshape %v1464 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1466 = stablehlo.reshape %v1465 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1467 = stablehlo.reverse %s3b2dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1468 = stablehlo.convolution(%v1466, %v1467)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1469 = stablehlo.reshape %v1468 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1470 = stablehlo.add %v1469, %dgapf : tensor<32x37632xf32>
    %v1471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1472 = stablehlo.reshape %v1356 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1473 = stablehlo.reshape %dgapf : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1474 = stablehlo.multiply %v1472, %v1473 : tensor<32x768x7x7xf32>
    %v1475 = stablehlo.reduce(%v1474 init: %v1471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1476 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1477 = stablehlo.multiply %v1475, %v1476 : tensor<768xf32>
    %v1478 = stablehlo.subtract %s3b2lg, %v1477 : tensor<768xf32>
    %v1479 = stablehlo.reshape %v1351 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1480 = stablehlo.reshape %v1389 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1481 = stablehlo.transpose %v1479, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1482 = stablehlo.transpose %v1480, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1483 = stablehlo.convolution(%v1481, %v1482)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1484 = stablehlo.transpose %v1483, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1485 = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %v1486 = stablehlo.multiply %v1484, %v1485 : tensor<768x3072x1x1xf32>
    %v1487 = stablehlo.subtract %s3b2pW, %v1486 : tensor<768x3072x1x1xf32>
    %v1488 = stablehlo.reshape %v1389 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1490 = stablehlo.reduce(%v1488 init: %v1489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1491 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1492 = stablehlo.multiply %v1490, %v1491 : tensor<768xf32>
    %v1493 = stablehlo.subtract %s3b2pb, %v1492 : tensor<768xf32>
    %v1494 = stablehlo.reshape %v1333 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1495 = stablehlo.reshape %v1417 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1496 = stablehlo.transpose %v1494, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1497 = stablehlo.transpose %v1495, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1498 = stablehlo.convolution(%v1496, %v1497)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1499 = stablehlo.transpose %v1498, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1500 = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %v1501 = stablehlo.multiply %v1499, %v1500 : tensor<3072x768x1x1xf32>
    %v1502 = stablehlo.subtract %s3b2eW, %v1501 : tensor<3072x768x1x1xf32>
    %v1503 = stablehlo.reshape %v1417 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1505 = stablehlo.reduce(%v1503 init: %v1504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1506 = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %v1507 = stablehlo.multiply %v1505, %v1506 : tensor<3072xf32>
    %v1508 = stablehlo.subtract %s3b2eb, %v1507 : tensor<3072xf32>
    %v1509 = stablehlo.reshape %v1299 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1510 = stablehlo.transpose %v1509, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1512 = stablehlo.reshape %v1422 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1513 = stablehlo.transpose %v1512, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1514 = stablehlo.reshape %v1513 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1515 = stablehlo.reshape %v1511 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1516 = stablehlo.reshape %v1514 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1518 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1519 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1520 = stablehlo.reduce(%v1515 init: %v1517) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1521 = stablehlo.broadcast_in_dim %v1520, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1522 = stablehlo.divide %v1521, %v1518 : tensor<32x49x768xf32>
    %v1523 = stablehlo.subtract %v1515, %v1522 : tensor<32x49x768xf32>
    %v1524 = stablehlo.multiply %v1523, %v1523 : tensor<32x49x768xf32>
    %v1525 = stablehlo.reduce(%v1524 init: %v1517) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1526 = stablehlo.broadcast_in_dim %v1525, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1527 = stablehlo.divide %v1526, %v1518 : tensor<32x49x768xf32>
    %v1528 = stablehlo.add %v1527, %v1519 : tensor<32x49x768xf32>
    %v1529 = stablehlo.rsqrt %v1528 : tensor<32x49x768xf32>
    %v1530 = stablehlo.multiply %v1523, %v1529 : tensor<32x49x768xf32>
    %v1531 = stablehlo.multiply %v1516, %v1530 : tensor<32x49x768xf32>
    %v1532 = stablehlo.reduce(%v1531 init: %v1517) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1533 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1534 = stablehlo.multiply %v1532, %v1533 : tensor<768xf32>
    %v1535 = stablehlo.subtract %s3b2ng, %v1534 : tensor<768xf32>
    %v1536 = stablehlo.reshape %v1422 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1537 = stablehlo.transpose %v1536, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1538 = stablehlo.reshape %v1537 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1540 = stablehlo.reshape %v1538 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1541 = stablehlo.reduce(%v1540 init: %v1539) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1542 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1543 = stablehlo.multiply %v1541, %v1542 : tensor<768xf32>
    %v1544 = stablehlo.subtract %s3b2nbt, %v1543 : tensor<768xf32>
    %v1545 = stablehlo.reshape %v1294 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1546 = stablehlo.reshape %v1465 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1547 = stablehlo.transpose %v1545, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1548 = stablehlo.transpose %v1546, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1549 = stablehlo.convolution(%v1547, %v1548)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1551 = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %v1552 = stablehlo.multiply %v1550, %v1551 : tensor<768x1x7x7xf32>
    %v1553 = stablehlo.subtract %s3b2dW, %v1552 : tensor<768x1x7x7xf32>
    %v1554 = stablehlo.reshape %v1465 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1555 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1556 = stablehlo.reduce(%v1554 init: %v1555) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1557 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1558 = stablehlo.multiply %v1556, %v1557 : tensor<768xf32>
    %v1559 = stablehlo.subtract %s3b2db, %v1558 : tensor<768xf32>
    %v1560 = stablehlo.reshape %v1470 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1561 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1562 = stablehlo.multiply %v1560, %v1561 : tensor<32x768x7x7xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1564 = stablehlo.reshape %v1563 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1565 = stablehlo.transpose %s3b1pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1566 = stablehlo.reverse %v1565, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1567 = stablehlo.convolution(%v1564, %v1566)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1569 = stablehlo.multiply %v1271, %v1271 : tensor<32x150528xf32>
    %v1570 = stablehlo.multiply %v1569, %v1271 : tensor<32x150528xf32>
    %v1571 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1572 = stablehlo.multiply %v1571, %v1570 : tensor<32x150528xf32>
    %v1573 = stablehlo.add %v1271, %v1572 : tensor<32x150528xf32>
    %v1574 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1575 = stablehlo.multiply %v1574, %v1573 : tensor<32x150528xf32>
    %v1576 = stablehlo.tanh %v1575 : tensor<32x150528xf32>
    %v1577 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1578 = stablehlo.add %v1577, %v1576 : tensor<32x150528xf32>
    %v1579 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1580 = stablehlo.multiply %v1579, %v1578 : tensor<32x150528xf32>
    %v1581 = stablehlo.multiply %v1576, %v1576 : tensor<32x150528xf32>
    %v1582 = stablehlo.subtract %v1577, %v1581 : tensor<32x150528xf32>
    %v1583 = stablehlo.multiply %v1579, %v1271 : tensor<32x150528xf32>
    %v1584 = stablehlo.multiply %v1583, %v1582 : tensor<32x150528xf32>
    %v1585 = stablehlo.constant dense<0.134145> : tensor<32x150528xf32>
    %v1586 = stablehlo.multiply %v1585, %v1569 : tensor<32x150528xf32>
    %v1587 = stablehlo.add %v1577, %v1586 : tensor<32x150528xf32>
    %v1588 = stablehlo.multiply %v1574, %v1587 : tensor<32x150528xf32>
    %v1589 = stablehlo.multiply %v1584, %v1588 : tensor<32x150528xf32>
    %v1590 = stablehlo.add %v1580, %v1589 : tensor<32x150528xf32>
    %v1591 = stablehlo.multiply %v1568, %v1590 : tensor<32x150528xf32>
    %v1592 = stablehlo.reshape %v1591 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1593 = stablehlo.transpose %s3b1eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1594 = stablehlo.reverse %v1593, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1595 = stablehlo.convolution(%v1592, %v1594)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1596 = stablehlo.reshape %v1595 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1597 = stablehlo.reshape %v1232 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1598 = stablehlo.transpose %v1597, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1599 = stablehlo.reshape %v1598 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1600 = stablehlo.reshape %v1596 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1601 = stablehlo.transpose %v1600, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1603 = stablehlo.reshape %v1602 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1604 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1605 = stablehlo.multiply %v1603, %v1604 : tensor<32x49x768xf32>
    %v1606 = stablehlo.reshape %v1605 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1608 = stablehlo.reshape %v1599 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1610 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1611 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1612 = stablehlo.reduce(%v1608 init: %v1609) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1613 = stablehlo.broadcast_in_dim %v1612, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1614 = stablehlo.divide %v1613, %v1610 : tensor<32x49x768xf32>
    %v1615 = stablehlo.subtract %v1608, %v1614 : tensor<32x49x768xf32>
    %v1616 = stablehlo.multiply %v1615, %v1615 : tensor<32x49x768xf32>
    %v1617 = stablehlo.reduce(%v1616 init: %v1609) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1618 = stablehlo.broadcast_in_dim %v1617, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1619 = stablehlo.divide %v1618, %v1610 : tensor<32x49x768xf32>
    %v1620 = stablehlo.add %v1619, %v1611 : tensor<32x49x768xf32>
    %v1621 = stablehlo.rsqrt %v1620 : tensor<32x49x768xf32>
    %v1622 = stablehlo.multiply %v1615, %v1621 : tensor<32x49x768xf32>
    %v1623 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1624 = stablehlo.multiply %v1623, %v1607 : tensor<32x49x768xf32>
    %v1625 = stablehlo.reduce(%v1624 init: %v1609) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1626 = stablehlo.broadcast_in_dim %v1625, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1627 = stablehlo.multiply %v1622, %v1624 : tensor<32x49x768xf32>
    %v1628 = stablehlo.reduce(%v1627 init: %v1609) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1629 = stablehlo.broadcast_in_dim %v1628, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1630 = stablehlo.multiply %v1624, %v1610 : tensor<32x49x768xf32>
    %v1631 = stablehlo.subtract %v1630, %v1626 : tensor<32x49x768xf32>
    %v1632 = stablehlo.multiply %v1622, %v1629 : tensor<32x49x768xf32>
    %v1633 = stablehlo.subtract %v1631, %v1632 : tensor<32x49x768xf32>
    %v1634 = stablehlo.divide %v1621, %v1610 : tensor<32x49x768xf32>
    %v1635 = stablehlo.multiply %v1634, %v1633 : tensor<32x49x768xf32>
    %v1636 = stablehlo.reshape %v1635 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1637 = stablehlo.reshape %v1636 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1638 = stablehlo.transpose %v1637, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1640 = stablehlo.reshape %v1639 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1641 = stablehlo.reverse %s3b1dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1642 = stablehlo.convolution(%v1640, %v1641)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1644 = stablehlo.add %v1643, %v1470 : tensor<32x37632xf32>
    %v1645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1646 = stablehlo.reshape %v1289 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1647 = stablehlo.reshape %v1470 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1648 = stablehlo.multiply %v1646, %v1647 : tensor<32x768x7x7xf32>
    %v1649 = stablehlo.reduce(%v1648 init: %v1645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1650 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1651 = stablehlo.multiply %v1649, %v1650 : tensor<768xf32>
    %v1652 = stablehlo.subtract %s3b1lg, %v1651 : tensor<768xf32>
    %v1653 = stablehlo.reshape %v1284 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1654 = stablehlo.reshape %v1563 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1655 = stablehlo.transpose %v1653, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1656 = stablehlo.transpose %v1654, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1657 = stablehlo.convolution(%v1655, %v1656)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1658 = stablehlo.transpose %v1657, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1659 = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %v1660 = stablehlo.multiply %v1658, %v1659 : tensor<768x3072x1x1xf32>
    %v1661 = stablehlo.subtract %s3b1pW, %v1660 : tensor<768x3072x1x1xf32>
    %v1662 = stablehlo.reshape %v1563 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1663 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1664 = stablehlo.reduce(%v1662 init: %v1663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1665 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1666 = stablehlo.multiply %v1664, %v1665 : tensor<768xf32>
    %v1667 = stablehlo.subtract %s3b1pb, %v1666 : tensor<768xf32>
    %v1668 = stablehlo.reshape %v1266 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1669 = stablehlo.reshape %v1591 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1670 = stablehlo.transpose %v1668, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1671 = stablehlo.transpose %v1669, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1672 = stablehlo.convolution(%v1670, %v1671)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1673 = stablehlo.transpose %v1672, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1674 = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %v1675 = stablehlo.multiply %v1673, %v1674 : tensor<3072x768x1x1xf32>
    %v1676 = stablehlo.subtract %s3b1eW, %v1675 : tensor<3072x768x1x1xf32>
    %v1677 = stablehlo.reshape %v1591 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1679 = stablehlo.reduce(%v1677 init: %v1678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1680 = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %v1681 = stablehlo.multiply %v1679, %v1680 : tensor<3072xf32>
    %v1682 = stablehlo.subtract %s3b1eb, %v1681 : tensor<3072xf32>
    %v1683 = stablehlo.reshape %v1232 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1684 = stablehlo.transpose %v1683, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1685 = stablehlo.reshape %v1684 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1686 = stablehlo.reshape %v1596 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1687 = stablehlo.transpose %v1686, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1688 = stablehlo.reshape %v1687 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1689 = stablehlo.reshape %v1685 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1690 = stablehlo.reshape %v1688 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1691 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1692 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1693 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1694 = stablehlo.reduce(%v1689 init: %v1691) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1695 = stablehlo.broadcast_in_dim %v1694, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1696 = stablehlo.divide %v1695, %v1692 : tensor<32x49x768xf32>
    %v1697 = stablehlo.subtract %v1689, %v1696 : tensor<32x49x768xf32>
    %v1698 = stablehlo.multiply %v1697, %v1697 : tensor<32x49x768xf32>
    %v1699 = stablehlo.reduce(%v1698 init: %v1691) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1700 = stablehlo.broadcast_in_dim %v1699, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1701 = stablehlo.divide %v1700, %v1692 : tensor<32x49x768xf32>
    %v1702 = stablehlo.add %v1701, %v1693 : tensor<32x49x768xf32>
    %v1703 = stablehlo.rsqrt %v1702 : tensor<32x49x768xf32>
    %v1704 = stablehlo.multiply %v1697, %v1703 : tensor<32x49x768xf32>
    %v1705 = stablehlo.multiply %v1690, %v1704 : tensor<32x49x768xf32>
    %v1706 = stablehlo.reduce(%v1705 init: %v1691) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1707 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1708 = stablehlo.multiply %v1706, %v1707 : tensor<768xf32>
    %v1709 = stablehlo.subtract %s3b1ng, %v1708 : tensor<768xf32>
    %v1710 = stablehlo.reshape %v1596 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1711 = stablehlo.transpose %v1710, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1712 = stablehlo.reshape %v1711 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1714 = stablehlo.reshape %v1712 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1715 = stablehlo.reduce(%v1714 init: %v1713) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1716 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1717 = stablehlo.multiply %v1715, %v1716 : tensor<768xf32>
    %v1718 = stablehlo.subtract %s3b1nbt, %v1717 : tensor<768xf32>
    %v1719 = stablehlo.reshape %v1227 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1720 = stablehlo.reshape %v1639 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1721 = stablehlo.transpose %v1719, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1722 = stablehlo.transpose %v1720, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1723 = stablehlo.convolution(%v1721, %v1722)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1724 = stablehlo.reshape %v1723 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1725 = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %v1726 = stablehlo.multiply %v1724, %v1725 : tensor<768x1x7x7xf32>
    %v1727 = stablehlo.subtract %s3b1dW, %v1726 : tensor<768x1x7x7xf32>
    %v1728 = stablehlo.reshape %v1639 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1730 = stablehlo.reduce(%v1728 init: %v1729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1731 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1732 = stablehlo.multiply %v1730, %v1731 : tensor<768xf32>
    %v1733 = stablehlo.subtract %s3b1db, %v1732 : tensor<768xf32>
    %v1734 = stablehlo.reshape %v1644 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1735 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1736 = stablehlo.multiply %v1734, %v1735 : tensor<32x768x7x7xf32>
    %v1737 = stablehlo.reshape %v1736 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1738 = stablehlo.reshape %v1737 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1739 = stablehlo.transpose %s3b0pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1740 = stablehlo.reverse %v1739, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1741 = stablehlo.convolution(%v1738, %v1740)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1742 = stablehlo.reshape %v1741 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1743 = stablehlo.multiply %v1204, %v1204 : tensor<32x150528xf32>
    %v1744 = stablehlo.multiply %v1743, %v1204 : tensor<32x150528xf32>
    %v1745 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1746 = stablehlo.multiply %v1745, %v1744 : tensor<32x150528xf32>
    %v1747 = stablehlo.add %v1204, %v1746 : tensor<32x150528xf32>
    %v1748 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1749 = stablehlo.multiply %v1748, %v1747 : tensor<32x150528xf32>
    %v1750 = stablehlo.tanh %v1749 : tensor<32x150528xf32>
    %v1751 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1752 = stablehlo.add %v1751, %v1750 : tensor<32x150528xf32>
    %v1753 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1754 = stablehlo.multiply %v1753, %v1752 : tensor<32x150528xf32>
    %v1755 = stablehlo.multiply %v1750, %v1750 : tensor<32x150528xf32>
    %v1756 = stablehlo.subtract %v1751, %v1755 : tensor<32x150528xf32>
    %v1757 = stablehlo.multiply %v1753, %v1204 : tensor<32x150528xf32>
    %v1758 = stablehlo.multiply %v1757, %v1756 : tensor<32x150528xf32>
    %v1759 = stablehlo.constant dense<0.134145> : tensor<32x150528xf32>
    %v1760 = stablehlo.multiply %v1759, %v1743 : tensor<32x150528xf32>
    %v1761 = stablehlo.add %v1751, %v1760 : tensor<32x150528xf32>
    %v1762 = stablehlo.multiply %v1748, %v1761 : tensor<32x150528xf32>
    %v1763 = stablehlo.multiply %v1758, %v1762 : tensor<32x150528xf32>
    %v1764 = stablehlo.add %v1754, %v1763 : tensor<32x150528xf32>
    %v1765 = stablehlo.multiply %v1742, %v1764 : tensor<32x150528xf32>
    %v1766 = stablehlo.reshape %v1765 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1767 = stablehlo.transpose %s3b0eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1768 = stablehlo.reverse %v1767, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1769 = stablehlo.convolution(%v1766, %v1768)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1770 = stablehlo.reshape %v1769 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1771 = stablehlo.reshape %v1165 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1772 = stablehlo.transpose %v1771, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1773 = stablehlo.reshape %v1772 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1774 = stablehlo.reshape %v1770 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1775 = stablehlo.transpose %v1774, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1776 = stablehlo.reshape %v1775 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1778 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1779 = stablehlo.multiply %v1777, %v1778 : tensor<32x49x768xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1782 = stablehlo.reshape %v1773 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1784 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1785 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1786 = stablehlo.reduce(%v1782 init: %v1783) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1787 = stablehlo.broadcast_in_dim %v1786, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1788 = stablehlo.divide %v1787, %v1784 : tensor<32x49x768xf32>
    %v1789 = stablehlo.subtract %v1782, %v1788 : tensor<32x49x768xf32>
    %v1790 = stablehlo.multiply %v1789, %v1789 : tensor<32x49x768xf32>
    %v1791 = stablehlo.reduce(%v1790 init: %v1783) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1792 = stablehlo.broadcast_in_dim %v1791, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1793 = stablehlo.divide %v1792, %v1784 : tensor<32x49x768xf32>
    %v1794 = stablehlo.add %v1793, %v1785 : tensor<32x49x768xf32>
    %v1795 = stablehlo.rsqrt %v1794 : tensor<32x49x768xf32>
    %v1796 = stablehlo.multiply %v1789, %v1795 : tensor<32x49x768xf32>
    %v1797 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1798 = stablehlo.multiply %v1797, %v1781 : tensor<32x49x768xf32>
    %v1799 = stablehlo.reduce(%v1798 init: %v1783) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1800 = stablehlo.broadcast_in_dim %v1799, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1801 = stablehlo.multiply %v1796, %v1798 : tensor<32x49x768xf32>
    %v1802 = stablehlo.reduce(%v1801 init: %v1783) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1803 = stablehlo.broadcast_in_dim %v1802, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1804 = stablehlo.multiply %v1798, %v1784 : tensor<32x49x768xf32>
    %v1805 = stablehlo.subtract %v1804, %v1800 : tensor<32x49x768xf32>
    %v1806 = stablehlo.multiply %v1796, %v1803 : tensor<32x49x768xf32>
    %v1807 = stablehlo.subtract %v1805, %v1806 : tensor<32x49x768xf32>
    %v1808 = stablehlo.divide %v1795, %v1784 : tensor<32x49x768xf32>
    %v1809 = stablehlo.multiply %v1808, %v1807 : tensor<32x49x768xf32>
    %v1810 = stablehlo.reshape %v1809 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1812 = stablehlo.transpose %v1811, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1813 = stablehlo.reshape %v1812 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1814 = stablehlo.reshape %v1813 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1815 = stablehlo.reverse %s3b0dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1816 = stablehlo.convolution(%v1814, %v1815)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1817 = stablehlo.reshape %v1816 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1818 = stablehlo.add %v1817, %v1644 : tensor<32x37632xf32>
    %v1819 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1820 = stablehlo.reshape %v1222 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1821 = stablehlo.reshape %v1644 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1822 = stablehlo.multiply %v1820, %v1821 : tensor<32x768x7x7xf32>
    %v1823 = stablehlo.reduce(%v1822 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1824 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1825 = stablehlo.multiply %v1823, %v1824 : tensor<768xf32>
    %v1826 = stablehlo.subtract %s3b0lg, %v1825 : tensor<768xf32>
    %v1827 = stablehlo.reshape %v1217 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1828 = stablehlo.reshape %v1737 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1829 = stablehlo.transpose %v1827, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1830 = stablehlo.transpose %v1828, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1831 = stablehlo.convolution(%v1829, %v1830)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1832 = stablehlo.transpose %v1831, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1833 = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %v1834 = stablehlo.multiply %v1832, %v1833 : tensor<768x3072x1x1xf32>
    %v1835 = stablehlo.subtract %s3b0pW, %v1834 : tensor<768x3072x1x1xf32>
    %v1836 = stablehlo.reshape %v1737 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1838 = stablehlo.reduce(%v1836 init: %v1837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1839 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1840 = stablehlo.multiply %v1838, %v1839 : tensor<768xf32>
    %v1841 = stablehlo.subtract %s3b0pb, %v1840 : tensor<768xf32>
    %v1842 = stablehlo.reshape %v1199 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1843 = stablehlo.reshape %v1765 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1844 = stablehlo.transpose %v1842, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1845 = stablehlo.transpose %v1843, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1846 = stablehlo.convolution(%v1844, %v1845)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1847 = stablehlo.transpose %v1846, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1848 = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %v1849 = stablehlo.multiply %v1847, %v1848 : tensor<3072x768x1x1xf32>
    %v1850 = stablehlo.subtract %s3b0eW, %v1849 : tensor<3072x768x1x1xf32>
    %v1851 = stablehlo.reshape %v1765 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1852 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1853 = stablehlo.reduce(%v1851 init: %v1852) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1854 = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %v1855 = stablehlo.multiply %v1853, %v1854 : tensor<3072xf32>
    %v1856 = stablehlo.subtract %s3b0eb, %v1855 : tensor<3072xf32>
    %v1857 = stablehlo.reshape %v1165 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1858 = stablehlo.transpose %v1857, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1859 = stablehlo.reshape %v1858 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1860 = stablehlo.reshape %v1770 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1861 = stablehlo.transpose %v1860, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1862 = stablehlo.reshape %v1861 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1863 = stablehlo.reshape %v1859 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1864 = stablehlo.reshape %v1862 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1866 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1867 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1868 = stablehlo.reduce(%v1863 init: %v1865) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1869 = stablehlo.broadcast_in_dim %v1868, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1870 = stablehlo.divide %v1869, %v1866 : tensor<32x49x768xf32>
    %v1871 = stablehlo.subtract %v1863, %v1870 : tensor<32x49x768xf32>
    %v1872 = stablehlo.multiply %v1871, %v1871 : tensor<32x49x768xf32>
    %v1873 = stablehlo.reduce(%v1872 init: %v1865) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1874 = stablehlo.broadcast_in_dim %v1873, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1875 = stablehlo.divide %v1874, %v1866 : tensor<32x49x768xf32>
    %v1876 = stablehlo.add %v1875, %v1867 : tensor<32x49x768xf32>
    %v1877 = stablehlo.rsqrt %v1876 : tensor<32x49x768xf32>
    %v1878 = stablehlo.multiply %v1871, %v1877 : tensor<32x49x768xf32>
    %v1879 = stablehlo.multiply %v1864, %v1878 : tensor<32x49x768xf32>
    %v1880 = stablehlo.reduce(%v1879 init: %v1865) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1881 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1882 = stablehlo.multiply %v1880, %v1881 : tensor<768xf32>
    %v1883 = stablehlo.subtract %s3b0ng, %v1882 : tensor<768xf32>
    %v1884 = stablehlo.reshape %v1770 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1885 = stablehlo.transpose %v1884, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1886 = stablehlo.reshape %v1885 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1888 = stablehlo.reshape %v1886 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1889 = stablehlo.reduce(%v1888 init: %v1887) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1890 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1891 = stablehlo.multiply %v1889, %v1890 : tensor<768xf32>
    %v1892 = stablehlo.subtract %s3b0nbt, %v1891 : tensor<768xf32>
    %v1893 = stablehlo.reshape %v1160 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1894 = stablehlo.reshape %v1813 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1895 = stablehlo.transpose %v1893, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1896 = stablehlo.transpose %v1894, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1897 = stablehlo.convolution(%v1895, %v1896)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1898 = stablehlo.reshape %v1897 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1899 = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %v1900 = stablehlo.multiply %v1898, %v1899 : tensor<768x1x7x7xf32>
    %v1901 = stablehlo.subtract %s3b0dW, %v1900 : tensor<768x1x7x7xf32>
    %v1902 = stablehlo.reshape %v1813 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1904 = stablehlo.reduce(%v1902 init: %v1903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1905 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1906 = stablehlo.multiply %v1904, %v1905 : tensor<768xf32>
    %v1907 = stablehlo.subtract %s3b0db, %v1906 : tensor<768xf32>
    %v1908 = stablehlo.reshape %v1818 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1910 = stablehlo.pad %v1908, %v1909, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x14x14xf32>
    %v1911 = stablehlo.transpose %d2W, dims = [1, 0, 2, 3] : (tensor<768x384x2x2xf32>) -> tensor<384x768x2x2xf32>
    %v1912 = stablehlo.reverse %v1911, dims = [2, 3] : tensor<384x768x2x2xf32>
    %v1913 = stablehlo.convolution(%v1910, %v1912)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x14x14xf32>, tensor<384x768x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v1914 = stablehlo.reshape %v1913 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1915 = stablehlo.reshape %v1121 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1916 = stablehlo.transpose %v1915, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1918 = stablehlo.reshape %v1914 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1919 = stablehlo.transpose %v1918, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1920 = stablehlo.reshape %v1919 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1921 = stablehlo.reshape %v1920 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1922 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1923 = stablehlo.multiply %v1921, %v1922 : tensor<32x196x384xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1925 = stablehlo.reshape %v1924 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1926 = stablehlo.reshape %v1917 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1928 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1929 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1930 = stablehlo.reduce(%v1926 init: %v1927) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1931 = stablehlo.broadcast_in_dim %v1930, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1932 = stablehlo.divide %v1931, %v1928 : tensor<32x196x384xf32>
    %v1933 = stablehlo.subtract %v1926, %v1932 : tensor<32x196x384xf32>
    %v1934 = stablehlo.multiply %v1933, %v1933 : tensor<32x196x384xf32>
    %v1935 = stablehlo.reduce(%v1934 init: %v1927) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1936 = stablehlo.broadcast_in_dim %v1935, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1937 = stablehlo.divide %v1936, %v1928 : tensor<32x196x384xf32>
    %v1938 = stablehlo.add %v1937, %v1929 : tensor<32x196x384xf32>
    %v1939 = stablehlo.rsqrt %v1938 : tensor<32x196x384xf32>
    %v1940 = stablehlo.multiply %v1933, %v1939 : tensor<32x196x384xf32>
    %v1941 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1942 = stablehlo.multiply %v1941, %v1925 : tensor<32x196x384xf32>
    %v1943 = stablehlo.reduce(%v1942 init: %v1927) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1944 = stablehlo.broadcast_in_dim %v1943, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1945 = stablehlo.multiply %v1940, %v1942 : tensor<32x196x384xf32>
    %v1946 = stablehlo.reduce(%v1945 init: %v1927) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1947 = stablehlo.broadcast_in_dim %v1946, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1948 = stablehlo.multiply %v1942, %v1928 : tensor<32x196x384xf32>
    %v1949 = stablehlo.subtract %v1948, %v1944 : tensor<32x196x384xf32>
    %v1950 = stablehlo.multiply %v1940, %v1947 : tensor<32x196x384xf32>
    %v1951 = stablehlo.subtract %v1949, %v1950 : tensor<32x196x384xf32>
    %v1952 = stablehlo.divide %v1939, %v1928 : tensor<32x196x384xf32>
    %v1953 = stablehlo.multiply %v1952, %v1951 : tensor<32x196x384xf32>
    %v1954 = stablehlo.reshape %v1953 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1955 = stablehlo.reshape %v1954 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1956 = stablehlo.transpose %v1955, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1957 = stablehlo.reshape %v1956 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1958 = stablehlo.reshape %v1818 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1960 = stablehlo.reduce(%v1958 init: %v1959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1961 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1962 = stablehlo.multiply %v1960, %v1961 : tensor<768xf32>
    %v1963 = stablehlo.subtract %d2b, %v1962 : tensor<768xf32>
    %v1964 = stablehlo.reshape %v1121 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1965 = stablehlo.transpose %v1964, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1966 = stablehlo.reshape %v1965 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1967 = stablehlo.reshape %v1914 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1968 = stablehlo.transpose %v1967, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1969 = stablehlo.reshape %v1968 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1970 = stablehlo.reshape %v1966 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1971 = stablehlo.reshape %v1969 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1972 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1973 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1974 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1975 = stablehlo.reduce(%v1970 init: %v1972) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1976 = stablehlo.broadcast_in_dim %v1975, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1977 = stablehlo.divide %v1976, %v1973 : tensor<32x196x384xf32>
    %v1978 = stablehlo.subtract %v1970, %v1977 : tensor<32x196x384xf32>
    %v1979 = stablehlo.multiply %v1978, %v1978 : tensor<32x196x384xf32>
    %v1980 = stablehlo.reduce(%v1979 init: %v1972) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1981 = stablehlo.broadcast_in_dim %v1980, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1982 = stablehlo.divide %v1981, %v1973 : tensor<32x196x384xf32>
    %v1983 = stablehlo.add %v1982, %v1974 : tensor<32x196x384xf32>
    %v1984 = stablehlo.rsqrt %v1983 : tensor<32x196x384xf32>
    %v1985 = stablehlo.multiply %v1978, %v1984 : tensor<32x196x384xf32>
    %v1986 = stablehlo.multiply %v1971, %v1985 : tensor<32x196x384xf32>
    %v1987 = stablehlo.reduce(%v1986 init: %v1972) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v1988 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1989 = stablehlo.multiply %v1987, %v1988 : tensor<384xf32>
    %v1990 = stablehlo.subtract %d2ng, %v1989 : tensor<384xf32>
    %v1991 = stablehlo.reshape %v1914 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1992 = stablehlo.transpose %v1991, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1993 = stablehlo.reshape %v1992 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1994 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1995 = stablehlo.reshape %v1993 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1996 = stablehlo.reduce(%v1995 init: %v1994) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v1997 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v1998 = stablehlo.multiply %v1996, %v1997 : tensor<384xf32>
    %v1999 = stablehlo.subtract %d2nbt, %v1998 : tensor<384xf32>
    %v2000 = stablehlo.reshape %v1155 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2001 = stablehlo.reshape %v1818 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2003 = stablehlo.pad %v2001, %v2002, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x13x13xf32>
    %v2004 = stablehlo.transpose %v2000, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2005 = stablehlo.transpose %v2003, dims = [1, 0, 2, 3] : (tensor<32x768x13x13xf32>) -> tensor<768x32x13x13xf32>
    %v2006 = stablehlo.convolution(%v2004, %v2005)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<768x32x13x13xf32>) -> tensor<384x768x2x2xf32>
    %v2007 = stablehlo.transpose %v2006, dims = [1, 0, 2, 3] : (tensor<384x768x2x2xf32>) -> tensor<768x384x2x2xf32>
    %v2008 = stablehlo.constant dense<0.1> : tensor<768x384x2x2xf32>
    %v2009 = stablehlo.multiply %v2007, %v2008 : tensor<768x384x2x2xf32>
    %v2010 = stablehlo.subtract %d2W, %v2009 : tensor<768x384x2x2xf32>
    %v2011 = stablehlo.reshape %v1957 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2012 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2013 = stablehlo.multiply %v2011, %v2012 : tensor<32x384x14x14xf32>
    %v2014 = stablehlo.reshape %v2013 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2015 = stablehlo.reshape %v2014 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2016 = stablehlo.transpose %s2b8pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2017 = stablehlo.reverse %v2016, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2018 = stablehlo.convolution(%v2015, %v2017)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2019 = stablehlo.reshape %v2018 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2020 = stablehlo.multiply %v1098, %v1098 : tensor<32x301056xf32>
    %v2021 = stablehlo.multiply %v2020, %v1098 : tensor<32x301056xf32>
    %v2022 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2023 = stablehlo.multiply %v2022, %v2021 : tensor<32x301056xf32>
    %v2024 = stablehlo.add %v1098, %v2023 : tensor<32x301056xf32>
    %v2025 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2026 = stablehlo.multiply %v2025, %v2024 : tensor<32x301056xf32>
    %v2027 = stablehlo.tanh %v2026 : tensor<32x301056xf32>
    %v2028 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2029 = stablehlo.add %v2028, %v2027 : tensor<32x301056xf32>
    %v2030 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2031 = stablehlo.multiply %v2030, %v2029 : tensor<32x301056xf32>
    %v2032 = stablehlo.multiply %v2027, %v2027 : tensor<32x301056xf32>
    %v2033 = stablehlo.subtract %v2028, %v2032 : tensor<32x301056xf32>
    %v2034 = stablehlo.multiply %v2030, %v1098 : tensor<32x301056xf32>
    %v2035 = stablehlo.multiply %v2034, %v2033 : tensor<32x301056xf32>
    %v2036 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2037 = stablehlo.multiply %v2036, %v2020 : tensor<32x301056xf32>
    %v2038 = stablehlo.add %v2028, %v2037 : tensor<32x301056xf32>
    %v2039 = stablehlo.multiply %v2025, %v2038 : tensor<32x301056xf32>
    %v2040 = stablehlo.multiply %v2035, %v2039 : tensor<32x301056xf32>
    %v2041 = stablehlo.add %v2031, %v2040 : tensor<32x301056xf32>
    %v2042 = stablehlo.multiply %v2019, %v2041 : tensor<32x301056xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2044 = stablehlo.transpose %s2b8eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2045 = stablehlo.reverse %v2044, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2046 = stablehlo.convolution(%v2043, %v2045)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2047 = stablehlo.reshape %v2046 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2048 = stablehlo.reshape %v1059 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2049 = stablehlo.transpose %v2048, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2050 = stablehlo.reshape %v2049 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2051 = stablehlo.reshape %v2047 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2052 = stablehlo.transpose %v2051, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2053 = stablehlo.reshape %v2052 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2054 = stablehlo.reshape %v2053 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2055 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2056 = stablehlo.multiply %v2054, %v2055 : tensor<32x196x384xf32>
    %v2057 = stablehlo.reshape %v2056 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2058 = stablehlo.reshape %v2057 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2059 = stablehlo.reshape %v2050 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2060 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2061 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2062 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2063 = stablehlo.reduce(%v2059 init: %v2060) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2064 = stablehlo.broadcast_in_dim %v2063, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2065 = stablehlo.divide %v2064, %v2061 : tensor<32x196x384xf32>
    %v2066 = stablehlo.subtract %v2059, %v2065 : tensor<32x196x384xf32>
    %v2067 = stablehlo.multiply %v2066, %v2066 : tensor<32x196x384xf32>
    %v2068 = stablehlo.reduce(%v2067 init: %v2060) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2069 = stablehlo.broadcast_in_dim %v2068, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2070 = stablehlo.divide %v2069, %v2061 : tensor<32x196x384xf32>
    %v2071 = stablehlo.add %v2070, %v2062 : tensor<32x196x384xf32>
    %v2072 = stablehlo.rsqrt %v2071 : tensor<32x196x384xf32>
    %v2073 = stablehlo.multiply %v2066, %v2072 : tensor<32x196x384xf32>
    %v2074 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2075 = stablehlo.multiply %v2074, %v2058 : tensor<32x196x384xf32>
    %v2076 = stablehlo.reduce(%v2075 init: %v2060) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2077 = stablehlo.broadcast_in_dim %v2076, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2078 = stablehlo.multiply %v2073, %v2075 : tensor<32x196x384xf32>
    %v2079 = stablehlo.reduce(%v2078 init: %v2060) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2080 = stablehlo.broadcast_in_dim %v2079, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2081 = stablehlo.multiply %v2075, %v2061 : tensor<32x196x384xf32>
    %v2082 = stablehlo.subtract %v2081, %v2077 : tensor<32x196x384xf32>
    %v2083 = stablehlo.multiply %v2073, %v2080 : tensor<32x196x384xf32>
    %v2084 = stablehlo.subtract %v2082, %v2083 : tensor<32x196x384xf32>
    %v2085 = stablehlo.divide %v2072, %v2061 : tensor<32x196x384xf32>
    %v2086 = stablehlo.multiply %v2085, %v2084 : tensor<32x196x384xf32>
    %v2087 = stablehlo.reshape %v2086 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2088 = stablehlo.reshape %v2087 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2089 = stablehlo.transpose %v2088, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2090 = stablehlo.reshape %v2089 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2091 = stablehlo.reshape %v2090 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2092 = stablehlo.reverse %s2b8dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2093 = stablehlo.convolution(%v2091, %v2092)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2094 = stablehlo.reshape %v2093 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2095 = stablehlo.add %v2094, %v1957 : tensor<32x75264xf32>
    %v2096 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2097 = stablehlo.reshape %v1116 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2098 = stablehlo.reshape %v1957 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2099 = stablehlo.multiply %v2097, %v2098 : tensor<32x384x14x14xf32>
    %v2100 = stablehlo.reduce(%v2099 init: %v2096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2101 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2102 = stablehlo.multiply %v2100, %v2101 : tensor<384xf32>
    %v2103 = stablehlo.subtract %s2b8lg, %v2102 : tensor<384xf32>
    %v2104 = stablehlo.reshape %v1111 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2105 = stablehlo.reshape %v2014 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2106 = stablehlo.transpose %v2104, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2107 = stablehlo.transpose %v2105, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2108 = stablehlo.convolution(%v2106, %v2107)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2109 = stablehlo.transpose %v2108, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2110 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2111 = stablehlo.multiply %v2109, %v2110 : tensor<384x1536x1x1xf32>
    %v2112 = stablehlo.subtract %s2b8pW, %v2111 : tensor<384x1536x1x1xf32>
    %v2113 = stablehlo.reshape %v2014 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2115 = stablehlo.reduce(%v2113 init: %v2114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2116 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2117 = stablehlo.multiply %v2115, %v2116 : tensor<384xf32>
    %v2118 = stablehlo.subtract %s2b8pb, %v2117 : tensor<384xf32>
    %v2119 = stablehlo.reshape %v1093 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2120 = stablehlo.reshape %v2042 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2121 = stablehlo.transpose %v2119, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2122 = stablehlo.transpose %v2120, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2123 = stablehlo.convolution(%v2121, %v2122)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2124 = stablehlo.transpose %v2123, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2125 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2126 = stablehlo.multiply %v2124, %v2125 : tensor<1536x384x1x1xf32>
    %v2127 = stablehlo.subtract %s2b8eW, %v2126 : tensor<1536x384x1x1xf32>
    %v2128 = stablehlo.reshape %v2042 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2130 = stablehlo.reduce(%v2128 init: %v2129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2131 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2132 = stablehlo.multiply %v2130, %v2131 : tensor<1536xf32>
    %v2133 = stablehlo.subtract %s2b8eb, %v2132 : tensor<1536xf32>
    %v2134 = stablehlo.reshape %v1059 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2135 = stablehlo.transpose %v2134, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2136 = stablehlo.reshape %v2135 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2137 = stablehlo.reshape %v2047 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2138 = stablehlo.transpose %v2137, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2139 = stablehlo.reshape %v2138 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2140 = stablehlo.reshape %v2136 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2141 = stablehlo.reshape %v2139 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2143 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2144 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2145 = stablehlo.reduce(%v2140 init: %v2142) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2146 = stablehlo.broadcast_in_dim %v2145, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2147 = stablehlo.divide %v2146, %v2143 : tensor<32x196x384xf32>
    %v2148 = stablehlo.subtract %v2140, %v2147 : tensor<32x196x384xf32>
    %v2149 = stablehlo.multiply %v2148, %v2148 : tensor<32x196x384xf32>
    %v2150 = stablehlo.reduce(%v2149 init: %v2142) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2151 = stablehlo.broadcast_in_dim %v2150, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2152 = stablehlo.divide %v2151, %v2143 : tensor<32x196x384xf32>
    %v2153 = stablehlo.add %v2152, %v2144 : tensor<32x196x384xf32>
    %v2154 = stablehlo.rsqrt %v2153 : tensor<32x196x384xf32>
    %v2155 = stablehlo.multiply %v2148, %v2154 : tensor<32x196x384xf32>
    %v2156 = stablehlo.multiply %v2141, %v2155 : tensor<32x196x384xf32>
    %v2157 = stablehlo.reduce(%v2156 init: %v2142) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2158 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2159 = stablehlo.multiply %v2157, %v2158 : tensor<384xf32>
    %v2160 = stablehlo.subtract %s2b8ng, %v2159 : tensor<384xf32>
    %v2161 = stablehlo.reshape %v2047 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2162 = stablehlo.transpose %v2161, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2163 = stablehlo.reshape %v2162 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2165 = stablehlo.reshape %v2163 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2166 = stablehlo.reduce(%v2165 init: %v2164) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2167 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2168 = stablehlo.multiply %v2166, %v2167 : tensor<384xf32>
    %v2169 = stablehlo.subtract %s2b8nbt, %v2168 : tensor<384xf32>
    %v2170 = stablehlo.reshape %v1054 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2171 = stablehlo.reshape %v2090 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2172 = stablehlo.transpose %v2170, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2173 = stablehlo.transpose %v2171, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2174 = stablehlo.convolution(%v2172, %v2173)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2175 = stablehlo.reshape %v2174 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2176 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2177 = stablehlo.multiply %v2175, %v2176 : tensor<384x1x7x7xf32>
    %v2178 = stablehlo.subtract %s2b8dW, %v2177 : tensor<384x1x7x7xf32>
    %v2179 = stablehlo.reshape %v2090 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2181 = stablehlo.reduce(%v2179 init: %v2180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2182 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2183 = stablehlo.multiply %v2181, %v2182 : tensor<384xf32>
    %v2184 = stablehlo.subtract %s2b8db, %v2183 : tensor<384xf32>
    %v2185 = stablehlo.reshape %v2095 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2186 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2187 = stablehlo.multiply %v2185, %v2186 : tensor<32x384x14x14xf32>
    %v2188 = stablehlo.reshape %v2187 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2189 = stablehlo.reshape %v2188 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2190 = stablehlo.transpose %s2b7pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2191 = stablehlo.reverse %v2190, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2192 = stablehlo.convolution(%v2189, %v2191)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2193 = stablehlo.reshape %v2192 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2194 = stablehlo.multiply %v1031, %v1031 : tensor<32x301056xf32>
    %v2195 = stablehlo.multiply %v2194, %v1031 : tensor<32x301056xf32>
    %v2196 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2197 = stablehlo.multiply %v2196, %v2195 : tensor<32x301056xf32>
    %v2198 = stablehlo.add %v1031, %v2197 : tensor<32x301056xf32>
    %v2199 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2200 = stablehlo.multiply %v2199, %v2198 : tensor<32x301056xf32>
    %v2201 = stablehlo.tanh %v2200 : tensor<32x301056xf32>
    %v2202 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2203 = stablehlo.add %v2202, %v2201 : tensor<32x301056xf32>
    %v2204 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2205 = stablehlo.multiply %v2204, %v2203 : tensor<32x301056xf32>
    %v2206 = stablehlo.multiply %v2201, %v2201 : tensor<32x301056xf32>
    %v2207 = stablehlo.subtract %v2202, %v2206 : tensor<32x301056xf32>
    %v2208 = stablehlo.multiply %v2204, %v1031 : tensor<32x301056xf32>
    %v2209 = stablehlo.multiply %v2208, %v2207 : tensor<32x301056xf32>
    %v2210 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2211 = stablehlo.multiply %v2210, %v2194 : tensor<32x301056xf32>
    %v2212 = stablehlo.add %v2202, %v2211 : tensor<32x301056xf32>
    %v2213 = stablehlo.multiply %v2199, %v2212 : tensor<32x301056xf32>
    %v2214 = stablehlo.multiply %v2209, %v2213 : tensor<32x301056xf32>
    %v2215 = stablehlo.add %v2205, %v2214 : tensor<32x301056xf32>
    %v2216 = stablehlo.multiply %v2193, %v2215 : tensor<32x301056xf32>
    %v2217 = stablehlo.reshape %v2216 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2218 = stablehlo.transpose %s2b7eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2219 = stablehlo.reverse %v2218, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2220 = stablehlo.convolution(%v2217, %v2219)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2221 = stablehlo.reshape %v2220 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2222 = stablehlo.reshape %v992 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2223 = stablehlo.transpose %v2222, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2224 = stablehlo.reshape %v2223 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2225 = stablehlo.reshape %v2221 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2226 = stablehlo.transpose %v2225, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2227 = stablehlo.reshape %v2226 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2228 = stablehlo.reshape %v2227 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2229 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2230 = stablehlo.multiply %v2228, %v2229 : tensor<32x196x384xf32>
    %v2231 = stablehlo.reshape %v2230 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2232 = stablehlo.reshape %v2231 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2233 = stablehlo.reshape %v2224 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2235 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2236 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2237 = stablehlo.reduce(%v2233 init: %v2234) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2238 = stablehlo.broadcast_in_dim %v2237, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2239 = stablehlo.divide %v2238, %v2235 : tensor<32x196x384xf32>
    %v2240 = stablehlo.subtract %v2233, %v2239 : tensor<32x196x384xf32>
    %v2241 = stablehlo.multiply %v2240, %v2240 : tensor<32x196x384xf32>
    %v2242 = stablehlo.reduce(%v2241 init: %v2234) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2243 = stablehlo.broadcast_in_dim %v2242, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2244 = stablehlo.divide %v2243, %v2235 : tensor<32x196x384xf32>
    %v2245 = stablehlo.add %v2244, %v2236 : tensor<32x196x384xf32>
    %v2246 = stablehlo.rsqrt %v2245 : tensor<32x196x384xf32>
    %v2247 = stablehlo.multiply %v2240, %v2246 : tensor<32x196x384xf32>
    %v2248 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2249 = stablehlo.multiply %v2248, %v2232 : tensor<32x196x384xf32>
    %v2250 = stablehlo.reduce(%v2249 init: %v2234) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2251 = stablehlo.broadcast_in_dim %v2250, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2252 = stablehlo.multiply %v2247, %v2249 : tensor<32x196x384xf32>
    %v2253 = stablehlo.reduce(%v2252 init: %v2234) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2254 = stablehlo.broadcast_in_dim %v2253, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2255 = stablehlo.multiply %v2249, %v2235 : tensor<32x196x384xf32>
    %v2256 = stablehlo.subtract %v2255, %v2251 : tensor<32x196x384xf32>
    %v2257 = stablehlo.multiply %v2247, %v2254 : tensor<32x196x384xf32>
    %v2258 = stablehlo.subtract %v2256, %v2257 : tensor<32x196x384xf32>
    %v2259 = stablehlo.divide %v2246, %v2235 : tensor<32x196x384xf32>
    %v2260 = stablehlo.multiply %v2259, %v2258 : tensor<32x196x384xf32>
    %v2261 = stablehlo.reshape %v2260 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2262 = stablehlo.reshape %v2261 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2263 = stablehlo.transpose %v2262, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2264 = stablehlo.reshape %v2263 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2266 = stablehlo.reverse %s2b7dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2267 = stablehlo.convolution(%v2265, %v2266)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2268 = stablehlo.reshape %v2267 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2269 = stablehlo.add %v2268, %v2095 : tensor<32x75264xf32>
    %v2270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2271 = stablehlo.reshape %v1049 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2272 = stablehlo.reshape %v2095 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2273 = stablehlo.multiply %v2271, %v2272 : tensor<32x384x14x14xf32>
    %v2274 = stablehlo.reduce(%v2273 init: %v2270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2275 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2276 = stablehlo.multiply %v2274, %v2275 : tensor<384xf32>
    %v2277 = stablehlo.subtract %s2b7lg, %v2276 : tensor<384xf32>
    %v2278 = stablehlo.reshape %v1044 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2279 = stablehlo.reshape %v2188 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2280 = stablehlo.transpose %v2278, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2281 = stablehlo.transpose %v2279, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2282 = stablehlo.convolution(%v2280, %v2281)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2283 = stablehlo.transpose %v2282, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2284 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2285 = stablehlo.multiply %v2283, %v2284 : tensor<384x1536x1x1xf32>
    %v2286 = stablehlo.subtract %s2b7pW, %v2285 : tensor<384x1536x1x1xf32>
    %v2287 = stablehlo.reshape %v2188 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2289 = stablehlo.reduce(%v2287 init: %v2288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2290 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2291 = stablehlo.multiply %v2289, %v2290 : tensor<384xf32>
    %v2292 = stablehlo.subtract %s2b7pb, %v2291 : tensor<384xf32>
    %v2293 = stablehlo.reshape %v1026 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2294 = stablehlo.reshape %v2216 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2295 = stablehlo.transpose %v2293, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2296 = stablehlo.transpose %v2294, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2297 = stablehlo.convolution(%v2295, %v2296)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2298 = stablehlo.transpose %v2297, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2299 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2300 = stablehlo.multiply %v2298, %v2299 : tensor<1536x384x1x1xf32>
    %v2301 = stablehlo.subtract %s2b7eW, %v2300 : tensor<1536x384x1x1xf32>
    %v2302 = stablehlo.reshape %v2216 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2304 = stablehlo.reduce(%v2302 init: %v2303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2305 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2306 = stablehlo.multiply %v2304, %v2305 : tensor<1536xf32>
    %v2307 = stablehlo.subtract %s2b7eb, %v2306 : tensor<1536xf32>
    %v2308 = stablehlo.reshape %v992 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2309 = stablehlo.transpose %v2308, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2310 = stablehlo.reshape %v2309 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2311 = stablehlo.reshape %v2221 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2312 = stablehlo.transpose %v2311, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2313 = stablehlo.reshape %v2312 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2314 = stablehlo.reshape %v2310 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2315 = stablehlo.reshape %v2313 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2317 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2318 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2319 = stablehlo.reduce(%v2314 init: %v2316) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2320 = stablehlo.broadcast_in_dim %v2319, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2321 = stablehlo.divide %v2320, %v2317 : tensor<32x196x384xf32>
    %v2322 = stablehlo.subtract %v2314, %v2321 : tensor<32x196x384xf32>
    %v2323 = stablehlo.multiply %v2322, %v2322 : tensor<32x196x384xf32>
    %v2324 = stablehlo.reduce(%v2323 init: %v2316) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2325 = stablehlo.broadcast_in_dim %v2324, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2326 = stablehlo.divide %v2325, %v2317 : tensor<32x196x384xf32>
    %v2327 = stablehlo.add %v2326, %v2318 : tensor<32x196x384xf32>
    %v2328 = stablehlo.rsqrt %v2327 : tensor<32x196x384xf32>
    %v2329 = stablehlo.multiply %v2322, %v2328 : tensor<32x196x384xf32>
    %v2330 = stablehlo.multiply %v2315, %v2329 : tensor<32x196x384xf32>
    %v2331 = stablehlo.reduce(%v2330 init: %v2316) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2332 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2333 = stablehlo.multiply %v2331, %v2332 : tensor<384xf32>
    %v2334 = stablehlo.subtract %s2b7ng, %v2333 : tensor<384xf32>
    %v2335 = stablehlo.reshape %v2221 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2336 = stablehlo.transpose %v2335, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2337 = stablehlo.reshape %v2336 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2339 = stablehlo.reshape %v2337 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2340 = stablehlo.reduce(%v2339 init: %v2338) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2341 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2342 = stablehlo.multiply %v2340, %v2341 : tensor<384xf32>
    %v2343 = stablehlo.subtract %s2b7nbt, %v2342 : tensor<384xf32>
    %v2344 = stablehlo.reshape %v987 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2345 = stablehlo.reshape %v2264 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2346 = stablehlo.transpose %v2344, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2347 = stablehlo.transpose %v2345, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2348 = stablehlo.convolution(%v2346, %v2347)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2350 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2351 = stablehlo.multiply %v2349, %v2350 : tensor<384x1x7x7xf32>
    %v2352 = stablehlo.subtract %s2b7dW, %v2351 : tensor<384x1x7x7xf32>
    %v2353 = stablehlo.reshape %v2264 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2354 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2355 = stablehlo.reduce(%v2353 init: %v2354) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2356 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2357 = stablehlo.multiply %v2355, %v2356 : tensor<384xf32>
    %v2358 = stablehlo.subtract %s2b7db, %v2357 : tensor<384xf32>
    %v2359 = stablehlo.reshape %v2269 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2360 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2361 = stablehlo.multiply %v2359, %v2360 : tensor<32x384x14x14xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2363 = stablehlo.reshape %v2362 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2364 = stablehlo.transpose %s2b6pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2365 = stablehlo.reverse %v2364, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2366 = stablehlo.convolution(%v2363, %v2365)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2367 = stablehlo.reshape %v2366 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2368 = stablehlo.multiply %v964, %v964 : tensor<32x301056xf32>
    %v2369 = stablehlo.multiply %v2368, %v964 : tensor<32x301056xf32>
    %v2370 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2371 = stablehlo.multiply %v2370, %v2369 : tensor<32x301056xf32>
    %v2372 = stablehlo.add %v964, %v2371 : tensor<32x301056xf32>
    %v2373 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2374 = stablehlo.multiply %v2373, %v2372 : tensor<32x301056xf32>
    %v2375 = stablehlo.tanh %v2374 : tensor<32x301056xf32>
    %v2376 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2377 = stablehlo.add %v2376, %v2375 : tensor<32x301056xf32>
    %v2378 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2379 = stablehlo.multiply %v2378, %v2377 : tensor<32x301056xf32>
    %v2380 = stablehlo.multiply %v2375, %v2375 : tensor<32x301056xf32>
    %v2381 = stablehlo.subtract %v2376, %v2380 : tensor<32x301056xf32>
    %v2382 = stablehlo.multiply %v2378, %v964 : tensor<32x301056xf32>
    %v2383 = stablehlo.multiply %v2382, %v2381 : tensor<32x301056xf32>
    %v2384 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2385 = stablehlo.multiply %v2384, %v2368 : tensor<32x301056xf32>
    %v2386 = stablehlo.add %v2376, %v2385 : tensor<32x301056xf32>
    %v2387 = stablehlo.multiply %v2373, %v2386 : tensor<32x301056xf32>
    %v2388 = stablehlo.multiply %v2383, %v2387 : tensor<32x301056xf32>
    %v2389 = stablehlo.add %v2379, %v2388 : tensor<32x301056xf32>
    %v2390 = stablehlo.multiply %v2367, %v2389 : tensor<32x301056xf32>
    %v2391 = stablehlo.reshape %v2390 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2392 = stablehlo.transpose %s2b6eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2393 = stablehlo.reverse %v2392, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2394 = stablehlo.convolution(%v2391, %v2393)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2396 = stablehlo.reshape %v925 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2397 = stablehlo.transpose %v2396, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2398 = stablehlo.reshape %v2397 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2399 = stablehlo.reshape %v2395 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2400 = stablehlo.transpose %v2399, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2401 = stablehlo.reshape %v2400 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2402 = stablehlo.reshape %v2401 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2403 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2404 = stablehlo.multiply %v2402, %v2403 : tensor<32x196x384xf32>
    %v2405 = stablehlo.reshape %v2404 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2406 = stablehlo.reshape %v2405 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2407 = stablehlo.reshape %v2398 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2408 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2409 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2410 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2411 = stablehlo.reduce(%v2407 init: %v2408) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2412 = stablehlo.broadcast_in_dim %v2411, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2413 = stablehlo.divide %v2412, %v2409 : tensor<32x196x384xf32>
    %v2414 = stablehlo.subtract %v2407, %v2413 : tensor<32x196x384xf32>
    %v2415 = stablehlo.multiply %v2414, %v2414 : tensor<32x196x384xf32>
    %v2416 = stablehlo.reduce(%v2415 init: %v2408) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2417 = stablehlo.broadcast_in_dim %v2416, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2418 = stablehlo.divide %v2417, %v2409 : tensor<32x196x384xf32>
    %v2419 = stablehlo.add %v2418, %v2410 : tensor<32x196x384xf32>
    %v2420 = stablehlo.rsqrt %v2419 : tensor<32x196x384xf32>
    %v2421 = stablehlo.multiply %v2414, %v2420 : tensor<32x196x384xf32>
    %v2422 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2423 = stablehlo.multiply %v2422, %v2406 : tensor<32x196x384xf32>
    %v2424 = stablehlo.reduce(%v2423 init: %v2408) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2425 = stablehlo.broadcast_in_dim %v2424, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2426 = stablehlo.multiply %v2421, %v2423 : tensor<32x196x384xf32>
    %v2427 = stablehlo.reduce(%v2426 init: %v2408) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2428 = stablehlo.broadcast_in_dim %v2427, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2429 = stablehlo.multiply %v2423, %v2409 : tensor<32x196x384xf32>
    %v2430 = stablehlo.subtract %v2429, %v2425 : tensor<32x196x384xf32>
    %v2431 = stablehlo.multiply %v2421, %v2428 : tensor<32x196x384xf32>
    %v2432 = stablehlo.subtract %v2430, %v2431 : tensor<32x196x384xf32>
    %v2433 = stablehlo.divide %v2420, %v2409 : tensor<32x196x384xf32>
    %v2434 = stablehlo.multiply %v2433, %v2432 : tensor<32x196x384xf32>
    %v2435 = stablehlo.reshape %v2434 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2436 = stablehlo.reshape %v2435 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2437 = stablehlo.transpose %v2436, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2438 = stablehlo.reshape %v2437 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2439 = stablehlo.reshape %v2438 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2440 = stablehlo.reverse %s2b6dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2441 = stablehlo.convolution(%v2439, %v2440)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2442 = stablehlo.reshape %v2441 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2443 = stablehlo.add %v2442, %v2269 : tensor<32x75264xf32>
    %v2444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2445 = stablehlo.reshape %v982 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2446 = stablehlo.reshape %v2269 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2447 = stablehlo.multiply %v2445, %v2446 : tensor<32x384x14x14xf32>
    %v2448 = stablehlo.reduce(%v2447 init: %v2444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2449 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2450 = stablehlo.multiply %v2448, %v2449 : tensor<384xf32>
    %v2451 = stablehlo.subtract %s2b6lg, %v2450 : tensor<384xf32>
    %v2452 = stablehlo.reshape %v977 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2453 = stablehlo.reshape %v2362 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2454 = stablehlo.transpose %v2452, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2455 = stablehlo.transpose %v2453, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2456 = stablehlo.convolution(%v2454, %v2455)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2457 = stablehlo.transpose %v2456, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2458 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2459 = stablehlo.multiply %v2457, %v2458 : tensor<384x1536x1x1xf32>
    %v2460 = stablehlo.subtract %s2b6pW, %v2459 : tensor<384x1536x1x1xf32>
    %v2461 = stablehlo.reshape %v2362 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2463 = stablehlo.reduce(%v2461 init: %v2462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2464 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2465 = stablehlo.multiply %v2463, %v2464 : tensor<384xf32>
    %v2466 = stablehlo.subtract %s2b6pb, %v2465 : tensor<384xf32>
    %v2467 = stablehlo.reshape %v959 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2468 = stablehlo.reshape %v2390 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2469 = stablehlo.transpose %v2467, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2470 = stablehlo.transpose %v2468, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2471 = stablehlo.convolution(%v2469, %v2470)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2472 = stablehlo.transpose %v2471, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2473 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2474 = stablehlo.multiply %v2472, %v2473 : tensor<1536x384x1x1xf32>
    %v2475 = stablehlo.subtract %s2b6eW, %v2474 : tensor<1536x384x1x1xf32>
    %v2476 = stablehlo.reshape %v2390 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2478 = stablehlo.reduce(%v2476 init: %v2477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2479 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2480 = stablehlo.multiply %v2478, %v2479 : tensor<1536xf32>
    %v2481 = stablehlo.subtract %s2b6eb, %v2480 : tensor<1536xf32>
    %v2482 = stablehlo.reshape %v925 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2483 = stablehlo.transpose %v2482, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2484 = stablehlo.reshape %v2483 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2485 = stablehlo.reshape %v2395 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2486 = stablehlo.transpose %v2485, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2487 = stablehlo.reshape %v2486 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2488 = stablehlo.reshape %v2484 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2489 = stablehlo.reshape %v2487 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2491 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2492 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2493 = stablehlo.reduce(%v2488 init: %v2490) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2494 = stablehlo.broadcast_in_dim %v2493, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2495 = stablehlo.divide %v2494, %v2491 : tensor<32x196x384xf32>
    %v2496 = stablehlo.subtract %v2488, %v2495 : tensor<32x196x384xf32>
    %v2497 = stablehlo.multiply %v2496, %v2496 : tensor<32x196x384xf32>
    %v2498 = stablehlo.reduce(%v2497 init: %v2490) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2499 = stablehlo.broadcast_in_dim %v2498, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2500 = stablehlo.divide %v2499, %v2491 : tensor<32x196x384xf32>
    %v2501 = stablehlo.add %v2500, %v2492 : tensor<32x196x384xf32>
    %v2502 = stablehlo.rsqrt %v2501 : tensor<32x196x384xf32>
    %v2503 = stablehlo.multiply %v2496, %v2502 : tensor<32x196x384xf32>
    %v2504 = stablehlo.multiply %v2489, %v2503 : tensor<32x196x384xf32>
    %v2505 = stablehlo.reduce(%v2504 init: %v2490) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2506 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2507 = stablehlo.multiply %v2505, %v2506 : tensor<384xf32>
    %v2508 = stablehlo.subtract %s2b6ng, %v2507 : tensor<384xf32>
    %v2509 = stablehlo.reshape %v2395 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2510 = stablehlo.transpose %v2509, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2511 = stablehlo.reshape %v2510 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2513 = stablehlo.reshape %v2511 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2514 = stablehlo.reduce(%v2513 init: %v2512) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2515 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2516 = stablehlo.multiply %v2514, %v2515 : tensor<384xf32>
    %v2517 = stablehlo.subtract %s2b6nbt, %v2516 : tensor<384xf32>
    %v2518 = stablehlo.reshape %v920 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2519 = stablehlo.reshape %v2438 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2520 = stablehlo.transpose %v2518, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2521 = stablehlo.transpose %v2519, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2522 = stablehlo.convolution(%v2520, %v2521)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2523 = stablehlo.reshape %v2522 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2524 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2525 = stablehlo.multiply %v2523, %v2524 : tensor<384x1x7x7xf32>
    %v2526 = stablehlo.subtract %s2b6dW, %v2525 : tensor<384x1x7x7xf32>
    %v2527 = stablehlo.reshape %v2438 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2529 = stablehlo.reduce(%v2527 init: %v2528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2530 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2531 = stablehlo.multiply %v2529, %v2530 : tensor<384xf32>
    %v2532 = stablehlo.subtract %s2b6db, %v2531 : tensor<384xf32>
    %v2533 = stablehlo.reshape %v2443 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2534 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2535 = stablehlo.multiply %v2533, %v2534 : tensor<32x384x14x14xf32>
    %v2536 = stablehlo.reshape %v2535 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2537 = stablehlo.reshape %v2536 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2538 = stablehlo.transpose %s2b5pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2539 = stablehlo.reverse %v2538, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2540 = stablehlo.convolution(%v2537, %v2539)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2541 = stablehlo.reshape %v2540 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2542 = stablehlo.multiply %v897, %v897 : tensor<32x301056xf32>
    %v2543 = stablehlo.multiply %v2542, %v897 : tensor<32x301056xf32>
    %v2544 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2545 = stablehlo.multiply %v2544, %v2543 : tensor<32x301056xf32>
    %v2546 = stablehlo.add %v897, %v2545 : tensor<32x301056xf32>
    %v2547 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2548 = stablehlo.multiply %v2547, %v2546 : tensor<32x301056xf32>
    %v2549 = stablehlo.tanh %v2548 : tensor<32x301056xf32>
    %v2550 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2551 = stablehlo.add %v2550, %v2549 : tensor<32x301056xf32>
    %v2552 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2553 = stablehlo.multiply %v2552, %v2551 : tensor<32x301056xf32>
    %v2554 = stablehlo.multiply %v2549, %v2549 : tensor<32x301056xf32>
    %v2555 = stablehlo.subtract %v2550, %v2554 : tensor<32x301056xf32>
    %v2556 = stablehlo.multiply %v2552, %v897 : tensor<32x301056xf32>
    %v2557 = stablehlo.multiply %v2556, %v2555 : tensor<32x301056xf32>
    %v2558 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2559 = stablehlo.multiply %v2558, %v2542 : tensor<32x301056xf32>
    %v2560 = stablehlo.add %v2550, %v2559 : tensor<32x301056xf32>
    %v2561 = stablehlo.multiply %v2547, %v2560 : tensor<32x301056xf32>
    %v2562 = stablehlo.multiply %v2557, %v2561 : tensor<32x301056xf32>
    %v2563 = stablehlo.add %v2553, %v2562 : tensor<32x301056xf32>
    %v2564 = stablehlo.multiply %v2541, %v2563 : tensor<32x301056xf32>
    %v2565 = stablehlo.reshape %v2564 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2566 = stablehlo.transpose %s2b5eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2567 = stablehlo.reverse %v2566, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2568 = stablehlo.convolution(%v2565, %v2567)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2569 = stablehlo.reshape %v2568 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2570 = stablehlo.reshape %v858 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2571 = stablehlo.transpose %v2570, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2572 = stablehlo.reshape %v2571 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2573 = stablehlo.reshape %v2569 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2574 = stablehlo.transpose %v2573, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2575 = stablehlo.reshape %v2574 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2576 = stablehlo.reshape %v2575 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2577 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2578 = stablehlo.multiply %v2576, %v2577 : tensor<32x196x384xf32>
    %v2579 = stablehlo.reshape %v2578 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2580 = stablehlo.reshape %v2579 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2581 = stablehlo.reshape %v2572 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2583 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2584 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2585 = stablehlo.reduce(%v2581 init: %v2582) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2586 = stablehlo.broadcast_in_dim %v2585, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2587 = stablehlo.divide %v2586, %v2583 : tensor<32x196x384xf32>
    %v2588 = stablehlo.subtract %v2581, %v2587 : tensor<32x196x384xf32>
    %v2589 = stablehlo.multiply %v2588, %v2588 : tensor<32x196x384xf32>
    %v2590 = stablehlo.reduce(%v2589 init: %v2582) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2591 = stablehlo.broadcast_in_dim %v2590, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2592 = stablehlo.divide %v2591, %v2583 : tensor<32x196x384xf32>
    %v2593 = stablehlo.add %v2592, %v2584 : tensor<32x196x384xf32>
    %v2594 = stablehlo.rsqrt %v2593 : tensor<32x196x384xf32>
    %v2595 = stablehlo.multiply %v2588, %v2594 : tensor<32x196x384xf32>
    %v2596 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2597 = stablehlo.multiply %v2596, %v2580 : tensor<32x196x384xf32>
    %v2598 = stablehlo.reduce(%v2597 init: %v2582) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2599 = stablehlo.broadcast_in_dim %v2598, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2600 = stablehlo.multiply %v2595, %v2597 : tensor<32x196x384xf32>
    %v2601 = stablehlo.reduce(%v2600 init: %v2582) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2602 = stablehlo.broadcast_in_dim %v2601, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2603 = stablehlo.multiply %v2597, %v2583 : tensor<32x196x384xf32>
    %v2604 = stablehlo.subtract %v2603, %v2599 : tensor<32x196x384xf32>
    %v2605 = stablehlo.multiply %v2595, %v2602 : tensor<32x196x384xf32>
    %v2606 = stablehlo.subtract %v2604, %v2605 : tensor<32x196x384xf32>
    %v2607 = stablehlo.divide %v2594, %v2583 : tensor<32x196x384xf32>
    %v2608 = stablehlo.multiply %v2607, %v2606 : tensor<32x196x384xf32>
    %v2609 = stablehlo.reshape %v2608 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2611 = stablehlo.transpose %v2610, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2612 = stablehlo.reshape %v2611 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2613 = stablehlo.reshape %v2612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2614 = stablehlo.reverse %s2b5dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2615 = stablehlo.convolution(%v2613, %v2614)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2616 = stablehlo.reshape %v2615 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2617 = stablehlo.add %v2616, %v2443 : tensor<32x75264xf32>
    %v2618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2619 = stablehlo.reshape %v915 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2620 = stablehlo.reshape %v2443 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2621 = stablehlo.multiply %v2619, %v2620 : tensor<32x384x14x14xf32>
    %v2622 = stablehlo.reduce(%v2621 init: %v2618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2623 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2624 = stablehlo.multiply %v2622, %v2623 : tensor<384xf32>
    %v2625 = stablehlo.subtract %s2b5lg, %v2624 : tensor<384xf32>
    %v2626 = stablehlo.reshape %v910 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2627 = stablehlo.reshape %v2536 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2628 = stablehlo.transpose %v2626, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2629 = stablehlo.transpose %v2627, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2630 = stablehlo.convolution(%v2628, %v2629)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2631 = stablehlo.transpose %v2630, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2632 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2633 = stablehlo.multiply %v2631, %v2632 : tensor<384x1536x1x1xf32>
    %v2634 = stablehlo.subtract %s2b5pW, %v2633 : tensor<384x1536x1x1xf32>
    %v2635 = stablehlo.reshape %v2536 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2637 = stablehlo.reduce(%v2635 init: %v2636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2638 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2639 = stablehlo.multiply %v2637, %v2638 : tensor<384xf32>
    %v2640 = stablehlo.subtract %s2b5pb, %v2639 : tensor<384xf32>
    %v2641 = stablehlo.reshape %v892 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2642 = stablehlo.reshape %v2564 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2643 = stablehlo.transpose %v2641, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2644 = stablehlo.transpose %v2642, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2645 = stablehlo.convolution(%v2643, %v2644)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2646 = stablehlo.transpose %v2645, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2647 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2648 = stablehlo.multiply %v2646, %v2647 : tensor<1536x384x1x1xf32>
    %v2649 = stablehlo.subtract %s2b5eW, %v2648 : tensor<1536x384x1x1xf32>
    %v2650 = stablehlo.reshape %v2564 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2651 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2652 = stablehlo.reduce(%v2650 init: %v2651) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2653 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2654 = stablehlo.multiply %v2652, %v2653 : tensor<1536xf32>
    %v2655 = stablehlo.subtract %s2b5eb, %v2654 : tensor<1536xf32>
    %v2656 = stablehlo.reshape %v858 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2657 = stablehlo.transpose %v2656, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2658 = stablehlo.reshape %v2657 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2659 = stablehlo.reshape %v2569 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2660 = stablehlo.transpose %v2659, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2661 = stablehlo.reshape %v2660 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2662 = stablehlo.reshape %v2658 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2663 = stablehlo.reshape %v2661 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2665 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2666 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2667 = stablehlo.reduce(%v2662 init: %v2664) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2668 = stablehlo.broadcast_in_dim %v2667, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2669 = stablehlo.divide %v2668, %v2665 : tensor<32x196x384xf32>
    %v2670 = stablehlo.subtract %v2662, %v2669 : tensor<32x196x384xf32>
    %v2671 = stablehlo.multiply %v2670, %v2670 : tensor<32x196x384xf32>
    %v2672 = stablehlo.reduce(%v2671 init: %v2664) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2673 = stablehlo.broadcast_in_dim %v2672, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2674 = stablehlo.divide %v2673, %v2665 : tensor<32x196x384xf32>
    %v2675 = stablehlo.add %v2674, %v2666 : tensor<32x196x384xf32>
    %v2676 = stablehlo.rsqrt %v2675 : tensor<32x196x384xf32>
    %v2677 = stablehlo.multiply %v2670, %v2676 : tensor<32x196x384xf32>
    %v2678 = stablehlo.multiply %v2663, %v2677 : tensor<32x196x384xf32>
    %v2679 = stablehlo.reduce(%v2678 init: %v2664) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2680 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2681 = stablehlo.multiply %v2679, %v2680 : tensor<384xf32>
    %v2682 = stablehlo.subtract %s2b5ng, %v2681 : tensor<384xf32>
    %v2683 = stablehlo.reshape %v2569 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2684 = stablehlo.transpose %v2683, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2685 = stablehlo.reshape %v2684 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2687 = stablehlo.reshape %v2685 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2688 = stablehlo.reduce(%v2687 init: %v2686) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2689 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2690 = stablehlo.multiply %v2688, %v2689 : tensor<384xf32>
    %v2691 = stablehlo.subtract %s2b5nbt, %v2690 : tensor<384xf32>
    %v2692 = stablehlo.reshape %v853 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2693 = stablehlo.reshape %v2612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2694 = stablehlo.transpose %v2692, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2695 = stablehlo.transpose %v2693, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2696 = stablehlo.convolution(%v2694, %v2695)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2697 = stablehlo.reshape %v2696 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2698 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2699 = stablehlo.multiply %v2697, %v2698 : tensor<384x1x7x7xf32>
    %v2700 = stablehlo.subtract %s2b5dW, %v2699 : tensor<384x1x7x7xf32>
    %v2701 = stablehlo.reshape %v2612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2703 = stablehlo.reduce(%v2701 init: %v2702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2704 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2705 = stablehlo.multiply %v2703, %v2704 : tensor<384xf32>
    %v2706 = stablehlo.subtract %s2b5db, %v2705 : tensor<384xf32>
    %v2707 = stablehlo.reshape %v2617 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2708 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2709 = stablehlo.multiply %v2707, %v2708 : tensor<32x384x14x14xf32>
    %v2710 = stablehlo.reshape %v2709 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2711 = stablehlo.reshape %v2710 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2712 = stablehlo.transpose %s2b4pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2713 = stablehlo.reverse %v2712, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2714 = stablehlo.convolution(%v2711, %v2713)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2715 = stablehlo.reshape %v2714 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2716 = stablehlo.multiply %v830, %v830 : tensor<32x301056xf32>
    %v2717 = stablehlo.multiply %v2716, %v830 : tensor<32x301056xf32>
    %v2718 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2719 = stablehlo.multiply %v2718, %v2717 : tensor<32x301056xf32>
    %v2720 = stablehlo.add %v830, %v2719 : tensor<32x301056xf32>
    %v2721 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2722 = stablehlo.multiply %v2721, %v2720 : tensor<32x301056xf32>
    %v2723 = stablehlo.tanh %v2722 : tensor<32x301056xf32>
    %v2724 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2725 = stablehlo.add %v2724, %v2723 : tensor<32x301056xf32>
    %v2726 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2727 = stablehlo.multiply %v2726, %v2725 : tensor<32x301056xf32>
    %v2728 = stablehlo.multiply %v2723, %v2723 : tensor<32x301056xf32>
    %v2729 = stablehlo.subtract %v2724, %v2728 : tensor<32x301056xf32>
    %v2730 = stablehlo.multiply %v2726, %v830 : tensor<32x301056xf32>
    %v2731 = stablehlo.multiply %v2730, %v2729 : tensor<32x301056xf32>
    %v2732 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2733 = stablehlo.multiply %v2732, %v2716 : tensor<32x301056xf32>
    %v2734 = stablehlo.add %v2724, %v2733 : tensor<32x301056xf32>
    %v2735 = stablehlo.multiply %v2721, %v2734 : tensor<32x301056xf32>
    %v2736 = stablehlo.multiply %v2731, %v2735 : tensor<32x301056xf32>
    %v2737 = stablehlo.add %v2727, %v2736 : tensor<32x301056xf32>
    %v2738 = stablehlo.multiply %v2715, %v2737 : tensor<32x301056xf32>
    %v2739 = stablehlo.reshape %v2738 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2740 = stablehlo.transpose %s2b4eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2741 = stablehlo.reverse %v2740, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2742 = stablehlo.convolution(%v2739, %v2741)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2743 = stablehlo.reshape %v2742 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2744 = stablehlo.reshape %v791 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2745 = stablehlo.transpose %v2744, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2746 = stablehlo.reshape %v2745 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2747 = stablehlo.reshape %v2743 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2748 = stablehlo.transpose %v2747, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2749 = stablehlo.reshape %v2748 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2750 = stablehlo.reshape %v2749 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2751 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2752 = stablehlo.multiply %v2750, %v2751 : tensor<32x196x384xf32>
    %v2753 = stablehlo.reshape %v2752 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2754 = stablehlo.reshape %v2753 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2755 = stablehlo.reshape %v2746 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2757 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2758 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2759 = stablehlo.reduce(%v2755 init: %v2756) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2760 = stablehlo.broadcast_in_dim %v2759, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2761 = stablehlo.divide %v2760, %v2757 : tensor<32x196x384xf32>
    %v2762 = stablehlo.subtract %v2755, %v2761 : tensor<32x196x384xf32>
    %v2763 = stablehlo.multiply %v2762, %v2762 : tensor<32x196x384xf32>
    %v2764 = stablehlo.reduce(%v2763 init: %v2756) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2765 = stablehlo.broadcast_in_dim %v2764, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2766 = stablehlo.divide %v2765, %v2757 : tensor<32x196x384xf32>
    %v2767 = stablehlo.add %v2766, %v2758 : tensor<32x196x384xf32>
    %v2768 = stablehlo.rsqrt %v2767 : tensor<32x196x384xf32>
    %v2769 = stablehlo.multiply %v2762, %v2768 : tensor<32x196x384xf32>
    %v2770 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2771 = stablehlo.multiply %v2770, %v2754 : tensor<32x196x384xf32>
    %v2772 = stablehlo.reduce(%v2771 init: %v2756) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2773 = stablehlo.broadcast_in_dim %v2772, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2774 = stablehlo.multiply %v2769, %v2771 : tensor<32x196x384xf32>
    %v2775 = stablehlo.reduce(%v2774 init: %v2756) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2776 = stablehlo.broadcast_in_dim %v2775, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2777 = stablehlo.multiply %v2771, %v2757 : tensor<32x196x384xf32>
    %v2778 = stablehlo.subtract %v2777, %v2773 : tensor<32x196x384xf32>
    %v2779 = stablehlo.multiply %v2769, %v2776 : tensor<32x196x384xf32>
    %v2780 = stablehlo.subtract %v2778, %v2779 : tensor<32x196x384xf32>
    %v2781 = stablehlo.divide %v2768, %v2757 : tensor<32x196x384xf32>
    %v2782 = stablehlo.multiply %v2781, %v2780 : tensor<32x196x384xf32>
    %v2783 = stablehlo.reshape %v2782 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2784 = stablehlo.reshape %v2783 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2785 = stablehlo.transpose %v2784, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2786 = stablehlo.reshape %v2785 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2787 = stablehlo.reshape %v2786 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2788 = stablehlo.reverse %s2b4dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2789 = stablehlo.convolution(%v2787, %v2788)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2790 = stablehlo.reshape %v2789 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2791 = stablehlo.add %v2790, %v2617 : tensor<32x75264xf32>
    %v2792 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2793 = stablehlo.reshape %v848 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2794 = stablehlo.reshape %v2617 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2795 = stablehlo.multiply %v2793, %v2794 : tensor<32x384x14x14xf32>
    %v2796 = stablehlo.reduce(%v2795 init: %v2792) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2797 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2798 = stablehlo.multiply %v2796, %v2797 : tensor<384xf32>
    %v2799 = stablehlo.subtract %s2b4lg, %v2798 : tensor<384xf32>
    %v2800 = stablehlo.reshape %v843 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2801 = stablehlo.reshape %v2710 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2802 = stablehlo.transpose %v2800, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2803 = stablehlo.transpose %v2801, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2804 = stablehlo.convolution(%v2802, %v2803)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2805 = stablehlo.transpose %v2804, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2806 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2807 = stablehlo.multiply %v2805, %v2806 : tensor<384x1536x1x1xf32>
    %v2808 = stablehlo.subtract %s2b4pW, %v2807 : tensor<384x1536x1x1xf32>
    %v2809 = stablehlo.reshape %v2710 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2811 = stablehlo.reduce(%v2809 init: %v2810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2812 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2813 = stablehlo.multiply %v2811, %v2812 : tensor<384xf32>
    %v2814 = stablehlo.subtract %s2b4pb, %v2813 : tensor<384xf32>
    %v2815 = stablehlo.reshape %v825 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2816 = stablehlo.reshape %v2738 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2817 = stablehlo.transpose %v2815, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2818 = stablehlo.transpose %v2816, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2819 = stablehlo.convolution(%v2817, %v2818)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2820 = stablehlo.transpose %v2819, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2821 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2822 = stablehlo.multiply %v2820, %v2821 : tensor<1536x384x1x1xf32>
    %v2823 = stablehlo.subtract %s2b4eW, %v2822 : tensor<1536x384x1x1xf32>
    %v2824 = stablehlo.reshape %v2738 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2826 = stablehlo.reduce(%v2824 init: %v2825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2827 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2828 = stablehlo.multiply %v2826, %v2827 : tensor<1536xf32>
    %v2829 = stablehlo.subtract %s2b4eb, %v2828 : tensor<1536xf32>
    %v2830 = stablehlo.reshape %v791 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2831 = stablehlo.transpose %v2830, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2832 = stablehlo.reshape %v2831 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2833 = stablehlo.reshape %v2743 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2834 = stablehlo.transpose %v2833, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2835 = stablehlo.reshape %v2834 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2836 = stablehlo.reshape %v2832 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2837 = stablehlo.reshape %v2835 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2839 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2840 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2841 = stablehlo.reduce(%v2836 init: %v2838) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2842 = stablehlo.broadcast_in_dim %v2841, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2843 = stablehlo.divide %v2842, %v2839 : tensor<32x196x384xf32>
    %v2844 = stablehlo.subtract %v2836, %v2843 : tensor<32x196x384xf32>
    %v2845 = stablehlo.multiply %v2844, %v2844 : tensor<32x196x384xf32>
    %v2846 = stablehlo.reduce(%v2845 init: %v2838) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2847 = stablehlo.broadcast_in_dim %v2846, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2848 = stablehlo.divide %v2847, %v2839 : tensor<32x196x384xf32>
    %v2849 = stablehlo.add %v2848, %v2840 : tensor<32x196x384xf32>
    %v2850 = stablehlo.rsqrt %v2849 : tensor<32x196x384xf32>
    %v2851 = stablehlo.multiply %v2844, %v2850 : tensor<32x196x384xf32>
    %v2852 = stablehlo.multiply %v2837, %v2851 : tensor<32x196x384xf32>
    %v2853 = stablehlo.reduce(%v2852 init: %v2838) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2854 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2855 = stablehlo.multiply %v2853, %v2854 : tensor<384xf32>
    %v2856 = stablehlo.subtract %s2b4ng, %v2855 : tensor<384xf32>
    %v2857 = stablehlo.reshape %v2743 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2858 = stablehlo.transpose %v2857, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2860 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2861 = stablehlo.reshape %v2859 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2862 = stablehlo.reduce(%v2861 init: %v2860) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2863 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2864 = stablehlo.multiply %v2862, %v2863 : tensor<384xf32>
    %v2865 = stablehlo.subtract %s2b4nbt, %v2864 : tensor<384xf32>
    %v2866 = stablehlo.reshape %v786 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2867 = stablehlo.reshape %v2786 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2868 = stablehlo.transpose %v2866, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2869 = stablehlo.transpose %v2867, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2870 = stablehlo.convolution(%v2868, %v2869)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2871 = stablehlo.reshape %v2870 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2872 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2873 = stablehlo.multiply %v2871, %v2872 : tensor<384x1x7x7xf32>
    %v2874 = stablehlo.subtract %s2b4dW, %v2873 : tensor<384x1x7x7xf32>
    %v2875 = stablehlo.reshape %v2786 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2876 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2877 = stablehlo.reduce(%v2875 init: %v2876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2878 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2879 = stablehlo.multiply %v2877, %v2878 : tensor<384xf32>
    %v2880 = stablehlo.subtract %s2b4db, %v2879 : tensor<384xf32>
    %v2881 = stablehlo.reshape %v2791 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2882 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2883 = stablehlo.multiply %v2881, %v2882 : tensor<32x384x14x14xf32>
    %v2884 = stablehlo.reshape %v2883 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2885 = stablehlo.reshape %v2884 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2886 = stablehlo.transpose %s2b3pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2887 = stablehlo.reverse %v2886, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2888 = stablehlo.convolution(%v2885, %v2887)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2889 = stablehlo.reshape %v2888 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2890 = stablehlo.multiply %v763, %v763 : tensor<32x301056xf32>
    %v2891 = stablehlo.multiply %v2890, %v763 : tensor<32x301056xf32>
    %v2892 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2893 = stablehlo.multiply %v2892, %v2891 : tensor<32x301056xf32>
    %v2894 = stablehlo.add %v763, %v2893 : tensor<32x301056xf32>
    %v2895 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2896 = stablehlo.multiply %v2895, %v2894 : tensor<32x301056xf32>
    %v2897 = stablehlo.tanh %v2896 : tensor<32x301056xf32>
    %v2898 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2899 = stablehlo.add %v2898, %v2897 : tensor<32x301056xf32>
    %v2900 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2901 = stablehlo.multiply %v2900, %v2899 : tensor<32x301056xf32>
    %v2902 = stablehlo.multiply %v2897, %v2897 : tensor<32x301056xf32>
    %v2903 = stablehlo.subtract %v2898, %v2902 : tensor<32x301056xf32>
    %v2904 = stablehlo.multiply %v2900, %v763 : tensor<32x301056xf32>
    %v2905 = stablehlo.multiply %v2904, %v2903 : tensor<32x301056xf32>
    %v2906 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2907 = stablehlo.multiply %v2906, %v2890 : tensor<32x301056xf32>
    %v2908 = stablehlo.add %v2898, %v2907 : tensor<32x301056xf32>
    %v2909 = stablehlo.multiply %v2895, %v2908 : tensor<32x301056xf32>
    %v2910 = stablehlo.multiply %v2905, %v2909 : tensor<32x301056xf32>
    %v2911 = stablehlo.add %v2901, %v2910 : tensor<32x301056xf32>
    %v2912 = stablehlo.multiply %v2889, %v2911 : tensor<32x301056xf32>
    %v2913 = stablehlo.reshape %v2912 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2914 = stablehlo.transpose %s2b3eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2915 = stablehlo.reverse %v2914, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2916 = stablehlo.convolution(%v2913, %v2915)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2917 = stablehlo.reshape %v2916 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2918 = stablehlo.reshape %v724 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2919 = stablehlo.transpose %v2918, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2920 = stablehlo.reshape %v2919 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2921 = stablehlo.reshape %v2917 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2922 = stablehlo.transpose %v2921, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2923 = stablehlo.reshape %v2922 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2924 = stablehlo.reshape %v2923 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2925 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2926 = stablehlo.multiply %v2924, %v2925 : tensor<32x196x384xf32>
    %v2927 = stablehlo.reshape %v2926 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2928 = stablehlo.reshape %v2927 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2929 = stablehlo.reshape %v2920 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2931 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2932 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2933 = stablehlo.reduce(%v2929 init: %v2930) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2934 = stablehlo.broadcast_in_dim %v2933, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2935 = stablehlo.divide %v2934, %v2931 : tensor<32x196x384xf32>
    %v2936 = stablehlo.subtract %v2929, %v2935 : tensor<32x196x384xf32>
    %v2937 = stablehlo.multiply %v2936, %v2936 : tensor<32x196x384xf32>
    %v2938 = stablehlo.reduce(%v2937 init: %v2930) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2939 = stablehlo.broadcast_in_dim %v2938, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2940 = stablehlo.divide %v2939, %v2931 : tensor<32x196x384xf32>
    %v2941 = stablehlo.add %v2940, %v2932 : tensor<32x196x384xf32>
    %v2942 = stablehlo.rsqrt %v2941 : tensor<32x196x384xf32>
    %v2943 = stablehlo.multiply %v2936, %v2942 : tensor<32x196x384xf32>
    %v2944 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2945 = stablehlo.multiply %v2944, %v2928 : tensor<32x196x384xf32>
    %v2946 = stablehlo.reduce(%v2945 init: %v2930) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2947 = stablehlo.broadcast_in_dim %v2946, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2948 = stablehlo.multiply %v2943, %v2945 : tensor<32x196x384xf32>
    %v2949 = stablehlo.reduce(%v2948 init: %v2930) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2950 = stablehlo.broadcast_in_dim %v2949, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2951 = stablehlo.multiply %v2945, %v2931 : tensor<32x196x384xf32>
    %v2952 = stablehlo.subtract %v2951, %v2947 : tensor<32x196x384xf32>
    %v2953 = stablehlo.multiply %v2943, %v2950 : tensor<32x196x384xf32>
    %v2954 = stablehlo.subtract %v2952, %v2953 : tensor<32x196x384xf32>
    %v2955 = stablehlo.divide %v2942, %v2931 : tensor<32x196x384xf32>
    %v2956 = stablehlo.multiply %v2955, %v2954 : tensor<32x196x384xf32>
    %v2957 = stablehlo.reshape %v2956 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2958 = stablehlo.reshape %v2957 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2959 = stablehlo.transpose %v2958, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2960 = stablehlo.reshape %v2959 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2961 = stablehlo.reshape %v2960 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2962 = stablehlo.reverse %s2b3dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2963 = stablehlo.convolution(%v2961, %v2962)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2964 = stablehlo.reshape %v2963 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2965 = stablehlo.add %v2964, %v2791 : tensor<32x75264xf32>
    %v2966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2967 = stablehlo.reshape %v781 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2968 = stablehlo.reshape %v2791 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2969 = stablehlo.multiply %v2967, %v2968 : tensor<32x384x14x14xf32>
    %v2970 = stablehlo.reduce(%v2969 init: %v2966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2971 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2972 = stablehlo.multiply %v2970, %v2971 : tensor<384xf32>
    %v2973 = stablehlo.subtract %s2b3lg, %v2972 : tensor<384xf32>
    %v2974 = stablehlo.reshape %v776 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2975 = stablehlo.reshape %v2884 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2976 = stablehlo.transpose %v2974, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2977 = stablehlo.transpose %v2975, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2978 = stablehlo.convolution(%v2976, %v2977)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2979 = stablehlo.transpose %v2978, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2980 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2981 = stablehlo.multiply %v2979, %v2980 : tensor<384x1536x1x1xf32>
    %v2982 = stablehlo.subtract %s2b3pW, %v2981 : tensor<384x1536x1x1xf32>
    %v2983 = stablehlo.reshape %v2884 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2984 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2985 = stablehlo.reduce(%v2983 init: %v2984) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2986 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2987 = stablehlo.multiply %v2985, %v2986 : tensor<384xf32>
    %v2988 = stablehlo.subtract %s2b3pb, %v2987 : tensor<384xf32>
    %v2989 = stablehlo.reshape %v758 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2990 = stablehlo.reshape %v2912 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2991 = stablehlo.transpose %v2989, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2992 = stablehlo.transpose %v2990, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2993 = stablehlo.convolution(%v2991, %v2992)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2994 = stablehlo.transpose %v2993, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2995 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2996 = stablehlo.multiply %v2994, %v2995 : tensor<1536x384x1x1xf32>
    %v2997 = stablehlo.subtract %s2b3eW, %v2996 : tensor<1536x384x1x1xf32>
    %v2998 = stablehlo.reshape %v2912 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2999 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3000 = stablehlo.reduce(%v2998 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3001 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3002 = stablehlo.multiply %v3000, %v3001 : tensor<1536xf32>
    %v3003 = stablehlo.subtract %s2b3eb, %v3002 : tensor<1536xf32>
    %v3004 = stablehlo.reshape %v724 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3005 = stablehlo.transpose %v3004, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3006 = stablehlo.reshape %v3005 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3007 = stablehlo.reshape %v2917 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3008 = stablehlo.transpose %v3007, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3009 = stablehlo.reshape %v3008 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3010 = stablehlo.reshape %v3006 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3011 = stablehlo.reshape %v3009 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3012 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3013 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3014 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3015 = stablehlo.reduce(%v3010 init: %v3012) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3016 = stablehlo.broadcast_in_dim %v3015, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3017 = stablehlo.divide %v3016, %v3013 : tensor<32x196x384xf32>
    %v3018 = stablehlo.subtract %v3010, %v3017 : tensor<32x196x384xf32>
    %v3019 = stablehlo.multiply %v3018, %v3018 : tensor<32x196x384xf32>
    %v3020 = stablehlo.reduce(%v3019 init: %v3012) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3021 = stablehlo.broadcast_in_dim %v3020, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3022 = stablehlo.divide %v3021, %v3013 : tensor<32x196x384xf32>
    %v3023 = stablehlo.add %v3022, %v3014 : tensor<32x196x384xf32>
    %v3024 = stablehlo.rsqrt %v3023 : tensor<32x196x384xf32>
    %v3025 = stablehlo.multiply %v3018, %v3024 : tensor<32x196x384xf32>
    %v3026 = stablehlo.multiply %v3011, %v3025 : tensor<32x196x384xf32>
    %v3027 = stablehlo.reduce(%v3026 init: %v3012) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3028 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3029 = stablehlo.multiply %v3027, %v3028 : tensor<384xf32>
    %v3030 = stablehlo.subtract %s2b3ng, %v3029 : tensor<384xf32>
    %v3031 = stablehlo.reshape %v2917 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3032 = stablehlo.transpose %v3031, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3033 = stablehlo.reshape %v3032 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3035 = stablehlo.reshape %v3033 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3036 = stablehlo.reduce(%v3035 init: %v3034) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3037 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3038 = stablehlo.multiply %v3036, %v3037 : tensor<384xf32>
    %v3039 = stablehlo.subtract %s2b3nbt, %v3038 : tensor<384xf32>
    %v3040 = stablehlo.reshape %v719 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3041 = stablehlo.reshape %v2960 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3042 = stablehlo.transpose %v3040, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3043 = stablehlo.transpose %v3041, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3044 = stablehlo.convolution(%v3042, %v3043)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3045 = stablehlo.reshape %v3044 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3046 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3047 = stablehlo.multiply %v3045, %v3046 : tensor<384x1x7x7xf32>
    %v3048 = stablehlo.subtract %s2b3dW, %v3047 : tensor<384x1x7x7xf32>
    %v3049 = stablehlo.reshape %v2960 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3051 = stablehlo.reduce(%v3049 init: %v3050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3052 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3053 = stablehlo.multiply %v3051, %v3052 : tensor<384xf32>
    %v3054 = stablehlo.subtract %s2b3db, %v3053 : tensor<384xf32>
    %v3055 = stablehlo.reshape %v2965 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3056 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3057 = stablehlo.multiply %v3055, %v3056 : tensor<32x384x14x14xf32>
    %v3058 = stablehlo.reshape %v3057 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3059 = stablehlo.reshape %v3058 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3060 = stablehlo.transpose %s2b2pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3061 = stablehlo.reverse %v3060, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3062 = stablehlo.convolution(%v3059, %v3061)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3063 = stablehlo.reshape %v3062 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3064 = stablehlo.multiply %v696, %v696 : tensor<32x301056xf32>
    %v3065 = stablehlo.multiply %v3064, %v696 : tensor<32x301056xf32>
    %v3066 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v3067 = stablehlo.multiply %v3066, %v3065 : tensor<32x301056xf32>
    %v3068 = stablehlo.add %v696, %v3067 : tensor<32x301056xf32>
    %v3069 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v3070 = stablehlo.multiply %v3069, %v3068 : tensor<32x301056xf32>
    %v3071 = stablehlo.tanh %v3070 : tensor<32x301056xf32>
    %v3072 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v3073 = stablehlo.add %v3072, %v3071 : tensor<32x301056xf32>
    %v3074 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v3075 = stablehlo.multiply %v3074, %v3073 : tensor<32x301056xf32>
    %v3076 = stablehlo.multiply %v3071, %v3071 : tensor<32x301056xf32>
    %v3077 = stablehlo.subtract %v3072, %v3076 : tensor<32x301056xf32>
    %v3078 = stablehlo.multiply %v3074, %v696 : tensor<32x301056xf32>
    %v3079 = stablehlo.multiply %v3078, %v3077 : tensor<32x301056xf32>
    %v3080 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v3081 = stablehlo.multiply %v3080, %v3064 : tensor<32x301056xf32>
    %v3082 = stablehlo.add %v3072, %v3081 : tensor<32x301056xf32>
    %v3083 = stablehlo.multiply %v3069, %v3082 : tensor<32x301056xf32>
    %v3084 = stablehlo.multiply %v3079, %v3083 : tensor<32x301056xf32>
    %v3085 = stablehlo.add %v3075, %v3084 : tensor<32x301056xf32>
    %v3086 = stablehlo.multiply %v3063, %v3085 : tensor<32x301056xf32>
    %v3087 = stablehlo.reshape %v3086 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3088 = stablehlo.transpose %s2b2eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3089 = stablehlo.reverse %v3088, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3090 = stablehlo.convolution(%v3087, %v3089)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3091 = stablehlo.reshape %v3090 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3092 = stablehlo.reshape %v657 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3093 = stablehlo.transpose %v3092, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3094 = stablehlo.reshape %v3093 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3095 = stablehlo.reshape %v3091 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3096 = stablehlo.transpose %v3095, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3097 = stablehlo.reshape %v3096 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3098 = stablehlo.reshape %v3097 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3099 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3100 = stablehlo.multiply %v3098, %v3099 : tensor<32x196x384xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3102 = stablehlo.reshape %v3101 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3103 = stablehlo.reshape %v3094 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3104 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3105 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3106 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3107 = stablehlo.reduce(%v3103 init: %v3104) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3108 = stablehlo.broadcast_in_dim %v3107, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3109 = stablehlo.divide %v3108, %v3105 : tensor<32x196x384xf32>
    %v3110 = stablehlo.subtract %v3103, %v3109 : tensor<32x196x384xf32>
    %v3111 = stablehlo.multiply %v3110, %v3110 : tensor<32x196x384xf32>
    %v3112 = stablehlo.reduce(%v3111 init: %v3104) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3113 = stablehlo.broadcast_in_dim %v3112, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3114 = stablehlo.divide %v3113, %v3105 : tensor<32x196x384xf32>
    %v3115 = stablehlo.add %v3114, %v3106 : tensor<32x196x384xf32>
    %v3116 = stablehlo.rsqrt %v3115 : tensor<32x196x384xf32>
    %v3117 = stablehlo.multiply %v3110, %v3116 : tensor<32x196x384xf32>
    %v3118 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3119 = stablehlo.multiply %v3118, %v3102 : tensor<32x196x384xf32>
    %v3120 = stablehlo.reduce(%v3119 init: %v3104) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3121 = stablehlo.broadcast_in_dim %v3120, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3122 = stablehlo.multiply %v3117, %v3119 : tensor<32x196x384xf32>
    %v3123 = stablehlo.reduce(%v3122 init: %v3104) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3124 = stablehlo.broadcast_in_dim %v3123, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3125 = stablehlo.multiply %v3119, %v3105 : tensor<32x196x384xf32>
    %v3126 = stablehlo.subtract %v3125, %v3121 : tensor<32x196x384xf32>
    %v3127 = stablehlo.multiply %v3117, %v3124 : tensor<32x196x384xf32>
    %v3128 = stablehlo.subtract %v3126, %v3127 : tensor<32x196x384xf32>
    %v3129 = stablehlo.divide %v3116, %v3105 : tensor<32x196x384xf32>
    %v3130 = stablehlo.multiply %v3129, %v3128 : tensor<32x196x384xf32>
    %v3131 = stablehlo.reshape %v3130 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3132 = stablehlo.reshape %v3131 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3133 = stablehlo.transpose %v3132, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3134 = stablehlo.reshape %v3133 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3135 = stablehlo.reshape %v3134 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3136 = stablehlo.reverse %s2b2dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3137 = stablehlo.convolution(%v3135, %v3136)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3138 = stablehlo.reshape %v3137 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3139 = stablehlo.add %v3138, %v2965 : tensor<32x75264xf32>
    %v3140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3141 = stablehlo.reshape %v714 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3142 = stablehlo.reshape %v2965 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3143 = stablehlo.multiply %v3141, %v3142 : tensor<32x384x14x14xf32>
    %v3144 = stablehlo.reduce(%v3143 init: %v3140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3145 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3146 = stablehlo.multiply %v3144, %v3145 : tensor<384xf32>
    %v3147 = stablehlo.subtract %s2b2lg, %v3146 : tensor<384xf32>
    %v3148 = stablehlo.reshape %v709 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3149 = stablehlo.reshape %v3058 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3150 = stablehlo.transpose %v3148, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3151 = stablehlo.transpose %v3149, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3152 = stablehlo.convolution(%v3150, %v3151)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3153 = stablehlo.transpose %v3152, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3154 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v3155 = stablehlo.multiply %v3153, %v3154 : tensor<384x1536x1x1xf32>
    %v3156 = stablehlo.subtract %s2b2pW, %v3155 : tensor<384x1536x1x1xf32>
    %v3157 = stablehlo.reshape %v3058 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3158 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3159 = stablehlo.reduce(%v3157 init: %v3158) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3160 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3161 = stablehlo.multiply %v3159, %v3160 : tensor<384xf32>
    %v3162 = stablehlo.subtract %s2b2pb, %v3161 : tensor<384xf32>
    %v3163 = stablehlo.reshape %v691 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3164 = stablehlo.reshape %v3086 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3165 = stablehlo.transpose %v3163, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3166 = stablehlo.transpose %v3164, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3167 = stablehlo.convolution(%v3165, %v3166)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3168 = stablehlo.transpose %v3167, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3169 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v3170 = stablehlo.multiply %v3168, %v3169 : tensor<1536x384x1x1xf32>
    %v3171 = stablehlo.subtract %s2b2eW, %v3170 : tensor<1536x384x1x1xf32>
    %v3172 = stablehlo.reshape %v3086 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3174 = stablehlo.reduce(%v3172 init: %v3173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3175 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3176 = stablehlo.multiply %v3174, %v3175 : tensor<1536xf32>
    %v3177 = stablehlo.subtract %s2b2eb, %v3176 : tensor<1536xf32>
    %v3178 = stablehlo.reshape %v657 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3179 = stablehlo.transpose %v3178, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3180 = stablehlo.reshape %v3179 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3181 = stablehlo.reshape %v3091 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3182 = stablehlo.transpose %v3181, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3183 = stablehlo.reshape %v3182 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3184 = stablehlo.reshape %v3180 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3185 = stablehlo.reshape %v3183 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3187 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3188 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3189 = stablehlo.reduce(%v3184 init: %v3186) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3190 = stablehlo.broadcast_in_dim %v3189, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3191 = stablehlo.divide %v3190, %v3187 : tensor<32x196x384xf32>
    %v3192 = stablehlo.subtract %v3184, %v3191 : tensor<32x196x384xf32>
    %v3193 = stablehlo.multiply %v3192, %v3192 : tensor<32x196x384xf32>
    %v3194 = stablehlo.reduce(%v3193 init: %v3186) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3195 = stablehlo.broadcast_in_dim %v3194, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3196 = stablehlo.divide %v3195, %v3187 : tensor<32x196x384xf32>
    %v3197 = stablehlo.add %v3196, %v3188 : tensor<32x196x384xf32>
    %v3198 = stablehlo.rsqrt %v3197 : tensor<32x196x384xf32>
    %v3199 = stablehlo.multiply %v3192, %v3198 : tensor<32x196x384xf32>
    %v3200 = stablehlo.multiply %v3185, %v3199 : tensor<32x196x384xf32>
    %v3201 = stablehlo.reduce(%v3200 init: %v3186) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3202 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3203 = stablehlo.multiply %v3201, %v3202 : tensor<384xf32>
    %v3204 = stablehlo.subtract %s2b2ng, %v3203 : tensor<384xf32>
    %v3205 = stablehlo.reshape %v3091 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3206 = stablehlo.transpose %v3205, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3207 = stablehlo.reshape %v3206 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3208 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3209 = stablehlo.reshape %v3207 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3210 = stablehlo.reduce(%v3209 init: %v3208) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3211 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3212 = stablehlo.multiply %v3210, %v3211 : tensor<384xf32>
    %v3213 = stablehlo.subtract %s2b2nbt, %v3212 : tensor<384xf32>
    %v3214 = stablehlo.reshape %v652 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3215 = stablehlo.reshape %v3134 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3216 = stablehlo.transpose %v3214, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3217 = stablehlo.transpose %v3215, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3218 = stablehlo.convolution(%v3216, %v3217)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3219 = stablehlo.reshape %v3218 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3220 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3221 = stablehlo.multiply %v3219, %v3220 : tensor<384x1x7x7xf32>
    %v3222 = stablehlo.subtract %s2b2dW, %v3221 : tensor<384x1x7x7xf32>
    %v3223 = stablehlo.reshape %v3134 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3225 = stablehlo.reduce(%v3223 init: %v3224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3226 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3227 = stablehlo.multiply %v3225, %v3226 : tensor<384xf32>
    %v3228 = stablehlo.subtract %s2b2db, %v3227 : tensor<384xf32>
    %v3229 = stablehlo.reshape %v3139 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3230 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3231 = stablehlo.multiply %v3229, %v3230 : tensor<32x384x14x14xf32>
    %v3232 = stablehlo.reshape %v3231 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3233 = stablehlo.reshape %v3232 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3234 = stablehlo.transpose %s2b1pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3235 = stablehlo.reverse %v3234, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3236 = stablehlo.convolution(%v3233, %v3235)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3237 = stablehlo.reshape %v3236 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3238 = stablehlo.multiply %v629, %v629 : tensor<32x301056xf32>
    %v3239 = stablehlo.multiply %v3238, %v629 : tensor<32x301056xf32>
    %v3240 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v3241 = stablehlo.multiply %v3240, %v3239 : tensor<32x301056xf32>
    %v3242 = stablehlo.add %v629, %v3241 : tensor<32x301056xf32>
    %v3243 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v3244 = stablehlo.multiply %v3243, %v3242 : tensor<32x301056xf32>
    %v3245 = stablehlo.tanh %v3244 : tensor<32x301056xf32>
    %v3246 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v3247 = stablehlo.add %v3246, %v3245 : tensor<32x301056xf32>
    %v3248 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v3249 = stablehlo.multiply %v3248, %v3247 : tensor<32x301056xf32>
    %v3250 = stablehlo.multiply %v3245, %v3245 : tensor<32x301056xf32>
    %v3251 = stablehlo.subtract %v3246, %v3250 : tensor<32x301056xf32>
    %v3252 = stablehlo.multiply %v3248, %v629 : tensor<32x301056xf32>
    %v3253 = stablehlo.multiply %v3252, %v3251 : tensor<32x301056xf32>
    %v3254 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v3255 = stablehlo.multiply %v3254, %v3238 : tensor<32x301056xf32>
    %v3256 = stablehlo.add %v3246, %v3255 : tensor<32x301056xf32>
    %v3257 = stablehlo.multiply %v3243, %v3256 : tensor<32x301056xf32>
    %v3258 = stablehlo.multiply %v3253, %v3257 : tensor<32x301056xf32>
    %v3259 = stablehlo.add %v3249, %v3258 : tensor<32x301056xf32>
    %v3260 = stablehlo.multiply %v3237, %v3259 : tensor<32x301056xf32>
    %v3261 = stablehlo.reshape %v3260 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3262 = stablehlo.transpose %s2b1eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3263 = stablehlo.reverse %v3262, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3264 = stablehlo.convolution(%v3261, %v3263)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3265 = stablehlo.reshape %v3264 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3266 = stablehlo.reshape %v590 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3267 = stablehlo.transpose %v3266, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3268 = stablehlo.reshape %v3267 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3269 = stablehlo.reshape %v3265 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3270 = stablehlo.transpose %v3269, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3271 = stablehlo.reshape %v3270 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3272 = stablehlo.reshape %v3271 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3273 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3274 = stablehlo.multiply %v3272, %v3273 : tensor<32x196x384xf32>
    %v3275 = stablehlo.reshape %v3274 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3276 = stablehlo.reshape %v3275 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3277 = stablehlo.reshape %v3268 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3279 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3280 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3281 = stablehlo.reduce(%v3277 init: %v3278) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3282 = stablehlo.broadcast_in_dim %v3281, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3283 = stablehlo.divide %v3282, %v3279 : tensor<32x196x384xf32>
    %v3284 = stablehlo.subtract %v3277, %v3283 : tensor<32x196x384xf32>
    %v3285 = stablehlo.multiply %v3284, %v3284 : tensor<32x196x384xf32>
    %v3286 = stablehlo.reduce(%v3285 init: %v3278) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3287 = stablehlo.broadcast_in_dim %v3286, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3288 = stablehlo.divide %v3287, %v3279 : tensor<32x196x384xf32>
    %v3289 = stablehlo.add %v3288, %v3280 : tensor<32x196x384xf32>
    %v3290 = stablehlo.rsqrt %v3289 : tensor<32x196x384xf32>
    %v3291 = stablehlo.multiply %v3284, %v3290 : tensor<32x196x384xf32>
    %v3292 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3293 = stablehlo.multiply %v3292, %v3276 : tensor<32x196x384xf32>
    %v3294 = stablehlo.reduce(%v3293 init: %v3278) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3295 = stablehlo.broadcast_in_dim %v3294, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3296 = stablehlo.multiply %v3291, %v3293 : tensor<32x196x384xf32>
    %v3297 = stablehlo.reduce(%v3296 init: %v3278) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3298 = stablehlo.broadcast_in_dim %v3297, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3299 = stablehlo.multiply %v3293, %v3279 : tensor<32x196x384xf32>
    %v3300 = stablehlo.subtract %v3299, %v3295 : tensor<32x196x384xf32>
    %v3301 = stablehlo.multiply %v3291, %v3298 : tensor<32x196x384xf32>
    %v3302 = stablehlo.subtract %v3300, %v3301 : tensor<32x196x384xf32>
    %v3303 = stablehlo.divide %v3290, %v3279 : tensor<32x196x384xf32>
    %v3304 = stablehlo.multiply %v3303, %v3302 : tensor<32x196x384xf32>
    %v3305 = stablehlo.reshape %v3304 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3306 = stablehlo.reshape %v3305 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3307 = stablehlo.transpose %v3306, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3308 = stablehlo.reshape %v3307 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3309 = stablehlo.reshape %v3308 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3310 = stablehlo.reverse %s2b1dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3311 = stablehlo.convolution(%v3309, %v3310)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3312 = stablehlo.reshape %v3311 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3313 = stablehlo.add %v3312, %v3139 : tensor<32x75264xf32>
    %v3314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3315 = stablehlo.reshape %v647 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3316 = stablehlo.reshape %v3139 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3317 = stablehlo.multiply %v3315, %v3316 : tensor<32x384x14x14xf32>
    %v3318 = stablehlo.reduce(%v3317 init: %v3314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3319 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3320 = stablehlo.multiply %v3318, %v3319 : tensor<384xf32>
    %v3321 = stablehlo.subtract %s2b1lg, %v3320 : tensor<384xf32>
    %v3322 = stablehlo.reshape %v642 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3323 = stablehlo.reshape %v3232 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3324 = stablehlo.transpose %v3322, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3325 = stablehlo.transpose %v3323, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3326 = stablehlo.convolution(%v3324, %v3325)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3327 = stablehlo.transpose %v3326, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3328 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v3329 = stablehlo.multiply %v3327, %v3328 : tensor<384x1536x1x1xf32>
    %v3330 = stablehlo.subtract %s2b1pW, %v3329 : tensor<384x1536x1x1xf32>
    %v3331 = stablehlo.reshape %v3232 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3333 = stablehlo.reduce(%v3331 init: %v3332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3334 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3335 = stablehlo.multiply %v3333, %v3334 : tensor<384xf32>
    %v3336 = stablehlo.subtract %s2b1pb, %v3335 : tensor<384xf32>
    %v3337 = stablehlo.reshape %v624 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3338 = stablehlo.reshape %v3260 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3339 = stablehlo.transpose %v3337, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3340 = stablehlo.transpose %v3338, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3341 = stablehlo.convolution(%v3339, %v3340)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3342 = stablehlo.transpose %v3341, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3343 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v3344 = stablehlo.multiply %v3342, %v3343 : tensor<1536x384x1x1xf32>
    %v3345 = stablehlo.subtract %s2b1eW, %v3344 : tensor<1536x384x1x1xf32>
    %v3346 = stablehlo.reshape %v3260 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3347 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3348 = stablehlo.reduce(%v3346 init: %v3347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3349 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3350 = stablehlo.multiply %v3348, %v3349 : tensor<1536xf32>
    %v3351 = stablehlo.subtract %s2b1eb, %v3350 : tensor<1536xf32>
    %v3352 = stablehlo.reshape %v590 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3353 = stablehlo.transpose %v3352, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3354 = stablehlo.reshape %v3353 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3355 = stablehlo.reshape %v3265 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3356 = stablehlo.transpose %v3355, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3357 = stablehlo.reshape %v3356 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3358 = stablehlo.reshape %v3354 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3359 = stablehlo.reshape %v3357 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3361 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3362 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3363 = stablehlo.reduce(%v3358 init: %v3360) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3364 = stablehlo.broadcast_in_dim %v3363, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3365 = stablehlo.divide %v3364, %v3361 : tensor<32x196x384xf32>
    %v3366 = stablehlo.subtract %v3358, %v3365 : tensor<32x196x384xf32>
    %v3367 = stablehlo.multiply %v3366, %v3366 : tensor<32x196x384xf32>
    %v3368 = stablehlo.reduce(%v3367 init: %v3360) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3369 = stablehlo.broadcast_in_dim %v3368, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3370 = stablehlo.divide %v3369, %v3361 : tensor<32x196x384xf32>
    %v3371 = stablehlo.add %v3370, %v3362 : tensor<32x196x384xf32>
    %v3372 = stablehlo.rsqrt %v3371 : tensor<32x196x384xf32>
    %v3373 = stablehlo.multiply %v3366, %v3372 : tensor<32x196x384xf32>
    %v3374 = stablehlo.multiply %v3359, %v3373 : tensor<32x196x384xf32>
    %v3375 = stablehlo.reduce(%v3374 init: %v3360) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3376 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3377 = stablehlo.multiply %v3375, %v3376 : tensor<384xf32>
    %v3378 = stablehlo.subtract %s2b1ng, %v3377 : tensor<384xf32>
    %v3379 = stablehlo.reshape %v3265 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3380 = stablehlo.transpose %v3379, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3381 = stablehlo.reshape %v3380 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3383 = stablehlo.reshape %v3381 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3384 = stablehlo.reduce(%v3383 init: %v3382) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3385 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3386 = stablehlo.multiply %v3384, %v3385 : tensor<384xf32>
    %v3387 = stablehlo.subtract %s2b1nbt, %v3386 : tensor<384xf32>
    %v3388 = stablehlo.reshape %v585 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3389 = stablehlo.reshape %v3308 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3390 = stablehlo.transpose %v3388, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3391 = stablehlo.transpose %v3389, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3392 = stablehlo.convolution(%v3390, %v3391)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3393 = stablehlo.reshape %v3392 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3394 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3395 = stablehlo.multiply %v3393, %v3394 : tensor<384x1x7x7xf32>
    %v3396 = stablehlo.subtract %s2b1dW, %v3395 : tensor<384x1x7x7xf32>
    %v3397 = stablehlo.reshape %v3308 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3399 = stablehlo.reduce(%v3397 init: %v3398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3400 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3401 = stablehlo.multiply %v3399, %v3400 : tensor<384xf32>
    %v3402 = stablehlo.subtract %s2b1db, %v3401 : tensor<384xf32>
    %v3403 = stablehlo.reshape %v3313 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3404 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3405 = stablehlo.multiply %v3403, %v3404 : tensor<32x384x14x14xf32>
    %v3406 = stablehlo.reshape %v3405 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3407 = stablehlo.reshape %v3406 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3408 = stablehlo.transpose %s2b0pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3409 = stablehlo.reverse %v3408, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3410 = stablehlo.convolution(%v3407, %v3409)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3411 = stablehlo.reshape %v3410 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3412 = stablehlo.multiply %v562, %v562 : tensor<32x301056xf32>
    %v3413 = stablehlo.multiply %v3412, %v562 : tensor<32x301056xf32>
    %v3414 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v3415 = stablehlo.multiply %v3414, %v3413 : tensor<32x301056xf32>
    %v3416 = stablehlo.add %v562, %v3415 : tensor<32x301056xf32>
    %v3417 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v3418 = stablehlo.multiply %v3417, %v3416 : tensor<32x301056xf32>
    %v3419 = stablehlo.tanh %v3418 : tensor<32x301056xf32>
    %v3420 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v3421 = stablehlo.add %v3420, %v3419 : tensor<32x301056xf32>
    %v3422 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v3423 = stablehlo.multiply %v3422, %v3421 : tensor<32x301056xf32>
    %v3424 = stablehlo.multiply %v3419, %v3419 : tensor<32x301056xf32>
    %v3425 = stablehlo.subtract %v3420, %v3424 : tensor<32x301056xf32>
    %v3426 = stablehlo.multiply %v3422, %v562 : tensor<32x301056xf32>
    %v3427 = stablehlo.multiply %v3426, %v3425 : tensor<32x301056xf32>
    %v3428 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v3429 = stablehlo.multiply %v3428, %v3412 : tensor<32x301056xf32>
    %v3430 = stablehlo.add %v3420, %v3429 : tensor<32x301056xf32>
    %v3431 = stablehlo.multiply %v3417, %v3430 : tensor<32x301056xf32>
    %v3432 = stablehlo.multiply %v3427, %v3431 : tensor<32x301056xf32>
    %v3433 = stablehlo.add %v3423, %v3432 : tensor<32x301056xf32>
    %v3434 = stablehlo.multiply %v3411, %v3433 : tensor<32x301056xf32>
    %v3435 = stablehlo.reshape %v3434 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3436 = stablehlo.transpose %s2b0eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3437 = stablehlo.reverse %v3436, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3438 = stablehlo.convolution(%v3435, %v3437)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3439 = stablehlo.reshape %v3438 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3440 = stablehlo.reshape %v523 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3441 = stablehlo.transpose %v3440, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3442 = stablehlo.reshape %v3441 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3443 = stablehlo.reshape %v3439 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3444 = stablehlo.transpose %v3443, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3445 = stablehlo.reshape %v3444 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3446 = stablehlo.reshape %v3445 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3447 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3448 = stablehlo.multiply %v3446, %v3447 : tensor<32x196x384xf32>
    %v3449 = stablehlo.reshape %v3448 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3450 = stablehlo.reshape %v3449 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3451 = stablehlo.reshape %v3442 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3453 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3454 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3455 = stablehlo.reduce(%v3451 init: %v3452) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3456 = stablehlo.broadcast_in_dim %v3455, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3457 = stablehlo.divide %v3456, %v3453 : tensor<32x196x384xf32>
    %v3458 = stablehlo.subtract %v3451, %v3457 : tensor<32x196x384xf32>
    %v3459 = stablehlo.multiply %v3458, %v3458 : tensor<32x196x384xf32>
    %v3460 = stablehlo.reduce(%v3459 init: %v3452) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3461 = stablehlo.broadcast_in_dim %v3460, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3462 = stablehlo.divide %v3461, %v3453 : tensor<32x196x384xf32>
    %v3463 = stablehlo.add %v3462, %v3454 : tensor<32x196x384xf32>
    %v3464 = stablehlo.rsqrt %v3463 : tensor<32x196x384xf32>
    %v3465 = stablehlo.multiply %v3458, %v3464 : tensor<32x196x384xf32>
    %v3466 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3467 = stablehlo.multiply %v3466, %v3450 : tensor<32x196x384xf32>
    %v3468 = stablehlo.reduce(%v3467 init: %v3452) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3469 = stablehlo.broadcast_in_dim %v3468, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3470 = stablehlo.multiply %v3465, %v3467 : tensor<32x196x384xf32>
    %v3471 = stablehlo.reduce(%v3470 init: %v3452) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3472 = stablehlo.broadcast_in_dim %v3471, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3473 = stablehlo.multiply %v3467, %v3453 : tensor<32x196x384xf32>
    %v3474 = stablehlo.subtract %v3473, %v3469 : tensor<32x196x384xf32>
    %v3475 = stablehlo.multiply %v3465, %v3472 : tensor<32x196x384xf32>
    %v3476 = stablehlo.subtract %v3474, %v3475 : tensor<32x196x384xf32>
    %v3477 = stablehlo.divide %v3464, %v3453 : tensor<32x196x384xf32>
    %v3478 = stablehlo.multiply %v3477, %v3476 : tensor<32x196x384xf32>
    %v3479 = stablehlo.reshape %v3478 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3480 = stablehlo.reshape %v3479 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3481 = stablehlo.transpose %v3480, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3482 = stablehlo.reshape %v3481 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3483 = stablehlo.reshape %v3482 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3484 = stablehlo.reverse %s2b0dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3485 = stablehlo.convolution(%v3483, %v3484)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3486 = stablehlo.reshape %v3485 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3487 = stablehlo.add %v3486, %v3313 : tensor<32x75264xf32>
    %v3488 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3489 = stablehlo.reshape %v580 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3490 = stablehlo.reshape %v3313 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3491 = stablehlo.multiply %v3489, %v3490 : tensor<32x384x14x14xf32>
    %v3492 = stablehlo.reduce(%v3491 init: %v3488) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3493 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3494 = stablehlo.multiply %v3492, %v3493 : tensor<384xf32>
    %v3495 = stablehlo.subtract %s2b0lg, %v3494 : tensor<384xf32>
    %v3496 = stablehlo.reshape %v575 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3497 = stablehlo.reshape %v3406 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3498 = stablehlo.transpose %v3496, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3499 = stablehlo.transpose %v3497, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3500 = stablehlo.convolution(%v3498, %v3499)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3501 = stablehlo.transpose %v3500, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3502 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v3503 = stablehlo.multiply %v3501, %v3502 : tensor<384x1536x1x1xf32>
    %v3504 = stablehlo.subtract %s2b0pW, %v3503 : tensor<384x1536x1x1xf32>
    %v3505 = stablehlo.reshape %v3406 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3507 = stablehlo.reduce(%v3505 init: %v3506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3508 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3509 = stablehlo.multiply %v3507, %v3508 : tensor<384xf32>
    %v3510 = stablehlo.subtract %s2b0pb, %v3509 : tensor<384xf32>
    %v3511 = stablehlo.reshape %v557 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3512 = stablehlo.reshape %v3434 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3513 = stablehlo.transpose %v3511, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3514 = stablehlo.transpose %v3512, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3515 = stablehlo.convolution(%v3513, %v3514)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3516 = stablehlo.transpose %v3515, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3517 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v3518 = stablehlo.multiply %v3516, %v3517 : tensor<1536x384x1x1xf32>
    %v3519 = stablehlo.subtract %s2b0eW, %v3518 : tensor<1536x384x1x1xf32>
    %v3520 = stablehlo.reshape %v3434 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3521 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3522 = stablehlo.reduce(%v3520 init: %v3521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3523 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3524 = stablehlo.multiply %v3522, %v3523 : tensor<1536xf32>
    %v3525 = stablehlo.subtract %s2b0eb, %v3524 : tensor<1536xf32>
    %v3526 = stablehlo.reshape %v523 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3527 = stablehlo.transpose %v3526, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3528 = stablehlo.reshape %v3527 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3529 = stablehlo.reshape %v3439 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3530 = stablehlo.transpose %v3529, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3531 = stablehlo.reshape %v3530 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3532 = stablehlo.reshape %v3528 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3533 = stablehlo.reshape %v3531 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3535 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3536 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3537 = stablehlo.reduce(%v3532 init: %v3534) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3538 = stablehlo.broadcast_in_dim %v3537, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3539 = stablehlo.divide %v3538, %v3535 : tensor<32x196x384xf32>
    %v3540 = stablehlo.subtract %v3532, %v3539 : tensor<32x196x384xf32>
    %v3541 = stablehlo.multiply %v3540, %v3540 : tensor<32x196x384xf32>
    %v3542 = stablehlo.reduce(%v3541 init: %v3534) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3543 = stablehlo.broadcast_in_dim %v3542, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3544 = stablehlo.divide %v3543, %v3535 : tensor<32x196x384xf32>
    %v3545 = stablehlo.add %v3544, %v3536 : tensor<32x196x384xf32>
    %v3546 = stablehlo.rsqrt %v3545 : tensor<32x196x384xf32>
    %v3547 = stablehlo.multiply %v3540, %v3546 : tensor<32x196x384xf32>
    %v3548 = stablehlo.multiply %v3533, %v3547 : tensor<32x196x384xf32>
    %v3549 = stablehlo.reduce(%v3548 init: %v3534) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3550 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3551 = stablehlo.multiply %v3549, %v3550 : tensor<384xf32>
    %v3552 = stablehlo.subtract %s2b0ng, %v3551 : tensor<384xf32>
    %v3553 = stablehlo.reshape %v3439 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3554 = stablehlo.transpose %v3553, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3555 = stablehlo.reshape %v3554 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3557 = stablehlo.reshape %v3555 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3558 = stablehlo.reduce(%v3557 init: %v3556) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3559 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3560 = stablehlo.multiply %v3558, %v3559 : tensor<384xf32>
    %v3561 = stablehlo.subtract %s2b0nbt, %v3560 : tensor<384xf32>
    %v3562 = stablehlo.reshape %v518 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3563 = stablehlo.reshape %v3482 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3564 = stablehlo.transpose %v3562, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3565 = stablehlo.transpose %v3563, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3566 = stablehlo.convolution(%v3564, %v3565)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3567 = stablehlo.reshape %v3566 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3568 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3569 = stablehlo.multiply %v3567, %v3568 : tensor<384x1x7x7xf32>
    %v3570 = stablehlo.subtract %s2b0dW, %v3569 : tensor<384x1x7x7xf32>
    %v3571 = stablehlo.reshape %v3482 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3572 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3573 = stablehlo.reduce(%v3571 init: %v3572) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3574 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3575 = stablehlo.multiply %v3573, %v3574 : tensor<384xf32>
    %v3576 = stablehlo.subtract %s2b0db, %v3575 : tensor<384xf32>
    %v3577 = stablehlo.reshape %v3487 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3578 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3579 = stablehlo.pad %v3577, %v3578, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x28x28xf32>
    %v3580 = stablehlo.transpose %d1W, dims = [1, 0, 2, 3] : (tensor<384x192x2x2xf32>) -> tensor<192x384x2x2xf32>
    %v3581 = stablehlo.reverse %v3580, dims = [2, 3] : tensor<192x384x2x2xf32>
    %v3582 = stablehlo.convolution(%v3579, %v3581)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x28x28xf32>, tensor<192x384x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v3583 = stablehlo.reshape %v3582 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3584 = stablehlo.reshape %v479 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3585 = stablehlo.transpose %v3584, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3586 = stablehlo.reshape %v3585 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3587 = stablehlo.reshape %v3583 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3588 = stablehlo.transpose %v3587, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3589 = stablehlo.reshape %v3588 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3590 = stablehlo.reshape %v3589 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3591 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v3592 = stablehlo.multiply %v3590, %v3591 : tensor<32x784x192xf32>
    %v3593 = stablehlo.reshape %v3592 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3594 = stablehlo.reshape %v3593 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3595 = stablehlo.reshape %v3586 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3596 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3597 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3598 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3599 = stablehlo.reduce(%v3595 init: %v3596) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3600 = stablehlo.broadcast_in_dim %v3599, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3601 = stablehlo.divide %v3600, %v3597 : tensor<32x784x192xf32>
    %v3602 = stablehlo.subtract %v3595, %v3601 : tensor<32x784x192xf32>
    %v3603 = stablehlo.multiply %v3602, %v3602 : tensor<32x784x192xf32>
    %v3604 = stablehlo.reduce(%v3603 init: %v3596) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3605 = stablehlo.broadcast_in_dim %v3604, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3606 = stablehlo.divide %v3605, %v3597 : tensor<32x784x192xf32>
    %v3607 = stablehlo.add %v3606, %v3598 : tensor<32x784x192xf32>
    %v3608 = stablehlo.rsqrt %v3607 : tensor<32x784x192xf32>
    %v3609 = stablehlo.multiply %v3602, %v3608 : tensor<32x784x192xf32>
    %v3610 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v3611 = stablehlo.multiply %v3610, %v3594 : tensor<32x784x192xf32>
    %v3612 = stablehlo.reduce(%v3611 init: %v3596) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3613 = stablehlo.broadcast_in_dim %v3612, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3614 = stablehlo.multiply %v3609, %v3611 : tensor<32x784x192xf32>
    %v3615 = stablehlo.reduce(%v3614 init: %v3596) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3616 = stablehlo.broadcast_in_dim %v3615, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3617 = stablehlo.multiply %v3611, %v3597 : tensor<32x784x192xf32>
    %v3618 = stablehlo.subtract %v3617, %v3613 : tensor<32x784x192xf32>
    %v3619 = stablehlo.multiply %v3609, %v3616 : tensor<32x784x192xf32>
    %v3620 = stablehlo.subtract %v3618, %v3619 : tensor<32x784x192xf32>
    %v3621 = stablehlo.divide %v3608, %v3597 : tensor<32x784x192xf32>
    %v3622 = stablehlo.multiply %v3621, %v3620 : tensor<32x784x192xf32>
    %v3623 = stablehlo.reshape %v3622 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3624 = stablehlo.reshape %v3623 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3625 = stablehlo.transpose %v3624, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v3626 = stablehlo.reshape %v3625 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v3627 = stablehlo.reshape %v3487 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3628 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3629 = stablehlo.reduce(%v3627 init: %v3628) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3630 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3631 = stablehlo.multiply %v3629, %v3630 : tensor<384xf32>
    %v3632 = stablehlo.subtract %d1b, %v3631 : tensor<384xf32>
    %v3633 = stablehlo.reshape %v479 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3634 = stablehlo.transpose %v3633, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3635 = stablehlo.reshape %v3634 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3636 = stablehlo.reshape %v3583 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3637 = stablehlo.transpose %v3636, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3638 = stablehlo.reshape %v3637 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3639 = stablehlo.reshape %v3635 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3640 = stablehlo.reshape %v3638 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3642 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3643 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3644 = stablehlo.reduce(%v3639 init: %v3641) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3645 = stablehlo.broadcast_in_dim %v3644, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3646 = stablehlo.divide %v3645, %v3642 : tensor<32x784x192xf32>
    %v3647 = stablehlo.subtract %v3639, %v3646 : tensor<32x784x192xf32>
    %v3648 = stablehlo.multiply %v3647, %v3647 : tensor<32x784x192xf32>
    %v3649 = stablehlo.reduce(%v3648 init: %v3641) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3650 = stablehlo.broadcast_in_dim %v3649, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3651 = stablehlo.divide %v3650, %v3642 : tensor<32x784x192xf32>
    %v3652 = stablehlo.add %v3651, %v3643 : tensor<32x784x192xf32>
    %v3653 = stablehlo.rsqrt %v3652 : tensor<32x784x192xf32>
    %v3654 = stablehlo.multiply %v3647, %v3653 : tensor<32x784x192xf32>
    %v3655 = stablehlo.multiply %v3640, %v3654 : tensor<32x784x192xf32>
    %v3656 = stablehlo.reduce(%v3655 init: %v3641) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3657 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3658 = stablehlo.multiply %v3656, %v3657 : tensor<192xf32>
    %v3659 = stablehlo.subtract %d1ng, %v3658 : tensor<192xf32>
    %v3660 = stablehlo.reshape %v3583 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3661 = stablehlo.transpose %v3660, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3662 = stablehlo.reshape %v3661 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3663 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3664 = stablehlo.reshape %v3662 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3665 = stablehlo.reduce(%v3664 init: %v3663) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3666 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3667 = stablehlo.multiply %v3665, %v3666 : tensor<192xf32>
    %v3668 = stablehlo.subtract %d1nbt, %v3667 : tensor<192xf32>
    %v3669 = stablehlo.reshape %v513 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3670 = stablehlo.reshape %v3487 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3672 = stablehlo.pad %v3670, %v3671, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x27x27xf32>
    %v3673 = stablehlo.transpose %v3669, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3674 = stablehlo.transpose %v3672, dims = [1, 0, 2, 3] : (tensor<32x384x27x27xf32>) -> tensor<384x32x27x27xf32>
    %v3675 = stablehlo.convolution(%v3673, %v3674)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<384x32x27x27xf32>) -> tensor<192x384x2x2xf32>
    %v3676 = stablehlo.transpose %v3675, dims = [1, 0, 2, 3] : (tensor<192x384x2x2xf32>) -> tensor<384x192x2x2xf32>
    %v3677 = stablehlo.constant dense<0.1> : tensor<384x192x2x2xf32>
    %v3678 = stablehlo.multiply %v3676, %v3677 : tensor<384x192x2x2xf32>
    %v3679 = stablehlo.subtract %d1W, %v3678 : tensor<384x192x2x2xf32>
    %v3680 = stablehlo.reshape %v3626 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3681 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3682 = stablehlo.multiply %v3680, %v3681 : tensor<32x192x28x28xf32>
    %v3683 = stablehlo.reshape %v3682 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3684 = stablehlo.reshape %v3683 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3685 = stablehlo.transpose %s1b2pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3686 = stablehlo.reverse %v3685, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v3687 = stablehlo.convolution(%v3684, %v3686)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v3688 = stablehlo.reshape %v3687 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3689 = stablehlo.multiply %v456, %v456 : tensor<32x602112xf32>
    %v3690 = stablehlo.multiply %v3689, %v456 : tensor<32x602112xf32>
    %v3691 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v3692 = stablehlo.multiply %v3691, %v3690 : tensor<32x602112xf32>
    %v3693 = stablehlo.add %v456, %v3692 : tensor<32x602112xf32>
    %v3694 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v3695 = stablehlo.multiply %v3694, %v3693 : tensor<32x602112xf32>
    %v3696 = stablehlo.tanh %v3695 : tensor<32x602112xf32>
    %v3697 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v3698 = stablehlo.add %v3697, %v3696 : tensor<32x602112xf32>
    %v3699 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v3700 = stablehlo.multiply %v3699, %v3698 : tensor<32x602112xf32>
    %v3701 = stablehlo.multiply %v3696, %v3696 : tensor<32x602112xf32>
    %v3702 = stablehlo.subtract %v3697, %v3701 : tensor<32x602112xf32>
    %v3703 = stablehlo.multiply %v3699, %v456 : tensor<32x602112xf32>
    %v3704 = stablehlo.multiply %v3703, %v3702 : tensor<32x602112xf32>
    %v3705 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v3706 = stablehlo.multiply %v3705, %v3689 : tensor<32x602112xf32>
    %v3707 = stablehlo.add %v3697, %v3706 : tensor<32x602112xf32>
    %v3708 = stablehlo.multiply %v3694, %v3707 : tensor<32x602112xf32>
    %v3709 = stablehlo.multiply %v3704, %v3708 : tensor<32x602112xf32>
    %v3710 = stablehlo.add %v3700, %v3709 : tensor<32x602112xf32>
    %v3711 = stablehlo.multiply %v3688, %v3710 : tensor<32x602112xf32>
    %v3712 = stablehlo.reshape %v3711 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3713 = stablehlo.transpose %s1b2eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3714 = stablehlo.reverse %v3713, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3715 = stablehlo.convolution(%v3712, %v3714)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3716 = stablehlo.reshape %v3715 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3717 = stablehlo.reshape %v417 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3718 = stablehlo.transpose %v3717, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3719 = stablehlo.reshape %v3718 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3720 = stablehlo.reshape %v3716 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3721 = stablehlo.transpose %v3720, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3722 = stablehlo.reshape %v3721 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3723 = stablehlo.reshape %v3722 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3724 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v3725 = stablehlo.multiply %v3723, %v3724 : tensor<32x784x192xf32>
    %v3726 = stablehlo.reshape %v3725 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3727 = stablehlo.reshape %v3726 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3728 = stablehlo.reshape %v3719 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3730 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3731 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3732 = stablehlo.reduce(%v3728 init: %v3729) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3733 = stablehlo.broadcast_in_dim %v3732, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3734 = stablehlo.divide %v3733, %v3730 : tensor<32x784x192xf32>
    %v3735 = stablehlo.subtract %v3728, %v3734 : tensor<32x784x192xf32>
    %v3736 = stablehlo.multiply %v3735, %v3735 : tensor<32x784x192xf32>
    %v3737 = stablehlo.reduce(%v3736 init: %v3729) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3738 = stablehlo.broadcast_in_dim %v3737, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3739 = stablehlo.divide %v3738, %v3730 : tensor<32x784x192xf32>
    %v3740 = stablehlo.add %v3739, %v3731 : tensor<32x784x192xf32>
    %v3741 = stablehlo.rsqrt %v3740 : tensor<32x784x192xf32>
    %v3742 = stablehlo.multiply %v3735, %v3741 : tensor<32x784x192xf32>
    %v3743 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v3744 = stablehlo.multiply %v3743, %v3727 : tensor<32x784x192xf32>
    %v3745 = stablehlo.reduce(%v3744 init: %v3729) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3746 = stablehlo.broadcast_in_dim %v3745, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3747 = stablehlo.multiply %v3742, %v3744 : tensor<32x784x192xf32>
    %v3748 = stablehlo.reduce(%v3747 init: %v3729) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3749 = stablehlo.broadcast_in_dim %v3748, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3750 = stablehlo.multiply %v3744, %v3730 : tensor<32x784x192xf32>
    %v3751 = stablehlo.subtract %v3750, %v3746 : tensor<32x784x192xf32>
    %v3752 = stablehlo.multiply %v3742, %v3749 : tensor<32x784x192xf32>
    %v3753 = stablehlo.subtract %v3751, %v3752 : tensor<32x784x192xf32>
    %v3754 = stablehlo.divide %v3741, %v3730 : tensor<32x784x192xf32>
    %v3755 = stablehlo.multiply %v3754, %v3753 : tensor<32x784x192xf32>
    %v3756 = stablehlo.reshape %v3755 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3757 = stablehlo.reshape %v3756 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3758 = stablehlo.transpose %v3757, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v3759 = stablehlo.reshape %v3758 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v3760 = stablehlo.reshape %v3759 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3761 = stablehlo.reverse %s1b2dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v3762 = stablehlo.convolution(%v3760, %v3761)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v3763 = stablehlo.reshape %v3762 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3764 = stablehlo.add %v3763, %v3626 : tensor<32x150528xf32>
    %v3765 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3766 = stablehlo.reshape %v474 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3767 = stablehlo.reshape %v3626 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3768 = stablehlo.multiply %v3766, %v3767 : tensor<32x192x28x28xf32>
    %v3769 = stablehlo.reduce(%v3768 init: %v3765) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3770 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3771 = stablehlo.multiply %v3769, %v3770 : tensor<192xf32>
    %v3772 = stablehlo.subtract %s1b2lg, %v3771 : tensor<192xf32>
    %v3773 = stablehlo.reshape %v469 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3774 = stablehlo.reshape %v3683 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3775 = stablehlo.transpose %v3773, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3776 = stablehlo.transpose %v3774, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3777 = stablehlo.convolution(%v3775, %v3776)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v3778 = stablehlo.transpose %v3777, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3779 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v3780 = stablehlo.multiply %v3778, %v3779 : tensor<192x768x1x1xf32>
    %v3781 = stablehlo.subtract %s1b2pW, %v3780 : tensor<192x768x1x1xf32>
    %v3782 = stablehlo.reshape %v3683 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3784 = stablehlo.reduce(%v3782 init: %v3783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3785 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3786 = stablehlo.multiply %v3784, %v3785 : tensor<192xf32>
    %v3787 = stablehlo.subtract %s1b2pb, %v3786 : tensor<192xf32>
    %v3788 = stablehlo.reshape %v451 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3789 = stablehlo.reshape %v3711 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3790 = stablehlo.transpose %v3788, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3791 = stablehlo.transpose %v3789, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3792 = stablehlo.convolution(%v3790, %v3791)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v3793 = stablehlo.transpose %v3792, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3794 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v3795 = stablehlo.multiply %v3793, %v3794 : tensor<768x192x1x1xf32>
    %v3796 = stablehlo.subtract %s1b2eW, %v3795 : tensor<768x192x1x1xf32>
    %v3797 = stablehlo.reshape %v3711 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3798 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3799 = stablehlo.reduce(%v3797 init: %v3798) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v3800 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v3801 = stablehlo.multiply %v3799, %v3800 : tensor<768xf32>
    %v3802 = stablehlo.subtract %s1b2eb, %v3801 : tensor<768xf32>
    %v3803 = stablehlo.reshape %v417 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3804 = stablehlo.transpose %v3803, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3805 = stablehlo.reshape %v3804 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3806 = stablehlo.reshape %v3716 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3807 = stablehlo.transpose %v3806, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3808 = stablehlo.reshape %v3807 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3809 = stablehlo.reshape %v3805 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3810 = stablehlo.reshape %v3808 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3812 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3813 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3814 = stablehlo.reduce(%v3809 init: %v3811) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3815 = stablehlo.broadcast_in_dim %v3814, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3816 = stablehlo.divide %v3815, %v3812 : tensor<32x784x192xf32>
    %v3817 = stablehlo.subtract %v3809, %v3816 : tensor<32x784x192xf32>
    %v3818 = stablehlo.multiply %v3817, %v3817 : tensor<32x784x192xf32>
    %v3819 = stablehlo.reduce(%v3818 init: %v3811) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3820 = stablehlo.broadcast_in_dim %v3819, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3821 = stablehlo.divide %v3820, %v3812 : tensor<32x784x192xf32>
    %v3822 = stablehlo.add %v3821, %v3813 : tensor<32x784x192xf32>
    %v3823 = stablehlo.rsqrt %v3822 : tensor<32x784x192xf32>
    %v3824 = stablehlo.multiply %v3817, %v3823 : tensor<32x784x192xf32>
    %v3825 = stablehlo.multiply %v3810, %v3824 : tensor<32x784x192xf32>
    %v3826 = stablehlo.reduce(%v3825 init: %v3811) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3827 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3828 = stablehlo.multiply %v3826, %v3827 : tensor<192xf32>
    %v3829 = stablehlo.subtract %s1b2ng, %v3828 : tensor<192xf32>
    %v3830 = stablehlo.reshape %v3716 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3831 = stablehlo.transpose %v3830, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3832 = stablehlo.reshape %v3831 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3834 = stablehlo.reshape %v3832 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3835 = stablehlo.reduce(%v3834 init: %v3833) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3836 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3837 = stablehlo.multiply %v3835, %v3836 : tensor<192xf32>
    %v3838 = stablehlo.subtract %s1b2nbt, %v3837 : tensor<192xf32>
    %v3839 = stablehlo.reshape %v412 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3840 = stablehlo.reshape %v3759 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3841 = stablehlo.transpose %v3839, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3842 = stablehlo.transpose %v3840, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3843 = stablehlo.convolution(%v3841, %v3842)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v3844 = stablehlo.reshape %v3843 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v3845 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v3846 = stablehlo.multiply %v3844, %v3845 : tensor<192x1x7x7xf32>
    %v3847 = stablehlo.subtract %s1b2dW, %v3846 : tensor<192x1x7x7xf32>
    %v3848 = stablehlo.reshape %v3759 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3850 = stablehlo.reduce(%v3848 init: %v3849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3851 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3852 = stablehlo.multiply %v3850, %v3851 : tensor<192xf32>
    %v3853 = stablehlo.subtract %s1b2db, %v3852 : tensor<192xf32>
    %v3854 = stablehlo.reshape %v3764 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3855 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3856 = stablehlo.multiply %v3854, %v3855 : tensor<32x192x28x28xf32>
    %v3857 = stablehlo.reshape %v3856 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3858 = stablehlo.reshape %v3857 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3859 = stablehlo.transpose %s1b1pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3860 = stablehlo.reverse %v3859, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v3861 = stablehlo.convolution(%v3858, %v3860)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v3862 = stablehlo.reshape %v3861 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3863 = stablehlo.multiply %v389, %v389 : tensor<32x602112xf32>
    %v3864 = stablehlo.multiply %v3863, %v389 : tensor<32x602112xf32>
    %v3865 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v3866 = stablehlo.multiply %v3865, %v3864 : tensor<32x602112xf32>
    %v3867 = stablehlo.add %v389, %v3866 : tensor<32x602112xf32>
    %v3868 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v3869 = stablehlo.multiply %v3868, %v3867 : tensor<32x602112xf32>
    %v3870 = stablehlo.tanh %v3869 : tensor<32x602112xf32>
    %v3871 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v3872 = stablehlo.add %v3871, %v3870 : tensor<32x602112xf32>
    %v3873 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v3874 = stablehlo.multiply %v3873, %v3872 : tensor<32x602112xf32>
    %v3875 = stablehlo.multiply %v3870, %v3870 : tensor<32x602112xf32>
    %v3876 = stablehlo.subtract %v3871, %v3875 : tensor<32x602112xf32>
    %v3877 = stablehlo.multiply %v3873, %v389 : tensor<32x602112xf32>
    %v3878 = stablehlo.multiply %v3877, %v3876 : tensor<32x602112xf32>
    %v3879 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v3880 = stablehlo.multiply %v3879, %v3863 : tensor<32x602112xf32>
    %v3881 = stablehlo.add %v3871, %v3880 : tensor<32x602112xf32>
    %v3882 = stablehlo.multiply %v3868, %v3881 : tensor<32x602112xf32>
    %v3883 = stablehlo.multiply %v3878, %v3882 : tensor<32x602112xf32>
    %v3884 = stablehlo.add %v3874, %v3883 : tensor<32x602112xf32>
    %v3885 = stablehlo.multiply %v3862, %v3884 : tensor<32x602112xf32>
    %v3886 = stablehlo.reshape %v3885 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3887 = stablehlo.transpose %s1b1eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3888 = stablehlo.reverse %v3887, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3889 = stablehlo.convolution(%v3886, %v3888)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3890 = stablehlo.reshape %v3889 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3891 = stablehlo.reshape %v350 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3892 = stablehlo.transpose %v3891, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3893 = stablehlo.reshape %v3892 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3894 = stablehlo.reshape %v3890 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3895 = stablehlo.transpose %v3894, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3896 = stablehlo.reshape %v3895 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3897 = stablehlo.reshape %v3896 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3898 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v3899 = stablehlo.multiply %v3897, %v3898 : tensor<32x784x192xf32>
    %v3900 = stablehlo.reshape %v3899 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3901 = stablehlo.reshape %v3900 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3902 = stablehlo.reshape %v3893 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3904 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3905 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3906 = stablehlo.reduce(%v3902 init: %v3903) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3907 = stablehlo.broadcast_in_dim %v3906, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3908 = stablehlo.divide %v3907, %v3904 : tensor<32x784x192xf32>
    %v3909 = stablehlo.subtract %v3902, %v3908 : tensor<32x784x192xf32>
    %v3910 = stablehlo.multiply %v3909, %v3909 : tensor<32x784x192xf32>
    %v3911 = stablehlo.reduce(%v3910 init: %v3903) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3912 = stablehlo.broadcast_in_dim %v3911, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3913 = stablehlo.divide %v3912, %v3904 : tensor<32x784x192xf32>
    %v3914 = stablehlo.add %v3913, %v3905 : tensor<32x784x192xf32>
    %v3915 = stablehlo.rsqrt %v3914 : tensor<32x784x192xf32>
    %v3916 = stablehlo.multiply %v3909, %v3915 : tensor<32x784x192xf32>
    %v3917 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v3918 = stablehlo.multiply %v3917, %v3901 : tensor<32x784x192xf32>
    %v3919 = stablehlo.reduce(%v3918 init: %v3903) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3920 = stablehlo.broadcast_in_dim %v3919, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3921 = stablehlo.multiply %v3916, %v3918 : tensor<32x784x192xf32>
    %v3922 = stablehlo.reduce(%v3921 init: %v3903) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3923 = stablehlo.broadcast_in_dim %v3922, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3924 = stablehlo.multiply %v3918, %v3904 : tensor<32x784x192xf32>
    %v3925 = stablehlo.subtract %v3924, %v3920 : tensor<32x784x192xf32>
    %v3926 = stablehlo.multiply %v3916, %v3923 : tensor<32x784x192xf32>
    %v3927 = stablehlo.subtract %v3925, %v3926 : tensor<32x784x192xf32>
    %v3928 = stablehlo.divide %v3915, %v3904 : tensor<32x784x192xf32>
    %v3929 = stablehlo.multiply %v3928, %v3927 : tensor<32x784x192xf32>
    %v3930 = stablehlo.reshape %v3929 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3931 = stablehlo.reshape %v3930 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3932 = stablehlo.transpose %v3931, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v3933 = stablehlo.reshape %v3932 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v3934 = stablehlo.reshape %v3933 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3935 = stablehlo.reverse %s1b1dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v3936 = stablehlo.convolution(%v3934, %v3935)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v3937 = stablehlo.reshape %v3936 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3938 = stablehlo.add %v3937, %v3764 : tensor<32x150528xf32>
    %v3939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3940 = stablehlo.reshape %v407 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3941 = stablehlo.reshape %v3764 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3942 = stablehlo.multiply %v3940, %v3941 : tensor<32x192x28x28xf32>
    %v3943 = stablehlo.reduce(%v3942 init: %v3939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3944 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3945 = stablehlo.multiply %v3943, %v3944 : tensor<192xf32>
    %v3946 = stablehlo.subtract %s1b1lg, %v3945 : tensor<192xf32>
    %v3947 = stablehlo.reshape %v402 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3948 = stablehlo.reshape %v3857 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3949 = stablehlo.transpose %v3947, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3950 = stablehlo.transpose %v3948, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3951 = stablehlo.convolution(%v3949, %v3950)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v3952 = stablehlo.transpose %v3951, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3953 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v3954 = stablehlo.multiply %v3952, %v3953 : tensor<192x768x1x1xf32>
    %v3955 = stablehlo.subtract %s1b1pW, %v3954 : tensor<192x768x1x1xf32>
    %v3956 = stablehlo.reshape %v3857 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3957 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3958 = stablehlo.reduce(%v3956 init: %v3957) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3959 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3960 = stablehlo.multiply %v3958, %v3959 : tensor<192xf32>
    %v3961 = stablehlo.subtract %s1b1pb, %v3960 : tensor<192xf32>
    %v3962 = stablehlo.reshape %v384 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3963 = stablehlo.reshape %v3885 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3964 = stablehlo.transpose %v3962, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3965 = stablehlo.transpose %v3963, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3966 = stablehlo.convolution(%v3964, %v3965)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v3967 = stablehlo.transpose %v3966, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3968 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v3969 = stablehlo.multiply %v3967, %v3968 : tensor<768x192x1x1xf32>
    %v3970 = stablehlo.subtract %s1b1eW, %v3969 : tensor<768x192x1x1xf32>
    %v3971 = stablehlo.reshape %v3885 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3972 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3973 = stablehlo.reduce(%v3971 init: %v3972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v3974 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v3975 = stablehlo.multiply %v3973, %v3974 : tensor<768xf32>
    %v3976 = stablehlo.subtract %s1b1eb, %v3975 : tensor<768xf32>
    %v3977 = stablehlo.reshape %v350 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3978 = stablehlo.transpose %v3977, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3979 = stablehlo.reshape %v3978 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3980 = stablehlo.reshape %v3890 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3981 = stablehlo.transpose %v3980, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3982 = stablehlo.reshape %v3981 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3983 = stablehlo.reshape %v3979 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3984 = stablehlo.reshape %v3982 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3985 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3986 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3987 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3988 = stablehlo.reduce(%v3983 init: %v3985) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3989 = stablehlo.broadcast_in_dim %v3988, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3990 = stablehlo.divide %v3989, %v3986 : tensor<32x784x192xf32>
    %v3991 = stablehlo.subtract %v3983, %v3990 : tensor<32x784x192xf32>
    %v3992 = stablehlo.multiply %v3991, %v3991 : tensor<32x784x192xf32>
    %v3993 = stablehlo.reduce(%v3992 init: %v3985) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3994 = stablehlo.broadcast_in_dim %v3993, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3995 = stablehlo.divide %v3994, %v3986 : tensor<32x784x192xf32>
    %v3996 = stablehlo.add %v3995, %v3987 : tensor<32x784x192xf32>
    %v3997 = stablehlo.rsqrt %v3996 : tensor<32x784x192xf32>
    %v3998 = stablehlo.multiply %v3991, %v3997 : tensor<32x784x192xf32>
    %v3999 = stablehlo.multiply %v3984, %v3998 : tensor<32x784x192xf32>
    %v4000 = stablehlo.reduce(%v3999 init: %v3985) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4001 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4002 = stablehlo.multiply %v4000, %v4001 : tensor<192xf32>
    %v4003 = stablehlo.subtract %s1b1ng, %v4002 : tensor<192xf32>
    %v4004 = stablehlo.reshape %v3890 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4005 = stablehlo.transpose %v4004, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4006 = stablehlo.reshape %v4005 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4008 = stablehlo.reshape %v4006 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4009 = stablehlo.reduce(%v4008 init: %v4007) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4010 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4011 = stablehlo.multiply %v4009, %v4010 : tensor<192xf32>
    %v4012 = stablehlo.subtract %s1b1nbt, %v4011 : tensor<192xf32>
    %v4013 = stablehlo.reshape %v345 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4014 = stablehlo.reshape %v3933 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4015 = stablehlo.transpose %v4013, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4016 = stablehlo.transpose %v4014, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4017 = stablehlo.convolution(%v4015, %v4016)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v4018 = stablehlo.reshape %v4017 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v4019 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v4020 = stablehlo.multiply %v4018, %v4019 : tensor<192x1x7x7xf32>
    %v4021 = stablehlo.subtract %s1b1dW, %v4020 : tensor<192x1x7x7xf32>
    %v4022 = stablehlo.reshape %v3933 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4024 = stablehlo.reduce(%v4022 init: %v4023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4025 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4026 = stablehlo.multiply %v4024, %v4025 : tensor<192xf32>
    %v4027 = stablehlo.subtract %s1b1db, %v4026 : tensor<192xf32>
    %v4028 = stablehlo.reshape %v3938 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4029 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4030 = stablehlo.multiply %v4028, %v4029 : tensor<32x192x28x28xf32>
    %v4031 = stablehlo.reshape %v4030 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4032 = stablehlo.reshape %v4031 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4033 = stablehlo.transpose %s1b0pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v4034 = stablehlo.reverse %v4033, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v4035 = stablehlo.convolution(%v4032, %v4034)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v4036 = stablehlo.reshape %v4035 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v4037 = stablehlo.multiply %v322, %v322 : tensor<32x602112xf32>
    %v4038 = stablehlo.multiply %v4037, %v322 : tensor<32x602112xf32>
    %v4039 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v4040 = stablehlo.multiply %v4039, %v4038 : tensor<32x602112xf32>
    %v4041 = stablehlo.add %v322, %v4040 : tensor<32x602112xf32>
    %v4042 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v4043 = stablehlo.multiply %v4042, %v4041 : tensor<32x602112xf32>
    %v4044 = stablehlo.tanh %v4043 : tensor<32x602112xf32>
    %v4045 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v4046 = stablehlo.add %v4045, %v4044 : tensor<32x602112xf32>
    %v4047 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v4048 = stablehlo.multiply %v4047, %v4046 : tensor<32x602112xf32>
    %v4049 = stablehlo.multiply %v4044, %v4044 : tensor<32x602112xf32>
    %v4050 = stablehlo.subtract %v4045, %v4049 : tensor<32x602112xf32>
    %v4051 = stablehlo.multiply %v4047, %v322 : tensor<32x602112xf32>
    %v4052 = stablehlo.multiply %v4051, %v4050 : tensor<32x602112xf32>
    %v4053 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v4054 = stablehlo.multiply %v4053, %v4037 : tensor<32x602112xf32>
    %v4055 = stablehlo.add %v4045, %v4054 : tensor<32x602112xf32>
    %v4056 = stablehlo.multiply %v4042, %v4055 : tensor<32x602112xf32>
    %v4057 = stablehlo.multiply %v4052, %v4056 : tensor<32x602112xf32>
    %v4058 = stablehlo.add %v4048, %v4057 : tensor<32x602112xf32>
    %v4059 = stablehlo.multiply %v4036, %v4058 : tensor<32x602112xf32>
    %v4060 = stablehlo.reshape %v4059 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4061 = stablehlo.transpose %s1b0eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v4062 = stablehlo.reverse %v4061, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v4063 = stablehlo.convolution(%v4060, %v4062)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4064 = stablehlo.reshape %v4063 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4065 = stablehlo.reshape %v283 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4066 = stablehlo.transpose %v4065, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4067 = stablehlo.reshape %v4066 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4068 = stablehlo.reshape %v4064 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4069 = stablehlo.transpose %v4068, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4070 = stablehlo.reshape %v4069 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4071 = stablehlo.reshape %v4070 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4072 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v4073 = stablehlo.multiply %v4071, %v4072 : tensor<32x784x192xf32>
    %v4074 = stablehlo.reshape %v4073 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4075 = stablehlo.reshape %v4074 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4076 = stablehlo.reshape %v4067 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4078 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v4079 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v4080 = stablehlo.reduce(%v4076 init: %v4077) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4081 = stablehlo.broadcast_in_dim %v4080, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4082 = stablehlo.divide %v4081, %v4078 : tensor<32x784x192xf32>
    %v4083 = stablehlo.subtract %v4076, %v4082 : tensor<32x784x192xf32>
    %v4084 = stablehlo.multiply %v4083, %v4083 : tensor<32x784x192xf32>
    %v4085 = stablehlo.reduce(%v4084 init: %v4077) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4086 = stablehlo.broadcast_in_dim %v4085, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4087 = stablehlo.divide %v4086, %v4078 : tensor<32x784x192xf32>
    %v4088 = stablehlo.add %v4087, %v4079 : tensor<32x784x192xf32>
    %v4089 = stablehlo.rsqrt %v4088 : tensor<32x784x192xf32>
    %v4090 = stablehlo.multiply %v4083, %v4089 : tensor<32x784x192xf32>
    %v4091 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v4092 = stablehlo.multiply %v4091, %v4075 : tensor<32x784x192xf32>
    %v4093 = stablehlo.reduce(%v4092 init: %v4077) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4094 = stablehlo.broadcast_in_dim %v4093, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4095 = stablehlo.multiply %v4090, %v4092 : tensor<32x784x192xf32>
    %v4096 = stablehlo.reduce(%v4095 init: %v4077) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4097 = stablehlo.broadcast_in_dim %v4096, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4098 = stablehlo.multiply %v4092, %v4078 : tensor<32x784x192xf32>
    %v4099 = stablehlo.subtract %v4098, %v4094 : tensor<32x784x192xf32>
    %v4100 = stablehlo.multiply %v4090, %v4097 : tensor<32x784x192xf32>
    %v4101 = stablehlo.subtract %v4099, %v4100 : tensor<32x784x192xf32>
    %v4102 = stablehlo.divide %v4089, %v4078 : tensor<32x784x192xf32>
    %v4103 = stablehlo.multiply %v4102, %v4101 : tensor<32x784x192xf32>
    %v4104 = stablehlo.reshape %v4103 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4105 = stablehlo.reshape %v4104 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4106 = stablehlo.transpose %v4105, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v4107 = stablehlo.reshape %v4106 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v4108 = stablehlo.reshape %v4107 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4109 = stablehlo.reverse %s1b0dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v4110 = stablehlo.convolution(%v4108, %v4109)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v4111 = stablehlo.reshape %v4110 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4112 = stablehlo.add %v4111, %v3938 : tensor<32x150528xf32>
    %v4113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4114 = stablehlo.reshape %v340 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4115 = stablehlo.reshape %v3938 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4116 = stablehlo.multiply %v4114, %v4115 : tensor<32x192x28x28xf32>
    %v4117 = stablehlo.reduce(%v4116 init: %v4113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4118 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4119 = stablehlo.multiply %v4117, %v4118 : tensor<192xf32>
    %v4120 = stablehlo.subtract %s1b0lg, %v4119 : tensor<192xf32>
    %v4121 = stablehlo.reshape %v335 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4122 = stablehlo.reshape %v4031 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4123 = stablehlo.transpose %v4121, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v4124 = stablehlo.transpose %v4122, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4125 = stablehlo.convolution(%v4123, %v4124)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v4126 = stablehlo.transpose %v4125, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v4127 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v4128 = stablehlo.multiply %v4126, %v4127 : tensor<192x768x1x1xf32>
    %v4129 = stablehlo.subtract %s1b0pW, %v4128 : tensor<192x768x1x1xf32>
    %v4130 = stablehlo.reshape %v4031 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4132 = stablehlo.reduce(%v4130 init: %v4131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4133 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4134 = stablehlo.multiply %v4132, %v4133 : tensor<192xf32>
    %v4135 = stablehlo.subtract %s1b0pb, %v4134 : tensor<192xf32>
    %v4136 = stablehlo.reshape %v317 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4137 = stablehlo.reshape %v4059 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4138 = stablehlo.transpose %v4136, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4139 = stablehlo.transpose %v4137, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v4140 = stablehlo.convolution(%v4138, %v4139)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v4141 = stablehlo.transpose %v4140, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v4142 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v4143 = stablehlo.multiply %v4141, %v4142 : tensor<768x192x1x1xf32>
    %v4144 = stablehlo.subtract %s1b0eW, %v4143 : tensor<768x192x1x1xf32>
    %v4145 = stablehlo.reshape %v4059 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4147 = stablehlo.reduce(%v4145 init: %v4146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v4148 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v4149 = stablehlo.multiply %v4147, %v4148 : tensor<768xf32>
    %v4150 = stablehlo.subtract %s1b0eb, %v4149 : tensor<768xf32>
    %v4151 = stablehlo.reshape %v283 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4152 = stablehlo.transpose %v4151, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4153 = stablehlo.reshape %v4152 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4154 = stablehlo.reshape %v4064 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4155 = stablehlo.transpose %v4154, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4156 = stablehlo.reshape %v4155 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4157 = stablehlo.reshape %v4153 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4158 = stablehlo.reshape %v4156 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4160 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v4161 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v4162 = stablehlo.reduce(%v4157 init: %v4159) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4163 = stablehlo.broadcast_in_dim %v4162, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4164 = stablehlo.divide %v4163, %v4160 : tensor<32x784x192xf32>
    %v4165 = stablehlo.subtract %v4157, %v4164 : tensor<32x784x192xf32>
    %v4166 = stablehlo.multiply %v4165, %v4165 : tensor<32x784x192xf32>
    %v4167 = stablehlo.reduce(%v4166 init: %v4159) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4168 = stablehlo.broadcast_in_dim %v4167, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4169 = stablehlo.divide %v4168, %v4160 : tensor<32x784x192xf32>
    %v4170 = stablehlo.add %v4169, %v4161 : tensor<32x784x192xf32>
    %v4171 = stablehlo.rsqrt %v4170 : tensor<32x784x192xf32>
    %v4172 = stablehlo.multiply %v4165, %v4171 : tensor<32x784x192xf32>
    %v4173 = stablehlo.multiply %v4158, %v4172 : tensor<32x784x192xf32>
    %v4174 = stablehlo.reduce(%v4173 init: %v4159) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4175 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4176 = stablehlo.multiply %v4174, %v4175 : tensor<192xf32>
    %v4177 = stablehlo.subtract %s1b0ng, %v4176 : tensor<192xf32>
    %v4178 = stablehlo.reshape %v4064 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4179 = stablehlo.transpose %v4178, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4180 = stablehlo.reshape %v4179 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4182 = stablehlo.reshape %v4180 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4183 = stablehlo.reduce(%v4182 init: %v4181) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4184 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4185 = stablehlo.multiply %v4183, %v4184 : tensor<192xf32>
    %v4186 = stablehlo.subtract %s1b0nbt, %v4185 : tensor<192xf32>
    %v4187 = stablehlo.reshape %v278 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4188 = stablehlo.reshape %v4107 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4189 = stablehlo.transpose %v4187, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4190 = stablehlo.transpose %v4188, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4191 = stablehlo.convolution(%v4189, %v4190)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v4192 = stablehlo.reshape %v4191 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v4193 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v4194 = stablehlo.multiply %v4192, %v4193 : tensor<192x1x7x7xf32>
    %v4195 = stablehlo.subtract %s1b0dW, %v4194 : tensor<192x1x7x7xf32>
    %v4196 = stablehlo.reshape %v4107 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4198 = stablehlo.reduce(%v4196 init: %v4197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4199 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4200 = stablehlo.multiply %v4198, %v4199 : tensor<192xf32>
    %v4201 = stablehlo.subtract %s1b0db, %v4200 : tensor<192xf32>
    %v4202 = stablehlo.reshape %v4112 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4204 = stablehlo.pad %v4202, %v4203, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x56x56xf32>
    %v4205 = stablehlo.transpose %d0W, dims = [1, 0, 2, 3] : (tensor<192x96x2x2xf32>) -> tensor<96x192x2x2xf32>
    %v4206 = stablehlo.reverse %v4205, dims = [2, 3] : tensor<96x192x2x2xf32>
    %v4207 = stablehlo.convolution(%v4204, %v4206)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x56x56xf32>, tensor<96x192x2x2xf32>) -> tensor<32x96x56x56xf32>
    %v4208 = stablehlo.reshape %v4207 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4209 = stablehlo.reshape %v239 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4210 = stablehlo.transpose %v4209, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4211 = stablehlo.reshape %v4210 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4212 = stablehlo.reshape %v4208 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4213 = stablehlo.transpose %v4212, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4214 = stablehlo.reshape %v4213 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4215 = stablehlo.reshape %v4214 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4216 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4217 = stablehlo.multiply %v4215, %v4216 : tensor<32x3136x96xf32>
    %v4218 = stablehlo.reshape %v4217 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4219 = stablehlo.reshape %v4218 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4220 = stablehlo.reshape %v4211 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4221 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4222 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4223 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4224 = stablehlo.reduce(%v4220 init: %v4221) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4225 = stablehlo.broadcast_in_dim %v4224, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4226 = stablehlo.divide %v4225, %v4222 : tensor<32x3136x96xf32>
    %v4227 = stablehlo.subtract %v4220, %v4226 : tensor<32x3136x96xf32>
    %v4228 = stablehlo.multiply %v4227, %v4227 : tensor<32x3136x96xf32>
    %v4229 = stablehlo.reduce(%v4228 init: %v4221) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4230 = stablehlo.broadcast_in_dim %v4229, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4231 = stablehlo.divide %v4230, %v4222 : tensor<32x3136x96xf32>
    %v4232 = stablehlo.add %v4231, %v4223 : tensor<32x3136x96xf32>
    %v4233 = stablehlo.rsqrt %v4232 : tensor<32x3136x96xf32>
    %v4234 = stablehlo.multiply %v4227, %v4233 : tensor<32x3136x96xf32>
    %v4235 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4236 = stablehlo.multiply %v4235, %v4219 : tensor<32x3136x96xf32>
    %v4237 = stablehlo.reduce(%v4236 init: %v4221) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4238 = stablehlo.broadcast_in_dim %v4237, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4239 = stablehlo.multiply %v4234, %v4236 : tensor<32x3136x96xf32>
    %v4240 = stablehlo.reduce(%v4239 init: %v4221) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4241 = stablehlo.broadcast_in_dim %v4240, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4242 = stablehlo.multiply %v4236, %v4222 : tensor<32x3136x96xf32>
    %v4243 = stablehlo.subtract %v4242, %v4238 : tensor<32x3136x96xf32>
    %v4244 = stablehlo.multiply %v4234, %v4241 : tensor<32x3136x96xf32>
    %v4245 = stablehlo.subtract %v4243, %v4244 : tensor<32x3136x96xf32>
    %v4246 = stablehlo.divide %v4233, %v4222 : tensor<32x3136x96xf32>
    %v4247 = stablehlo.multiply %v4246, %v4245 : tensor<32x3136x96xf32>
    %v4248 = stablehlo.reshape %v4247 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4249 = stablehlo.reshape %v4248 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4250 = stablehlo.transpose %v4249, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4251 = stablehlo.reshape %v4250 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4252 = stablehlo.reshape %v4112 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4254 = stablehlo.reduce(%v4252 init: %v4253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4255 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4256 = stablehlo.multiply %v4254, %v4255 : tensor<192xf32>
    %v4257 = stablehlo.subtract %d0b, %v4256 : tensor<192xf32>
    %v4258 = stablehlo.reshape %v239 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4259 = stablehlo.transpose %v4258, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4260 = stablehlo.reshape %v4259 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4261 = stablehlo.reshape %v4208 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4262 = stablehlo.transpose %v4261, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4263 = stablehlo.reshape %v4262 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4264 = stablehlo.reshape %v4260 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4265 = stablehlo.reshape %v4263 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4267 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4268 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4269 = stablehlo.reduce(%v4264 init: %v4266) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4270 = stablehlo.broadcast_in_dim %v4269, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4271 = stablehlo.divide %v4270, %v4267 : tensor<32x3136x96xf32>
    %v4272 = stablehlo.subtract %v4264, %v4271 : tensor<32x3136x96xf32>
    %v4273 = stablehlo.multiply %v4272, %v4272 : tensor<32x3136x96xf32>
    %v4274 = stablehlo.reduce(%v4273 init: %v4266) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4275 = stablehlo.broadcast_in_dim %v4274, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4276 = stablehlo.divide %v4275, %v4267 : tensor<32x3136x96xf32>
    %v4277 = stablehlo.add %v4276, %v4268 : tensor<32x3136x96xf32>
    %v4278 = stablehlo.rsqrt %v4277 : tensor<32x3136x96xf32>
    %v4279 = stablehlo.multiply %v4272, %v4278 : tensor<32x3136x96xf32>
    %v4280 = stablehlo.multiply %v4265, %v4279 : tensor<32x3136x96xf32>
    %v4281 = stablehlo.reduce(%v4280 init: %v4266) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4282 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4283 = stablehlo.multiply %v4281, %v4282 : tensor<96xf32>
    %v4284 = stablehlo.subtract %d0ng, %v4283 : tensor<96xf32>
    %v4285 = stablehlo.reshape %v4208 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4286 = stablehlo.transpose %v4285, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4287 = stablehlo.reshape %v4286 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4289 = stablehlo.reshape %v4287 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4290 = stablehlo.reduce(%v4289 init: %v4288) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4291 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4292 = stablehlo.multiply %v4290, %v4291 : tensor<96xf32>
    %v4293 = stablehlo.subtract %d0nbt, %v4292 : tensor<96xf32>
    %v4294 = stablehlo.reshape %v273 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4295 = stablehlo.reshape %v4112 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4297 = stablehlo.pad %v4295, %v4296, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x55x55xf32>
    %v4298 = stablehlo.transpose %v4294, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4299 = stablehlo.transpose %v4297, dims = [1, 0, 2, 3] : (tensor<32x192x55x55xf32>) -> tensor<192x32x55x55xf32>
    %v4300 = stablehlo.convolution(%v4298, %v4299)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<192x32x55x55xf32>) -> tensor<96x192x2x2xf32>
    %v4301 = stablehlo.transpose %v4300, dims = [1, 0, 2, 3] : (tensor<96x192x2x2xf32>) -> tensor<192x96x2x2xf32>
    %v4302 = stablehlo.constant dense<0.1> : tensor<192x96x2x2xf32>
    %v4303 = stablehlo.multiply %v4301, %v4302 : tensor<192x96x2x2xf32>
    %v4304 = stablehlo.subtract %d0W, %v4303 : tensor<192x96x2x2xf32>
    %v4305 = stablehlo.reshape %v4251 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4306 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4307 = stablehlo.multiply %v4305, %v4306 : tensor<32x96x56x56xf32>
    %v4308 = stablehlo.reshape %v4307 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4309 = stablehlo.reshape %v4308 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4310 = stablehlo.transpose %s0b2pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4311 = stablehlo.reverse %v4310, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v4312 = stablehlo.convolution(%v4309, %v4311)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v4313 = stablehlo.reshape %v4312 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4314 = stablehlo.multiply %v216, %v216 : tensor<32x1204224xf32>
    %v4315 = stablehlo.multiply %v4314, %v216 : tensor<32x1204224xf32>
    %v4316 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v4317 = stablehlo.multiply %v4316, %v4315 : tensor<32x1204224xf32>
    %v4318 = stablehlo.add %v216, %v4317 : tensor<32x1204224xf32>
    %v4319 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v4320 = stablehlo.multiply %v4319, %v4318 : tensor<32x1204224xf32>
    %v4321 = stablehlo.tanh %v4320 : tensor<32x1204224xf32>
    %v4322 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v4323 = stablehlo.add %v4322, %v4321 : tensor<32x1204224xf32>
    %v4324 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v4325 = stablehlo.multiply %v4324, %v4323 : tensor<32x1204224xf32>
    %v4326 = stablehlo.multiply %v4321, %v4321 : tensor<32x1204224xf32>
    %v4327 = stablehlo.subtract %v4322, %v4326 : tensor<32x1204224xf32>
    %v4328 = stablehlo.multiply %v4324, %v216 : tensor<32x1204224xf32>
    %v4329 = stablehlo.multiply %v4328, %v4327 : tensor<32x1204224xf32>
    %v4330 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v4331 = stablehlo.multiply %v4330, %v4314 : tensor<32x1204224xf32>
    %v4332 = stablehlo.add %v4322, %v4331 : tensor<32x1204224xf32>
    %v4333 = stablehlo.multiply %v4319, %v4332 : tensor<32x1204224xf32>
    %v4334 = stablehlo.multiply %v4329, %v4333 : tensor<32x1204224xf32>
    %v4335 = stablehlo.add %v4325, %v4334 : tensor<32x1204224xf32>
    %v4336 = stablehlo.multiply %v4313, %v4335 : tensor<32x1204224xf32>
    %v4337 = stablehlo.reshape %v4336 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4338 = stablehlo.transpose %s0b2eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4339 = stablehlo.reverse %v4338, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v4340 = stablehlo.convolution(%v4337, %v4339)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4341 = stablehlo.reshape %v4340 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4342 = stablehlo.reshape %v177 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4343 = stablehlo.transpose %v4342, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4344 = stablehlo.reshape %v4343 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4345 = stablehlo.reshape %v4341 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4346 = stablehlo.transpose %v4345, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4347 = stablehlo.reshape %v4346 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4348 = stablehlo.reshape %v4347 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4349 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4350 = stablehlo.multiply %v4348, %v4349 : tensor<32x3136x96xf32>
    %v4351 = stablehlo.reshape %v4350 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4352 = stablehlo.reshape %v4351 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4353 = stablehlo.reshape %v4344 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4354 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4355 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4356 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4357 = stablehlo.reduce(%v4353 init: %v4354) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4358 = stablehlo.broadcast_in_dim %v4357, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4359 = stablehlo.divide %v4358, %v4355 : tensor<32x3136x96xf32>
    %v4360 = stablehlo.subtract %v4353, %v4359 : tensor<32x3136x96xf32>
    %v4361 = stablehlo.multiply %v4360, %v4360 : tensor<32x3136x96xf32>
    %v4362 = stablehlo.reduce(%v4361 init: %v4354) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4363 = stablehlo.broadcast_in_dim %v4362, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4364 = stablehlo.divide %v4363, %v4355 : tensor<32x3136x96xf32>
    %v4365 = stablehlo.add %v4364, %v4356 : tensor<32x3136x96xf32>
    %v4366 = stablehlo.rsqrt %v4365 : tensor<32x3136x96xf32>
    %v4367 = stablehlo.multiply %v4360, %v4366 : tensor<32x3136x96xf32>
    %v4368 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4369 = stablehlo.multiply %v4368, %v4352 : tensor<32x3136x96xf32>
    %v4370 = stablehlo.reduce(%v4369 init: %v4354) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4371 = stablehlo.broadcast_in_dim %v4370, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4372 = stablehlo.multiply %v4367, %v4369 : tensor<32x3136x96xf32>
    %v4373 = stablehlo.reduce(%v4372 init: %v4354) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4374 = stablehlo.broadcast_in_dim %v4373, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4375 = stablehlo.multiply %v4369, %v4355 : tensor<32x3136x96xf32>
    %v4376 = stablehlo.subtract %v4375, %v4371 : tensor<32x3136x96xf32>
    %v4377 = stablehlo.multiply %v4367, %v4374 : tensor<32x3136x96xf32>
    %v4378 = stablehlo.subtract %v4376, %v4377 : tensor<32x3136x96xf32>
    %v4379 = stablehlo.divide %v4366, %v4355 : tensor<32x3136x96xf32>
    %v4380 = stablehlo.multiply %v4379, %v4378 : tensor<32x3136x96xf32>
    %v4381 = stablehlo.reshape %v4380 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4382 = stablehlo.reshape %v4381 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4383 = stablehlo.transpose %v4382, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4384 = stablehlo.reshape %v4383 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4385 = stablehlo.reshape %v4384 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4386 = stablehlo.reverse %s0b2dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v4387 = stablehlo.convolution(%v4385, %v4386)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v4388 = stablehlo.reshape %v4387 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4389 = stablehlo.add %v4388, %v4251 : tensor<32x301056xf32>
    %v4390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4391 = stablehlo.reshape %v234 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4392 = stablehlo.reshape %v4251 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4393 = stablehlo.multiply %v4391, %v4392 : tensor<32x96x56x56xf32>
    %v4394 = stablehlo.reduce(%v4393 init: %v4390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4395 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4396 = stablehlo.multiply %v4394, %v4395 : tensor<96xf32>
    %v4397 = stablehlo.subtract %s0b2lg, %v4396 : tensor<96xf32>
    %v4398 = stablehlo.reshape %v229 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4399 = stablehlo.reshape %v4308 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4400 = stablehlo.transpose %v4398, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4401 = stablehlo.transpose %v4399, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4402 = stablehlo.convolution(%v4400, %v4401)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v4403 = stablehlo.transpose %v4402, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4404 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v4405 = stablehlo.multiply %v4403, %v4404 : tensor<96x384x1x1xf32>
    %v4406 = stablehlo.subtract %s0b2pW, %v4405 : tensor<96x384x1x1xf32>
    %v4407 = stablehlo.reshape %v4308 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4408 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4409 = stablehlo.reduce(%v4407 init: %v4408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4410 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4411 = stablehlo.multiply %v4409, %v4410 : tensor<96xf32>
    %v4412 = stablehlo.subtract %s0b2pb, %v4411 : tensor<96xf32>
    %v4413 = stablehlo.reshape %v211 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4414 = stablehlo.reshape %v4336 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4415 = stablehlo.transpose %v4413, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4416 = stablehlo.transpose %v4414, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4417 = stablehlo.convolution(%v4415, %v4416)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v4418 = stablehlo.transpose %v4417, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4419 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v4420 = stablehlo.multiply %v4418, %v4419 : tensor<384x96x1x1xf32>
    %v4421 = stablehlo.subtract %s0b2eW, %v4420 : tensor<384x96x1x1xf32>
    %v4422 = stablehlo.reshape %v4336 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4424 = stablehlo.reduce(%v4422 init: %v4423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v4425 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v4426 = stablehlo.multiply %v4424, %v4425 : tensor<384xf32>
    %v4427 = stablehlo.subtract %s0b2eb, %v4426 : tensor<384xf32>
    %v4428 = stablehlo.reshape %v177 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4429 = stablehlo.transpose %v4428, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4430 = stablehlo.reshape %v4429 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4431 = stablehlo.reshape %v4341 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4432 = stablehlo.transpose %v4431, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4433 = stablehlo.reshape %v4432 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4434 = stablehlo.reshape %v4430 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4435 = stablehlo.reshape %v4433 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4436 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4437 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4438 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4439 = stablehlo.reduce(%v4434 init: %v4436) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4440 = stablehlo.broadcast_in_dim %v4439, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4441 = stablehlo.divide %v4440, %v4437 : tensor<32x3136x96xf32>
    %v4442 = stablehlo.subtract %v4434, %v4441 : tensor<32x3136x96xf32>
    %v4443 = stablehlo.multiply %v4442, %v4442 : tensor<32x3136x96xf32>
    %v4444 = stablehlo.reduce(%v4443 init: %v4436) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4445 = stablehlo.broadcast_in_dim %v4444, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4446 = stablehlo.divide %v4445, %v4437 : tensor<32x3136x96xf32>
    %v4447 = stablehlo.add %v4446, %v4438 : tensor<32x3136x96xf32>
    %v4448 = stablehlo.rsqrt %v4447 : tensor<32x3136x96xf32>
    %v4449 = stablehlo.multiply %v4442, %v4448 : tensor<32x3136x96xf32>
    %v4450 = stablehlo.multiply %v4435, %v4449 : tensor<32x3136x96xf32>
    %v4451 = stablehlo.reduce(%v4450 init: %v4436) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4452 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4453 = stablehlo.multiply %v4451, %v4452 : tensor<96xf32>
    %v4454 = stablehlo.subtract %s0b2ng, %v4453 : tensor<96xf32>
    %v4455 = stablehlo.reshape %v4341 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4456 = stablehlo.transpose %v4455, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4457 = stablehlo.reshape %v4456 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4459 = stablehlo.reshape %v4457 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4460 = stablehlo.reduce(%v4459 init: %v4458) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4461 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4462 = stablehlo.multiply %v4460, %v4461 : tensor<96xf32>
    %v4463 = stablehlo.subtract %s0b2nbt, %v4462 : tensor<96xf32>
    %v4464 = stablehlo.reshape %v172 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4465 = stablehlo.reshape %v4384 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4466 = stablehlo.transpose %v4464, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4467 = stablehlo.transpose %v4465, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4468 = stablehlo.convolution(%v4466, %v4467)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v4469 = stablehlo.reshape %v4468 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v4470 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v4471 = stablehlo.multiply %v4469, %v4470 : tensor<96x1x7x7xf32>
    %v4472 = stablehlo.subtract %s0b2dW, %v4471 : tensor<96x1x7x7xf32>
    %v4473 = stablehlo.reshape %v4384 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4475 = stablehlo.reduce(%v4473 init: %v4474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4476 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4477 = stablehlo.multiply %v4475, %v4476 : tensor<96xf32>
    %v4478 = stablehlo.subtract %s0b2db, %v4477 : tensor<96xf32>
    %v4479 = stablehlo.reshape %v4389 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4480 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4481 = stablehlo.multiply %v4479, %v4480 : tensor<32x96x56x56xf32>
    %v4482 = stablehlo.reshape %v4481 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4483 = stablehlo.reshape %v4482 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4484 = stablehlo.transpose %s0b1pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4485 = stablehlo.reverse %v4484, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v4486 = stablehlo.convolution(%v4483, %v4485)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v4487 = stablehlo.reshape %v4486 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4488 = stablehlo.multiply %v149, %v149 : tensor<32x1204224xf32>
    %v4489 = stablehlo.multiply %v4488, %v149 : tensor<32x1204224xf32>
    %v4490 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v4491 = stablehlo.multiply %v4490, %v4489 : tensor<32x1204224xf32>
    %v4492 = stablehlo.add %v149, %v4491 : tensor<32x1204224xf32>
    %v4493 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v4494 = stablehlo.multiply %v4493, %v4492 : tensor<32x1204224xf32>
    %v4495 = stablehlo.tanh %v4494 : tensor<32x1204224xf32>
    %v4496 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v4497 = stablehlo.add %v4496, %v4495 : tensor<32x1204224xf32>
    %v4498 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v4499 = stablehlo.multiply %v4498, %v4497 : tensor<32x1204224xf32>
    %v4500 = stablehlo.multiply %v4495, %v4495 : tensor<32x1204224xf32>
    %v4501 = stablehlo.subtract %v4496, %v4500 : tensor<32x1204224xf32>
    %v4502 = stablehlo.multiply %v4498, %v149 : tensor<32x1204224xf32>
    %v4503 = stablehlo.multiply %v4502, %v4501 : tensor<32x1204224xf32>
    %v4504 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v4505 = stablehlo.multiply %v4504, %v4488 : tensor<32x1204224xf32>
    %v4506 = stablehlo.add %v4496, %v4505 : tensor<32x1204224xf32>
    %v4507 = stablehlo.multiply %v4493, %v4506 : tensor<32x1204224xf32>
    %v4508 = stablehlo.multiply %v4503, %v4507 : tensor<32x1204224xf32>
    %v4509 = stablehlo.add %v4499, %v4508 : tensor<32x1204224xf32>
    %v4510 = stablehlo.multiply %v4487, %v4509 : tensor<32x1204224xf32>
    %v4511 = stablehlo.reshape %v4510 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4512 = stablehlo.transpose %s0b1eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4513 = stablehlo.reverse %v4512, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v4514 = stablehlo.convolution(%v4511, %v4513)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4515 = stablehlo.reshape %v4514 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4516 = stablehlo.reshape %v110 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4517 = stablehlo.transpose %v4516, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4518 = stablehlo.reshape %v4517 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4519 = stablehlo.reshape %v4515 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4520 = stablehlo.transpose %v4519, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4521 = stablehlo.reshape %v4520 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4522 = stablehlo.reshape %v4521 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4523 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4524 = stablehlo.multiply %v4522, %v4523 : tensor<32x3136x96xf32>
    %v4525 = stablehlo.reshape %v4524 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4526 = stablehlo.reshape %v4525 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4527 = stablehlo.reshape %v4518 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4529 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4530 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4531 = stablehlo.reduce(%v4527 init: %v4528) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4532 = stablehlo.broadcast_in_dim %v4531, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4533 = stablehlo.divide %v4532, %v4529 : tensor<32x3136x96xf32>
    %v4534 = stablehlo.subtract %v4527, %v4533 : tensor<32x3136x96xf32>
    %v4535 = stablehlo.multiply %v4534, %v4534 : tensor<32x3136x96xf32>
    %v4536 = stablehlo.reduce(%v4535 init: %v4528) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4537 = stablehlo.broadcast_in_dim %v4536, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4538 = stablehlo.divide %v4537, %v4529 : tensor<32x3136x96xf32>
    %v4539 = stablehlo.add %v4538, %v4530 : tensor<32x3136x96xf32>
    %v4540 = stablehlo.rsqrt %v4539 : tensor<32x3136x96xf32>
    %v4541 = stablehlo.multiply %v4534, %v4540 : tensor<32x3136x96xf32>
    %v4542 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4543 = stablehlo.multiply %v4542, %v4526 : tensor<32x3136x96xf32>
    %v4544 = stablehlo.reduce(%v4543 init: %v4528) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4545 = stablehlo.broadcast_in_dim %v4544, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4546 = stablehlo.multiply %v4541, %v4543 : tensor<32x3136x96xf32>
    %v4547 = stablehlo.reduce(%v4546 init: %v4528) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4548 = stablehlo.broadcast_in_dim %v4547, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4549 = stablehlo.multiply %v4543, %v4529 : tensor<32x3136x96xf32>
    %v4550 = stablehlo.subtract %v4549, %v4545 : tensor<32x3136x96xf32>
    %v4551 = stablehlo.multiply %v4541, %v4548 : tensor<32x3136x96xf32>
    %v4552 = stablehlo.subtract %v4550, %v4551 : tensor<32x3136x96xf32>
    %v4553 = stablehlo.divide %v4540, %v4529 : tensor<32x3136x96xf32>
    %v4554 = stablehlo.multiply %v4553, %v4552 : tensor<32x3136x96xf32>
    %v4555 = stablehlo.reshape %v4554 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4556 = stablehlo.reshape %v4555 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4557 = stablehlo.transpose %v4556, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4558 = stablehlo.reshape %v4557 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4559 = stablehlo.reshape %v4558 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4560 = stablehlo.reverse %s0b1dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v4561 = stablehlo.convolution(%v4559, %v4560)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v4562 = stablehlo.reshape %v4561 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4563 = stablehlo.add %v4562, %v4389 : tensor<32x301056xf32>
    %v4564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4565 = stablehlo.reshape %v167 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4566 = stablehlo.reshape %v4389 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4567 = stablehlo.multiply %v4565, %v4566 : tensor<32x96x56x56xf32>
    %v4568 = stablehlo.reduce(%v4567 init: %v4564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4569 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4570 = stablehlo.multiply %v4568, %v4569 : tensor<96xf32>
    %v4571 = stablehlo.subtract %s0b1lg, %v4570 : tensor<96xf32>
    %v4572 = stablehlo.reshape %v162 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4573 = stablehlo.reshape %v4482 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4574 = stablehlo.transpose %v4572, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4575 = stablehlo.transpose %v4573, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4576 = stablehlo.convolution(%v4574, %v4575)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v4577 = stablehlo.transpose %v4576, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4578 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v4579 = stablehlo.multiply %v4577, %v4578 : tensor<96x384x1x1xf32>
    %v4580 = stablehlo.subtract %s0b1pW, %v4579 : tensor<96x384x1x1xf32>
    %v4581 = stablehlo.reshape %v4482 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4583 = stablehlo.reduce(%v4581 init: %v4582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4584 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4585 = stablehlo.multiply %v4583, %v4584 : tensor<96xf32>
    %v4586 = stablehlo.subtract %s0b1pb, %v4585 : tensor<96xf32>
    %v4587 = stablehlo.reshape %v144 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4588 = stablehlo.reshape %v4510 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4589 = stablehlo.transpose %v4587, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4590 = stablehlo.transpose %v4588, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4591 = stablehlo.convolution(%v4589, %v4590)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v4592 = stablehlo.transpose %v4591, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4593 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v4594 = stablehlo.multiply %v4592, %v4593 : tensor<384x96x1x1xf32>
    %v4595 = stablehlo.subtract %s0b1eW, %v4594 : tensor<384x96x1x1xf32>
    %v4596 = stablehlo.reshape %v4510 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4598 = stablehlo.reduce(%v4596 init: %v4597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v4599 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v4600 = stablehlo.multiply %v4598, %v4599 : tensor<384xf32>
    %v4601 = stablehlo.subtract %s0b1eb, %v4600 : tensor<384xf32>
    %v4602 = stablehlo.reshape %v110 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4603 = stablehlo.transpose %v4602, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4604 = stablehlo.reshape %v4603 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4605 = stablehlo.reshape %v4515 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4606 = stablehlo.transpose %v4605, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4607 = stablehlo.reshape %v4606 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4608 = stablehlo.reshape %v4604 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4609 = stablehlo.reshape %v4607 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4610 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4611 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4612 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4613 = stablehlo.reduce(%v4608 init: %v4610) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4614 = stablehlo.broadcast_in_dim %v4613, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4615 = stablehlo.divide %v4614, %v4611 : tensor<32x3136x96xf32>
    %v4616 = stablehlo.subtract %v4608, %v4615 : tensor<32x3136x96xf32>
    %v4617 = stablehlo.multiply %v4616, %v4616 : tensor<32x3136x96xf32>
    %v4618 = stablehlo.reduce(%v4617 init: %v4610) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4619 = stablehlo.broadcast_in_dim %v4618, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4620 = stablehlo.divide %v4619, %v4611 : tensor<32x3136x96xf32>
    %v4621 = stablehlo.add %v4620, %v4612 : tensor<32x3136x96xf32>
    %v4622 = stablehlo.rsqrt %v4621 : tensor<32x3136x96xf32>
    %v4623 = stablehlo.multiply %v4616, %v4622 : tensor<32x3136x96xf32>
    %v4624 = stablehlo.multiply %v4609, %v4623 : tensor<32x3136x96xf32>
    %v4625 = stablehlo.reduce(%v4624 init: %v4610) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4626 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4627 = stablehlo.multiply %v4625, %v4626 : tensor<96xf32>
    %v4628 = stablehlo.subtract %s0b1ng, %v4627 : tensor<96xf32>
    %v4629 = stablehlo.reshape %v4515 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4630 = stablehlo.transpose %v4629, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4631 = stablehlo.reshape %v4630 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4632 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4633 = stablehlo.reshape %v4631 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4634 = stablehlo.reduce(%v4633 init: %v4632) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4635 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4636 = stablehlo.multiply %v4634, %v4635 : tensor<96xf32>
    %v4637 = stablehlo.subtract %s0b1nbt, %v4636 : tensor<96xf32>
    %v4638 = stablehlo.reshape %v105 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4639 = stablehlo.reshape %v4558 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4640 = stablehlo.transpose %v4638, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4641 = stablehlo.transpose %v4639, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4642 = stablehlo.convolution(%v4640, %v4641)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v4643 = stablehlo.reshape %v4642 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v4644 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v4645 = stablehlo.multiply %v4643, %v4644 : tensor<96x1x7x7xf32>
    %v4646 = stablehlo.subtract %s0b1dW, %v4645 : tensor<96x1x7x7xf32>
    %v4647 = stablehlo.reshape %v4558 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4648 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4649 = stablehlo.reduce(%v4647 init: %v4648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4650 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4651 = stablehlo.multiply %v4649, %v4650 : tensor<96xf32>
    %v4652 = stablehlo.subtract %s0b1db, %v4651 : tensor<96xf32>
    %v4653 = stablehlo.reshape %v4563 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4654 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4655 = stablehlo.multiply %v4653, %v4654 : tensor<32x96x56x56xf32>
    %v4656 = stablehlo.reshape %v4655 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4657 = stablehlo.reshape %v4656 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4658 = stablehlo.transpose %s0b0pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4659 = stablehlo.reverse %v4658, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v4660 = stablehlo.convolution(%v4657, %v4659)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v4661 = stablehlo.reshape %v4660 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4662 = stablehlo.multiply %v82, %v82 : tensor<32x1204224xf32>
    %v4663 = stablehlo.multiply %v4662, %v82 : tensor<32x1204224xf32>
    %v4664 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v4665 = stablehlo.multiply %v4664, %v4663 : tensor<32x1204224xf32>
    %v4666 = stablehlo.add %v82, %v4665 : tensor<32x1204224xf32>
    %v4667 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v4668 = stablehlo.multiply %v4667, %v4666 : tensor<32x1204224xf32>
    %v4669 = stablehlo.tanh %v4668 : tensor<32x1204224xf32>
    %v4670 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v4671 = stablehlo.add %v4670, %v4669 : tensor<32x1204224xf32>
    %v4672 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v4673 = stablehlo.multiply %v4672, %v4671 : tensor<32x1204224xf32>
    %v4674 = stablehlo.multiply %v4669, %v4669 : tensor<32x1204224xf32>
    %v4675 = stablehlo.subtract %v4670, %v4674 : tensor<32x1204224xf32>
    %v4676 = stablehlo.multiply %v4672, %v82 : tensor<32x1204224xf32>
    %v4677 = stablehlo.multiply %v4676, %v4675 : tensor<32x1204224xf32>
    %v4678 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v4679 = stablehlo.multiply %v4678, %v4662 : tensor<32x1204224xf32>
    %v4680 = stablehlo.add %v4670, %v4679 : tensor<32x1204224xf32>
    %v4681 = stablehlo.multiply %v4667, %v4680 : tensor<32x1204224xf32>
    %v4682 = stablehlo.multiply %v4677, %v4681 : tensor<32x1204224xf32>
    %v4683 = stablehlo.add %v4673, %v4682 : tensor<32x1204224xf32>
    %v4684 = stablehlo.multiply %v4661, %v4683 : tensor<32x1204224xf32>
    %v4685 = stablehlo.reshape %v4684 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4686 = stablehlo.transpose %s0b0eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4687 = stablehlo.reverse %v4686, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v4688 = stablehlo.convolution(%v4685, %v4687)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4689 = stablehlo.reshape %v4688 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4690 = stablehlo.reshape %v43 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4691 = stablehlo.transpose %v4690, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4692 = stablehlo.reshape %v4691 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4693 = stablehlo.reshape %v4689 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4694 = stablehlo.transpose %v4693, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4695 = stablehlo.reshape %v4694 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4696 = stablehlo.reshape %v4695 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4697 = stablehlo.broadcast_in_dim %s0b0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4698 = stablehlo.multiply %v4696, %v4697 : tensor<32x3136x96xf32>
    %v4699 = stablehlo.reshape %v4698 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4700 = stablehlo.reshape %v4699 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4701 = stablehlo.reshape %v4692 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4703 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4704 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4705 = stablehlo.reduce(%v4701 init: %v4702) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4706 = stablehlo.broadcast_in_dim %v4705, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4707 = stablehlo.divide %v4706, %v4703 : tensor<32x3136x96xf32>
    %v4708 = stablehlo.subtract %v4701, %v4707 : tensor<32x3136x96xf32>
    %v4709 = stablehlo.multiply %v4708, %v4708 : tensor<32x3136x96xf32>
    %v4710 = stablehlo.reduce(%v4709 init: %v4702) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4711 = stablehlo.broadcast_in_dim %v4710, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4712 = stablehlo.divide %v4711, %v4703 : tensor<32x3136x96xf32>
    %v4713 = stablehlo.add %v4712, %v4704 : tensor<32x3136x96xf32>
    %v4714 = stablehlo.rsqrt %v4713 : tensor<32x3136x96xf32>
    %v4715 = stablehlo.multiply %v4708, %v4714 : tensor<32x3136x96xf32>
    %v4716 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4717 = stablehlo.multiply %v4716, %v4700 : tensor<32x3136x96xf32>
    %v4718 = stablehlo.reduce(%v4717 init: %v4702) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4719 = stablehlo.broadcast_in_dim %v4718, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4720 = stablehlo.multiply %v4715, %v4717 : tensor<32x3136x96xf32>
    %v4721 = stablehlo.reduce(%v4720 init: %v4702) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4722 = stablehlo.broadcast_in_dim %v4721, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4723 = stablehlo.multiply %v4717, %v4703 : tensor<32x3136x96xf32>
    %v4724 = stablehlo.subtract %v4723, %v4719 : tensor<32x3136x96xf32>
    %v4725 = stablehlo.multiply %v4715, %v4722 : tensor<32x3136x96xf32>
    %v4726 = stablehlo.subtract %v4724, %v4725 : tensor<32x3136x96xf32>
    %v4727 = stablehlo.divide %v4714, %v4703 : tensor<32x3136x96xf32>
    %v4728 = stablehlo.multiply %v4727, %v4726 : tensor<32x3136x96xf32>
    %v4729 = stablehlo.reshape %v4728 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4730 = stablehlo.reshape %v4729 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4731 = stablehlo.transpose %v4730, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4732 = stablehlo.reshape %v4731 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4733 = stablehlo.reshape %v4732 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4734 = stablehlo.reverse %s0b0dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v4735 = stablehlo.convolution(%v4733, %v4734)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v4736 = stablehlo.reshape %v4735 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4737 = stablehlo.add %v4736, %v4563 : tensor<32x301056xf32>
    %v4738 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4739 = stablehlo.reshape %v100 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4740 = stablehlo.reshape %v4563 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4741 = stablehlo.multiply %v4739, %v4740 : tensor<32x96x56x56xf32>
    %v4742 = stablehlo.reduce(%v4741 init: %v4738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4743 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4744 = stablehlo.multiply %v4742, %v4743 : tensor<96xf32>
    %v4745 = stablehlo.subtract %s0b0lg, %v4744 : tensor<96xf32>
    %v4746 = stablehlo.reshape %v95 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4747 = stablehlo.reshape %v4656 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4748 = stablehlo.transpose %v4746, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4749 = stablehlo.transpose %v4747, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4750 = stablehlo.convolution(%v4748, %v4749)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v4751 = stablehlo.transpose %v4750, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4752 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v4753 = stablehlo.multiply %v4751, %v4752 : tensor<96x384x1x1xf32>
    %v4754 = stablehlo.subtract %s0b0pW, %v4753 : tensor<96x384x1x1xf32>
    %v4755 = stablehlo.reshape %v4656 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4757 = stablehlo.reduce(%v4755 init: %v4756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4758 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4759 = stablehlo.multiply %v4757, %v4758 : tensor<96xf32>
    %v4760 = stablehlo.subtract %s0b0pb, %v4759 : tensor<96xf32>
    %v4761 = stablehlo.reshape %v77 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4762 = stablehlo.reshape %v4684 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4763 = stablehlo.transpose %v4761, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4764 = stablehlo.transpose %v4762, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4765 = stablehlo.convolution(%v4763, %v4764)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v4766 = stablehlo.transpose %v4765, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4767 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v4768 = stablehlo.multiply %v4766, %v4767 : tensor<384x96x1x1xf32>
    %v4769 = stablehlo.subtract %s0b0eW, %v4768 : tensor<384x96x1x1xf32>
    %v4770 = stablehlo.reshape %v4684 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4772 = stablehlo.reduce(%v4770 init: %v4771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v4773 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v4774 = stablehlo.multiply %v4772, %v4773 : tensor<384xf32>
    %v4775 = stablehlo.subtract %s0b0eb, %v4774 : tensor<384xf32>
    %v4776 = stablehlo.reshape %v43 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4777 = stablehlo.transpose %v4776, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4778 = stablehlo.reshape %v4777 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4779 = stablehlo.reshape %v4689 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4780 = stablehlo.transpose %v4779, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4781 = stablehlo.reshape %v4780 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4782 = stablehlo.reshape %v4778 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4783 = stablehlo.reshape %v4781 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4785 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4786 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4787 = stablehlo.reduce(%v4782 init: %v4784) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4788 = stablehlo.broadcast_in_dim %v4787, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4789 = stablehlo.divide %v4788, %v4785 : tensor<32x3136x96xf32>
    %v4790 = stablehlo.subtract %v4782, %v4789 : tensor<32x3136x96xf32>
    %v4791 = stablehlo.multiply %v4790, %v4790 : tensor<32x3136x96xf32>
    %v4792 = stablehlo.reduce(%v4791 init: %v4784) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4793 = stablehlo.broadcast_in_dim %v4792, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4794 = stablehlo.divide %v4793, %v4785 : tensor<32x3136x96xf32>
    %v4795 = stablehlo.add %v4794, %v4786 : tensor<32x3136x96xf32>
    %v4796 = stablehlo.rsqrt %v4795 : tensor<32x3136x96xf32>
    %v4797 = stablehlo.multiply %v4790, %v4796 : tensor<32x3136x96xf32>
    %v4798 = stablehlo.multiply %v4783, %v4797 : tensor<32x3136x96xf32>
    %v4799 = stablehlo.reduce(%v4798 init: %v4784) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4800 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4801 = stablehlo.multiply %v4799, %v4800 : tensor<96xf32>
    %v4802 = stablehlo.subtract %s0b0ng, %v4801 : tensor<96xf32>
    %v4803 = stablehlo.reshape %v4689 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4804 = stablehlo.transpose %v4803, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4805 = stablehlo.reshape %v4804 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4807 = stablehlo.reshape %v4805 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4808 = stablehlo.reduce(%v4807 init: %v4806) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4809 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4810 = stablehlo.multiply %v4808, %v4809 : tensor<96xf32>
    %v4811 = stablehlo.subtract %s0b0nbt, %v4810 : tensor<96xf32>
    %v4812 = stablehlo.reshape %v38 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4813 = stablehlo.reshape %v4732 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4814 = stablehlo.transpose %v4812, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4815 = stablehlo.transpose %v4813, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4816 = stablehlo.convolution(%v4814, %v4815)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v4817 = stablehlo.reshape %v4816 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v4818 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v4819 = stablehlo.multiply %v4817, %v4818 : tensor<96x1x7x7xf32>
    %v4820 = stablehlo.subtract %s0b0dW, %v4819 : tensor<96x1x7x7xf32>
    %v4821 = stablehlo.reshape %v4732 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4823 = stablehlo.reduce(%v4821 init: %v4822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4824 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4825 = stablehlo.multiply %v4823, %v4824 : tensor<96xf32>
    %v4826 = stablehlo.subtract %s0b0db, %v4825 : tensor<96xf32>
    %v4827 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4828 = stablehlo.transpose %v4827, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4829 = stablehlo.reshape %v4828 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4830 = stablehlo.reshape %v4737 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4831 = stablehlo.transpose %v4830, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4832 = stablehlo.reshape %v4831 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4833 = stablehlo.reshape %v4829 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4834 = stablehlo.reshape %v4832 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4835 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4836 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4837 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4838 = stablehlo.reduce(%v4833 init: %v4835) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4839 = stablehlo.broadcast_in_dim %v4838, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4840 = stablehlo.divide %v4839, %v4836 : tensor<32x3136x96xf32>
    %v4841 = stablehlo.subtract %v4833, %v4840 : tensor<32x3136x96xf32>
    %v4842 = stablehlo.multiply %v4841, %v4841 : tensor<32x3136x96xf32>
    %v4843 = stablehlo.reduce(%v4842 init: %v4835) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4844 = stablehlo.broadcast_in_dim %v4843, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4845 = stablehlo.divide %v4844, %v4836 : tensor<32x3136x96xf32>
    %v4846 = stablehlo.add %v4845, %v4837 : tensor<32x3136x96xf32>
    %v4847 = stablehlo.rsqrt %v4846 : tensor<32x3136x96xf32>
    %v4848 = stablehlo.multiply %v4841, %v4847 : tensor<32x3136x96xf32>
    %v4849 = stablehlo.multiply %v4834, %v4848 : tensor<32x3136x96xf32>
    %v4850 = stablehlo.reduce(%v4849 init: %v4835) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4851 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4852 = stablehlo.multiply %v4850, %v4851 : tensor<96xf32>
    %v4853 = stablehlo.subtract %psng, %v4852 : tensor<96xf32>
    %v4854 = stablehlo.reshape %v4737 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4855 = stablehlo.transpose %v4854, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4856 = stablehlo.reshape %v4855 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4857 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4858 = stablehlo.reshape %v4856 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4859 = stablehlo.reduce(%v4858 init: %v4857) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4860 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4861 = stablehlo.multiply %v4859, %v4860 : tensor<96xf32>
    %v4862 = stablehlo.subtract %psnbt, %v4861 : tensor<96xf32>
    %v4863 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4864 = stablehlo.transpose %v4863, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4865 = stablehlo.reshape %v4864 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4866 = stablehlo.reshape %v4737 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4867 = stablehlo.transpose %v4866, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4868 = stablehlo.reshape %v4867 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4869 = stablehlo.reshape %v4868 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4870 = stablehlo.broadcast_in_dim %psng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4871 = stablehlo.multiply %v4869, %v4870 : tensor<32x3136x96xf32>
    %v4872 = stablehlo.reshape %v4871 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4873 = stablehlo.reshape %v4872 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4874 = stablehlo.reshape %v4865 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4876 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4877 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4878 = stablehlo.reduce(%v4874 init: %v4875) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4879 = stablehlo.broadcast_in_dim %v4878, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4880 = stablehlo.divide %v4879, %v4876 : tensor<32x3136x96xf32>
    %v4881 = stablehlo.subtract %v4874, %v4880 : tensor<32x3136x96xf32>
    %v4882 = stablehlo.multiply %v4881, %v4881 : tensor<32x3136x96xf32>
    %v4883 = stablehlo.reduce(%v4882 init: %v4875) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4884 = stablehlo.broadcast_in_dim %v4883, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4885 = stablehlo.divide %v4884, %v4876 : tensor<32x3136x96xf32>
    %v4886 = stablehlo.add %v4885, %v4877 : tensor<32x3136x96xf32>
    %v4887 = stablehlo.rsqrt %v4886 : tensor<32x3136x96xf32>
    %v4888 = stablehlo.multiply %v4881, %v4887 : tensor<32x3136x96xf32>
    %v4889 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4890 = stablehlo.multiply %v4889, %v4873 : tensor<32x3136x96xf32>
    %v4891 = stablehlo.reduce(%v4890 init: %v4875) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4892 = stablehlo.broadcast_in_dim %v4891, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4893 = stablehlo.multiply %v4888, %v4890 : tensor<32x3136x96xf32>
    %v4894 = stablehlo.reduce(%v4893 init: %v4875) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4895 = stablehlo.broadcast_in_dim %v4894, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4896 = stablehlo.multiply %v4890, %v4876 : tensor<32x3136x96xf32>
    %v4897 = stablehlo.subtract %v4896, %v4892 : tensor<32x3136x96xf32>
    %v4898 = stablehlo.multiply %v4888, %v4895 : tensor<32x3136x96xf32>
    %v4899 = stablehlo.subtract %v4897, %v4898 : tensor<32x3136x96xf32>
    %v4900 = stablehlo.divide %v4887, %v4876 : tensor<32x3136x96xf32>
    %v4901 = stablehlo.multiply %v4900, %v4899 : tensor<32x3136x96xf32>
    %v4902 = stablehlo.reshape %v4901 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4903 = stablehlo.reshape %v4902 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4904 = stablehlo.transpose %v4903, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4905 = stablehlo.reshape %v4904 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4912 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v4913 = stablehlo.reshape %v4905 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4914 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4915 = stablehlo.pad %v4913, %v4914, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x221x221xf32>
    %v4916 = stablehlo.transpose %v4912, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v4917 = stablehlo.transpose %v4915, dims = [1, 0, 2, 3] : (tensor<32x96x221x221xf32>) -> tensor<96x32x221x221xf32>
    %v4918 = stablehlo.convolution(%v4916, %v4917)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<96x32x221x221xf32>) -> tensor<3x96x4x4xf32>
    %v4919 = stablehlo.transpose %v4918, dims = [1, 0, 2, 3] : (tensor<3x96x4x4xf32>) -> tensor<96x3x4x4xf32>
    %psWl = stablehlo.constant dense<0.1> : tensor<96x3x4x4xf32>
    %psWs = stablehlo.multiply %v4919, %psWl : tensor<96x3x4x4xf32>
    %psWn = stablehlo.subtract %psW, %psWs : tensor<96x3x4x4xf32>
    %v4906 = stablehlo.reshape %v4905 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4907 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4908 = stablehlo.reduce(%v4906 init: %v4907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4909 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4910 = stablehlo.multiply %v4908, %v4909 : tensor<96xf32>
    %v4911 = stablehlo.subtract %psb, %v4910 : tensor<96xf32>
    return %psWn, %v4911, %v4853, %v4862, %v4820, %v4826, %v4802, %v4811, %v4769, %v4775, %v4754, %v4760, %v4745, %v4646, %v4652, %v4628, %v4637, %v4595, %v4601, %v4580, %v4586, %v4571, %v4472, %v4478, %v4454, %v4463, %v4421, %v4427, %v4406, %v4412, %v4397, %v4284, %v4293, %v4304, %v4257, %v4195, %v4201, %v4177, %v4186, %v4144, %v4150, %v4129, %v4135, %v4120, %v4021, %v4027, %v4003, %v4012, %v3970, %v3976, %v3955, %v3961, %v3946, %v3847, %v3853, %v3829, %v3838, %v3796, %v3802, %v3781, %v3787, %v3772, %v3659, %v3668, %v3679, %v3632, %v3570, %v3576, %v3552, %v3561, %v3519, %v3525, %v3504, %v3510, %v3495, %v3396, %v3402, %v3378, %v3387, %v3345, %v3351, %v3330, %v3336, %v3321, %v3222, %v3228, %v3204, %v3213, %v3171, %v3177, %v3156, %v3162, %v3147, %v3048, %v3054, %v3030, %v3039, %v2997, %v3003, %v2982, %v2988, %v2973, %v2874, %v2880, %v2856, %v2865, %v2823, %v2829, %v2808, %v2814, %v2799, %v2700, %v2706, %v2682, %v2691, %v2649, %v2655, %v2634, %v2640, %v2625, %v2526, %v2532, %v2508, %v2517, %v2475, %v2481, %v2460, %v2466, %v2451, %v2352, %v2358, %v2334, %v2343, %v2301, %v2307, %v2286, %v2292, %v2277, %v2178, %v2184, %v2160, %v2169, %v2127, %v2133, %v2112, %v2118, %v2103, %v1990, %v1999, %v2010, %v1963, %v1901, %v1907, %v1883, %v1892, %v1850, %v1856, %v1835, %v1841, %v1826, %v1727, %v1733, %v1709, %v1718, %v1676, %v1682, %v1661, %v1667, %v1652, %v1553, %v1559, %v1535, %v1544, %v1502, %v1508, %v1487, %v1493, %v1478, %v1380, %v1385 : tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x10xf32>, tensor<10xf32>
  }
}
