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
    %v1457 = stablehlo.dot_general %v1456, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x10xf32>) -> tensor<32x10xf32>
    %v1458 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1459 = stablehlo.add %v1457, %v1458 : tensor<32x10xf32>
    %v1460 = stablehlo.exponential %v1459 : tensor<32x10xf32>
    %v1461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1462 = stablehlo.reduce(%v1460 init: %v1461) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v1463 = stablehlo.broadcast_in_dim %v1462, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v1464 = stablehlo.divide %v1460, %v1463 : tensor<32x10xf32>
    %v1465 = stablehlo.subtract %v1464, %onehot : tensor<32x10xf32>
    %dy = stablehlo.divide %v1465, %bsc : tensor<32x10xf32>
    %v1466 = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<768x10xf32>) -> tensor<32x768xf32>
    %v1467 = stablehlo.dot_general %v1456, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<32x10xf32>) -> tensor<768x10xf32>
    %v1468 = stablehlo.constant dense<0.1> : tensor<768x10xf32>
    %v1469 = stablehlo.multiply %v1467, %v1468 : tensor<768x10xf32>
    %v1470 = stablehlo.subtract %Wd, %v1469 : tensor<768x10xf32>
    %v1471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1472 = stablehlo.reduce(%dy init: %v1471) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1473 = stablehlo.constant dense<0.1> : tensor<10xf32>
    %v1474 = stablehlo.multiply %v1472, %v1473 : tensor<10xf32>
    %v1475 = stablehlo.subtract %bd, %v1474 : tensor<10xf32>
    %dgi = stablehlo.reshape %v1466 : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : (tensor<32x768x1x1xf32>) -> tensor<32x768x7x7xf32>
    %dgn = stablehlo.constant dense<49.0> : tensor<32x768x7x7xf32>
    %dgd = stablehlo.divide %dgb, %dgn : tensor<32x768x7x7xf32>
    %dgapf = stablehlo.reshape %dgd : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1476 = stablehlo.reshape %dgapf : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1477 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1478 = stablehlo.multiply %v1476, %v1477 : tensor<32x768x7x7xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1481 = stablehlo.transpose %s3b2pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1482 = stablehlo.reverse %v1481, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1483 = stablehlo.convolution(%v1480, %v1482)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1485 = stablehlo.reshape %v1484 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1486 = stablehlo.reshape %v1423 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1487 = stablehlo.multiply %v1486, %v1486 : tensor<32x3072x7x7xf32>
    %v1488 = stablehlo.multiply %v1487, %v1486 : tensor<32x3072x7x7xf32>
    %v1489 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1490 = stablehlo.multiply %v1489, %v1488 : tensor<32x3072x7x7xf32>
    %v1491 = stablehlo.add %v1486, %v1490 : tensor<32x3072x7x7xf32>
    %v1492 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1493 = stablehlo.multiply %v1492, %v1491 : tensor<32x3072x7x7xf32>
    %v1494 = stablehlo.tanh %v1493 : tensor<32x3072x7x7xf32>
    %v1495 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1496 = stablehlo.add %v1495, %v1494 : tensor<32x3072x7x7xf32>
    %v1497 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1498 = stablehlo.multiply %v1497, %v1496 : tensor<32x3072x7x7xf32>
    %v1499 = stablehlo.multiply %v1494, %v1494 : tensor<32x3072x7x7xf32>
    %v1500 = stablehlo.subtract %v1495, %v1499 : tensor<32x3072x7x7xf32>
    %v1501 = stablehlo.multiply %v1497, %v1486 : tensor<32x3072x7x7xf32>
    %v1502 = stablehlo.multiply %v1501, %v1500 : tensor<32x3072x7x7xf32>
    %v1503 = stablehlo.constant dense<0.134145> : tensor<32x3072x7x7xf32>
    %v1504 = stablehlo.multiply %v1503, %v1487 : tensor<32x3072x7x7xf32>
    %v1505 = stablehlo.add %v1495, %v1504 : tensor<32x3072x7x7xf32>
    %v1506 = stablehlo.multiply %v1492, %v1505 : tensor<32x3072x7x7xf32>
    %v1507 = stablehlo.multiply %v1502, %v1506 : tensor<32x3072x7x7xf32>
    %v1508 = stablehlo.add %v1498, %v1507 : tensor<32x3072x7x7xf32>
    %v1509 = stablehlo.multiply %v1485, %v1508 : tensor<32x3072x7x7xf32>
    %v1510 = stablehlo.reshape %v1509 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1512 = stablehlo.transpose %s3b2eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1513 = stablehlo.reverse %v1512, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1514 = stablehlo.convolution(%v1511, %v1513)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1515 = stablehlo.reshape %v1514 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1516 = stablehlo.reshape %v1384 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1517 = stablehlo.transpose %v1516, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1518 = stablehlo.reshape %v1517 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1519 = stablehlo.reshape %v1515 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1520 = stablehlo.transpose %v1519, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1523 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1524 = stablehlo.multiply %v1522, %v1523 : tensor<32x49x768xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1527 = stablehlo.reshape %v1518 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1529 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1530 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1531 = stablehlo.reduce(%v1527 init: %v1528) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1532 = stablehlo.broadcast_in_dim %v1531, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1533 = stablehlo.divide %v1532, %v1529 : tensor<32x49x768xf32>
    %v1534 = stablehlo.subtract %v1527, %v1533 : tensor<32x49x768xf32>
    %v1535 = stablehlo.multiply %v1534, %v1534 : tensor<32x49x768xf32>
    %v1536 = stablehlo.reduce(%v1535 init: %v1528) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1537 = stablehlo.broadcast_in_dim %v1536, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1538 = stablehlo.divide %v1537, %v1529 : tensor<32x49x768xf32>
    %v1539 = stablehlo.add %v1538, %v1530 : tensor<32x49x768xf32>
    %v1540 = stablehlo.rsqrt %v1539 : tensor<32x49x768xf32>
    %v1541 = stablehlo.multiply %v1534, %v1540 : tensor<32x49x768xf32>
    %v1542 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1543 = stablehlo.multiply %v1542, %v1526 : tensor<32x49x768xf32>
    %v1544 = stablehlo.reduce(%v1543 init: %v1528) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1545 = stablehlo.broadcast_in_dim %v1544, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1546 = stablehlo.multiply %v1541, %v1543 : tensor<32x49x768xf32>
    %v1547 = stablehlo.reduce(%v1546 init: %v1528) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1548 = stablehlo.broadcast_in_dim %v1547, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1549 = stablehlo.multiply %v1543, %v1529 : tensor<32x49x768xf32>
    %v1550 = stablehlo.subtract %v1549, %v1545 : tensor<32x49x768xf32>
    %v1551 = stablehlo.multiply %v1541, %v1548 : tensor<32x49x768xf32>
    %v1552 = stablehlo.subtract %v1550, %v1551 : tensor<32x49x768xf32>
    %v1553 = stablehlo.divide %v1540, %v1529 : tensor<32x49x768xf32>
    %v1554 = stablehlo.multiply %v1553, %v1552 : tensor<32x49x768xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1557 = stablehlo.transpose %v1556, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1558 = stablehlo.reshape %v1557 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1560 = stablehlo.reverse %s3b2dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1561 = stablehlo.convolution(%v1559, %v1560)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1562 = stablehlo.reshape %v1561 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1564 = stablehlo.reshape %dgapf : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1565 = stablehlo.add %v1563, %v1564 : tensor<32x768x7x7xf32>
    %v1566 = stablehlo.reshape %v1565 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1567 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1568 = stablehlo.reshape %v1443 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1569 = stablehlo.reshape %dgapf : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1570 = stablehlo.multiply %v1568, %v1569 : tensor<32x768x7x7xf32>
    %v1571 = stablehlo.reduce(%v1570 init: %v1567) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1572 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1573 = stablehlo.multiply %v1571, %v1572 : tensor<768xf32>
    %v1574 = stablehlo.subtract %s3b2lg, %v1573 : tensor<768xf32>
    %v1575 = stablehlo.reshape %v1438 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1576 = stablehlo.reshape %v1479 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1577 = stablehlo.transpose %v1575, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1578 = stablehlo.transpose %v1576, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1579 = stablehlo.convolution(%v1577, %v1578)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1580 = stablehlo.transpose %v1579, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1581 = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %v1582 = stablehlo.multiply %v1580, %v1581 : tensor<768x3072x1x1xf32>
    %v1583 = stablehlo.subtract %s3b2pW, %v1582 : tensor<768x3072x1x1xf32>
    %v1584 = stablehlo.reshape %v1479 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1586 = stablehlo.reduce(%v1584 init: %v1585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1587 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1588 = stablehlo.multiply %v1586, %v1587 : tensor<768xf32>
    %v1589 = stablehlo.subtract %s3b2pb, %v1588 : tensor<768xf32>
    %v1590 = stablehlo.reshape %v1418 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1591 = stablehlo.reshape %v1510 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1592 = stablehlo.transpose %v1590, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1593 = stablehlo.transpose %v1591, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1594 = stablehlo.convolution(%v1592, %v1593)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1595 = stablehlo.transpose %v1594, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1596 = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %v1597 = stablehlo.multiply %v1595, %v1596 : tensor<3072x768x1x1xf32>
    %v1598 = stablehlo.subtract %s3b2eW, %v1597 : tensor<3072x768x1x1xf32>
    %v1599 = stablehlo.reshape %v1510 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1600 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1601 = stablehlo.reduce(%v1599 init: %v1600) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1602 = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %v1603 = stablehlo.multiply %v1601, %v1602 : tensor<3072xf32>
    %v1604 = stablehlo.subtract %s3b2eb, %v1603 : tensor<3072xf32>
    %v1605 = stablehlo.reshape %v1384 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1606 = stablehlo.transpose %v1605, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1608 = stablehlo.reshape %v1515 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1609 = stablehlo.transpose %v1608, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1610 = stablehlo.reshape %v1609 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1611 = stablehlo.reshape %v1607 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1612 = stablehlo.reshape %v1610 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1614 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1615 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1616 = stablehlo.reduce(%v1611 init: %v1613) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1617 = stablehlo.broadcast_in_dim %v1616, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1618 = stablehlo.divide %v1617, %v1614 : tensor<32x49x768xf32>
    %v1619 = stablehlo.subtract %v1611, %v1618 : tensor<32x49x768xf32>
    %v1620 = stablehlo.multiply %v1619, %v1619 : tensor<32x49x768xf32>
    %v1621 = stablehlo.reduce(%v1620 init: %v1613) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1622 = stablehlo.broadcast_in_dim %v1621, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1623 = stablehlo.divide %v1622, %v1614 : tensor<32x49x768xf32>
    %v1624 = stablehlo.add %v1623, %v1615 : tensor<32x49x768xf32>
    %v1625 = stablehlo.rsqrt %v1624 : tensor<32x49x768xf32>
    %v1626 = stablehlo.multiply %v1619, %v1625 : tensor<32x49x768xf32>
    %v1627 = stablehlo.multiply %v1612, %v1626 : tensor<32x49x768xf32>
    %v1628 = stablehlo.reduce(%v1627 init: %v1613) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1629 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1630 = stablehlo.multiply %v1628, %v1629 : tensor<768xf32>
    %v1631 = stablehlo.subtract %s3b2ng, %v1630 : tensor<768xf32>
    %v1632 = stablehlo.reshape %v1515 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1633 = stablehlo.transpose %v1632, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1635 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1636 = stablehlo.reshape %v1634 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1637 = stablehlo.reduce(%v1636 init: %v1635) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1638 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1639 = stablehlo.multiply %v1637, %v1638 : tensor<768xf32>
    %v1640 = stablehlo.subtract %s3b2nbt, %v1639 : tensor<768xf32>
    %v1641 = stablehlo.reshape %v1379 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1642 = stablehlo.reshape %v1558 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1643 = stablehlo.transpose %v1641, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1644 = stablehlo.transpose %v1642, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1645 = stablehlo.convolution(%v1643, %v1644)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1646 = stablehlo.reshape %v1645 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1647 = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %v1648 = stablehlo.multiply %v1646, %v1647 : tensor<768x1x7x7xf32>
    %v1649 = stablehlo.subtract %s3b2dW, %v1648 : tensor<768x1x7x7xf32>
    %v1650 = stablehlo.reshape %v1558 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1651 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1652 = stablehlo.reduce(%v1650 init: %v1651) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1653 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1654 = stablehlo.multiply %v1652, %v1653 : tensor<768xf32>
    %v1655 = stablehlo.subtract %s3b2db, %v1654 : tensor<768xf32>
    %v1656 = stablehlo.reshape %v1566 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1657 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1658 = stablehlo.multiply %v1656, %v1657 : tensor<32x768x7x7xf32>
    %v1659 = stablehlo.reshape %v1658 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1660 = stablehlo.reshape %v1659 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1661 = stablehlo.transpose %s3b1pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1662 = stablehlo.reverse %v1661, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1663 = stablehlo.convolution(%v1660, %v1662)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1664 = stablehlo.reshape %v1663 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1666 = stablehlo.reshape %v1351 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1667 = stablehlo.multiply %v1666, %v1666 : tensor<32x3072x7x7xf32>
    %v1668 = stablehlo.multiply %v1667, %v1666 : tensor<32x3072x7x7xf32>
    %v1669 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1670 = stablehlo.multiply %v1669, %v1668 : tensor<32x3072x7x7xf32>
    %v1671 = stablehlo.add %v1666, %v1670 : tensor<32x3072x7x7xf32>
    %v1672 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1673 = stablehlo.multiply %v1672, %v1671 : tensor<32x3072x7x7xf32>
    %v1674 = stablehlo.tanh %v1673 : tensor<32x3072x7x7xf32>
    %v1675 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1676 = stablehlo.add %v1675, %v1674 : tensor<32x3072x7x7xf32>
    %v1677 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1678 = stablehlo.multiply %v1677, %v1676 : tensor<32x3072x7x7xf32>
    %v1679 = stablehlo.multiply %v1674, %v1674 : tensor<32x3072x7x7xf32>
    %v1680 = stablehlo.subtract %v1675, %v1679 : tensor<32x3072x7x7xf32>
    %v1681 = stablehlo.multiply %v1677, %v1666 : tensor<32x3072x7x7xf32>
    %v1682 = stablehlo.multiply %v1681, %v1680 : tensor<32x3072x7x7xf32>
    %v1683 = stablehlo.constant dense<0.134145> : tensor<32x3072x7x7xf32>
    %v1684 = stablehlo.multiply %v1683, %v1667 : tensor<32x3072x7x7xf32>
    %v1685 = stablehlo.add %v1675, %v1684 : tensor<32x3072x7x7xf32>
    %v1686 = stablehlo.multiply %v1672, %v1685 : tensor<32x3072x7x7xf32>
    %v1687 = stablehlo.multiply %v1682, %v1686 : tensor<32x3072x7x7xf32>
    %v1688 = stablehlo.add %v1678, %v1687 : tensor<32x3072x7x7xf32>
    %v1689 = stablehlo.multiply %v1665, %v1688 : tensor<32x3072x7x7xf32>
    %v1690 = stablehlo.reshape %v1689 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1691 = stablehlo.reshape %v1690 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1692 = stablehlo.transpose %s3b1eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1693 = stablehlo.reverse %v1692, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1694 = stablehlo.convolution(%v1691, %v1693)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1695 = stablehlo.reshape %v1694 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1696 = stablehlo.reshape %v1312 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1697 = stablehlo.transpose %v1696, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1698 = stablehlo.reshape %v1697 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1699 = stablehlo.reshape %v1695 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1700 = stablehlo.transpose %v1699, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1702 = stablehlo.reshape %v1701 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1703 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1704 = stablehlo.multiply %v1702, %v1703 : tensor<32x49x768xf32>
    %v1705 = stablehlo.reshape %v1704 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1706 = stablehlo.reshape %v1705 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1707 = stablehlo.reshape %v1698 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1709 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1710 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1711 = stablehlo.reduce(%v1707 init: %v1708) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1712 = stablehlo.broadcast_in_dim %v1711, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1713 = stablehlo.divide %v1712, %v1709 : tensor<32x49x768xf32>
    %v1714 = stablehlo.subtract %v1707, %v1713 : tensor<32x49x768xf32>
    %v1715 = stablehlo.multiply %v1714, %v1714 : tensor<32x49x768xf32>
    %v1716 = stablehlo.reduce(%v1715 init: %v1708) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1717 = stablehlo.broadcast_in_dim %v1716, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1718 = stablehlo.divide %v1717, %v1709 : tensor<32x49x768xf32>
    %v1719 = stablehlo.add %v1718, %v1710 : tensor<32x49x768xf32>
    %v1720 = stablehlo.rsqrt %v1719 : tensor<32x49x768xf32>
    %v1721 = stablehlo.multiply %v1714, %v1720 : tensor<32x49x768xf32>
    %v1722 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1723 = stablehlo.multiply %v1722, %v1706 : tensor<32x49x768xf32>
    %v1724 = stablehlo.reduce(%v1723 init: %v1708) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1725 = stablehlo.broadcast_in_dim %v1724, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1726 = stablehlo.multiply %v1721, %v1723 : tensor<32x49x768xf32>
    %v1727 = stablehlo.reduce(%v1726 init: %v1708) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1728 = stablehlo.broadcast_in_dim %v1727, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1729 = stablehlo.multiply %v1723, %v1709 : tensor<32x49x768xf32>
    %v1730 = stablehlo.subtract %v1729, %v1725 : tensor<32x49x768xf32>
    %v1731 = stablehlo.multiply %v1721, %v1728 : tensor<32x49x768xf32>
    %v1732 = stablehlo.subtract %v1730, %v1731 : tensor<32x49x768xf32>
    %v1733 = stablehlo.divide %v1720, %v1709 : tensor<32x49x768xf32>
    %v1734 = stablehlo.multiply %v1733, %v1732 : tensor<32x49x768xf32>
    %v1735 = stablehlo.reshape %v1734 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1736 = stablehlo.reshape %v1735 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1737 = stablehlo.transpose %v1736, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1738 = stablehlo.reshape %v1737 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1739 = stablehlo.reshape %v1738 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1740 = stablehlo.reverse %s3b1dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1741 = stablehlo.convolution(%v1739, %v1740)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1742 = stablehlo.reshape %v1741 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1743 = stablehlo.reshape %v1742 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1744 = stablehlo.reshape %v1566 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1745 = stablehlo.add %v1743, %v1744 : tensor<32x768x7x7xf32>
    %v1746 = stablehlo.reshape %v1745 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1748 = stablehlo.reshape %v1371 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1749 = stablehlo.reshape %v1566 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1750 = stablehlo.multiply %v1748, %v1749 : tensor<32x768x7x7xf32>
    %v1751 = stablehlo.reduce(%v1750 init: %v1747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1752 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1753 = stablehlo.multiply %v1751, %v1752 : tensor<768xf32>
    %v1754 = stablehlo.subtract %s3b1lg, %v1753 : tensor<768xf32>
    %v1755 = stablehlo.reshape %v1366 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1756 = stablehlo.reshape %v1659 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1757 = stablehlo.transpose %v1755, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1758 = stablehlo.transpose %v1756, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1759 = stablehlo.convolution(%v1757, %v1758)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1760 = stablehlo.transpose %v1759, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1761 = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %v1762 = stablehlo.multiply %v1760, %v1761 : tensor<768x3072x1x1xf32>
    %v1763 = stablehlo.subtract %s3b1pW, %v1762 : tensor<768x3072x1x1xf32>
    %v1764 = stablehlo.reshape %v1659 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1765 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1766 = stablehlo.reduce(%v1764 init: %v1765) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1767 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1768 = stablehlo.multiply %v1766, %v1767 : tensor<768xf32>
    %v1769 = stablehlo.subtract %s3b1pb, %v1768 : tensor<768xf32>
    %v1770 = stablehlo.reshape %v1346 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1771 = stablehlo.reshape %v1690 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1772 = stablehlo.transpose %v1770, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1773 = stablehlo.transpose %v1771, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1774 = stablehlo.convolution(%v1772, %v1773)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1775 = stablehlo.transpose %v1774, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1776 = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %v1777 = stablehlo.multiply %v1775, %v1776 : tensor<3072x768x1x1xf32>
    %v1778 = stablehlo.subtract %s3b1eW, %v1777 : tensor<3072x768x1x1xf32>
    %v1779 = stablehlo.reshape %v1690 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1781 = stablehlo.reduce(%v1779 init: %v1780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1782 = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %v1783 = stablehlo.multiply %v1781, %v1782 : tensor<3072xf32>
    %v1784 = stablehlo.subtract %s3b1eb, %v1783 : tensor<3072xf32>
    %v1785 = stablehlo.reshape %v1312 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1786 = stablehlo.transpose %v1785, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1787 = stablehlo.reshape %v1786 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1788 = stablehlo.reshape %v1695 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1789 = stablehlo.transpose %v1788, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1790 = stablehlo.reshape %v1789 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1791 = stablehlo.reshape %v1787 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1792 = stablehlo.reshape %v1790 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1794 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1795 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1796 = stablehlo.reduce(%v1791 init: %v1793) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1797 = stablehlo.broadcast_in_dim %v1796, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1798 = stablehlo.divide %v1797, %v1794 : tensor<32x49x768xf32>
    %v1799 = stablehlo.subtract %v1791, %v1798 : tensor<32x49x768xf32>
    %v1800 = stablehlo.multiply %v1799, %v1799 : tensor<32x49x768xf32>
    %v1801 = stablehlo.reduce(%v1800 init: %v1793) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1802 = stablehlo.broadcast_in_dim %v1801, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1803 = stablehlo.divide %v1802, %v1794 : tensor<32x49x768xf32>
    %v1804 = stablehlo.add %v1803, %v1795 : tensor<32x49x768xf32>
    %v1805 = stablehlo.rsqrt %v1804 : tensor<32x49x768xf32>
    %v1806 = stablehlo.multiply %v1799, %v1805 : tensor<32x49x768xf32>
    %v1807 = stablehlo.multiply %v1792, %v1806 : tensor<32x49x768xf32>
    %v1808 = stablehlo.reduce(%v1807 init: %v1793) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1809 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1810 = stablehlo.multiply %v1808, %v1809 : tensor<768xf32>
    %v1811 = stablehlo.subtract %s3b1ng, %v1810 : tensor<768xf32>
    %v1812 = stablehlo.reshape %v1695 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1813 = stablehlo.transpose %v1812, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1814 = stablehlo.reshape %v1813 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1816 = stablehlo.reshape %v1814 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1817 = stablehlo.reduce(%v1816 init: %v1815) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1818 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1819 = stablehlo.multiply %v1817, %v1818 : tensor<768xf32>
    %v1820 = stablehlo.subtract %s3b1nbt, %v1819 : tensor<768xf32>
    %v1821 = stablehlo.reshape %v1307 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1822 = stablehlo.reshape %v1738 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1823 = stablehlo.transpose %v1821, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1824 = stablehlo.transpose %v1822, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1825 = stablehlo.convolution(%v1823, %v1824)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1826 = stablehlo.reshape %v1825 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1827 = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %v1828 = stablehlo.multiply %v1826, %v1827 : tensor<768x1x7x7xf32>
    %v1829 = stablehlo.subtract %s3b1dW, %v1828 : tensor<768x1x7x7xf32>
    %v1830 = stablehlo.reshape %v1738 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1832 = stablehlo.reduce(%v1830 init: %v1831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1833 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1834 = stablehlo.multiply %v1832, %v1833 : tensor<768xf32>
    %v1835 = stablehlo.subtract %s3b1db, %v1834 : tensor<768xf32>
    %v1836 = stablehlo.reshape %v1746 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1837 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1838 = stablehlo.multiply %v1836, %v1837 : tensor<32x768x7x7xf32>
    %v1839 = stablehlo.reshape %v1838 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1840 = stablehlo.reshape %v1839 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1841 = stablehlo.transpose %s3b0pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1842 = stablehlo.reverse %v1841, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1843 = stablehlo.convolution(%v1840, %v1842)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1844 = stablehlo.reshape %v1843 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1845 = stablehlo.reshape %v1844 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1846 = stablehlo.reshape %v1279 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1847 = stablehlo.multiply %v1846, %v1846 : tensor<32x3072x7x7xf32>
    %v1848 = stablehlo.multiply %v1847, %v1846 : tensor<32x3072x7x7xf32>
    %v1849 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1850 = stablehlo.multiply %v1849, %v1848 : tensor<32x3072x7x7xf32>
    %v1851 = stablehlo.add %v1846, %v1850 : tensor<32x3072x7x7xf32>
    %v1852 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1853 = stablehlo.multiply %v1852, %v1851 : tensor<32x3072x7x7xf32>
    %v1854 = stablehlo.tanh %v1853 : tensor<32x3072x7x7xf32>
    %v1855 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1856 = stablehlo.add %v1855, %v1854 : tensor<32x3072x7x7xf32>
    %v1857 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1858 = stablehlo.multiply %v1857, %v1856 : tensor<32x3072x7x7xf32>
    %v1859 = stablehlo.multiply %v1854, %v1854 : tensor<32x3072x7x7xf32>
    %v1860 = stablehlo.subtract %v1855, %v1859 : tensor<32x3072x7x7xf32>
    %v1861 = stablehlo.multiply %v1857, %v1846 : tensor<32x3072x7x7xf32>
    %v1862 = stablehlo.multiply %v1861, %v1860 : tensor<32x3072x7x7xf32>
    %v1863 = stablehlo.constant dense<0.134145> : tensor<32x3072x7x7xf32>
    %v1864 = stablehlo.multiply %v1863, %v1847 : tensor<32x3072x7x7xf32>
    %v1865 = stablehlo.add %v1855, %v1864 : tensor<32x3072x7x7xf32>
    %v1866 = stablehlo.multiply %v1852, %v1865 : tensor<32x3072x7x7xf32>
    %v1867 = stablehlo.multiply %v1862, %v1866 : tensor<32x3072x7x7xf32>
    %v1868 = stablehlo.add %v1858, %v1867 : tensor<32x3072x7x7xf32>
    %v1869 = stablehlo.multiply %v1845, %v1868 : tensor<32x3072x7x7xf32>
    %v1870 = stablehlo.reshape %v1869 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1871 = stablehlo.reshape %v1870 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1872 = stablehlo.transpose %s3b0eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1873 = stablehlo.reverse %v1872, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1874 = stablehlo.convolution(%v1871, %v1873)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1875 = stablehlo.reshape %v1874 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1876 = stablehlo.reshape %v1240 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1877 = stablehlo.transpose %v1876, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1878 = stablehlo.reshape %v1877 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1879 = stablehlo.reshape %v1875 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1880 = stablehlo.transpose %v1879, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1881 = stablehlo.reshape %v1880 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1882 = stablehlo.reshape %v1881 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1883 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1884 = stablehlo.multiply %v1882, %v1883 : tensor<32x49x768xf32>
    %v1885 = stablehlo.reshape %v1884 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1886 = stablehlo.reshape %v1885 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1887 = stablehlo.reshape %v1878 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1889 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1890 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1891 = stablehlo.reduce(%v1887 init: %v1888) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1892 = stablehlo.broadcast_in_dim %v1891, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1893 = stablehlo.divide %v1892, %v1889 : tensor<32x49x768xf32>
    %v1894 = stablehlo.subtract %v1887, %v1893 : tensor<32x49x768xf32>
    %v1895 = stablehlo.multiply %v1894, %v1894 : tensor<32x49x768xf32>
    %v1896 = stablehlo.reduce(%v1895 init: %v1888) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1897 = stablehlo.broadcast_in_dim %v1896, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1898 = stablehlo.divide %v1897, %v1889 : tensor<32x49x768xf32>
    %v1899 = stablehlo.add %v1898, %v1890 : tensor<32x49x768xf32>
    %v1900 = stablehlo.rsqrt %v1899 : tensor<32x49x768xf32>
    %v1901 = stablehlo.multiply %v1894, %v1900 : tensor<32x49x768xf32>
    %v1902 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1903 = stablehlo.multiply %v1902, %v1886 : tensor<32x49x768xf32>
    %v1904 = stablehlo.reduce(%v1903 init: %v1888) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1905 = stablehlo.broadcast_in_dim %v1904, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1906 = stablehlo.multiply %v1901, %v1903 : tensor<32x49x768xf32>
    %v1907 = stablehlo.reduce(%v1906 init: %v1888) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1908 = stablehlo.broadcast_in_dim %v1907, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1909 = stablehlo.multiply %v1903, %v1889 : tensor<32x49x768xf32>
    %v1910 = stablehlo.subtract %v1909, %v1905 : tensor<32x49x768xf32>
    %v1911 = stablehlo.multiply %v1901, %v1908 : tensor<32x49x768xf32>
    %v1912 = stablehlo.subtract %v1910, %v1911 : tensor<32x49x768xf32>
    %v1913 = stablehlo.divide %v1900, %v1889 : tensor<32x49x768xf32>
    %v1914 = stablehlo.multiply %v1913, %v1912 : tensor<32x49x768xf32>
    %v1915 = stablehlo.reshape %v1914 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1917 = stablehlo.transpose %v1916, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1918 = stablehlo.reshape %v1917 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1919 = stablehlo.reshape %v1918 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1920 = stablehlo.reverse %s3b0dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1921 = stablehlo.convolution(%v1919, %v1920)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1922 = stablehlo.reshape %v1921 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1923 = stablehlo.reshape %v1922 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1924 = stablehlo.reshape %v1746 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1925 = stablehlo.add %v1923, %v1924 : tensor<32x768x7x7xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1928 = stablehlo.reshape %v1299 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1929 = stablehlo.reshape %v1746 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1930 = stablehlo.multiply %v1928, %v1929 : tensor<32x768x7x7xf32>
    %v1931 = stablehlo.reduce(%v1930 init: %v1927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1932 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1933 = stablehlo.multiply %v1931, %v1932 : tensor<768xf32>
    %v1934 = stablehlo.subtract %s3b0lg, %v1933 : tensor<768xf32>
    %v1935 = stablehlo.reshape %v1294 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1936 = stablehlo.reshape %v1839 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1937 = stablehlo.transpose %v1935, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1938 = stablehlo.transpose %v1936, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1939 = stablehlo.convolution(%v1937, %v1938)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1940 = stablehlo.transpose %v1939, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1941 = stablehlo.constant dense<0.1> : tensor<768x3072x1x1xf32>
    %v1942 = stablehlo.multiply %v1940, %v1941 : tensor<768x3072x1x1xf32>
    %v1943 = stablehlo.subtract %s3b0pW, %v1942 : tensor<768x3072x1x1xf32>
    %v1944 = stablehlo.reshape %v1839 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1946 = stablehlo.reduce(%v1944 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1947 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1948 = stablehlo.multiply %v1946, %v1947 : tensor<768xf32>
    %v1949 = stablehlo.subtract %s3b0pb, %v1948 : tensor<768xf32>
    %v1950 = stablehlo.reshape %v1274 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1951 = stablehlo.reshape %v1870 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1952 = stablehlo.transpose %v1950, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1953 = stablehlo.transpose %v1951, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1954 = stablehlo.convolution(%v1952, %v1953)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1955 = stablehlo.transpose %v1954, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1956 = stablehlo.constant dense<0.1> : tensor<3072x768x1x1xf32>
    %v1957 = stablehlo.multiply %v1955, %v1956 : tensor<3072x768x1x1xf32>
    %v1958 = stablehlo.subtract %s3b0eW, %v1957 : tensor<3072x768x1x1xf32>
    %v1959 = stablehlo.reshape %v1870 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1960 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1961 = stablehlo.reduce(%v1959 init: %v1960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1962 = stablehlo.constant dense<0.1> : tensor<3072xf32>
    %v1963 = stablehlo.multiply %v1961, %v1962 : tensor<3072xf32>
    %v1964 = stablehlo.subtract %s3b0eb, %v1963 : tensor<3072xf32>
    %v1965 = stablehlo.reshape %v1240 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1966 = stablehlo.transpose %v1965, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1967 = stablehlo.reshape %v1966 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1968 = stablehlo.reshape %v1875 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1969 = stablehlo.transpose %v1968, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1970 = stablehlo.reshape %v1969 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1971 = stablehlo.reshape %v1967 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1972 = stablehlo.reshape %v1970 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1973 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1974 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1975 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1976 = stablehlo.reduce(%v1971 init: %v1973) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1977 = stablehlo.broadcast_in_dim %v1976, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1978 = stablehlo.divide %v1977, %v1974 : tensor<32x49x768xf32>
    %v1979 = stablehlo.subtract %v1971, %v1978 : tensor<32x49x768xf32>
    %v1980 = stablehlo.multiply %v1979, %v1979 : tensor<32x49x768xf32>
    %v1981 = stablehlo.reduce(%v1980 init: %v1973) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1982 = stablehlo.broadcast_in_dim %v1981, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1983 = stablehlo.divide %v1982, %v1974 : tensor<32x49x768xf32>
    %v1984 = stablehlo.add %v1983, %v1975 : tensor<32x49x768xf32>
    %v1985 = stablehlo.rsqrt %v1984 : tensor<32x49x768xf32>
    %v1986 = stablehlo.multiply %v1979, %v1985 : tensor<32x49x768xf32>
    %v1987 = stablehlo.multiply %v1972, %v1986 : tensor<32x49x768xf32>
    %v1988 = stablehlo.reduce(%v1987 init: %v1973) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1989 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1990 = stablehlo.multiply %v1988, %v1989 : tensor<768xf32>
    %v1991 = stablehlo.subtract %s3b0ng, %v1990 : tensor<768xf32>
    %v1992 = stablehlo.reshape %v1875 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1993 = stablehlo.transpose %v1992, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1994 = stablehlo.reshape %v1993 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1995 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1996 = stablehlo.reshape %v1994 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1997 = stablehlo.reduce(%v1996 init: %v1995) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1998 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v1999 = stablehlo.multiply %v1997, %v1998 : tensor<768xf32>
    %v2000 = stablehlo.subtract %s3b0nbt, %v1999 : tensor<768xf32>
    %v2001 = stablehlo.reshape %v1235 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2002 = stablehlo.reshape %v1918 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2003 = stablehlo.transpose %v2001, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v2004 = stablehlo.transpose %v2002, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v2005 = stablehlo.convolution(%v2003, %v2004)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v2006 = stablehlo.reshape %v2005 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v2007 = stablehlo.constant dense<0.1> : tensor<768x1x7x7xf32>
    %v2008 = stablehlo.multiply %v2006, %v2007 : tensor<768x1x7x7xf32>
    %v2009 = stablehlo.subtract %s3b0dW, %v2008 : tensor<768x1x7x7xf32>
    %v2010 = stablehlo.reshape %v1918 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2012 = stablehlo.reduce(%v2010 init: %v2011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v2013 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v2014 = stablehlo.multiply %v2012, %v2013 : tensor<768xf32>
    %v2015 = stablehlo.subtract %s3b0db, %v2014 : tensor<768xf32>
    %v2016 = stablehlo.reshape %v1926 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2018 = stablehlo.pad %v2016, %v2017, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x14x14xf32>
    %v2019 = stablehlo.transpose %d2W, dims = [1, 0, 2, 3] : (tensor<768x384x2x2xf32>) -> tensor<384x768x2x2xf32>
    %v2020 = stablehlo.reverse %v2019, dims = [2, 3] : tensor<384x768x2x2xf32>
    %v2021 = stablehlo.convolution(%v2018, %v2020)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x14x14xf32>, tensor<384x768x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v2022 = stablehlo.reshape %v2021 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2023 = stablehlo.reshape %v1196 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2024 = stablehlo.transpose %v2023, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2025 = stablehlo.reshape %v2024 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2026 = stablehlo.reshape %v2022 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2027 = stablehlo.transpose %v2026, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2028 = stablehlo.reshape %v2027 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2029 = stablehlo.reshape %v2028 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2030 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2031 = stablehlo.multiply %v2029, %v2030 : tensor<32x196x384xf32>
    %v2032 = stablehlo.reshape %v2031 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2033 = stablehlo.reshape %v2032 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2034 = stablehlo.reshape %v2025 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2035 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2036 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2037 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2038 = stablehlo.reduce(%v2034 init: %v2035) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2039 = stablehlo.broadcast_in_dim %v2038, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2040 = stablehlo.divide %v2039, %v2036 : tensor<32x196x384xf32>
    %v2041 = stablehlo.subtract %v2034, %v2040 : tensor<32x196x384xf32>
    %v2042 = stablehlo.multiply %v2041, %v2041 : tensor<32x196x384xf32>
    %v2043 = stablehlo.reduce(%v2042 init: %v2035) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2044 = stablehlo.broadcast_in_dim %v2043, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2045 = stablehlo.divide %v2044, %v2036 : tensor<32x196x384xf32>
    %v2046 = stablehlo.add %v2045, %v2037 : tensor<32x196x384xf32>
    %v2047 = stablehlo.rsqrt %v2046 : tensor<32x196x384xf32>
    %v2048 = stablehlo.multiply %v2041, %v2047 : tensor<32x196x384xf32>
    %v2049 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2050 = stablehlo.multiply %v2049, %v2033 : tensor<32x196x384xf32>
    %v2051 = stablehlo.reduce(%v2050 init: %v2035) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2052 = stablehlo.broadcast_in_dim %v2051, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2053 = stablehlo.multiply %v2048, %v2050 : tensor<32x196x384xf32>
    %v2054 = stablehlo.reduce(%v2053 init: %v2035) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2055 = stablehlo.broadcast_in_dim %v2054, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2056 = stablehlo.multiply %v2050, %v2036 : tensor<32x196x384xf32>
    %v2057 = stablehlo.subtract %v2056, %v2052 : tensor<32x196x384xf32>
    %v2058 = stablehlo.multiply %v2048, %v2055 : tensor<32x196x384xf32>
    %v2059 = stablehlo.subtract %v2057, %v2058 : tensor<32x196x384xf32>
    %v2060 = stablehlo.divide %v2047, %v2036 : tensor<32x196x384xf32>
    %v2061 = stablehlo.multiply %v2060, %v2059 : tensor<32x196x384xf32>
    %v2062 = stablehlo.reshape %v2061 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2063 = stablehlo.reshape %v2062 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2064 = stablehlo.transpose %v2063, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2065 = stablehlo.reshape %v2064 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2066 = stablehlo.reshape %v1926 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2068 = stablehlo.reduce(%v2066 init: %v2067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v2069 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v2070 = stablehlo.multiply %v2068, %v2069 : tensor<768xf32>
    %v2071 = stablehlo.subtract %d2b, %v2070 : tensor<768xf32>
    %v2072 = stablehlo.reshape %v1196 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2073 = stablehlo.transpose %v2072, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2074 = stablehlo.reshape %v2073 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2075 = stablehlo.reshape %v2022 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2076 = stablehlo.transpose %v2075, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2077 = stablehlo.reshape %v2076 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2078 = stablehlo.reshape %v2074 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2079 = stablehlo.reshape %v2077 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2080 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2081 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2082 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2083 = stablehlo.reduce(%v2078 init: %v2080) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2084 = stablehlo.broadcast_in_dim %v2083, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2085 = stablehlo.divide %v2084, %v2081 : tensor<32x196x384xf32>
    %v2086 = stablehlo.subtract %v2078, %v2085 : tensor<32x196x384xf32>
    %v2087 = stablehlo.multiply %v2086, %v2086 : tensor<32x196x384xf32>
    %v2088 = stablehlo.reduce(%v2087 init: %v2080) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2089 = stablehlo.broadcast_in_dim %v2088, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2090 = stablehlo.divide %v2089, %v2081 : tensor<32x196x384xf32>
    %v2091 = stablehlo.add %v2090, %v2082 : tensor<32x196x384xf32>
    %v2092 = stablehlo.rsqrt %v2091 : tensor<32x196x384xf32>
    %v2093 = stablehlo.multiply %v2086, %v2092 : tensor<32x196x384xf32>
    %v2094 = stablehlo.multiply %v2079, %v2093 : tensor<32x196x384xf32>
    %v2095 = stablehlo.reduce(%v2094 init: %v2080) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2096 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2097 = stablehlo.multiply %v2095, %v2096 : tensor<384xf32>
    %v2098 = stablehlo.subtract %d2ng, %v2097 : tensor<384xf32>
    %v2099 = stablehlo.reshape %v2022 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2100 = stablehlo.transpose %v2099, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2101 = stablehlo.reshape %v2100 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2103 = stablehlo.reshape %v2101 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2104 = stablehlo.reduce(%v2103 init: %v2102) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2105 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2106 = stablehlo.multiply %v2104, %v2105 : tensor<384xf32>
    %v2107 = stablehlo.subtract %d2nbt, %v2106 : tensor<384xf32>
    %v2108 = stablehlo.reshape %v1230 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2109 = stablehlo.reshape %v1926 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v2110 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2111 = stablehlo.pad %v2109, %v2110, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x13x13xf32>
    %v2112 = stablehlo.transpose %v2108, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2113 = stablehlo.transpose %v2111, dims = [1, 0, 2, 3] : (tensor<32x768x13x13xf32>) -> tensor<768x32x13x13xf32>
    %v2114 = stablehlo.convolution(%v2112, %v2113)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<768x32x13x13xf32>) -> tensor<384x768x2x2xf32>
    %v2115 = stablehlo.transpose %v2114, dims = [1, 0, 2, 3] : (tensor<384x768x2x2xf32>) -> tensor<768x384x2x2xf32>
    %v2116 = stablehlo.constant dense<0.1> : tensor<768x384x2x2xf32>
    %v2117 = stablehlo.multiply %v2115, %v2116 : tensor<768x384x2x2xf32>
    %v2118 = stablehlo.subtract %d2W, %v2117 : tensor<768x384x2x2xf32>
    %v2119 = stablehlo.reshape %v2065 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2120 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2121 = stablehlo.multiply %v2119, %v2120 : tensor<32x384x14x14xf32>
    %v2122 = stablehlo.reshape %v2121 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2123 = stablehlo.reshape %v2122 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2124 = stablehlo.transpose %s2b8pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2125 = stablehlo.reverse %v2124, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2126 = stablehlo.convolution(%v2123, %v2125)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2127 = stablehlo.reshape %v2126 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2128 = stablehlo.reshape %v2127 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2129 = stablehlo.reshape %v1168 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2130 = stablehlo.multiply %v2129, %v2129 : tensor<32x1536x14x14xf32>
    %v2131 = stablehlo.multiply %v2130, %v2129 : tensor<32x1536x14x14xf32>
    %v2132 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v2133 = stablehlo.multiply %v2132, %v2131 : tensor<32x1536x14x14xf32>
    %v2134 = stablehlo.add %v2129, %v2133 : tensor<32x1536x14x14xf32>
    %v2135 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v2136 = stablehlo.multiply %v2135, %v2134 : tensor<32x1536x14x14xf32>
    %v2137 = stablehlo.tanh %v2136 : tensor<32x1536x14x14xf32>
    %v2138 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v2139 = stablehlo.add %v2138, %v2137 : tensor<32x1536x14x14xf32>
    %v2140 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2141 = stablehlo.multiply %v2140, %v2139 : tensor<32x1536x14x14xf32>
    %v2142 = stablehlo.multiply %v2137, %v2137 : tensor<32x1536x14x14xf32>
    %v2143 = stablehlo.subtract %v2138, %v2142 : tensor<32x1536x14x14xf32>
    %v2144 = stablehlo.multiply %v2140, %v2129 : tensor<32x1536x14x14xf32>
    %v2145 = stablehlo.multiply %v2144, %v2143 : tensor<32x1536x14x14xf32>
    %v2146 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v2147 = stablehlo.multiply %v2146, %v2130 : tensor<32x1536x14x14xf32>
    %v2148 = stablehlo.add %v2138, %v2147 : tensor<32x1536x14x14xf32>
    %v2149 = stablehlo.multiply %v2135, %v2148 : tensor<32x1536x14x14xf32>
    %v2150 = stablehlo.multiply %v2145, %v2149 : tensor<32x1536x14x14xf32>
    %v2151 = stablehlo.add %v2141, %v2150 : tensor<32x1536x14x14xf32>
    %v2152 = stablehlo.multiply %v2128, %v2151 : tensor<32x1536x14x14xf32>
    %v2153 = stablehlo.reshape %v2152 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2155 = stablehlo.transpose %s2b8eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2156 = stablehlo.reverse %v2155, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2157 = stablehlo.convolution(%v2154, %v2156)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2158 = stablehlo.reshape %v2157 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2159 = stablehlo.reshape %v1129 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2160 = stablehlo.transpose %v2159, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2161 = stablehlo.reshape %v2160 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2162 = stablehlo.reshape %v2158 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2163 = stablehlo.transpose %v2162, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2164 = stablehlo.reshape %v2163 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2166 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2167 = stablehlo.multiply %v2165, %v2166 : tensor<32x196x384xf32>
    %v2168 = stablehlo.reshape %v2167 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2169 = stablehlo.reshape %v2168 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2170 = stablehlo.reshape %v2161 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2172 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2173 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2174 = stablehlo.reduce(%v2170 init: %v2171) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2175 = stablehlo.broadcast_in_dim %v2174, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2176 = stablehlo.divide %v2175, %v2172 : tensor<32x196x384xf32>
    %v2177 = stablehlo.subtract %v2170, %v2176 : tensor<32x196x384xf32>
    %v2178 = stablehlo.multiply %v2177, %v2177 : tensor<32x196x384xf32>
    %v2179 = stablehlo.reduce(%v2178 init: %v2171) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2180 = stablehlo.broadcast_in_dim %v2179, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2181 = stablehlo.divide %v2180, %v2172 : tensor<32x196x384xf32>
    %v2182 = stablehlo.add %v2181, %v2173 : tensor<32x196x384xf32>
    %v2183 = stablehlo.rsqrt %v2182 : tensor<32x196x384xf32>
    %v2184 = stablehlo.multiply %v2177, %v2183 : tensor<32x196x384xf32>
    %v2185 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2186 = stablehlo.multiply %v2185, %v2169 : tensor<32x196x384xf32>
    %v2187 = stablehlo.reduce(%v2186 init: %v2171) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2188 = stablehlo.broadcast_in_dim %v2187, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2189 = stablehlo.multiply %v2184, %v2186 : tensor<32x196x384xf32>
    %v2190 = stablehlo.reduce(%v2189 init: %v2171) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2191 = stablehlo.broadcast_in_dim %v2190, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2192 = stablehlo.multiply %v2186, %v2172 : tensor<32x196x384xf32>
    %v2193 = stablehlo.subtract %v2192, %v2188 : tensor<32x196x384xf32>
    %v2194 = stablehlo.multiply %v2184, %v2191 : tensor<32x196x384xf32>
    %v2195 = stablehlo.subtract %v2193, %v2194 : tensor<32x196x384xf32>
    %v2196 = stablehlo.divide %v2183, %v2172 : tensor<32x196x384xf32>
    %v2197 = stablehlo.multiply %v2196, %v2195 : tensor<32x196x384xf32>
    %v2198 = stablehlo.reshape %v2197 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2199 = stablehlo.reshape %v2198 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2200 = stablehlo.transpose %v2199, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2201 = stablehlo.reshape %v2200 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2202 = stablehlo.reshape %v2201 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2203 = stablehlo.reverse %s2b8dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2204 = stablehlo.convolution(%v2202, %v2203)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2205 = stablehlo.reshape %v2204 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2206 = stablehlo.reshape %v2205 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2207 = stablehlo.reshape %v2065 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2208 = stablehlo.add %v2206, %v2207 : tensor<32x384x14x14xf32>
    %v2209 = stablehlo.reshape %v2208 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2211 = stablehlo.reshape %v1188 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2212 = stablehlo.reshape %v2065 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2213 = stablehlo.multiply %v2211, %v2212 : tensor<32x384x14x14xf32>
    %v2214 = stablehlo.reduce(%v2213 init: %v2210) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2215 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2216 = stablehlo.multiply %v2214, %v2215 : tensor<384xf32>
    %v2217 = stablehlo.subtract %s2b8lg, %v2216 : tensor<384xf32>
    %v2218 = stablehlo.reshape %v1183 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2219 = stablehlo.reshape %v2122 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2220 = stablehlo.transpose %v2218, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2221 = stablehlo.transpose %v2219, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2222 = stablehlo.convolution(%v2220, %v2221)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2223 = stablehlo.transpose %v2222, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2224 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2225 = stablehlo.multiply %v2223, %v2224 : tensor<384x1536x1x1xf32>
    %v2226 = stablehlo.subtract %s2b8pW, %v2225 : tensor<384x1536x1x1xf32>
    %v2227 = stablehlo.reshape %v2122 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2229 = stablehlo.reduce(%v2227 init: %v2228) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2230 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2231 = stablehlo.multiply %v2229, %v2230 : tensor<384xf32>
    %v2232 = stablehlo.subtract %s2b8pb, %v2231 : tensor<384xf32>
    %v2233 = stablehlo.reshape %v1163 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2234 = stablehlo.reshape %v2153 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2235 = stablehlo.transpose %v2233, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2236 = stablehlo.transpose %v2234, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2237 = stablehlo.convolution(%v2235, %v2236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2238 = stablehlo.transpose %v2237, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2239 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2240 = stablehlo.multiply %v2238, %v2239 : tensor<1536x384x1x1xf32>
    %v2241 = stablehlo.subtract %s2b8eW, %v2240 : tensor<1536x384x1x1xf32>
    %v2242 = stablehlo.reshape %v2153 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2243 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2244 = stablehlo.reduce(%v2242 init: %v2243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2245 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2246 = stablehlo.multiply %v2244, %v2245 : tensor<1536xf32>
    %v2247 = stablehlo.subtract %s2b8eb, %v2246 : tensor<1536xf32>
    %v2248 = stablehlo.reshape %v1129 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2249 = stablehlo.transpose %v2248, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2250 = stablehlo.reshape %v2249 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2251 = stablehlo.reshape %v2158 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2252 = stablehlo.transpose %v2251, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2253 = stablehlo.reshape %v2252 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2254 = stablehlo.reshape %v2250 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2255 = stablehlo.reshape %v2253 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2257 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2258 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2259 = stablehlo.reduce(%v2254 init: %v2256) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2260 = stablehlo.broadcast_in_dim %v2259, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2261 = stablehlo.divide %v2260, %v2257 : tensor<32x196x384xf32>
    %v2262 = stablehlo.subtract %v2254, %v2261 : tensor<32x196x384xf32>
    %v2263 = stablehlo.multiply %v2262, %v2262 : tensor<32x196x384xf32>
    %v2264 = stablehlo.reduce(%v2263 init: %v2256) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2265 = stablehlo.broadcast_in_dim %v2264, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2266 = stablehlo.divide %v2265, %v2257 : tensor<32x196x384xf32>
    %v2267 = stablehlo.add %v2266, %v2258 : tensor<32x196x384xf32>
    %v2268 = stablehlo.rsqrt %v2267 : tensor<32x196x384xf32>
    %v2269 = stablehlo.multiply %v2262, %v2268 : tensor<32x196x384xf32>
    %v2270 = stablehlo.multiply %v2255, %v2269 : tensor<32x196x384xf32>
    %v2271 = stablehlo.reduce(%v2270 init: %v2256) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2272 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2273 = stablehlo.multiply %v2271, %v2272 : tensor<384xf32>
    %v2274 = stablehlo.subtract %s2b8ng, %v2273 : tensor<384xf32>
    %v2275 = stablehlo.reshape %v2158 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2276 = stablehlo.transpose %v2275, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2277 = stablehlo.reshape %v2276 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2279 = stablehlo.reshape %v2277 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2280 = stablehlo.reduce(%v2279 init: %v2278) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2281 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2282 = stablehlo.multiply %v2280, %v2281 : tensor<384xf32>
    %v2283 = stablehlo.subtract %s2b8nbt, %v2282 : tensor<384xf32>
    %v2284 = stablehlo.reshape %v1124 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2285 = stablehlo.reshape %v2201 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2286 = stablehlo.transpose %v2284, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2287 = stablehlo.transpose %v2285, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2288 = stablehlo.convolution(%v2286, %v2287)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2289 = stablehlo.reshape %v2288 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2290 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2291 = stablehlo.multiply %v2289, %v2290 : tensor<384x1x7x7xf32>
    %v2292 = stablehlo.subtract %s2b8dW, %v2291 : tensor<384x1x7x7xf32>
    %v2293 = stablehlo.reshape %v2201 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2295 = stablehlo.reduce(%v2293 init: %v2294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2296 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2297 = stablehlo.multiply %v2295, %v2296 : tensor<384xf32>
    %v2298 = stablehlo.subtract %s2b8db, %v2297 : tensor<384xf32>
    %v2299 = stablehlo.reshape %v2209 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2300 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2301 = stablehlo.multiply %v2299, %v2300 : tensor<32x384x14x14xf32>
    %v2302 = stablehlo.reshape %v2301 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2303 = stablehlo.reshape %v2302 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2304 = stablehlo.transpose %s2b7pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2305 = stablehlo.reverse %v2304, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2306 = stablehlo.convolution(%v2303, %v2305)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2307 = stablehlo.reshape %v2306 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2308 = stablehlo.reshape %v2307 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2309 = stablehlo.reshape %v1096 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2310 = stablehlo.multiply %v2309, %v2309 : tensor<32x1536x14x14xf32>
    %v2311 = stablehlo.multiply %v2310, %v2309 : tensor<32x1536x14x14xf32>
    %v2312 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v2313 = stablehlo.multiply %v2312, %v2311 : tensor<32x1536x14x14xf32>
    %v2314 = stablehlo.add %v2309, %v2313 : tensor<32x1536x14x14xf32>
    %v2315 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v2316 = stablehlo.multiply %v2315, %v2314 : tensor<32x1536x14x14xf32>
    %v2317 = stablehlo.tanh %v2316 : tensor<32x1536x14x14xf32>
    %v2318 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v2319 = stablehlo.add %v2318, %v2317 : tensor<32x1536x14x14xf32>
    %v2320 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2321 = stablehlo.multiply %v2320, %v2319 : tensor<32x1536x14x14xf32>
    %v2322 = stablehlo.multiply %v2317, %v2317 : tensor<32x1536x14x14xf32>
    %v2323 = stablehlo.subtract %v2318, %v2322 : tensor<32x1536x14x14xf32>
    %v2324 = stablehlo.multiply %v2320, %v2309 : tensor<32x1536x14x14xf32>
    %v2325 = stablehlo.multiply %v2324, %v2323 : tensor<32x1536x14x14xf32>
    %v2326 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v2327 = stablehlo.multiply %v2326, %v2310 : tensor<32x1536x14x14xf32>
    %v2328 = stablehlo.add %v2318, %v2327 : tensor<32x1536x14x14xf32>
    %v2329 = stablehlo.multiply %v2315, %v2328 : tensor<32x1536x14x14xf32>
    %v2330 = stablehlo.multiply %v2325, %v2329 : tensor<32x1536x14x14xf32>
    %v2331 = stablehlo.add %v2321, %v2330 : tensor<32x1536x14x14xf32>
    %v2332 = stablehlo.multiply %v2308, %v2331 : tensor<32x1536x14x14xf32>
    %v2333 = stablehlo.reshape %v2332 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2334 = stablehlo.reshape %v2333 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2335 = stablehlo.transpose %s2b7eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2336 = stablehlo.reverse %v2335, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2337 = stablehlo.convolution(%v2334, %v2336)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2338 = stablehlo.reshape %v2337 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2339 = stablehlo.reshape %v1057 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2340 = stablehlo.transpose %v2339, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2342 = stablehlo.reshape %v2338 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2343 = stablehlo.transpose %v2342, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2344 = stablehlo.reshape %v2343 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2345 = stablehlo.reshape %v2344 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2346 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2347 = stablehlo.multiply %v2345, %v2346 : tensor<32x196x384xf32>
    %v2348 = stablehlo.reshape %v2347 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2350 = stablehlo.reshape %v2341 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2352 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2353 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2354 = stablehlo.reduce(%v2350 init: %v2351) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2355 = stablehlo.broadcast_in_dim %v2354, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2356 = stablehlo.divide %v2355, %v2352 : tensor<32x196x384xf32>
    %v2357 = stablehlo.subtract %v2350, %v2356 : tensor<32x196x384xf32>
    %v2358 = stablehlo.multiply %v2357, %v2357 : tensor<32x196x384xf32>
    %v2359 = stablehlo.reduce(%v2358 init: %v2351) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2360 = stablehlo.broadcast_in_dim %v2359, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2361 = stablehlo.divide %v2360, %v2352 : tensor<32x196x384xf32>
    %v2362 = stablehlo.add %v2361, %v2353 : tensor<32x196x384xf32>
    %v2363 = stablehlo.rsqrt %v2362 : tensor<32x196x384xf32>
    %v2364 = stablehlo.multiply %v2357, %v2363 : tensor<32x196x384xf32>
    %v2365 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2366 = stablehlo.multiply %v2365, %v2349 : tensor<32x196x384xf32>
    %v2367 = stablehlo.reduce(%v2366 init: %v2351) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2368 = stablehlo.broadcast_in_dim %v2367, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2369 = stablehlo.multiply %v2364, %v2366 : tensor<32x196x384xf32>
    %v2370 = stablehlo.reduce(%v2369 init: %v2351) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2371 = stablehlo.broadcast_in_dim %v2370, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2372 = stablehlo.multiply %v2366, %v2352 : tensor<32x196x384xf32>
    %v2373 = stablehlo.subtract %v2372, %v2368 : tensor<32x196x384xf32>
    %v2374 = stablehlo.multiply %v2364, %v2371 : tensor<32x196x384xf32>
    %v2375 = stablehlo.subtract %v2373, %v2374 : tensor<32x196x384xf32>
    %v2376 = stablehlo.divide %v2363, %v2352 : tensor<32x196x384xf32>
    %v2377 = stablehlo.multiply %v2376, %v2375 : tensor<32x196x384xf32>
    %v2378 = stablehlo.reshape %v2377 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2379 = stablehlo.reshape %v2378 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2380 = stablehlo.transpose %v2379, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2381 = stablehlo.reshape %v2380 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2382 = stablehlo.reshape %v2381 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2383 = stablehlo.reverse %s2b7dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2384 = stablehlo.convolution(%v2382, %v2383)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2385 = stablehlo.reshape %v2384 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2386 = stablehlo.reshape %v2385 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2387 = stablehlo.reshape %v2209 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2388 = stablehlo.add %v2386, %v2387 : tensor<32x384x14x14xf32>
    %v2389 = stablehlo.reshape %v2388 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2391 = stablehlo.reshape %v1116 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2392 = stablehlo.reshape %v2209 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2393 = stablehlo.multiply %v2391, %v2392 : tensor<32x384x14x14xf32>
    %v2394 = stablehlo.reduce(%v2393 init: %v2390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2395 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2396 = stablehlo.multiply %v2394, %v2395 : tensor<384xf32>
    %v2397 = stablehlo.subtract %s2b7lg, %v2396 : tensor<384xf32>
    %v2398 = stablehlo.reshape %v1111 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2399 = stablehlo.reshape %v2302 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2400 = stablehlo.transpose %v2398, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2401 = stablehlo.transpose %v2399, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2402 = stablehlo.convolution(%v2400, %v2401)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2403 = stablehlo.transpose %v2402, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2404 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2405 = stablehlo.multiply %v2403, %v2404 : tensor<384x1536x1x1xf32>
    %v2406 = stablehlo.subtract %s2b7pW, %v2405 : tensor<384x1536x1x1xf32>
    %v2407 = stablehlo.reshape %v2302 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2408 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2409 = stablehlo.reduce(%v2407 init: %v2408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2410 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2411 = stablehlo.multiply %v2409, %v2410 : tensor<384xf32>
    %v2412 = stablehlo.subtract %s2b7pb, %v2411 : tensor<384xf32>
    %v2413 = stablehlo.reshape %v1091 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2414 = stablehlo.reshape %v2333 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2415 = stablehlo.transpose %v2413, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2416 = stablehlo.transpose %v2414, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2417 = stablehlo.convolution(%v2415, %v2416)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2418 = stablehlo.transpose %v2417, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2419 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2420 = stablehlo.multiply %v2418, %v2419 : tensor<1536x384x1x1xf32>
    %v2421 = stablehlo.subtract %s2b7eW, %v2420 : tensor<1536x384x1x1xf32>
    %v2422 = stablehlo.reshape %v2333 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2424 = stablehlo.reduce(%v2422 init: %v2423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2425 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2426 = stablehlo.multiply %v2424, %v2425 : tensor<1536xf32>
    %v2427 = stablehlo.subtract %s2b7eb, %v2426 : tensor<1536xf32>
    %v2428 = stablehlo.reshape %v1057 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2429 = stablehlo.transpose %v2428, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2430 = stablehlo.reshape %v2429 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2431 = stablehlo.reshape %v2338 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2432 = stablehlo.transpose %v2431, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2433 = stablehlo.reshape %v2432 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2434 = stablehlo.reshape %v2430 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2435 = stablehlo.reshape %v2433 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2436 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2437 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2438 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2439 = stablehlo.reduce(%v2434 init: %v2436) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2440 = stablehlo.broadcast_in_dim %v2439, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2441 = stablehlo.divide %v2440, %v2437 : tensor<32x196x384xf32>
    %v2442 = stablehlo.subtract %v2434, %v2441 : tensor<32x196x384xf32>
    %v2443 = stablehlo.multiply %v2442, %v2442 : tensor<32x196x384xf32>
    %v2444 = stablehlo.reduce(%v2443 init: %v2436) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2445 = stablehlo.broadcast_in_dim %v2444, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2446 = stablehlo.divide %v2445, %v2437 : tensor<32x196x384xf32>
    %v2447 = stablehlo.add %v2446, %v2438 : tensor<32x196x384xf32>
    %v2448 = stablehlo.rsqrt %v2447 : tensor<32x196x384xf32>
    %v2449 = stablehlo.multiply %v2442, %v2448 : tensor<32x196x384xf32>
    %v2450 = stablehlo.multiply %v2435, %v2449 : tensor<32x196x384xf32>
    %v2451 = stablehlo.reduce(%v2450 init: %v2436) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2452 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2453 = stablehlo.multiply %v2451, %v2452 : tensor<384xf32>
    %v2454 = stablehlo.subtract %s2b7ng, %v2453 : tensor<384xf32>
    %v2455 = stablehlo.reshape %v2338 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2456 = stablehlo.transpose %v2455, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2457 = stablehlo.reshape %v2456 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2459 = stablehlo.reshape %v2457 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2460 = stablehlo.reduce(%v2459 init: %v2458) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2461 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2462 = stablehlo.multiply %v2460, %v2461 : tensor<384xf32>
    %v2463 = stablehlo.subtract %s2b7nbt, %v2462 : tensor<384xf32>
    %v2464 = stablehlo.reshape %v1052 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2465 = stablehlo.reshape %v2381 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2466 = stablehlo.transpose %v2464, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2467 = stablehlo.transpose %v2465, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2468 = stablehlo.convolution(%v2466, %v2467)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2469 = stablehlo.reshape %v2468 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2470 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2471 = stablehlo.multiply %v2469, %v2470 : tensor<384x1x7x7xf32>
    %v2472 = stablehlo.subtract %s2b7dW, %v2471 : tensor<384x1x7x7xf32>
    %v2473 = stablehlo.reshape %v2381 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2475 = stablehlo.reduce(%v2473 init: %v2474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2476 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2477 = stablehlo.multiply %v2475, %v2476 : tensor<384xf32>
    %v2478 = stablehlo.subtract %s2b7db, %v2477 : tensor<384xf32>
    %v2479 = stablehlo.reshape %v2389 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2480 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2481 = stablehlo.multiply %v2479, %v2480 : tensor<32x384x14x14xf32>
    %v2482 = stablehlo.reshape %v2481 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2483 = stablehlo.reshape %v2482 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2484 = stablehlo.transpose %s2b6pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2485 = stablehlo.reverse %v2484, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2486 = stablehlo.convolution(%v2483, %v2485)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2487 = stablehlo.reshape %v2486 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2488 = stablehlo.reshape %v2487 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2489 = stablehlo.reshape %v1024 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2490 = stablehlo.multiply %v2489, %v2489 : tensor<32x1536x14x14xf32>
    %v2491 = stablehlo.multiply %v2490, %v2489 : tensor<32x1536x14x14xf32>
    %v2492 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v2493 = stablehlo.multiply %v2492, %v2491 : tensor<32x1536x14x14xf32>
    %v2494 = stablehlo.add %v2489, %v2493 : tensor<32x1536x14x14xf32>
    %v2495 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v2496 = stablehlo.multiply %v2495, %v2494 : tensor<32x1536x14x14xf32>
    %v2497 = stablehlo.tanh %v2496 : tensor<32x1536x14x14xf32>
    %v2498 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v2499 = stablehlo.add %v2498, %v2497 : tensor<32x1536x14x14xf32>
    %v2500 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2501 = stablehlo.multiply %v2500, %v2499 : tensor<32x1536x14x14xf32>
    %v2502 = stablehlo.multiply %v2497, %v2497 : tensor<32x1536x14x14xf32>
    %v2503 = stablehlo.subtract %v2498, %v2502 : tensor<32x1536x14x14xf32>
    %v2504 = stablehlo.multiply %v2500, %v2489 : tensor<32x1536x14x14xf32>
    %v2505 = stablehlo.multiply %v2504, %v2503 : tensor<32x1536x14x14xf32>
    %v2506 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v2507 = stablehlo.multiply %v2506, %v2490 : tensor<32x1536x14x14xf32>
    %v2508 = stablehlo.add %v2498, %v2507 : tensor<32x1536x14x14xf32>
    %v2509 = stablehlo.multiply %v2495, %v2508 : tensor<32x1536x14x14xf32>
    %v2510 = stablehlo.multiply %v2505, %v2509 : tensor<32x1536x14x14xf32>
    %v2511 = stablehlo.add %v2501, %v2510 : tensor<32x1536x14x14xf32>
    %v2512 = stablehlo.multiply %v2488, %v2511 : tensor<32x1536x14x14xf32>
    %v2513 = stablehlo.reshape %v2512 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2514 = stablehlo.reshape %v2513 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2515 = stablehlo.transpose %s2b6eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2516 = stablehlo.reverse %v2515, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2517 = stablehlo.convolution(%v2514, %v2516)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2518 = stablehlo.reshape %v2517 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2519 = stablehlo.reshape %v985 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2520 = stablehlo.transpose %v2519, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2521 = stablehlo.reshape %v2520 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2522 = stablehlo.reshape %v2518 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2523 = stablehlo.transpose %v2522, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2524 = stablehlo.reshape %v2523 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2525 = stablehlo.reshape %v2524 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2526 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2527 = stablehlo.multiply %v2525, %v2526 : tensor<32x196x384xf32>
    %v2528 = stablehlo.reshape %v2527 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2529 = stablehlo.reshape %v2528 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2530 = stablehlo.reshape %v2521 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2532 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2533 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2534 = stablehlo.reduce(%v2530 init: %v2531) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2535 = stablehlo.broadcast_in_dim %v2534, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2536 = stablehlo.divide %v2535, %v2532 : tensor<32x196x384xf32>
    %v2537 = stablehlo.subtract %v2530, %v2536 : tensor<32x196x384xf32>
    %v2538 = stablehlo.multiply %v2537, %v2537 : tensor<32x196x384xf32>
    %v2539 = stablehlo.reduce(%v2538 init: %v2531) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2540 = stablehlo.broadcast_in_dim %v2539, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2541 = stablehlo.divide %v2540, %v2532 : tensor<32x196x384xf32>
    %v2542 = stablehlo.add %v2541, %v2533 : tensor<32x196x384xf32>
    %v2543 = stablehlo.rsqrt %v2542 : tensor<32x196x384xf32>
    %v2544 = stablehlo.multiply %v2537, %v2543 : tensor<32x196x384xf32>
    %v2545 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2546 = stablehlo.multiply %v2545, %v2529 : tensor<32x196x384xf32>
    %v2547 = stablehlo.reduce(%v2546 init: %v2531) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2548 = stablehlo.broadcast_in_dim %v2547, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2549 = stablehlo.multiply %v2544, %v2546 : tensor<32x196x384xf32>
    %v2550 = stablehlo.reduce(%v2549 init: %v2531) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2551 = stablehlo.broadcast_in_dim %v2550, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2552 = stablehlo.multiply %v2546, %v2532 : tensor<32x196x384xf32>
    %v2553 = stablehlo.subtract %v2552, %v2548 : tensor<32x196x384xf32>
    %v2554 = stablehlo.multiply %v2544, %v2551 : tensor<32x196x384xf32>
    %v2555 = stablehlo.subtract %v2553, %v2554 : tensor<32x196x384xf32>
    %v2556 = stablehlo.divide %v2543, %v2532 : tensor<32x196x384xf32>
    %v2557 = stablehlo.multiply %v2556, %v2555 : tensor<32x196x384xf32>
    %v2558 = stablehlo.reshape %v2557 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2559 = stablehlo.reshape %v2558 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2560 = stablehlo.transpose %v2559, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2561 = stablehlo.reshape %v2560 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2562 = stablehlo.reshape %v2561 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2563 = stablehlo.reverse %s2b6dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2564 = stablehlo.convolution(%v2562, %v2563)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2565 = stablehlo.reshape %v2564 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2566 = stablehlo.reshape %v2565 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2567 = stablehlo.reshape %v2389 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2568 = stablehlo.add %v2566, %v2567 : tensor<32x384x14x14xf32>
    %v2569 = stablehlo.reshape %v2568 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2571 = stablehlo.reshape %v1044 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2572 = stablehlo.reshape %v2389 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2573 = stablehlo.multiply %v2571, %v2572 : tensor<32x384x14x14xf32>
    %v2574 = stablehlo.reduce(%v2573 init: %v2570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2575 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2576 = stablehlo.multiply %v2574, %v2575 : tensor<384xf32>
    %v2577 = stablehlo.subtract %s2b6lg, %v2576 : tensor<384xf32>
    %v2578 = stablehlo.reshape %v1039 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2579 = stablehlo.reshape %v2482 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2580 = stablehlo.transpose %v2578, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2581 = stablehlo.transpose %v2579, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2582 = stablehlo.convolution(%v2580, %v2581)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2583 = stablehlo.transpose %v2582, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2584 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2585 = stablehlo.multiply %v2583, %v2584 : tensor<384x1536x1x1xf32>
    %v2586 = stablehlo.subtract %s2b6pW, %v2585 : tensor<384x1536x1x1xf32>
    %v2587 = stablehlo.reshape %v2482 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2588 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2589 = stablehlo.reduce(%v2587 init: %v2588) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2590 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2591 = stablehlo.multiply %v2589, %v2590 : tensor<384xf32>
    %v2592 = stablehlo.subtract %s2b6pb, %v2591 : tensor<384xf32>
    %v2593 = stablehlo.reshape %v1019 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2594 = stablehlo.reshape %v2513 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2595 = stablehlo.transpose %v2593, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2596 = stablehlo.transpose %v2594, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2597 = stablehlo.convolution(%v2595, %v2596)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2598 = stablehlo.transpose %v2597, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2599 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2600 = stablehlo.multiply %v2598, %v2599 : tensor<1536x384x1x1xf32>
    %v2601 = stablehlo.subtract %s2b6eW, %v2600 : tensor<1536x384x1x1xf32>
    %v2602 = stablehlo.reshape %v2513 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2604 = stablehlo.reduce(%v2602 init: %v2603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2605 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2606 = stablehlo.multiply %v2604, %v2605 : tensor<1536xf32>
    %v2607 = stablehlo.subtract %s2b6eb, %v2606 : tensor<1536xf32>
    %v2608 = stablehlo.reshape %v985 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2609 = stablehlo.transpose %v2608, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2611 = stablehlo.reshape %v2518 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2612 = stablehlo.transpose %v2611, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2613 = stablehlo.reshape %v2612 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2614 = stablehlo.reshape %v2610 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2615 = stablehlo.reshape %v2613 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2617 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2618 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2619 = stablehlo.reduce(%v2614 init: %v2616) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2620 = stablehlo.broadcast_in_dim %v2619, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2621 = stablehlo.divide %v2620, %v2617 : tensor<32x196x384xf32>
    %v2622 = stablehlo.subtract %v2614, %v2621 : tensor<32x196x384xf32>
    %v2623 = stablehlo.multiply %v2622, %v2622 : tensor<32x196x384xf32>
    %v2624 = stablehlo.reduce(%v2623 init: %v2616) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2625 = stablehlo.broadcast_in_dim %v2624, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2626 = stablehlo.divide %v2625, %v2617 : tensor<32x196x384xf32>
    %v2627 = stablehlo.add %v2626, %v2618 : tensor<32x196x384xf32>
    %v2628 = stablehlo.rsqrt %v2627 : tensor<32x196x384xf32>
    %v2629 = stablehlo.multiply %v2622, %v2628 : tensor<32x196x384xf32>
    %v2630 = stablehlo.multiply %v2615, %v2629 : tensor<32x196x384xf32>
    %v2631 = stablehlo.reduce(%v2630 init: %v2616) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2632 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2633 = stablehlo.multiply %v2631, %v2632 : tensor<384xf32>
    %v2634 = stablehlo.subtract %s2b6ng, %v2633 : tensor<384xf32>
    %v2635 = stablehlo.reshape %v2518 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2636 = stablehlo.transpose %v2635, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2637 = stablehlo.reshape %v2636 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2639 = stablehlo.reshape %v2637 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2640 = stablehlo.reduce(%v2639 init: %v2638) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2641 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2642 = stablehlo.multiply %v2640, %v2641 : tensor<384xf32>
    %v2643 = stablehlo.subtract %s2b6nbt, %v2642 : tensor<384xf32>
    %v2644 = stablehlo.reshape %v980 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2645 = stablehlo.reshape %v2561 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2646 = stablehlo.transpose %v2644, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2647 = stablehlo.transpose %v2645, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2648 = stablehlo.convolution(%v2646, %v2647)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2649 = stablehlo.reshape %v2648 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2650 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2651 = stablehlo.multiply %v2649, %v2650 : tensor<384x1x7x7xf32>
    %v2652 = stablehlo.subtract %s2b6dW, %v2651 : tensor<384x1x7x7xf32>
    %v2653 = stablehlo.reshape %v2561 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2655 = stablehlo.reduce(%v2653 init: %v2654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2656 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2657 = stablehlo.multiply %v2655, %v2656 : tensor<384xf32>
    %v2658 = stablehlo.subtract %s2b6db, %v2657 : tensor<384xf32>
    %v2659 = stablehlo.reshape %v2569 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2660 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2661 = stablehlo.multiply %v2659, %v2660 : tensor<32x384x14x14xf32>
    %v2662 = stablehlo.reshape %v2661 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2663 = stablehlo.reshape %v2662 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2664 = stablehlo.transpose %s2b5pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2665 = stablehlo.reverse %v2664, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2666 = stablehlo.convolution(%v2663, %v2665)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2667 = stablehlo.reshape %v2666 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2668 = stablehlo.reshape %v2667 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2669 = stablehlo.reshape %v952 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2670 = stablehlo.multiply %v2669, %v2669 : tensor<32x1536x14x14xf32>
    %v2671 = stablehlo.multiply %v2670, %v2669 : tensor<32x1536x14x14xf32>
    %v2672 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v2673 = stablehlo.multiply %v2672, %v2671 : tensor<32x1536x14x14xf32>
    %v2674 = stablehlo.add %v2669, %v2673 : tensor<32x1536x14x14xf32>
    %v2675 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v2676 = stablehlo.multiply %v2675, %v2674 : tensor<32x1536x14x14xf32>
    %v2677 = stablehlo.tanh %v2676 : tensor<32x1536x14x14xf32>
    %v2678 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v2679 = stablehlo.add %v2678, %v2677 : tensor<32x1536x14x14xf32>
    %v2680 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2681 = stablehlo.multiply %v2680, %v2679 : tensor<32x1536x14x14xf32>
    %v2682 = stablehlo.multiply %v2677, %v2677 : tensor<32x1536x14x14xf32>
    %v2683 = stablehlo.subtract %v2678, %v2682 : tensor<32x1536x14x14xf32>
    %v2684 = stablehlo.multiply %v2680, %v2669 : tensor<32x1536x14x14xf32>
    %v2685 = stablehlo.multiply %v2684, %v2683 : tensor<32x1536x14x14xf32>
    %v2686 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v2687 = stablehlo.multiply %v2686, %v2670 : tensor<32x1536x14x14xf32>
    %v2688 = stablehlo.add %v2678, %v2687 : tensor<32x1536x14x14xf32>
    %v2689 = stablehlo.multiply %v2675, %v2688 : tensor<32x1536x14x14xf32>
    %v2690 = stablehlo.multiply %v2685, %v2689 : tensor<32x1536x14x14xf32>
    %v2691 = stablehlo.add %v2681, %v2690 : tensor<32x1536x14x14xf32>
    %v2692 = stablehlo.multiply %v2668, %v2691 : tensor<32x1536x14x14xf32>
    %v2693 = stablehlo.reshape %v2692 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2694 = stablehlo.reshape %v2693 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2695 = stablehlo.transpose %s2b5eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2696 = stablehlo.reverse %v2695, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2697 = stablehlo.convolution(%v2694, %v2696)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2698 = stablehlo.reshape %v2697 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2699 = stablehlo.reshape %v913 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2700 = stablehlo.transpose %v2699, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2701 = stablehlo.reshape %v2700 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2702 = stablehlo.reshape %v2698 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2703 = stablehlo.transpose %v2702, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2704 = stablehlo.reshape %v2703 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2705 = stablehlo.reshape %v2704 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2706 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2707 = stablehlo.multiply %v2705, %v2706 : tensor<32x196x384xf32>
    %v2708 = stablehlo.reshape %v2707 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2709 = stablehlo.reshape %v2708 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2710 = stablehlo.reshape %v2701 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2712 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2713 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2714 = stablehlo.reduce(%v2710 init: %v2711) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2715 = stablehlo.broadcast_in_dim %v2714, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2716 = stablehlo.divide %v2715, %v2712 : tensor<32x196x384xf32>
    %v2717 = stablehlo.subtract %v2710, %v2716 : tensor<32x196x384xf32>
    %v2718 = stablehlo.multiply %v2717, %v2717 : tensor<32x196x384xf32>
    %v2719 = stablehlo.reduce(%v2718 init: %v2711) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2720 = stablehlo.broadcast_in_dim %v2719, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2721 = stablehlo.divide %v2720, %v2712 : tensor<32x196x384xf32>
    %v2722 = stablehlo.add %v2721, %v2713 : tensor<32x196x384xf32>
    %v2723 = stablehlo.rsqrt %v2722 : tensor<32x196x384xf32>
    %v2724 = stablehlo.multiply %v2717, %v2723 : tensor<32x196x384xf32>
    %v2725 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2726 = stablehlo.multiply %v2725, %v2709 : tensor<32x196x384xf32>
    %v2727 = stablehlo.reduce(%v2726 init: %v2711) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2728 = stablehlo.broadcast_in_dim %v2727, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2729 = stablehlo.multiply %v2724, %v2726 : tensor<32x196x384xf32>
    %v2730 = stablehlo.reduce(%v2729 init: %v2711) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2731 = stablehlo.broadcast_in_dim %v2730, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2732 = stablehlo.multiply %v2726, %v2712 : tensor<32x196x384xf32>
    %v2733 = stablehlo.subtract %v2732, %v2728 : tensor<32x196x384xf32>
    %v2734 = stablehlo.multiply %v2724, %v2731 : tensor<32x196x384xf32>
    %v2735 = stablehlo.subtract %v2733, %v2734 : tensor<32x196x384xf32>
    %v2736 = stablehlo.divide %v2723, %v2712 : tensor<32x196x384xf32>
    %v2737 = stablehlo.multiply %v2736, %v2735 : tensor<32x196x384xf32>
    %v2738 = stablehlo.reshape %v2737 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2739 = stablehlo.reshape %v2738 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2740 = stablehlo.transpose %v2739, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2741 = stablehlo.reshape %v2740 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2742 = stablehlo.reshape %v2741 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2743 = stablehlo.reverse %s2b5dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2744 = stablehlo.convolution(%v2742, %v2743)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2745 = stablehlo.reshape %v2744 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2746 = stablehlo.reshape %v2745 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2747 = stablehlo.reshape %v2569 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2748 = stablehlo.add %v2746, %v2747 : tensor<32x384x14x14xf32>
    %v2749 = stablehlo.reshape %v2748 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2751 = stablehlo.reshape %v972 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2752 = stablehlo.reshape %v2569 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2753 = stablehlo.multiply %v2751, %v2752 : tensor<32x384x14x14xf32>
    %v2754 = stablehlo.reduce(%v2753 init: %v2750) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2755 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2756 = stablehlo.multiply %v2754, %v2755 : tensor<384xf32>
    %v2757 = stablehlo.subtract %s2b5lg, %v2756 : tensor<384xf32>
    %v2758 = stablehlo.reshape %v967 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2759 = stablehlo.reshape %v2662 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2760 = stablehlo.transpose %v2758, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2761 = stablehlo.transpose %v2759, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2762 = stablehlo.convolution(%v2760, %v2761)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2763 = stablehlo.transpose %v2762, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2764 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2765 = stablehlo.multiply %v2763, %v2764 : tensor<384x1536x1x1xf32>
    %v2766 = stablehlo.subtract %s2b5pW, %v2765 : tensor<384x1536x1x1xf32>
    %v2767 = stablehlo.reshape %v2662 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2768 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2769 = stablehlo.reduce(%v2767 init: %v2768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2770 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2771 = stablehlo.multiply %v2769, %v2770 : tensor<384xf32>
    %v2772 = stablehlo.subtract %s2b5pb, %v2771 : tensor<384xf32>
    %v2773 = stablehlo.reshape %v947 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2774 = stablehlo.reshape %v2693 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2775 = stablehlo.transpose %v2773, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2776 = stablehlo.transpose %v2774, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2777 = stablehlo.convolution(%v2775, %v2776)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2778 = stablehlo.transpose %v2777, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2779 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2780 = stablehlo.multiply %v2778, %v2779 : tensor<1536x384x1x1xf32>
    %v2781 = stablehlo.subtract %s2b5eW, %v2780 : tensor<1536x384x1x1xf32>
    %v2782 = stablehlo.reshape %v2693 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2784 = stablehlo.reduce(%v2782 init: %v2783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2785 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2786 = stablehlo.multiply %v2784, %v2785 : tensor<1536xf32>
    %v2787 = stablehlo.subtract %s2b5eb, %v2786 : tensor<1536xf32>
    %v2788 = stablehlo.reshape %v913 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2789 = stablehlo.transpose %v2788, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2790 = stablehlo.reshape %v2789 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2791 = stablehlo.reshape %v2698 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2792 = stablehlo.transpose %v2791, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2793 = stablehlo.reshape %v2792 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2794 = stablehlo.reshape %v2790 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2795 = stablehlo.reshape %v2793 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2797 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2798 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2799 = stablehlo.reduce(%v2794 init: %v2796) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2800 = stablehlo.broadcast_in_dim %v2799, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2801 = stablehlo.divide %v2800, %v2797 : tensor<32x196x384xf32>
    %v2802 = stablehlo.subtract %v2794, %v2801 : tensor<32x196x384xf32>
    %v2803 = stablehlo.multiply %v2802, %v2802 : tensor<32x196x384xf32>
    %v2804 = stablehlo.reduce(%v2803 init: %v2796) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2805 = stablehlo.broadcast_in_dim %v2804, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2806 = stablehlo.divide %v2805, %v2797 : tensor<32x196x384xf32>
    %v2807 = stablehlo.add %v2806, %v2798 : tensor<32x196x384xf32>
    %v2808 = stablehlo.rsqrt %v2807 : tensor<32x196x384xf32>
    %v2809 = stablehlo.multiply %v2802, %v2808 : tensor<32x196x384xf32>
    %v2810 = stablehlo.multiply %v2795, %v2809 : tensor<32x196x384xf32>
    %v2811 = stablehlo.reduce(%v2810 init: %v2796) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2812 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2813 = stablehlo.multiply %v2811, %v2812 : tensor<384xf32>
    %v2814 = stablehlo.subtract %s2b5ng, %v2813 : tensor<384xf32>
    %v2815 = stablehlo.reshape %v2698 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2816 = stablehlo.transpose %v2815, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2817 = stablehlo.reshape %v2816 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2819 = stablehlo.reshape %v2817 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2820 = stablehlo.reduce(%v2819 init: %v2818) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2821 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2822 = stablehlo.multiply %v2820, %v2821 : tensor<384xf32>
    %v2823 = stablehlo.subtract %s2b5nbt, %v2822 : tensor<384xf32>
    %v2824 = stablehlo.reshape %v908 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2825 = stablehlo.reshape %v2741 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2826 = stablehlo.transpose %v2824, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2827 = stablehlo.transpose %v2825, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2828 = stablehlo.convolution(%v2826, %v2827)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2829 = stablehlo.reshape %v2828 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2830 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v2831 = stablehlo.multiply %v2829, %v2830 : tensor<384x1x7x7xf32>
    %v2832 = stablehlo.subtract %s2b5dW, %v2831 : tensor<384x1x7x7xf32>
    %v2833 = stablehlo.reshape %v2741 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2835 = stablehlo.reduce(%v2833 init: %v2834) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2836 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2837 = stablehlo.multiply %v2835, %v2836 : tensor<384xf32>
    %v2838 = stablehlo.subtract %s2b5db, %v2837 : tensor<384xf32>
    %v2839 = stablehlo.reshape %v2749 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2840 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2841 = stablehlo.multiply %v2839, %v2840 : tensor<32x384x14x14xf32>
    %v2842 = stablehlo.reshape %v2841 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2843 = stablehlo.reshape %v2842 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2844 = stablehlo.transpose %s2b4pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2845 = stablehlo.reverse %v2844, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2846 = stablehlo.convolution(%v2843, %v2845)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2847 = stablehlo.reshape %v2846 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2849 = stablehlo.reshape %v880 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2850 = stablehlo.multiply %v2849, %v2849 : tensor<32x1536x14x14xf32>
    %v2851 = stablehlo.multiply %v2850, %v2849 : tensor<32x1536x14x14xf32>
    %v2852 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v2853 = stablehlo.multiply %v2852, %v2851 : tensor<32x1536x14x14xf32>
    %v2854 = stablehlo.add %v2849, %v2853 : tensor<32x1536x14x14xf32>
    %v2855 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v2856 = stablehlo.multiply %v2855, %v2854 : tensor<32x1536x14x14xf32>
    %v2857 = stablehlo.tanh %v2856 : tensor<32x1536x14x14xf32>
    %v2858 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v2859 = stablehlo.add %v2858, %v2857 : tensor<32x1536x14x14xf32>
    %v2860 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v2861 = stablehlo.multiply %v2860, %v2859 : tensor<32x1536x14x14xf32>
    %v2862 = stablehlo.multiply %v2857, %v2857 : tensor<32x1536x14x14xf32>
    %v2863 = stablehlo.subtract %v2858, %v2862 : tensor<32x1536x14x14xf32>
    %v2864 = stablehlo.multiply %v2860, %v2849 : tensor<32x1536x14x14xf32>
    %v2865 = stablehlo.multiply %v2864, %v2863 : tensor<32x1536x14x14xf32>
    %v2866 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v2867 = stablehlo.multiply %v2866, %v2850 : tensor<32x1536x14x14xf32>
    %v2868 = stablehlo.add %v2858, %v2867 : tensor<32x1536x14x14xf32>
    %v2869 = stablehlo.multiply %v2855, %v2868 : tensor<32x1536x14x14xf32>
    %v2870 = stablehlo.multiply %v2865, %v2869 : tensor<32x1536x14x14xf32>
    %v2871 = stablehlo.add %v2861, %v2870 : tensor<32x1536x14x14xf32>
    %v2872 = stablehlo.multiply %v2848, %v2871 : tensor<32x1536x14x14xf32>
    %v2873 = stablehlo.reshape %v2872 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2874 = stablehlo.reshape %v2873 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2875 = stablehlo.transpose %s2b4eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2876 = stablehlo.reverse %v2875, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2877 = stablehlo.convolution(%v2874, %v2876)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2878 = stablehlo.reshape %v2877 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2879 = stablehlo.reshape %v841 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2880 = stablehlo.transpose %v2879, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2881 = stablehlo.reshape %v2880 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2882 = stablehlo.reshape %v2878 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2883 = stablehlo.transpose %v2882, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2884 = stablehlo.reshape %v2883 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2885 = stablehlo.reshape %v2884 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2886 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2887 = stablehlo.multiply %v2885, %v2886 : tensor<32x196x384xf32>
    %v2888 = stablehlo.reshape %v2887 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2889 = stablehlo.reshape %v2888 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2890 = stablehlo.reshape %v2881 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2892 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2893 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2894 = stablehlo.reduce(%v2890 init: %v2891) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2895 = stablehlo.broadcast_in_dim %v2894, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2896 = stablehlo.divide %v2895, %v2892 : tensor<32x196x384xf32>
    %v2897 = stablehlo.subtract %v2890, %v2896 : tensor<32x196x384xf32>
    %v2898 = stablehlo.multiply %v2897, %v2897 : tensor<32x196x384xf32>
    %v2899 = stablehlo.reduce(%v2898 init: %v2891) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2900 = stablehlo.broadcast_in_dim %v2899, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2901 = stablehlo.divide %v2900, %v2892 : tensor<32x196x384xf32>
    %v2902 = stablehlo.add %v2901, %v2893 : tensor<32x196x384xf32>
    %v2903 = stablehlo.rsqrt %v2902 : tensor<32x196x384xf32>
    %v2904 = stablehlo.multiply %v2897, %v2903 : tensor<32x196x384xf32>
    %v2905 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2906 = stablehlo.multiply %v2905, %v2889 : tensor<32x196x384xf32>
    %v2907 = stablehlo.reduce(%v2906 init: %v2891) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2908 = stablehlo.broadcast_in_dim %v2907, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2909 = stablehlo.multiply %v2904, %v2906 : tensor<32x196x384xf32>
    %v2910 = stablehlo.reduce(%v2909 init: %v2891) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2911 = stablehlo.broadcast_in_dim %v2910, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2912 = stablehlo.multiply %v2906, %v2892 : tensor<32x196x384xf32>
    %v2913 = stablehlo.subtract %v2912, %v2908 : tensor<32x196x384xf32>
    %v2914 = stablehlo.multiply %v2904, %v2911 : tensor<32x196x384xf32>
    %v2915 = stablehlo.subtract %v2913, %v2914 : tensor<32x196x384xf32>
    %v2916 = stablehlo.divide %v2903, %v2892 : tensor<32x196x384xf32>
    %v2917 = stablehlo.multiply %v2916, %v2915 : tensor<32x196x384xf32>
    %v2918 = stablehlo.reshape %v2917 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2919 = stablehlo.reshape %v2918 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2920 = stablehlo.transpose %v2919, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2921 = stablehlo.reshape %v2920 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2922 = stablehlo.reshape %v2921 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2923 = stablehlo.reverse %s2b4dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2924 = stablehlo.convolution(%v2922, %v2923)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2925 = stablehlo.reshape %v2924 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2926 = stablehlo.reshape %v2925 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2927 = stablehlo.reshape %v2749 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2928 = stablehlo.add %v2926, %v2927 : tensor<32x384x14x14xf32>
    %v2929 = stablehlo.reshape %v2928 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2931 = stablehlo.reshape %v900 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2932 = stablehlo.reshape %v2749 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2933 = stablehlo.multiply %v2931, %v2932 : tensor<32x384x14x14xf32>
    %v2934 = stablehlo.reduce(%v2933 init: %v2930) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2935 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2936 = stablehlo.multiply %v2934, %v2935 : tensor<384xf32>
    %v2937 = stablehlo.subtract %s2b4lg, %v2936 : tensor<384xf32>
    %v2938 = stablehlo.reshape %v895 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2939 = stablehlo.reshape %v2842 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2940 = stablehlo.transpose %v2938, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2941 = stablehlo.transpose %v2939, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2942 = stablehlo.convolution(%v2940, %v2941)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2943 = stablehlo.transpose %v2942, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2944 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v2945 = stablehlo.multiply %v2943, %v2944 : tensor<384x1536x1x1xf32>
    %v2946 = stablehlo.subtract %s2b4pW, %v2945 : tensor<384x1536x1x1xf32>
    %v2947 = stablehlo.reshape %v2842 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2949 = stablehlo.reduce(%v2947 init: %v2948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2950 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2951 = stablehlo.multiply %v2949, %v2950 : tensor<384xf32>
    %v2952 = stablehlo.subtract %s2b4pb, %v2951 : tensor<384xf32>
    %v2953 = stablehlo.reshape %v875 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2954 = stablehlo.reshape %v2873 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2955 = stablehlo.transpose %v2953, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2956 = stablehlo.transpose %v2954, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2957 = stablehlo.convolution(%v2955, %v2956)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2958 = stablehlo.transpose %v2957, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2959 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v2960 = stablehlo.multiply %v2958, %v2959 : tensor<1536x384x1x1xf32>
    %v2961 = stablehlo.subtract %s2b4eW, %v2960 : tensor<1536x384x1x1xf32>
    %v2962 = stablehlo.reshape %v2873 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2963 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2964 = stablehlo.reduce(%v2962 init: %v2963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2965 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v2966 = stablehlo.multiply %v2964, %v2965 : tensor<1536xf32>
    %v2967 = stablehlo.subtract %s2b4eb, %v2966 : tensor<1536xf32>
    %v2968 = stablehlo.reshape %v841 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2969 = stablehlo.transpose %v2968, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2970 = stablehlo.reshape %v2969 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2971 = stablehlo.reshape %v2878 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2972 = stablehlo.transpose %v2971, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2973 = stablehlo.reshape %v2972 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2974 = stablehlo.reshape %v2970 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2975 = stablehlo.reshape %v2973 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2976 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2977 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2978 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2979 = stablehlo.reduce(%v2974 init: %v2976) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2980 = stablehlo.broadcast_in_dim %v2979, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2981 = stablehlo.divide %v2980, %v2977 : tensor<32x196x384xf32>
    %v2982 = stablehlo.subtract %v2974, %v2981 : tensor<32x196x384xf32>
    %v2983 = stablehlo.multiply %v2982, %v2982 : tensor<32x196x384xf32>
    %v2984 = stablehlo.reduce(%v2983 init: %v2976) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2985 = stablehlo.broadcast_in_dim %v2984, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2986 = stablehlo.divide %v2985, %v2977 : tensor<32x196x384xf32>
    %v2987 = stablehlo.add %v2986, %v2978 : tensor<32x196x384xf32>
    %v2988 = stablehlo.rsqrt %v2987 : tensor<32x196x384xf32>
    %v2989 = stablehlo.multiply %v2982, %v2988 : tensor<32x196x384xf32>
    %v2990 = stablehlo.multiply %v2975, %v2989 : tensor<32x196x384xf32>
    %v2991 = stablehlo.reduce(%v2990 init: %v2976) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2992 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v2993 = stablehlo.multiply %v2991, %v2992 : tensor<384xf32>
    %v2994 = stablehlo.subtract %s2b4ng, %v2993 : tensor<384xf32>
    %v2995 = stablehlo.reshape %v2878 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2996 = stablehlo.transpose %v2995, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2997 = stablehlo.reshape %v2996 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2999 = stablehlo.reshape %v2997 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3000 = stablehlo.reduce(%v2999 init: %v2998) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3001 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3002 = stablehlo.multiply %v3000, %v3001 : tensor<384xf32>
    %v3003 = stablehlo.subtract %s2b4nbt, %v3002 : tensor<384xf32>
    %v3004 = stablehlo.reshape %v836 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3005 = stablehlo.reshape %v2921 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3006 = stablehlo.transpose %v3004, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3007 = stablehlo.transpose %v3005, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3008 = stablehlo.convolution(%v3006, %v3007)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3009 = stablehlo.reshape %v3008 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3010 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3011 = stablehlo.multiply %v3009, %v3010 : tensor<384x1x7x7xf32>
    %v3012 = stablehlo.subtract %s2b4dW, %v3011 : tensor<384x1x7x7xf32>
    %v3013 = stablehlo.reshape %v2921 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3015 = stablehlo.reduce(%v3013 init: %v3014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3016 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3017 = stablehlo.multiply %v3015, %v3016 : tensor<384xf32>
    %v3018 = stablehlo.subtract %s2b4db, %v3017 : tensor<384xf32>
    %v3019 = stablehlo.reshape %v2929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3020 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3021 = stablehlo.multiply %v3019, %v3020 : tensor<32x384x14x14xf32>
    %v3022 = stablehlo.reshape %v3021 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3023 = stablehlo.reshape %v3022 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3024 = stablehlo.transpose %s2b3pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3025 = stablehlo.reverse %v3024, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3026 = stablehlo.convolution(%v3023, %v3025)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3027 = stablehlo.reshape %v3026 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3028 = stablehlo.reshape %v3027 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3029 = stablehlo.reshape %v808 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3030 = stablehlo.multiply %v3029, %v3029 : tensor<32x1536x14x14xf32>
    %v3031 = stablehlo.multiply %v3030, %v3029 : tensor<32x1536x14x14xf32>
    %v3032 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v3033 = stablehlo.multiply %v3032, %v3031 : tensor<32x1536x14x14xf32>
    %v3034 = stablehlo.add %v3029, %v3033 : tensor<32x1536x14x14xf32>
    %v3035 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v3036 = stablehlo.multiply %v3035, %v3034 : tensor<32x1536x14x14xf32>
    %v3037 = stablehlo.tanh %v3036 : tensor<32x1536x14x14xf32>
    %v3038 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v3039 = stablehlo.add %v3038, %v3037 : tensor<32x1536x14x14xf32>
    %v3040 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v3041 = stablehlo.multiply %v3040, %v3039 : tensor<32x1536x14x14xf32>
    %v3042 = stablehlo.multiply %v3037, %v3037 : tensor<32x1536x14x14xf32>
    %v3043 = stablehlo.subtract %v3038, %v3042 : tensor<32x1536x14x14xf32>
    %v3044 = stablehlo.multiply %v3040, %v3029 : tensor<32x1536x14x14xf32>
    %v3045 = stablehlo.multiply %v3044, %v3043 : tensor<32x1536x14x14xf32>
    %v3046 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v3047 = stablehlo.multiply %v3046, %v3030 : tensor<32x1536x14x14xf32>
    %v3048 = stablehlo.add %v3038, %v3047 : tensor<32x1536x14x14xf32>
    %v3049 = stablehlo.multiply %v3035, %v3048 : tensor<32x1536x14x14xf32>
    %v3050 = stablehlo.multiply %v3045, %v3049 : tensor<32x1536x14x14xf32>
    %v3051 = stablehlo.add %v3041, %v3050 : tensor<32x1536x14x14xf32>
    %v3052 = stablehlo.multiply %v3028, %v3051 : tensor<32x1536x14x14xf32>
    %v3053 = stablehlo.reshape %v3052 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3054 = stablehlo.reshape %v3053 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3055 = stablehlo.transpose %s2b3eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3056 = stablehlo.reverse %v3055, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3057 = stablehlo.convolution(%v3054, %v3056)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3058 = stablehlo.reshape %v3057 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3059 = stablehlo.reshape %v769 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3060 = stablehlo.transpose %v3059, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3061 = stablehlo.reshape %v3060 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3062 = stablehlo.reshape %v3058 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3063 = stablehlo.transpose %v3062, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3064 = stablehlo.reshape %v3063 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3065 = stablehlo.reshape %v3064 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3066 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3067 = stablehlo.multiply %v3065, %v3066 : tensor<32x196x384xf32>
    %v3068 = stablehlo.reshape %v3067 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3069 = stablehlo.reshape %v3068 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3070 = stablehlo.reshape %v3061 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3071 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3072 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3073 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3074 = stablehlo.reduce(%v3070 init: %v3071) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3075 = stablehlo.broadcast_in_dim %v3074, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3076 = stablehlo.divide %v3075, %v3072 : tensor<32x196x384xf32>
    %v3077 = stablehlo.subtract %v3070, %v3076 : tensor<32x196x384xf32>
    %v3078 = stablehlo.multiply %v3077, %v3077 : tensor<32x196x384xf32>
    %v3079 = stablehlo.reduce(%v3078 init: %v3071) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3080 = stablehlo.broadcast_in_dim %v3079, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3081 = stablehlo.divide %v3080, %v3072 : tensor<32x196x384xf32>
    %v3082 = stablehlo.add %v3081, %v3073 : tensor<32x196x384xf32>
    %v3083 = stablehlo.rsqrt %v3082 : tensor<32x196x384xf32>
    %v3084 = stablehlo.multiply %v3077, %v3083 : tensor<32x196x384xf32>
    %v3085 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3086 = stablehlo.multiply %v3085, %v3069 : tensor<32x196x384xf32>
    %v3087 = stablehlo.reduce(%v3086 init: %v3071) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3088 = stablehlo.broadcast_in_dim %v3087, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3089 = stablehlo.multiply %v3084, %v3086 : tensor<32x196x384xf32>
    %v3090 = stablehlo.reduce(%v3089 init: %v3071) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3091 = stablehlo.broadcast_in_dim %v3090, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3092 = stablehlo.multiply %v3086, %v3072 : tensor<32x196x384xf32>
    %v3093 = stablehlo.subtract %v3092, %v3088 : tensor<32x196x384xf32>
    %v3094 = stablehlo.multiply %v3084, %v3091 : tensor<32x196x384xf32>
    %v3095 = stablehlo.subtract %v3093, %v3094 : tensor<32x196x384xf32>
    %v3096 = stablehlo.divide %v3083, %v3072 : tensor<32x196x384xf32>
    %v3097 = stablehlo.multiply %v3096, %v3095 : tensor<32x196x384xf32>
    %v3098 = stablehlo.reshape %v3097 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3099 = stablehlo.reshape %v3098 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3100 = stablehlo.transpose %v3099, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3102 = stablehlo.reshape %v3101 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3103 = stablehlo.reverse %s2b3dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3104 = stablehlo.convolution(%v3102, %v3103)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3105 = stablehlo.reshape %v3104 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3106 = stablehlo.reshape %v3105 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3107 = stablehlo.reshape %v2929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3108 = stablehlo.add %v3106, %v3107 : tensor<32x384x14x14xf32>
    %v3109 = stablehlo.reshape %v3108 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3110 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3111 = stablehlo.reshape %v828 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3112 = stablehlo.reshape %v2929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3113 = stablehlo.multiply %v3111, %v3112 : tensor<32x384x14x14xf32>
    %v3114 = stablehlo.reduce(%v3113 init: %v3110) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3115 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3116 = stablehlo.multiply %v3114, %v3115 : tensor<384xf32>
    %v3117 = stablehlo.subtract %s2b3lg, %v3116 : tensor<384xf32>
    %v3118 = stablehlo.reshape %v823 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3119 = stablehlo.reshape %v3022 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3120 = stablehlo.transpose %v3118, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3121 = stablehlo.transpose %v3119, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3122 = stablehlo.convolution(%v3120, %v3121)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3123 = stablehlo.transpose %v3122, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3124 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v3125 = stablehlo.multiply %v3123, %v3124 : tensor<384x1536x1x1xf32>
    %v3126 = stablehlo.subtract %s2b3pW, %v3125 : tensor<384x1536x1x1xf32>
    %v3127 = stablehlo.reshape %v3022 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3129 = stablehlo.reduce(%v3127 init: %v3128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3130 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3131 = stablehlo.multiply %v3129, %v3130 : tensor<384xf32>
    %v3132 = stablehlo.subtract %s2b3pb, %v3131 : tensor<384xf32>
    %v3133 = stablehlo.reshape %v803 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3134 = stablehlo.reshape %v3053 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3135 = stablehlo.transpose %v3133, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3136 = stablehlo.transpose %v3134, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3137 = stablehlo.convolution(%v3135, %v3136)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3138 = stablehlo.transpose %v3137, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3139 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v3140 = stablehlo.multiply %v3138, %v3139 : tensor<1536x384x1x1xf32>
    %v3141 = stablehlo.subtract %s2b3eW, %v3140 : tensor<1536x384x1x1xf32>
    %v3142 = stablehlo.reshape %v3053 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3144 = stablehlo.reduce(%v3142 init: %v3143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3145 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3146 = stablehlo.multiply %v3144, %v3145 : tensor<1536xf32>
    %v3147 = stablehlo.subtract %s2b3eb, %v3146 : tensor<1536xf32>
    %v3148 = stablehlo.reshape %v769 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3149 = stablehlo.transpose %v3148, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3150 = stablehlo.reshape %v3149 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3151 = stablehlo.reshape %v3058 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3152 = stablehlo.transpose %v3151, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3153 = stablehlo.reshape %v3152 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3154 = stablehlo.reshape %v3150 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3155 = stablehlo.reshape %v3153 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3157 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3158 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3159 = stablehlo.reduce(%v3154 init: %v3156) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3160 = stablehlo.broadcast_in_dim %v3159, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3161 = stablehlo.divide %v3160, %v3157 : tensor<32x196x384xf32>
    %v3162 = stablehlo.subtract %v3154, %v3161 : tensor<32x196x384xf32>
    %v3163 = stablehlo.multiply %v3162, %v3162 : tensor<32x196x384xf32>
    %v3164 = stablehlo.reduce(%v3163 init: %v3156) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3165 = stablehlo.broadcast_in_dim %v3164, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3166 = stablehlo.divide %v3165, %v3157 : tensor<32x196x384xf32>
    %v3167 = stablehlo.add %v3166, %v3158 : tensor<32x196x384xf32>
    %v3168 = stablehlo.rsqrt %v3167 : tensor<32x196x384xf32>
    %v3169 = stablehlo.multiply %v3162, %v3168 : tensor<32x196x384xf32>
    %v3170 = stablehlo.multiply %v3155, %v3169 : tensor<32x196x384xf32>
    %v3171 = stablehlo.reduce(%v3170 init: %v3156) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3172 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3173 = stablehlo.multiply %v3171, %v3172 : tensor<384xf32>
    %v3174 = stablehlo.subtract %s2b3ng, %v3173 : tensor<384xf32>
    %v3175 = stablehlo.reshape %v3058 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3176 = stablehlo.transpose %v3175, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3177 = stablehlo.reshape %v3176 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3179 = stablehlo.reshape %v3177 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3180 = stablehlo.reduce(%v3179 init: %v3178) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3181 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3182 = stablehlo.multiply %v3180, %v3181 : tensor<384xf32>
    %v3183 = stablehlo.subtract %s2b3nbt, %v3182 : tensor<384xf32>
    %v3184 = stablehlo.reshape %v764 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3185 = stablehlo.reshape %v3101 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3186 = stablehlo.transpose %v3184, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3187 = stablehlo.transpose %v3185, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3188 = stablehlo.convolution(%v3186, %v3187)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3189 = stablehlo.reshape %v3188 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3190 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3191 = stablehlo.multiply %v3189, %v3190 : tensor<384x1x7x7xf32>
    %v3192 = stablehlo.subtract %s2b3dW, %v3191 : tensor<384x1x7x7xf32>
    %v3193 = stablehlo.reshape %v3101 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3195 = stablehlo.reduce(%v3193 init: %v3194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3196 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3197 = stablehlo.multiply %v3195, %v3196 : tensor<384xf32>
    %v3198 = stablehlo.subtract %s2b3db, %v3197 : tensor<384xf32>
    %v3199 = stablehlo.reshape %v3109 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3200 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3201 = stablehlo.multiply %v3199, %v3200 : tensor<32x384x14x14xf32>
    %v3202 = stablehlo.reshape %v3201 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3203 = stablehlo.reshape %v3202 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3204 = stablehlo.transpose %s2b2pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3205 = stablehlo.reverse %v3204, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3206 = stablehlo.convolution(%v3203, %v3205)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3207 = stablehlo.reshape %v3206 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3208 = stablehlo.reshape %v3207 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3209 = stablehlo.reshape %v736 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3210 = stablehlo.multiply %v3209, %v3209 : tensor<32x1536x14x14xf32>
    %v3211 = stablehlo.multiply %v3210, %v3209 : tensor<32x1536x14x14xf32>
    %v3212 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v3213 = stablehlo.multiply %v3212, %v3211 : tensor<32x1536x14x14xf32>
    %v3214 = stablehlo.add %v3209, %v3213 : tensor<32x1536x14x14xf32>
    %v3215 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v3216 = stablehlo.multiply %v3215, %v3214 : tensor<32x1536x14x14xf32>
    %v3217 = stablehlo.tanh %v3216 : tensor<32x1536x14x14xf32>
    %v3218 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v3219 = stablehlo.add %v3218, %v3217 : tensor<32x1536x14x14xf32>
    %v3220 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v3221 = stablehlo.multiply %v3220, %v3219 : tensor<32x1536x14x14xf32>
    %v3222 = stablehlo.multiply %v3217, %v3217 : tensor<32x1536x14x14xf32>
    %v3223 = stablehlo.subtract %v3218, %v3222 : tensor<32x1536x14x14xf32>
    %v3224 = stablehlo.multiply %v3220, %v3209 : tensor<32x1536x14x14xf32>
    %v3225 = stablehlo.multiply %v3224, %v3223 : tensor<32x1536x14x14xf32>
    %v3226 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v3227 = stablehlo.multiply %v3226, %v3210 : tensor<32x1536x14x14xf32>
    %v3228 = stablehlo.add %v3218, %v3227 : tensor<32x1536x14x14xf32>
    %v3229 = stablehlo.multiply %v3215, %v3228 : tensor<32x1536x14x14xf32>
    %v3230 = stablehlo.multiply %v3225, %v3229 : tensor<32x1536x14x14xf32>
    %v3231 = stablehlo.add %v3221, %v3230 : tensor<32x1536x14x14xf32>
    %v3232 = stablehlo.multiply %v3208, %v3231 : tensor<32x1536x14x14xf32>
    %v3233 = stablehlo.reshape %v3232 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3234 = stablehlo.reshape %v3233 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3235 = stablehlo.transpose %s2b2eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3236 = stablehlo.reverse %v3235, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3237 = stablehlo.convolution(%v3234, %v3236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3238 = stablehlo.reshape %v3237 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3239 = stablehlo.reshape %v697 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3240 = stablehlo.transpose %v3239, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3241 = stablehlo.reshape %v3240 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3242 = stablehlo.reshape %v3238 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3243 = stablehlo.transpose %v3242, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3244 = stablehlo.reshape %v3243 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3245 = stablehlo.reshape %v3244 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3246 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3247 = stablehlo.multiply %v3245, %v3246 : tensor<32x196x384xf32>
    %v3248 = stablehlo.reshape %v3247 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3249 = stablehlo.reshape %v3248 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3250 = stablehlo.reshape %v3241 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3252 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3253 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3254 = stablehlo.reduce(%v3250 init: %v3251) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3255 = stablehlo.broadcast_in_dim %v3254, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3256 = stablehlo.divide %v3255, %v3252 : tensor<32x196x384xf32>
    %v3257 = stablehlo.subtract %v3250, %v3256 : tensor<32x196x384xf32>
    %v3258 = stablehlo.multiply %v3257, %v3257 : tensor<32x196x384xf32>
    %v3259 = stablehlo.reduce(%v3258 init: %v3251) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3260 = stablehlo.broadcast_in_dim %v3259, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3261 = stablehlo.divide %v3260, %v3252 : tensor<32x196x384xf32>
    %v3262 = stablehlo.add %v3261, %v3253 : tensor<32x196x384xf32>
    %v3263 = stablehlo.rsqrt %v3262 : tensor<32x196x384xf32>
    %v3264 = stablehlo.multiply %v3257, %v3263 : tensor<32x196x384xf32>
    %v3265 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3266 = stablehlo.multiply %v3265, %v3249 : tensor<32x196x384xf32>
    %v3267 = stablehlo.reduce(%v3266 init: %v3251) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3268 = stablehlo.broadcast_in_dim %v3267, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3269 = stablehlo.multiply %v3264, %v3266 : tensor<32x196x384xf32>
    %v3270 = stablehlo.reduce(%v3269 init: %v3251) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3271 = stablehlo.broadcast_in_dim %v3270, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3272 = stablehlo.multiply %v3266, %v3252 : tensor<32x196x384xf32>
    %v3273 = stablehlo.subtract %v3272, %v3268 : tensor<32x196x384xf32>
    %v3274 = stablehlo.multiply %v3264, %v3271 : tensor<32x196x384xf32>
    %v3275 = stablehlo.subtract %v3273, %v3274 : tensor<32x196x384xf32>
    %v3276 = stablehlo.divide %v3263, %v3252 : tensor<32x196x384xf32>
    %v3277 = stablehlo.multiply %v3276, %v3275 : tensor<32x196x384xf32>
    %v3278 = stablehlo.reshape %v3277 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3279 = stablehlo.reshape %v3278 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3280 = stablehlo.transpose %v3279, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3281 = stablehlo.reshape %v3280 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3282 = stablehlo.reshape %v3281 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3283 = stablehlo.reverse %s2b2dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3284 = stablehlo.convolution(%v3282, %v3283)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3285 = stablehlo.reshape %v3284 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3286 = stablehlo.reshape %v3285 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3287 = stablehlo.reshape %v3109 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3288 = stablehlo.add %v3286, %v3287 : tensor<32x384x14x14xf32>
    %v3289 = stablehlo.reshape %v3288 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3291 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3292 = stablehlo.reshape %v3109 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3293 = stablehlo.multiply %v3291, %v3292 : tensor<32x384x14x14xf32>
    %v3294 = stablehlo.reduce(%v3293 init: %v3290) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3295 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3296 = stablehlo.multiply %v3294, %v3295 : tensor<384xf32>
    %v3297 = stablehlo.subtract %s2b2lg, %v3296 : tensor<384xf32>
    %v3298 = stablehlo.reshape %v751 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3299 = stablehlo.reshape %v3202 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3300 = stablehlo.transpose %v3298, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3301 = stablehlo.transpose %v3299, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3302 = stablehlo.convolution(%v3300, %v3301)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3303 = stablehlo.transpose %v3302, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3304 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v3305 = stablehlo.multiply %v3303, %v3304 : tensor<384x1536x1x1xf32>
    %v3306 = stablehlo.subtract %s2b2pW, %v3305 : tensor<384x1536x1x1xf32>
    %v3307 = stablehlo.reshape %v3202 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3309 = stablehlo.reduce(%v3307 init: %v3308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3310 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3311 = stablehlo.multiply %v3309, %v3310 : tensor<384xf32>
    %v3312 = stablehlo.subtract %s2b2pb, %v3311 : tensor<384xf32>
    %v3313 = stablehlo.reshape %v731 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3314 = stablehlo.reshape %v3233 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3315 = stablehlo.transpose %v3313, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3316 = stablehlo.transpose %v3314, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3317 = stablehlo.convolution(%v3315, %v3316)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3318 = stablehlo.transpose %v3317, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3319 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v3320 = stablehlo.multiply %v3318, %v3319 : tensor<1536x384x1x1xf32>
    %v3321 = stablehlo.subtract %s2b2eW, %v3320 : tensor<1536x384x1x1xf32>
    %v3322 = stablehlo.reshape %v3233 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3324 = stablehlo.reduce(%v3322 init: %v3323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3325 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3326 = stablehlo.multiply %v3324, %v3325 : tensor<1536xf32>
    %v3327 = stablehlo.subtract %s2b2eb, %v3326 : tensor<1536xf32>
    %v3328 = stablehlo.reshape %v697 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3329 = stablehlo.transpose %v3328, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3330 = stablehlo.reshape %v3329 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3331 = stablehlo.reshape %v3238 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3332 = stablehlo.transpose %v3331, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3333 = stablehlo.reshape %v3332 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3334 = stablehlo.reshape %v3330 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3335 = stablehlo.reshape %v3333 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3337 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3338 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3339 = stablehlo.reduce(%v3334 init: %v3336) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3340 = stablehlo.broadcast_in_dim %v3339, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3341 = stablehlo.divide %v3340, %v3337 : tensor<32x196x384xf32>
    %v3342 = stablehlo.subtract %v3334, %v3341 : tensor<32x196x384xf32>
    %v3343 = stablehlo.multiply %v3342, %v3342 : tensor<32x196x384xf32>
    %v3344 = stablehlo.reduce(%v3343 init: %v3336) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3345 = stablehlo.broadcast_in_dim %v3344, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3346 = stablehlo.divide %v3345, %v3337 : tensor<32x196x384xf32>
    %v3347 = stablehlo.add %v3346, %v3338 : tensor<32x196x384xf32>
    %v3348 = stablehlo.rsqrt %v3347 : tensor<32x196x384xf32>
    %v3349 = stablehlo.multiply %v3342, %v3348 : tensor<32x196x384xf32>
    %v3350 = stablehlo.multiply %v3335, %v3349 : tensor<32x196x384xf32>
    %v3351 = stablehlo.reduce(%v3350 init: %v3336) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3352 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3353 = stablehlo.multiply %v3351, %v3352 : tensor<384xf32>
    %v3354 = stablehlo.subtract %s2b2ng, %v3353 : tensor<384xf32>
    %v3355 = stablehlo.reshape %v3238 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3356 = stablehlo.transpose %v3355, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3357 = stablehlo.reshape %v3356 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3359 = stablehlo.reshape %v3357 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3360 = stablehlo.reduce(%v3359 init: %v3358) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3361 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3362 = stablehlo.multiply %v3360, %v3361 : tensor<384xf32>
    %v3363 = stablehlo.subtract %s2b2nbt, %v3362 : tensor<384xf32>
    %v3364 = stablehlo.reshape %v692 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3365 = stablehlo.reshape %v3281 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3366 = stablehlo.transpose %v3364, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3367 = stablehlo.transpose %v3365, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3368 = stablehlo.convolution(%v3366, %v3367)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3369 = stablehlo.reshape %v3368 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3370 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3371 = stablehlo.multiply %v3369, %v3370 : tensor<384x1x7x7xf32>
    %v3372 = stablehlo.subtract %s2b2dW, %v3371 : tensor<384x1x7x7xf32>
    %v3373 = stablehlo.reshape %v3281 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3375 = stablehlo.reduce(%v3373 init: %v3374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3376 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3377 = stablehlo.multiply %v3375, %v3376 : tensor<384xf32>
    %v3378 = stablehlo.subtract %s2b2db, %v3377 : tensor<384xf32>
    %v3379 = stablehlo.reshape %v3289 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3380 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3381 = stablehlo.multiply %v3379, %v3380 : tensor<32x384x14x14xf32>
    %v3382 = stablehlo.reshape %v3381 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3383 = stablehlo.reshape %v3382 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3384 = stablehlo.transpose %s2b1pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3385 = stablehlo.reverse %v3384, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3386 = stablehlo.convolution(%v3383, %v3385)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3387 = stablehlo.reshape %v3386 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3388 = stablehlo.reshape %v3387 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3389 = stablehlo.reshape %v664 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3390 = stablehlo.multiply %v3389, %v3389 : tensor<32x1536x14x14xf32>
    %v3391 = stablehlo.multiply %v3390, %v3389 : tensor<32x1536x14x14xf32>
    %v3392 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v3393 = stablehlo.multiply %v3392, %v3391 : tensor<32x1536x14x14xf32>
    %v3394 = stablehlo.add %v3389, %v3393 : tensor<32x1536x14x14xf32>
    %v3395 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v3396 = stablehlo.multiply %v3395, %v3394 : tensor<32x1536x14x14xf32>
    %v3397 = stablehlo.tanh %v3396 : tensor<32x1536x14x14xf32>
    %v3398 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v3399 = stablehlo.add %v3398, %v3397 : tensor<32x1536x14x14xf32>
    %v3400 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v3401 = stablehlo.multiply %v3400, %v3399 : tensor<32x1536x14x14xf32>
    %v3402 = stablehlo.multiply %v3397, %v3397 : tensor<32x1536x14x14xf32>
    %v3403 = stablehlo.subtract %v3398, %v3402 : tensor<32x1536x14x14xf32>
    %v3404 = stablehlo.multiply %v3400, %v3389 : tensor<32x1536x14x14xf32>
    %v3405 = stablehlo.multiply %v3404, %v3403 : tensor<32x1536x14x14xf32>
    %v3406 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v3407 = stablehlo.multiply %v3406, %v3390 : tensor<32x1536x14x14xf32>
    %v3408 = stablehlo.add %v3398, %v3407 : tensor<32x1536x14x14xf32>
    %v3409 = stablehlo.multiply %v3395, %v3408 : tensor<32x1536x14x14xf32>
    %v3410 = stablehlo.multiply %v3405, %v3409 : tensor<32x1536x14x14xf32>
    %v3411 = stablehlo.add %v3401, %v3410 : tensor<32x1536x14x14xf32>
    %v3412 = stablehlo.multiply %v3388, %v3411 : tensor<32x1536x14x14xf32>
    %v3413 = stablehlo.reshape %v3412 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3414 = stablehlo.reshape %v3413 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3415 = stablehlo.transpose %s2b1eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3416 = stablehlo.reverse %v3415, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3417 = stablehlo.convolution(%v3414, %v3416)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3418 = stablehlo.reshape %v3417 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3419 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3420 = stablehlo.transpose %v3419, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3421 = stablehlo.reshape %v3420 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3422 = stablehlo.reshape %v3418 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3423 = stablehlo.transpose %v3422, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3424 = stablehlo.reshape %v3423 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3425 = stablehlo.reshape %v3424 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3426 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3427 = stablehlo.multiply %v3425, %v3426 : tensor<32x196x384xf32>
    %v3428 = stablehlo.reshape %v3427 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3429 = stablehlo.reshape %v3428 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3430 = stablehlo.reshape %v3421 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3432 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3433 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3434 = stablehlo.reduce(%v3430 init: %v3431) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3435 = stablehlo.broadcast_in_dim %v3434, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3436 = stablehlo.divide %v3435, %v3432 : tensor<32x196x384xf32>
    %v3437 = stablehlo.subtract %v3430, %v3436 : tensor<32x196x384xf32>
    %v3438 = stablehlo.multiply %v3437, %v3437 : tensor<32x196x384xf32>
    %v3439 = stablehlo.reduce(%v3438 init: %v3431) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3440 = stablehlo.broadcast_in_dim %v3439, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3441 = stablehlo.divide %v3440, %v3432 : tensor<32x196x384xf32>
    %v3442 = stablehlo.add %v3441, %v3433 : tensor<32x196x384xf32>
    %v3443 = stablehlo.rsqrt %v3442 : tensor<32x196x384xf32>
    %v3444 = stablehlo.multiply %v3437, %v3443 : tensor<32x196x384xf32>
    %v3445 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3446 = stablehlo.multiply %v3445, %v3429 : tensor<32x196x384xf32>
    %v3447 = stablehlo.reduce(%v3446 init: %v3431) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3448 = stablehlo.broadcast_in_dim %v3447, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3449 = stablehlo.multiply %v3444, %v3446 : tensor<32x196x384xf32>
    %v3450 = stablehlo.reduce(%v3449 init: %v3431) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3451 = stablehlo.broadcast_in_dim %v3450, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3452 = stablehlo.multiply %v3446, %v3432 : tensor<32x196x384xf32>
    %v3453 = stablehlo.subtract %v3452, %v3448 : tensor<32x196x384xf32>
    %v3454 = stablehlo.multiply %v3444, %v3451 : tensor<32x196x384xf32>
    %v3455 = stablehlo.subtract %v3453, %v3454 : tensor<32x196x384xf32>
    %v3456 = stablehlo.divide %v3443, %v3432 : tensor<32x196x384xf32>
    %v3457 = stablehlo.multiply %v3456, %v3455 : tensor<32x196x384xf32>
    %v3458 = stablehlo.reshape %v3457 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3459 = stablehlo.reshape %v3458 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3460 = stablehlo.transpose %v3459, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3461 = stablehlo.reshape %v3460 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3462 = stablehlo.reshape %v3461 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3463 = stablehlo.reverse %s2b1dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3464 = stablehlo.convolution(%v3462, %v3463)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3465 = stablehlo.reshape %v3464 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3466 = stablehlo.reshape %v3465 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3467 = stablehlo.reshape %v3289 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3468 = stablehlo.add %v3466, %v3467 : tensor<32x384x14x14xf32>
    %v3469 = stablehlo.reshape %v3468 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3471 = stablehlo.reshape %v684 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3472 = stablehlo.reshape %v3289 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3473 = stablehlo.multiply %v3471, %v3472 : tensor<32x384x14x14xf32>
    %v3474 = stablehlo.reduce(%v3473 init: %v3470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3475 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3476 = stablehlo.multiply %v3474, %v3475 : tensor<384xf32>
    %v3477 = stablehlo.subtract %s2b1lg, %v3476 : tensor<384xf32>
    %v3478 = stablehlo.reshape %v679 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3479 = stablehlo.reshape %v3382 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3480 = stablehlo.transpose %v3478, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3481 = stablehlo.transpose %v3479, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3482 = stablehlo.convolution(%v3480, %v3481)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3483 = stablehlo.transpose %v3482, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3484 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v3485 = stablehlo.multiply %v3483, %v3484 : tensor<384x1536x1x1xf32>
    %v3486 = stablehlo.subtract %s2b1pW, %v3485 : tensor<384x1536x1x1xf32>
    %v3487 = stablehlo.reshape %v3382 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3488 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3489 = stablehlo.reduce(%v3487 init: %v3488) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3490 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3491 = stablehlo.multiply %v3489, %v3490 : tensor<384xf32>
    %v3492 = stablehlo.subtract %s2b1pb, %v3491 : tensor<384xf32>
    %v3493 = stablehlo.reshape %v659 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3494 = stablehlo.reshape %v3413 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3495 = stablehlo.transpose %v3493, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3496 = stablehlo.transpose %v3494, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3497 = stablehlo.convolution(%v3495, %v3496)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3498 = stablehlo.transpose %v3497, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3499 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v3500 = stablehlo.multiply %v3498, %v3499 : tensor<1536x384x1x1xf32>
    %v3501 = stablehlo.subtract %s2b1eW, %v3500 : tensor<1536x384x1x1xf32>
    %v3502 = stablehlo.reshape %v3413 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3504 = stablehlo.reduce(%v3502 init: %v3503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3505 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3506 = stablehlo.multiply %v3504, %v3505 : tensor<1536xf32>
    %v3507 = stablehlo.subtract %s2b1eb, %v3506 : tensor<1536xf32>
    %v3508 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3509 = stablehlo.transpose %v3508, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3510 = stablehlo.reshape %v3509 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3511 = stablehlo.reshape %v3418 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3512 = stablehlo.transpose %v3511, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3513 = stablehlo.reshape %v3512 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3514 = stablehlo.reshape %v3510 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3515 = stablehlo.reshape %v3513 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3517 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3518 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3519 = stablehlo.reduce(%v3514 init: %v3516) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3520 = stablehlo.broadcast_in_dim %v3519, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3521 = stablehlo.divide %v3520, %v3517 : tensor<32x196x384xf32>
    %v3522 = stablehlo.subtract %v3514, %v3521 : tensor<32x196x384xf32>
    %v3523 = stablehlo.multiply %v3522, %v3522 : tensor<32x196x384xf32>
    %v3524 = stablehlo.reduce(%v3523 init: %v3516) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3525 = stablehlo.broadcast_in_dim %v3524, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3526 = stablehlo.divide %v3525, %v3517 : tensor<32x196x384xf32>
    %v3527 = stablehlo.add %v3526, %v3518 : tensor<32x196x384xf32>
    %v3528 = stablehlo.rsqrt %v3527 : tensor<32x196x384xf32>
    %v3529 = stablehlo.multiply %v3522, %v3528 : tensor<32x196x384xf32>
    %v3530 = stablehlo.multiply %v3515, %v3529 : tensor<32x196x384xf32>
    %v3531 = stablehlo.reduce(%v3530 init: %v3516) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3532 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3533 = stablehlo.multiply %v3531, %v3532 : tensor<384xf32>
    %v3534 = stablehlo.subtract %s2b1ng, %v3533 : tensor<384xf32>
    %v3535 = stablehlo.reshape %v3418 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3536 = stablehlo.transpose %v3535, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3537 = stablehlo.reshape %v3536 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3539 = stablehlo.reshape %v3537 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3540 = stablehlo.reduce(%v3539 init: %v3538) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3541 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3542 = stablehlo.multiply %v3540, %v3541 : tensor<384xf32>
    %v3543 = stablehlo.subtract %s2b1nbt, %v3542 : tensor<384xf32>
    %v3544 = stablehlo.reshape %v620 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3545 = stablehlo.reshape %v3461 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3546 = stablehlo.transpose %v3544, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3547 = stablehlo.transpose %v3545, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3548 = stablehlo.convolution(%v3546, %v3547)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3549 = stablehlo.reshape %v3548 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3550 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3551 = stablehlo.multiply %v3549, %v3550 : tensor<384x1x7x7xf32>
    %v3552 = stablehlo.subtract %s2b1dW, %v3551 : tensor<384x1x7x7xf32>
    %v3553 = stablehlo.reshape %v3461 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3555 = stablehlo.reduce(%v3553 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3556 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3557 = stablehlo.multiply %v3555, %v3556 : tensor<384xf32>
    %v3558 = stablehlo.subtract %s2b1db, %v3557 : tensor<384xf32>
    %v3559 = stablehlo.reshape %v3469 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3560 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3561 = stablehlo.multiply %v3559, %v3560 : tensor<32x384x14x14xf32>
    %v3562 = stablehlo.reshape %v3561 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3563 = stablehlo.reshape %v3562 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3564 = stablehlo.transpose %s2b0pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3565 = stablehlo.reverse %v3564, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3566 = stablehlo.convolution(%v3563, %v3565)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3567 = stablehlo.reshape %v3566 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3568 = stablehlo.reshape %v3567 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3569 = stablehlo.reshape %v592 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3570 = stablehlo.multiply %v3569, %v3569 : tensor<32x1536x14x14xf32>
    %v3571 = stablehlo.multiply %v3570, %v3569 : tensor<32x1536x14x14xf32>
    %v3572 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v3573 = stablehlo.multiply %v3572, %v3571 : tensor<32x1536x14x14xf32>
    %v3574 = stablehlo.add %v3569, %v3573 : tensor<32x1536x14x14xf32>
    %v3575 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v3576 = stablehlo.multiply %v3575, %v3574 : tensor<32x1536x14x14xf32>
    %v3577 = stablehlo.tanh %v3576 : tensor<32x1536x14x14xf32>
    %v3578 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v3579 = stablehlo.add %v3578, %v3577 : tensor<32x1536x14x14xf32>
    %v3580 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v3581 = stablehlo.multiply %v3580, %v3579 : tensor<32x1536x14x14xf32>
    %v3582 = stablehlo.multiply %v3577, %v3577 : tensor<32x1536x14x14xf32>
    %v3583 = stablehlo.subtract %v3578, %v3582 : tensor<32x1536x14x14xf32>
    %v3584 = stablehlo.multiply %v3580, %v3569 : tensor<32x1536x14x14xf32>
    %v3585 = stablehlo.multiply %v3584, %v3583 : tensor<32x1536x14x14xf32>
    %v3586 = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %v3587 = stablehlo.multiply %v3586, %v3570 : tensor<32x1536x14x14xf32>
    %v3588 = stablehlo.add %v3578, %v3587 : tensor<32x1536x14x14xf32>
    %v3589 = stablehlo.multiply %v3575, %v3588 : tensor<32x1536x14x14xf32>
    %v3590 = stablehlo.multiply %v3585, %v3589 : tensor<32x1536x14x14xf32>
    %v3591 = stablehlo.add %v3581, %v3590 : tensor<32x1536x14x14xf32>
    %v3592 = stablehlo.multiply %v3568, %v3591 : tensor<32x1536x14x14xf32>
    %v3593 = stablehlo.reshape %v3592 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3594 = stablehlo.reshape %v3593 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3595 = stablehlo.transpose %s2b0eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3596 = stablehlo.reverse %v3595, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3597 = stablehlo.convolution(%v3594, %v3596)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3598 = stablehlo.reshape %v3597 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3599 = stablehlo.reshape %v553 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3600 = stablehlo.transpose %v3599, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3601 = stablehlo.reshape %v3600 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3602 = stablehlo.reshape %v3598 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3603 = stablehlo.transpose %v3602, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3604 = stablehlo.reshape %v3603 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3605 = stablehlo.reshape %v3604 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3606 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3607 = stablehlo.multiply %v3605, %v3606 : tensor<32x196x384xf32>
    %v3608 = stablehlo.reshape %v3607 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3609 = stablehlo.reshape %v3608 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3610 = stablehlo.reshape %v3601 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3612 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3613 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3614 = stablehlo.reduce(%v3610 init: %v3611) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3615 = stablehlo.broadcast_in_dim %v3614, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3616 = stablehlo.divide %v3615, %v3612 : tensor<32x196x384xf32>
    %v3617 = stablehlo.subtract %v3610, %v3616 : tensor<32x196x384xf32>
    %v3618 = stablehlo.multiply %v3617, %v3617 : tensor<32x196x384xf32>
    %v3619 = stablehlo.reduce(%v3618 init: %v3611) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3620 = stablehlo.broadcast_in_dim %v3619, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3621 = stablehlo.divide %v3620, %v3612 : tensor<32x196x384xf32>
    %v3622 = stablehlo.add %v3621, %v3613 : tensor<32x196x384xf32>
    %v3623 = stablehlo.rsqrt %v3622 : tensor<32x196x384xf32>
    %v3624 = stablehlo.multiply %v3617, %v3623 : tensor<32x196x384xf32>
    %v3625 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3626 = stablehlo.multiply %v3625, %v3609 : tensor<32x196x384xf32>
    %v3627 = stablehlo.reduce(%v3626 init: %v3611) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3628 = stablehlo.broadcast_in_dim %v3627, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3629 = stablehlo.multiply %v3624, %v3626 : tensor<32x196x384xf32>
    %v3630 = stablehlo.reduce(%v3629 init: %v3611) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3631 = stablehlo.broadcast_in_dim %v3630, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3632 = stablehlo.multiply %v3626, %v3612 : tensor<32x196x384xf32>
    %v3633 = stablehlo.subtract %v3632, %v3628 : tensor<32x196x384xf32>
    %v3634 = stablehlo.multiply %v3624, %v3631 : tensor<32x196x384xf32>
    %v3635 = stablehlo.subtract %v3633, %v3634 : tensor<32x196x384xf32>
    %v3636 = stablehlo.divide %v3623, %v3612 : tensor<32x196x384xf32>
    %v3637 = stablehlo.multiply %v3636, %v3635 : tensor<32x196x384xf32>
    %v3638 = stablehlo.reshape %v3637 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3639 = stablehlo.reshape %v3638 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3640 = stablehlo.transpose %v3639, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3641 = stablehlo.reshape %v3640 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3642 = stablehlo.reshape %v3641 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3643 = stablehlo.reverse %s2b0dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3644 = stablehlo.convolution(%v3642, %v3643)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3645 = stablehlo.reshape %v3644 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3646 = stablehlo.reshape %v3645 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3647 = stablehlo.reshape %v3469 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3648 = stablehlo.add %v3646, %v3647 : tensor<32x384x14x14xf32>
    %v3649 = stablehlo.reshape %v3648 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3651 = stablehlo.reshape %v612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3652 = stablehlo.reshape %v3469 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3653 = stablehlo.multiply %v3651, %v3652 : tensor<32x384x14x14xf32>
    %v3654 = stablehlo.reduce(%v3653 init: %v3650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3655 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3656 = stablehlo.multiply %v3654, %v3655 : tensor<384xf32>
    %v3657 = stablehlo.subtract %s2b0lg, %v3656 : tensor<384xf32>
    %v3658 = stablehlo.reshape %v607 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3659 = stablehlo.reshape %v3562 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3660 = stablehlo.transpose %v3658, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3661 = stablehlo.transpose %v3659, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3662 = stablehlo.convolution(%v3660, %v3661)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3663 = stablehlo.transpose %v3662, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3664 = stablehlo.constant dense<0.1> : tensor<384x1536x1x1xf32>
    %v3665 = stablehlo.multiply %v3663, %v3664 : tensor<384x1536x1x1xf32>
    %v3666 = stablehlo.subtract %s2b0pW, %v3665 : tensor<384x1536x1x1xf32>
    %v3667 = stablehlo.reshape %v3562 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3668 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3669 = stablehlo.reduce(%v3667 init: %v3668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3670 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3671 = stablehlo.multiply %v3669, %v3670 : tensor<384xf32>
    %v3672 = stablehlo.subtract %s2b0pb, %v3671 : tensor<384xf32>
    %v3673 = stablehlo.reshape %v587 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3674 = stablehlo.reshape %v3593 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3675 = stablehlo.transpose %v3673, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3676 = stablehlo.transpose %v3674, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3677 = stablehlo.convolution(%v3675, %v3676)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3678 = stablehlo.transpose %v3677, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3679 = stablehlo.constant dense<0.1> : tensor<1536x384x1x1xf32>
    %v3680 = stablehlo.multiply %v3678, %v3679 : tensor<1536x384x1x1xf32>
    %v3681 = stablehlo.subtract %s2b0eW, %v3680 : tensor<1536x384x1x1xf32>
    %v3682 = stablehlo.reshape %v3593 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3683 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3684 = stablehlo.reduce(%v3682 init: %v3683) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3685 = stablehlo.constant dense<0.1> : tensor<1536xf32>
    %v3686 = stablehlo.multiply %v3684, %v3685 : tensor<1536xf32>
    %v3687 = stablehlo.subtract %s2b0eb, %v3686 : tensor<1536xf32>
    %v3688 = stablehlo.reshape %v553 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3689 = stablehlo.transpose %v3688, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3690 = stablehlo.reshape %v3689 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3691 = stablehlo.reshape %v3598 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3692 = stablehlo.transpose %v3691, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3693 = stablehlo.reshape %v3692 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3694 = stablehlo.reshape %v3690 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3695 = stablehlo.reshape %v3693 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3696 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3697 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3698 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3699 = stablehlo.reduce(%v3694 init: %v3696) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3700 = stablehlo.broadcast_in_dim %v3699, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3701 = stablehlo.divide %v3700, %v3697 : tensor<32x196x384xf32>
    %v3702 = stablehlo.subtract %v3694, %v3701 : tensor<32x196x384xf32>
    %v3703 = stablehlo.multiply %v3702, %v3702 : tensor<32x196x384xf32>
    %v3704 = stablehlo.reduce(%v3703 init: %v3696) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3705 = stablehlo.broadcast_in_dim %v3704, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3706 = stablehlo.divide %v3705, %v3697 : tensor<32x196x384xf32>
    %v3707 = stablehlo.add %v3706, %v3698 : tensor<32x196x384xf32>
    %v3708 = stablehlo.rsqrt %v3707 : tensor<32x196x384xf32>
    %v3709 = stablehlo.multiply %v3702, %v3708 : tensor<32x196x384xf32>
    %v3710 = stablehlo.multiply %v3695, %v3709 : tensor<32x196x384xf32>
    %v3711 = stablehlo.reduce(%v3710 init: %v3696) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3712 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3713 = stablehlo.multiply %v3711, %v3712 : tensor<384xf32>
    %v3714 = stablehlo.subtract %s2b0ng, %v3713 : tensor<384xf32>
    %v3715 = stablehlo.reshape %v3598 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3716 = stablehlo.transpose %v3715, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3717 = stablehlo.reshape %v3716 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3719 = stablehlo.reshape %v3717 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3720 = stablehlo.reduce(%v3719 init: %v3718) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3721 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3722 = stablehlo.multiply %v3720, %v3721 : tensor<384xf32>
    %v3723 = stablehlo.subtract %s2b0nbt, %v3722 : tensor<384xf32>
    %v3724 = stablehlo.reshape %v548 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3725 = stablehlo.reshape %v3641 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3726 = stablehlo.transpose %v3724, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3727 = stablehlo.transpose %v3725, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3728 = stablehlo.convolution(%v3726, %v3727)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3729 = stablehlo.reshape %v3728 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3730 = stablehlo.constant dense<0.1> : tensor<384x1x7x7xf32>
    %v3731 = stablehlo.multiply %v3729, %v3730 : tensor<384x1x7x7xf32>
    %v3732 = stablehlo.subtract %s2b0dW, %v3731 : tensor<384x1x7x7xf32>
    %v3733 = stablehlo.reshape %v3641 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3735 = stablehlo.reduce(%v3733 init: %v3734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3736 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3737 = stablehlo.multiply %v3735, %v3736 : tensor<384xf32>
    %v3738 = stablehlo.subtract %s2b0db, %v3737 : tensor<384xf32>
    %v3739 = stablehlo.reshape %v3649 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3740 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3741 = stablehlo.pad %v3739, %v3740, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x28x28xf32>
    %v3742 = stablehlo.transpose %d1W, dims = [1, 0, 2, 3] : (tensor<384x192x2x2xf32>) -> tensor<192x384x2x2xf32>
    %v3743 = stablehlo.reverse %v3742, dims = [2, 3] : tensor<192x384x2x2xf32>
    %v3744 = stablehlo.convolution(%v3741, %v3743)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x28x28xf32>, tensor<192x384x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v3745 = stablehlo.reshape %v3744 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3746 = stablehlo.reshape %v509 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3747 = stablehlo.transpose %v3746, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3748 = stablehlo.reshape %v3747 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3749 = stablehlo.reshape %v3745 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3750 = stablehlo.transpose %v3749, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3751 = stablehlo.reshape %v3750 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3752 = stablehlo.reshape %v3751 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3753 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v3754 = stablehlo.multiply %v3752, %v3753 : tensor<32x784x192xf32>
    %v3755 = stablehlo.reshape %v3754 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3756 = stablehlo.reshape %v3755 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3757 = stablehlo.reshape %v3748 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3759 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3760 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3761 = stablehlo.reduce(%v3757 init: %v3758) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3762 = stablehlo.broadcast_in_dim %v3761, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3763 = stablehlo.divide %v3762, %v3759 : tensor<32x784x192xf32>
    %v3764 = stablehlo.subtract %v3757, %v3763 : tensor<32x784x192xf32>
    %v3765 = stablehlo.multiply %v3764, %v3764 : tensor<32x784x192xf32>
    %v3766 = stablehlo.reduce(%v3765 init: %v3758) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3767 = stablehlo.broadcast_in_dim %v3766, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3768 = stablehlo.divide %v3767, %v3759 : tensor<32x784x192xf32>
    %v3769 = stablehlo.add %v3768, %v3760 : tensor<32x784x192xf32>
    %v3770 = stablehlo.rsqrt %v3769 : tensor<32x784x192xf32>
    %v3771 = stablehlo.multiply %v3764, %v3770 : tensor<32x784x192xf32>
    %v3772 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v3773 = stablehlo.multiply %v3772, %v3756 : tensor<32x784x192xf32>
    %v3774 = stablehlo.reduce(%v3773 init: %v3758) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3775 = stablehlo.broadcast_in_dim %v3774, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3776 = stablehlo.multiply %v3771, %v3773 : tensor<32x784x192xf32>
    %v3777 = stablehlo.reduce(%v3776 init: %v3758) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3778 = stablehlo.broadcast_in_dim %v3777, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3779 = stablehlo.multiply %v3773, %v3759 : tensor<32x784x192xf32>
    %v3780 = stablehlo.subtract %v3779, %v3775 : tensor<32x784x192xf32>
    %v3781 = stablehlo.multiply %v3771, %v3778 : tensor<32x784x192xf32>
    %v3782 = stablehlo.subtract %v3780, %v3781 : tensor<32x784x192xf32>
    %v3783 = stablehlo.divide %v3770, %v3759 : tensor<32x784x192xf32>
    %v3784 = stablehlo.multiply %v3783, %v3782 : tensor<32x784x192xf32>
    %v3785 = stablehlo.reshape %v3784 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3786 = stablehlo.reshape %v3785 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3787 = stablehlo.transpose %v3786, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v3788 = stablehlo.reshape %v3787 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v3789 = stablehlo.reshape %v3649 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3791 = stablehlo.reduce(%v3789 init: %v3790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3792 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v3793 = stablehlo.multiply %v3791, %v3792 : tensor<384xf32>
    %v3794 = stablehlo.subtract %d1b, %v3793 : tensor<384xf32>
    %v3795 = stablehlo.reshape %v509 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3796 = stablehlo.transpose %v3795, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3797 = stablehlo.reshape %v3796 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3798 = stablehlo.reshape %v3745 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3799 = stablehlo.transpose %v3798, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3800 = stablehlo.reshape %v3799 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3801 = stablehlo.reshape %v3797 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3802 = stablehlo.reshape %v3800 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3804 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3805 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3806 = stablehlo.reduce(%v3801 init: %v3803) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3807 = stablehlo.broadcast_in_dim %v3806, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3808 = stablehlo.divide %v3807, %v3804 : tensor<32x784x192xf32>
    %v3809 = stablehlo.subtract %v3801, %v3808 : tensor<32x784x192xf32>
    %v3810 = stablehlo.multiply %v3809, %v3809 : tensor<32x784x192xf32>
    %v3811 = stablehlo.reduce(%v3810 init: %v3803) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3812 = stablehlo.broadcast_in_dim %v3811, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3813 = stablehlo.divide %v3812, %v3804 : tensor<32x784x192xf32>
    %v3814 = stablehlo.add %v3813, %v3805 : tensor<32x784x192xf32>
    %v3815 = stablehlo.rsqrt %v3814 : tensor<32x784x192xf32>
    %v3816 = stablehlo.multiply %v3809, %v3815 : tensor<32x784x192xf32>
    %v3817 = stablehlo.multiply %v3802, %v3816 : tensor<32x784x192xf32>
    %v3818 = stablehlo.reduce(%v3817 init: %v3803) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3819 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3820 = stablehlo.multiply %v3818, %v3819 : tensor<192xf32>
    %v3821 = stablehlo.subtract %d1ng, %v3820 : tensor<192xf32>
    %v3822 = stablehlo.reshape %v3745 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3823 = stablehlo.transpose %v3822, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3824 = stablehlo.reshape %v3823 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3826 = stablehlo.reshape %v3824 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3827 = stablehlo.reduce(%v3826 init: %v3825) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3828 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3829 = stablehlo.multiply %v3827, %v3828 : tensor<192xf32>
    %v3830 = stablehlo.subtract %d1nbt, %v3829 : tensor<192xf32>
    %v3831 = stablehlo.reshape %v543 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3832 = stablehlo.reshape %v3649 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3834 = stablehlo.pad %v3832, %v3833, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x27x27xf32>
    %v3835 = stablehlo.transpose %v3831, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3836 = stablehlo.transpose %v3834, dims = [1, 0, 2, 3] : (tensor<32x384x27x27xf32>) -> tensor<384x32x27x27xf32>
    %v3837 = stablehlo.convolution(%v3835, %v3836)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<384x32x27x27xf32>) -> tensor<192x384x2x2xf32>
    %v3838 = stablehlo.transpose %v3837, dims = [1, 0, 2, 3] : (tensor<192x384x2x2xf32>) -> tensor<384x192x2x2xf32>
    %v3839 = stablehlo.constant dense<0.1> : tensor<384x192x2x2xf32>
    %v3840 = stablehlo.multiply %v3838, %v3839 : tensor<384x192x2x2xf32>
    %v3841 = stablehlo.subtract %d1W, %v3840 : tensor<384x192x2x2xf32>
    %v3842 = stablehlo.reshape %v3788 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3843 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3844 = stablehlo.multiply %v3842, %v3843 : tensor<32x192x28x28xf32>
    %v3845 = stablehlo.reshape %v3844 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3846 = stablehlo.reshape %v3845 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3847 = stablehlo.transpose %s1b2pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3848 = stablehlo.reverse %v3847, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v3849 = stablehlo.convolution(%v3846, %v3848)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v3850 = stablehlo.reshape %v3849 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3851 = stablehlo.reshape %v3850 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3852 = stablehlo.reshape %v481 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3853 = stablehlo.multiply %v3852, %v3852 : tensor<32x768x28x28xf32>
    %v3854 = stablehlo.multiply %v3853, %v3852 : tensor<32x768x28x28xf32>
    %v3855 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v3856 = stablehlo.multiply %v3855, %v3854 : tensor<32x768x28x28xf32>
    %v3857 = stablehlo.add %v3852, %v3856 : tensor<32x768x28x28xf32>
    %v3858 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v3859 = stablehlo.multiply %v3858, %v3857 : tensor<32x768x28x28xf32>
    %v3860 = stablehlo.tanh %v3859 : tensor<32x768x28x28xf32>
    %v3861 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v3862 = stablehlo.add %v3861, %v3860 : tensor<32x768x28x28xf32>
    %v3863 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v3864 = stablehlo.multiply %v3863, %v3862 : tensor<32x768x28x28xf32>
    %v3865 = stablehlo.multiply %v3860, %v3860 : tensor<32x768x28x28xf32>
    %v3866 = stablehlo.subtract %v3861, %v3865 : tensor<32x768x28x28xf32>
    %v3867 = stablehlo.multiply %v3863, %v3852 : tensor<32x768x28x28xf32>
    %v3868 = stablehlo.multiply %v3867, %v3866 : tensor<32x768x28x28xf32>
    %v3869 = stablehlo.constant dense<0.134145> : tensor<32x768x28x28xf32>
    %v3870 = stablehlo.multiply %v3869, %v3853 : tensor<32x768x28x28xf32>
    %v3871 = stablehlo.add %v3861, %v3870 : tensor<32x768x28x28xf32>
    %v3872 = stablehlo.multiply %v3858, %v3871 : tensor<32x768x28x28xf32>
    %v3873 = stablehlo.multiply %v3868, %v3872 : tensor<32x768x28x28xf32>
    %v3874 = stablehlo.add %v3864, %v3873 : tensor<32x768x28x28xf32>
    %v3875 = stablehlo.multiply %v3851, %v3874 : tensor<32x768x28x28xf32>
    %v3876 = stablehlo.reshape %v3875 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3877 = stablehlo.reshape %v3876 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3878 = stablehlo.transpose %s1b2eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3879 = stablehlo.reverse %v3878, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3880 = stablehlo.convolution(%v3877, %v3879)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3881 = stablehlo.reshape %v3880 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3882 = stablehlo.reshape %v442 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3883 = stablehlo.transpose %v3882, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3884 = stablehlo.reshape %v3883 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3885 = stablehlo.reshape %v3881 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3886 = stablehlo.transpose %v3885, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3887 = stablehlo.reshape %v3886 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3888 = stablehlo.reshape %v3887 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3889 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v3890 = stablehlo.multiply %v3888, %v3889 : tensor<32x784x192xf32>
    %v3891 = stablehlo.reshape %v3890 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3892 = stablehlo.reshape %v3891 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3893 = stablehlo.reshape %v3884 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3895 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3896 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3897 = stablehlo.reduce(%v3893 init: %v3894) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3898 = stablehlo.broadcast_in_dim %v3897, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3899 = stablehlo.divide %v3898, %v3895 : tensor<32x784x192xf32>
    %v3900 = stablehlo.subtract %v3893, %v3899 : tensor<32x784x192xf32>
    %v3901 = stablehlo.multiply %v3900, %v3900 : tensor<32x784x192xf32>
    %v3902 = stablehlo.reduce(%v3901 init: %v3894) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3903 = stablehlo.broadcast_in_dim %v3902, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3904 = stablehlo.divide %v3903, %v3895 : tensor<32x784x192xf32>
    %v3905 = stablehlo.add %v3904, %v3896 : tensor<32x784x192xf32>
    %v3906 = stablehlo.rsqrt %v3905 : tensor<32x784x192xf32>
    %v3907 = stablehlo.multiply %v3900, %v3906 : tensor<32x784x192xf32>
    %v3908 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v3909 = stablehlo.multiply %v3908, %v3892 : tensor<32x784x192xf32>
    %v3910 = stablehlo.reduce(%v3909 init: %v3894) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3911 = stablehlo.broadcast_in_dim %v3910, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3912 = stablehlo.multiply %v3907, %v3909 : tensor<32x784x192xf32>
    %v3913 = stablehlo.reduce(%v3912 init: %v3894) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3914 = stablehlo.broadcast_in_dim %v3913, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3915 = stablehlo.multiply %v3909, %v3895 : tensor<32x784x192xf32>
    %v3916 = stablehlo.subtract %v3915, %v3911 : tensor<32x784x192xf32>
    %v3917 = stablehlo.multiply %v3907, %v3914 : tensor<32x784x192xf32>
    %v3918 = stablehlo.subtract %v3916, %v3917 : tensor<32x784x192xf32>
    %v3919 = stablehlo.divide %v3906, %v3895 : tensor<32x784x192xf32>
    %v3920 = stablehlo.multiply %v3919, %v3918 : tensor<32x784x192xf32>
    %v3921 = stablehlo.reshape %v3920 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3922 = stablehlo.reshape %v3921 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3923 = stablehlo.transpose %v3922, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v3924 = stablehlo.reshape %v3923 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v3925 = stablehlo.reshape %v3924 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3926 = stablehlo.reverse %s1b2dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v3927 = stablehlo.convolution(%v3925, %v3926)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v3928 = stablehlo.reshape %v3927 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3929 = stablehlo.reshape %v3928 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3930 = stablehlo.reshape %v3788 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3931 = stablehlo.add %v3929, %v3930 : tensor<32x192x28x28xf32>
    %v3932 = stablehlo.reshape %v3931 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3934 = stablehlo.reshape %v501 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3935 = stablehlo.reshape %v3788 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3936 = stablehlo.multiply %v3934, %v3935 : tensor<32x192x28x28xf32>
    %v3937 = stablehlo.reduce(%v3936 init: %v3933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3938 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3939 = stablehlo.multiply %v3937, %v3938 : tensor<192xf32>
    %v3940 = stablehlo.subtract %s1b2lg, %v3939 : tensor<192xf32>
    %v3941 = stablehlo.reshape %v496 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3942 = stablehlo.reshape %v3845 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3943 = stablehlo.transpose %v3941, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3944 = stablehlo.transpose %v3942, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3945 = stablehlo.convolution(%v3943, %v3944)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v3946 = stablehlo.transpose %v3945, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3947 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v3948 = stablehlo.multiply %v3946, %v3947 : tensor<192x768x1x1xf32>
    %v3949 = stablehlo.subtract %s1b2pW, %v3948 : tensor<192x768x1x1xf32>
    %v3950 = stablehlo.reshape %v3845 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3952 = stablehlo.reduce(%v3950 init: %v3951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3953 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3954 = stablehlo.multiply %v3952, %v3953 : tensor<192xf32>
    %v3955 = stablehlo.subtract %s1b2pb, %v3954 : tensor<192xf32>
    %v3956 = stablehlo.reshape %v476 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3957 = stablehlo.reshape %v3876 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3958 = stablehlo.transpose %v3956, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3959 = stablehlo.transpose %v3957, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3960 = stablehlo.convolution(%v3958, %v3959)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v3961 = stablehlo.transpose %v3960, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3962 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v3963 = stablehlo.multiply %v3961, %v3962 : tensor<768x192x1x1xf32>
    %v3964 = stablehlo.subtract %s1b2eW, %v3963 : tensor<768x192x1x1xf32>
    %v3965 = stablehlo.reshape %v3876 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3967 = stablehlo.reduce(%v3965 init: %v3966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v3968 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v3969 = stablehlo.multiply %v3967, %v3968 : tensor<768xf32>
    %v3970 = stablehlo.subtract %s1b2eb, %v3969 : tensor<768xf32>
    %v3971 = stablehlo.reshape %v442 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3972 = stablehlo.transpose %v3971, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3973 = stablehlo.reshape %v3972 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3974 = stablehlo.reshape %v3881 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3975 = stablehlo.transpose %v3974, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3976 = stablehlo.reshape %v3975 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3977 = stablehlo.reshape %v3973 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3978 = stablehlo.reshape %v3976 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3980 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3981 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3982 = stablehlo.reduce(%v3977 init: %v3979) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3983 = stablehlo.broadcast_in_dim %v3982, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3984 = stablehlo.divide %v3983, %v3980 : tensor<32x784x192xf32>
    %v3985 = stablehlo.subtract %v3977, %v3984 : tensor<32x784x192xf32>
    %v3986 = stablehlo.multiply %v3985, %v3985 : tensor<32x784x192xf32>
    %v3987 = stablehlo.reduce(%v3986 init: %v3979) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3988 = stablehlo.broadcast_in_dim %v3987, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3989 = stablehlo.divide %v3988, %v3980 : tensor<32x784x192xf32>
    %v3990 = stablehlo.add %v3989, %v3981 : tensor<32x784x192xf32>
    %v3991 = stablehlo.rsqrt %v3990 : tensor<32x784x192xf32>
    %v3992 = stablehlo.multiply %v3985, %v3991 : tensor<32x784x192xf32>
    %v3993 = stablehlo.multiply %v3978, %v3992 : tensor<32x784x192xf32>
    %v3994 = stablehlo.reduce(%v3993 init: %v3979) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3995 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v3996 = stablehlo.multiply %v3994, %v3995 : tensor<192xf32>
    %v3997 = stablehlo.subtract %s1b2ng, %v3996 : tensor<192xf32>
    %v3998 = stablehlo.reshape %v3881 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3999 = stablehlo.transpose %v3998, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4000 = stablehlo.reshape %v3999 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4001 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4002 = stablehlo.reshape %v4000 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4003 = stablehlo.reduce(%v4002 init: %v4001) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4004 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4005 = stablehlo.multiply %v4003, %v4004 : tensor<192xf32>
    %v4006 = stablehlo.subtract %s1b2nbt, %v4005 : tensor<192xf32>
    %v4007 = stablehlo.reshape %v437 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4008 = stablehlo.reshape %v3924 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4009 = stablehlo.transpose %v4007, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4010 = stablehlo.transpose %v4008, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4011 = stablehlo.convolution(%v4009, %v4010)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v4012 = stablehlo.reshape %v4011 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v4013 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v4014 = stablehlo.multiply %v4012, %v4013 : tensor<192x1x7x7xf32>
    %v4015 = stablehlo.subtract %s1b2dW, %v4014 : tensor<192x1x7x7xf32>
    %v4016 = stablehlo.reshape %v3924 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4018 = stablehlo.reduce(%v4016 init: %v4017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4019 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4020 = stablehlo.multiply %v4018, %v4019 : tensor<192xf32>
    %v4021 = stablehlo.subtract %s1b2db, %v4020 : tensor<192xf32>
    %v4022 = stablehlo.reshape %v3932 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4023 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4024 = stablehlo.multiply %v4022, %v4023 : tensor<32x192x28x28xf32>
    %v4025 = stablehlo.reshape %v4024 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4026 = stablehlo.reshape %v4025 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4027 = stablehlo.transpose %s1b1pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v4028 = stablehlo.reverse %v4027, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v4029 = stablehlo.convolution(%v4026, %v4028)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v4030 = stablehlo.reshape %v4029 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v4031 = stablehlo.reshape %v4030 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4032 = stablehlo.reshape %v409 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4033 = stablehlo.multiply %v4032, %v4032 : tensor<32x768x28x28xf32>
    %v4034 = stablehlo.multiply %v4033, %v4032 : tensor<32x768x28x28xf32>
    %v4035 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v4036 = stablehlo.multiply %v4035, %v4034 : tensor<32x768x28x28xf32>
    %v4037 = stablehlo.add %v4032, %v4036 : tensor<32x768x28x28xf32>
    %v4038 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v4039 = stablehlo.multiply %v4038, %v4037 : tensor<32x768x28x28xf32>
    %v4040 = stablehlo.tanh %v4039 : tensor<32x768x28x28xf32>
    %v4041 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v4042 = stablehlo.add %v4041, %v4040 : tensor<32x768x28x28xf32>
    %v4043 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v4044 = stablehlo.multiply %v4043, %v4042 : tensor<32x768x28x28xf32>
    %v4045 = stablehlo.multiply %v4040, %v4040 : tensor<32x768x28x28xf32>
    %v4046 = stablehlo.subtract %v4041, %v4045 : tensor<32x768x28x28xf32>
    %v4047 = stablehlo.multiply %v4043, %v4032 : tensor<32x768x28x28xf32>
    %v4048 = stablehlo.multiply %v4047, %v4046 : tensor<32x768x28x28xf32>
    %v4049 = stablehlo.constant dense<0.134145> : tensor<32x768x28x28xf32>
    %v4050 = stablehlo.multiply %v4049, %v4033 : tensor<32x768x28x28xf32>
    %v4051 = stablehlo.add %v4041, %v4050 : tensor<32x768x28x28xf32>
    %v4052 = stablehlo.multiply %v4038, %v4051 : tensor<32x768x28x28xf32>
    %v4053 = stablehlo.multiply %v4048, %v4052 : tensor<32x768x28x28xf32>
    %v4054 = stablehlo.add %v4044, %v4053 : tensor<32x768x28x28xf32>
    %v4055 = stablehlo.multiply %v4031, %v4054 : tensor<32x768x28x28xf32>
    %v4056 = stablehlo.reshape %v4055 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v4057 = stablehlo.reshape %v4056 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4058 = stablehlo.transpose %s1b1eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v4059 = stablehlo.reverse %v4058, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v4060 = stablehlo.convolution(%v4057, %v4059)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4061 = stablehlo.reshape %v4060 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4062 = stablehlo.reshape %v370 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4063 = stablehlo.transpose %v4062, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4064 = stablehlo.reshape %v4063 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4065 = stablehlo.reshape %v4061 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4066 = stablehlo.transpose %v4065, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4067 = stablehlo.reshape %v4066 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4068 = stablehlo.reshape %v4067 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4069 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v4070 = stablehlo.multiply %v4068, %v4069 : tensor<32x784x192xf32>
    %v4071 = stablehlo.reshape %v4070 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4072 = stablehlo.reshape %v4071 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4073 = stablehlo.reshape %v4064 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4074 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4075 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v4076 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v4077 = stablehlo.reduce(%v4073 init: %v4074) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4078 = stablehlo.broadcast_in_dim %v4077, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4079 = stablehlo.divide %v4078, %v4075 : tensor<32x784x192xf32>
    %v4080 = stablehlo.subtract %v4073, %v4079 : tensor<32x784x192xf32>
    %v4081 = stablehlo.multiply %v4080, %v4080 : tensor<32x784x192xf32>
    %v4082 = stablehlo.reduce(%v4081 init: %v4074) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4083 = stablehlo.broadcast_in_dim %v4082, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4084 = stablehlo.divide %v4083, %v4075 : tensor<32x784x192xf32>
    %v4085 = stablehlo.add %v4084, %v4076 : tensor<32x784x192xf32>
    %v4086 = stablehlo.rsqrt %v4085 : tensor<32x784x192xf32>
    %v4087 = stablehlo.multiply %v4080, %v4086 : tensor<32x784x192xf32>
    %v4088 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v4089 = stablehlo.multiply %v4088, %v4072 : tensor<32x784x192xf32>
    %v4090 = stablehlo.reduce(%v4089 init: %v4074) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4091 = stablehlo.broadcast_in_dim %v4090, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4092 = stablehlo.multiply %v4087, %v4089 : tensor<32x784x192xf32>
    %v4093 = stablehlo.reduce(%v4092 init: %v4074) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4094 = stablehlo.broadcast_in_dim %v4093, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4095 = stablehlo.multiply %v4089, %v4075 : tensor<32x784x192xf32>
    %v4096 = stablehlo.subtract %v4095, %v4091 : tensor<32x784x192xf32>
    %v4097 = stablehlo.multiply %v4087, %v4094 : tensor<32x784x192xf32>
    %v4098 = stablehlo.subtract %v4096, %v4097 : tensor<32x784x192xf32>
    %v4099 = stablehlo.divide %v4086, %v4075 : tensor<32x784x192xf32>
    %v4100 = stablehlo.multiply %v4099, %v4098 : tensor<32x784x192xf32>
    %v4101 = stablehlo.reshape %v4100 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4102 = stablehlo.reshape %v4101 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4103 = stablehlo.transpose %v4102, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v4104 = stablehlo.reshape %v4103 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v4105 = stablehlo.reshape %v4104 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4106 = stablehlo.reverse %s1b1dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v4107 = stablehlo.convolution(%v4105, %v4106)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v4108 = stablehlo.reshape %v4107 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4109 = stablehlo.reshape %v4108 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4110 = stablehlo.reshape %v3932 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4111 = stablehlo.add %v4109, %v4110 : tensor<32x192x28x28xf32>
    %v4112 = stablehlo.reshape %v4111 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4114 = stablehlo.reshape %v429 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4115 = stablehlo.reshape %v3932 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4116 = stablehlo.multiply %v4114, %v4115 : tensor<32x192x28x28xf32>
    %v4117 = stablehlo.reduce(%v4116 init: %v4113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4118 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4119 = stablehlo.multiply %v4117, %v4118 : tensor<192xf32>
    %v4120 = stablehlo.subtract %s1b1lg, %v4119 : tensor<192xf32>
    %v4121 = stablehlo.reshape %v424 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4122 = stablehlo.reshape %v4025 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4123 = stablehlo.transpose %v4121, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v4124 = stablehlo.transpose %v4122, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4125 = stablehlo.convolution(%v4123, %v4124)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v4126 = stablehlo.transpose %v4125, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v4127 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v4128 = stablehlo.multiply %v4126, %v4127 : tensor<192x768x1x1xf32>
    %v4129 = stablehlo.subtract %s1b1pW, %v4128 : tensor<192x768x1x1xf32>
    %v4130 = stablehlo.reshape %v4025 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4132 = stablehlo.reduce(%v4130 init: %v4131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4133 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4134 = stablehlo.multiply %v4132, %v4133 : tensor<192xf32>
    %v4135 = stablehlo.subtract %s1b1pb, %v4134 : tensor<192xf32>
    %v4136 = stablehlo.reshape %v404 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4137 = stablehlo.reshape %v4056 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4138 = stablehlo.transpose %v4136, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4139 = stablehlo.transpose %v4137, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v4140 = stablehlo.convolution(%v4138, %v4139)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v4141 = stablehlo.transpose %v4140, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v4142 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v4143 = stablehlo.multiply %v4141, %v4142 : tensor<768x192x1x1xf32>
    %v4144 = stablehlo.subtract %s1b1eW, %v4143 : tensor<768x192x1x1xf32>
    %v4145 = stablehlo.reshape %v4056 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4147 = stablehlo.reduce(%v4145 init: %v4146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v4148 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v4149 = stablehlo.multiply %v4147, %v4148 : tensor<768xf32>
    %v4150 = stablehlo.subtract %s1b1eb, %v4149 : tensor<768xf32>
    %v4151 = stablehlo.reshape %v370 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4152 = stablehlo.transpose %v4151, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4153 = stablehlo.reshape %v4152 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4154 = stablehlo.reshape %v4061 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
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
    %v4177 = stablehlo.subtract %s1b1ng, %v4176 : tensor<192xf32>
    %v4178 = stablehlo.reshape %v4061 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4179 = stablehlo.transpose %v4178, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4180 = stablehlo.reshape %v4179 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4182 = stablehlo.reshape %v4180 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4183 = stablehlo.reduce(%v4182 init: %v4181) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4184 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4185 = stablehlo.multiply %v4183, %v4184 : tensor<192xf32>
    %v4186 = stablehlo.subtract %s1b1nbt, %v4185 : tensor<192xf32>
    %v4187 = stablehlo.reshape %v365 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4188 = stablehlo.reshape %v4104 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4189 = stablehlo.transpose %v4187, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4190 = stablehlo.transpose %v4188, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4191 = stablehlo.convolution(%v4189, %v4190)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v4192 = stablehlo.reshape %v4191 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v4193 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v4194 = stablehlo.multiply %v4192, %v4193 : tensor<192x1x7x7xf32>
    %v4195 = stablehlo.subtract %s1b1dW, %v4194 : tensor<192x1x7x7xf32>
    %v4196 = stablehlo.reshape %v4104 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4198 = stablehlo.reduce(%v4196 init: %v4197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4199 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4200 = stablehlo.multiply %v4198, %v4199 : tensor<192xf32>
    %v4201 = stablehlo.subtract %s1b1db, %v4200 : tensor<192xf32>
    %v4202 = stablehlo.reshape %v4112 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4203 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4204 = stablehlo.multiply %v4202, %v4203 : tensor<32x192x28x28xf32>
    %v4205 = stablehlo.reshape %v4204 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4206 = stablehlo.reshape %v4205 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4207 = stablehlo.transpose %s1b0pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v4208 = stablehlo.reverse %v4207, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v4209 = stablehlo.convolution(%v4206, %v4208)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v4210 = stablehlo.reshape %v4209 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v4211 = stablehlo.reshape %v4210 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4212 = stablehlo.reshape %v337 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4213 = stablehlo.multiply %v4212, %v4212 : tensor<32x768x28x28xf32>
    %v4214 = stablehlo.multiply %v4213, %v4212 : tensor<32x768x28x28xf32>
    %v4215 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v4216 = stablehlo.multiply %v4215, %v4214 : tensor<32x768x28x28xf32>
    %v4217 = stablehlo.add %v4212, %v4216 : tensor<32x768x28x28xf32>
    %v4218 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v4219 = stablehlo.multiply %v4218, %v4217 : tensor<32x768x28x28xf32>
    %v4220 = stablehlo.tanh %v4219 : tensor<32x768x28x28xf32>
    %v4221 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v4222 = stablehlo.add %v4221, %v4220 : tensor<32x768x28x28xf32>
    %v4223 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v4224 = stablehlo.multiply %v4223, %v4222 : tensor<32x768x28x28xf32>
    %v4225 = stablehlo.multiply %v4220, %v4220 : tensor<32x768x28x28xf32>
    %v4226 = stablehlo.subtract %v4221, %v4225 : tensor<32x768x28x28xf32>
    %v4227 = stablehlo.multiply %v4223, %v4212 : tensor<32x768x28x28xf32>
    %v4228 = stablehlo.multiply %v4227, %v4226 : tensor<32x768x28x28xf32>
    %v4229 = stablehlo.constant dense<0.134145> : tensor<32x768x28x28xf32>
    %v4230 = stablehlo.multiply %v4229, %v4213 : tensor<32x768x28x28xf32>
    %v4231 = stablehlo.add %v4221, %v4230 : tensor<32x768x28x28xf32>
    %v4232 = stablehlo.multiply %v4218, %v4231 : tensor<32x768x28x28xf32>
    %v4233 = stablehlo.multiply %v4228, %v4232 : tensor<32x768x28x28xf32>
    %v4234 = stablehlo.add %v4224, %v4233 : tensor<32x768x28x28xf32>
    %v4235 = stablehlo.multiply %v4211, %v4234 : tensor<32x768x28x28xf32>
    %v4236 = stablehlo.reshape %v4235 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v4237 = stablehlo.reshape %v4236 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4238 = stablehlo.transpose %s1b0eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v4239 = stablehlo.reverse %v4238, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v4240 = stablehlo.convolution(%v4237, %v4239)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4241 = stablehlo.reshape %v4240 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4242 = stablehlo.reshape %v298 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4243 = stablehlo.transpose %v4242, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4244 = stablehlo.reshape %v4243 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4245 = stablehlo.reshape %v4241 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4246 = stablehlo.transpose %v4245, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4247 = stablehlo.reshape %v4246 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4248 = stablehlo.reshape %v4247 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4249 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v4250 = stablehlo.multiply %v4248, %v4249 : tensor<32x784x192xf32>
    %v4251 = stablehlo.reshape %v4250 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4252 = stablehlo.reshape %v4251 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4253 = stablehlo.reshape %v4244 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4255 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v4256 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v4257 = stablehlo.reduce(%v4253 init: %v4254) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4258 = stablehlo.broadcast_in_dim %v4257, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4259 = stablehlo.divide %v4258, %v4255 : tensor<32x784x192xf32>
    %v4260 = stablehlo.subtract %v4253, %v4259 : tensor<32x784x192xf32>
    %v4261 = stablehlo.multiply %v4260, %v4260 : tensor<32x784x192xf32>
    %v4262 = stablehlo.reduce(%v4261 init: %v4254) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4263 = stablehlo.broadcast_in_dim %v4262, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4264 = stablehlo.divide %v4263, %v4255 : tensor<32x784x192xf32>
    %v4265 = stablehlo.add %v4264, %v4256 : tensor<32x784x192xf32>
    %v4266 = stablehlo.rsqrt %v4265 : tensor<32x784x192xf32>
    %v4267 = stablehlo.multiply %v4260, %v4266 : tensor<32x784x192xf32>
    %v4268 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v4269 = stablehlo.multiply %v4268, %v4252 : tensor<32x784x192xf32>
    %v4270 = stablehlo.reduce(%v4269 init: %v4254) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4271 = stablehlo.broadcast_in_dim %v4270, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4272 = stablehlo.multiply %v4267, %v4269 : tensor<32x784x192xf32>
    %v4273 = stablehlo.reduce(%v4272 init: %v4254) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4274 = stablehlo.broadcast_in_dim %v4273, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4275 = stablehlo.multiply %v4269, %v4255 : tensor<32x784x192xf32>
    %v4276 = stablehlo.subtract %v4275, %v4271 : tensor<32x784x192xf32>
    %v4277 = stablehlo.multiply %v4267, %v4274 : tensor<32x784x192xf32>
    %v4278 = stablehlo.subtract %v4276, %v4277 : tensor<32x784x192xf32>
    %v4279 = stablehlo.divide %v4266, %v4255 : tensor<32x784x192xf32>
    %v4280 = stablehlo.multiply %v4279, %v4278 : tensor<32x784x192xf32>
    %v4281 = stablehlo.reshape %v4280 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4282 = stablehlo.reshape %v4281 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4283 = stablehlo.transpose %v4282, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v4284 = stablehlo.reshape %v4283 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v4285 = stablehlo.reshape %v4284 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4286 = stablehlo.reverse %s1b0dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v4287 = stablehlo.convolution(%v4285, %v4286)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v4288 = stablehlo.reshape %v4287 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4289 = stablehlo.reshape %v4288 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4290 = stablehlo.reshape %v4112 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4291 = stablehlo.add %v4289, %v4290 : tensor<32x192x28x28xf32>
    %v4292 = stablehlo.reshape %v4291 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4294 = stablehlo.reshape %v357 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4295 = stablehlo.reshape %v4112 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4296 = stablehlo.multiply %v4294, %v4295 : tensor<32x192x28x28xf32>
    %v4297 = stablehlo.reduce(%v4296 init: %v4293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4298 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4299 = stablehlo.multiply %v4297, %v4298 : tensor<192xf32>
    %v4300 = stablehlo.subtract %s1b0lg, %v4299 : tensor<192xf32>
    %v4301 = stablehlo.reshape %v352 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4302 = stablehlo.reshape %v4205 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4303 = stablehlo.transpose %v4301, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v4304 = stablehlo.transpose %v4302, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4305 = stablehlo.convolution(%v4303, %v4304)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v4306 = stablehlo.transpose %v4305, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v4307 = stablehlo.constant dense<0.1> : tensor<192x768x1x1xf32>
    %v4308 = stablehlo.multiply %v4306, %v4307 : tensor<192x768x1x1xf32>
    %v4309 = stablehlo.subtract %s1b0pW, %v4308 : tensor<192x768x1x1xf32>
    %v4310 = stablehlo.reshape %v4205 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4311 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4312 = stablehlo.reduce(%v4310 init: %v4311) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4313 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4314 = stablehlo.multiply %v4312, %v4313 : tensor<192xf32>
    %v4315 = stablehlo.subtract %s1b0pb, %v4314 : tensor<192xf32>
    %v4316 = stablehlo.reshape %v332 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4317 = stablehlo.reshape %v4236 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4318 = stablehlo.transpose %v4316, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4319 = stablehlo.transpose %v4317, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v4320 = stablehlo.convolution(%v4318, %v4319)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v4321 = stablehlo.transpose %v4320, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v4322 = stablehlo.constant dense<0.1> : tensor<768x192x1x1xf32>
    %v4323 = stablehlo.multiply %v4321, %v4322 : tensor<768x192x1x1xf32>
    %v4324 = stablehlo.subtract %s1b0eW, %v4323 : tensor<768x192x1x1xf32>
    %v4325 = stablehlo.reshape %v4236 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v4326 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4327 = stablehlo.reduce(%v4325 init: %v4326) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v4328 = stablehlo.constant dense<0.1> : tensor<768xf32>
    %v4329 = stablehlo.multiply %v4327, %v4328 : tensor<768xf32>
    %v4330 = stablehlo.subtract %s1b0eb, %v4329 : tensor<768xf32>
    %v4331 = stablehlo.reshape %v298 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4332 = stablehlo.transpose %v4331, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4333 = stablehlo.reshape %v4332 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4334 = stablehlo.reshape %v4241 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4335 = stablehlo.transpose %v4334, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4336 = stablehlo.reshape %v4335 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4337 = stablehlo.reshape %v4333 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4338 = stablehlo.reshape %v4336 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4340 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v4341 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v4342 = stablehlo.reduce(%v4337 init: %v4339) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4343 = stablehlo.broadcast_in_dim %v4342, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4344 = stablehlo.divide %v4343, %v4340 : tensor<32x784x192xf32>
    %v4345 = stablehlo.subtract %v4337, %v4344 : tensor<32x784x192xf32>
    %v4346 = stablehlo.multiply %v4345, %v4345 : tensor<32x784x192xf32>
    %v4347 = stablehlo.reduce(%v4346 init: %v4339) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v4348 = stablehlo.broadcast_in_dim %v4347, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v4349 = stablehlo.divide %v4348, %v4340 : tensor<32x784x192xf32>
    %v4350 = stablehlo.add %v4349, %v4341 : tensor<32x784x192xf32>
    %v4351 = stablehlo.rsqrt %v4350 : tensor<32x784x192xf32>
    %v4352 = stablehlo.multiply %v4345, %v4351 : tensor<32x784x192xf32>
    %v4353 = stablehlo.multiply %v4338, %v4352 : tensor<32x784x192xf32>
    %v4354 = stablehlo.reduce(%v4353 init: %v4339) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4355 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4356 = stablehlo.multiply %v4354, %v4355 : tensor<192xf32>
    %v4357 = stablehlo.subtract %s1b0ng, %v4356 : tensor<192xf32>
    %v4358 = stablehlo.reshape %v4241 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v4359 = stablehlo.transpose %v4358, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v4360 = stablehlo.reshape %v4359 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v4361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4362 = stablehlo.reshape %v4360 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v4363 = stablehlo.reduce(%v4362 init: %v4361) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4364 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4365 = stablehlo.multiply %v4363, %v4364 : tensor<192xf32>
    %v4366 = stablehlo.subtract %s1b0nbt, %v4365 : tensor<192xf32>
    %v4367 = stablehlo.reshape %v293 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4368 = stablehlo.reshape %v4284 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4369 = stablehlo.transpose %v4367, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4370 = stablehlo.transpose %v4368, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4371 = stablehlo.convolution(%v4369, %v4370)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v4372 = stablehlo.reshape %v4371 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v4373 = stablehlo.constant dense<0.1> : tensor<192x1x7x7xf32>
    %v4374 = stablehlo.multiply %v4372, %v4373 : tensor<192x1x7x7xf32>
    %v4375 = stablehlo.subtract %s1b0dW, %v4374 : tensor<192x1x7x7xf32>
    %v4376 = stablehlo.reshape %v4284 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4378 = stablehlo.reduce(%v4376 init: %v4377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4379 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4380 = stablehlo.multiply %v4378, %v4379 : tensor<192xf32>
    %v4381 = stablehlo.subtract %s1b0db, %v4380 : tensor<192xf32>
    %v4382 = stablehlo.reshape %v4292 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4384 = stablehlo.pad %v4382, %v4383, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x56x56xf32>
    %v4385 = stablehlo.transpose %d0W, dims = [1, 0, 2, 3] : (tensor<192x96x2x2xf32>) -> tensor<96x192x2x2xf32>
    %v4386 = stablehlo.reverse %v4385, dims = [2, 3] : tensor<96x192x2x2xf32>
    %v4387 = stablehlo.convolution(%v4384, %v4386)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x56x56xf32>, tensor<96x192x2x2xf32>) -> tensor<32x96x56x56xf32>
    %v4388 = stablehlo.reshape %v4387 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4389 = stablehlo.reshape %v254 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4390 = stablehlo.transpose %v4389, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4391 = stablehlo.reshape %v4390 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4392 = stablehlo.reshape %v4388 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4393 = stablehlo.transpose %v4392, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4394 = stablehlo.reshape %v4393 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4395 = stablehlo.reshape %v4394 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4396 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4397 = stablehlo.multiply %v4395, %v4396 : tensor<32x3136x96xf32>
    %v4398 = stablehlo.reshape %v4397 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4399 = stablehlo.reshape %v4398 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4400 = stablehlo.reshape %v4391 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4402 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4403 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4404 = stablehlo.reduce(%v4400 init: %v4401) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4405 = stablehlo.broadcast_in_dim %v4404, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4406 = stablehlo.divide %v4405, %v4402 : tensor<32x3136x96xf32>
    %v4407 = stablehlo.subtract %v4400, %v4406 : tensor<32x3136x96xf32>
    %v4408 = stablehlo.multiply %v4407, %v4407 : tensor<32x3136x96xf32>
    %v4409 = stablehlo.reduce(%v4408 init: %v4401) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4410 = stablehlo.broadcast_in_dim %v4409, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4411 = stablehlo.divide %v4410, %v4402 : tensor<32x3136x96xf32>
    %v4412 = stablehlo.add %v4411, %v4403 : tensor<32x3136x96xf32>
    %v4413 = stablehlo.rsqrt %v4412 : tensor<32x3136x96xf32>
    %v4414 = stablehlo.multiply %v4407, %v4413 : tensor<32x3136x96xf32>
    %v4415 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4416 = stablehlo.multiply %v4415, %v4399 : tensor<32x3136x96xf32>
    %v4417 = stablehlo.reduce(%v4416 init: %v4401) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4418 = stablehlo.broadcast_in_dim %v4417, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4419 = stablehlo.multiply %v4414, %v4416 : tensor<32x3136x96xf32>
    %v4420 = stablehlo.reduce(%v4419 init: %v4401) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4421 = stablehlo.broadcast_in_dim %v4420, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4422 = stablehlo.multiply %v4416, %v4402 : tensor<32x3136x96xf32>
    %v4423 = stablehlo.subtract %v4422, %v4418 : tensor<32x3136x96xf32>
    %v4424 = stablehlo.multiply %v4414, %v4421 : tensor<32x3136x96xf32>
    %v4425 = stablehlo.subtract %v4423, %v4424 : tensor<32x3136x96xf32>
    %v4426 = stablehlo.divide %v4413, %v4402 : tensor<32x3136x96xf32>
    %v4427 = stablehlo.multiply %v4426, %v4425 : tensor<32x3136x96xf32>
    %v4428 = stablehlo.reshape %v4427 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4429 = stablehlo.reshape %v4428 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4430 = stablehlo.transpose %v4429, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4431 = stablehlo.reshape %v4430 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4432 = stablehlo.reshape %v4292 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4433 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4434 = stablehlo.reduce(%v4432 init: %v4433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4435 = stablehlo.constant dense<0.1> : tensor<192xf32>
    %v4436 = stablehlo.multiply %v4434, %v4435 : tensor<192xf32>
    %v4437 = stablehlo.subtract %d0b, %v4436 : tensor<192xf32>
    %v4438 = stablehlo.reshape %v254 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4439 = stablehlo.transpose %v4438, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4440 = stablehlo.reshape %v4439 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4441 = stablehlo.reshape %v4388 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4442 = stablehlo.transpose %v4441, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4443 = stablehlo.reshape %v4442 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4444 = stablehlo.reshape %v4440 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4445 = stablehlo.reshape %v4443 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4446 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4447 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4448 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4449 = stablehlo.reduce(%v4444 init: %v4446) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4450 = stablehlo.broadcast_in_dim %v4449, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4451 = stablehlo.divide %v4450, %v4447 : tensor<32x3136x96xf32>
    %v4452 = stablehlo.subtract %v4444, %v4451 : tensor<32x3136x96xf32>
    %v4453 = stablehlo.multiply %v4452, %v4452 : tensor<32x3136x96xf32>
    %v4454 = stablehlo.reduce(%v4453 init: %v4446) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4455 = stablehlo.broadcast_in_dim %v4454, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4456 = stablehlo.divide %v4455, %v4447 : tensor<32x3136x96xf32>
    %v4457 = stablehlo.add %v4456, %v4448 : tensor<32x3136x96xf32>
    %v4458 = stablehlo.rsqrt %v4457 : tensor<32x3136x96xf32>
    %v4459 = stablehlo.multiply %v4452, %v4458 : tensor<32x3136x96xf32>
    %v4460 = stablehlo.multiply %v4445, %v4459 : tensor<32x3136x96xf32>
    %v4461 = stablehlo.reduce(%v4460 init: %v4446) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4462 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4463 = stablehlo.multiply %v4461, %v4462 : tensor<96xf32>
    %v4464 = stablehlo.subtract %d0ng, %v4463 : tensor<96xf32>
    %v4465 = stablehlo.reshape %v4388 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4466 = stablehlo.transpose %v4465, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4467 = stablehlo.reshape %v4466 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4469 = stablehlo.reshape %v4467 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4470 = stablehlo.reduce(%v4469 init: %v4468) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4471 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4472 = stablehlo.multiply %v4470, %v4471 : tensor<96xf32>
    %v4473 = stablehlo.subtract %d0nbt, %v4472 : tensor<96xf32>
    %v4474 = stablehlo.reshape %v288 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4475 = stablehlo.reshape %v4292 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4476 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4477 = stablehlo.pad %v4475, %v4476, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x55x55xf32>
    %v4478 = stablehlo.transpose %v4474, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4479 = stablehlo.transpose %v4477, dims = [1, 0, 2, 3] : (tensor<32x192x55x55xf32>) -> tensor<192x32x55x55xf32>
    %v4480 = stablehlo.convolution(%v4478, %v4479)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<192x32x55x55xf32>) -> tensor<96x192x2x2xf32>
    %v4481 = stablehlo.transpose %v4480, dims = [1, 0, 2, 3] : (tensor<96x192x2x2xf32>) -> tensor<192x96x2x2xf32>
    %v4482 = stablehlo.constant dense<0.1> : tensor<192x96x2x2xf32>
    %v4483 = stablehlo.multiply %v4481, %v4482 : tensor<192x96x2x2xf32>
    %v4484 = stablehlo.subtract %d0W, %v4483 : tensor<192x96x2x2xf32>
    %v4485 = stablehlo.reshape %v4431 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4486 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4487 = stablehlo.multiply %v4485, %v4486 : tensor<32x96x56x56xf32>
    %v4488 = stablehlo.reshape %v4487 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4489 = stablehlo.reshape %v4488 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4490 = stablehlo.transpose %s0b2pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4491 = stablehlo.reverse %v4490, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v4492 = stablehlo.convolution(%v4489, %v4491)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v4493 = stablehlo.reshape %v4492 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4494 = stablehlo.reshape %v4493 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4495 = stablehlo.reshape %v226 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4496 = stablehlo.multiply %v4495, %v4495 : tensor<32x384x56x56xf32>
    %v4497 = stablehlo.multiply %v4496, %v4495 : tensor<32x384x56x56xf32>
    %v4498 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v4499 = stablehlo.multiply %v4498, %v4497 : tensor<32x384x56x56xf32>
    %v4500 = stablehlo.add %v4495, %v4499 : tensor<32x384x56x56xf32>
    %v4501 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v4502 = stablehlo.multiply %v4501, %v4500 : tensor<32x384x56x56xf32>
    %v4503 = stablehlo.tanh %v4502 : tensor<32x384x56x56xf32>
    %v4504 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v4505 = stablehlo.add %v4504, %v4503 : tensor<32x384x56x56xf32>
    %v4506 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v4507 = stablehlo.multiply %v4506, %v4505 : tensor<32x384x56x56xf32>
    %v4508 = stablehlo.multiply %v4503, %v4503 : tensor<32x384x56x56xf32>
    %v4509 = stablehlo.subtract %v4504, %v4508 : tensor<32x384x56x56xf32>
    %v4510 = stablehlo.multiply %v4506, %v4495 : tensor<32x384x56x56xf32>
    %v4511 = stablehlo.multiply %v4510, %v4509 : tensor<32x384x56x56xf32>
    %v4512 = stablehlo.constant dense<0.134145> : tensor<32x384x56x56xf32>
    %v4513 = stablehlo.multiply %v4512, %v4496 : tensor<32x384x56x56xf32>
    %v4514 = stablehlo.add %v4504, %v4513 : tensor<32x384x56x56xf32>
    %v4515 = stablehlo.multiply %v4501, %v4514 : tensor<32x384x56x56xf32>
    %v4516 = stablehlo.multiply %v4511, %v4515 : tensor<32x384x56x56xf32>
    %v4517 = stablehlo.add %v4507, %v4516 : tensor<32x384x56x56xf32>
    %v4518 = stablehlo.multiply %v4494, %v4517 : tensor<32x384x56x56xf32>
    %v4519 = stablehlo.reshape %v4518 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4520 = stablehlo.reshape %v4519 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4521 = stablehlo.transpose %s0b2eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4522 = stablehlo.reverse %v4521, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v4523 = stablehlo.convolution(%v4520, %v4522)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4524 = stablehlo.reshape %v4523 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4525 = stablehlo.reshape %v187 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4526 = stablehlo.transpose %v4525, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4527 = stablehlo.reshape %v4526 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4528 = stablehlo.reshape %v4524 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4529 = stablehlo.transpose %v4528, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4530 = stablehlo.reshape %v4529 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4531 = stablehlo.reshape %v4530 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4532 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4533 = stablehlo.multiply %v4531, %v4532 : tensor<32x3136x96xf32>
    %v4534 = stablehlo.reshape %v4533 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4535 = stablehlo.reshape %v4534 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4536 = stablehlo.reshape %v4527 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4538 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4539 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4540 = stablehlo.reduce(%v4536 init: %v4537) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4541 = stablehlo.broadcast_in_dim %v4540, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4542 = stablehlo.divide %v4541, %v4538 : tensor<32x3136x96xf32>
    %v4543 = stablehlo.subtract %v4536, %v4542 : tensor<32x3136x96xf32>
    %v4544 = stablehlo.multiply %v4543, %v4543 : tensor<32x3136x96xf32>
    %v4545 = stablehlo.reduce(%v4544 init: %v4537) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4546 = stablehlo.broadcast_in_dim %v4545, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4547 = stablehlo.divide %v4546, %v4538 : tensor<32x3136x96xf32>
    %v4548 = stablehlo.add %v4547, %v4539 : tensor<32x3136x96xf32>
    %v4549 = stablehlo.rsqrt %v4548 : tensor<32x3136x96xf32>
    %v4550 = stablehlo.multiply %v4543, %v4549 : tensor<32x3136x96xf32>
    %v4551 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4552 = stablehlo.multiply %v4551, %v4535 : tensor<32x3136x96xf32>
    %v4553 = stablehlo.reduce(%v4552 init: %v4537) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4554 = stablehlo.broadcast_in_dim %v4553, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4555 = stablehlo.multiply %v4550, %v4552 : tensor<32x3136x96xf32>
    %v4556 = stablehlo.reduce(%v4555 init: %v4537) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4557 = stablehlo.broadcast_in_dim %v4556, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4558 = stablehlo.multiply %v4552, %v4538 : tensor<32x3136x96xf32>
    %v4559 = stablehlo.subtract %v4558, %v4554 : tensor<32x3136x96xf32>
    %v4560 = stablehlo.multiply %v4550, %v4557 : tensor<32x3136x96xf32>
    %v4561 = stablehlo.subtract %v4559, %v4560 : tensor<32x3136x96xf32>
    %v4562 = stablehlo.divide %v4549, %v4538 : tensor<32x3136x96xf32>
    %v4563 = stablehlo.multiply %v4562, %v4561 : tensor<32x3136x96xf32>
    %v4564 = stablehlo.reshape %v4563 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4565 = stablehlo.reshape %v4564 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4566 = stablehlo.transpose %v4565, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4567 = stablehlo.reshape %v4566 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4568 = stablehlo.reshape %v4567 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4569 = stablehlo.reverse %s0b2dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v4570 = stablehlo.convolution(%v4568, %v4569)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v4571 = stablehlo.reshape %v4570 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4572 = stablehlo.reshape %v4571 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4573 = stablehlo.reshape %v4431 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4574 = stablehlo.add %v4572, %v4573 : tensor<32x96x56x56xf32>
    %v4575 = stablehlo.reshape %v4574 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4576 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4577 = stablehlo.reshape %v246 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4578 = stablehlo.reshape %v4431 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4579 = stablehlo.multiply %v4577, %v4578 : tensor<32x96x56x56xf32>
    %v4580 = stablehlo.reduce(%v4579 init: %v4576) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4581 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4582 = stablehlo.multiply %v4580, %v4581 : tensor<96xf32>
    %v4583 = stablehlo.subtract %s0b2lg, %v4582 : tensor<96xf32>
    %v4584 = stablehlo.reshape %v241 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4585 = stablehlo.reshape %v4488 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4586 = stablehlo.transpose %v4584, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4587 = stablehlo.transpose %v4585, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4588 = stablehlo.convolution(%v4586, %v4587)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v4589 = stablehlo.transpose %v4588, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4590 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v4591 = stablehlo.multiply %v4589, %v4590 : tensor<96x384x1x1xf32>
    %v4592 = stablehlo.subtract %s0b2pW, %v4591 : tensor<96x384x1x1xf32>
    %v4593 = stablehlo.reshape %v4488 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4594 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4595 = stablehlo.reduce(%v4593 init: %v4594) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4596 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4597 = stablehlo.multiply %v4595, %v4596 : tensor<96xf32>
    %v4598 = stablehlo.subtract %s0b2pb, %v4597 : tensor<96xf32>
    %v4599 = stablehlo.reshape %v221 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4600 = stablehlo.reshape %v4519 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4601 = stablehlo.transpose %v4599, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4602 = stablehlo.transpose %v4600, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4603 = stablehlo.convolution(%v4601, %v4602)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v4604 = stablehlo.transpose %v4603, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4605 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v4606 = stablehlo.multiply %v4604, %v4605 : tensor<384x96x1x1xf32>
    %v4607 = stablehlo.subtract %s0b2eW, %v4606 : tensor<384x96x1x1xf32>
    %v4608 = stablehlo.reshape %v4519 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4610 = stablehlo.reduce(%v4608 init: %v4609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v4611 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v4612 = stablehlo.multiply %v4610, %v4611 : tensor<384xf32>
    %v4613 = stablehlo.subtract %s0b2eb, %v4612 : tensor<384xf32>
    %v4614 = stablehlo.reshape %v187 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4615 = stablehlo.transpose %v4614, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4616 = stablehlo.reshape %v4615 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4617 = stablehlo.reshape %v4524 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4618 = stablehlo.transpose %v4617, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4619 = stablehlo.reshape %v4618 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4620 = stablehlo.reshape %v4616 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4621 = stablehlo.reshape %v4619 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4622 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4623 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4624 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4625 = stablehlo.reduce(%v4620 init: %v4622) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4626 = stablehlo.broadcast_in_dim %v4625, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4627 = stablehlo.divide %v4626, %v4623 : tensor<32x3136x96xf32>
    %v4628 = stablehlo.subtract %v4620, %v4627 : tensor<32x3136x96xf32>
    %v4629 = stablehlo.multiply %v4628, %v4628 : tensor<32x3136x96xf32>
    %v4630 = stablehlo.reduce(%v4629 init: %v4622) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4631 = stablehlo.broadcast_in_dim %v4630, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4632 = stablehlo.divide %v4631, %v4623 : tensor<32x3136x96xf32>
    %v4633 = stablehlo.add %v4632, %v4624 : tensor<32x3136x96xf32>
    %v4634 = stablehlo.rsqrt %v4633 : tensor<32x3136x96xf32>
    %v4635 = stablehlo.multiply %v4628, %v4634 : tensor<32x3136x96xf32>
    %v4636 = stablehlo.multiply %v4621, %v4635 : tensor<32x3136x96xf32>
    %v4637 = stablehlo.reduce(%v4636 init: %v4622) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4638 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4639 = stablehlo.multiply %v4637, %v4638 : tensor<96xf32>
    %v4640 = stablehlo.subtract %s0b2ng, %v4639 : tensor<96xf32>
    %v4641 = stablehlo.reshape %v4524 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4642 = stablehlo.transpose %v4641, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4643 = stablehlo.reshape %v4642 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4645 = stablehlo.reshape %v4643 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4646 = stablehlo.reduce(%v4645 init: %v4644) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4647 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4648 = stablehlo.multiply %v4646, %v4647 : tensor<96xf32>
    %v4649 = stablehlo.subtract %s0b2nbt, %v4648 : tensor<96xf32>
    %v4650 = stablehlo.reshape %v182 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4651 = stablehlo.reshape %v4567 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4652 = stablehlo.transpose %v4650, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4653 = stablehlo.transpose %v4651, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4654 = stablehlo.convolution(%v4652, %v4653)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v4655 = stablehlo.reshape %v4654 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v4656 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v4657 = stablehlo.multiply %v4655, %v4656 : tensor<96x1x7x7xf32>
    %v4658 = stablehlo.subtract %s0b2dW, %v4657 : tensor<96x1x7x7xf32>
    %v4659 = stablehlo.reshape %v4567 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4660 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4661 = stablehlo.reduce(%v4659 init: %v4660) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4662 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4663 = stablehlo.multiply %v4661, %v4662 : tensor<96xf32>
    %v4664 = stablehlo.subtract %s0b2db, %v4663 : tensor<96xf32>
    %v4665 = stablehlo.reshape %v4575 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4666 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4667 = stablehlo.multiply %v4665, %v4666 : tensor<32x96x56x56xf32>
    %v4668 = stablehlo.reshape %v4667 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4669 = stablehlo.reshape %v4668 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4670 = stablehlo.transpose %s0b1pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4671 = stablehlo.reverse %v4670, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v4672 = stablehlo.convolution(%v4669, %v4671)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v4673 = stablehlo.reshape %v4672 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4674 = stablehlo.reshape %v4673 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4675 = stablehlo.reshape %v154 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4676 = stablehlo.multiply %v4675, %v4675 : tensor<32x384x56x56xf32>
    %v4677 = stablehlo.multiply %v4676, %v4675 : tensor<32x384x56x56xf32>
    %v4678 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v4679 = stablehlo.multiply %v4678, %v4677 : tensor<32x384x56x56xf32>
    %v4680 = stablehlo.add %v4675, %v4679 : tensor<32x384x56x56xf32>
    %v4681 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v4682 = stablehlo.multiply %v4681, %v4680 : tensor<32x384x56x56xf32>
    %v4683 = stablehlo.tanh %v4682 : tensor<32x384x56x56xf32>
    %v4684 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v4685 = stablehlo.add %v4684, %v4683 : tensor<32x384x56x56xf32>
    %v4686 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v4687 = stablehlo.multiply %v4686, %v4685 : tensor<32x384x56x56xf32>
    %v4688 = stablehlo.multiply %v4683, %v4683 : tensor<32x384x56x56xf32>
    %v4689 = stablehlo.subtract %v4684, %v4688 : tensor<32x384x56x56xf32>
    %v4690 = stablehlo.multiply %v4686, %v4675 : tensor<32x384x56x56xf32>
    %v4691 = stablehlo.multiply %v4690, %v4689 : tensor<32x384x56x56xf32>
    %v4692 = stablehlo.constant dense<0.134145> : tensor<32x384x56x56xf32>
    %v4693 = stablehlo.multiply %v4692, %v4676 : tensor<32x384x56x56xf32>
    %v4694 = stablehlo.add %v4684, %v4693 : tensor<32x384x56x56xf32>
    %v4695 = stablehlo.multiply %v4681, %v4694 : tensor<32x384x56x56xf32>
    %v4696 = stablehlo.multiply %v4691, %v4695 : tensor<32x384x56x56xf32>
    %v4697 = stablehlo.add %v4687, %v4696 : tensor<32x384x56x56xf32>
    %v4698 = stablehlo.multiply %v4674, %v4697 : tensor<32x384x56x56xf32>
    %v4699 = stablehlo.reshape %v4698 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4700 = stablehlo.reshape %v4699 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4701 = stablehlo.transpose %s0b1eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4702 = stablehlo.reverse %v4701, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v4703 = stablehlo.convolution(%v4700, %v4702)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4704 = stablehlo.reshape %v4703 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4705 = stablehlo.reshape %v115 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4706 = stablehlo.transpose %v4705, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4707 = stablehlo.reshape %v4706 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4708 = stablehlo.reshape %v4704 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4709 = stablehlo.transpose %v4708, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4710 = stablehlo.reshape %v4709 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4711 = stablehlo.reshape %v4710 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4712 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4713 = stablehlo.multiply %v4711, %v4712 : tensor<32x3136x96xf32>
    %v4714 = stablehlo.reshape %v4713 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4715 = stablehlo.reshape %v4714 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4716 = stablehlo.reshape %v4707 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4717 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4718 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4719 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4720 = stablehlo.reduce(%v4716 init: %v4717) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4721 = stablehlo.broadcast_in_dim %v4720, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4722 = stablehlo.divide %v4721, %v4718 : tensor<32x3136x96xf32>
    %v4723 = stablehlo.subtract %v4716, %v4722 : tensor<32x3136x96xf32>
    %v4724 = stablehlo.multiply %v4723, %v4723 : tensor<32x3136x96xf32>
    %v4725 = stablehlo.reduce(%v4724 init: %v4717) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4726 = stablehlo.broadcast_in_dim %v4725, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4727 = stablehlo.divide %v4726, %v4718 : tensor<32x3136x96xf32>
    %v4728 = stablehlo.add %v4727, %v4719 : tensor<32x3136x96xf32>
    %v4729 = stablehlo.rsqrt %v4728 : tensor<32x3136x96xf32>
    %v4730 = stablehlo.multiply %v4723, %v4729 : tensor<32x3136x96xf32>
    %v4731 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4732 = stablehlo.multiply %v4731, %v4715 : tensor<32x3136x96xf32>
    %v4733 = stablehlo.reduce(%v4732 init: %v4717) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4734 = stablehlo.broadcast_in_dim %v4733, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4735 = stablehlo.multiply %v4730, %v4732 : tensor<32x3136x96xf32>
    %v4736 = stablehlo.reduce(%v4735 init: %v4717) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4737 = stablehlo.broadcast_in_dim %v4736, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4738 = stablehlo.multiply %v4732, %v4718 : tensor<32x3136x96xf32>
    %v4739 = stablehlo.subtract %v4738, %v4734 : tensor<32x3136x96xf32>
    %v4740 = stablehlo.multiply %v4730, %v4737 : tensor<32x3136x96xf32>
    %v4741 = stablehlo.subtract %v4739, %v4740 : tensor<32x3136x96xf32>
    %v4742 = stablehlo.divide %v4729, %v4718 : tensor<32x3136x96xf32>
    %v4743 = stablehlo.multiply %v4742, %v4741 : tensor<32x3136x96xf32>
    %v4744 = stablehlo.reshape %v4743 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4745 = stablehlo.reshape %v4744 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4746 = stablehlo.transpose %v4745, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4747 = stablehlo.reshape %v4746 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4748 = stablehlo.reshape %v4747 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4749 = stablehlo.reverse %s0b1dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v4750 = stablehlo.convolution(%v4748, %v4749)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v4751 = stablehlo.reshape %v4750 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4752 = stablehlo.reshape %v4751 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4753 = stablehlo.reshape %v4575 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4754 = stablehlo.add %v4752, %v4753 : tensor<32x96x56x56xf32>
    %v4755 = stablehlo.reshape %v4754 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4757 = stablehlo.reshape %v174 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4758 = stablehlo.reshape %v4575 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4759 = stablehlo.multiply %v4757, %v4758 : tensor<32x96x56x56xf32>
    %v4760 = stablehlo.reduce(%v4759 init: %v4756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4761 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4762 = stablehlo.multiply %v4760, %v4761 : tensor<96xf32>
    %v4763 = stablehlo.subtract %s0b1lg, %v4762 : tensor<96xf32>
    %v4764 = stablehlo.reshape %v169 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4765 = stablehlo.reshape %v4668 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4766 = stablehlo.transpose %v4764, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4767 = stablehlo.transpose %v4765, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4768 = stablehlo.convolution(%v4766, %v4767)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v4769 = stablehlo.transpose %v4768, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4770 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v4771 = stablehlo.multiply %v4769, %v4770 : tensor<96x384x1x1xf32>
    %v4772 = stablehlo.subtract %s0b1pW, %v4771 : tensor<96x384x1x1xf32>
    %v4773 = stablehlo.reshape %v4668 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4775 = stablehlo.reduce(%v4773 init: %v4774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4776 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4777 = stablehlo.multiply %v4775, %v4776 : tensor<96xf32>
    %v4778 = stablehlo.subtract %s0b1pb, %v4777 : tensor<96xf32>
    %v4779 = stablehlo.reshape %v149 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4780 = stablehlo.reshape %v4699 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4781 = stablehlo.transpose %v4779, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4782 = stablehlo.transpose %v4780, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4783 = stablehlo.convolution(%v4781, %v4782)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v4784 = stablehlo.transpose %v4783, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4785 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v4786 = stablehlo.multiply %v4784, %v4785 : tensor<384x96x1x1xf32>
    %v4787 = stablehlo.subtract %s0b1eW, %v4786 : tensor<384x96x1x1xf32>
    %v4788 = stablehlo.reshape %v4699 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4790 = stablehlo.reduce(%v4788 init: %v4789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v4791 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v4792 = stablehlo.multiply %v4790, %v4791 : tensor<384xf32>
    %v4793 = stablehlo.subtract %s0b1eb, %v4792 : tensor<384xf32>
    %v4794 = stablehlo.reshape %v115 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4795 = stablehlo.transpose %v4794, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4796 = stablehlo.reshape %v4795 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4797 = stablehlo.reshape %v4704 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4798 = stablehlo.transpose %v4797, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4799 = stablehlo.reshape %v4798 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4800 = stablehlo.reshape %v4796 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4801 = stablehlo.reshape %v4799 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4803 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4804 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4805 = stablehlo.reduce(%v4800 init: %v4802) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4806 = stablehlo.broadcast_in_dim %v4805, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4807 = stablehlo.divide %v4806, %v4803 : tensor<32x3136x96xf32>
    %v4808 = stablehlo.subtract %v4800, %v4807 : tensor<32x3136x96xf32>
    %v4809 = stablehlo.multiply %v4808, %v4808 : tensor<32x3136x96xf32>
    %v4810 = stablehlo.reduce(%v4809 init: %v4802) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4811 = stablehlo.broadcast_in_dim %v4810, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4812 = stablehlo.divide %v4811, %v4803 : tensor<32x3136x96xf32>
    %v4813 = stablehlo.add %v4812, %v4804 : tensor<32x3136x96xf32>
    %v4814 = stablehlo.rsqrt %v4813 : tensor<32x3136x96xf32>
    %v4815 = stablehlo.multiply %v4808, %v4814 : tensor<32x3136x96xf32>
    %v4816 = stablehlo.multiply %v4801, %v4815 : tensor<32x3136x96xf32>
    %v4817 = stablehlo.reduce(%v4816 init: %v4802) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4818 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4819 = stablehlo.multiply %v4817, %v4818 : tensor<96xf32>
    %v4820 = stablehlo.subtract %s0b1ng, %v4819 : tensor<96xf32>
    %v4821 = stablehlo.reshape %v4704 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4822 = stablehlo.transpose %v4821, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4823 = stablehlo.reshape %v4822 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4824 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4825 = stablehlo.reshape %v4823 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4826 = stablehlo.reduce(%v4825 init: %v4824) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4827 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4828 = stablehlo.multiply %v4826, %v4827 : tensor<96xf32>
    %v4829 = stablehlo.subtract %s0b1nbt, %v4828 : tensor<96xf32>
    %v4830 = stablehlo.reshape %v110 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4831 = stablehlo.reshape %v4747 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4832 = stablehlo.transpose %v4830, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4833 = stablehlo.transpose %v4831, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4834 = stablehlo.convolution(%v4832, %v4833)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v4835 = stablehlo.reshape %v4834 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v4836 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v4837 = stablehlo.multiply %v4835, %v4836 : tensor<96x1x7x7xf32>
    %v4838 = stablehlo.subtract %s0b1dW, %v4837 : tensor<96x1x7x7xf32>
    %v4839 = stablehlo.reshape %v4747 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4841 = stablehlo.reduce(%v4839 init: %v4840) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4842 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4843 = stablehlo.multiply %v4841, %v4842 : tensor<96xf32>
    %v4844 = stablehlo.subtract %s0b1db, %v4843 : tensor<96xf32>
    %v4845 = stablehlo.reshape %v4755 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4846 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4847 = stablehlo.multiply %v4845, %v4846 : tensor<32x96x56x56xf32>
    %v4848 = stablehlo.reshape %v4847 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4849 = stablehlo.reshape %v4848 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4850 = stablehlo.transpose %s0b0pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4851 = stablehlo.reverse %v4850, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v4852 = stablehlo.convolution(%v4849, %v4851)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v4853 = stablehlo.reshape %v4852 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4854 = stablehlo.reshape %v4853 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4855 = stablehlo.reshape %v82 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4856 = stablehlo.multiply %v4855, %v4855 : tensor<32x384x56x56xf32>
    %v4857 = stablehlo.multiply %v4856, %v4855 : tensor<32x384x56x56xf32>
    %v4858 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v4859 = stablehlo.multiply %v4858, %v4857 : tensor<32x384x56x56xf32>
    %v4860 = stablehlo.add %v4855, %v4859 : tensor<32x384x56x56xf32>
    %v4861 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v4862 = stablehlo.multiply %v4861, %v4860 : tensor<32x384x56x56xf32>
    %v4863 = stablehlo.tanh %v4862 : tensor<32x384x56x56xf32>
    %v4864 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v4865 = stablehlo.add %v4864, %v4863 : tensor<32x384x56x56xf32>
    %v4866 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v4867 = stablehlo.multiply %v4866, %v4865 : tensor<32x384x56x56xf32>
    %v4868 = stablehlo.multiply %v4863, %v4863 : tensor<32x384x56x56xf32>
    %v4869 = stablehlo.subtract %v4864, %v4868 : tensor<32x384x56x56xf32>
    %v4870 = stablehlo.multiply %v4866, %v4855 : tensor<32x384x56x56xf32>
    %v4871 = stablehlo.multiply %v4870, %v4869 : tensor<32x384x56x56xf32>
    %v4872 = stablehlo.constant dense<0.134145> : tensor<32x384x56x56xf32>
    %v4873 = stablehlo.multiply %v4872, %v4856 : tensor<32x384x56x56xf32>
    %v4874 = stablehlo.add %v4864, %v4873 : tensor<32x384x56x56xf32>
    %v4875 = stablehlo.multiply %v4861, %v4874 : tensor<32x384x56x56xf32>
    %v4876 = stablehlo.multiply %v4871, %v4875 : tensor<32x384x56x56xf32>
    %v4877 = stablehlo.add %v4867, %v4876 : tensor<32x384x56x56xf32>
    %v4878 = stablehlo.multiply %v4854, %v4877 : tensor<32x384x56x56xf32>
    %v4879 = stablehlo.reshape %v4878 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4880 = stablehlo.reshape %v4879 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4881 = stablehlo.transpose %s0b0eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4882 = stablehlo.reverse %v4881, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v4883 = stablehlo.convolution(%v4880, %v4882)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4884 = stablehlo.reshape %v4883 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4885 = stablehlo.reshape %v43 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4886 = stablehlo.transpose %v4885, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4887 = stablehlo.reshape %v4886 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4888 = stablehlo.reshape %v4884 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4889 = stablehlo.transpose %v4888, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4890 = stablehlo.reshape %v4889 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4891 = stablehlo.reshape %v4890 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4892 = stablehlo.broadcast_in_dim %s0b0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4893 = stablehlo.multiply %v4891, %v4892 : tensor<32x3136x96xf32>
    %v4894 = stablehlo.reshape %v4893 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4895 = stablehlo.reshape %v4894 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4896 = stablehlo.reshape %v4887 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4898 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4899 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4900 = stablehlo.reduce(%v4896 init: %v4897) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4901 = stablehlo.broadcast_in_dim %v4900, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4902 = stablehlo.divide %v4901, %v4898 : tensor<32x3136x96xf32>
    %v4903 = stablehlo.subtract %v4896, %v4902 : tensor<32x3136x96xf32>
    %v4904 = stablehlo.multiply %v4903, %v4903 : tensor<32x3136x96xf32>
    %v4905 = stablehlo.reduce(%v4904 init: %v4897) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4906 = stablehlo.broadcast_in_dim %v4905, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4907 = stablehlo.divide %v4906, %v4898 : tensor<32x3136x96xf32>
    %v4908 = stablehlo.add %v4907, %v4899 : tensor<32x3136x96xf32>
    %v4909 = stablehlo.rsqrt %v4908 : tensor<32x3136x96xf32>
    %v4910 = stablehlo.multiply %v4903, %v4909 : tensor<32x3136x96xf32>
    %v4911 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4912 = stablehlo.multiply %v4911, %v4895 : tensor<32x3136x96xf32>
    %v4913 = stablehlo.reduce(%v4912 init: %v4897) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4914 = stablehlo.broadcast_in_dim %v4913, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4915 = stablehlo.multiply %v4910, %v4912 : tensor<32x3136x96xf32>
    %v4916 = stablehlo.reduce(%v4915 init: %v4897) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4917 = stablehlo.broadcast_in_dim %v4916, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4918 = stablehlo.multiply %v4912, %v4898 : tensor<32x3136x96xf32>
    %v4919 = stablehlo.subtract %v4918, %v4914 : tensor<32x3136x96xf32>
    %v4920 = stablehlo.multiply %v4910, %v4917 : tensor<32x3136x96xf32>
    %v4921 = stablehlo.subtract %v4919, %v4920 : tensor<32x3136x96xf32>
    %v4922 = stablehlo.divide %v4909, %v4898 : tensor<32x3136x96xf32>
    %v4923 = stablehlo.multiply %v4922, %v4921 : tensor<32x3136x96xf32>
    %v4924 = stablehlo.reshape %v4923 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4925 = stablehlo.reshape %v4924 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4926 = stablehlo.transpose %v4925, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4927 = stablehlo.reshape %v4926 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4928 = stablehlo.reshape %v4927 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4929 = stablehlo.reverse %s0b0dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v4930 = stablehlo.convolution(%v4928, %v4929)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v4931 = stablehlo.reshape %v4930 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4932 = stablehlo.reshape %v4931 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4933 = stablehlo.reshape %v4755 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4934 = stablehlo.add %v4932, %v4933 : tensor<32x96x56x56xf32>
    %v4935 = stablehlo.reshape %v4934 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4937 = stablehlo.reshape %v102 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4938 = stablehlo.reshape %v4755 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4939 = stablehlo.multiply %v4937, %v4938 : tensor<32x96x56x56xf32>
    %v4940 = stablehlo.reduce(%v4939 init: %v4936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4941 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4942 = stablehlo.multiply %v4940, %v4941 : tensor<96xf32>
    %v4943 = stablehlo.subtract %s0b0lg, %v4942 : tensor<96xf32>
    %v4944 = stablehlo.reshape %v97 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4945 = stablehlo.reshape %v4848 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4946 = stablehlo.transpose %v4944, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4947 = stablehlo.transpose %v4945, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4948 = stablehlo.convolution(%v4946, %v4947)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v4949 = stablehlo.transpose %v4948, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4950 = stablehlo.constant dense<0.1> : tensor<96x384x1x1xf32>
    %v4951 = stablehlo.multiply %v4949, %v4950 : tensor<96x384x1x1xf32>
    %v4952 = stablehlo.subtract %s0b0pW, %v4951 : tensor<96x384x1x1xf32>
    %v4953 = stablehlo.reshape %v4848 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4955 = stablehlo.reduce(%v4953 init: %v4954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4956 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4957 = stablehlo.multiply %v4955, %v4956 : tensor<96xf32>
    %v4958 = stablehlo.subtract %s0b0pb, %v4957 : tensor<96xf32>
    %v4959 = stablehlo.reshape %v77 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4960 = stablehlo.reshape %v4879 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4961 = stablehlo.transpose %v4959, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4962 = stablehlo.transpose %v4960, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4963 = stablehlo.convolution(%v4961, %v4962)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v4964 = stablehlo.transpose %v4963, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4965 = stablehlo.constant dense<0.1> : tensor<384x96x1x1xf32>
    %v4966 = stablehlo.multiply %v4964, %v4965 : tensor<384x96x1x1xf32>
    %v4967 = stablehlo.subtract %s0b0eW, %v4966 : tensor<384x96x1x1xf32>
    %v4968 = stablehlo.reshape %v4879 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4969 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4970 = stablehlo.reduce(%v4968 init: %v4969) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v4971 = stablehlo.constant dense<0.1> : tensor<384xf32>
    %v4972 = stablehlo.multiply %v4970, %v4971 : tensor<384xf32>
    %v4973 = stablehlo.subtract %s0b0eb, %v4972 : tensor<384xf32>
    %v4974 = stablehlo.reshape %v43 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4975 = stablehlo.transpose %v4974, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4976 = stablehlo.reshape %v4975 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4977 = stablehlo.reshape %v4884 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4978 = stablehlo.transpose %v4977, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4979 = stablehlo.reshape %v4978 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4980 = stablehlo.reshape %v4976 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4981 = stablehlo.reshape %v4979 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4983 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4984 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4985 = stablehlo.reduce(%v4980 init: %v4982) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4986 = stablehlo.broadcast_in_dim %v4985, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4987 = stablehlo.divide %v4986, %v4983 : tensor<32x3136x96xf32>
    %v4988 = stablehlo.subtract %v4980, %v4987 : tensor<32x3136x96xf32>
    %v4989 = stablehlo.multiply %v4988, %v4988 : tensor<32x3136x96xf32>
    %v4990 = stablehlo.reduce(%v4989 init: %v4982) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4991 = stablehlo.broadcast_in_dim %v4990, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4992 = stablehlo.divide %v4991, %v4983 : tensor<32x3136x96xf32>
    %v4993 = stablehlo.add %v4992, %v4984 : tensor<32x3136x96xf32>
    %v4994 = stablehlo.rsqrt %v4993 : tensor<32x3136x96xf32>
    %v4995 = stablehlo.multiply %v4988, %v4994 : tensor<32x3136x96xf32>
    %v4996 = stablehlo.multiply %v4981, %v4995 : tensor<32x3136x96xf32>
    %v4997 = stablehlo.reduce(%v4996 init: %v4982) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4998 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v4999 = stablehlo.multiply %v4997, %v4998 : tensor<96xf32>
    %v5000 = stablehlo.subtract %s0b0ng, %v4999 : tensor<96xf32>
    %v5001 = stablehlo.reshape %v4884 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5002 = stablehlo.transpose %v5001, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5003 = stablehlo.reshape %v5002 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5004 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5005 = stablehlo.reshape %v5003 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5006 = stablehlo.reduce(%v5005 init: %v5004) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v5007 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5008 = stablehlo.multiply %v5006, %v5007 : tensor<96xf32>
    %v5009 = stablehlo.subtract %s0b0nbt, %v5008 : tensor<96xf32>
    %v5010 = stablehlo.reshape %v38 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5011 = stablehlo.reshape %v4927 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5012 = stablehlo.transpose %v5010, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v5013 = stablehlo.transpose %v5011, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v5014 = stablehlo.convolution(%v5012, %v5013)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v5015 = stablehlo.reshape %v5014 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v5016 = stablehlo.constant dense<0.1> : tensor<96x1x7x7xf32>
    %v5017 = stablehlo.multiply %v5015, %v5016 : tensor<96x1x7x7xf32>
    %v5018 = stablehlo.subtract %s0b0dW, %v5017 : tensor<96x1x7x7xf32>
    %v5019 = stablehlo.reshape %v4927 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5020 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5021 = stablehlo.reduce(%v5019 init: %v5020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5022 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5023 = stablehlo.multiply %v5021, %v5022 : tensor<96xf32>
    %v5024 = stablehlo.subtract %s0b0db, %v5023 : tensor<96xf32>
    %v5025 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5026 = stablehlo.transpose %v5025, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5027 = stablehlo.reshape %v5026 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5028 = stablehlo.reshape %v4935 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5029 = stablehlo.transpose %v5028, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5030 = stablehlo.reshape %v5029 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5031 = stablehlo.reshape %v5027 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5032 = stablehlo.reshape %v5030 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5033 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5034 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v5035 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v5036 = stablehlo.reduce(%v5031 init: %v5033) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5037 = stablehlo.broadcast_in_dim %v5036, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5038 = stablehlo.divide %v5037, %v5034 : tensor<32x3136x96xf32>
    %v5039 = stablehlo.subtract %v5031, %v5038 : tensor<32x3136x96xf32>
    %v5040 = stablehlo.multiply %v5039, %v5039 : tensor<32x3136x96xf32>
    %v5041 = stablehlo.reduce(%v5040 init: %v5033) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5042 = stablehlo.broadcast_in_dim %v5041, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5043 = stablehlo.divide %v5042, %v5034 : tensor<32x3136x96xf32>
    %v5044 = stablehlo.add %v5043, %v5035 : tensor<32x3136x96xf32>
    %v5045 = stablehlo.rsqrt %v5044 : tensor<32x3136x96xf32>
    %v5046 = stablehlo.multiply %v5039, %v5045 : tensor<32x3136x96xf32>
    %v5047 = stablehlo.multiply %v5032, %v5046 : tensor<32x3136x96xf32>
    %v5048 = stablehlo.reduce(%v5047 init: %v5033) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v5049 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5050 = stablehlo.multiply %v5048, %v5049 : tensor<96xf32>
    %v5051 = stablehlo.subtract %psng, %v5050 : tensor<96xf32>
    %v5052 = stablehlo.reshape %v4935 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5053 = stablehlo.transpose %v5052, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5054 = stablehlo.reshape %v5053 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5056 = stablehlo.reshape %v5054 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5057 = stablehlo.reduce(%v5056 init: %v5055) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v5058 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5059 = stablehlo.multiply %v5057, %v5058 : tensor<96xf32>
    %v5060 = stablehlo.subtract %psnbt, %v5059 : tensor<96xf32>
    %v5061 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5062 = stablehlo.transpose %v5061, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5063 = stablehlo.reshape %v5062 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5064 = stablehlo.reshape %v4935 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v5065 = stablehlo.transpose %v5064, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v5066 = stablehlo.reshape %v5065 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5067 = stablehlo.reshape %v5066 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5068 = stablehlo.broadcast_in_dim %psng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v5069 = stablehlo.multiply %v5067, %v5068 : tensor<32x3136x96xf32>
    %v5070 = stablehlo.reshape %v5069 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5071 = stablehlo.reshape %v5070 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5072 = stablehlo.reshape %v5063 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5074 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v5075 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v5076 = stablehlo.reduce(%v5072 init: %v5073) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5077 = stablehlo.broadcast_in_dim %v5076, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5078 = stablehlo.divide %v5077, %v5074 : tensor<32x3136x96xf32>
    %v5079 = stablehlo.subtract %v5072, %v5078 : tensor<32x3136x96xf32>
    %v5080 = stablehlo.multiply %v5079, %v5079 : tensor<32x3136x96xf32>
    %v5081 = stablehlo.reduce(%v5080 init: %v5073) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5082 = stablehlo.broadcast_in_dim %v5081, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5083 = stablehlo.divide %v5082, %v5074 : tensor<32x3136x96xf32>
    %v5084 = stablehlo.add %v5083, %v5075 : tensor<32x3136x96xf32>
    %v5085 = stablehlo.rsqrt %v5084 : tensor<32x3136x96xf32>
    %v5086 = stablehlo.multiply %v5079, %v5085 : tensor<32x3136x96xf32>
    %v5087 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v5088 = stablehlo.multiply %v5087, %v5071 : tensor<32x3136x96xf32>
    %v5089 = stablehlo.reduce(%v5088 init: %v5073) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5090 = stablehlo.broadcast_in_dim %v5089, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5091 = stablehlo.multiply %v5086, %v5088 : tensor<32x3136x96xf32>
    %v5092 = stablehlo.reduce(%v5091 init: %v5073) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v5093 = stablehlo.broadcast_in_dim %v5092, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v5094 = stablehlo.multiply %v5088, %v5074 : tensor<32x3136x96xf32>
    %v5095 = stablehlo.subtract %v5094, %v5090 : tensor<32x3136x96xf32>
    %v5096 = stablehlo.multiply %v5086, %v5093 : tensor<32x3136x96xf32>
    %v5097 = stablehlo.subtract %v5095, %v5096 : tensor<32x3136x96xf32>
    %v5098 = stablehlo.divide %v5085, %v5074 : tensor<32x3136x96xf32>
    %v5099 = stablehlo.multiply %v5098, %v5097 : tensor<32x3136x96xf32>
    %v5100 = stablehlo.reshape %v5099 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v5101 = stablehlo.reshape %v5100 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v5102 = stablehlo.transpose %v5101, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v5103 = stablehlo.reshape %v5102 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v5110 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v5111 = stablehlo.reshape %v5103 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5113 = stablehlo.pad %v5111, %v5112, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x221x221xf32>
    %v5114 = stablehlo.transpose %v5110, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v5115 = stablehlo.transpose %v5113, dims = [1, 0, 2, 3] : (tensor<32x96x221x221xf32>) -> tensor<96x32x221x221xf32>
    %v5116 = stablehlo.convolution(%v5114, %v5115)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<96x32x221x221xf32>) -> tensor<3x96x4x4xf32>
    %v5117 = stablehlo.transpose %v5116, dims = [1, 0, 2, 3] : (tensor<3x96x4x4xf32>) -> tensor<96x3x4x4xf32>
    %psWl = stablehlo.constant dense<0.1> : tensor<96x3x4x4xf32>
    %psWs = stablehlo.multiply %v5117, %psWl : tensor<96x3x4x4xf32>
    %psWn = stablehlo.subtract %psW, %psWs : tensor<96x3x4x4xf32>
    %v5104 = stablehlo.reshape %v5103 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5106 = stablehlo.reduce(%v5104 init: %v5105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5107 = stablehlo.constant dense<0.1> : tensor<96xf32>
    %v5108 = stablehlo.multiply %v5106, %v5107 : tensor<96xf32>
    %v5109 = stablehlo.subtract %psb, %v5108 : tensor<96xf32>
    return %psWn, %v5109, %v5051, %v5060, %v5018, %v5024, %v5000, %v5009, %v4967, %v4973, %v4952, %v4958, %v4943, %v4838, %v4844, %v4820, %v4829, %v4787, %v4793, %v4772, %v4778, %v4763, %v4658, %v4664, %v4640, %v4649, %v4607, %v4613, %v4592, %v4598, %v4583, %v4464, %v4473, %v4484, %v4437, %v4375, %v4381, %v4357, %v4366, %v4324, %v4330, %v4309, %v4315, %v4300, %v4195, %v4201, %v4177, %v4186, %v4144, %v4150, %v4129, %v4135, %v4120, %v4015, %v4021, %v3997, %v4006, %v3964, %v3970, %v3949, %v3955, %v3940, %v3821, %v3830, %v3841, %v3794, %v3732, %v3738, %v3714, %v3723, %v3681, %v3687, %v3666, %v3672, %v3657, %v3552, %v3558, %v3534, %v3543, %v3501, %v3507, %v3486, %v3492, %v3477, %v3372, %v3378, %v3354, %v3363, %v3321, %v3327, %v3306, %v3312, %v3297, %v3192, %v3198, %v3174, %v3183, %v3141, %v3147, %v3126, %v3132, %v3117, %v3012, %v3018, %v2994, %v3003, %v2961, %v2967, %v2946, %v2952, %v2937, %v2832, %v2838, %v2814, %v2823, %v2781, %v2787, %v2766, %v2772, %v2757, %v2652, %v2658, %v2634, %v2643, %v2601, %v2607, %v2586, %v2592, %v2577, %v2472, %v2478, %v2454, %v2463, %v2421, %v2427, %v2406, %v2412, %v2397, %v2292, %v2298, %v2274, %v2283, %v2241, %v2247, %v2226, %v2232, %v2217, %v2098, %v2107, %v2118, %v2071, %v2009, %v2015, %v1991, %v2000, %v1958, %v1964, %v1943, %v1949, %v1934, %v1829, %v1835, %v1811, %v1820, %v1778, %v1784, %v1763, %v1769, %v1754, %v1649, %v1655, %v1631, %v1640, %v1598, %v1604, %v1583, %v1589, %v1574, %v1470, %v1475 : tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x10xf32>, tensor<10xf32>
  }
}
