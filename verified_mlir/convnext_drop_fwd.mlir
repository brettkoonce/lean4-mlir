module @m {
  func.func @convnext_drop_fwd(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %psng: tensor<96xf32>, %psnbt: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<96xf32>, %s0b0nbt: tensor<96xf32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<96xf32>, %s0b1nbt: tensor<96xf32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<96xf32>, %s0b2nbt: tensor<96xf32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<96xf32>, %d0nbt: tensor<96xf32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<192xf32>, %s1b0nbt: tensor<192xf32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<192xf32>, %s1b1nbt: tensor<192xf32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<192xf32>, %s1b2nbt: tensor<192xf32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<192xf32>, %d1nbt: tensor<192xf32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<384xf32>, %s2b0nbt: tensor<384xf32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<384xf32>, %s2b1nbt: tensor<384xf32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<384xf32>, %s2b2nbt: tensor<384xf32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<384xf32>, %s2b3nbt: tensor<384xf32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<384xf32>, %s2b4nbt: tensor<384xf32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<384xf32>, %s2b5nbt: tensor<384xf32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<384xf32>, %s2b6nbt: tensor<384xf32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<384xf32>, %s2b7nbt: tensor<384xf32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<384xf32>, %s2b8nbt: tensor<384xf32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<384xf32>, %d2nbt: tensor<384xf32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<768xf32>, %s3b0nbt: tensor<768xf32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<768xf32>, %s3b1nbt: tensor<768xf32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<768xf32>, %s3b2nbt: tensor<768xf32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<768xf32>, %hnbt: tensor<768xf32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>, %dp0: tensor<32xf32>, %dp1: tensor<32xf32>, %dp2: tensor<32xf32>, %dp3: tensor<32xf32>, %dp4: tensor<32xf32>, %dp5: tensor<32xf32>, %dp6: tensor<32xf32>, %dp7: tensor<32xf32>, %dp8: tensor<32xf32>, %dp9: tensor<32xf32>, %dp10: tensor<32xf32>, %dp11: tensor<32xf32>, %dp12: tensor<32xf32>, %dp13: tensor<32xf32>, %dp14: tensor<32xf32>, %dp15: tensor<32xf32>, %dp16: tensor<32xf32>, %dp17: tensor<32xf32>) -> tensor<32x10xf32> {
    // ── ConvNeXt-T forward at the BATCHED index N := B, with STOCHASTIC DEPTH ──
    // 18 drop sites, one per block, on the RESIDUAL BRANCH (between LayerScale and the
    // skip add). Emitted in the forward too, at an all-ones mask supplied by the driver:
    // exactly the identity (Proofs.dropPath_ones_id), so this stays a byte-prefix of the
    // SD train step and the forward-subset-train-step audit keeps a partner.
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
    %v108 = stablehlo.broadcast_in_dim %dp0, dims = [0] : (tensor<32xf32>) -> tensor<32x96x56x56xf32>
    %v109 = stablehlo.multiply %v108, %v107 : tensor<32x96x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v112 = stablehlo.reshape %v38 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v113 = stablehlo.add %v111, %v112 : tensor<32x96x56x56xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v116 = stablehlo.convolution(%v115, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v117 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v118 = stablehlo.add %v116, %v117 : tensor<32x96x56x56xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v121 = stablehlo.transpose %v120, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v125 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v126 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v127 = stablehlo.reduce(%v123 init: %v124) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v129 = stablehlo.divide %v128, %v125 : tensor<32x3136x96xf32>
    %v130 = stablehlo.subtract %v123, %v129 : tensor<32x3136x96xf32>
    %v131 = stablehlo.multiply %v130, %v130 : tensor<32x3136x96xf32>
    %v132 = stablehlo.reduce(%v131 init: %v124) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v133 = stablehlo.broadcast_in_dim %v132, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v134 = stablehlo.divide %v133, %v125 : tensor<32x3136x96xf32>
    %v135 = stablehlo.add %v134, %v126 : tensor<32x3136x96xf32>
    %v136 = stablehlo.rsqrt %v135 : tensor<32x3136x96xf32>
    %v137 = stablehlo.multiply %v130, %v136 : tensor<32x3136x96xf32>
    %v138 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v139 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v140 = stablehlo.multiply %v137, %v138 : tensor<32x3136x96xf32>
    %v141 = stablehlo.add %v140, %v139 : tensor<32x3136x96xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v144 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v145 = stablehlo.multiply %v143, %v144 : tensor<32x3136x96xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v148 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v149 = stablehlo.add %v147, %v148 : tensor<32x3136x96xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v152 = stablehlo.transpose %v151, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v155 = stablehlo.convolution(%v154, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v156 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v157 = stablehlo.add %v155, %v156 : tensor<32x384x56x56xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v159 = stablehlo.reshape %v158 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v160 = stablehlo.multiply %v159, %v159 : tensor<32x384x56x56xf32>
    %v161 = stablehlo.multiply %v160, %v159 : tensor<32x384x56x56xf32>
    %v162 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v163 = stablehlo.multiply %v162, %v161 : tensor<32x384x56x56xf32>
    %v164 = stablehlo.add %v159, %v163 : tensor<32x384x56x56xf32>
    %v165 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v166 = stablehlo.multiply %v165, %v164 : tensor<32x384x56x56xf32>
    %v167 = stablehlo.tanh %v166 : tensor<32x384x56x56xf32>
    %v168 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v169 = stablehlo.add %v168, %v167 : tensor<32x384x56x56xf32>
    %v170 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v171 = stablehlo.multiply %v170, %v159 : tensor<32x384x56x56xf32>
    %v172 = stablehlo.multiply %v171, %v169 : tensor<32x384x56x56xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v175 = stablehlo.convolution(%v174, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v176 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v177 = stablehlo.add %v175, %v176 : tensor<32x96x56x56xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v180 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v181 = stablehlo.multiply %v179, %v180 : tensor<32x96x56x56xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v184 = stablehlo.broadcast_in_dim %dp1, dims = [0] : (tensor<32xf32>) -> tensor<32x96x56x56xf32>
    %v185 = stablehlo.multiply %v184, %v183 : tensor<32x96x56x56xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v188 = stablehlo.reshape %v114 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v189 = stablehlo.add %v187, %v188 : tensor<32x96x56x56xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v192 = stablehlo.convolution(%v191, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v193 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v194 = stablehlo.add %v192, %v193 : tensor<32x96x56x56xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v197 = stablehlo.transpose %v196, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v200 = stablehlo.constant dense<0.0> : tensor<f32>
    %v201 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v202 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v203 = stablehlo.reduce(%v199 init: %v200) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v204 = stablehlo.broadcast_in_dim %v203, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v205 = stablehlo.divide %v204, %v201 : tensor<32x3136x96xf32>
    %v206 = stablehlo.subtract %v199, %v205 : tensor<32x3136x96xf32>
    %v207 = stablehlo.multiply %v206, %v206 : tensor<32x3136x96xf32>
    %v208 = stablehlo.reduce(%v207 init: %v200) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v209 = stablehlo.broadcast_in_dim %v208, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v210 = stablehlo.divide %v209, %v201 : tensor<32x3136x96xf32>
    %v211 = stablehlo.add %v210, %v202 : tensor<32x3136x96xf32>
    %v212 = stablehlo.rsqrt %v211 : tensor<32x3136x96xf32>
    %v213 = stablehlo.multiply %v206, %v212 : tensor<32x3136x96xf32>
    %v214 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v215 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v216 = stablehlo.multiply %v213, %v214 : tensor<32x3136x96xf32>
    %v217 = stablehlo.add %v216, %v215 : tensor<32x3136x96xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v220 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v221 = stablehlo.multiply %v219, %v220 : tensor<32x3136x96xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v224 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v225 = stablehlo.add %v223, %v224 : tensor<32x3136x96xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v228 = stablehlo.transpose %v227, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v231 = stablehlo.convolution(%v230, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v232 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v233 = stablehlo.add %v231, %v232 : tensor<32x384x56x56xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v236 = stablehlo.multiply %v235, %v235 : tensor<32x384x56x56xf32>
    %v237 = stablehlo.multiply %v236, %v235 : tensor<32x384x56x56xf32>
    %v238 = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %v239 = stablehlo.multiply %v238, %v237 : tensor<32x384x56x56xf32>
    %v240 = stablehlo.add %v235, %v239 : tensor<32x384x56x56xf32>
    %v241 = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %v242 = stablehlo.multiply %v241, %v240 : tensor<32x384x56x56xf32>
    %v243 = stablehlo.tanh %v242 : tensor<32x384x56x56xf32>
    %v244 = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %v245 = stablehlo.add %v244, %v243 : tensor<32x384x56x56xf32>
    %v246 = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %v247 = stablehlo.multiply %v246, %v235 : tensor<32x384x56x56xf32>
    %v248 = stablehlo.multiply %v247, %v245 : tensor<32x384x56x56xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v251 = stablehlo.convolution(%v250, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v252 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v253 = stablehlo.add %v251, %v252 : tensor<32x96x56x56xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v256 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v257 = stablehlo.multiply %v255, %v256 : tensor<32x96x56x56xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v260 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x96x56x56xf32>
    %v261 = stablehlo.multiply %v260, %v259 : tensor<32x96x56x56xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v264 = stablehlo.reshape %v190 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v265 = stablehlo.add %v263, %v264 : tensor<32x96x56x56xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v268 = stablehlo.transpose %v267, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v272 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v273 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v274 = stablehlo.reduce(%v270 init: %v271) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v275 = stablehlo.broadcast_in_dim %v274, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v276 = stablehlo.divide %v275, %v272 : tensor<32x3136x96xf32>
    %v277 = stablehlo.subtract %v270, %v276 : tensor<32x3136x96xf32>
    %v278 = stablehlo.multiply %v277, %v277 : tensor<32x3136x96xf32>
    %v279 = stablehlo.reduce(%v278 init: %v271) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v280 = stablehlo.broadcast_in_dim %v279, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v281 = stablehlo.divide %v280, %v272 : tensor<32x3136x96xf32>
    %v282 = stablehlo.add %v281, %v273 : tensor<32x3136x96xf32>
    %v283 = stablehlo.rsqrt %v282 : tensor<32x3136x96xf32>
    %v284 = stablehlo.multiply %v277, %v283 : tensor<32x3136x96xf32>
    %v285 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v286 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v287 = stablehlo.multiply %v284, %v285 : tensor<32x3136x96xf32>
    %v288 = stablehlo.add %v287, %v286 : tensor<32x3136x96xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v291 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v292 = stablehlo.multiply %v290, %v291 : tensor<32x3136x96xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v294 = stablehlo.reshape %v293 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v295 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v296 = stablehlo.add %v294, %v295 : tensor<32x3136x96xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v299 = stablehlo.transpose %v298, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v302 = stablehlo.convolution(%v301, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v303 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v304 = stablehlo.add %v302, %v303 : tensor<32x192x28x28xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v307 = stablehlo.convolution(%v306, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<32x192x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v312 = stablehlo.transpose %v311, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v316 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v317 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v318 = stablehlo.reduce(%v314 init: %v315) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v319 = stablehlo.broadcast_in_dim %v318, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v320 = stablehlo.divide %v319, %v316 : tensor<32x784x192xf32>
    %v321 = stablehlo.subtract %v314, %v320 : tensor<32x784x192xf32>
    %v322 = stablehlo.multiply %v321, %v321 : tensor<32x784x192xf32>
    %v323 = stablehlo.reduce(%v322 init: %v315) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v324 = stablehlo.broadcast_in_dim %v323, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v325 = stablehlo.divide %v324, %v316 : tensor<32x784x192xf32>
    %v326 = stablehlo.add %v325, %v317 : tensor<32x784x192xf32>
    %v327 = stablehlo.rsqrt %v326 : tensor<32x784x192xf32>
    %v328 = stablehlo.multiply %v321, %v327 : tensor<32x784x192xf32>
    %v329 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v330 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v331 = stablehlo.multiply %v328, %v329 : tensor<32x784x192xf32>
    %v332 = stablehlo.add %v331, %v330 : tensor<32x784x192xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v335 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v336 = stablehlo.multiply %v334, %v335 : tensor<32x784x192xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v339 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<32x784x192xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v343 = stablehlo.transpose %v342, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v344 = stablehlo.reshape %v343 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v345 = stablehlo.reshape %v344 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v346 = stablehlo.convolution(%v345, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v347 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v348 = stablehlo.add %v346, %v347 : tensor<32x768x28x28xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v351 = stablehlo.multiply %v350, %v350 : tensor<32x768x28x28xf32>
    %v352 = stablehlo.multiply %v351, %v350 : tensor<32x768x28x28xf32>
    %v353 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v354 = stablehlo.multiply %v353, %v352 : tensor<32x768x28x28xf32>
    %v355 = stablehlo.add %v350, %v354 : tensor<32x768x28x28xf32>
    %v356 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v357 = stablehlo.multiply %v356, %v355 : tensor<32x768x28x28xf32>
    %v358 = stablehlo.tanh %v357 : tensor<32x768x28x28xf32>
    %v359 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v360 = stablehlo.add %v359, %v358 : tensor<32x768x28x28xf32>
    %v361 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v362 = stablehlo.multiply %v361, %v350 : tensor<32x768x28x28xf32>
    %v363 = stablehlo.multiply %v362, %v360 : tensor<32x768x28x28xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v366 = stablehlo.convolution(%v365, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v367 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v368 = stablehlo.add %v366, %v367 : tensor<32x192x28x28xf32>
    %v369 = stablehlo.reshape %v368 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v371 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v372 = stablehlo.multiply %v370, %v371 : tensor<32x192x28x28xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v375 = stablehlo.broadcast_in_dim %dp3, dims = [0] : (tensor<32xf32>) -> tensor<32x192x28x28xf32>
    %v376 = stablehlo.multiply %v375, %v374 : tensor<32x192x28x28xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v379 = stablehlo.reshape %v305 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v380 = stablehlo.add %v378, %v379 : tensor<32x192x28x28xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v383 = stablehlo.convolution(%v382, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v384 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v385 = stablehlo.add %v383, %v384 : tensor<32x192x28x28xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v388 = stablehlo.transpose %v387, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v391 = stablehlo.constant dense<0.0> : tensor<f32>
    %v392 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v393 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v394 = stablehlo.reduce(%v390 init: %v391) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v395 = stablehlo.broadcast_in_dim %v394, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v396 = stablehlo.divide %v395, %v392 : tensor<32x784x192xf32>
    %v397 = stablehlo.subtract %v390, %v396 : tensor<32x784x192xf32>
    %v398 = stablehlo.multiply %v397, %v397 : tensor<32x784x192xf32>
    %v399 = stablehlo.reduce(%v398 init: %v391) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v400 = stablehlo.broadcast_in_dim %v399, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v401 = stablehlo.divide %v400, %v392 : tensor<32x784x192xf32>
    %v402 = stablehlo.add %v401, %v393 : tensor<32x784x192xf32>
    %v403 = stablehlo.rsqrt %v402 : tensor<32x784x192xf32>
    %v404 = stablehlo.multiply %v397, %v403 : tensor<32x784x192xf32>
    %v405 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v406 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v407 = stablehlo.multiply %v404, %v405 : tensor<32x784x192xf32>
    %v408 = stablehlo.add %v407, %v406 : tensor<32x784x192xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v411 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v412 = stablehlo.multiply %v410, %v411 : tensor<32x784x192xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v415 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v416 = stablehlo.add %v414, %v415 : tensor<32x784x192xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v419 = stablehlo.transpose %v418, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v420 = stablehlo.reshape %v419 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v422 = stablehlo.convolution(%v421, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v423 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v424 = stablehlo.add %v422, %v423 : tensor<32x768x28x28xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v427 = stablehlo.multiply %v426, %v426 : tensor<32x768x28x28xf32>
    %v428 = stablehlo.multiply %v427, %v426 : tensor<32x768x28x28xf32>
    %v429 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v430 = stablehlo.multiply %v429, %v428 : tensor<32x768x28x28xf32>
    %v431 = stablehlo.add %v426, %v430 : tensor<32x768x28x28xf32>
    %v432 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v433 = stablehlo.multiply %v432, %v431 : tensor<32x768x28x28xf32>
    %v434 = stablehlo.tanh %v433 : tensor<32x768x28x28xf32>
    %v435 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v436 = stablehlo.add %v435, %v434 : tensor<32x768x28x28xf32>
    %v437 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v438 = stablehlo.multiply %v437, %v426 : tensor<32x768x28x28xf32>
    %v439 = stablehlo.multiply %v438, %v436 : tensor<32x768x28x28xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v442 = stablehlo.convolution(%v441, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v443 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v444 = stablehlo.add %v442, %v443 : tensor<32x192x28x28xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v447 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v448 = stablehlo.multiply %v446, %v447 : tensor<32x192x28x28xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v451 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x192x28x28xf32>
    %v452 = stablehlo.multiply %v451, %v450 : tensor<32x192x28x28xf32>
    %v453 = stablehlo.reshape %v452 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v455 = stablehlo.reshape %v381 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v456 = stablehlo.add %v454, %v455 : tensor<32x192x28x28xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v459 = stablehlo.convolution(%v458, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v460 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v461 = stablehlo.add %v459, %v460 : tensor<32x192x28x28xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v464 = stablehlo.transpose %v463, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v467 = stablehlo.constant dense<0.0> : tensor<f32>
    %v468 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v469 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v470 = stablehlo.reduce(%v466 init: %v467) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v471 = stablehlo.broadcast_in_dim %v470, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v472 = stablehlo.divide %v471, %v468 : tensor<32x784x192xf32>
    %v473 = stablehlo.subtract %v466, %v472 : tensor<32x784x192xf32>
    %v474 = stablehlo.multiply %v473, %v473 : tensor<32x784x192xf32>
    %v475 = stablehlo.reduce(%v474 init: %v467) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v476 = stablehlo.broadcast_in_dim %v475, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v477 = stablehlo.divide %v476, %v468 : tensor<32x784x192xf32>
    %v478 = stablehlo.add %v477, %v469 : tensor<32x784x192xf32>
    %v479 = stablehlo.rsqrt %v478 : tensor<32x784x192xf32>
    %v480 = stablehlo.multiply %v473, %v479 : tensor<32x784x192xf32>
    %v481 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v482 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v483 = stablehlo.multiply %v480, %v481 : tensor<32x784x192xf32>
    %v484 = stablehlo.add %v483, %v482 : tensor<32x784x192xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v487 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v488 = stablehlo.multiply %v486, %v487 : tensor<32x784x192xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v491 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v492 = stablehlo.add %v490, %v491 : tensor<32x784x192xf32>
    %v493 = stablehlo.reshape %v492 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v495 = stablehlo.transpose %v494, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v498 = stablehlo.convolution(%v497, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v499 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v500 = stablehlo.add %v498, %v499 : tensor<32x768x28x28xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v503 = stablehlo.multiply %v502, %v502 : tensor<32x768x28x28xf32>
    %v504 = stablehlo.multiply %v503, %v502 : tensor<32x768x28x28xf32>
    %v505 = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %v506 = stablehlo.multiply %v505, %v504 : tensor<32x768x28x28xf32>
    %v507 = stablehlo.add %v502, %v506 : tensor<32x768x28x28xf32>
    %v508 = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %v509 = stablehlo.multiply %v508, %v507 : tensor<32x768x28x28xf32>
    %v510 = stablehlo.tanh %v509 : tensor<32x768x28x28xf32>
    %v511 = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %v512 = stablehlo.add %v511, %v510 : tensor<32x768x28x28xf32>
    %v513 = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %v514 = stablehlo.multiply %v513, %v502 : tensor<32x768x28x28xf32>
    %v515 = stablehlo.multiply %v514, %v512 : tensor<32x768x28x28xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v518 = stablehlo.convolution(%v517, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v519 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v520 = stablehlo.add %v518, %v519 : tensor<32x192x28x28xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v523 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v524 = stablehlo.multiply %v522, %v523 : tensor<32x192x28x28xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v527 = stablehlo.broadcast_in_dim %dp5, dims = [0] : (tensor<32xf32>) -> tensor<32x192x28x28xf32>
    %v528 = stablehlo.multiply %v527, %v526 : tensor<32x192x28x28xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v531 = stablehlo.reshape %v457 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<32x192x28x28xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v535 = stablehlo.transpose %v534, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v539 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v540 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v541 = stablehlo.reduce(%v537 init: %v538) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v542 = stablehlo.broadcast_in_dim %v541, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v543 = stablehlo.divide %v542, %v539 : tensor<32x784x192xf32>
    %v544 = stablehlo.subtract %v537, %v543 : tensor<32x784x192xf32>
    %v545 = stablehlo.multiply %v544, %v544 : tensor<32x784x192xf32>
    %v546 = stablehlo.reduce(%v545 init: %v538) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v547 = stablehlo.broadcast_in_dim %v546, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v548 = stablehlo.divide %v547, %v539 : tensor<32x784x192xf32>
    %v549 = stablehlo.add %v548, %v540 : tensor<32x784x192xf32>
    %v550 = stablehlo.rsqrt %v549 : tensor<32x784x192xf32>
    %v551 = stablehlo.multiply %v544, %v550 : tensor<32x784x192xf32>
    %v552 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v553 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v554 = stablehlo.multiply %v551, %v552 : tensor<32x784x192xf32>
    %v555 = stablehlo.add %v554, %v553 : tensor<32x784x192xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v558 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v559 = stablehlo.multiply %v557, %v558 : tensor<32x784x192xf32>
    %v560 = stablehlo.reshape %v559 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v562 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v563 = stablehlo.add %v561, %v562 : tensor<32x784x192xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v566 = stablehlo.transpose %v565, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v569 = stablehlo.convolution(%v568, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v570 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<32x384x14x14xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v574 = stablehlo.convolution(%v573, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v575 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v576 = stablehlo.add %v574, %v575 : tensor<32x384x14x14xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v579 = stablehlo.transpose %v578, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v583 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v584 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v585 = stablehlo.reduce(%v581 init: %v582) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v586 = stablehlo.broadcast_in_dim %v585, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v587 = stablehlo.divide %v586, %v583 : tensor<32x196x384xf32>
    %v588 = stablehlo.subtract %v581, %v587 : tensor<32x196x384xf32>
    %v589 = stablehlo.multiply %v588, %v588 : tensor<32x196x384xf32>
    %v590 = stablehlo.reduce(%v589 init: %v582) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v591 = stablehlo.broadcast_in_dim %v590, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v592 = stablehlo.divide %v591, %v583 : tensor<32x196x384xf32>
    %v593 = stablehlo.add %v592, %v584 : tensor<32x196x384xf32>
    %v594 = stablehlo.rsqrt %v593 : tensor<32x196x384xf32>
    %v595 = stablehlo.multiply %v588, %v594 : tensor<32x196x384xf32>
    %v596 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v597 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v598 = stablehlo.multiply %v595, %v596 : tensor<32x196x384xf32>
    %v599 = stablehlo.add %v598, %v597 : tensor<32x196x384xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v602 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v603 = stablehlo.multiply %v601, %v602 : tensor<32x196x384xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v606 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v607 = stablehlo.add %v605, %v606 : tensor<32x196x384xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v609 = stablehlo.reshape %v608 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v610 = stablehlo.transpose %v609, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v613 = stablehlo.convolution(%v612, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v615 = stablehlo.add %v613, %v614 : tensor<32x1536x14x14xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v618 = stablehlo.multiply %v617, %v617 : tensor<32x1536x14x14xf32>
    %v619 = stablehlo.multiply %v618, %v617 : tensor<32x1536x14x14xf32>
    %v620 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v621 = stablehlo.multiply %v620, %v619 : tensor<32x1536x14x14xf32>
    %v622 = stablehlo.add %v617, %v621 : tensor<32x1536x14x14xf32>
    %v623 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v624 = stablehlo.multiply %v623, %v622 : tensor<32x1536x14x14xf32>
    %v625 = stablehlo.tanh %v624 : tensor<32x1536x14x14xf32>
    %v626 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v627 = stablehlo.add %v626, %v625 : tensor<32x1536x14x14xf32>
    %v628 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v629 = stablehlo.multiply %v628, %v617 : tensor<32x1536x14x14xf32>
    %v630 = stablehlo.multiply %v629, %v627 : tensor<32x1536x14x14xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v633 = stablehlo.convolution(%v632, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v634 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<32x384x14x14xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v638 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v639 = stablehlo.multiply %v637, %v638 : tensor<32x384x14x14xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v642 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v643 = stablehlo.multiply %v642, %v641 : tensor<32x384x14x14xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v646 = stablehlo.reshape %v572 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v647 = stablehlo.add %v645, %v646 : tensor<32x384x14x14xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v650 = stablehlo.convolution(%v649, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v652 = stablehlo.add %v650, %v651 : tensor<32x384x14x14xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v654 = stablehlo.reshape %v653 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v655 = stablehlo.transpose %v654, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v658 = stablehlo.constant dense<0.0> : tensor<f32>
    %v659 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v660 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v661 = stablehlo.reduce(%v657 init: %v658) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v662 = stablehlo.broadcast_in_dim %v661, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v663 = stablehlo.divide %v662, %v659 : tensor<32x196x384xf32>
    %v664 = stablehlo.subtract %v657, %v663 : tensor<32x196x384xf32>
    %v665 = stablehlo.multiply %v664, %v664 : tensor<32x196x384xf32>
    %v666 = stablehlo.reduce(%v665 init: %v658) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v667 = stablehlo.broadcast_in_dim %v666, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v668 = stablehlo.divide %v667, %v659 : tensor<32x196x384xf32>
    %v669 = stablehlo.add %v668, %v660 : tensor<32x196x384xf32>
    %v670 = stablehlo.rsqrt %v669 : tensor<32x196x384xf32>
    %v671 = stablehlo.multiply %v664, %v670 : tensor<32x196x384xf32>
    %v672 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v673 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v674 = stablehlo.multiply %v671, %v672 : tensor<32x196x384xf32>
    %v675 = stablehlo.add %v674, %v673 : tensor<32x196x384xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v678 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v679 = stablehlo.multiply %v677, %v678 : tensor<32x196x384xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v682 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v683 = stablehlo.add %v681, %v682 : tensor<32x196x384xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v686 = stablehlo.transpose %v685, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v689 = stablehlo.convolution(%v688, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v691 = stablehlo.add %v689, %v690 : tensor<32x1536x14x14xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v694 = stablehlo.multiply %v693, %v693 : tensor<32x1536x14x14xf32>
    %v695 = stablehlo.multiply %v694, %v693 : tensor<32x1536x14x14xf32>
    %v696 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v697 = stablehlo.multiply %v696, %v695 : tensor<32x1536x14x14xf32>
    %v698 = stablehlo.add %v693, %v697 : tensor<32x1536x14x14xf32>
    %v699 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v700 = stablehlo.multiply %v699, %v698 : tensor<32x1536x14x14xf32>
    %v701 = stablehlo.tanh %v700 : tensor<32x1536x14x14xf32>
    %v702 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v703 = stablehlo.add %v702, %v701 : tensor<32x1536x14x14xf32>
    %v704 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v705 = stablehlo.multiply %v704, %v693 : tensor<32x1536x14x14xf32>
    %v706 = stablehlo.multiply %v705, %v703 : tensor<32x1536x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v709 = stablehlo.convolution(%v708, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v710 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v711 = stablehlo.add %v709, %v710 : tensor<32x384x14x14xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v714 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v715 = stablehlo.multiply %v713, %v714 : tensor<32x384x14x14xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v718 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v719 = stablehlo.multiply %v718, %v717 : tensor<32x384x14x14xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v722 = stablehlo.reshape %v648 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v723 = stablehlo.add %v721, %v722 : tensor<32x384x14x14xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v726 = stablehlo.convolution(%v725, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v727 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v728 = stablehlo.add %v726, %v727 : tensor<32x384x14x14xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v731 = stablehlo.transpose %v730, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v735 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v736 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v737 = stablehlo.reduce(%v733 init: %v734) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v738 = stablehlo.broadcast_in_dim %v737, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v739 = stablehlo.divide %v738, %v735 : tensor<32x196x384xf32>
    %v740 = stablehlo.subtract %v733, %v739 : tensor<32x196x384xf32>
    %v741 = stablehlo.multiply %v740, %v740 : tensor<32x196x384xf32>
    %v742 = stablehlo.reduce(%v741 init: %v734) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v743 = stablehlo.broadcast_in_dim %v742, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v744 = stablehlo.divide %v743, %v735 : tensor<32x196x384xf32>
    %v745 = stablehlo.add %v744, %v736 : tensor<32x196x384xf32>
    %v746 = stablehlo.rsqrt %v745 : tensor<32x196x384xf32>
    %v747 = stablehlo.multiply %v740, %v746 : tensor<32x196x384xf32>
    %v748 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v749 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v750 = stablehlo.multiply %v747, %v748 : tensor<32x196x384xf32>
    %v751 = stablehlo.add %v750, %v749 : tensor<32x196x384xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v754 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v755 = stablehlo.multiply %v753, %v754 : tensor<32x196x384xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v758 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v759 = stablehlo.add %v757, %v758 : tensor<32x196x384xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v762 = stablehlo.transpose %v761, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v765 = stablehlo.convolution(%v764, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v766 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v767 = stablehlo.add %v765, %v766 : tensor<32x1536x14x14xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v770 = stablehlo.multiply %v769, %v769 : tensor<32x1536x14x14xf32>
    %v771 = stablehlo.multiply %v770, %v769 : tensor<32x1536x14x14xf32>
    %v772 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v773 = stablehlo.multiply %v772, %v771 : tensor<32x1536x14x14xf32>
    %v774 = stablehlo.add %v769, %v773 : tensor<32x1536x14x14xf32>
    %v775 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v776 = stablehlo.multiply %v775, %v774 : tensor<32x1536x14x14xf32>
    %v777 = stablehlo.tanh %v776 : tensor<32x1536x14x14xf32>
    %v778 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v779 = stablehlo.add %v778, %v777 : tensor<32x1536x14x14xf32>
    %v780 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v781 = stablehlo.multiply %v780, %v769 : tensor<32x1536x14x14xf32>
    %v782 = stablehlo.multiply %v781, %v779 : tensor<32x1536x14x14xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v785 = stablehlo.convolution(%v784, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v787 = stablehlo.add %v785, %v786 : tensor<32x384x14x14xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v790 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v791 = stablehlo.multiply %v789, %v790 : tensor<32x384x14x14xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %dp8, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v795 = stablehlo.multiply %v794, %v793 : tensor<32x384x14x14xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v798 = stablehlo.reshape %v724 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<32x384x14x14xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v802 = stablehlo.convolution(%v801, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v803 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v804 = stablehlo.add %v802, %v803 : tensor<32x384x14x14xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v807 = stablehlo.transpose %v806, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v811 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v812 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v813 = stablehlo.reduce(%v809 init: %v810) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v814 = stablehlo.broadcast_in_dim %v813, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v815 = stablehlo.divide %v814, %v811 : tensor<32x196x384xf32>
    %v816 = stablehlo.subtract %v809, %v815 : tensor<32x196x384xf32>
    %v817 = stablehlo.multiply %v816, %v816 : tensor<32x196x384xf32>
    %v818 = stablehlo.reduce(%v817 init: %v810) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v819 = stablehlo.broadcast_in_dim %v818, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v820 = stablehlo.divide %v819, %v811 : tensor<32x196x384xf32>
    %v821 = stablehlo.add %v820, %v812 : tensor<32x196x384xf32>
    %v822 = stablehlo.rsqrt %v821 : tensor<32x196x384xf32>
    %v823 = stablehlo.multiply %v816, %v822 : tensor<32x196x384xf32>
    %v824 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v825 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v826 = stablehlo.multiply %v823, %v824 : tensor<32x196x384xf32>
    %v827 = stablehlo.add %v826, %v825 : tensor<32x196x384xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v830 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v831 = stablehlo.multiply %v829, %v830 : tensor<32x196x384xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v834 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v835 = stablehlo.add %v833, %v834 : tensor<32x196x384xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v838 = stablehlo.transpose %v837, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v841 = stablehlo.convolution(%v840, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v842 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v843 = stablehlo.add %v841, %v842 : tensor<32x1536x14x14xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v846 = stablehlo.multiply %v845, %v845 : tensor<32x1536x14x14xf32>
    %v847 = stablehlo.multiply %v846, %v845 : tensor<32x1536x14x14xf32>
    %v848 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v849 = stablehlo.multiply %v848, %v847 : tensor<32x1536x14x14xf32>
    %v850 = stablehlo.add %v845, %v849 : tensor<32x1536x14x14xf32>
    %v851 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v852 = stablehlo.multiply %v851, %v850 : tensor<32x1536x14x14xf32>
    %v853 = stablehlo.tanh %v852 : tensor<32x1536x14x14xf32>
    %v854 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v855 = stablehlo.add %v854, %v853 : tensor<32x1536x14x14xf32>
    %v856 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v857 = stablehlo.multiply %v856, %v845 : tensor<32x1536x14x14xf32>
    %v858 = stablehlo.multiply %v857, %v855 : tensor<32x1536x14x14xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v861 = stablehlo.convolution(%v860, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v862 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v863 = stablehlo.add %v861, %v862 : tensor<32x384x14x14xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v866 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v867 = stablehlo.multiply %v865, %v866 : tensor<32x384x14x14xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v870 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v871 = stablehlo.multiply %v870, %v869 : tensor<32x384x14x14xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v874 = stablehlo.reshape %v800 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v875 = stablehlo.add %v873, %v874 : tensor<32x384x14x14xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v877 = stablehlo.reshape %v876 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v878 = stablehlo.convolution(%v877, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v879 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v880 = stablehlo.add %v878, %v879 : tensor<32x384x14x14xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v883 = stablehlo.transpose %v882, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v887 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v888 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v889 = stablehlo.reduce(%v885 init: %v886) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v890 = stablehlo.broadcast_in_dim %v889, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v891 = stablehlo.divide %v890, %v887 : tensor<32x196x384xf32>
    %v892 = stablehlo.subtract %v885, %v891 : tensor<32x196x384xf32>
    %v893 = stablehlo.multiply %v892, %v892 : tensor<32x196x384xf32>
    %v894 = stablehlo.reduce(%v893 init: %v886) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v895 = stablehlo.broadcast_in_dim %v894, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v896 = stablehlo.divide %v895, %v887 : tensor<32x196x384xf32>
    %v897 = stablehlo.add %v896, %v888 : tensor<32x196x384xf32>
    %v898 = stablehlo.rsqrt %v897 : tensor<32x196x384xf32>
    %v899 = stablehlo.multiply %v892, %v898 : tensor<32x196x384xf32>
    %v900 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v901 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v902 = stablehlo.multiply %v899, %v900 : tensor<32x196x384xf32>
    %v903 = stablehlo.add %v902, %v901 : tensor<32x196x384xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v906 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v907 = stablehlo.multiply %v905, %v906 : tensor<32x196x384xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v910 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v911 = stablehlo.add %v909, %v910 : tensor<32x196x384xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v914 = stablehlo.transpose %v913, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v917 = stablehlo.convolution(%v916, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v918 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v919 = stablehlo.add %v917, %v918 : tensor<32x1536x14x14xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v921 = stablehlo.reshape %v920 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v922 = stablehlo.multiply %v921, %v921 : tensor<32x1536x14x14xf32>
    %v923 = stablehlo.multiply %v922, %v921 : tensor<32x1536x14x14xf32>
    %v924 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v925 = stablehlo.multiply %v924, %v923 : tensor<32x1536x14x14xf32>
    %v926 = stablehlo.add %v921, %v925 : tensor<32x1536x14x14xf32>
    %v927 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v928 = stablehlo.multiply %v927, %v926 : tensor<32x1536x14x14xf32>
    %v929 = stablehlo.tanh %v928 : tensor<32x1536x14x14xf32>
    %v930 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v931 = stablehlo.add %v930, %v929 : tensor<32x1536x14x14xf32>
    %v932 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v933 = stablehlo.multiply %v932, %v921 : tensor<32x1536x14x14xf32>
    %v934 = stablehlo.multiply %v933, %v931 : tensor<32x1536x14x14xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v937 = stablehlo.convolution(%v936, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v938 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v939 = stablehlo.add %v937, %v938 : tensor<32x384x14x14xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v942 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v943 = stablehlo.multiply %v941, %v942 : tensor<32x384x14x14xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v946 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v947 = stablehlo.multiply %v946, %v945 : tensor<32x384x14x14xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v950 = stablehlo.reshape %v876 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v951 = stablehlo.add %v949, %v950 : tensor<32x384x14x14xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v954 = stablehlo.convolution(%v953, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v955 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v956 = stablehlo.add %v954, %v955 : tensor<32x384x14x14xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v959 = stablehlo.transpose %v958, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v963 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v964 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v965 = stablehlo.reduce(%v961 init: %v962) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v966 = stablehlo.broadcast_in_dim %v965, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v967 = stablehlo.divide %v966, %v963 : tensor<32x196x384xf32>
    %v968 = stablehlo.subtract %v961, %v967 : tensor<32x196x384xf32>
    %v969 = stablehlo.multiply %v968, %v968 : tensor<32x196x384xf32>
    %v970 = stablehlo.reduce(%v969 init: %v962) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v971 = stablehlo.broadcast_in_dim %v970, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v972 = stablehlo.divide %v971, %v963 : tensor<32x196x384xf32>
    %v973 = stablehlo.add %v972, %v964 : tensor<32x196x384xf32>
    %v974 = stablehlo.rsqrt %v973 : tensor<32x196x384xf32>
    %v975 = stablehlo.multiply %v968, %v974 : tensor<32x196x384xf32>
    %v976 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v977 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v978 = stablehlo.multiply %v975, %v976 : tensor<32x196x384xf32>
    %v979 = stablehlo.add %v978, %v977 : tensor<32x196x384xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v982 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v983 = stablehlo.multiply %v981, %v982 : tensor<32x196x384xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v986 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v987 = stablehlo.add %v985, %v986 : tensor<32x196x384xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v990 = stablehlo.transpose %v989, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v993 = stablehlo.convolution(%v992, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v994 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v995 = stablehlo.add %v993, %v994 : tensor<32x1536x14x14xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v997 = stablehlo.reshape %v996 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v998 = stablehlo.multiply %v997, %v997 : tensor<32x1536x14x14xf32>
    %v999 = stablehlo.multiply %v998, %v997 : tensor<32x1536x14x14xf32>
    %v1000 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v1001 = stablehlo.multiply %v1000, %v999 : tensor<32x1536x14x14xf32>
    %v1002 = stablehlo.add %v997, %v1001 : tensor<32x1536x14x14xf32>
    %v1003 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v1004 = stablehlo.multiply %v1003, %v1002 : tensor<32x1536x14x14xf32>
    %v1005 = stablehlo.tanh %v1004 : tensor<32x1536x14x14xf32>
    %v1006 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v1007 = stablehlo.add %v1006, %v1005 : tensor<32x1536x14x14xf32>
    %v1008 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1009 = stablehlo.multiply %v1008, %v997 : tensor<32x1536x14x14xf32>
    %v1010 = stablehlo.multiply %v1009, %v1007 : tensor<32x1536x14x14xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1013 = stablehlo.convolution(%v1012, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1014 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1015 = stablehlo.add %v1013, %v1014 : tensor<32x384x14x14xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1018 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1019 = stablehlo.multiply %v1017, %v1018 : tensor<32x384x14x14xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1021 = stablehlo.reshape %v1020 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1022 = stablehlo.broadcast_in_dim %dp11, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1023 = stablehlo.multiply %v1022, %v1021 : tensor<32x384x14x14xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1025 = stablehlo.reshape %v1024 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1026 = stablehlo.reshape %v952 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1027 = stablehlo.add %v1025, %v1026 : tensor<32x384x14x14xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1030 = stablehlo.convolution(%v1029, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1031 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1032 = stablehlo.add %v1030, %v1031 : tensor<32x384x14x14xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1035 = stablehlo.transpose %v1034, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1039 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1040 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1041 = stablehlo.reduce(%v1037 init: %v1038) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1042 = stablehlo.broadcast_in_dim %v1041, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1043 = stablehlo.divide %v1042, %v1039 : tensor<32x196x384xf32>
    %v1044 = stablehlo.subtract %v1037, %v1043 : tensor<32x196x384xf32>
    %v1045 = stablehlo.multiply %v1044, %v1044 : tensor<32x196x384xf32>
    %v1046 = stablehlo.reduce(%v1045 init: %v1038) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1047 = stablehlo.broadcast_in_dim %v1046, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1048 = stablehlo.divide %v1047, %v1039 : tensor<32x196x384xf32>
    %v1049 = stablehlo.add %v1048, %v1040 : tensor<32x196x384xf32>
    %v1050 = stablehlo.rsqrt %v1049 : tensor<32x196x384xf32>
    %v1051 = stablehlo.multiply %v1044, %v1050 : tensor<32x196x384xf32>
    %v1052 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1053 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1054 = stablehlo.multiply %v1051, %v1052 : tensor<32x196x384xf32>
    %v1055 = stablehlo.add %v1054, %v1053 : tensor<32x196x384xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1058 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1059 = stablehlo.multiply %v1057, %v1058 : tensor<32x196x384xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1062 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1063 = stablehlo.add %v1061, %v1062 : tensor<32x196x384xf32>
    %v1064 = stablehlo.reshape %v1063 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1066 = stablehlo.transpose %v1065, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1069 = stablehlo.convolution(%v1068, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1070 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1071 = stablehlo.add %v1069, %v1070 : tensor<32x1536x14x14xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1074 = stablehlo.multiply %v1073, %v1073 : tensor<32x1536x14x14xf32>
    %v1075 = stablehlo.multiply %v1074, %v1073 : tensor<32x1536x14x14xf32>
    %v1076 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v1077 = stablehlo.multiply %v1076, %v1075 : tensor<32x1536x14x14xf32>
    %v1078 = stablehlo.add %v1073, %v1077 : tensor<32x1536x14x14xf32>
    %v1079 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v1080 = stablehlo.multiply %v1079, %v1078 : tensor<32x1536x14x14xf32>
    %v1081 = stablehlo.tanh %v1080 : tensor<32x1536x14x14xf32>
    %v1082 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v1083 = stablehlo.add %v1082, %v1081 : tensor<32x1536x14x14xf32>
    %v1084 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1085 = stablehlo.multiply %v1084, %v1073 : tensor<32x1536x14x14xf32>
    %v1086 = stablehlo.multiply %v1085, %v1083 : tensor<32x1536x14x14xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1089 = stablehlo.convolution(%v1088, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1090 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1091 = stablehlo.add %v1089, %v1090 : tensor<32x384x14x14xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1094 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1095 = stablehlo.multiply %v1093, %v1094 : tensor<32x384x14x14xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1098 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1099 = stablehlo.multiply %v1098, %v1097 : tensor<32x384x14x14xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1102 = stablehlo.reshape %v1028 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1103 = stablehlo.add %v1101, %v1102 : tensor<32x384x14x14xf32>
    %v1104 = stablehlo.reshape %v1103 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1106 = stablehlo.convolution(%v1105, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1107 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1108 = stablehlo.add %v1106, %v1107 : tensor<32x384x14x14xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1110 = stablehlo.reshape %v1109 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1111 = stablehlo.transpose %v1110, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1113 = stablehlo.reshape %v1112 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1115 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1116 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1117 = stablehlo.reduce(%v1113 init: %v1114) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1118 = stablehlo.broadcast_in_dim %v1117, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1119 = stablehlo.divide %v1118, %v1115 : tensor<32x196x384xf32>
    %v1120 = stablehlo.subtract %v1113, %v1119 : tensor<32x196x384xf32>
    %v1121 = stablehlo.multiply %v1120, %v1120 : tensor<32x196x384xf32>
    %v1122 = stablehlo.reduce(%v1121 init: %v1114) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1123 = stablehlo.broadcast_in_dim %v1122, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1124 = stablehlo.divide %v1123, %v1115 : tensor<32x196x384xf32>
    %v1125 = stablehlo.add %v1124, %v1116 : tensor<32x196x384xf32>
    %v1126 = stablehlo.rsqrt %v1125 : tensor<32x196x384xf32>
    %v1127 = stablehlo.multiply %v1120, %v1126 : tensor<32x196x384xf32>
    %v1128 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1129 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1130 = stablehlo.multiply %v1127, %v1128 : tensor<32x196x384xf32>
    %v1131 = stablehlo.add %v1130, %v1129 : tensor<32x196x384xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1134 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1135 = stablehlo.multiply %v1133, %v1134 : tensor<32x196x384xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1138 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1139 = stablehlo.add %v1137, %v1138 : tensor<32x196x384xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1142 = stablehlo.transpose %v1141, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1145 = stablehlo.convolution(%v1144, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1146 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1147 = stablehlo.add %v1145, %v1146 : tensor<32x1536x14x14xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1150 = stablehlo.multiply %v1149, %v1149 : tensor<32x1536x14x14xf32>
    %v1151 = stablehlo.multiply %v1150, %v1149 : tensor<32x1536x14x14xf32>
    %v1152 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v1153 = stablehlo.multiply %v1152, %v1151 : tensor<32x1536x14x14xf32>
    %v1154 = stablehlo.add %v1149, %v1153 : tensor<32x1536x14x14xf32>
    %v1155 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v1156 = stablehlo.multiply %v1155, %v1154 : tensor<32x1536x14x14xf32>
    %v1157 = stablehlo.tanh %v1156 : tensor<32x1536x14x14xf32>
    %v1158 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v1159 = stablehlo.add %v1158, %v1157 : tensor<32x1536x14x14xf32>
    %v1160 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1161 = stablehlo.multiply %v1160, %v1149 : tensor<32x1536x14x14xf32>
    %v1162 = stablehlo.multiply %v1161, %v1159 : tensor<32x1536x14x14xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1165 = stablehlo.convolution(%v1164, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1166 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1167 = stablehlo.add %v1165, %v1166 : tensor<32x384x14x14xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1170 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1171 = stablehlo.multiply %v1169, %v1170 : tensor<32x384x14x14xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1174 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1175 = stablehlo.multiply %v1174, %v1173 : tensor<32x384x14x14xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1178 = stablehlo.reshape %v1104 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1179 = stablehlo.add %v1177, %v1178 : tensor<32x384x14x14xf32>
    %v1180 = stablehlo.reshape %v1179 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1181 = stablehlo.reshape %v1180 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1182 = stablehlo.convolution(%v1181, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1183 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1184 = stablehlo.add %v1182, %v1183 : tensor<32x384x14x14xf32>
    %v1185 = stablehlo.reshape %v1184 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1186 = stablehlo.reshape %v1185 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1187 = stablehlo.transpose %v1186, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1191 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1192 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1193 = stablehlo.reduce(%v1189 init: %v1190) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1194 = stablehlo.broadcast_in_dim %v1193, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1195 = stablehlo.divide %v1194, %v1191 : tensor<32x196x384xf32>
    %v1196 = stablehlo.subtract %v1189, %v1195 : tensor<32x196x384xf32>
    %v1197 = stablehlo.multiply %v1196, %v1196 : tensor<32x196x384xf32>
    %v1198 = stablehlo.reduce(%v1197 init: %v1190) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1199 = stablehlo.broadcast_in_dim %v1198, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1200 = stablehlo.divide %v1199, %v1191 : tensor<32x196x384xf32>
    %v1201 = stablehlo.add %v1200, %v1192 : tensor<32x196x384xf32>
    %v1202 = stablehlo.rsqrt %v1201 : tensor<32x196x384xf32>
    %v1203 = stablehlo.multiply %v1196, %v1202 : tensor<32x196x384xf32>
    %v1204 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1205 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1206 = stablehlo.multiply %v1203, %v1204 : tensor<32x196x384xf32>
    %v1207 = stablehlo.add %v1206, %v1205 : tensor<32x196x384xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1210 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1211 = stablehlo.multiply %v1209, %v1210 : tensor<32x196x384xf32>
    %v1212 = stablehlo.reshape %v1211 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1213 = stablehlo.reshape %v1212 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1214 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1215 = stablehlo.add %v1213, %v1214 : tensor<32x196x384xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1218 = stablehlo.transpose %v1217, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1221 = stablehlo.convolution(%v1220, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1222 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1223 = stablehlo.add %v1221, %v1222 : tensor<32x1536x14x14xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1226 = stablehlo.multiply %v1225, %v1225 : tensor<32x1536x14x14xf32>
    %v1227 = stablehlo.multiply %v1226, %v1225 : tensor<32x1536x14x14xf32>
    %v1228 = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %v1229 = stablehlo.multiply %v1228, %v1227 : tensor<32x1536x14x14xf32>
    %v1230 = stablehlo.add %v1225, %v1229 : tensor<32x1536x14x14xf32>
    %v1231 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %v1232 = stablehlo.multiply %v1231, %v1230 : tensor<32x1536x14x14xf32>
    %v1233 = stablehlo.tanh %v1232 : tensor<32x1536x14x14xf32>
    %v1234 = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %v1235 = stablehlo.add %v1234, %v1233 : tensor<32x1536x14x14xf32>
    %v1236 = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %v1237 = stablehlo.multiply %v1236, %v1225 : tensor<32x1536x14x14xf32>
    %v1238 = stablehlo.multiply %v1237, %v1235 : tensor<32x1536x14x14xf32>
    %v1239 = stablehlo.reshape %v1238 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1241 = stablehlo.convolution(%v1240, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1242 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1243 = stablehlo.add %v1241, %v1242 : tensor<32x384x14x14xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1246 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1247 = stablehlo.multiply %v1245, %v1246 : tensor<32x384x14x14xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1250 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x384x14x14xf32>
    %v1251 = stablehlo.multiply %v1250, %v1249 : tensor<32x384x14x14xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1254 = stablehlo.reshape %v1180 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1255 = stablehlo.add %v1253, %v1254 : tensor<32x384x14x14xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1258 = stablehlo.transpose %v1257, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1262 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1263 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1264 = stablehlo.reduce(%v1260 init: %v1261) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1265 = stablehlo.broadcast_in_dim %v1264, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1266 = stablehlo.divide %v1265, %v1262 : tensor<32x196x384xf32>
    %v1267 = stablehlo.subtract %v1260, %v1266 : tensor<32x196x384xf32>
    %v1268 = stablehlo.multiply %v1267, %v1267 : tensor<32x196x384xf32>
    %v1269 = stablehlo.reduce(%v1268 init: %v1261) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1270 = stablehlo.broadcast_in_dim %v1269, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1271 = stablehlo.divide %v1270, %v1262 : tensor<32x196x384xf32>
    %v1272 = stablehlo.add %v1271, %v1263 : tensor<32x196x384xf32>
    %v1273 = stablehlo.rsqrt %v1272 : tensor<32x196x384xf32>
    %v1274 = stablehlo.multiply %v1267, %v1273 : tensor<32x196x384xf32>
    %v1275 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1276 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1277 = stablehlo.multiply %v1274, %v1275 : tensor<32x196x384xf32>
    %v1278 = stablehlo.add %v1277, %v1276 : tensor<32x196x384xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1281 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1282 = stablehlo.multiply %v1280, %v1281 : tensor<32x196x384xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1285 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1286 = stablehlo.add %v1284, %v1285 : tensor<32x196x384xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1289 = stablehlo.transpose %v1288, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1292 = stablehlo.convolution(%v1291, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %v1293 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1294 = stablehlo.add %v1292, %v1293 : tensor<32x768x7x7xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1297 = stablehlo.convolution(%v1296, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1298 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1299 = stablehlo.add %v1297, %v1298 : tensor<32x768x7x7xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1302 = stablehlo.transpose %v1301, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1305 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1306 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1307 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1308 = stablehlo.reduce(%v1304 init: %v1305) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1309 = stablehlo.broadcast_in_dim %v1308, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1310 = stablehlo.divide %v1309, %v1306 : tensor<32x49x768xf32>
    %v1311 = stablehlo.subtract %v1304, %v1310 : tensor<32x49x768xf32>
    %v1312 = stablehlo.multiply %v1311, %v1311 : tensor<32x49x768xf32>
    %v1313 = stablehlo.reduce(%v1312 init: %v1305) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1314 = stablehlo.broadcast_in_dim %v1313, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1315 = stablehlo.divide %v1314, %v1306 : tensor<32x49x768xf32>
    %v1316 = stablehlo.add %v1315, %v1307 : tensor<32x49x768xf32>
    %v1317 = stablehlo.rsqrt %v1316 : tensor<32x49x768xf32>
    %v1318 = stablehlo.multiply %v1311, %v1317 : tensor<32x49x768xf32>
    %v1319 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1320 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1321 = stablehlo.multiply %v1318, %v1319 : tensor<32x49x768xf32>
    %v1322 = stablehlo.add %v1321, %v1320 : tensor<32x49x768xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1325 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1326 = stablehlo.multiply %v1324, %v1325 : tensor<32x49x768xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1329 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1330 = stablehlo.add %v1328, %v1329 : tensor<32x49x768xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1333 = stablehlo.transpose %v1332, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1334 = stablehlo.reshape %v1333 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1336 = stablehlo.convolution(%v1335, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1337 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1338 = stablehlo.add %v1336, %v1337 : tensor<32x3072x7x7xf32>
    %v1339 = stablehlo.reshape %v1338 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1341 = stablehlo.multiply %v1340, %v1340 : tensor<32x3072x7x7xf32>
    %v1342 = stablehlo.multiply %v1341, %v1340 : tensor<32x3072x7x7xf32>
    %v1343 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1344 = stablehlo.multiply %v1343, %v1342 : tensor<32x3072x7x7xf32>
    %v1345 = stablehlo.add %v1340, %v1344 : tensor<32x3072x7x7xf32>
    %v1346 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1347 = stablehlo.multiply %v1346, %v1345 : tensor<32x3072x7x7xf32>
    %v1348 = stablehlo.tanh %v1347 : tensor<32x3072x7x7xf32>
    %v1349 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1350 = stablehlo.add %v1349, %v1348 : tensor<32x3072x7x7xf32>
    %v1351 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1352 = stablehlo.multiply %v1351, %v1340 : tensor<32x3072x7x7xf32>
    %v1353 = stablehlo.multiply %v1352, %v1350 : tensor<32x3072x7x7xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1356 = stablehlo.convolution(%v1355, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1357 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1358 = stablehlo.add %v1356, %v1357 : tensor<32x768x7x7xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1361 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1362 = stablehlo.multiply %v1360, %v1361 : tensor<32x768x7x7xf32>
    %v1363 = stablehlo.reshape %v1362 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1365 = stablehlo.broadcast_in_dim %dp15, dims = [0] : (tensor<32xf32>) -> tensor<32x768x7x7xf32>
    %v1366 = stablehlo.multiply %v1365, %v1364 : tensor<32x768x7x7xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1369 = stablehlo.reshape %v1295 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1370 = stablehlo.add %v1368, %v1369 : tensor<32x768x7x7xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1373 = stablehlo.convolution(%v1372, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1374 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1375 = stablehlo.add %v1373, %v1374 : tensor<32x768x7x7xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1378 = stablehlo.transpose %v1377, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1382 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1383 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1384 = stablehlo.reduce(%v1380 init: %v1381) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1385 = stablehlo.broadcast_in_dim %v1384, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1386 = stablehlo.divide %v1385, %v1382 : tensor<32x49x768xf32>
    %v1387 = stablehlo.subtract %v1380, %v1386 : tensor<32x49x768xf32>
    %v1388 = stablehlo.multiply %v1387, %v1387 : tensor<32x49x768xf32>
    %v1389 = stablehlo.reduce(%v1388 init: %v1381) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1390 = stablehlo.broadcast_in_dim %v1389, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1391 = stablehlo.divide %v1390, %v1382 : tensor<32x49x768xf32>
    %v1392 = stablehlo.add %v1391, %v1383 : tensor<32x49x768xf32>
    %v1393 = stablehlo.rsqrt %v1392 : tensor<32x49x768xf32>
    %v1394 = stablehlo.multiply %v1387, %v1393 : tensor<32x49x768xf32>
    %v1395 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1396 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1397 = stablehlo.multiply %v1394, %v1395 : tensor<32x49x768xf32>
    %v1398 = stablehlo.add %v1397, %v1396 : tensor<32x49x768xf32>
    %v1399 = stablehlo.reshape %v1398 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1401 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1402 = stablehlo.multiply %v1400, %v1401 : tensor<32x49x768xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1404 = stablehlo.reshape %v1403 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1405 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1406 = stablehlo.add %v1404, %v1405 : tensor<32x49x768xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1409 = stablehlo.transpose %v1408, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1410 = stablehlo.reshape %v1409 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1412 = stablehlo.convolution(%v1411, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1413 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1414 = stablehlo.add %v1412, %v1413 : tensor<32x3072x7x7xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1417 = stablehlo.multiply %v1416, %v1416 : tensor<32x3072x7x7xf32>
    %v1418 = stablehlo.multiply %v1417, %v1416 : tensor<32x3072x7x7xf32>
    %v1419 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1420 = stablehlo.multiply %v1419, %v1418 : tensor<32x3072x7x7xf32>
    %v1421 = stablehlo.add %v1416, %v1420 : tensor<32x3072x7x7xf32>
    %v1422 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1423 = stablehlo.multiply %v1422, %v1421 : tensor<32x3072x7x7xf32>
    %v1424 = stablehlo.tanh %v1423 : tensor<32x3072x7x7xf32>
    %v1425 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1426 = stablehlo.add %v1425, %v1424 : tensor<32x3072x7x7xf32>
    %v1427 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1428 = stablehlo.multiply %v1427, %v1416 : tensor<32x3072x7x7xf32>
    %v1429 = stablehlo.multiply %v1428, %v1426 : tensor<32x3072x7x7xf32>
    %v1430 = stablehlo.reshape %v1429 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1432 = stablehlo.convolution(%v1431, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1433 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1434 = stablehlo.add %v1432, %v1433 : tensor<32x768x7x7xf32>
    %v1435 = stablehlo.reshape %v1434 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1437 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1438 = stablehlo.multiply %v1436, %v1437 : tensor<32x768x7x7xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1440 = stablehlo.reshape %v1439 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1441 = stablehlo.broadcast_in_dim %dp16, dims = [0] : (tensor<32xf32>) -> tensor<32x768x7x7xf32>
    %v1442 = stablehlo.multiply %v1441, %v1440 : tensor<32x768x7x7xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1445 = stablehlo.reshape %v1371 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1446 = stablehlo.add %v1444, %v1445 : tensor<32x768x7x7xf32>
    %v1447 = stablehlo.reshape %v1446 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1449 = stablehlo.convolution(%v1448, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1450 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1451 = stablehlo.add %v1449, %v1450 : tensor<32x768x7x7xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1454 = stablehlo.transpose %v1453, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1458 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1459 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1460 = stablehlo.reduce(%v1456 init: %v1457) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1461 = stablehlo.broadcast_in_dim %v1460, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1462 = stablehlo.divide %v1461, %v1458 : tensor<32x49x768xf32>
    %v1463 = stablehlo.subtract %v1456, %v1462 : tensor<32x49x768xf32>
    %v1464 = stablehlo.multiply %v1463, %v1463 : tensor<32x49x768xf32>
    %v1465 = stablehlo.reduce(%v1464 init: %v1457) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1466 = stablehlo.broadcast_in_dim %v1465, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1467 = stablehlo.divide %v1466, %v1458 : tensor<32x49x768xf32>
    %v1468 = stablehlo.add %v1467, %v1459 : tensor<32x49x768xf32>
    %v1469 = stablehlo.rsqrt %v1468 : tensor<32x49x768xf32>
    %v1470 = stablehlo.multiply %v1463, %v1469 : tensor<32x49x768xf32>
    %v1471 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1472 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1473 = stablehlo.multiply %v1470, %v1471 : tensor<32x49x768xf32>
    %v1474 = stablehlo.add %v1473, %v1472 : tensor<32x49x768xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1476 = stablehlo.reshape %v1475 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1477 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1478 = stablehlo.multiply %v1476, %v1477 : tensor<32x49x768xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1481 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1482 = stablehlo.add %v1480, %v1481 : tensor<32x49x768xf32>
    %v1483 = stablehlo.reshape %v1482 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1485 = stablehlo.transpose %v1484, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1486 = stablehlo.reshape %v1485 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1488 = stablehlo.convolution(%v1487, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1489 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1490 = stablehlo.add %v1488, %v1489 : tensor<32x3072x7x7xf32>
    %v1491 = stablehlo.reshape %v1490 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1493 = stablehlo.multiply %v1492, %v1492 : tensor<32x3072x7x7xf32>
    %v1494 = stablehlo.multiply %v1493, %v1492 : tensor<32x3072x7x7xf32>
    %v1495 = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %v1496 = stablehlo.multiply %v1495, %v1494 : tensor<32x3072x7x7xf32>
    %v1497 = stablehlo.add %v1492, %v1496 : tensor<32x3072x7x7xf32>
    %v1498 = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %v1499 = stablehlo.multiply %v1498, %v1497 : tensor<32x3072x7x7xf32>
    %v1500 = stablehlo.tanh %v1499 : tensor<32x3072x7x7xf32>
    %v1501 = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %v1502 = stablehlo.add %v1501, %v1500 : tensor<32x3072x7x7xf32>
    %v1503 = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %v1504 = stablehlo.multiply %v1503, %v1492 : tensor<32x3072x7x7xf32>
    %v1505 = stablehlo.multiply %v1504, %v1502 : tensor<32x3072x7x7xf32>
    %v1506 = stablehlo.reshape %v1505 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1507 = stablehlo.reshape %v1506 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1508 = stablehlo.convolution(%v1507, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1509 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1510 = stablehlo.add %v1508, %v1509 : tensor<32x768x7x7xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1512 = stablehlo.reshape %v1511 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1513 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1514 = stablehlo.multiply %v1512, %v1513 : tensor<32x768x7x7xf32>
    %v1515 = stablehlo.reshape %v1514 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1516 = stablehlo.reshape %v1515 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1517 = stablehlo.broadcast_in_dim %dp17, dims = [0] : (tensor<32xf32>) -> tensor<32x768x7x7xf32>
    %v1518 = stablehlo.multiply %v1517, %v1516 : tensor<32x768x7x7xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1520 = stablehlo.reshape %v1519 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1521 = stablehlo.reshape %v1447 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1522 = stablehlo.add %v1520, %v1521 : tensor<32x768x7x7xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1526 = stablehlo.reduce(%v1524 init: %v1525) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %v1527 = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %v1528 = stablehlo.divide %v1526, %v1527 : tensor<32x768xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1531 = stablehlo.constant dense<768.0> : tensor<32x1x768xf32>
    %v1532 = stablehlo.constant dense<1.0e-6> : tensor<32x1x768xf32>
    %v1533 = stablehlo.reduce(%v1529 init: %v1530) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1534 = stablehlo.broadcast_in_dim %v1533, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v1535 = stablehlo.divide %v1534, %v1531 : tensor<32x1x768xf32>
    %v1536 = stablehlo.subtract %v1529, %v1535 : tensor<32x1x768xf32>
    %v1537 = stablehlo.multiply %v1536, %v1536 : tensor<32x1x768xf32>
    %v1538 = stablehlo.reduce(%v1537 init: %v1530) applies stablehlo.add across dimensions = [2] : (tensor<32x1x768xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1539 = stablehlo.broadcast_in_dim %v1538, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x768xf32>
    %v1540 = stablehlo.divide %v1539, %v1531 : tensor<32x1x768xf32>
    %v1541 = stablehlo.add %v1540, %v1532 : tensor<32x1x768xf32>
    %v1542 = stablehlo.rsqrt %v1541 : tensor<32x1x768xf32>
    %v1543 = stablehlo.multiply %v1536, %v1542 : tensor<32x1x768xf32>
    %v1544 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x1x768xf32>
    %v1545 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x1x768xf32>
    %v1546 = stablehlo.multiply %v1543, %v1544 : tensor<32x1x768xf32>
    %v1547 = stablehlo.add %v1546, %v1545 : tensor<32x1x768xf32>
    %v1548 = stablehlo.reshape %v1547 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1550 = stablehlo.broadcast_in_dim %hng, dims = [2] : (tensor<768xf32>) -> tensor<32x1x768xf32>
    %v1551 = stablehlo.multiply %v1549, %v1550 : tensor<32x1x768xf32>
    %v1552 = stablehlo.reshape %v1551 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v1553 = stablehlo.reshape %v1552 : (tensor<32x768xf32>) -> tensor<32x1x768xf32>
    %v1554 = stablehlo.broadcast_in_dim %hnbt, dims = [2] : (tensor<768xf32>) -> tensor<32x1x768xf32>
    %v1555 = stablehlo.add %v1553, %v1554 : tensor<32x1x768xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v1557 = stablehlo.dot_general %v1556, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x10xf32>) -> tensor<32x10xf32>
    %v1558 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1559 = stablehlo.add %v1557, %v1558 : tensor<32x10xf32>
    return %v1559 : tensor<32x10xf32>
  }
}
