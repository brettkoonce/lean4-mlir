module @m {
  func.func @convnextsin_erf_fwd(%x: tensor<64x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %psng: tensor<96xf32>, %psnbt: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<96xf32>, %s0b0nbt: tensor<96xf32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<96xf32>, %s0b1nbt: tensor<96xf32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<96xf32>, %s0b2nbt: tensor<96xf32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<96xf32>, %d0nbt: tensor<96xf32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<192xf32>, %s1b0nbt: tensor<192xf32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<192xf32>, %s1b1nbt: tensor<192xf32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<192xf32>, %s1b2nbt: tensor<192xf32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<192xf32>, %d1nbt: tensor<192xf32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<384xf32>, %s2b0nbt: tensor<384xf32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<384xf32>, %s2b1nbt: tensor<384xf32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<384xf32>, %s2b2nbt: tensor<384xf32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<384xf32>, %s2b3nbt: tensor<384xf32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<384xf32>, %s2b4nbt: tensor<384xf32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<384xf32>, %s2b5nbt: tensor<384xf32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<384xf32>, %s2b6nbt: tensor<384xf32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<384xf32>, %s2b7nbt: tensor<384xf32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<384xf32>, %s2b8nbt: tensor<384xf32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %s2b9dW: tensor<384x1x7x7xf32>, %s2b9db: tensor<384xf32>, %s2b9ng: tensor<384xf32>, %s2b9nbt: tensor<384xf32>, %s2b9eW: tensor<1536x384x1x1xf32>, %s2b9eb: tensor<1536xf32>, %s2b9pW: tensor<384x1536x1x1xf32>, %s2b9pb: tensor<384xf32>, %s2b9lg: tensor<384xf32>, %s2b10dW: tensor<384x1x7x7xf32>, %s2b10db: tensor<384xf32>, %s2b10ng: tensor<384xf32>, %s2b10nbt: tensor<384xf32>, %s2b10eW: tensor<1536x384x1x1xf32>, %s2b10eb: tensor<1536xf32>, %s2b10pW: tensor<384x1536x1x1xf32>, %s2b10pb: tensor<384xf32>, %s2b10lg: tensor<384xf32>, %s2b11dW: tensor<384x1x7x7xf32>, %s2b11db: tensor<384xf32>, %s2b11ng: tensor<384xf32>, %s2b11nbt: tensor<384xf32>, %s2b11eW: tensor<1536x384x1x1xf32>, %s2b11eb: tensor<1536xf32>, %s2b11pW: tensor<384x1536x1x1xf32>, %s2b11pb: tensor<384xf32>, %s2b11lg: tensor<384xf32>, %s2b12dW: tensor<384x1x7x7xf32>, %s2b12db: tensor<384xf32>, %s2b12ng: tensor<384xf32>, %s2b12nbt: tensor<384xf32>, %s2b12eW: tensor<1536x384x1x1xf32>, %s2b12eb: tensor<1536xf32>, %s2b12pW: tensor<384x1536x1x1xf32>, %s2b12pb: tensor<384xf32>, %s2b12lg: tensor<384xf32>, %s2b13dW: tensor<384x1x7x7xf32>, %s2b13db: tensor<384xf32>, %s2b13ng: tensor<384xf32>, %s2b13nbt: tensor<384xf32>, %s2b13eW: tensor<1536x384x1x1xf32>, %s2b13eb: tensor<1536xf32>, %s2b13pW: tensor<384x1536x1x1xf32>, %s2b13pb: tensor<384xf32>, %s2b13lg: tensor<384xf32>, %s2b14dW: tensor<384x1x7x7xf32>, %s2b14db: tensor<384xf32>, %s2b14ng: tensor<384xf32>, %s2b14nbt: tensor<384xf32>, %s2b14eW: tensor<1536x384x1x1xf32>, %s2b14eb: tensor<1536xf32>, %s2b14pW: tensor<384x1536x1x1xf32>, %s2b14pb: tensor<384xf32>, %s2b14lg: tensor<384xf32>, %s2b15dW: tensor<384x1x7x7xf32>, %s2b15db: tensor<384xf32>, %s2b15ng: tensor<384xf32>, %s2b15nbt: tensor<384xf32>, %s2b15eW: tensor<1536x384x1x1xf32>, %s2b15eb: tensor<1536xf32>, %s2b15pW: tensor<384x1536x1x1xf32>, %s2b15pb: tensor<384xf32>, %s2b15lg: tensor<384xf32>, %s2b16dW: tensor<384x1x7x7xf32>, %s2b16db: tensor<384xf32>, %s2b16ng: tensor<384xf32>, %s2b16nbt: tensor<384xf32>, %s2b16eW: tensor<1536x384x1x1xf32>, %s2b16eb: tensor<1536xf32>, %s2b16pW: tensor<384x1536x1x1xf32>, %s2b16pb: tensor<384xf32>, %s2b16lg: tensor<384xf32>, %s2b17dW: tensor<384x1x7x7xf32>, %s2b17db: tensor<384xf32>, %s2b17ng: tensor<384xf32>, %s2b17nbt: tensor<384xf32>, %s2b17eW: tensor<1536x384x1x1xf32>, %s2b17eb: tensor<1536xf32>, %s2b17pW: tensor<384x1536x1x1xf32>, %s2b17pb: tensor<384xf32>, %s2b17lg: tensor<384xf32>, %s2b18dW: tensor<384x1x7x7xf32>, %s2b18db: tensor<384xf32>, %s2b18ng: tensor<384xf32>, %s2b18nbt: tensor<384xf32>, %s2b18eW: tensor<1536x384x1x1xf32>, %s2b18eb: tensor<1536xf32>, %s2b18pW: tensor<384x1536x1x1xf32>, %s2b18pb: tensor<384xf32>, %s2b18lg: tensor<384xf32>, %s2b19dW: tensor<384x1x7x7xf32>, %s2b19db: tensor<384xf32>, %s2b19ng: tensor<384xf32>, %s2b19nbt: tensor<384xf32>, %s2b19eW: tensor<1536x384x1x1xf32>, %s2b19eb: tensor<1536xf32>, %s2b19pW: tensor<384x1536x1x1xf32>, %s2b19pb: tensor<384xf32>, %s2b19lg: tensor<384xf32>, %s2b20dW: tensor<384x1x7x7xf32>, %s2b20db: tensor<384xf32>, %s2b20ng: tensor<384xf32>, %s2b20nbt: tensor<384xf32>, %s2b20eW: tensor<1536x384x1x1xf32>, %s2b20eb: tensor<1536xf32>, %s2b20pW: tensor<384x1536x1x1xf32>, %s2b20pb: tensor<384xf32>, %s2b20lg: tensor<384xf32>, %s2b21dW: tensor<384x1x7x7xf32>, %s2b21db: tensor<384xf32>, %s2b21ng: tensor<384xf32>, %s2b21nbt: tensor<384xf32>, %s2b21eW: tensor<1536x384x1x1xf32>, %s2b21eb: tensor<1536xf32>, %s2b21pW: tensor<384x1536x1x1xf32>, %s2b21pb: tensor<384xf32>, %s2b21lg: tensor<384xf32>, %s2b22dW: tensor<384x1x7x7xf32>, %s2b22db: tensor<384xf32>, %s2b22ng: tensor<384xf32>, %s2b22nbt: tensor<384xf32>, %s2b22eW: tensor<1536x384x1x1xf32>, %s2b22eb: tensor<1536xf32>, %s2b22pW: tensor<384x1536x1x1xf32>, %s2b22pb: tensor<384xf32>, %s2b22lg: tensor<384xf32>, %s2b23dW: tensor<384x1x7x7xf32>, %s2b23db: tensor<384xf32>, %s2b23ng: tensor<384xf32>, %s2b23nbt: tensor<384xf32>, %s2b23eW: tensor<1536x384x1x1xf32>, %s2b23eb: tensor<1536xf32>, %s2b23pW: tensor<384x1536x1x1xf32>, %s2b23pb: tensor<384xf32>, %s2b23lg: tensor<384xf32>, %s2b24dW: tensor<384x1x7x7xf32>, %s2b24db: tensor<384xf32>, %s2b24ng: tensor<384xf32>, %s2b24nbt: tensor<384xf32>, %s2b24eW: tensor<1536x384x1x1xf32>, %s2b24eb: tensor<1536xf32>, %s2b24pW: tensor<384x1536x1x1xf32>, %s2b24pb: tensor<384xf32>, %s2b24lg: tensor<384xf32>, %s2b25dW: tensor<384x1x7x7xf32>, %s2b25db: tensor<384xf32>, %s2b25ng: tensor<384xf32>, %s2b25nbt: tensor<384xf32>, %s2b25eW: tensor<1536x384x1x1xf32>, %s2b25eb: tensor<1536xf32>, %s2b25pW: tensor<384x1536x1x1xf32>, %s2b25pb: tensor<384xf32>, %s2b25lg: tensor<384xf32>, %s2b26dW: tensor<384x1x7x7xf32>, %s2b26db: tensor<384xf32>, %s2b26ng: tensor<384xf32>, %s2b26nbt: tensor<384xf32>, %s2b26eW: tensor<1536x384x1x1xf32>, %s2b26eb: tensor<1536xf32>, %s2b26pW: tensor<384x1536x1x1xf32>, %s2b26pb: tensor<384xf32>, %s2b26lg: tensor<384xf32>, %d2ng: tensor<384xf32>, %d2nbt: tensor<384xf32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<768xf32>, %s3b0nbt: tensor<768xf32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<768xf32>, %s3b1nbt: tensor<768xf32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<768xf32>, %s3b2nbt: tensor<768xf32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<768xf32>, %hnbt: tensor<768xf32>, %Wd: tensor<768x1000xf32>, %bd: tensor<1000xf32>) -> tensor<64x1000xf32> {
    // ── ConvNeXt-S forward: every op is pretty(verified AST node) except the %one/%zero LayerNorm constants ──
    // The channel-LN chain normalises with lnRowF at γ=1/β=0 and applies the REAL
    // per-channel affine with rowScaleF/rowBiasF, so these two are its scalar identities.
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %psW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3x224x224xf32>, tensor<96x3x4x4xf32>) -> tensor<64x96x56x56xf32>
    %v2 = stablehlo.broadcast_in_dim %psb, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<64x96x56x56xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<64x301056xf32>) -> tensor<64x96x3136xf32>
    %v6 = stablehlo.transpose %v5, dims = [0, 2, 1] : (tensor<64x96x3136xf32>) -> tensor<64x3136x96xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v9 = stablehlo.constant dense<0.0> : tensor<f32>
    %v10 = stablehlo.constant dense<96.0> : tensor<64x3136x96xf32>
    %v11 = stablehlo.constant dense<1.0e-6> : tensor<64x3136x96xf32>
    %v12 = stablehlo.reduce(%v8 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v13 = stablehlo.broadcast_in_dim %v12, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v14 = stablehlo.divide %v13, %v10 : tensor<64x3136x96xf32>
    %v15 = stablehlo.subtract %v8, %v14 : tensor<64x3136x96xf32>
    %v16 = stablehlo.multiply %v15, %v15 : tensor<64x3136x96xf32>
    %v17 = stablehlo.reduce(%v16 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v18 = stablehlo.broadcast_in_dim %v17, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v19 = stablehlo.divide %v18, %v10 : tensor<64x3136x96xf32>
    %v20 = stablehlo.add %v19, %v11 : tensor<64x3136x96xf32>
    %v21 = stablehlo.rsqrt %v20 : tensor<64x3136x96xf32>
    %v22 = stablehlo.multiply %v15, %v21 : tensor<64x3136x96xf32>
    %v23 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v24 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v25 = stablehlo.multiply %v22, %v23 : tensor<64x3136x96xf32>
    %v26 = stablehlo.add %v25, %v24 : tensor<64x3136x96xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v29 = stablehlo.broadcast_in_dim %psng, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v30 = stablehlo.multiply %v28, %v29 : tensor<64x3136x96xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v33 = stablehlo.broadcast_in_dim %psnbt, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<64x3136x96xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v37 = stablehlo.transpose %v36, dims = [0, 2, 1] : (tensor<64x3136x96xf32>) -> tensor<64x96x3136xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<64x96x3136xf32>) -> tensor<64x301056xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v40 = stablehlo.convolution(%v39, %s0b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<64x96x56x56xf32>
    %v41 = stablehlo.broadcast_in_dim %s0b0db, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<64x96x56x56xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<64x301056xf32>) -> tensor<64x96x3136xf32>
    %v45 = stablehlo.transpose %v44, dims = [0, 2, 1] : (tensor<64x96x3136xf32>) -> tensor<64x3136x96xf32>
    %v46 = stablehlo.reshape %v45 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v48 = stablehlo.constant dense<0.0> : tensor<f32>
    %v49 = stablehlo.constant dense<96.0> : tensor<64x3136x96xf32>
    %v50 = stablehlo.constant dense<1.0e-6> : tensor<64x3136x96xf32>
    %v51 = stablehlo.reduce(%v47 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v52 = stablehlo.broadcast_in_dim %v51, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v53 = stablehlo.divide %v52, %v49 : tensor<64x3136x96xf32>
    %v54 = stablehlo.subtract %v47, %v53 : tensor<64x3136x96xf32>
    %v55 = stablehlo.multiply %v54, %v54 : tensor<64x3136x96xf32>
    %v56 = stablehlo.reduce(%v55 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v57 = stablehlo.broadcast_in_dim %v56, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v58 = stablehlo.divide %v57, %v49 : tensor<64x3136x96xf32>
    %v59 = stablehlo.add %v58, %v50 : tensor<64x3136x96xf32>
    %v60 = stablehlo.rsqrt %v59 : tensor<64x3136x96xf32>
    %v61 = stablehlo.multiply %v54, %v60 : tensor<64x3136x96xf32>
    %v62 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v63 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v64 = stablehlo.multiply %v61, %v62 : tensor<64x3136x96xf32>
    %v65 = stablehlo.add %v64, %v63 : tensor<64x3136x96xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v68 = stablehlo.broadcast_in_dim %s0b0ng, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v69 = stablehlo.multiply %v67, %v68 : tensor<64x3136x96xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v72 = stablehlo.broadcast_in_dim %s0b0nbt, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v73 = stablehlo.add %v71, %v72 : tensor<64x3136x96xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v76 = stablehlo.transpose %v75, dims = [0, 2, 1] : (tensor<64x3136x96xf32>) -> tensor<64x96x3136xf32>
    %v77 = stablehlo.reshape %v76 : (tensor<64x96x3136xf32>) -> tensor<64x301056xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v79 = stablehlo.convolution(%v78, %s0b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<64x384x56x56xf32>
    %v80 = stablehlo.broadcast_in_dim %s0b0eb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x56x56xf32>
    %v81 = stablehlo.add %v79, %v80 : tensor<64x384x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<64x384x56x56xf32>) -> tensor<64x1204224xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<64x1204224xf32>) -> tensor<64x384x56x56xf32>
    %v84 = stablehlo.constant dense<0.5> : tensor<64x384x56x56xf32>
    %v85 = stablehlo.multiply %v84, %v83 : tensor<64x384x56x56xf32>
    %v86 = stablehlo.negate %v83 : tensor<64x384x56x56xf32>
    %v87 = stablehlo.constant dense<0.7071067811865476> : tensor<64x384x56x56xf32>
    %v88 = stablehlo.multiply %v86, %v87 : tensor<64x384x56x56xf32>
    %v89 = chlo.erfc %v88 : tensor<64x384x56x56xf32> -> tensor<64x384x56x56xf32>
    %v90 = stablehlo.multiply %v85, %v89 : tensor<64x384x56x56xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<64x384x56x56xf32>) -> tensor<64x1204224xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<64x1204224xf32>) -> tensor<64x384x56x56xf32>
    %v93 = stablehlo.convolution(%v92, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<64x96x56x56xf32>
    %v94 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v95 = stablehlo.add %v93, %v94 : tensor<64x96x56x56xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v98 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v99 = stablehlo.multiply %v97, %v98 : tensor<64x96x56x56xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v102 = stablehlo.reshape %v38 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v103 = stablehlo.add %v101, %v102 : tensor<64x96x56x56xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v106 = stablehlo.convolution(%v105, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<64x96x56x56xf32>
    %v107 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v108 = stablehlo.add %v106, %v107 : tensor<64x96x56x56xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<64x301056xf32>) -> tensor<64x96x3136xf32>
    %v111 = stablehlo.transpose %v110, dims = [0, 2, 1] : (tensor<64x96x3136xf32>) -> tensor<64x3136x96xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v115 = stablehlo.constant dense<96.0> : tensor<64x3136x96xf32>
    %v116 = stablehlo.constant dense<1.0e-6> : tensor<64x3136x96xf32>
    %v117 = stablehlo.reduce(%v113 init: %v114) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v118 = stablehlo.broadcast_in_dim %v117, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v119 = stablehlo.divide %v118, %v115 : tensor<64x3136x96xf32>
    %v120 = stablehlo.subtract %v113, %v119 : tensor<64x3136x96xf32>
    %v121 = stablehlo.multiply %v120, %v120 : tensor<64x3136x96xf32>
    %v122 = stablehlo.reduce(%v121 init: %v114) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v123 = stablehlo.broadcast_in_dim %v122, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v124 = stablehlo.divide %v123, %v115 : tensor<64x3136x96xf32>
    %v125 = stablehlo.add %v124, %v116 : tensor<64x3136x96xf32>
    %v126 = stablehlo.rsqrt %v125 : tensor<64x3136x96xf32>
    %v127 = stablehlo.multiply %v120, %v126 : tensor<64x3136x96xf32>
    %v128 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v129 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v130 = stablehlo.multiply %v127, %v128 : tensor<64x3136x96xf32>
    %v131 = stablehlo.add %v130, %v129 : tensor<64x3136x96xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v134 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v135 = stablehlo.multiply %v133, %v134 : tensor<64x3136x96xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v138 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v139 = stablehlo.add %v137, %v138 : tensor<64x3136x96xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v142 = stablehlo.transpose %v141, dims = [0, 2, 1] : (tensor<64x3136x96xf32>) -> tensor<64x96x3136xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<64x96x3136xf32>) -> tensor<64x301056xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v145 = stablehlo.convolution(%v144, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<64x384x56x56xf32>
    %v146 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x56x56xf32>
    %v147 = stablehlo.add %v145, %v146 : tensor<64x384x56x56xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<64x384x56x56xf32>) -> tensor<64x1204224xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<64x1204224xf32>) -> tensor<64x384x56x56xf32>
    %v150 = stablehlo.constant dense<0.5> : tensor<64x384x56x56xf32>
    %v151 = stablehlo.multiply %v150, %v149 : tensor<64x384x56x56xf32>
    %v152 = stablehlo.negate %v149 : tensor<64x384x56x56xf32>
    %v153 = stablehlo.constant dense<0.7071067811865476> : tensor<64x384x56x56xf32>
    %v154 = stablehlo.multiply %v152, %v153 : tensor<64x384x56x56xf32>
    %v155 = chlo.erfc %v154 : tensor<64x384x56x56xf32> -> tensor<64x384x56x56xf32>
    %v156 = stablehlo.multiply %v151, %v155 : tensor<64x384x56x56xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<64x384x56x56xf32>) -> tensor<64x1204224xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<64x1204224xf32>) -> tensor<64x384x56x56xf32>
    %v159 = stablehlo.convolution(%v158, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<64x96x56x56xf32>
    %v160 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v161 = stablehlo.add %v159, %v160 : tensor<64x96x56x56xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v164 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v165 = stablehlo.multiply %v163, %v164 : tensor<64x96x56x56xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v168 = stablehlo.reshape %v104 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v169 = stablehlo.add %v167, %v168 : tensor<64x96x56x56xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v172 = stablehlo.convolution(%v171, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<64x96x56x56xf32>
    %v173 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v174 = stablehlo.add %v172, %v173 : tensor<64x96x56x56xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<64x301056xf32>) -> tensor<64x96x3136xf32>
    %v177 = stablehlo.transpose %v176, dims = [0, 2, 1] : (tensor<64x96x3136xf32>) -> tensor<64x3136x96xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v181 = stablehlo.constant dense<96.0> : tensor<64x3136x96xf32>
    %v182 = stablehlo.constant dense<1.0e-6> : tensor<64x3136x96xf32>
    %v183 = stablehlo.reduce(%v179 init: %v180) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v184 = stablehlo.broadcast_in_dim %v183, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v185 = stablehlo.divide %v184, %v181 : tensor<64x3136x96xf32>
    %v186 = stablehlo.subtract %v179, %v185 : tensor<64x3136x96xf32>
    %v187 = stablehlo.multiply %v186, %v186 : tensor<64x3136x96xf32>
    %v188 = stablehlo.reduce(%v187 init: %v180) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v189 = stablehlo.broadcast_in_dim %v188, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v190 = stablehlo.divide %v189, %v181 : tensor<64x3136x96xf32>
    %v191 = stablehlo.add %v190, %v182 : tensor<64x3136x96xf32>
    %v192 = stablehlo.rsqrt %v191 : tensor<64x3136x96xf32>
    %v193 = stablehlo.multiply %v186, %v192 : tensor<64x3136x96xf32>
    %v194 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v195 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v196 = stablehlo.multiply %v193, %v194 : tensor<64x3136x96xf32>
    %v197 = stablehlo.add %v196, %v195 : tensor<64x3136x96xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v200 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v201 = stablehlo.multiply %v199, %v200 : tensor<64x3136x96xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v204 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v205 = stablehlo.add %v203, %v204 : tensor<64x3136x96xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v208 = stablehlo.transpose %v207, dims = [0, 2, 1] : (tensor<64x3136x96xf32>) -> tensor<64x96x3136xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<64x96x3136xf32>) -> tensor<64x301056xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v211 = stablehlo.convolution(%v210, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<64x384x56x56xf32>
    %v212 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x56x56xf32>
    %v213 = stablehlo.add %v211, %v212 : tensor<64x384x56x56xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<64x384x56x56xf32>) -> tensor<64x1204224xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<64x1204224xf32>) -> tensor<64x384x56x56xf32>
    %v216 = stablehlo.constant dense<0.5> : tensor<64x384x56x56xf32>
    %v217 = stablehlo.multiply %v216, %v215 : tensor<64x384x56x56xf32>
    %v218 = stablehlo.negate %v215 : tensor<64x384x56x56xf32>
    %v219 = stablehlo.constant dense<0.7071067811865476> : tensor<64x384x56x56xf32>
    %v220 = stablehlo.multiply %v218, %v219 : tensor<64x384x56x56xf32>
    %v221 = chlo.erfc %v220 : tensor<64x384x56x56xf32> -> tensor<64x384x56x56xf32>
    %v222 = stablehlo.multiply %v217, %v221 : tensor<64x384x56x56xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<64x384x56x56xf32>) -> tensor<64x1204224xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<64x1204224xf32>) -> tensor<64x384x56x56xf32>
    %v225 = stablehlo.convolution(%v224, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<64x96x56x56xf32>
    %v226 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v227 = stablehlo.add %v225, %v226 : tensor<64x96x56x56xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v230 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v231 = stablehlo.multiply %v229, %v230 : tensor<64x96x56x56xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v234 = stablehlo.reshape %v170 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v235 = stablehlo.add %v233, %v234 : tensor<64x96x56x56xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<64x301056xf32>) -> tensor<64x96x3136xf32>
    %v238 = stablehlo.transpose %v237, dims = [0, 2, 1] : (tensor<64x96x3136xf32>) -> tensor<64x3136x96xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v241 = stablehlo.constant dense<0.0> : tensor<f32>
    %v242 = stablehlo.constant dense<96.0> : tensor<64x3136x96xf32>
    %v243 = stablehlo.constant dense<1.0e-6> : tensor<64x3136x96xf32>
    %v244 = stablehlo.reduce(%v240 init: %v241) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v245 = stablehlo.broadcast_in_dim %v244, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v246 = stablehlo.divide %v245, %v242 : tensor<64x3136x96xf32>
    %v247 = stablehlo.subtract %v240, %v246 : tensor<64x3136x96xf32>
    %v248 = stablehlo.multiply %v247, %v247 : tensor<64x3136x96xf32>
    %v249 = stablehlo.reduce(%v248 init: %v241) applies stablehlo.add across dimensions = [2] : (tensor<64x3136x96xf32>, tensor<f32>) -> tensor<64x3136xf32>
    %v250 = stablehlo.broadcast_in_dim %v249, dims = [0, 1] : (tensor<64x3136xf32>) -> tensor<64x3136x96xf32>
    %v251 = stablehlo.divide %v250, %v242 : tensor<64x3136x96xf32>
    %v252 = stablehlo.add %v251, %v243 : tensor<64x3136x96xf32>
    %v253 = stablehlo.rsqrt %v252 : tensor<64x3136x96xf32>
    %v254 = stablehlo.multiply %v247, %v253 : tensor<64x3136x96xf32>
    %v255 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v256 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x3136x96xf32>
    %v257 = stablehlo.multiply %v254, %v255 : tensor<64x3136x96xf32>
    %v258 = stablehlo.add %v257, %v256 : tensor<64x3136x96xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v261 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v262 = stablehlo.multiply %v260, %v261 : tensor<64x3136x96xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v265 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<96xf32>) -> tensor<64x3136x96xf32>
    %v266 = stablehlo.add %v264, %v265 : tensor<64x3136x96xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<64x3136x96xf32>) -> tensor<64x301056xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<64x301056xf32>) -> tensor<64x3136x96xf32>
    %v269 = stablehlo.transpose %v268, dims = [0, 2, 1] : (tensor<64x3136x96xf32>) -> tensor<64x96x3136xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<64x96x3136xf32>) -> tensor<64x301056xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v272 = stablehlo.convolution(%v271, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<64x192x28x28xf32>
    %v273 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v274 = stablehlo.add %v272, %v273 : tensor<64x192x28x28xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v277 = stablehlo.convolution(%v276, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<64x192x28x28xf32>
    %v278 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<64x192x28x28xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<64x150528xf32>) -> tensor<64x192x784xf32>
    %v282 = stablehlo.transpose %v281, dims = [0, 2, 1] : (tensor<64x192x784xf32>) -> tensor<64x784x192xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v286 = stablehlo.constant dense<192.0> : tensor<64x784x192xf32>
    %v287 = stablehlo.constant dense<1.0e-6> : tensor<64x784x192xf32>
    %v288 = stablehlo.reduce(%v284 init: %v285) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v289 = stablehlo.broadcast_in_dim %v288, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v290 = stablehlo.divide %v289, %v286 : tensor<64x784x192xf32>
    %v291 = stablehlo.subtract %v284, %v290 : tensor<64x784x192xf32>
    %v292 = stablehlo.multiply %v291, %v291 : tensor<64x784x192xf32>
    %v293 = stablehlo.reduce(%v292 init: %v285) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v294 = stablehlo.broadcast_in_dim %v293, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v295 = stablehlo.divide %v294, %v286 : tensor<64x784x192xf32>
    %v296 = stablehlo.add %v295, %v287 : tensor<64x784x192xf32>
    %v297 = stablehlo.rsqrt %v296 : tensor<64x784x192xf32>
    %v298 = stablehlo.multiply %v291, %v297 : tensor<64x784x192xf32>
    %v299 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v300 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v301 = stablehlo.multiply %v298, %v299 : tensor<64x784x192xf32>
    %v302 = stablehlo.add %v301, %v300 : tensor<64x784x192xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v305 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v306 = stablehlo.multiply %v304, %v305 : tensor<64x784x192xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v309 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v310 = stablehlo.add %v308, %v309 : tensor<64x784x192xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v313 = stablehlo.transpose %v312, dims = [0, 2, 1] : (tensor<64x784x192xf32>) -> tensor<64x192x784xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<64x192x784xf32>) -> tensor<64x150528xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v316 = stablehlo.convolution(%v315, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<64x768x28x28xf32>
    %v317 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x28x28xf32>
    %v318 = stablehlo.add %v316, %v317 : tensor<64x768x28x28xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<64x768x28x28xf32>) -> tensor<64x602112xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<64x602112xf32>) -> tensor<64x768x28x28xf32>
    %v321 = stablehlo.constant dense<0.5> : tensor<64x768x28x28xf32>
    %v322 = stablehlo.multiply %v321, %v320 : tensor<64x768x28x28xf32>
    %v323 = stablehlo.negate %v320 : tensor<64x768x28x28xf32>
    %v324 = stablehlo.constant dense<0.7071067811865476> : tensor<64x768x28x28xf32>
    %v325 = stablehlo.multiply %v323, %v324 : tensor<64x768x28x28xf32>
    %v326 = chlo.erfc %v325 : tensor<64x768x28x28xf32> -> tensor<64x768x28x28xf32>
    %v327 = stablehlo.multiply %v322, %v326 : tensor<64x768x28x28xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<64x768x28x28xf32>) -> tensor<64x602112xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<64x602112xf32>) -> tensor<64x768x28x28xf32>
    %v330 = stablehlo.convolution(%v329, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<64x192x28x28xf32>
    %v331 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v332 = stablehlo.add %v330, %v331 : tensor<64x192x28x28xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v335 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v336 = stablehlo.multiply %v334, %v335 : tensor<64x192x28x28xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v339 = stablehlo.reshape %v275 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<64x192x28x28xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v343 = stablehlo.convolution(%v342, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<64x192x28x28xf32>
    %v344 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v345 = stablehlo.add %v343, %v344 : tensor<64x192x28x28xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<64x150528xf32>) -> tensor<64x192x784xf32>
    %v348 = stablehlo.transpose %v347, dims = [0, 2, 1] : (tensor<64x192x784xf32>) -> tensor<64x784x192xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v352 = stablehlo.constant dense<192.0> : tensor<64x784x192xf32>
    %v353 = stablehlo.constant dense<1.0e-6> : tensor<64x784x192xf32>
    %v354 = stablehlo.reduce(%v350 init: %v351) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v355 = stablehlo.broadcast_in_dim %v354, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v356 = stablehlo.divide %v355, %v352 : tensor<64x784x192xf32>
    %v357 = stablehlo.subtract %v350, %v356 : tensor<64x784x192xf32>
    %v358 = stablehlo.multiply %v357, %v357 : tensor<64x784x192xf32>
    %v359 = stablehlo.reduce(%v358 init: %v351) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v360 = stablehlo.broadcast_in_dim %v359, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v361 = stablehlo.divide %v360, %v352 : tensor<64x784x192xf32>
    %v362 = stablehlo.add %v361, %v353 : tensor<64x784x192xf32>
    %v363 = stablehlo.rsqrt %v362 : tensor<64x784x192xf32>
    %v364 = stablehlo.multiply %v357, %v363 : tensor<64x784x192xf32>
    %v365 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v366 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v367 = stablehlo.multiply %v364, %v365 : tensor<64x784x192xf32>
    %v368 = stablehlo.add %v367, %v366 : tensor<64x784x192xf32>
    %v369 = stablehlo.reshape %v368 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v371 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v372 = stablehlo.multiply %v370, %v371 : tensor<64x784x192xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v375 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v376 = stablehlo.add %v374, %v375 : tensor<64x784x192xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v379 = stablehlo.transpose %v378, dims = [0, 2, 1] : (tensor<64x784x192xf32>) -> tensor<64x192x784xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<64x192x784xf32>) -> tensor<64x150528xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v382 = stablehlo.convolution(%v381, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<64x768x28x28xf32>
    %v383 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x28x28xf32>
    %v384 = stablehlo.add %v382, %v383 : tensor<64x768x28x28xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<64x768x28x28xf32>) -> tensor<64x602112xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<64x602112xf32>) -> tensor<64x768x28x28xf32>
    %v387 = stablehlo.constant dense<0.5> : tensor<64x768x28x28xf32>
    %v388 = stablehlo.multiply %v387, %v386 : tensor<64x768x28x28xf32>
    %v389 = stablehlo.negate %v386 : tensor<64x768x28x28xf32>
    %v390 = stablehlo.constant dense<0.7071067811865476> : tensor<64x768x28x28xf32>
    %v391 = stablehlo.multiply %v389, %v390 : tensor<64x768x28x28xf32>
    %v392 = chlo.erfc %v391 : tensor<64x768x28x28xf32> -> tensor<64x768x28x28xf32>
    %v393 = stablehlo.multiply %v388, %v392 : tensor<64x768x28x28xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<64x768x28x28xf32>) -> tensor<64x602112xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<64x602112xf32>) -> tensor<64x768x28x28xf32>
    %v396 = stablehlo.convolution(%v395, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<64x192x28x28xf32>
    %v397 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v398 = stablehlo.add %v396, %v397 : tensor<64x192x28x28xf32>
    %v399 = stablehlo.reshape %v398 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v400 = stablehlo.reshape %v399 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v401 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v402 = stablehlo.multiply %v400, %v401 : tensor<64x192x28x28xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v405 = stablehlo.reshape %v341 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v406 = stablehlo.add %v404, %v405 : tensor<64x192x28x28xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v409 = stablehlo.convolution(%v408, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<64x192x28x28xf32>
    %v410 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v411 = stablehlo.add %v409, %v410 : tensor<64x192x28x28xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<64x150528xf32>) -> tensor<64x192x784xf32>
    %v414 = stablehlo.transpose %v413, dims = [0, 2, 1] : (tensor<64x192x784xf32>) -> tensor<64x784x192xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v418 = stablehlo.constant dense<192.0> : tensor<64x784x192xf32>
    %v419 = stablehlo.constant dense<1.0e-6> : tensor<64x784x192xf32>
    %v420 = stablehlo.reduce(%v416 init: %v417) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v421 = stablehlo.broadcast_in_dim %v420, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v422 = stablehlo.divide %v421, %v418 : tensor<64x784x192xf32>
    %v423 = stablehlo.subtract %v416, %v422 : tensor<64x784x192xf32>
    %v424 = stablehlo.multiply %v423, %v423 : tensor<64x784x192xf32>
    %v425 = stablehlo.reduce(%v424 init: %v417) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v426 = stablehlo.broadcast_in_dim %v425, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v427 = stablehlo.divide %v426, %v418 : tensor<64x784x192xf32>
    %v428 = stablehlo.add %v427, %v419 : tensor<64x784x192xf32>
    %v429 = stablehlo.rsqrt %v428 : tensor<64x784x192xf32>
    %v430 = stablehlo.multiply %v423, %v429 : tensor<64x784x192xf32>
    %v431 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v432 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v433 = stablehlo.multiply %v430, %v431 : tensor<64x784x192xf32>
    %v434 = stablehlo.add %v433, %v432 : tensor<64x784x192xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v437 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v438 = stablehlo.multiply %v436, %v437 : tensor<64x784x192xf32>
    %v439 = stablehlo.reshape %v438 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v441 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v442 = stablehlo.add %v440, %v441 : tensor<64x784x192xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v444 = stablehlo.reshape %v443 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v445 = stablehlo.transpose %v444, dims = [0, 2, 1] : (tensor<64x784x192xf32>) -> tensor<64x192x784xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<64x192x784xf32>) -> tensor<64x150528xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v448 = stablehlo.convolution(%v447, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<64x768x28x28xf32>
    %v449 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x28x28xf32>
    %v450 = stablehlo.add %v448, %v449 : tensor<64x768x28x28xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<64x768x28x28xf32>) -> tensor<64x602112xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<64x602112xf32>) -> tensor<64x768x28x28xf32>
    %v453 = stablehlo.constant dense<0.5> : tensor<64x768x28x28xf32>
    %v454 = stablehlo.multiply %v453, %v452 : tensor<64x768x28x28xf32>
    %v455 = stablehlo.negate %v452 : tensor<64x768x28x28xf32>
    %v456 = stablehlo.constant dense<0.7071067811865476> : tensor<64x768x28x28xf32>
    %v457 = stablehlo.multiply %v455, %v456 : tensor<64x768x28x28xf32>
    %v458 = chlo.erfc %v457 : tensor<64x768x28x28xf32> -> tensor<64x768x28x28xf32>
    %v459 = stablehlo.multiply %v454, %v458 : tensor<64x768x28x28xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<64x768x28x28xf32>) -> tensor<64x602112xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<64x602112xf32>) -> tensor<64x768x28x28xf32>
    %v462 = stablehlo.convolution(%v461, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<64x192x28x28xf32>
    %v463 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v464 = stablehlo.add %v462, %v463 : tensor<64x192x28x28xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v467 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v468 = stablehlo.multiply %v466, %v467 : tensor<64x192x28x28xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v471 = stablehlo.reshape %v407 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v472 = stablehlo.add %v470, %v471 : tensor<64x192x28x28xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<64x150528xf32>) -> tensor<64x192x784xf32>
    %v475 = stablehlo.transpose %v474, dims = [0, 2, 1] : (tensor<64x192x784xf32>) -> tensor<64x784x192xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v479 = stablehlo.constant dense<192.0> : tensor<64x784x192xf32>
    %v480 = stablehlo.constant dense<1.0e-6> : tensor<64x784x192xf32>
    %v481 = stablehlo.reduce(%v477 init: %v478) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v482 = stablehlo.broadcast_in_dim %v481, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v483 = stablehlo.divide %v482, %v479 : tensor<64x784x192xf32>
    %v484 = stablehlo.subtract %v477, %v483 : tensor<64x784x192xf32>
    %v485 = stablehlo.multiply %v484, %v484 : tensor<64x784x192xf32>
    %v486 = stablehlo.reduce(%v485 init: %v478) applies stablehlo.add across dimensions = [2] : (tensor<64x784x192xf32>, tensor<f32>) -> tensor<64x784xf32>
    %v487 = stablehlo.broadcast_in_dim %v486, dims = [0, 1] : (tensor<64x784xf32>) -> tensor<64x784x192xf32>
    %v488 = stablehlo.divide %v487, %v479 : tensor<64x784x192xf32>
    %v489 = stablehlo.add %v488, %v480 : tensor<64x784x192xf32>
    %v490 = stablehlo.rsqrt %v489 : tensor<64x784x192xf32>
    %v491 = stablehlo.multiply %v484, %v490 : tensor<64x784x192xf32>
    %v492 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v493 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x784x192xf32>
    %v494 = stablehlo.multiply %v491, %v492 : tensor<64x784x192xf32>
    %v495 = stablehlo.add %v494, %v493 : tensor<64x784x192xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v498 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v499 = stablehlo.multiply %v497, %v498 : tensor<64x784x192xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v502 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<192xf32>) -> tensor<64x784x192xf32>
    %v503 = stablehlo.add %v501, %v502 : tensor<64x784x192xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<64x784x192xf32>) -> tensor<64x150528xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<64x150528xf32>) -> tensor<64x784x192xf32>
    %v506 = stablehlo.transpose %v505, dims = [0, 2, 1] : (tensor<64x784x192xf32>) -> tensor<64x192x784xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<64x192x784xf32>) -> tensor<64x150528xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v509 = stablehlo.convolution(%v508, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<64x384x14x14xf32>
    %v510 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v511 = stablehlo.add %v509, %v510 : tensor<64x384x14x14xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v514 = stablehlo.convolution(%v513, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v515 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v516 = stablehlo.add %v514, %v515 : tensor<64x384x14x14xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v519 = stablehlo.transpose %v518, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v523 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v524 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v525 = stablehlo.reduce(%v521 init: %v522) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v526 = stablehlo.broadcast_in_dim %v525, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v527 = stablehlo.divide %v526, %v523 : tensor<64x196x384xf32>
    %v528 = stablehlo.subtract %v521, %v527 : tensor<64x196x384xf32>
    %v529 = stablehlo.multiply %v528, %v528 : tensor<64x196x384xf32>
    %v530 = stablehlo.reduce(%v529 init: %v522) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v532 = stablehlo.divide %v531, %v523 : tensor<64x196x384xf32>
    %v533 = stablehlo.add %v532, %v524 : tensor<64x196x384xf32>
    %v534 = stablehlo.rsqrt %v533 : tensor<64x196x384xf32>
    %v535 = stablehlo.multiply %v528, %v534 : tensor<64x196x384xf32>
    %v536 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v537 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v538 = stablehlo.multiply %v535, %v536 : tensor<64x196x384xf32>
    %v539 = stablehlo.add %v538, %v537 : tensor<64x196x384xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v542 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v543 = stablehlo.multiply %v541, %v542 : tensor<64x196x384xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v546 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v547 = stablehlo.add %v545, %v546 : tensor<64x196x384xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v550 = stablehlo.transpose %v549, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v553 = stablehlo.convolution(%v552, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v554 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v555 = stablehlo.add %v553, %v554 : tensor<64x1536x14x14xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v558 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v559 = stablehlo.multiply %v558, %v557 : tensor<64x1536x14x14xf32>
    %v560 = stablehlo.negate %v557 : tensor<64x1536x14x14xf32>
    %v561 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v562 = stablehlo.multiply %v560, %v561 : tensor<64x1536x14x14xf32>
    %v563 = chlo.erfc %v562 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v564 = stablehlo.multiply %v559, %v563 : tensor<64x1536x14x14xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v566 = stablehlo.reshape %v565 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v567 = stablehlo.convolution(%v566, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v568 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v569 = stablehlo.add %v567, %v568 : tensor<64x384x14x14xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v571 = stablehlo.reshape %v570 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v572 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v573 = stablehlo.multiply %v571, %v572 : tensor<64x384x14x14xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v576 = stablehlo.reshape %v512 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v577 = stablehlo.add %v575, %v576 : tensor<64x384x14x14xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v579 = stablehlo.reshape %v578 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v580 = stablehlo.convolution(%v579, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v581 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v582 = stablehlo.add %v580, %v581 : tensor<64x384x14x14xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v585 = stablehlo.transpose %v584, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v588 = stablehlo.constant dense<0.0> : tensor<f32>
    %v589 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v590 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v591 = stablehlo.reduce(%v587 init: %v588) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v592 = stablehlo.broadcast_in_dim %v591, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v593 = stablehlo.divide %v592, %v589 : tensor<64x196x384xf32>
    %v594 = stablehlo.subtract %v587, %v593 : tensor<64x196x384xf32>
    %v595 = stablehlo.multiply %v594, %v594 : tensor<64x196x384xf32>
    %v596 = stablehlo.reduce(%v595 init: %v588) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v597 = stablehlo.broadcast_in_dim %v596, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v598 = stablehlo.divide %v597, %v589 : tensor<64x196x384xf32>
    %v599 = stablehlo.add %v598, %v590 : tensor<64x196x384xf32>
    %v600 = stablehlo.rsqrt %v599 : tensor<64x196x384xf32>
    %v601 = stablehlo.multiply %v594, %v600 : tensor<64x196x384xf32>
    %v602 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v603 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v604 = stablehlo.multiply %v601, %v602 : tensor<64x196x384xf32>
    %v605 = stablehlo.add %v604, %v603 : tensor<64x196x384xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v608 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v609 = stablehlo.multiply %v607, %v608 : tensor<64x196x384xf32>
    %v610 = stablehlo.reshape %v609 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v612 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v613 = stablehlo.add %v611, %v612 : tensor<64x196x384xf32>
    %v614 = stablehlo.reshape %v613 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v615 = stablehlo.reshape %v614 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v616 = stablehlo.transpose %v615, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v619 = stablehlo.convolution(%v618, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v620 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v621 = stablehlo.add %v619, %v620 : tensor<64x1536x14x14xf32>
    %v622 = stablehlo.reshape %v621 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v623 = stablehlo.reshape %v622 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v624 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v625 = stablehlo.multiply %v624, %v623 : tensor<64x1536x14x14xf32>
    %v626 = stablehlo.negate %v623 : tensor<64x1536x14x14xf32>
    %v627 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v628 = stablehlo.multiply %v626, %v627 : tensor<64x1536x14x14xf32>
    %v629 = chlo.erfc %v628 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v630 = stablehlo.multiply %v625, %v629 : tensor<64x1536x14x14xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v633 = stablehlo.convolution(%v632, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v634 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<64x384x14x14xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v638 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v639 = stablehlo.multiply %v637, %v638 : tensor<64x384x14x14xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v642 = stablehlo.reshape %v578 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v643 = stablehlo.add %v641, %v642 : tensor<64x384x14x14xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v646 = stablehlo.convolution(%v645, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v647 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v648 = stablehlo.add %v646, %v647 : tensor<64x384x14x14xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v650 = stablehlo.reshape %v649 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v651 = stablehlo.transpose %v650, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v655 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v656 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v657 = stablehlo.reduce(%v653 init: %v654) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v658 = stablehlo.broadcast_in_dim %v657, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v659 = stablehlo.divide %v658, %v655 : tensor<64x196x384xf32>
    %v660 = stablehlo.subtract %v653, %v659 : tensor<64x196x384xf32>
    %v661 = stablehlo.multiply %v660, %v660 : tensor<64x196x384xf32>
    %v662 = stablehlo.reduce(%v661 init: %v654) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v663 = stablehlo.broadcast_in_dim %v662, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v664 = stablehlo.divide %v663, %v655 : tensor<64x196x384xf32>
    %v665 = stablehlo.add %v664, %v656 : tensor<64x196x384xf32>
    %v666 = stablehlo.rsqrt %v665 : tensor<64x196x384xf32>
    %v667 = stablehlo.multiply %v660, %v666 : tensor<64x196x384xf32>
    %v668 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v669 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v670 = stablehlo.multiply %v667, %v668 : tensor<64x196x384xf32>
    %v671 = stablehlo.add %v670, %v669 : tensor<64x196x384xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v674 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v675 = stablehlo.multiply %v673, %v674 : tensor<64x196x384xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v678 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v679 = stablehlo.add %v677, %v678 : tensor<64x196x384xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v682 = stablehlo.transpose %v681, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v685 = stablehlo.convolution(%v684, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v686 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v687 = stablehlo.add %v685, %v686 : tensor<64x1536x14x14xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v690 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v691 = stablehlo.multiply %v690, %v689 : tensor<64x1536x14x14xf32>
    %v692 = stablehlo.negate %v689 : tensor<64x1536x14x14xf32>
    %v693 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v694 = stablehlo.multiply %v692, %v693 : tensor<64x1536x14x14xf32>
    %v695 = chlo.erfc %v694 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v696 = stablehlo.multiply %v691, %v695 : tensor<64x1536x14x14xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v699 = stablehlo.convolution(%v698, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v700 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v701 = stablehlo.add %v699, %v700 : tensor<64x384x14x14xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v704 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v705 = stablehlo.multiply %v703, %v704 : tensor<64x384x14x14xf32>
    %v706 = stablehlo.reshape %v705 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v708 = stablehlo.reshape %v644 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v709 = stablehlo.add %v707, %v708 : tensor<64x384x14x14xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v712 = stablehlo.convolution(%v711, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v713 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v714 = stablehlo.add %v712, %v713 : tensor<64x384x14x14xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v717 = stablehlo.transpose %v716, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v720 = stablehlo.constant dense<0.0> : tensor<f32>
    %v721 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v722 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v723 = stablehlo.reduce(%v719 init: %v720) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v724 = stablehlo.broadcast_in_dim %v723, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v725 = stablehlo.divide %v724, %v721 : tensor<64x196x384xf32>
    %v726 = stablehlo.subtract %v719, %v725 : tensor<64x196x384xf32>
    %v727 = stablehlo.multiply %v726, %v726 : tensor<64x196x384xf32>
    %v728 = stablehlo.reduce(%v727 init: %v720) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v729 = stablehlo.broadcast_in_dim %v728, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v730 = stablehlo.divide %v729, %v721 : tensor<64x196x384xf32>
    %v731 = stablehlo.add %v730, %v722 : tensor<64x196x384xf32>
    %v732 = stablehlo.rsqrt %v731 : tensor<64x196x384xf32>
    %v733 = stablehlo.multiply %v726, %v732 : tensor<64x196x384xf32>
    %v734 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v735 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v736 = stablehlo.multiply %v733, %v734 : tensor<64x196x384xf32>
    %v737 = stablehlo.add %v736, %v735 : tensor<64x196x384xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v740 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v741 = stablehlo.multiply %v739, %v740 : tensor<64x196x384xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v744 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v745 = stablehlo.add %v743, %v744 : tensor<64x196x384xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v748 = stablehlo.transpose %v747, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v751 = stablehlo.convolution(%v750, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v752 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v753 = stablehlo.add %v751, %v752 : tensor<64x1536x14x14xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v756 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v757 = stablehlo.multiply %v756, %v755 : tensor<64x1536x14x14xf32>
    %v758 = stablehlo.negate %v755 : tensor<64x1536x14x14xf32>
    %v759 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v760 = stablehlo.multiply %v758, %v759 : tensor<64x1536x14x14xf32>
    %v761 = chlo.erfc %v760 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v762 = stablehlo.multiply %v757, %v761 : tensor<64x1536x14x14xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v765 = stablehlo.convolution(%v764, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v766 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v767 = stablehlo.add %v765, %v766 : tensor<64x384x14x14xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v770 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v771 = stablehlo.multiply %v769, %v770 : tensor<64x384x14x14xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v774 = stablehlo.reshape %v710 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v775 = stablehlo.add %v773, %v774 : tensor<64x384x14x14xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v778 = stablehlo.convolution(%v777, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v779 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v780 = stablehlo.add %v778, %v779 : tensor<64x384x14x14xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v783 = stablehlo.transpose %v782, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v787 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v788 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v789 = stablehlo.reduce(%v785 init: %v786) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v790 = stablehlo.broadcast_in_dim %v789, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v791 = stablehlo.divide %v790, %v787 : tensor<64x196x384xf32>
    %v792 = stablehlo.subtract %v785, %v791 : tensor<64x196x384xf32>
    %v793 = stablehlo.multiply %v792, %v792 : tensor<64x196x384xf32>
    %v794 = stablehlo.reduce(%v793 init: %v786) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v795 = stablehlo.broadcast_in_dim %v794, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v796 = stablehlo.divide %v795, %v787 : tensor<64x196x384xf32>
    %v797 = stablehlo.add %v796, %v788 : tensor<64x196x384xf32>
    %v798 = stablehlo.rsqrt %v797 : tensor<64x196x384xf32>
    %v799 = stablehlo.multiply %v792, %v798 : tensor<64x196x384xf32>
    %v800 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v801 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v802 = stablehlo.multiply %v799, %v800 : tensor<64x196x384xf32>
    %v803 = stablehlo.add %v802, %v801 : tensor<64x196x384xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v806 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v807 = stablehlo.multiply %v805, %v806 : tensor<64x196x384xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v810 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v811 = stablehlo.add %v809, %v810 : tensor<64x196x384xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v814 = stablehlo.transpose %v813, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v816 = stablehlo.reshape %v815 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v817 = stablehlo.convolution(%v816, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v818 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v819 = stablehlo.add %v817, %v818 : tensor<64x1536x14x14xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v822 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v823 = stablehlo.multiply %v822, %v821 : tensor<64x1536x14x14xf32>
    %v824 = stablehlo.negate %v821 : tensor<64x1536x14x14xf32>
    %v825 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v826 = stablehlo.multiply %v824, %v825 : tensor<64x1536x14x14xf32>
    %v827 = chlo.erfc %v826 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v828 = stablehlo.multiply %v823, %v827 : tensor<64x1536x14x14xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v831 = stablehlo.convolution(%v830, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v832 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v833 = stablehlo.add %v831, %v832 : tensor<64x384x14x14xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v836 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v837 = stablehlo.multiply %v835, %v836 : tensor<64x384x14x14xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v840 = stablehlo.reshape %v776 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v841 = stablehlo.add %v839, %v840 : tensor<64x384x14x14xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v844 = stablehlo.convolution(%v843, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v845 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v846 = stablehlo.add %v844, %v845 : tensor<64x384x14x14xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v849 = stablehlo.transpose %v848, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v852 = stablehlo.constant dense<0.0> : tensor<f32>
    %v853 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v854 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v855 = stablehlo.reduce(%v851 init: %v852) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v856 = stablehlo.broadcast_in_dim %v855, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v857 = stablehlo.divide %v856, %v853 : tensor<64x196x384xf32>
    %v858 = stablehlo.subtract %v851, %v857 : tensor<64x196x384xf32>
    %v859 = stablehlo.multiply %v858, %v858 : tensor<64x196x384xf32>
    %v860 = stablehlo.reduce(%v859 init: %v852) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v861 = stablehlo.broadcast_in_dim %v860, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v862 = stablehlo.divide %v861, %v853 : tensor<64x196x384xf32>
    %v863 = stablehlo.add %v862, %v854 : tensor<64x196x384xf32>
    %v864 = stablehlo.rsqrt %v863 : tensor<64x196x384xf32>
    %v865 = stablehlo.multiply %v858, %v864 : tensor<64x196x384xf32>
    %v866 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v867 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v868 = stablehlo.multiply %v865, %v866 : tensor<64x196x384xf32>
    %v869 = stablehlo.add %v868, %v867 : tensor<64x196x384xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v872 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v873 = stablehlo.multiply %v871, %v872 : tensor<64x196x384xf32>
    %v874 = stablehlo.reshape %v873 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v876 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v877 = stablehlo.add %v875, %v876 : tensor<64x196x384xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v880 = stablehlo.transpose %v879, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v883 = stablehlo.convolution(%v882, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v884 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v885 = stablehlo.add %v883, %v884 : tensor<64x1536x14x14xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v887 = stablehlo.reshape %v886 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v888 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v889 = stablehlo.multiply %v888, %v887 : tensor<64x1536x14x14xf32>
    %v890 = stablehlo.negate %v887 : tensor<64x1536x14x14xf32>
    %v891 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v892 = stablehlo.multiply %v890, %v891 : tensor<64x1536x14x14xf32>
    %v893 = chlo.erfc %v892 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v894 = stablehlo.multiply %v889, %v893 : tensor<64x1536x14x14xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v897 = stablehlo.convolution(%v896, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v898 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v899 = stablehlo.add %v897, %v898 : tensor<64x384x14x14xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v901 = stablehlo.reshape %v900 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v902 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v903 = stablehlo.multiply %v901, %v902 : tensor<64x384x14x14xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v906 = stablehlo.reshape %v842 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v907 = stablehlo.add %v905, %v906 : tensor<64x384x14x14xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v910 = stablehlo.convolution(%v909, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v911 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v912 = stablehlo.add %v910, %v911 : tensor<64x384x14x14xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v915 = stablehlo.transpose %v914, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v919 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v920 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v921 = stablehlo.reduce(%v917 init: %v918) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v922 = stablehlo.broadcast_in_dim %v921, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v923 = stablehlo.divide %v922, %v919 : tensor<64x196x384xf32>
    %v924 = stablehlo.subtract %v917, %v923 : tensor<64x196x384xf32>
    %v925 = stablehlo.multiply %v924, %v924 : tensor<64x196x384xf32>
    %v926 = stablehlo.reduce(%v925 init: %v918) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v927 = stablehlo.broadcast_in_dim %v926, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v928 = stablehlo.divide %v927, %v919 : tensor<64x196x384xf32>
    %v929 = stablehlo.add %v928, %v920 : tensor<64x196x384xf32>
    %v930 = stablehlo.rsqrt %v929 : tensor<64x196x384xf32>
    %v931 = stablehlo.multiply %v924, %v930 : tensor<64x196x384xf32>
    %v932 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v933 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v934 = stablehlo.multiply %v931, %v932 : tensor<64x196x384xf32>
    %v935 = stablehlo.add %v934, %v933 : tensor<64x196x384xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v938 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v939 = stablehlo.multiply %v937, %v938 : tensor<64x196x384xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v942 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v943 = stablehlo.add %v941, %v942 : tensor<64x196x384xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v946 = stablehlo.transpose %v945, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v949 = stablehlo.convolution(%v948, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v950 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v951 = stablehlo.add %v949, %v950 : tensor<64x1536x14x14xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v954 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v955 = stablehlo.multiply %v954, %v953 : tensor<64x1536x14x14xf32>
    %v956 = stablehlo.negate %v953 : tensor<64x1536x14x14xf32>
    %v957 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v958 = stablehlo.multiply %v956, %v957 : tensor<64x1536x14x14xf32>
    %v959 = chlo.erfc %v958 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v960 = stablehlo.multiply %v955, %v959 : tensor<64x1536x14x14xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v963 = stablehlo.convolution(%v962, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v964 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v965 = stablehlo.add %v963, %v964 : tensor<64x384x14x14xf32>
    %v966 = stablehlo.reshape %v965 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v968 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v969 = stablehlo.multiply %v967, %v968 : tensor<64x384x14x14xf32>
    %v970 = stablehlo.reshape %v969 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v971 = stablehlo.reshape %v970 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v972 = stablehlo.reshape %v908 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v973 = stablehlo.add %v971, %v972 : tensor<64x384x14x14xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v976 = stablehlo.convolution(%v975, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v977 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v978 = stablehlo.add %v976, %v977 : tensor<64x384x14x14xf32>
    %v979 = stablehlo.reshape %v978 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v981 = stablehlo.transpose %v980, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v984 = stablehlo.constant dense<0.0> : tensor<f32>
    %v985 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v986 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v987 = stablehlo.reduce(%v983 init: %v984) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v988 = stablehlo.broadcast_in_dim %v987, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v989 = stablehlo.divide %v988, %v985 : tensor<64x196x384xf32>
    %v990 = stablehlo.subtract %v983, %v989 : tensor<64x196x384xf32>
    %v991 = stablehlo.multiply %v990, %v990 : tensor<64x196x384xf32>
    %v992 = stablehlo.reduce(%v991 init: %v984) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v993 = stablehlo.broadcast_in_dim %v992, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v994 = stablehlo.divide %v993, %v985 : tensor<64x196x384xf32>
    %v995 = stablehlo.add %v994, %v986 : tensor<64x196x384xf32>
    %v996 = stablehlo.rsqrt %v995 : tensor<64x196x384xf32>
    %v997 = stablehlo.multiply %v990, %v996 : tensor<64x196x384xf32>
    %v998 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v999 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1000 = stablehlo.multiply %v997, %v998 : tensor<64x196x384xf32>
    %v1001 = stablehlo.add %v1000, %v999 : tensor<64x196x384xf32>
    %v1002 = stablehlo.reshape %v1001 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1004 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1005 = stablehlo.multiply %v1003, %v1004 : tensor<64x196x384xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1008 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1009 = stablehlo.add %v1007, %v1008 : tensor<64x196x384xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1012 = stablehlo.transpose %v1011, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1015 = stablehlo.convolution(%v1014, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1016 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1017 = stablehlo.add %v1015, %v1016 : tensor<64x1536x14x14xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1020 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1021 = stablehlo.multiply %v1020, %v1019 : tensor<64x1536x14x14xf32>
    %v1022 = stablehlo.negate %v1019 : tensor<64x1536x14x14xf32>
    %v1023 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1024 = stablehlo.multiply %v1022, %v1023 : tensor<64x1536x14x14xf32>
    %v1025 = chlo.erfc %v1024 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1026 = stablehlo.multiply %v1021, %v1025 : tensor<64x1536x14x14xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1029 = stablehlo.convolution(%v1028, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1030 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1031 = stablehlo.add %v1029, %v1030 : tensor<64x384x14x14xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1034 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1035 = stablehlo.multiply %v1033, %v1034 : tensor<64x384x14x14xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1038 = stablehlo.reshape %v974 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1039 = stablehlo.add %v1037, %v1038 : tensor<64x384x14x14xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1041 = stablehlo.reshape %v1040 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1042 = stablehlo.convolution(%v1041, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1043 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1044 = stablehlo.add %v1042, %v1043 : tensor<64x384x14x14xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1046 = stablehlo.reshape %v1045 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1047 = stablehlo.transpose %v1046, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1051 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1052 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1053 = stablehlo.reduce(%v1049 init: %v1050) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1054 = stablehlo.broadcast_in_dim %v1053, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1055 = stablehlo.divide %v1054, %v1051 : tensor<64x196x384xf32>
    %v1056 = stablehlo.subtract %v1049, %v1055 : tensor<64x196x384xf32>
    %v1057 = stablehlo.multiply %v1056, %v1056 : tensor<64x196x384xf32>
    %v1058 = stablehlo.reduce(%v1057 init: %v1050) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1059 = stablehlo.broadcast_in_dim %v1058, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1060 = stablehlo.divide %v1059, %v1051 : tensor<64x196x384xf32>
    %v1061 = stablehlo.add %v1060, %v1052 : tensor<64x196x384xf32>
    %v1062 = stablehlo.rsqrt %v1061 : tensor<64x196x384xf32>
    %v1063 = stablehlo.multiply %v1056, %v1062 : tensor<64x196x384xf32>
    %v1064 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1065 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1066 = stablehlo.multiply %v1063, %v1064 : tensor<64x196x384xf32>
    %v1067 = stablehlo.add %v1066, %v1065 : tensor<64x196x384xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1070 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1071 = stablehlo.multiply %v1069, %v1070 : tensor<64x196x384xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1074 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1075 = stablehlo.add %v1073, %v1074 : tensor<64x196x384xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1078 = stablehlo.transpose %v1077, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1081 = stablehlo.convolution(%v1080, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1082 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1083 = stablehlo.add %v1081, %v1082 : tensor<64x1536x14x14xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1086 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1087 = stablehlo.multiply %v1086, %v1085 : tensor<64x1536x14x14xf32>
    %v1088 = stablehlo.negate %v1085 : tensor<64x1536x14x14xf32>
    %v1089 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1090 = stablehlo.multiply %v1088, %v1089 : tensor<64x1536x14x14xf32>
    %v1091 = chlo.erfc %v1090 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1092 = stablehlo.multiply %v1087, %v1091 : tensor<64x1536x14x14xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1095 = stablehlo.convolution(%v1094, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1096 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1097 = stablehlo.add %v1095, %v1096 : tensor<64x384x14x14xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1100 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1101 = stablehlo.multiply %v1099, %v1100 : tensor<64x384x14x14xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1104 = stablehlo.reshape %v1040 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1105 = stablehlo.add %v1103, %v1104 : tensor<64x384x14x14xf32>
    %v1106 = stablehlo.reshape %v1105 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1107 = stablehlo.reshape %v1106 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1108 = stablehlo.convolution(%v1107, %s2b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1109 = stablehlo.broadcast_in_dim %s2b9db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1110 = stablehlo.add %v1108, %v1109 : tensor<64x384x14x14xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1113 = stablehlo.transpose %v1112, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1117 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1118 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1119 = stablehlo.reduce(%v1115 init: %v1116) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1120 = stablehlo.broadcast_in_dim %v1119, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1121 = stablehlo.divide %v1120, %v1117 : tensor<64x196x384xf32>
    %v1122 = stablehlo.subtract %v1115, %v1121 : tensor<64x196x384xf32>
    %v1123 = stablehlo.multiply %v1122, %v1122 : tensor<64x196x384xf32>
    %v1124 = stablehlo.reduce(%v1123 init: %v1116) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1125 = stablehlo.broadcast_in_dim %v1124, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1126 = stablehlo.divide %v1125, %v1117 : tensor<64x196x384xf32>
    %v1127 = stablehlo.add %v1126, %v1118 : tensor<64x196x384xf32>
    %v1128 = stablehlo.rsqrt %v1127 : tensor<64x196x384xf32>
    %v1129 = stablehlo.multiply %v1122, %v1128 : tensor<64x196x384xf32>
    %v1130 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1131 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1132 = stablehlo.multiply %v1129, %v1130 : tensor<64x196x384xf32>
    %v1133 = stablehlo.add %v1132, %v1131 : tensor<64x196x384xf32>
    %v1134 = stablehlo.reshape %v1133 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1135 = stablehlo.reshape %v1134 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1136 = stablehlo.broadcast_in_dim %s2b9ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1137 = stablehlo.multiply %v1135, %v1136 : tensor<64x196x384xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1140 = stablehlo.broadcast_in_dim %s2b9nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1141 = stablehlo.add %v1139, %v1140 : tensor<64x196x384xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1144 = stablehlo.transpose %v1143, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1147 = stablehlo.convolution(%v1146, %s2b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1148 = stablehlo.broadcast_in_dim %s2b9eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1149 = stablehlo.add %v1147, %v1148 : tensor<64x1536x14x14xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1151 = stablehlo.reshape %v1150 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1152 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1153 = stablehlo.multiply %v1152, %v1151 : tensor<64x1536x14x14xf32>
    %v1154 = stablehlo.negate %v1151 : tensor<64x1536x14x14xf32>
    %v1155 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1156 = stablehlo.multiply %v1154, %v1155 : tensor<64x1536x14x14xf32>
    %v1157 = chlo.erfc %v1156 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1158 = stablehlo.multiply %v1153, %v1157 : tensor<64x1536x14x14xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1161 = stablehlo.convolution(%v1160, %s2b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1162 = stablehlo.broadcast_in_dim %s2b9pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1163 = stablehlo.add %v1161, %v1162 : tensor<64x384x14x14xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1166 = stablehlo.broadcast_in_dim %s2b9lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1167 = stablehlo.multiply %v1165, %v1166 : tensor<64x384x14x14xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1170 = stablehlo.reshape %v1106 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1171 = stablehlo.add %v1169, %v1170 : tensor<64x384x14x14xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1174 = stablehlo.convolution(%v1173, %s2b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1175 = stablehlo.broadcast_in_dim %s2b10db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1176 = stablehlo.add %v1174, %v1175 : tensor<64x384x14x14xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1178 = stablehlo.reshape %v1177 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1179 = stablehlo.transpose %v1178, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1180 = stablehlo.reshape %v1179 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1181 = stablehlo.reshape %v1180 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1183 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1184 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1185 = stablehlo.reduce(%v1181 init: %v1182) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1186 = stablehlo.broadcast_in_dim %v1185, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1187 = stablehlo.divide %v1186, %v1183 : tensor<64x196x384xf32>
    %v1188 = stablehlo.subtract %v1181, %v1187 : tensor<64x196x384xf32>
    %v1189 = stablehlo.multiply %v1188, %v1188 : tensor<64x196x384xf32>
    %v1190 = stablehlo.reduce(%v1189 init: %v1182) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1191 = stablehlo.broadcast_in_dim %v1190, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1192 = stablehlo.divide %v1191, %v1183 : tensor<64x196x384xf32>
    %v1193 = stablehlo.add %v1192, %v1184 : tensor<64x196x384xf32>
    %v1194 = stablehlo.rsqrt %v1193 : tensor<64x196x384xf32>
    %v1195 = stablehlo.multiply %v1188, %v1194 : tensor<64x196x384xf32>
    %v1196 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1197 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1198 = stablehlo.multiply %v1195, %v1196 : tensor<64x196x384xf32>
    %v1199 = stablehlo.add %v1198, %v1197 : tensor<64x196x384xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1201 = stablehlo.reshape %v1200 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1202 = stablehlo.broadcast_in_dim %s2b10ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1203 = stablehlo.multiply %v1201, %v1202 : tensor<64x196x384xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1205 = stablehlo.reshape %v1204 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1206 = stablehlo.broadcast_in_dim %s2b10nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1207 = stablehlo.add %v1205, %v1206 : tensor<64x196x384xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1210 = stablehlo.transpose %v1209, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1211 = stablehlo.reshape %v1210 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1212 = stablehlo.reshape %v1211 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1213 = stablehlo.convolution(%v1212, %s2b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1214 = stablehlo.broadcast_in_dim %s2b10eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1215 = stablehlo.add %v1213, %v1214 : tensor<64x1536x14x14xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1218 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1219 = stablehlo.multiply %v1218, %v1217 : tensor<64x1536x14x14xf32>
    %v1220 = stablehlo.negate %v1217 : tensor<64x1536x14x14xf32>
    %v1221 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1222 = stablehlo.multiply %v1220, %v1221 : tensor<64x1536x14x14xf32>
    %v1223 = chlo.erfc %v1222 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1224 = stablehlo.multiply %v1219, %v1223 : tensor<64x1536x14x14xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1227 = stablehlo.convolution(%v1226, %s2b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1228 = stablehlo.broadcast_in_dim %s2b10pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1229 = stablehlo.add %v1227, %v1228 : tensor<64x384x14x14xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1232 = stablehlo.broadcast_in_dim %s2b10lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1233 = stablehlo.multiply %v1231, %v1232 : tensor<64x384x14x14xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1236 = stablehlo.reshape %v1172 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1237 = stablehlo.add %v1235, %v1236 : tensor<64x384x14x14xf32>
    %v1238 = stablehlo.reshape %v1237 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1239 = stablehlo.reshape %v1238 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1240 = stablehlo.convolution(%v1239, %s2b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1241 = stablehlo.broadcast_in_dim %s2b11db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1242 = stablehlo.add %v1240, %v1241 : tensor<64x384x14x14xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1245 = stablehlo.transpose %v1244, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1246 = stablehlo.reshape %v1245 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1248 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1249 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1250 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1251 = stablehlo.reduce(%v1247 init: %v1248) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1252 = stablehlo.broadcast_in_dim %v1251, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1253 = stablehlo.divide %v1252, %v1249 : tensor<64x196x384xf32>
    %v1254 = stablehlo.subtract %v1247, %v1253 : tensor<64x196x384xf32>
    %v1255 = stablehlo.multiply %v1254, %v1254 : tensor<64x196x384xf32>
    %v1256 = stablehlo.reduce(%v1255 init: %v1248) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1257 = stablehlo.broadcast_in_dim %v1256, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1258 = stablehlo.divide %v1257, %v1249 : tensor<64x196x384xf32>
    %v1259 = stablehlo.add %v1258, %v1250 : tensor<64x196x384xf32>
    %v1260 = stablehlo.rsqrt %v1259 : tensor<64x196x384xf32>
    %v1261 = stablehlo.multiply %v1254, %v1260 : tensor<64x196x384xf32>
    %v1262 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1263 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1264 = stablehlo.multiply %v1261, %v1262 : tensor<64x196x384xf32>
    %v1265 = stablehlo.add %v1264, %v1263 : tensor<64x196x384xf32>
    %v1266 = stablehlo.reshape %v1265 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1268 = stablehlo.broadcast_in_dim %s2b11ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1269 = stablehlo.multiply %v1267, %v1268 : tensor<64x196x384xf32>
    %v1270 = stablehlo.reshape %v1269 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1272 = stablehlo.broadcast_in_dim %s2b11nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1273 = stablehlo.add %v1271, %v1272 : tensor<64x196x384xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1276 = stablehlo.transpose %v1275, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1279 = stablehlo.convolution(%v1278, %s2b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1280 = stablehlo.broadcast_in_dim %s2b11eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1281 = stablehlo.add %v1279, %v1280 : tensor<64x1536x14x14xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1284 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1285 = stablehlo.multiply %v1284, %v1283 : tensor<64x1536x14x14xf32>
    %v1286 = stablehlo.negate %v1283 : tensor<64x1536x14x14xf32>
    %v1287 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1288 = stablehlo.multiply %v1286, %v1287 : tensor<64x1536x14x14xf32>
    %v1289 = chlo.erfc %v1288 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1290 = stablehlo.multiply %v1285, %v1289 : tensor<64x1536x14x14xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1293 = stablehlo.convolution(%v1292, %s2b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1294 = stablehlo.broadcast_in_dim %s2b11pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1295 = stablehlo.add %v1293, %v1294 : tensor<64x384x14x14xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1297 = stablehlo.reshape %v1296 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1298 = stablehlo.broadcast_in_dim %s2b11lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1299 = stablehlo.multiply %v1297, %v1298 : tensor<64x384x14x14xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1302 = stablehlo.reshape %v1238 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1303 = stablehlo.add %v1301, %v1302 : tensor<64x384x14x14xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1306 = stablehlo.convolution(%v1305, %s2b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1307 = stablehlo.broadcast_in_dim %s2b12db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1308 = stablehlo.add %v1306, %v1307 : tensor<64x384x14x14xf32>
    %v1309 = stablehlo.reshape %v1308 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1311 = stablehlo.transpose %v1310, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1315 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1316 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1317 = stablehlo.reduce(%v1313 init: %v1314) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1318 = stablehlo.broadcast_in_dim %v1317, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1319 = stablehlo.divide %v1318, %v1315 : tensor<64x196x384xf32>
    %v1320 = stablehlo.subtract %v1313, %v1319 : tensor<64x196x384xf32>
    %v1321 = stablehlo.multiply %v1320, %v1320 : tensor<64x196x384xf32>
    %v1322 = stablehlo.reduce(%v1321 init: %v1314) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1323 = stablehlo.broadcast_in_dim %v1322, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1324 = stablehlo.divide %v1323, %v1315 : tensor<64x196x384xf32>
    %v1325 = stablehlo.add %v1324, %v1316 : tensor<64x196x384xf32>
    %v1326 = stablehlo.rsqrt %v1325 : tensor<64x196x384xf32>
    %v1327 = stablehlo.multiply %v1320, %v1326 : tensor<64x196x384xf32>
    %v1328 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1329 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1330 = stablehlo.multiply %v1327, %v1328 : tensor<64x196x384xf32>
    %v1331 = stablehlo.add %v1330, %v1329 : tensor<64x196x384xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1334 = stablehlo.broadcast_in_dim %s2b12ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1335 = stablehlo.multiply %v1333, %v1334 : tensor<64x196x384xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1338 = stablehlo.broadcast_in_dim %s2b12nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1339 = stablehlo.add %v1337, %v1338 : tensor<64x196x384xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1342 = stablehlo.transpose %v1341, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1345 = stablehlo.convolution(%v1344, %s2b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1346 = stablehlo.broadcast_in_dim %s2b12eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1347 = stablehlo.add %v1345, %v1346 : tensor<64x1536x14x14xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1350 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1351 = stablehlo.multiply %v1350, %v1349 : tensor<64x1536x14x14xf32>
    %v1352 = stablehlo.negate %v1349 : tensor<64x1536x14x14xf32>
    %v1353 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1354 = stablehlo.multiply %v1352, %v1353 : tensor<64x1536x14x14xf32>
    %v1355 = chlo.erfc %v1354 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1356 = stablehlo.multiply %v1351, %v1355 : tensor<64x1536x14x14xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1359 = stablehlo.convolution(%v1358, %s2b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1360 = stablehlo.broadcast_in_dim %s2b12pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1361 = stablehlo.add %v1359, %v1360 : tensor<64x384x14x14xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1363 = stablehlo.reshape %v1362 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1364 = stablehlo.broadcast_in_dim %s2b12lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1365 = stablehlo.multiply %v1363, %v1364 : tensor<64x384x14x14xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1368 = stablehlo.reshape %v1304 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1369 = stablehlo.add %v1367, %v1368 : tensor<64x384x14x14xf32>
    %v1370 = stablehlo.reshape %v1369 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1372 = stablehlo.convolution(%v1371, %s2b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1373 = stablehlo.broadcast_in_dim %s2b13db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1374 = stablehlo.add %v1372, %v1373 : tensor<64x384x14x14xf32>
    %v1375 = stablehlo.reshape %v1374 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1377 = stablehlo.transpose %v1376, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1378 = stablehlo.reshape %v1377 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1381 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1382 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1383 = stablehlo.reduce(%v1379 init: %v1380) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1384 = stablehlo.broadcast_in_dim %v1383, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1385 = stablehlo.divide %v1384, %v1381 : tensor<64x196x384xf32>
    %v1386 = stablehlo.subtract %v1379, %v1385 : tensor<64x196x384xf32>
    %v1387 = stablehlo.multiply %v1386, %v1386 : tensor<64x196x384xf32>
    %v1388 = stablehlo.reduce(%v1387 init: %v1380) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1389 = stablehlo.broadcast_in_dim %v1388, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1390 = stablehlo.divide %v1389, %v1381 : tensor<64x196x384xf32>
    %v1391 = stablehlo.add %v1390, %v1382 : tensor<64x196x384xf32>
    %v1392 = stablehlo.rsqrt %v1391 : tensor<64x196x384xf32>
    %v1393 = stablehlo.multiply %v1386, %v1392 : tensor<64x196x384xf32>
    %v1394 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1395 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1396 = stablehlo.multiply %v1393, %v1394 : tensor<64x196x384xf32>
    %v1397 = stablehlo.add %v1396, %v1395 : tensor<64x196x384xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1399 = stablehlo.reshape %v1398 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1400 = stablehlo.broadcast_in_dim %s2b13ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1401 = stablehlo.multiply %v1399, %v1400 : tensor<64x196x384xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1404 = stablehlo.broadcast_in_dim %s2b13nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1405 = stablehlo.add %v1403, %v1404 : tensor<64x196x384xf32>
    %v1406 = stablehlo.reshape %v1405 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1408 = stablehlo.transpose %v1407, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1410 = stablehlo.reshape %v1409 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1411 = stablehlo.convolution(%v1410, %s2b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1412 = stablehlo.broadcast_in_dim %s2b13eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1413 = stablehlo.add %v1411, %v1412 : tensor<64x1536x14x14xf32>
    %v1414 = stablehlo.reshape %v1413 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1416 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1417 = stablehlo.multiply %v1416, %v1415 : tensor<64x1536x14x14xf32>
    %v1418 = stablehlo.negate %v1415 : tensor<64x1536x14x14xf32>
    %v1419 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1420 = stablehlo.multiply %v1418, %v1419 : tensor<64x1536x14x14xf32>
    %v1421 = chlo.erfc %v1420 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1422 = stablehlo.multiply %v1417, %v1421 : tensor<64x1536x14x14xf32>
    %v1423 = stablehlo.reshape %v1422 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1424 = stablehlo.reshape %v1423 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1425 = stablehlo.convolution(%v1424, %s2b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1426 = stablehlo.broadcast_in_dim %s2b13pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1427 = stablehlo.add %v1425, %v1426 : tensor<64x384x14x14xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1430 = stablehlo.broadcast_in_dim %s2b13lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1431 = stablehlo.multiply %v1429, %v1430 : tensor<64x384x14x14xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1434 = stablehlo.reshape %v1370 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1435 = stablehlo.add %v1433, %v1434 : tensor<64x384x14x14xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1438 = stablehlo.convolution(%v1437, %s2b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1439 = stablehlo.broadcast_in_dim %s2b14db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1440 = stablehlo.add %v1438, %v1439 : tensor<64x384x14x14xf32>
    %v1441 = stablehlo.reshape %v1440 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1442 = stablehlo.reshape %v1441 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1443 = stablehlo.transpose %v1442, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1446 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1447 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1448 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1449 = stablehlo.reduce(%v1445 init: %v1446) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1450 = stablehlo.broadcast_in_dim %v1449, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1451 = stablehlo.divide %v1450, %v1447 : tensor<64x196x384xf32>
    %v1452 = stablehlo.subtract %v1445, %v1451 : tensor<64x196x384xf32>
    %v1453 = stablehlo.multiply %v1452, %v1452 : tensor<64x196x384xf32>
    %v1454 = stablehlo.reduce(%v1453 init: %v1446) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1455 = stablehlo.broadcast_in_dim %v1454, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1456 = stablehlo.divide %v1455, %v1447 : tensor<64x196x384xf32>
    %v1457 = stablehlo.add %v1456, %v1448 : tensor<64x196x384xf32>
    %v1458 = stablehlo.rsqrt %v1457 : tensor<64x196x384xf32>
    %v1459 = stablehlo.multiply %v1452, %v1458 : tensor<64x196x384xf32>
    %v1460 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1461 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1462 = stablehlo.multiply %v1459, %v1460 : tensor<64x196x384xf32>
    %v1463 = stablehlo.add %v1462, %v1461 : tensor<64x196x384xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1465 = stablehlo.reshape %v1464 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1466 = stablehlo.broadcast_in_dim %s2b14ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1467 = stablehlo.multiply %v1465, %v1466 : tensor<64x196x384xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1469 = stablehlo.reshape %v1468 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1470 = stablehlo.broadcast_in_dim %s2b14nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1471 = stablehlo.add %v1469, %v1470 : tensor<64x196x384xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1474 = stablehlo.transpose %v1473, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1476 = stablehlo.reshape %v1475 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1477 = stablehlo.convolution(%v1476, %s2b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1478 = stablehlo.broadcast_in_dim %s2b14eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1479 = stablehlo.add %v1477, %v1478 : tensor<64x1536x14x14xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1481 = stablehlo.reshape %v1480 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1482 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1483 = stablehlo.multiply %v1482, %v1481 : tensor<64x1536x14x14xf32>
    %v1484 = stablehlo.negate %v1481 : tensor<64x1536x14x14xf32>
    %v1485 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1486 = stablehlo.multiply %v1484, %v1485 : tensor<64x1536x14x14xf32>
    %v1487 = chlo.erfc %v1486 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1488 = stablehlo.multiply %v1483, %v1487 : tensor<64x1536x14x14xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1491 = stablehlo.convolution(%v1490, %s2b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1492 = stablehlo.broadcast_in_dim %s2b14pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1493 = stablehlo.add %v1491, %v1492 : tensor<64x384x14x14xf32>
    %v1494 = stablehlo.reshape %v1493 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1495 = stablehlo.reshape %v1494 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1496 = stablehlo.broadcast_in_dim %s2b14lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1497 = stablehlo.multiply %v1495, %v1496 : tensor<64x384x14x14xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1499 = stablehlo.reshape %v1498 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1500 = stablehlo.reshape %v1436 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1501 = stablehlo.add %v1499, %v1500 : tensor<64x384x14x14xf32>
    %v1502 = stablehlo.reshape %v1501 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1503 = stablehlo.reshape %v1502 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1504 = stablehlo.convolution(%v1503, %s2b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1505 = stablehlo.broadcast_in_dim %s2b15db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1506 = stablehlo.add %v1504, %v1505 : tensor<64x384x14x14xf32>
    %v1507 = stablehlo.reshape %v1506 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1508 = stablehlo.reshape %v1507 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1509 = stablehlo.transpose %v1508, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1510 = stablehlo.reshape %v1509 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1513 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1514 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1515 = stablehlo.reduce(%v1511 init: %v1512) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1516 = stablehlo.broadcast_in_dim %v1515, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1517 = stablehlo.divide %v1516, %v1513 : tensor<64x196x384xf32>
    %v1518 = stablehlo.subtract %v1511, %v1517 : tensor<64x196x384xf32>
    %v1519 = stablehlo.multiply %v1518, %v1518 : tensor<64x196x384xf32>
    %v1520 = stablehlo.reduce(%v1519 init: %v1512) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1521 = stablehlo.broadcast_in_dim %v1520, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1522 = stablehlo.divide %v1521, %v1513 : tensor<64x196x384xf32>
    %v1523 = stablehlo.add %v1522, %v1514 : tensor<64x196x384xf32>
    %v1524 = stablehlo.rsqrt %v1523 : tensor<64x196x384xf32>
    %v1525 = stablehlo.multiply %v1518, %v1524 : tensor<64x196x384xf32>
    %v1526 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1527 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1528 = stablehlo.multiply %v1525, %v1526 : tensor<64x196x384xf32>
    %v1529 = stablehlo.add %v1528, %v1527 : tensor<64x196x384xf32>
    %v1530 = stablehlo.reshape %v1529 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1532 = stablehlo.broadcast_in_dim %s2b15ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1533 = stablehlo.multiply %v1531, %v1532 : tensor<64x196x384xf32>
    %v1534 = stablehlo.reshape %v1533 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1535 = stablehlo.reshape %v1534 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1536 = stablehlo.broadcast_in_dim %s2b15nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1537 = stablehlo.add %v1535, %v1536 : tensor<64x196x384xf32>
    %v1538 = stablehlo.reshape %v1537 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1539 = stablehlo.reshape %v1538 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1540 = stablehlo.transpose %v1539, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1541 = stablehlo.reshape %v1540 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1542 = stablehlo.reshape %v1541 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1543 = stablehlo.convolution(%v1542, %s2b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1544 = stablehlo.broadcast_in_dim %s2b15eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1545 = stablehlo.add %v1543, %v1544 : tensor<64x1536x14x14xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1547 = stablehlo.reshape %v1546 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1548 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1549 = stablehlo.multiply %v1548, %v1547 : tensor<64x1536x14x14xf32>
    %v1550 = stablehlo.negate %v1547 : tensor<64x1536x14x14xf32>
    %v1551 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1552 = stablehlo.multiply %v1550, %v1551 : tensor<64x1536x14x14xf32>
    %v1553 = chlo.erfc %v1552 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1554 = stablehlo.multiply %v1549, %v1553 : tensor<64x1536x14x14xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1557 = stablehlo.convolution(%v1556, %s2b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1558 = stablehlo.broadcast_in_dim %s2b15pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1559 = stablehlo.add %v1557, %v1558 : tensor<64x384x14x14xf32>
    %v1560 = stablehlo.reshape %v1559 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1562 = stablehlo.broadcast_in_dim %s2b15lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1563 = stablehlo.multiply %v1561, %v1562 : tensor<64x384x14x14xf32>
    %v1564 = stablehlo.reshape %v1563 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1566 = stablehlo.reshape %v1502 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1567 = stablehlo.add %v1565, %v1566 : tensor<64x384x14x14xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1570 = stablehlo.convolution(%v1569, %s2b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1571 = stablehlo.broadcast_in_dim %s2b16db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1572 = stablehlo.add %v1570, %v1571 : tensor<64x384x14x14xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1574 = stablehlo.reshape %v1573 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1575 = stablehlo.transpose %v1574, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1576 = stablehlo.reshape %v1575 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1577 = stablehlo.reshape %v1576 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1578 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1579 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1580 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1581 = stablehlo.reduce(%v1577 init: %v1578) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1582 = stablehlo.broadcast_in_dim %v1581, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1583 = stablehlo.divide %v1582, %v1579 : tensor<64x196x384xf32>
    %v1584 = stablehlo.subtract %v1577, %v1583 : tensor<64x196x384xf32>
    %v1585 = stablehlo.multiply %v1584, %v1584 : tensor<64x196x384xf32>
    %v1586 = stablehlo.reduce(%v1585 init: %v1578) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1587 = stablehlo.broadcast_in_dim %v1586, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1588 = stablehlo.divide %v1587, %v1579 : tensor<64x196x384xf32>
    %v1589 = stablehlo.add %v1588, %v1580 : tensor<64x196x384xf32>
    %v1590 = stablehlo.rsqrt %v1589 : tensor<64x196x384xf32>
    %v1591 = stablehlo.multiply %v1584, %v1590 : tensor<64x196x384xf32>
    %v1592 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1593 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1594 = stablehlo.multiply %v1591, %v1592 : tensor<64x196x384xf32>
    %v1595 = stablehlo.add %v1594, %v1593 : tensor<64x196x384xf32>
    %v1596 = stablehlo.reshape %v1595 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1597 = stablehlo.reshape %v1596 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1598 = stablehlo.broadcast_in_dim %s2b16ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1599 = stablehlo.multiply %v1597, %v1598 : tensor<64x196x384xf32>
    %v1600 = stablehlo.reshape %v1599 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1601 = stablehlo.reshape %v1600 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1602 = stablehlo.broadcast_in_dim %s2b16nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1603 = stablehlo.add %v1601, %v1602 : tensor<64x196x384xf32>
    %v1604 = stablehlo.reshape %v1603 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1605 = stablehlo.reshape %v1604 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1606 = stablehlo.transpose %v1605, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1608 = stablehlo.reshape %v1607 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1609 = stablehlo.convolution(%v1608, %s2b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1610 = stablehlo.broadcast_in_dim %s2b16eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1611 = stablehlo.add %v1609, %v1610 : tensor<64x1536x14x14xf32>
    %v1612 = stablehlo.reshape %v1611 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1613 = stablehlo.reshape %v1612 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1614 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1615 = stablehlo.multiply %v1614, %v1613 : tensor<64x1536x14x14xf32>
    %v1616 = stablehlo.negate %v1613 : tensor<64x1536x14x14xf32>
    %v1617 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1618 = stablehlo.multiply %v1616, %v1617 : tensor<64x1536x14x14xf32>
    %v1619 = chlo.erfc %v1618 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1620 = stablehlo.multiply %v1615, %v1619 : tensor<64x1536x14x14xf32>
    %v1621 = stablehlo.reshape %v1620 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1623 = stablehlo.convolution(%v1622, %s2b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1624 = stablehlo.broadcast_in_dim %s2b16pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1625 = stablehlo.add %v1623, %v1624 : tensor<64x384x14x14xf32>
    %v1626 = stablehlo.reshape %v1625 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1627 = stablehlo.reshape %v1626 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1628 = stablehlo.broadcast_in_dim %s2b16lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1629 = stablehlo.multiply %v1627, %v1628 : tensor<64x384x14x14xf32>
    %v1630 = stablehlo.reshape %v1629 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1631 = stablehlo.reshape %v1630 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1632 = stablehlo.reshape %v1568 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1633 = stablehlo.add %v1631, %v1632 : tensor<64x384x14x14xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1635 = stablehlo.reshape %v1634 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1636 = stablehlo.convolution(%v1635, %s2b17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1637 = stablehlo.broadcast_in_dim %s2b17db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1638 = stablehlo.add %v1636, %v1637 : tensor<64x384x14x14xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1640 = stablehlo.reshape %v1639 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1641 = stablehlo.transpose %v1640, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1645 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1646 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1647 = stablehlo.reduce(%v1643 init: %v1644) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1648 = stablehlo.broadcast_in_dim %v1647, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1649 = stablehlo.divide %v1648, %v1645 : tensor<64x196x384xf32>
    %v1650 = stablehlo.subtract %v1643, %v1649 : tensor<64x196x384xf32>
    %v1651 = stablehlo.multiply %v1650, %v1650 : tensor<64x196x384xf32>
    %v1652 = stablehlo.reduce(%v1651 init: %v1644) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1653 = stablehlo.broadcast_in_dim %v1652, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1654 = stablehlo.divide %v1653, %v1645 : tensor<64x196x384xf32>
    %v1655 = stablehlo.add %v1654, %v1646 : tensor<64x196x384xf32>
    %v1656 = stablehlo.rsqrt %v1655 : tensor<64x196x384xf32>
    %v1657 = stablehlo.multiply %v1650, %v1656 : tensor<64x196x384xf32>
    %v1658 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1659 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1660 = stablehlo.multiply %v1657, %v1658 : tensor<64x196x384xf32>
    %v1661 = stablehlo.add %v1660, %v1659 : tensor<64x196x384xf32>
    %v1662 = stablehlo.reshape %v1661 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1664 = stablehlo.broadcast_in_dim %s2b17ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1665 = stablehlo.multiply %v1663, %v1664 : tensor<64x196x384xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1667 = stablehlo.reshape %v1666 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1668 = stablehlo.broadcast_in_dim %s2b17nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1669 = stablehlo.add %v1667, %v1668 : tensor<64x196x384xf32>
    %v1670 = stablehlo.reshape %v1669 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1671 = stablehlo.reshape %v1670 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1672 = stablehlo.transpose %v1671, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1674 = stablehlo.reshape %v1673 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1675 = stablehlo.convolution(%v1674, %s2b17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1676 = stablehlo.broadcast_in_dim %s2b17eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1677 = stablehlo.add %v1675, %v1676 : tensor<64x1536x14x14xf32>
    %v1678 = stablehlo.reshape %v1677 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1679 = stablehlo.reshape %v1678 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1680 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1681 = stablehlo.multiply %v1680, %v1679 : tensor<64x1536x14x14xf32>
    %v1682 = stablehlo.negate %v1679 : tensor<64x1536x14x14xf32>
    %v1683 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1684 = stablehlo.multiply %v1682, %v1683 : tensor<64x1536x14x14xf32>
    %v1685 = chlo.erfc %v1684 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1686 = stablehlo.multiply %v1681, %v1685 : tensor<64x1536x14x14xf32>
    %v1687 = stablehlo.reshape %v1686 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1688 = stablehlo.reshape %v1687 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1689 = stablehlo.convolution(%v1688, %s2b17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1690 = stablehlo.broadcast_in_dim %s2b17pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1691 = stablehlo.add %v1689, %v1690 : tensor<64x384x14x14xf32>
    %v1692 = stablehlo.reshape %v1691 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1694 = stablehlo.broadcast_in_dim %s2b17lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1695 = stablehlo.multiply %v1693, %v1694 : tensor<64x384x14x14xf32>
    %v1696 = stablehlo.reshape %v1695 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1698 = stablehlo.reshape %v1634 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1699 = stablehlo.add %v1697, %v1698 : tensor<64x384x14x14xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1702 = stablehlo.convolution(%v1701, %s2b18dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1703 = stablehlo.broadcast_in_dim %s2b18db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1704 = stablehlo.add %v1702, %v1703 : tensor<64x384x14x14xf32>
    %v1705 = stablehlo.reshape %v1704 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1706 = stablehlo.reshape %v1705 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1707 = stablehlo.transpose %v1706, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1708 = stablehlo.reshape %v1707 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1709 = stablehlo.reshape %v1708 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1710 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1711 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1712 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1713 = stablehlo.reduce(%v1709 init: %v1710) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1714 = stablehlo.broadcast_in_dim %v1713, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1715 = stablehlo.divide %v1714, %v1711 : tensor<64x196x384xf32>
    %v1716 = stablehlo.subtract %v1709, %v1715 : tensor<64x196x384xf32>
    %v1717 = stablehlo.multiply %v1716, %v1716 : tensor<64x196x384xf32>
    %v1718 = stablehlo.reduce(%v1717 init: %v1710) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1719 = stablehlo.broadcast_in_dim %v1718, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1720 = stablehlo.divide %v1719, %v1711 : tensor<64x196x384xf32>
    %v1721 = stablehlo.add %v1720, %v1712 : tensor<64x196x384xf32>
    %v1722 = stablehlo.rsqrt %v1721 : tensor<64x196x384xf32>
    %v1723 = stablehlo.multiply %v1716, %v1722 : tensor<64x196x384xf32>
    %v1724 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1725 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1726 = stablehlo.multiply %v1723, %v1724 : tensor<64x196x384xf32>
    %v1727 = stablehlo.add %v1726, %v1725 : tensor<64x196x384xf32>
    %v1728 = stablehlo.reshape %v1727 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1729 = stablehlo.reshape %v1728 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1730 = stablehlo.broadcast_in_dim %s2b18ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1731 = stablehlo.multiply %v1729, %v1730 : tensor<64x196x384xf32>
    %v1732 = stablehlo.reshape %v1731 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1733 = stablehlo.reshape %v1732 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1734 = stablehlo.broadcast_in_dim %s2b18nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1735 = stablehlo.add %v1733, %v1734 : tensor<64x196x384xf32>
    %v1736 = stablehlo.reshape %v1735 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1737 = stablehlo.reshape %v1736 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1738 = stablehlo.transpose %v1737, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1739 = stablehlo.reshape %v1738 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1740 = stablehlo.reshape %v1739 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1741 = stablehlo.convolution(%v1740, %s2b18eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1742 = stablehlo.broadcast_in_dim %s2b18eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1743 = stablehlo.add %v1741, %v1742 : tensor<64x1536x14x14xf32>
    %v1744 = stablehlo.reshape %v1743 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1745 = stablehlo.reshape %v1744 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1746 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1747 = stablehlo.multiply %v1746, %v1745 : tensor<64x1536x14x14xf32>
    %v1748 = stablehlo.negate %v1745 : tensor<64x1536x14x14xf32>
    %v1749 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1750 = stablehlo.multiply %v1748, %v1749 : tensor<64x1536x14x14xf32>
    %v1751 = chlo.erfc %v1750 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1752 = stablehlo.multiply %v1747, %v1751 : tensor<64x1536x14x14xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1754 = stablehlo.reshape %v1753 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1755 = stablehlo.convolution(%v1754, %s2b18pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1756 = stablehlo.broadcast_in_dim %s2b18pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1757 = stablehlo.add %v1755, %v1756 : tensor<64x384x14x14xf32>
    %v1758 = stablehlo.reshape %v1757 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1759 = stablehlo.reshape %v1758 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1760 = stablehlo.broadcast_in_dim %s2b18lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1761 = stablehlo.multiply %v1759, %v1760 : tensor<64x384x14x14xf32>
    %v1762 = stablehlo.reshape %v1761 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1763 = stablehlo.reshape %v1762 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1764 = stablehlo.reshape %v1700 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1765 = stablehlo.add %v1763, %v1764 : tensor<64x384x14x14xf32>
    %v1766 = stablehlo.reshape %v1765 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1767 = stablehlo.reshape %v1766 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1768 = stablehlo.convolution(%v1767, %s2b19dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1769 = stablehlo.broadcast_in_dim %s2b19db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1770 = stablehlo.add %v1768, %v1769 : tensor<64x384x14x14xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1772 = stablehlo.reshape %v1771 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1773 = stablehlo.transpose %v1772, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1774 = stablehlo.reshape %v1773 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1775 = stablehlo.reshape %v1774 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1776 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1777 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1778 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1779 = stablehlo.reduce(%v1775 init: %v1776) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1780 = stablehlo.broadcast_in_dim %v1779, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1781 = stablehlo.divide %v1780, %v1777 : tensor<64x196x384xf32>
    %v1782 = stablehlo.subtract %v1775, %v1781 : tensor<64x196x384xf32>
    %v1783 = stablehlo.multiply %v1782, %v1782 : tensor<64x196x384xf32>
    %v1784 = stablehlo.reduce(%v1783 init: %v1776) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1785 = stablehlo.broadcast_in_dim %v1784, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1786 = stablehlo.divide %v1785, %v1777 : tensor<64x196x384xf32>
    %v1787 = stablehlo.add %v1786, %v1778 : tensor<64x196x384xf32>
    %v1788 = stablehlo.rsqrt %v1787 : tensor<64x196x384xf32>
    %v1789 = stablehlo.multiply %v1782, %v1788 : tensor<64x196x384xf32>
    %v1790 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1791 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1792 = stablehlo.multiply %v1789, %v1790 : tensor<64x196x384xf32>
    %v1793 = stablehlo.add %v1792, %v1791 : tensor<64x196x384xf32>
    %v1794 = stablehlo.reshape %v1793 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1795 = stablehlo.reshape %v1794 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1796 = stablehlo.broadcast_in_dim %s2b19ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1797 = stablehlo.multiply %v1795, %v1796 : tensor<64x196x384xf32>
    %v1798 = stablehlo.reshape %v1797 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1799 = stablehlo.reshape %v1798 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1800 = stablehlo.broadcast_in_dim %s2b19nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1801 = stablehlo.add %v1799, %v1800 : tensor<64x196x384xf32>
    %v1802 = stablehlo.reshape %v1801 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1804 = stablehlo.transpose %v1803, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1805 = stablehlo.reshape %v1804 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1806 = stablehlo.reshape %v1805 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1807 = stablehlo.convolution(%v1806, %s2b19eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1808 = stablehlo.broadcast_in_dim %s2b19eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1809 = stablehlo.add %v1807, %v1808 : tensor<64x1536x14x14xf32>
    %v1810 = stablehlo.reshape %v1809 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1812 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1813 = stablehlo.multiply %v1812, %v1811 : tensor<64x1536x14x14xf32>
    %v1814 = stablehlo.negate %v1811 : tensor<64x1536x14x14xf32>
    %v1815 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1816 = stablehlo.multiply %v1814, %v1815 : tensor<64x1536x14x14xf32>
    %v1817 = chlo.erfc %v1816 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1818 = stablehlo.multiply %v1813, %v1817 : tensor<64x1536x14x14xf32>
    %v1819 = stablehlo.reshape %v1818 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1820 = stablehlo.reshape %v1819 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1821 = stablehlo.convolution(%v1820, %s2b19pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1822 = stablehlo.broadcast_in_dim %s2b19pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1823 = stablehlo.add %v1821, %v1822 : tensor<64x384x14x14xf32>
    %v1824 = stablehlo.reshape %v1823 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1825 = stablehlo.reshape %v1824 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1826 = stablehlo.broadcast_in_dim %s2b19lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1827 = stablehlo.multiply %v1825, %v1826 : tensor<64x384x14x14xf32>
    %v1828 = stablehlo.reshape %v1827 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1829 = stablehlo.reshape %v1828 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1830 = stablehlo.reshape %v1766 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1831 = stablehlo.add %v1829, %v1830 : tensor<64x384x14x14xf32>
    %v1832 = stablehlo.reshape %v1831 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1834 = stablehlo.convolution(%v1833, %s2b20dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1835 = stablehlo.broadcast_in_dim %s2b20db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1836 = stablehlo.add %v1834, %v1835 : tensor<64x384x14x14xf32>
    %v1837 = stablehlo.reshape %v1836 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1838 = stablehlo.reshape %v1837 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1839 = stablehlo.transpose %v1838, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1840 = stablehlo.reshape %v1839 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1841 = stablehlo.reshape %v1840 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1843 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1844 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1845 = stablehlo.reduce(%v1841 init: %v1842) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1846 = stablehlo.broadcast_in_dim %v1845, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1847 = stablehlo.divide %v1846, %v1843 : tensor<64x196x384xf32>
    %v1848 = stablehlo.subtract %v1841, %v1847 : tensor<64x196x384xf32>
    %v1849 = stablehlo.multiply %v1848, %v1848 : tensor<64x196x384xf32>
    %v1850 = stablehlo.reduce(%v1849 init: %v1842) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1851 = stablehlo.broadcast_in_dim %v1850, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1852 = stablehlo.divide %v1851, %v1843 : tensor<64x196x384xf32>
    %v1853 = stablehlo.add %v1852, %v1844 : tensor<64x196x384xf32>
    %v1854 = stablehlo.rsqrt %v1853 : tensor<64x196x384xf32>
    %v1855 = stablehlo.multiply %v1848, %v1854 : tensor<64x196x384xf32>
    %v1856 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1857 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1858 = stablehlo.multiply %v1855, %v1856 : tensor<64x196x384xf32>
    %v1859 = stablehlo.add %v1858, %v1857 : tensor<64x196x384xf32>
    %v1860 = stablehlo.reshape %v1859 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1861 = stablehlo.reshape %v1860 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1862 = stablehlo.broadcast_in_dim %s2b20ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1863 = stablehlo.multiply %v1861, %v1862 : tensor<64x196x384xf32>
    %v1864 = stablehlo.reshape %v1863 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1865 = stablehlo.reshape %v1864 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1866 = stablehlo.broadcast_in_dim %s2b20nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1867 = stablehlo.add %v1865, %v1866 : tensor<64x196x384xf32>
    %v1868 = stablehlo.reshape %v1867 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1869 = stablehlo.reshape %v1868 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1870 = stablehlo.transpose %v1869, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1871 = stablehlo.reshape %v1870 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1872 = stablehlo.reshape %v1871 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1873 = stablehlo.convolution(%v1872, %s2b20eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1874 = stablehlo.broadcast_in_dim %s2b20eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1875 = stablehlo.add %v1873, %v1874 : tensor<64x1536x14x14xf32>
    %v1876 = stablehlo.reshape %v1875 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1877 = stablehlo.reshape %v1876 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1878 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1879 = stablehlo.multiply %v1878, %v1877 : tensor<64x1536x14x14xf32>
    %v1880 = stablehlo.negate %v1877 : tensor<64x1536x14x14xf32>
    %v1881 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1882 = stablehlo.multiply %v1880, %v1881 : tensor<64x1536x14x14xf32>
    %v1883 = chlo.erfc %v1882 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1884 = stablehlo.multiply %v1879, %v1883 : tensor<64x1536x14x14xf32>
    %v1885 = stablehlo.reshape %v1884 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1886 = stablehlo.reshape %v1885 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1887 = stablehlo.convolution(%v1886, %s2b20pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1888 = stablehlo.broadcast_in_dim %s2b20pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1889 = stablehlo.add %v1887, %v1888 : tensor<64x384x14x14xf32>
    %v1890 = stablehlo.reshape %v1889 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1891 = stablehlo.reshape %v1890 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1892 = stablehlo.broadcast_in_dim %s2b20lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1893 = stablehlo.multiply %v1891, %v1892 : tensor<64x384x14x14xf32>
    %v1894 = stablehlo.reshape %v1893 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1895 = stablehlo.reshape %v1894 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1896 = stablehlo.reshape %v1832 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1897 = stablehlo.add %v1895, %v1896 : tensor<64x384x14x14xf32>
    %v1898 = stablehlo.reshape %v1897 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1899 = stablehlo.reshape %v1898 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1900 = stablehlo.convolution(%v1899, %s2b21dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1901 = stablehlo.broadcast_in_dim %s2b21db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1902 = stablehlo.add %v1900, %v1901 : tensor<64x384x14x14xf32>
    %v1903 = stablehlo.reshape %v1902 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1905 = stablehlo.transpose %v1904, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1906 = stablehlo.reshape %v1905 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1908 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1909 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1910 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1911 = stablehlo.reduce(%v1907 init: %v1908) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1912 = stablehlo.broadcast_in_dim %v1911, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1913 = stablehlo.divide %v1912, %v1909 : tensor<64x196x384xf32>
    %v1914 = stablehlo.subtract %v1907, %v1913 : tensor<64x196x384xf32>
    %v1915 = stablehlo.multiply %v1914, %v1914 : tensor<64x196x384xf32>
    %v1916 = stablehlo.reduce(%v1915 init: %v1908) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1917 = stablehlo.broadcast_in_dim %v1916, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1918 = stablehlo.divide %v1917, %v1909 : tensor<64x196x384xf32>
    %v1919 = stablehlo.add %v1918, %v1910 : tensor<64x196x384xf32>
    %v1920 = stablehlo.rsqrt %v1919 : tensor<64x196x384xf32>
    %v1921 = stablehlo.multiply %v1914, %v1920 : tensor<64x196x384xf32>
    %v1922 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1923 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1924 = stablehlo.multiply %v1921, %v1922 : tensor<64x196x384xf32>
    %v1925 = stablehlo.add %v1924, %v1923 : tensor<64x196x384xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1927 = stablehlo.reshape %v1926 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1928 = stablehlo.broadcast_in_dim %s2b21ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1929 = stablehlo.multiply %v1927, %v1928 : tensor<64x196x384xf32>
    %v1930 = stablehlo.reshape %v1929 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1931 = stablehlo.reshape %v1930 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1932 = stablehlo.broadcast_in_dim %s2b21nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1933 = stablehlo.add %v1931, %v1932 : tensor<64x196x384xf32>
    %v1934 = stablehlo.reshape %v1933 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1935 = stablehlo.reshape %v1934 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1936 = stablehlo.transpose %v1935, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v1937 = stablehlo.reshape %v1936 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v1938 = stablehlo.reshape %v1937 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1939 = stablehlo.convolution(%v1938, %s2b21eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v1940 = stablehlo.broadcast_in_dim %s2b21eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v1941 = stablehlo.add %v1939, %v1940 : tensor<64x1536x14x14xf32>
    %v1942 = stablehlo.reshape %v1941 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1944 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v1945 = stablehlo.multiply %v1944, %v1943 : tensor<64x1536x14x14xf32>
    %v1946 = stablehlo.negate %v1943 : tensor<64x1536x14x14xf32>
    %v1947 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v1948 = stablehlo.multiply %v1946, %v1947 : tensor<64x1536x14x14xf32>
    %v1949 = chlo.erfc %v1948 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v1950 = stablehlo.multiply %v1945, %v1949 : tensor<64x1536x14x14xf32>
    %v1951 = stablehlo.reshape %v1950 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v1952 = stablehlo.reshape %v1951 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v1953 = stablehlo.convolution(%v1952, %s2b21pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v1954 = stablehlo.broadcast_in_dim %s2b21pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1955 = stablehlo.add %v1953, %v1954 : tensor<64x384x14x14xf32>
    %v1956 = stablehlo.reshape %v1955 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1957 = stablehlo.reshape %v1956 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1958 = stablehlo.broadcast_in_dim %s2b21lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1959 = stablehlo.multiply %v1957, %v1958 : tensor<64x384x14x14xf32>
    %v1960 = stablehlo.reshape %v1959 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1961 = stablehlo.reshape %v1960 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1962 = stablehlo.reshape %v1898 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1963 = stablehlo.add %v1961, %v1962 : tensor<64x384x14x14xf32>
    %v1964 = stablehlo.reshape %v1963 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1965 = stablehlo.reshape %v1964 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v1966 = stablehlo.convolution(%v1965, %s2b22dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v1967 = stablehlo.broadcast_in_dim %s2b22db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v1968 = stablehlo.add %v1966, %v1967 : tensor<64x384x14x14xf32>
    %v1969 = stablehlo.reshape %v1968 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v1970 = stablehlo.reshape %v1969 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v1971 = stablehlo.transpose %v1970, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v1972 = stablehlo.reshape %v1971 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1973 = stablehlo.reshape %v1972 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1974 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1975 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v1976 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v1977 = stablehlo.reduce(%v1973 init: %v1974) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1978 = stablehlo.broadcast_in_dim %v1977, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1979 = stablehlo.divide %v1978, %v1975 : tensor<64x196x384xf32>
    %v1980 = stablehlo.subtract %v1973, %v1979 : tensor<64x196x384xf32>
    %v1981 = stablehlo.multiply %v1980, %v1980 : tensor<64x196x384xf32>
    %v1982 = stablehlo.reduce(%v1981 init: %v1974) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v1983 = stablehlo.broadcast_in_dim %v1982, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v1984 = stablehlo.divide %v1983, %v1975 : tensor<64x196x384xf32>
    %v1985 = stablehlo.add %v1984, %v1976 : tensor<64x196x384xf32>
    %v1986 = stablehlo.rsqrt %v1985 : tensor<64x196x384xf32>
    %v1987 = stablehlo.multiply %v1980, %v1986 : tensor<64x196x384xf32>
    %v1988 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1989 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v1990 = stablehlo.multiply %v1987, %v1988 : tensor<64x196x384xf32>
    %v1991 = stablehlo.add %v1990, %v1989 : tensor<64x196x384xf32>
    %v1992 = stablehlo.reshape %v1991 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1993 = stablehlo.reshape %v1992 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1994 = stablehlo.broadcast_in_dim %s2b22ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1995 = stablehlo.multiply %v1993, %v1994 : tensor<64x196x384xf32>
    %v1996 = stablehlo.reshape %v1995 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v1997 = stablehlo.reshape %v1996 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v1998 = stablehlo.broadcast_in_dim %s2b22nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v1999 = stablehlo.add %v1997, %v1998 : tensor<64x196x384xf32>
    %v2000 = stablehlo.reshape %v1999 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2001 = stablehlo.reshape %v2000 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2002 = stablehlo.transpose %v2001, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v2003 = stablehlo.reshape %v2002 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v2004 = stablehlo.reshape %v2003 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2005 = stablehlo.convolution(%v2004, %s2b22eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v2006 = stablehlo.broadcast_in_dim %s2b22eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v2007 = stablehlo.add %v2005, %v2006 : tensor<64x1536x14x14xf32>
    %v2008 = stablehlo.reshape %v2007 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v2009 = stablehlo.reshape %v2008 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v2010 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v2011 = stablehlo.multiply %v2010, %v2009 : tensor<64x1536x14x14xf32>
    %v2012 = stablehlo.negate %v2009 : tensor<64x1536x14x14xf32>
    %v2013 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v2014 = stablehlo.multiply %v2012, %v2013 : tensor<64x1536x14x14xf32>
    %v2015 = chlo.erfc %v2014 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v2016 = stablehlo.multiply %v2011, %v2015 : tensor<64x1536x14x14xf32>
    %v2017 = stablehlo.reshape %v2016 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v2018 = stablehlo.reshape %v2017 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v2019 = stablehlo.convolution(%v2018, %s2b22pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v2020 = stablehlo.broadcast_in_dim %s2b22pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2021 = stablehlo.add %v2019, %v2020 : tensor<64x384x14x14xf32>
    %v2022 = stablehlo.reshape %v2021 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2023 = stablehlo.reshape %v2022 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2024 = stablehlo.broadcast_in_dim %s2b22lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2025 = stablehlo.multiply %v2023, %v2024 : tensor<64x384x14x14xf32>
    %v2026 = stablehlo.reshape %v2025 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2027 = stablehlo.reshape %v2026 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2028 = stablehlo.reshape %v1964 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2029 = stablehlo.add %v2027, %v2028 : tensor<64x384x14x14xf32>
    %v2030 = stablehlo.reshape %v2029 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2031 = stablehlo.reshape %v2030 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2032 = stablehlo.convolution(%v2031, %s2b23dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v2033 = stablehlo.broadcast_in_dim %s2b23db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2034 = stablehlo.add %v2032, %v2033 : tensor<64x384x14x14xf32>
    %v2035 = stablehlo.reshape %v2034 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2036 = stablehlo.reshape %v2035 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v2037 = stablehlo.transpose %v2036, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2041 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v2042 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v2043 = stablehlo.reduce(%v2039 init: %v2040) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v2044 = stablehlo.broadcast_in_dim %v2043, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v2045 = stablehlo.divide %v2044, %v2041 : tensor<64x196x384xf32>
    %v2046 = stablehlo.subtract %v2039, %v2045 : tensor<64x196x384xf32>
    %v2047 = stablehlo.multiply %v2046, %v2046 : tensor<64x196x384xf32>
    %v2048 = stablehlo.reduce(%v2047 init: %v2040) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v2049 = stablehlo.broadcast_in_dim %v2048, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v2050 = stablehlo.divide %v2049, %v2041 : tensor<64x196x384xf32>
    %v2051 = stablehlo.add %v2050, %v2042 : tensor<64x196x384xf32>
    %v2052 = stablehlo.rsqrt %v2051 : tensor<64x196x384xf32>
    %v2053 = stablehlo.multiply %v2046, %v2052 : tensor<64x196x384xf32>
    %v2054 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v2055 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v2056 = stablehlo.multiply %v2053, %v2054 : tensor<64x196x384xf32>
    %v2057 = stablehlo.add %v2056, %v2055 : tensor<64x196x384xf32>
    %v2058 = stablehlo.reshape %v2057 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2059 = stablehlo.reshape %v2058 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2060 = stablehlo.broadcast_in_dim %s2b23ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v2061 = stablehlo.multiply %v2059, %v2060 : tensor<64x196x384xf32>
    %v2062 = stablehlo.reshape %v2061 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2063 = stablehlo.reshape %v2062 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2064 = stablehlo.broadcast_in_dim %s2b23nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v2065 = stablehlo.add %v2063, %v2064 : tensor<64x196x384xf32>
    %v2066 = stablehlo.reshape %v2065 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2067 = stablehlo.reshape %v2066 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2068 = stablehlo.transpose %v2067, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v2069 = stablehlo.reshape %v2068 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v2070 = stablehlo.reshape %v2069 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2071 = stablehlo.convolution(%v2070, %s2b23eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v2072 = stablehlo.broadcast_in_dim %s2b23eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v2073 = stablehlo.add %v2071, %v2072 : tensor<64x1536x14x14xf32>
    %v2074 = stablehlo.reshape %v2073 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v2075 = stablehlo.reshape %v2074 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v2076 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v2077 = stablehlo.multiply %v2076, %v2075 : tensor<64x1536x14x14xf32>
    %v2078 = stablehlo.negate %v2075 : tensor<64x1536x14x14xf32>
    %v2079 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v2080 = stablehlo.multiply %v2078, %v2079 : tensor<64x1536x14x14xf32>
    %v2081 = chlo.erfc %v2080 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v2082 = stablehlo.multiply %v2077, %v2081 : tensor<64x1536x14x14xf32>
    %v2083 = stablehlo.reshape %v2082 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v2084 = stablehlo.reshape %v2083 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v2085 = stablehlo.convolution(%v2084, %s2b23pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v2086 = stablehlo.broadcast_in_dim %s2b23pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2087 = stablehlo.add %v2085, %v2086 : tensor<64x384x14x14xf32>
    %v2088 = stablehlo.reshape %v2087 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2089 = stablehlo.reshape %v2088 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2090 = stablehlo.broadcast_in_dim %s2b23lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2091 = stablehlo.multiply %v2089, %v2090 : tensor<64x384x14x14xf32>
    %v2092 = stablehlo.reshape %v2091 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2093 = stablehlo.reshape %v2092 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2094 = stablehlo.reshape %v2030 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2095 = stablehlo.add %v2093, %v2094 : tensor<64x384x14x14xf32>
    %v2096 = stablehlo.reshape %v2095 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2097 = stablehlo.reshape %v2096 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2098 = stablehlo.convolution(%v2097, %s2b24dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v2099 = stablehlo.broadcast_in_dim %s2b24db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2100 = stablehlo.add %v2098, %v2099 : tensor<64x384x14x14xf32>
    %v2101 = stablehlo.reshape %v2100 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2102 = stablehlo.reshape %v2101 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v2103 = stablehlo.transpose %v2102, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v2104 = stablehlo.reshape %v2103 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2105 = stablehlo.reshape %v2104 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2107 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v2108 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v2109 = stablehlo.reduce(%v2105 init: %v2106) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v2110 = stablehlo.broadcast_in_dim %v2109, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v2111 = stablehlo.divide %v2110, %v2107 : tensor<64x196x384xf32>
    %v2112 = stablehlo.subtract %v2105, %v2111 : tensor<64x196x384xf32>
    %v2113 = stablehlo.multiply %v2112, %v2112 : tensor<64x196x384xf32>
    %v2114 = stablehlo.reduce(%v2113 init: %v2106) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v2115 = stablehlo.broadcast_in_dim %v2114, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v2116 = stablehlo.divide %v2115, %v2107 : tensor<64x196x384xf32>
    %v2117 = stablehlo.add %v2116, %v2108 : tensor<64x196x384xf32>
    %v2118 = stablehlo.rsqrt %v2117 : tensor<64x196x384xf32>
    %v2119 = stablehlo.multiply %v2112, %v2118 : tensor<64x196x384xf32>
    %v2120 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v2121 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v2122 = stablehlo.multiply %v2119, %v2120 : tensor<64x196x384xf32>
    %v2123 = stablehlo.add %v2122, %v2121 : tensor<64x196x384xf32>
    %v2124 = stablehlo.reshape %v2123 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2126 = stablehlo.broadcast_in_dim %s2b24ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v2127 = stablehlo.multiply %v2125, %v2126 : tensor<64x196x384xf32>
    %v2128 = stablehlo.reshape %v2127 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2129 = stablehlo.reshape %v2128 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2130 = stablehlo.broadcast_in_dim %s2b24nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v2131 = stablehlo.add %v2129, %v2130 : tensor<64x196x384xf32>
    %v2132 = stablehlo.reshape %v2131 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2133 = stablehlo.reshape %v2132 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2134 = stablehlo.transpose %v2133, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v2135 = stablehlo.reshape %v2134 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v2136 = stablehlo.reshape %v2135 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2137 = stablehlo.convolution(%v2136, %s2b24eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v2138 = stablehlo.broadcast_in_dim %s2b24eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v2139 = stablehlo.add %v2137, %v2138 : tensor<64x1536x14x14xf32>
    %v2140 = stablehlo.reshape %v2139 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v2141 = stablehlo.reshape %v2140 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v2142 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v2143 = stablehlo.multiply %v2142, %v2141 : tensor<64x1536x14x14xf32>
    %v2144 = stablehlo.negate %v2141 : tensor<64x1536x14x14xf32>
    %v2145 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v2146 = stablehlo.multiply %v2144, %v2145 : tensor<64x1536x14x14xf32>
    %v2147 = chlo.erfc %v2146 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v2148 = stablehlo.multiply %v2143, %v2147 : tensor<64x1536x14x14xf32>
    %v2149 = stablehlo.reshape %v2148 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v2150 = stablehlo.reshape %v2149 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v2151 = stablehlo.convolution(%v2150, %s2b24pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v2152 = stablehlo.broadcast_in_dim %s2b24pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2153 = stablehlo.add %v2151, %v2152 : tensor<64x384x14x14xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2155 = stablehlo.reshape %v2154 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2156 = stablehlo.broadcast_in_dim %s2b24lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2157 = stablehlo.multiply %v2155, %v2156 : tensor<64x384x14x14xf32>
    %v2158 = stablehlo.reshape %v2157 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2159 = stablehlo.reshape %v2158 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2160 = stablehlo.reshape %v2096 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2161 = stablehlo.add %v2159, %v2160 : tensor<64x384x14x14xf32>
    %v2162 = stablehlo.reshape %v2161 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2163 = stablehlo.reshape %v2162 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2164 = stablehlo.convolution(%v2163, %s2b25dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v2165 = stablehlo.broadcast_in_dim %s2b25db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2166 = stablehlo.add %v2164, %v2165 : tensor<64x384x14x14xf32>
    %v2167 = stablehlo.reshape %v2166 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2168 = stablehlo.reshape %v2167 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v2169 = stablehlo.transpose %v2168, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v2170 = stablehlo.reshape %v2169 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2171 = stablehlo.reshape %v2170 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2173 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v2174 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v2175 = stablehlo.reduce(%v2171 init: %v2172) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v2176 = stablehlo.broadcast_in_dim %v2175, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v2177 = stablehlo.divide %v2176, %v2173 : tensor<64x196x384xf32>
    %v2178 = stablehlo.subtract %v2171, %v2177 : tensor<64x196x384xf32>
    %v2179 = stablehlo.multiply %v2178, %v2178 : tensor<64x196x384xf32>
    %v2180 = stablehlo.reduce(%v2179 init: %v2172) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v2181 = stablehlo.broadcast_in_dim %v2180, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v2182 = stablehlo.divide %v2181, %v2173 : tensor<64x196x384xf32>
    %v2183 = stablehlo.add %v2182, %v2174 : tensor<64x196x384xf32>
    %v2184 = stablehlo.rsqrt %v2183 : tensor<64x196x384xf32>
    %v2185 = stablehlo.multiply %v2178, %v2184 : tensor<64x196x384xf32>
    %v2186 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v2187 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v2188 = stablehlo.multiply %v2185, %v2186 : tensor<64x196x384xf32>
    %v2189 = stablehlo.add %v2188, %v2187 : tensor<64x196x384xf32>
    %v2190 = stablehlo.reshape %v2189 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2192 = stablehlo.broadcast_in_dim %s2b25ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v2193 = stablehlo.multiply %v2191, %v2192 : tensor<64x196x384xf32>
    %v2194 = stablehlo.reshape %v2193 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2195 = stablehlo.reshape %v2194 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2196 = stablehlo.broadcast_in_dim %s2b25nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v2197 = stablehlo.add %v2195, %v2196 : tensor<64x196x384xf32>
    %v2198 = stablehlo.reshape %v2197 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2199 = stablehlo.reshape %v2198 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2200 = stablehlo.transpose %v2199, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v2201 = stablehlo.reshape %v2200 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v2202 = stablehlo.reshape %v2201 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2203 = stablehlo.convolution(%v2202, %s2b25eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v2204 = stablehlo.broadcast_in_dim %s2b25eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v2205 = stablehlo.add %v2203, %v2204 : tensor<64x1536x14x14xf32>
    %v2206 = stablehlo.reshape %v2205 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v2207 = stablehlo.reshape %v2206 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v2208 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v2209 = stablehlo.multiply %v2208, %v2207 : tensor<64x1536x14x14xf32>
    %v2210 = stablehlo.negate %v2207 : tensor<64x1536x14x14xf32>
    %v2211 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v2212 = stablehlo.multiply %v2210, %v2211 : tensor<64x1536x14x14xf32>
    %v2213 = chlo.erfc %v2212 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v2214 = stablehlo.multiply %v2209, %v2213 : tensor<64x1536x14x14xf32>
    %v2215 = stablehlo.reshape %v2214 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v2216 = stablehlo.reshape %v2215 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v2217 = stablehlo.convolution(%v2216, %s2b25pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v2218 = stablehlo.broadcast_in_dim %s2b25pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2219 = stablehlo.add %v2217, %v2218 : tensor<64x384x14x14xf32>
    %v2220 = stablehlo.reshape %v2219 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2221 = stablehlo.reshape %v2220 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2222 = stablehlo.broadcast_in_dim %s2b25lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2223 = stablehlo.multiply %v2221, %v2222 : tensor<64x384x14x14xf32>
    %v2224 = stablehlo.reshape %v2223 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2225 = stablehlo.reshape %v2224 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2226 = stablehlo.reshape %v2162 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2227 = stablehlo.add %v2225, %v2226 : tensor<64x384x14x14xf32>
    %v2228 = stablehlo.reshape %v2227 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2229 = stablehlo.reshape %v2228 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2230 = stablehlo.convolution(%v2229, %s2b26dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<64x384x14x14xf32>
    %v2231 = stablehlo.broadcast_in_dim %s2b26db, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2232 = stablehlo.add %v2230, %v2231 : tensor<64x384x14x14xf32>
    %v2233 = stablehlo.reshape %v2232 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2234 = stablehlo.reshape %v2233 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v2235 = stablehlo.transpose %v2234, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v2236 = stablehlo.reshape %v2235 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2237 = stablehlo.reshape %v2236 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2238 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2239 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v2240 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v2241 = stablehlo.reduce(%v2237 init: %v2238) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v2242 = stablehlo.broadcast_in_dim %v2241, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v2243 = stablehlo.divide %v2242, %v2239 : tensor<64x196x384xf32>
    %v2244 = stablehlo.subtract %v2237, %v2243 : tensor<64x196x384xf32>
    %v2245 = stablehlo.multiply %v2244, %v2244 : tensor<64x196x384xf32>
    %v2246 = stablehlo.reduce(%v2245 init: %v2238) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v2247 = stablehlo.broadcast_in_dim %v2246, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v2248 = stablehlo.divide %v2247, %v2239 : tensor<64x196x384xf32>
    %v2249 = stablehlo.add %v2248, %v2240 : tensor<64x196x384xf32>
    %v2250 = stablehlo.rsqrt %v2249 : tensor<64x196x384xf32>
    %v2251 = stablehlo.multiply %v2244, %v2250 : tensor<64x196x384xf32>
    %v2252 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v2253 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v2254 = stablehlo.multiply %v2251, %v2252 : tensor<64x196x384xf32>
    %v2255 = stablehlo.add %v2254, %v2253 : tensor<64x196x384xf32>
    %v2256 = stablehlo.reshape %v2255 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2257 = stablehlo.reshape %v2256 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2258 = stablehlo.broadcast_in_dim %s2b26ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v2259 = stablehlo.multiply %v2257, %v2258 : tensor<64x196x384xf32>
    %v2260 = stablehlo.reshape %v2259 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2261 = stablehlo.reshape %v2260 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2262 = stablehlo.broadcast_in_dim %s2b26nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v2263 = stablehlo.add %v2261, %v2262 : tensor<64x196x384xf32>
    %v2264 = stablehlo.reshape %v2263 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2266 = stablehlo.transpose %v2265, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v2267 = stablehlo.reshape %v2266 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v2268 = stablehlo.reshape %v2267 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2269 = stablehlo.convolution(%v2268, %s2b26eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<64x1536x14x14xf32>
    %v2270 = stablehlo.broadcast_in_dim %s2b26eb, dims = [1] : (tensor<1536xf32>) -> tensor<64x1536x14x14xf32>
    %v2271 = stablehlo.add %v2269, %v2270 : tensor<64x1536x14x14xf32>
    %v2272 = stablehlo.reshape %v2271 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v2273 = stablehlo.reshape %v2272 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v2274 = stablehlo.constant dense<0.5> : tensor<64x1536x14x14xf32>
    %v2275 = stablehlo.multiply %v2274, %v2273 : tensor<64x1536x14x14xf32>
    %v2276 = stablehlo.negate %v2273 : tensor<64x1536x14x14xf32>
    %v2277 = stablehlo.constant dense<0.7071067811865476> : tensor<64x1536x14x14xf32>
    %v2278 = stablehlo.multiply %v2276, %v2277 : tensor<64x1536x14x14xf32>
    %v2279 = chlo.erfc %v2278 : tensor<64x1536x14x14xf32> -> tensor<64x1536x14x14xf32>
    %v2280 = stablehlo.multiply %v2275, %v2279 : tensor<64x1536x14x14xf32>
    %v2281 = stablehlo.reshape %v2280 : (tensor<64x1536x14x14xf32>) -> tensor<64x301056xf32>
    %v2282 = stablehlo.reshape %v2281 : (tensor<64x301056xf32>) -> tensor<64x1536x14x14xf32>
    %v2283 = stablehlo.convolution(%v2282, %s2b26pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v2284 = stablehlo.broadcast_in_dim %s2b26pb, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2285 = stablehlo.add %v2283, %v2284 : tensor<64x384x14x14xf32>
    %v2286 = stablehlo.reshape %v2285 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2287 = stablehlo.reshape %v2286 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2288 = stablehlo.broadcast_in_dim %s2b26lg, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v2289 = stablehlo.multiply %v2287, %v2288 : tensor<64x384x14x14xf32>
    %v2290 = stablehlo.reshape %v2289 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2291 = stablehlo.reshape %v2290 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2292 = stablehlo.reshape %v2228 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2293 = stablehlo.add %v2291, %v2292 : tensor<64x384x14x14xf32>
    %v2294 = stablehlo.reshape %v2293 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v2295 = stablehlo.reshape %v2294 : (tensor<64x75264xf32>) -> tensor<64x384x196xf32>
    %v2296 = stablehlo.transpose %v2295, dims = [0, 2, 1] : (tensor<64x384x196xf32>) -> tensor<64x196x384xf32>
    %v2297 = stablehlo.reshape %v2296 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2298 = stablehlo.reshape %v2297 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2299 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2300 = stablehlo.constant dense<384.0> : tensor<64x196x384xf32>
    %v2301 = stablehlo.constant dense<1.0e-6> : tensor<64x196x384xf32>
    %v2302 = stablehlo.reduce(%v2298 init: %v2299) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v2303 = stablehlo.broadcast_in_dim %v2302, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v2304 = stablehlo.divide %v2303, %v2300 : tensor<64x196x384xf32>
    %v2305 = stablehlo.subtract %v2298, %v2304 : tensor<64x196x384xf32>
    %v2306 = stablehlo.multiply %v2305, %v2305 : tensor<64x196x384xf32>
    %v2307 = stablehlo.reduce(%v2306 init: %v2299) applies stablehlo.add across dimensions = [2] : (tensor<64x196x384xf32>, tensor<f32>) -> tensor<64x196xf32>
    %v2308 = stablehlo.broadcast_in_dim %v2307, dims = [0, 1] : (tensor<64x196xf32>) -> tensor<64x196x384xf32>
    %v2309 = stablehlo.divide %v2308, %v2300 : tensor<64x196x384xf32>
    %v2310 = stablehlo.add %v2309, %v2301 : tensor<64x196x384xf32>
    %v2311 = stablehlo.rsqrt %v2310 : tensor<64x196x384xf32>
    %v2312 = stablehlo.multiply %v2305, %v2311 : tensor<64x196x384xf32>
    %v2313 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v2314 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x196x384xf32>
    %v2315 = stablehlo.multiply %v2312, %v2313 : tensor<64x196x384xf32>
    %v2316 = stablehlo.add %v2315, %v2314 : tensor<64x196x384xf32>
    %v2317 = stablehlo.reshape %v2316 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2318 = stablehlo.reshape %v2317 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2319 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v2320 = stablehlo.multiply %v2318, %v2319 : tensor<64x196x384xf32>
    %v2321 = stablehlo.reshape %v2320 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2322 = stablehlo.reshape %v2321 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2323 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<384xf32>) -> tensor<64x196x384xf32>
    %v2324 = stablehlo.add %v2322, %v2323 : tensor<64x196x384xf32>
    %v2325 = stablehlo.reshape %v2324 : (tensor<64x196x384xf32>) -> tensor<64x75264xf32>
    %v2326 = stablehlo.reshape %v2325 : (tensor<64x75264xf32>) -> tensor<64x196x384xf32>
    %v2327 = stablehlo.transpose %v2326, dims = [0, 2, 1] : (tensor<64x196x384xf32>) -> tensor<64x384x196xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<64x384x196xf32>) -> tensor<64x75264xf32>
    %v2329 = stablehlo.reshape %v2328 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v2330 = stablehlo.convolution(%v2329, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<64x768x7x7xf32>
    %v2331 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v2332 = stablehlo.add %v2330, %v2331 : tensor<64x768x7x7xf32>
    %v2333 = stablehlo.reshape %v2332 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2334 = stablehlo.reshape %v2333 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2335 = stablehlo.convolution(%v2334, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<64x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<64x768x7x7xf32>
    %v2336 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v2337 = stablehlo.add %v2335, %v2336 : tensor<64x768x7x7xf32>
    %v2338 = stablehlo.reshape %v2337 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2339 = stablehlo.reshape %v2338 : (tensor<64x37632xf32>) -> tensor<64x768x49xf32>
    %v2340 = stablehlo.transpose %v2339, dims = [0, 2, 1] : (tensor<64x768x49xf32>) -> tensor<64x49x768xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v2342 = stablehlo.reshape %v2341 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v2343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2344 = stablehlo.constant dense<768.0> : tensor<64x49x768xf32>
    %v2345 = stablehlo.constant dense<1.0e-6> : tensor<64x49x768xf32>
    %v2346 = stablehlo.reduce(%v2342 init: %v2343) applies stablehlo.add across dimensions = [2] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x49xf32>
    %v2347 = stablehlo.broadcast_in_dim %v2346, dims = [0, 1] : (tensor<64x49xf32>) -> tensor<64x49x768xf32>
    %v2348 = stablehlo.divide %v2347, %v2344 : tensor<64x49x768xf32>
    %v2349 = stablehlo.subtract %v2342, %v2348 : tensor<64x49x768xf32>
    %v2350 = stablehlo.multiply %v2349, %v2349 : tensor<64x49x768xf32>
    %v2351 = stablehlo.reduce(%v2350 init: %v2343) applies stablehlo.add across dimensions = [2] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x49xf32>
    %v2352 = stablehlo.broadcast_in_dim %v2351, dims = [0, 1] : (tensor<64x49xf32>) -> tensor<64x49x768xf32>
    %v2353 = stablehlo.divide %v2352, %v2344 : tensor<64x49x768xf32>
    %v2354 = stablehlo.add %v2353, %v2345 : tensor<64x49x768xf32>
    %v2355 = stablehlo.rsqrt %v2354 : tensor<64x49x768xf32>
    %v2356 = stablehlo.multiply %v2349, %v2355 : tensor<64x49x768xf32>
    %v2357 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x49x768xf32>
    %v2358 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x49x768xf32>
    %v2359 = stablehlo.multiply %v2356, %v2357 : tensor<64x49x768xf32>
    %v2360 = stablehlo.add %v2359, %v2358 : tensor<64x49x768xf32>
    %v2361 = stablehlo.reshape %v2360 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v2363 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<64x49x768xf32>
    %v2364 = stablehlo.multiply %v2362, %v2363 : tensor<64x49x768xf32>
    %v2365 = stablehlo.reshape %v2364 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v2366 = stablehlo.reshape %v2365 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v2367 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<768xf32>) -> tensor<64x49x768xf32>
    %v2368 = stablehlo.add %v2366, %v2367 : tensor<64x49x768xf32>
    %v2369 = stablehlo.reshape %v2368 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v2370 = stablehlo.reshape %v2369 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v2371 = stablehlo.transpose %v2370, dims = [0, 2, 1] : (tensor<64x49x768xf32>) -> tensor<64x768x49xf32>
    %v2372 = stablehlo.reshape %v2371 : (tensor<64x768x49xf32>) -> tensor<64x37632xf32>
    %v2373 = stablehlo.reshape %v2372 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2374 = stablehlo.convolution(%v2373, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<64x3072x7x7xf32>
    %v2375 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<64x3072x7x7xf32>
    %v2376 = stablehlo.add %v2374, %v2375 : tensor<64x3072x7x7xf32>
    %v2377 = stablehlo.reshape %v2376 : (tensor<64x3072x7x7xf32>) -> tensor<64x150528xf32>
    %v2378 = stablehlo.reshape %v2377 : (tensor<64x150528xf32>) -> tensor<64x3072x7x7xf32>
    %v2379 = stablehlo.constant dense<0.5> : tensor<64x3072x7x7xf32>
    %v2380 = stablehlo.multiply %v2379, %v2378 : tensor<64x3072x7x7xf32>
    %v2381 = stablehlo.negate %v2378 : tensor<64x3072x7x7xf32>
    %v2382 = stablehlo.constant dense<0.7071067811865476> : tensor<64x3072x7x7xf32>
    %v2383 = stablehlo.multiply %v2381, %v2382 : tensor<64x3072x7x7xf32>
    %v2384 = chlo.erfc %v2383 : tensor<64x3072x7x7xf32> -> tensor<64x3072x7x7xf32>
    %v2385 = stablehlo.multiply %v2380, %v2384 : tensor<64x3072x7x7xf32>
    %v2386 = stablehlo.reshape %v2385 : (tensor<64x3072x7x7xf32>) -> tensor<64x150528xf32>
    %v2387 = stablehlo.reshape %v2386 : (tensor<64x150528xf32>) -> tensor<64x3072x7x7xf32>
    %v2388 = stablehlo.convolution(%v2387, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<64x768x7x7xf32>
    %v2389 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v2390 = stablehlo.add %v2388, %v2389 : tensor<64x768x7x7xf32>
    %v2391 = stablehlo.reshape %v2390 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2393 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v2394 = stablehlo.multiply %v2392, %v2393 : tensor<64x768x7x7xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2396 = stablehlo.reshape %v2395 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2397 = stablehlo.reshape %v2333 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2398 = stablehlo.add %v2396, %v2397 : tensor<64x768x7x7xf32>
    %v2399 = stablehlo.reshape %v2398 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2400 = stablehlo.reshape %v2399 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2401 = stablehlo.convolution(%v2400, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<64x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<64x768x7x7xf32>
    %v2402 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v2403 = stablehlo.add %v2401, %v2402 : tensor<64x768x7x7xf32>
    %v2404 = stablehlo.reshape %v2403 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2405 = stablehlo.reshape %v2404 : (tensor<64x37632xf32>) -> tensor<64x768x49xf32>
    %v2406 = stablehlo.transpose %v2405, dims = [0, 2, 1] : (tensor<64x768x49xf32>) -> tensor<64x49x768xf32>
    %v2407 = stablehlo.reshape %v2406 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v2408 = stablehlo.reshape %v2407 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v2409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2410 = stablehlo.constant dense<768.0> : tensor<64x49x768xf32>
    %v2411 = stablehlo.constant dense<1.0e-6> : tensor<64x49x768xf32>
    %v2412 = stablehlo.reduce(%v2408 init: %v2409) applies stablehlo.add across dimensions = [2] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x49xf32>
    %v2413 = stablehlo.broadcast_in_dim %v2412, dims = [0, 1] : (tensor<64x49xf32>) -> tensor<64x49x768xf32>
    %v2414 = stablehlo.divide %v2413, %v2410 : tensor<64x49x768xf32>
    %v2415 = stablehlo.subtract %v2408, %v2414 : tensor<64x49x768xf32>
    %v2416 = stablehlo.multiply %v2415, %v2415 : tensor<64x49x768xf32>
    %v2417 = stablehlo.reduce(%v2416 init: %v2409) applies stablehlo.add across dimensions = [2] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x49xf32>
    %v2418 = stablehlo.broadcast_in_dim %v2417, dims = [0, 1] : (tensor<64x49xf32>) -> tensor<64x49x768xf32>
    %v2419 = stablehlo.divide %v2418, %v2410 : tensor<64x49x768xf32>
    %v2420 = stablehlo.add %v2419, %v2411 : tensor<64x49x768xf32>
    %v2421 = stablehlo.rsqrt %v2420 : tensor<64x49x768xf32>
    %v2422 = stablehlo.multiply %v2415, %v2421 : tensor<64x49x768xf32>
    %v2423 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x49x768xf32>
    %v2424 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x49x768xf32>
    %v2425 = stablehlo.multiply %v2422, %v2423 : tensor<64x49x768xf32>
    %v2426 = stablehlo.add %v2425, %v2424 : tensor<64x49x768xf32>
    %v2427 = stablehlo.reshape %v2426 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v2428 = stablehlo.reshape %v2427 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v2429 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<64x49x768xf32>
    %v2430 = stablehlo.multiply %v2428, %v2429 : tensor<64x49x768xf32>
    %v2431 = stablehlo.reshape %v2430 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v2432 = stablehlo.reshape %v2431 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v2433 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<768xf32>) -> tensor<64x49x768xf32>
    %v2434 = stablehlo.add %v2432, %v2433 : tensor<64x49x768xf32>
    %v2435 = stablehlo.reshape %v2434 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v2436 = stablehlo.reshape %v2435 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v2437 = stablehlo.transpose %v2436, dims = [0, 2, 1] : (tensor<64x49x768xf32>) -> tensor<64x768x49xf32>
    %v2438 = stablehlo.reshape %v2437 : (tensor<64x768x49xf32>) -> tensor<64x37632xf32>
    %v2439 = stablehlo.reshape %v2438 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2440 = stablehlo.convolution(%v2439, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<64x3072x7x7xf32>
    %v2441 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<64x3072x7x7xf32>
    %v2442 = stablehlo.add %v2440, %v2441 : tensor<64x3072x7x7xf32>
    %v2443 = stablehlo.reshape %v2442 : (tensor<64x3072x7x7xf32>) -> tensor<64x150528xf32>
    %v2444 = stablehlo.reshape %v2443 : (tensor<64x150528xf32>) -> tensor<64x3072x7x7xf32>
    %v2445 = stablehlo.constant dense<0.5> : tensor<64x3072x7x7xf32>
    %v2446 = stablehlo.multiply %v2445, %v2444 : tensor<64x3072x7x7xf32>
    %v2447 = stablehlo.negate %v2444 : tensor<64x3072x7x7xf32>
    %v2448 = stablehlo.constant dense<0.7071067811865476> : tensor<64x3072x7x7xf32>
    %v2449 = stablehlo.multiply %v2447, %v2448 : tensor<64x3072x7x7xf32>
    %v2450 = chlo.erfc %v2449 : tensor<64x3072x7x7xf32> -> tensor<64x3072x7x7xf32>
    %v2451 = stablehlo.multiply %v2446, %v2450 : tensor<64x3072x7x7xf32>
    %v2452 = stablehlo.reshape %v2451 : (tensor<64x3072x7x7xf32>) -> tensor<64x150528xf32>
    %v2453 = stablehlo.reshape %v2452 : (tensor<64x150528xf32>) -> tensor<64x3072x7x7xf32>
    %v2454 = stablehlo.convolution(%v2453, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<64x768x7x7xf32>
    %v2455 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v2456 = stablehlo.add %v2454, %v2455 : tensor<64x768x7x7xf32>
    %v2457 = stablehlo.reshape %v2456 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2458 = stablehlo.reshape %v2457 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2459 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v2460 = stablehlo.multiply %v2458, %v2459 : tensor<64x768x7x7xf32>
    %v2461 = stablehlo.reshape %v2460 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2462 = stablehlo.reshape %v2461 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2463 = stablehlo.reshape %v2399 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2464 = stablehlo.add %v2462, %v2463 : tensor<64x768x7x7xf32>
    %v2465 = stablehlo.reshape %v2464 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2466 = stablehlo.reshape %v2465 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2467 = stablehlo.convolution(%v2466, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<64x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<64x768x7x7xf32>
    %v2468 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v2469 = stablehlo.add %v2467, %v2468 : tensor<64x768x7x7xf32>
    %v2470 = stablehlo.reshape %v2469 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2471 = stablehlo.reshape %v2470 : (tensor<64x37632xf32>) -> tensor<64x768x49xf32>
    %v2472 = stablehlo.transpose %v2471, dims = [0, 2, 1] : (tensor<64x768x49xf32>) -> tensor<64x49x768xf32>
    %v2473 = stablehlo.reshape %v2472 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v2474 = stablehlo.reshape %v2473 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v2475 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2476 = stablehlo.constant dense<768.0> : tensor<64x49x768xf32>
    %v2477 = stablehlo.constant dense<1.0e-6> : tensor<64x49x768xf32>
    %v2478 = stablehlo.reduce(%v2474 init: %v2475) applies stablehlo.add across dimensions = [2] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x49xf32>
    %v2479 = stablehlo.broadcast_in_dim %v2478, dims = [0, 1] : (tensor<64x49xf32>) -> tensor<64x49x768xf32>
    %v2480 = stablehlo.divide %v2479, %v2476 : tensor<64x49x768xf32>
    %v2481 = stablehlo.subtract %v2474, %v2480 : tensor<64x49x768xf32>
    %v2482 = stablehlo.multiply %v2481, %v2481 : tensor<64x49x768xf32>
    %v2483 = stablehlo.reduce(%v2482 init: %v2475) applies stablehlo.add across dimensions = [2] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x49xf32>
    %v2484 = stablehlo.broadcast_in_dim %v2483, dims = [0, 1] : (tensor<64x49xf32>) -> tensor<64x49x768xf32>
    %v2485 = stablehlo.divide %v2484, %v2476 : tensor<64x49x768xf32>
    %v2486 = stablehlo.add %v2485, %v2477 : tensor<64x49x768xf32>
    %v2487 = stablehlo.rsqrt %v2486 : tensor<64x49x768xf32>
    %v2488 = stablehlo.multiply %v2481, %v2487 : tensor<64x49x768xf32>
    %v2489 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x49x768xf32>
    %v2490 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x49x768xf32>
    %v2491 = stablehlo.multiply %v2488, %v2489 : tensor<64x49x768xf32>
    %v2492 = stablehlo.add %v2491, %v2490 : tensor<64x49x768xf32>
    %v2493 = stablehlo.reshape %v2492 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v2494 = stablehlo.reshape %v2493 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v2495 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<64x49x768xf32>
    %v2496 = stablehlo.multiply %v2494, %v2495 : tensor<64x49x768xf32>
    %v2497 = stablehlo.reshape %v2496 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v2498 = stablehlo.reshape %v2497 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v2499 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<768xf32>) -> tensor<64x49x768xf32>
    %v2500 = stablehlo.add %v2498, %v2499 : tensor<64x49x768xf32>
    %v2501 = stablehlo.reshape %v2500 : (tensor<64x49x768xf32>) -> tensor<64x37632xf32>
    %v2502 = stablehlo.reshape %v2501 : (tensor<64x37632xf32>) -> tensor<64x49x768xf32>
    %v2503 = stablehlo.transpose %v2502, dims = [0, 2, 1] : (tensor<64x49x768xf32>) -> tensor<64x768x49xf32>
    %v2504 = stablehlo.reshape %v2503 : (tensor<64x768x49xf32>) -> tensor<64x37632xf32>
    %v2505 = stablehlo.reshape %v2504 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2506 = stablehlo.convolution(%v2505, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<64x3072x7x7xf32>
    %v2507 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<64x3072x7x7xf32>
    %v2508 = stablehlo.add %v2506, %v2507 : tensor<64x3072x7x7xf32>
    %v2509 = stablehlo.reshape %v2508 : (tensor<64x3072x7x7xf32>) -> tensor<64x150528xf32>
    %v2510 = stablehlo.reshape %v2509 : (tensor<64x150528xf32>) -> tensor<64x3072x7x7xf32>
    %v2511 = stablehlo.constant dense<0.5> : tensor<64x3072x7x7xf32>
    %v2512 = stablehlo.multiply %v2511, %v2510 : tensor<64x3072x7x7xf32>
    %v2513 = stablehlo.negate %v2510 : tensor<64x3072x7x7xf32>
    %v2514 = stablehlo.constant dense<0.7071067811865476> : tensor<64x3072x7x7xf32>
    %v2515 = stablehlo.multiply %v2513, %v2514 : tensor<64x3072x7x7xf32>
    %v2516 = chlo.erfc %v2515 : tensor<64x3072x7x7xf32> -> tensor<64x3072x7x7xf32>
    %v2517 = stablehlo.multiply %v2512, %v2516 : tensor<64x3072x7x7xf32>
    %v2518 = stablehlo.reshape %v2517 : (tensor<64x3072x7x7xf32>) -> tensor<64x150528xf32>
    %v2519 = stablehlo.reshape %v2518 : (tensor<64x150528xf32>) -> tensor<64x3072x7x7xf32>
    %v2520 = stablehlo.convolution(%v2519, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<64x768x7x7xf32>
    %v2521 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v2522 = stablehlo.add %v2520, %v2521 : tensor<64x768x7x7xf32>
    %v2523 = stablehlo.reshape %v2522 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2524 = stablehlo.reshape %v2523 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2525 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<64x768x7x7xf32>
    %v2526 = stablehlo.multiply %v2524, %v2525 : tensor<64x768x7x7xf32>
    %v2527 = stablehlo.reshape %v2526 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2528 = stablehlo.reshape %v2527 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2529 = stablehlo.reshape %v2465 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2530 = stablehlo.add %v2528, %v2529 : tensor<64x768x7x7xf32>
    %v2531 = stablehlo.reshape %v2530 : (tensor<64x768x7x7xf32>) -> tensor<64x37632xf32>
    %v2532 = stablehlo.reshape %v2531 : (tensor<64x37632xf32>) -> tensor<64x768x7x7xf32>
    %v2533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2534 = stablehlo.reduce(%v2532 init: %v2533) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x768x7x7xf32>, tensor<f32>) -> tensor<64x768xf32>
    %v2535 = stablehlo.constant dense<49.0> : tensor<64x768xf32>
    %v2536 = stablehlo.divide %v2534, %v2535 : tensor<64x768xf32>
    %v2537 = stablehlo.reshape %v2536 : (tensor<64x768xf32>) -> tensor<64x1x768xf32>
    %v2538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2539 = stablehlo.constant dense<768.0> : tensor<64x1x768xf32>
    %v2540 = stablehlo.constant dense<1.0e-6> : tensor<64x1x768xf32>
    %v2541 = stablehlo.reduce(%v2537 init: %v2538) applies stablehlo.add across dimensions = [2] : (tensor<64x1x768xf32>, tensor<f32>) -> tensor<64x1xf32>
    %v2542 = stablehlo.broadcast_in_dim %v2541, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x768xf32>
    %v2543 = stablehlo.divide %v2542, %v2539 : tensor<64x1x768xf32>
    %v2544 = stablehlo.subtract %v2537, %v2543 : tensor<64x1x768xf32>
    %v2545 = stablehlo.multiply %v2544, %v2544 : tensor<64x1x768xf32>
    %v2546 = stablehlo.reduce(%v2545 init: %v2538) applies stablehlo.add across dimensions = [2] : (tensor<64x1x768xf32>, tensor<f32>) -> tensor<64x1xf32>
    %v2547 = stablehlo.broadcast_in_dim %v2546, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x768xf32>
    %v2548 = stablehlo.divide %v2547, %v2539 : tensor<64x1x768xf32>
    %v2549 = stablehlo.add %v2548, %v2540 : tensor<64x1x768xf32>
    %v2550 = stablehlo.rsqrt %v2549 : tensor<64x1x768xf32>
    %v2551 = stablehlo.multiply %v2544, %v2550 : tensor<64x1x768xf32>
    %v2552 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<64x1x768xf32>
    %v2553 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<64x1x768xf32>
    %v2554 = stablehlo.multiply %v2551, %v2552 : tensor<64x1x768xf32>
    %v2555 = stablehlo.add %v2554, %v2553 : tensor<64x1x768xf32>
    %v2556 = stablehlo.reshape %v2555 : (tensor<64x1x768xf32>) -> tensor<64x768xf32>
    %v2557 = stablehlo.reshape %v2556 : (tensor<64x768xf32>) -> tensor<64x1x768xf32>
    %v2558 = stablehlo.broadcast_in_dim %hng, dims = [2] : (tensor<768xf32>) -> tensor<64x1x768xf32>
    %v2559 = stablehlo.multiply %v2557, %v2558 : tensor<64x1x768xf32>
    %v2560 = stablehlo.reshape %v2559 : (tensor<64x1x768xf32>) -> tensor<64x768xf32>
    %v2561 = stablehlo.reshape %v2560 : (tensor<64x768xf32>) -> tensor<64x1x768xf32>
    %v2562 = stablehlo.broadcast_in_dim %hnbt, dims = [2] : (tensor<768xf32>) -> tensor<64x1x768xf32>
    %v2563 = stablehlo.add %v2561, %v2562 : tensor<64x1x768xf32>
    %v2564 = stablehlo.reshape %v2563 : (tensor<64x1x768xf32>) -> tensor<64x768xf32>
    %v2565 = stablehlo.dot_general %v2564, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x768xf32>, tensor<768x1000xf32>) -> tensor<64x1000xf32>
    %v2566 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v2567 = stablehlo.add %v2565, %v2566 : tensor<64x1000xf32>
    return %v2567 : tensor<64x1000xf32>
  }
}
