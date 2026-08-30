module @m {
  func.func @convnextbin_fwd(%x: tensor<32x150528xf32>, %psW: tensor<128x3x4x4xf32>, %psb: tensor<128xf32>, %psng: tensor<128xf32>, %psnbt: tensor<128xf32>, %s0b0dW: tensor<128x1x7x7xf32>, %s0b0db: tensor<128xf32>, %s0b0ng: tensor<128xf32>, %s0b0nbt: tensor<128xf32>, %s0b0eW: tensor<512x128x1x1xf32>, %s0b0eb: tensor<512xf32>, %s0b0pW: tensor<128x512x1x1xf32>, %s0b0pb: tensor<128xf32>, %s0b0lg: tensor<128xf32>, %s0b1dW: tensor<128x1x7x7xf32>, %s0b1db: tensor<128xf32>, %s0b1ng: tensor<128xf32>, %s0b1nbt: tensor<128xf32>, %s0b1eW: tensor<512x128x1x1xf32>, %s0b1eb: tensor<512xf32>, %s0b1pW: tensor<128x512x1x1xf32>, %s0b1pb: tensor<128xf32>, %s0b1lg: tensor<128xf32>, %s0b2dW: tensor<128x1x7x7xf32>, %s0b2db: tensor<128xf32>, %s0b2ng: tensor<128xf32>, %s0b2nbt: tensor<128xf32>, %s0b2eW: tensor<512x128x1x1xf32>, %s0b2eb: tensor<512xf32>, %s0b2pW: tensor<128x512x1x1xf32>, %s0b2pb: tensor<128xf32>, %s0b2lg: tensor<128xf32>, %d0ng: tensor<128xf32>, %d0nbt: tensor<128xf32>, %d0W: tensor<256x128x2x2xf32>, %d0b: tensor<256xf32>, %s1b0dW: tensor<256x1x7x7xf32>, %s1b0db: tensor<256xf32>, %s1b0ng: tensor<256xf32>, %s1b0nbt: tensor<256xf32>, %s1b0eW: tensor<1024x256x1x1xf32>, %s1b0eb: tensor<1024xf32>, %s1b0pW: tensor<256x1024x1x1xf32>, %s1b0pb: tensor<256xf32>, %s1b0lg: tensor<256xf32>, %s1b1dW: tensor<256x1x7x7xf32>, %s1b1db: tensor<256xf32>, %s1b1ng: tensor<256xf32>, %s1b1nbt: tensor<256xf32>, %s1b1eW: tensor<1024x256x1x1xf32>, %s1b1eb: tensor<1024xf32>, %s1b1pW: tensor<256x1024x1x1xf32>, %s1b1pb: tensor<256xf32>, %s1b1lg: tensor<256xf32>, %s1b2dW: tensor<256x1x7x7xf32>, %s1b2db: tensor<256xf32>, %s1b2ng: tensor<256xf32>, %s1b2nbt: tensor<256xf32>, %s1b2eW: tensor<1024x256x1x1xf32>, %s1b2eb: tensor<1024xf32>, %s1b2pW: tensor<256x1024x1x1xf32>, %s1b2pb: tensor<256xf32>, %s1b2lg: tensor<256xf32>, %d1ng: tensor<256xf32>, %d1nbt: tensor<256xf32>, %d1W: tensor<512x256x2x2xf32>, %d1b: tensor<512xf32>, %s2b0dW: tensor<512x1x7x7xf32>, %s2b0db: tensor<512xf32>, %s2b0ng: tensor<512xf32>, %s2b0nbt: tensor<512xf32>, %s2b0eW: tensor<2048x512x1x1xf32>, %s2b0eb: tensor<2048xf32>, %s2b0pW: tensor<512x2048x1x1xf32>, %s2b0pb: tensor<512xf32>, %s2b0lg: tensor<512xf32>, %s2b1dW: tensor<512x1x7x7xf32>, %s2b1db: tensor<512xf32>, %s2b1ng: tensor<512xf32>, %s2b1nbt: tensor<512xf32>, %s2b1eW: tensor<2048x512x1x1xf32>, %s2b1eb: tensor<2048xf32>, %s2b1pW: tensor<512x2048x1x1xf32>, %s2b1pb: tensor<512xf32>, %s2b1lg: tensor<512xf32>, %s2b2dW: tensor<512x1x7x7xf32>, %s2b2db: tensor<512xf32>, %s2b2ng: tensor<512xf32>, %s2b2nbt: tensor<512xf32>, %s2b2eW: tensor<2048x512x1x1xf32>, %s2b2eb: tensor<2048xf32>, %s2b2pW: tensor<512x2048x1x1xf32>, %s2b2pb: tensor<512xf32>, %s2b2lg: tensor<512xf32>, %s2b3dW: tensor<512x1x7x7xf32>, %s2b3db: tensor<512xf32>, %s2b3ng: tensor<512xf32>, %s2b3nbt: tensor<512xf32>, %s2b3eW: tensor<2048x512x1x1xf32>, %s2b3eb: tensor<2048xf32>, %s2b3pW: tensor<512x2048x1x1xf32>, %s2b3pb: tensor<512xf32>, %s2b3lg: tensor<512xf32>, %s2b4dW: tensor<512x1x7x7xf32>, %s2b4db: tensor<512xf32>, %s2b4ng: tensor<512xf32>, %s2b4nbt: tensor<512xf32>, %s2b4eW: tensor<2048x512x1x1xf32>, %s2b4eb: tensor<2048xf32>, %s2b4pW: tensor<512x2048x1x1xf32>, %s2b4pb: tensor<512xf32>, %s2b4lg: tensor<512xf32>, %s2b5dW: tensor<512x1x7x7xf32>, %s2b5db: tensor<512xf32>, %s2b5ng: tensor<512xf32>, %s2b5nbt: tensor<512xf32>, %s2b5eW: tensor<2048x512x1x1xf32>, %s2b5eb: tensor<2048xf32>, %s2b5pW: tensor<512x2048x1x1xf32>, %s2b5pb: tensor<512xf32>, %s2b5lg: tensor<512xf32>, %s2b6dW: tensor<512x1x7x7xf32>, %s2b6db: tensor<512xf32>, %s2b6ng: tensor<512xf32>, %s2b6nbt: tensor<512xf32>, %s2b6eW: tensor<2048x512x1x1xf32>, %s2b6eb: tensor<2048xf32>, %s2b6pW: tensor<512x2048x1x1xf32>, %s2b6pb: tensor<512xf32>, %s2b6lg: tensor<512xf32>, %s2b7dW: tensor<512x1x7x7xf32>, %s2b7db: tensor<512xf32>, %s2b7ng: tensor<512xf32>, %s2b7nbt: tensor<512xf32>, %s2b7eW: tensor<2048x512x1x1xf32>, %s2b7eb: tensor<2048xf32>, %s2b7pW: tensor<512x2048x1x1xf32>, %s2b7pb: tensor<512xf32>, %s2b7lg: tensor<512xf32>, %s2b8dW: tensor<512x1x7x7xf32>, %s2b8db: tensor<512xf32>, %s2b8ng: tensor<512xf32>, %s2b8nbt: tensor<512xf32>, %s2b8eW: tensor<2048x512x1x1xf32>, %s2b8eb: tensor<2048xf32>, %s2b8pW: tensor<512x2048x1x1xf32>, %s2b8pb: tensor<512xf32>, %s2b8lg: tensor<512xf32>, %s2b9dW: tensor<512x1x7x7xf32>, %s2b9db: tensor<512xf32>, %s2b9ng: tensor<512xf32>, %s2b9nbt: tensor<512xf32>, %s2b9eW: tensor<2048x512x1x1xf32>, %s2b9eb: tensor<2048xf32>, %s2b9pW: tensor<512x2048x1x1xf32>, %s2b9pb: tensor<512xf32>, %s2b9lg: tensor<512xf32>, %s2b10dW: tensor<512x1x7x7xf32>, %s2b10db: tensor<512xf32>, %s2b10ng: tensor<512xf32>, %s2b10nbt: tensor<512xf32>, %s2b10eW: tensor<2048x512x1x1xf32>, %s2b10eb: tensor<2048xf32>, %s2b10pW: tensor<512x2048x1x1xf32>, %s2b10pb: tensor<512xf32>, %s2b10lg: tensor<512xf32>, %s2b11dW: tensor<512x1x7x7xf32>, %s2b11db: tensor<512xf32>, %s2b11ng: tensor<512xf32>, %s2b11nbt: tensor<512xf32>, %s2b11eW: tensor<2048x512x1x1xf32>, %s2b11eb: tensor<2048xf32>, %s2b11pW: tensor<512x2048x1x1xf32>, %s2b11pb: tensor<512xf32>, %s2b11lg: tensor<512xf32>, %s2b12dW: tensor<512x1x7x7xf32>, %s2b12db: tensor<512xf32>, %s2b12ng: tensor<512xf32>, %s2b12nbt: tensor<512xf32>, %s2b12eW: tensor<2048x512x1x1xf32>, %s2b12eb: tensor<2048xf32>, %s2b12pW: tensor<512x2048x1x1xf32>, %s2b12pb: tensor<512xf32>, %s2b12lg: tensor<512xf32>, %s2b13dW: tensor<512x1x7x7xf32>, %s2b13db: tensor<512xf32>, %s2b13ng: tensor<512xf32>, %s2b13nbt: tensor<512xf32>, %s2b13eW: tensor<2048x512x1x1xf32>, %s2b13eb: tensor<2048xf32>, %s2b13pW: tensor<512x2048x1x1xf32>, %s2b13pb: tensor<512xf32>, %s2b13lg: tensor<512xf32>, %s2b14dW: tensor<512x1x7x7xf32>, %s2b14db: tensor<512xf32>, %s2b14ng: tensor<512xf32>, %s2b14nbt: tensor<512xf32>, %s2b14eW: tensor<2048x512x1x1xf32>, %s2b14eb: tensor<2048xf32>, %s2b14pW: tensor<512x2048x1x1xf32>, %s2b14pb: tensor<512xf32>, %s2b14lg: tensor<512xf32>, %s2b15dW: tensor<512x1x7x7xf32>, %s2b15db: tensor<512xf32>, %s2b15ng: tensor<512xf32>, %s2b15nbt: tensor<512xf32>, %s2b15eW: tensor<2048x512x1x1xf32>, %s2b15eb: tensor<2048xf32>, %s2b15pW: tensor<512x2048x1x1xf32>, %s2b15pb: tensor<512xf32>, %s2b15lg: tensor<512xf32>, %s2b16dW: tensor<512x1x7x7xf32>, %s2b16db: tensor<512xf32>, %s2b16ng: tensor<512xf32>, %s2b16nbt: tensor<512xf32>, %s2b16eW: tensor<2048x512x1x1xf32>, %s2b16eb: tensor<2048xf32>, %s2b16pW: tensor<512x2048x1x1xf32>, %s2b16pb: tensor<512xf32>, %s2b16lg: tensor<512xf32>, %s2b17dW: tensor<512x1x7x7xf32>, %s2b17db: tensor<512xf32>, %s2b17ng: tensor<512xf32>, %s2b17nbt: tensor<512xf32>, %s2b17eW: tensor<2048x512x1x1xf32>, %s2b17eb: tensor<2048xf32>, %s2b17pW: tensor<512x2048x1x1xf32>, %s2b17pb: tensor<512xf32>, %s2b17lg: tensor<512xf32>, %s2b18dW: tensor<512x1x7x7xf32>, %s2b18db: tensor<512xf32>, %s2b18ng: tensor<512xf32>, %s2b18nbt: tensor<512xf32>, %s2b18eW: tensor<2048x512x1x1xf32>, %s2b18eb: tensor<2048xf32>, %s2b18pW: tensor<512x2048x1x1xf32>, %s2b18pb: tensor<512xf32>, %s2b18lg: tensor<512xf32>, %s2b19dW: tensor<512x1x7x7xf32>, %s2b19db: tensor<512xf32>, %s2b19ng: tensor<512xf32>, %s2b19nbt: tensor<512xf32>, %s2b19eW: tensor<2048x512x1x1xf32>, %s2b19eb: tensor<2048xf32>, %s2b19pW: tensor<512x2048x1x1xf32>, %s2b19pb: tensor<512xf32>, %s2b19lg: tensor<512xf32>, %s2b20dW: tensor<512x1x7x7xf32>, %s2b20db: tensor<512xf32>, %s2b20ng: tensor<512xf32>, %s2b20nbt: tensor<512xf32>, %s2b20eW: tensor<2048x512x1x1xf32>, %s2b20eb: tensor<2048xf32>, %s2b20pW: tensor<512x2048x1x1xf32>, %s2b20pb: tensor<512xf32>, %s2b20lg: tensor<512xf32>, %s2b21dW: tensor<512x1x7x7xf32>, %s2b21db: tensor<512xf32>, %s2b21ng: tensor<512xf32>, %s2b21nbt: tensor<512xf32>, %s2b21eW: tensor<2048x512x1x1xf32>, %s2b21eb: tensor<2048xf32>, %s2b21pW: tensor<512x2048x1x1xf32>, %s2b21pb: tensor<512xf32>, %s2b21lg: tensor<512xf32>, %s2b22dW: tensor<512x1x7x7xf32>, %s2b22db: tensor<512xf32>, %s2b22ng: tensor<512xf32>, %s2b22nbt: tensor<512xf32>, %s2b22eW: tensor<2048x512x1x1xf32>, %s2b22eb: tensor<2048xf32>, %s2b22pW: tensor<512x2048x1x1xf32>, %s2b22pb: tensor<512xf32>, %s2b22lg: tensor<512xf32>, %s2b23dW: tensor<512x1x7x7xf32>, %s2b23db: tensor<512xf32>, %s2b23ng: tensor<512xf32>, %s2b23nbt: tensor<512xf32>, %s2b23eW: tensor<2048x512x1x1xf32>, %s2b23eb: tensor<2048xf32>, %s2b23pW: tensor<512x2048x1x1xf32>, %s2b23pb: tensor<512xf32>, %s2b23lg: tensor<512xf32>, %s2b24dW: tensor<512x1x7x7xf32>, %s2b24db: tensor<512xf32>, %s2b24ng: tensor<512xf32>, %s2b24nbt: tensor<512xf32>, %s2b24eW: tensor<2048x512x1x1xf32>, %s2b24eb: tensor<2048xf32>, %s2b24pW: tensor<512x2048x1x1xf32>, %s2b24pb: tensor<512xf32>, %s2b24lg: tensor<512xf32>, %s2b25dW: tensor<512x1x7x7xf32>, %s2b25db: tensor<512xf32>, %s2b25ng: tensor<512xf32>, %s2b25nbt: tensor<512xf32>, %s2b25eW: tensor<2048x512x1x1xf32>, %s2b25eb: tensor<2048xf32>, %s2b25pW: tensor<512x2048x1x1xf32>, %s2b25pb: tensor<512xf32>, %s2b25lg: tensor<512xf32>, %s2b26dW: tensor<512x1x7x7xf32>, %s2b26db: tensor<512xf32>, %s2b26ng: tensor<512xf32>, %s2b26nbt: tensor<512xf32>, %s2b26eW: tensor<2048x512x1x1xf32>, %s2b26eb: tensor<2048xf32>, %s2b26pW: tensor<512x2048x1x1xf32>, %s2b26pb: tensor<512xf32>, %s2b26lg: tensor<512xf32>, %d2ng: tensor<512xf32>, %d2nbt: tensor<512xf32>, %d2W: tensor<1024x512x2x2xf32>, %d2b: tensor<1024xf32>, %s3b0dW: tensor<1024x1x7x7xf32>, %s3b0db: tensor<1024xf32>, %s3b0ng: tensor<1024xf32>, %s3b0nbt: tensor<1024xf32>, %s3b0eW: tensor<4096x1024x1x1xf32>, %s3b0eb: tensor<4096xf32>, %s3b0pW: tensor<1024x4096x1x1xf32>, %s3b0pb: tensor<1024xf32>, %s3b0lg: tensor<1024xf32>, %s3b1dW: tensor<1024x1x7x7xf32>, %s3b1db: tensor<1024xf32>, %s3b1ng: tensor<1024xf32>, %s3b1nbt: tensor<1024xf32>, %s3b1eW: tensor<4096x1024x1x1xf32>, %s3b1eb: tensor<4096xf32>, %s3b1pW: tensor<1024x4096x1x1xf32>, %s3b1pb: tensor<1024xf32>, %s3b1lg: tensor<1024xf32>, %s3b2dW: tensor<1024x1x7x7xf32>, %s3b2db: tensor<1024xf32>, %s3b2ng: tensor<1024xf32>, %s3b2nbt: tensor<1024xf32>, %s3b2eW: tensor<4096x1024x1x1xf32>, %s3b2eb: tensor<4096xf32>, %s3b2pW: tensor<1024x4096x1x1xf32>, %s3b2pb: tensor<1024xf32>, %s3b2lg: tensor<1024xf32>, %hng: tensor<1024xf32>, %hnbt: tensor<1024xf32>, %Wd: tensor<1024x1000xf32>, %bd: tensor<1000xf32>) -> tensor<32x1000xf32> {
    // ── ConvNeXt-B forward: every line is pretty(verified AST node) ──
    // §2m: the channel-LN chain normalises with lnRowF at γ=1/β=0 and applies the REAL
    // per-channel affine with rowScaleF/rowBiasF, so these two are its scalar identities.
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %psW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<128x3x4x4xf32>) -> tensor<32x128x56x56xf32>
    %v2 = stablehlo.broadcast_in_dim %psb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x128x56x56xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v6 = stablehlo.transpose %v5, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v9 = stablehlo.constant dense<0.0> : tensor<f32>
    %v10 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v11 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v12 = stablehlo.reduce(%v8 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v13 = stablehlo.broadcast_in_dim %v12, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v14 = stablehlo.divide %v13, %v10 : tensor<32x3136x128xf32>
    %v15 = stablehlo.subtract %v8, %v14 : tensor<32x3136x128xf32>
    %v16 = stablehlo.multiply %v15, %v15 : tensor<32x3136x128xf32>
    %v17 = stablehlo.reduce(%v16 init: %v9) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v18 = stablehlo.broadcast_in_dim %v17, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v19 = stablehlo.divide %v18, %v10 : tensor<32x3136x128xf32>
    %v20 = stablehlo.add %v19, %v11 : tensor<32x3136x128xf32>
    %v21 = stablehlo.rsqrt %v20 : tensor<32x3136x128xf32>
    %v22 = stablehlo.multiply %v15, %v21 : tensor<32x3136x128xf32>
    %v23 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v24 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v25 = stablehlo.multiply %v22, %v23 : tensor<32x3136x128xf32>
    %v26 = stablehlo.add %v25, %v24 : tensor<32x3136x128xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v29 = stablehlo.broadcast_in_dim %psng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v30 = stablehlo.multiply %v28, %v29 : tensor<32x3136x128xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v33 = stablehlo.broadcast_in_dim %psnbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<32x3136x128xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v37 = stablehlo.transpose %v36, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v40 = stablehlo.convolution(%v39, %s0b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x56x56xf32>, tensor<128x1x7x7xf32>) -> tensor<32x128x56x56xf32>
    %v41 = stablehlo.broadcast_in_dim %s0b0db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<32x128x56x56xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v45 = stablehlo.transpose %v44, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v46 = stablehlo.reshape %v45 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v48 = stablehlo.constant dense<0.0> : tensor<f32>
    %v49 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v50 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v51 = stablehlo.reduce(%v47 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v52 = stablehlo.broadcast_in_dim %v51, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v53 = stablehlo.divide %v52, %v49 : tensor<32x3136x128xf32>
    %v54 = stablehlo.subtract %v47, %v53 : tensor<32x3136x128xf32>
    %v55 = stablehlo.multiply %v54, %v54 : tensor<32x3136x128xf32>
    %v56 = stablehlo.reduce(%v55 init: %v48) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v57 = stablehlo.broadcast_in_dim %v56, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v58 = stablehlo.divide %v57, %v49 : tensor<32x3136x128xf32>
    %v59 = stablehlo.add %v58, %v50 : tensor<32x3136x128xf32>
    %v60 = stablehlo.rsqrt %v59 : tensor<32x3136x128xf32>
    %v61 = stablehlo.multiply %v54, %v60 : tensor<32x3136x128xf32>
    %v62 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v63 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v64 = stablehlo.multiply %v61, %v62 : tensor<32x3136x128xf32>
    %v65 = stablehlo.add %v64, %v63 : tensor<32x3136x128xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v68 = stablehlo.broadcast_in_dim %s0b0ng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v69 = stablehlo.multiply %v67, %v68 : tensor<32x3136x128xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v72 = stablehlo.broadcast_in_dim %s0b0nbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v73 = stablehlo.add %v71, %v72 : tensor<32x3136x128xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v76 = stablehlo.transpose %v75, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v77 = stablehlo.reshape %v76 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v79 = stablehlo.convolution(%v78, %s0b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<512x128x1x1xf32>) -> tensor<32x512x56x56xf32>
    %v80 = stablehlo.broadcast_in_dim %s0b0eb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x56x56xf32>
    %v81 = stablehlo.add %v79, %v80 : tensor<32x512x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x512x56x56xf32>) -> tensor<32x1605632xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x1605632xf32>) -> tensor<32x512x56x56xf32>
    %v84 = stablehlo.multiply %v83, %v83 : tensor<32x512x56x56xf32>
    %v85 = stablehlo.multiply %v84, %v83 : tensor<32x512x56x56xf32>
    %v86 = stablehlo.constant dense<0.044715> : tensor<32x512x56x56xf32>
    %v87 = stablehlo.multiply %v86, %v85 : tensor<32x512x56x56xf32>
    %v88 = stablehlo.add %v83, %v87 : tensor<32x512x56x56xf32>
    %v89 = stablehlo.constant dense<0.7978845608028654> : tensor<32x512x56x56xf32>
    %v90 = stablehlo.multiply %v89, %v88 : tensor<32x512x56x56xf32>
    %v91 = stablehlo.tanh %v90 : tensor<32x512x56x56xf32>
    %v92 = stablehlo.constant dense<1.0> : tensor<32x512x56x56xf32>
    %v93 = stablehlo.add %v92, %v91 : tensor<32x512x56x56xf32>
    %v94 = stablehlo.constant dense<0.5> : tensor<32x512x56x56xf32>
    %v95 = stablehlo.multiply %v94, %v83 : tensor<32x512x56x56xf32>
    %v96 = stablehlo.multiply %v95, %v93 : tensor<32x512x56x56xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<32x512x56x56xf32>) -> tensor<32x1605632xf32>
    %v98 = stablehlo.reshape %v97 : (tensor<32x1605632xf32>) -> tensor<32x512x56x56xf32>
    %v99 = stablehlo.convolution(%v98, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x56x56xf32>, tensor<128x512x1x1xf32>) -> tensor<32x128x56x56xf32>
    %v100 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v101 = stablehlo.add %v99, %v100 : tensor<32x128x56x56xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v104 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v105 = stablehlo.multiply %v103, %v104 : tensor<32x128x56x56xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v108 = stablehlo.reshape %v38 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v109 = stablehlo.add %v107, %v108 : tensor<32x128x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v112 = stablehlo.convolution(%v111, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x56x56xf32>, tensor<128x1x7x7xf32>) -> tensor<32x128x56x56xf32>
    %v113 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v114 = stablehlo.add %v112, %v113 : tensor<32x128x56x56xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v117 = stablehlo.transpose %v116, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v121 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v122 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v123 = stablehlo.reduce(%v119 init: %v120) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v124 = stablehlo.broadcast_in_dim %v123, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v125 = stablehlo.divide %v124, %v121 : tensor<32x3136x128xf32>
    %v126 = stablehlo.subtract %v119, %v125 : tensor<32x3136x128xf32>
    %v127 = stablehlo.multiply %v126, %v126 : tensor<32x3136x128xf32>
    %v128 = stablehlo.reduce(%v127 init: %v120) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v129 = stablehlo.broadcast_in_dim %v128, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v130 = stablehlo.divide %v129, %v121 : tensor<32x3136x128xf32>
    %v131 = stablehlo.add %v130, %v122 : tensor<32x3136x128xf32>
    %v132 = stablehlo.rsqrt %v131 : tensor<32x3136x128xf32>
    %v133 = stablehlo.multiply %v126, %v132 : tensor<32x3136x128xf32>
    %v134 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v135 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v136 = stablehlo.multiply %v133, %v134 : tensor<32x3136x128xf32>
    %v137 = stablehlo.add %v136, %v135 : tensor<32x3136x128xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v140 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v141 = stablehlo.multiply %v139, %v140 : tensor<32x3136x128xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v144 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v145 = stablehlo.add %v143, %v144 : tensor<32x3136x128xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v148 = stablehlo.transpose %v147, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v151 = stablehlo.convolution(%v150, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<512x128x1x1xf32>) -> tensor<32x512x56x56xf32>
    %v152 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x56x56xf32>
    %v153 = stablehlo.add %v151, %v152 : tensor<32x512x56x56xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<32x512x56x56xf32>) -> tensor<32x1605632xf32>
    %v155 = stablehlo.reshape %v154 : (tensor<32x1605632xf32>) -> tensor<32x512x56x56xf32>
    %v156 = stablehlo.multiply %v155, %v155 : tensor<32x512x56x56xf32>
    %v157 = stablehlo.multiply %v156, %v155 : tensor<32x512x56x56xf32>
    %v158 = stablehlo.constant dense<0.044715> : tensor<32x512x56x56xf32>
    %v159 = stablehlo.multiply %v158, %v157 : tensor<32x512x56x56xf32>
    %v160 = stablehlo.add %v155, %v159 : tensor<32x512x56x56xf32>
    %v161 = stablehlo.constant dense<0.7978845608028654> : tensor<32x512x56x56xf32>
    %v162 = stablehlo.multiply %v161, %v160 : tensor<32x512x56x56xf32>
    %v163 = stablehlo.tanh %v162 : tensor<32x512x56x56xf32>
    %v164 = stablehlo.constant dense<1.0> : tensor<32x512x56x56xf32>
    %v165 = stablehlo.add %v164, %v163 : tensor<32x512x56x56xf32>
    %v166 = stablehlo.constant dense<0.5> : tensor<32x512x56x56xf32>
    %v167 = stablehlo.multiply %v166, %v155 : tensor<32x512x56x56xf32>
    %v168 = stablehlo.multiply %v167, %v165 : tensor<32x512x56x56xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<32x512x56x56xf32>) -> tensor<32x1605632xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x1605632xf32>) -> tensor<32x512x56x56xf32>
    %v171 = stablehlo.convolution(%v170, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x56x56xf32>, tensor<128x512x1x1xf32>) -> tensor<32x128x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v173 = stablehlo.add %v171, %v172 : tensor<32x128x56x56xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v176 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v177 = stablehlo.multiply %v175, %v176 : tensor<32x128x56x56xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v180 = stablehlo.reshape %v110 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v181 = stablehlo.add %v179, %v180 : tensor<32x128x56x56xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v184 = stablehlo.convolution(%v183, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x56x56xf32>, tensor<128x1x7x7xf32>) -> tensor<32x128x56x56xf32>
    %v185 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v186 = stablehlo.add %v184, %v185 : tensor<32x128x56x56xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v189 = stablehlo.transpose %v188, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v194 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v195 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v197 = stablehlo.divide %v196, %v193 : tensor<32x3136x128xf32>
    %v198 = stablehlo.subtract %v191, %v197 : tensor<32x3136x128xf32>
    %v199 = stablehlo.multiply %v198, %v198 : tensor<32x3136x128xf32>
    %v200 = stablehlo.reduce(%v199 init: %v192) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v202 = stablehlo.divide %v201, %v193 : tensor<32x3136x128xf32>
    %v203 = stablehlo.add %v202, %v194 : tensor<32x3136x128xf32>
    %v204 = stablehlo.rsqrt %v203 : tensor<32x3136x128xf32>
    %v205 = stablehlo.multiply %v198, %v204 : tensor<32x3136x128xf32>
    %v206 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v207 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v208 = stablehlo.multiply %v205, %v206 : tensor<32x3136x128xf32>
    %v209 = stablehlo.add %v208, %v207 : tensor<32x3136x128xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v212 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v213 = stablehlo.multiply %v211, %v212 : tensor<32x3136x128xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v216 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v217 = stablehlo.add %v215, %v216 : tensor<32x3136x128xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v220 = stablehlo.transpose %v219, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v223 = stablehlo.convolution(%v222, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<512x128x1x1xf32>) -> tensor<32x512x56x56xf32>
    %v224 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x56x56xf32>
    %v225 = stablehlo.add %v223, %v224 : tensor<32x512x56x56xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x512x56x56xf32>) -> tensor<32x1605632xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<32x1605632xf32>) -> tensor<32x512x56x56xf32>
    %v228 = stablehlo.multiply %v227, %v227 : tensor<32x512x56x56xf32>
    %v229 = stablehlo.multiply %v228, %v227 : tensor<32x512x56x56xf32>
    %v230 = stablehlo.constant dense<0.044715> : tensor<32x512x56x56xf32>
    %v231 = stablehlo.multiply %v230, %v229 : tensor<32x512x56x56xf32>
    %v232 = stablehlo.add %v227, %v231 : tensor<32x512x56x56xf32>
    %v233 = stablehlo.constant dense<0.7978845608028654> : tensor<32x512x56x56xf32>
    %v234 = stablehlo.multiply %v233, %v232 : tensor<32x512x56x56xf32>
    %v235 = stablehlo.tanh %v234 : tensor<32x512x56x56xf32>
    %v236 = stablehlo.constant dense<1.0> : tensor<32x512x56x56xf32>
    %v237 = stablehlo.add %v236, %v235 : tensor<32x512x56x56xf32>
    %v238 = stablehlo.constant dense<0.5> : tensor<32x512x56x56xf32>
    %v239 = stablehlo.multiply %v238, %v227 : tensor<32x512x56x56xf32>
    %v240 = stablehlo.multiply %v239, %v237 : tensor<32x512x56x56xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x512x56x56xf32>) -> tensor<32x1605632xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<32x1605632xf32>) -> tensor<32x512x56x56xf32>
    %v243 = stablehlo.convolution(%v242, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x56x56xf32>, tensor<128x512x1x1xf32>) -> tensor<32x128x56x56xf32>
    %v244 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v245 = stablehlo.add %v243, %v244 : tensor<32x128x56x56xf32>
    %v246 = stablehlo.reshape %v245 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v248 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v249 = stablehlo.multiply %v247, %v248 : tensor<32x128x56x56xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v252 = stablehlo.reshape %v182 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v253 = stablehlo.add %v251, %v252 : tensor<32x128x56x56xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v256 = stablehlo.transpose %v255, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v257 = stablehlo.reshape %v256 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v260 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v261 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v262 = stablehlo.reduce(%v258 init: %v259) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v263 = stablehlo.broadcast_in_dim %v262, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v264 = stablehlo.divide %v263, %v260 : tensor<32x3136x128xf32>
    %v265 = stablehlo.subtract %v258, %v264 : tensor<32x3136x128xf32>
    %v266 = stablehlo.multiply %v265, %v265 : tensor<32x3136x128xf32>
    %v267 = stablehlo.reduce(%v266 init: %v259) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v268 = stablehlo.broadcast_in_dim %v267, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v269 = stablehlo.divide %v268, %v260 : tensor<32x3136x128xf32>
    %v270 = stablehlo.add %v269, %v261 : tensor<32x3136x128xf32>
    %v271 = stablehlo.rsqrt %v270 : tensor<32x3136x128xf32>
    %v272 = stablehlo.multiply %v265, %v271 : tensor<32x3136x128xf32>
    %v273 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v274 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v275 = stablehlo.multiply %v272, %v273 : tensor<32x3136x128xf32>
    %v276 = stablehlo.add %v275, %v274 : tensor<32x3136x128xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v279 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v280 = stablehlo.multiply %v278, %v279 : tensor<32x3136x128xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v283 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v284 = stablehlo.add %v282, %v283 : tensor<32x3136x128xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v287 = stablehlo.transpose %v286, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v290 = stablehlo.convolution(%v289, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<256x128x2x2xf32>) -> tensor<32x256x28x28xf32>
    %v291 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v292 = stablehlo.add %v290, %v291 : tensor<32x256x28x28xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v294 = stablehlo.reshape %v293 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v295 = stablehlo.convolution(%v294, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x28x28xf32>, tensor<256x1x7x7xf32>) -> tensor<32x256x28x28xf32>
    %v296 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v297 = stablehlo.add %v295, %v296 : tensor<32x256x28x28xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v299 = stablehlo.reshape %v298 : (tensor<32x200704xf32>) -> tensor<32x256x784xf32>
    %v300 = stablehlo.transpose %v299, dims = [0, 2, 1] : (tensor<32x256x784xf32>) -> tensor<32x784x256xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v304 = stablehlo.constant dense<256.0> : tensor<32x784x256xf32>
    %v305 = stablehlo.constant dense<1.0e-6> : tensor<32x784x256xf32>
    %v306 = stablehlo.reduce(%v302 init: %v303) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v307 = stablehlo.broadcast_in_dim %v306, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v308 = stablehlo.divide %v307, %v304 : tensor<32x784x256xf32>
    %v309 = stablehlo.subtract %v302, %v308 : tensor<32x784x256xf32>
    %v310 = stablehlo.multiply %v309, %v309 : tensor<32x784x256xf32>
    %v311 = stablehlo.reduce(%v310 init: %v303) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v312 = stablehlo.broadcast_in_dim %v311, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v313 = stablehlo.divide %v312, %v304 : tensor<32x784x256xf32>
    %v314 = stablehlo.add %v313, %v305 : tensor<32x784x256xf32>
    %v315 = stablehlo.rsqrt %v314 : tensor<32x784x256xf32>
    %v316 = stablehlo.multiply %v309, %v315 : tensor<32x784x256xf32>
    %v317 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v318 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v319 = stablehlo.multiply %v316, %v317 : tensor<32x784x256xf32>
    %v320 = stablehlo.add %v319, %v318 : tensor<32x784x256xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v323 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v324 = stablehlo.multiply %v322, %v323 : tensor<32x784x256xf32>
    %v325 = stablehlo.reshape %v324 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v327 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v328 = stablehlo.add %v326, %v327 : tensor<32x784x256xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v331 = stablehlo.transpose %v330, dims = [0, 2, 1] : (tensor<32x784x256xf32>) -> tensor<32x256x784xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x256x784xf32>) -> tensor<32x200704xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v334 = stablehlo.convolution(%v333, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x28x28xf32>
    %v335 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x28x28xf32>
    %v336 = stablehlo.add %v334, %v335 : tensor<32x1024x28x28xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x1024x28x28xf32>) -> tensor<32x802816xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x802816xf32>) -> tensor<32x1024x28x28xf32>
    %v339 = stablehlo.multiply %v338, %v338 : tensor<32x1024x28x28xf32>
    %v340 = stablehlo.multiply %v339, %v338 : tensor<32x1024x28x28xf32>
    %v341 = stablehlo.constant dense<0.044715> : tensor<32x1024x28x28xf32>
    %v342 = stablehlo.multiply %v341, %v340 : tensor<32x1024x28x28xf32>
    %v343 = stablehlo.add %v338, %v342 : tensor<32x1024x28x28xf32>
    %v344 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1024x28x28xf32>
    %v345 = stablehlo.multiply %v344, %v343 : tensor<32x1024x28x28xf32>
    %v346 = stablehlo.tanh %v345 : tensor<32x1024x28x28xf32>
    %v347 = stablehlo.constant dense<1.0> : tensor<32x1024x28x28xf32>
    %v348 = stablehlo.add %v347, %v346 : tensor<32x1024x28x28xf32>
    %v349 = stablehlo.constant dense<0.5> : tensor<32x1024x28x28xf32>
    %v350 = stablehlo.multiply %v349, %v338 : tensor<32x1024x28x28xf32>
    %v351 = stablehlo.multiply %v350, %v348 : tensor<32x1024x28x28xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x1024x28x28xf32>) -> tensor<32x802816xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x802816xf32>) -> tensor<32x1024x28x28xf32>
    %v354 = stablehlo.convolution(%v353, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x28x28xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x28x28xf32>
    %v355 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v356 = stablehlo.add %v354, %v355 : tensor<32x256x28x28xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v359 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v360 = stablehlo.multiply %v358, %v359 : tensor<32x256x28x28xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v363 = stablehlo.reshape %v293 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v364 = stablehlo.add %v362, %v363 : tensor<32x256x28x28xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v366 = stablehlo.reshape %v365 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v367 = stablehlo.convolution(%v366, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x28x28xf32>, tensor<256x1x7x7xf32>) -> tensor<32x256x28x28xf32>
    %v368 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v369 = stablehlo.add %v367, %v368 : tensor<32x256x28x28xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v371 = stablehlo.reshape %v370 : (tensor<32x200704xf32>) -> tensor<32x256x784xf32>
    %v372 = stablehlo.transpose %v371, dims = [0, 2, 1] : (tensor<32x256x784xf32>) -> tensor<32x784x256xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v376 = stablehlo.constant dense<256.0> : tensor<32x784x256xf32>
    %v377 = stablehlo.constant dense<1.0e-6> : tensor<32x784x256xf32>
    %v378 = stablehlo.reduce(%v374 init: %v375) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v379 = stablehlo.broadcast_in_dim %v378, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v380 = stablehlo.divide %v379, %v376 : tensor<32x784x256xf32>
    %v381 = stablehlo.subtract %v374, %v380 : tensor<32x784x256xf32>
    %v382 = stablehlo.multiply %v381, %v381 : tensor<32x784x256xf32>
    %v383 = stablehlo.reduce(%v382 init: %v375) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v384 = stablehlo.broadcast_in_dim %v383, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v385 = stablehlo.divide %v384, %v376 : tensor<32x784x256xf32>
    %v386 = stablehlo.add %v385, %v377 : tensor<32x784x256xf32>
    %v387 = stablehlo.rsqrt %v386 : tensor<32x784x256xf32>
    %v388 = stablehlo.multiply %v381, %v387 : tensor<32x784x256xf32>
    %v389 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v390 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v391 = stablehlo.multiply %v388, %v389 : tensor<32x784x256xf32>
    %v392 = stablehlo.add %v391, %v390 : tensor<32x784x256xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v395 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v396 = stablehlo.multiply %v394, %v395 : tensor<32x784x256xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v399 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<32x784x256xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v403 = stablehlo.transpose %v402, dims = [0, 2, 1] : (tensor<32x784x256xf32>) -> tensor<32x256x784xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<32x256x784xf32>) -> tensor<32x200704xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v406 = stablehlo.convolution(%v405, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x28x28xf32>
    %v407 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x28x28xf32>
    %v408 = stablehlo.add %v406, %v407 : tensor<32x1024x28x28xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<32x1024x28x28xf32>) -> tensor<32x802816xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x802816xf32>) -> tensor<32x1024x28x28xf32>
    %v411 = stablehlo.multiply %v410, %v410 : tensor<32x1024x28x28xf32>
    %v412 = stablehlo.multiply %v411, %v410 : tensor<32x1024x28x28xf32>
    %v413 = stablehlo.constant dense<0.044715> : tensor<32x1024x28x28xf32>
    %v414 = stablehlo.multiply %v413, %v412 : tensor<32x1024x28x28xf32>
    %v415 = stablehlo.add %v410, %v414 : tensor<32x1024x28x28xf32>
    %v416 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1024x28x28xf32>
    %v417 = stablehlo.multiply %v416, %v415 : tensor<32x1024x28x28xf32>
    %v418 = stablehlo.tanh %v417 : tensor<32x1024x28x28xf32>
    %v419 = stablehlo.constant dense<1.0> : tensor<32x1024x28x28xf32>
    %v420 = stablehlo.add %v419, %v418 : tensor<32x1024x28x28xf32>
    %v421 = stablehlo.constant dense<0.5> : tensor<32x1024x28x28xf32>
    %v422 = stablehlo.multiply %v421, %v410 : tensor<32x1024x28x28xf32>
    %v423 = stablehlo.multiply %v422, %v420 : tensor<32x1024x28x28xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<32x1024x28x28xf32>) -> tensor<32x802816xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x802816xf32>) -> tensor<32x1024x28x28xf32>
    %v426 = stablehlo.convolution(%v425, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x28x28xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x28x28xf32>
    %v427 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v428 = stablehlo.add %v426, %v427 : tensor<32x256x28x28xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v431 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v432 = stablehlo.multiply %v430, %v431 : tensor<32x256x28x28xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v435 = stablehlo.reshape %v365 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v436 = stablehlo.add %v434, %v435 : tensor<32x256x28x28xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v439 = stablehlo.convolution(%v438, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x28x28xf32>, tensor<256x1x7x7xf32>) -> tensor<32x256x28x28xf32>
    %v440 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v441 = stablehlo.add %v439, %v440 : tensor<32x256x28x28xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x200704xf32>) -> tensor<32x256x784xf32>
    %v444 = stablehlo.transpose %v443, dims = [0, 2, 1] : (tensor<32x256x784xf32>) -> tensor<32x784x256xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v448 = stablehlo.constant dense<256.0> : tensor<32x784x256xf32>
    %v449 = stablehlo.constant dense<1.0e-6> : tensor<32x784x256xf32>
    %v450 = stablehlo.reduce(%v446 init: %v447) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v452 = stablehlo.divide %v451, %v448 : tensor<32x784x256xf32>
    %v453 = stablehlo.subtract %v446, %v452 : tensor<32x784x256xf32>
    %v454 = stablehlo.multiply %v453, %v453 : tensor<32x784x256xf32>
    %v455 = stablehlo.reduce(%v454 init: %v447) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v456 = stablehlo.broadcast_in_dim %v455, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v457 = stablehlo.divide %v456, %v448 : tensor<32x784x256xf32>
    %v458 = stablehlo.add %v457, %v449 : tensor<32x784x256xf32>
    %v459 = stablehlo.rsqrt %v458 : tensor<32x784x256xf32>
    %v460 = stablehlo.multiply %v453, %v459 : tensor<32x784x256xf32>
    %v461 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v462 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v463 = stablehlo.multiply %v460, %v461 : tensor<32x784x256xf32>
    %v464 = stablehlo.add %v463, %v462 : tensor<32x784x256xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v467 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v468 = stablehlo.multiply %v466, %v467 : tensor<32x784x256xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v471 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v472 = stablehlo.add %v470, %v471 : tensor<32x784x256xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v475 = stablehlo.transpose %v474, dims = [0, 2, 1] : (tensor<32x784x256xf32>) -> tensor<32x256x784xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x256x784xf32>) -> tensor<32x200704xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v478 = stablehlo.convolution(%v477, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x28x28xf32>
    %v479 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x28x28xf32>
    %v480 = stablehlo.add %v478, %v479 : tensor<32x1024x28x28xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x1024x28x28xf32>) -> tensor<32x802816xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x802816xf32>) -> tensor<32x1024x28x28xf32>
    %v483 = stablehlo.multiply %v482, %v482 : tensor<32x1024x28x28xf32>
    %v484 = stablehlo.multiply %v483, %v482 : tensor<32x1024x28x28xf32>
    %v485 = stablehlo.constant dense<0.044715> : tensor<32x1024x28x28xf32>
    %v486 = stablehlo.multiply %v485, %v484 : tensor<32x1024x28x28xf32>
    %v487 = stablehlo.add %v482, %v486 : tensor<32x1024x28x28xf32>
    %v488 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1024x28x28xf32>
    %v489 = stablehlo.multiply %v488, %v487 : tensor<32x1024x28x28xf32>
    %v490 = stablehlo.tanh %v489 : tensor<32x1024x28x28xf32>
    %v491 = stablehlo.constant dense<1.0> : tensor<32x1024x28x28xf32>
    %v492 = stablehlo.add %v491, %v490 : tensor<32x1024x28x28xf32>
    %v493 = stablehlo.constant dense<0.5> : tensor<32x1024x28x28xf32>
    %v494 = stablehlo.multiply %v493, %v482 : tensor<32x1024x28x28xf32>
    %v495 = stablehlo.multiply %v494, %v492 : tensor<32x1024x28x28xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<32x1024x28x28xf32>) -> tensor<32x802816xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x802816xf32>) -> tensor<32x1024x28x28xf32>
    %v498 = stablehlo.convolution(%v497, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x28x28xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x28x28xf32>
    %v499 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v500 = stablehlo.add %v498, %v499 : tensor<32x256x28x28xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v503 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v504 = stablehlo.multiply %v502, %v503 : tensor<32x256x28x28xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v507 = stablehlo.reshape %v437 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<32x256x28x28xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x200704xf32>) -> tensor<32x256x784xf32>
    %v511 = stablehlo.transpose %v510, dims = [0, 2, 1] : (tensor<32x256x784xf32>) -> tensor<32x784x256xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v515 = stablehlo.constant dense<256.0> : tensor<32x784x256xf32>
    %v516 = stablehlo.constant dense<1.0e-6> : tensor<32x784x256xf32>
    %v517 = stablehlo.reduce(%v513 init: %v514) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v518 = stablehlo.broadcast_in_dim %v517, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v519 = stablehlo.divide %v518, %v515 : tensor<32x784x256xf32>
    %v520 = stablehlo.subtract %v513, %v519 : tensor<32x784x256xf32>
    %v521 = stablehlo.multiply %v520, %v520 : tensor<32x784x256xf32>
    %v522 = stablehlo.reduce(%v521 init: %v514) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v523 = stablehlo.broadcast_in_dim %v522, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v524 = stablehlo.divide %v523, %v515 : tensor<32x784x256xf32>
    %v525 = stablehlo.add %v524, %v516 : tensor<32x784x256xf32>
    %v526 = stablehlo.rsqrt %v525 : tensor<32x784x256xf32>
    %v527 = stablehlo.multiply %v520, %v526 : tensor<32x784x256xf32>
    %v528 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v529 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v530 = stablehlo.multiply %v527, %v528 : tensor<32x784x256xf32>
    %v531 = stablehlo.add %v530, %v529 : tensor<32x784x256xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v534 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v535 = stablehlo.multiply %v533, %v534 : tensor<32x784x256xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v538 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v539 = stablehlo.add %v537, %v538 : tensor<32x784x256xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v542 = stablehlo.transpose %v541, dims = [0, 2, 1] : (tensor<32x784x256xf32>) -> tensor<32x256x784xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<32x256x784xf32>) -> tensor<32x200704xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v545 = stablehlo.convolution(%v544, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<512x256x2x2xf32>) -> tensor<32x512x14x14xf32>
    %v546 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v547 = stablehlo.add %v545, %v546 : tensor<32x512x14x14xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v550 = stablehlo.convolution(%v549, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v551 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v552 = stablehlo.add %v550, %v551 : tensor<32x512x14x14xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v555 = stablehlo.transpose %v554, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v558 = stablehlo.constant dense<0.0> : tensor<f32>
    %v559 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v560 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v561 = stablehlo.reduce(%v557 init: %v558) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v562 = stablehlo.broadcast_in_dim %v561, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v563 = stablehlo.divide %v562, %v559 : tensor<32x196x512xf32>
    %v564 = stablehlo.subtract %v557, %v563 : tensor<32x196x512xf32>
    %v565 = stablehlo.multiply %v564, %v564 : tensor<32x196x512xf32>
    %v566 = stablehlo.reduce(%v565 init: %v558) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v567 = stablehlo.broadcast_in_dim %v566, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v568 = stablehlo.divide %v567, %v559 : tensor<32x196x512xf32>
    %v569 = stablehlo.add %v568, %v560 : tensor<32x196x512xf32>
    %v570 = stablehlo.rsqrt %v569 : tensor<32x196x512xf32>
    %v571 = stablehlo.multiply %v564, %v570 : tensor<32x196x512xf32>
    %v572 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v573 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v574 = stablehlo.multiply %v571, %v572 : tensor<32x196x512xf32>
    %v575 = stablehlo.add %v574, %v573 : tensor<32x196x512xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v578 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v579 = stablehlo.multiply %v577, %v578 : tensor<32x196x512xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v582 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v583 = stablehlo.add %v581, %v582 : tensor<32x196x512xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v586 = stablehlo.transpose %v585, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v589 = stablehlo.convolution(%v588, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v590 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v591 = stablehlo.add %v589, %v590 : tensor<32x2048x14x14xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v594 = stablehlo.multiply %v593, %v593 : tensor<32x2048x14x14xf32>
    %v595 = stablehlo.multiply %v594, %v593 : tensor<32x2048x14x14xf32>
    %v596 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v597 = stablehlo.multiply %v596, %v595 : tensor<32x2048x14x14xf32>
    %v598 = stablehlo.add %v593, %v597 : tensor<32x2048x14x14xf32>
    %v599 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v600 = stablehlo.multiply %v599, %v598 : tensor<32x2048x14x14xf32>
    %v601 = stablehlo.tanh %v600 : tensor<32x2048x14x14xf32>
    %v602 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v603 = stablehlo.add %v602, %v601 : tensor<32x2048x14x14xf32>
    %v604 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v605 = stablehlo.multiply %v604, %v593 : tensor<32x2048x14x14xf32>
    %v606 = stablehlo.multiply %v605, %v603 : tensor<32x2048x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v609 = stablehlo.convolution(%v608, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<32x512x14x14xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v615 = stablehlo.multiply %v613, %v614 : tensor<32x512x14x14xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v618 = stablehlo.reshape %v548 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v619 = stablehlo.add %v617, %v618 : tensor<32x512x14x14xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v622 = stablehlo.convolution(%v621, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v624 = stablehlo.add %v622, %v623 : tensor<32x512x14x14xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v627 = stablehlo.transpose %v626, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v631 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v632 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v633 = stablehlo.reduce(%v629 init: %v630) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v634 = stablehlo.broadcast_in_dim %v633, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v635 = stablehlo.divide %v634, %v631 : tensor<32x196x512xf32>
    %v636 = stablehlo.subtract %v629, %v635 : tensor<32x196x512xf32>
    %v637 = stablehlo.multiply %v636, %v636 : tensor<32x196x512xf32>
    %v638 = stablehlo.reduce(%v637 init: %v630) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v639 = stablehlo.broadcast_in_dim %v638, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v640 = stablehlo.divide %v639, %v631 : tensor<32x196x512xf32>
    %v641 = stablehlo.add %v640, %v632 : tensor<32x196x512xf32>
    %v642 = stablehlo.rsqrt %v641 : tensor<32x196x512xf32>
    %v643 = stablehlo.multiply %v636, %v642 : tensor<32x196x512xf32>
    %v644 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v645 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v646 = stablehlo.multiply %v643, %v644 : tensor<32x196x512xf32>
    %v647 = stablehlo.add %v646, %v645 : tensor<32x196x512xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v650 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v651 = stablehlo.multiply %v649, %v650 : tensor<32x196x512xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v654 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v655 = stablehlo.add %v653, %v654 : tensor<32x196x512xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v658 = stablehlo.transpose %v657, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v661 = stablehlo.convolution(%v660, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v663 = stablehlo.add %v661, %v662 : tensor<32x2048x14x14xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v666 = stablehlo.multiply %v665, %v665 : tensor<32x2048x14x14xf32>
    %v667 = stablehlo.multiply %v666, %v665 : tensor<32x2048x14x14xf32>
    %v668 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v669 = stablehlo.multiply %v668, %v667 : tensor<32x2048x14x14xf32>
    %v670 = stablehlo.add %v665, %v669 : tensor<32x2048x14x14xf32>
    %v671 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v672 = stablehlo.multiply %v671, %v670 : tensor<32x2048x14x14xf32>
    %v673 = stablehlo.tanh %v672 : tensor<32x2048x14x14xf32>
    %v674 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v675 = stablehlo.add %v674, %v673 : tensor<32x2048x14x14xf32>
    %v676 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v677 = stablehlo.multiply %v676, %v665 : tensor<32x2048x14x14xf32>
    %v678 = stablehlo.multiply %v677, %v675 : tensor<32x2048x14x14xf32>
    %v679 = stablehlo.reshape %v678 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v681 = stablehlo.convolution(%v680, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v682 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v683 = stablehlo.add %v681, %v682 : tensor<32x512x14x14xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v686 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v687 = stablehlo.multiply %v685, %v686 : tensor<32x512x14x14xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v690 = stablehlo.reshape %v620 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v691 = stablehlo.add %v689, %v690 : tensor<32x512x14x14xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v694 = stablehlo.convolution(%v693, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v695 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v696 = stablehlo.add %v694, %v695 : tensor<32x512x14x14xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v699 = stablehlo.transpose %v698, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v703 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v704 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v705 = stablehlo.reduce(%v701 init: %v702) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v706 = stablehlo.broadcast_in_dim %v705, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v707 = stablehlo.divide %v706, %v703 : tensor<32x196x512xf32>
    %v708 = stablehlo.subtract %v701, %v707 : tensor<32x196x512xf32>
    %v709 = stablehlo.multiply %v708, %v708 : tensor<32x196x512xf32>
    %v710 = stablehlo.reduce(%v709 init: %v702) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v711 = stablehlo.broadcast_in_dim %v710, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v712 = stablehlo.divide %v711, %v703 : tensor<32x196x512xf32>
    %v713 = stablehlo.add %v712, %v704 : tensor<32x196x512xf32>
    %v714 = stablehlo.rsqrt %v713 : tensor<32x196x512xf32>
    %v715 = stablehlo.multiply %v708, %v714 : tensor<32x196x512xf32>
    %v716 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v717 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v718 = stablehlo.multiply %v715, %v716 : tensor<32x196x512xf32>
    %v719 = stablehlo.add %v718, %v717 : tensor<32x196x512xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v722 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v723 = stablehlo.multiply %v721, %v722 : tensor<32x196x512xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v726 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v727 = stablehlo.add %v725, %v726 : tensor<32x196x512xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v730 = stablehlo.transpose %v729, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v733 = stablehlo.convolution(%v732, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<32x2048x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v738 = stablehlo.multiply %v737, %v737 : tensor<32x2048x14x14xf32>
    %v739 = stablehlo.multiply %v738, %v737 : tensor<32x2048x14x14xf32>
    %v740 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v741 = stablehlo.multiply %v740, %v739 : tensor<32x2048x14x14xf32>
    %v742 = stablehlo.add %v737, %v741 : tensor<32x2048x14x14xf32>
    %v743 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v744 = stablehlo.multiply %v743, %v742 : tensor<32x2048x14x14xf32>
    %v745 = stablehlo.tanh %v744 : tensor<32x2048x14x14xf32>
    %v746 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v747 = stablehlo.add %v746, %v745 : tensor<32x2048x14x14xf32>
    %v748 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v749 = stablehlo.multiply %v748, %v737 : tensor<32x2048x14x14xf32>
    %v750 = stablehlo.multiply %v749, %v747 : tensor<32x2048x14x14xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v753 = stablehlo.convolution(%v752, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v754 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v755 = stablehlo.add %v753, %v754 : tensor<32x512x14x14xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v758 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v759 = stablehlo.multiply %v757, %v758 : tensor<32x512x14x14xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v762 = stablehlo.reshape %v692 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v763 = stablehlo.add %v761, %v762 : tensor<32x512x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v766 = stablehlo.convolution(%v765, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x512x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v771 = stablehlo.transpose %v770, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v775 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v776 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v777 = stablehlo.reduce(%v773 init: %v774) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v778 = stablehlo.broadcast_in_dim %v777, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v779 = stablehlo.divide %v778, %v775 : tensor<32x196x512xf32>
    %v780 = stablehlo.subtract %v773, %v779 : tensor<32x196x512xf32>
    %v781 = stablehlo.multiply %v780, %v780 : tensor<32x196x512xf32>
    %v782 = stablehlo.reduce(%v781 init: %v774) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v783 = stablehlo.broadcast_in_dim %v782, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v784 = stablehlo.divide %v783, %v775 : tensor<32x196x512xf32>
    %v785 = stablehlo.add %v784, %v776 : tensor<32x196x512xf32>
    %v786 = stablehlo.rsqrt %v785 : tensor<32x196x512xf32>
    %v787 = stablehlo.multiply %v780, %v786 : tensor<32x196x512xf32>
    %v788 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v789 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v790 = stablehlo.multiply %v787, %v788 : tensor<32x196x512xf32>
    %v791 = stablehlo.add %v790, %v789 : tensor<32x196x512xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v794 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v795 = stablehlo.multiply %v793, %v794 : tensor<32x196x512xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v798 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<32x196x512xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v802 = stablehlo.transpose %v801, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v805 = stablehlo.convolution(%v804, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v806 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v807 = stablehlo.add %v805, %v806 : tensor<32x2048x14x14xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v810 = stablehlo.multiply %v809, %v809 : tensor<32x2048x14x14xf32>
    %v811 = stablehlo.multiply %v810, %v809 : tensor<32x2048x14x14xf32>
    %v812 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v813 = stablehlo.multiply %v812, %v811 : tensor<32x2048x14x14xf32>
    %v814 = stablehlo.add %v809, %v813 : tensor<32x2048x14x14xf32>
    %v815 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v816 = stablehlo.multiply %v815, %v814 : tensor<32x2048x14x14xf32>
    %v817 = stablehlo.tanh %v816 : tensor<32x2048x14x14xf32>
    %v818 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v819 = stablehlo.add %v818, %v817 : tensor<32x2048x14x14xf32>
    %v820 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v821 = stablehlo.multiply %v820, %v809 : tensor<32x2048x14x14xf32>
    %v822 = stablehlo.multiply %v821, %v819 : tensor<32x2048x14x14xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v825 = stablehlo.convolution(%v824, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v826 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v827 = stablehlo.add %v825, %v826 : tensor<32x512x14x14xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v830 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v831 = stablehlo.multiply %v829, %v830 : tensor<32x512x14x14xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v834 = stablehlo.reshape %v764 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v835 = stablehlo.add %v833, %v834 : tensor<32x512x14x14xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v838 = stablehlo.convolution(%v837, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v839 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v840 = stablehlo.add %v838, %v839 : tensor<32x512x14x14xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v843 = stablehlo.transpose %v842, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v847 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v848 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v849 = stablehlo.reduce(%v845 init: %v846) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v850 = stablehlo.broadcast_in_dim %v849, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v851 = stablehlo.divide %v850, %v847 : tensor<32x196x512xf32>
    %v852 = stablehlo.subtract %v845, %v851 : tensor<32x196x512xf32>
    %v853 = stablehlo.multiply %v852, %v852 : tensor<32x196x512xf32>
    %v854 = stablehlo.reduce(%v853 init: %v846) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v855 = stablehlo.broadcast_in_dim %v854, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v856 = stablehlo.divide %v855, %v847 : tensor<32x196x512xf32>
    %v857 = stablehlo.add %v856, %v848 : tensor<32x196x512xf32>
    %v858 = stablehlo.rsqrt %v857 : tensor<32x196x512xf32>
    %v859 = stablehlo.multiply %v852, %v858 : tensor<32x196x512xf32>
    %v860 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v861 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v862 = stablehlo.multiply %v859, %v860 : tensor<32x196x512xf32>
    %v863 = stablehlo.add %v862, %v861 : tensor<32x196x512xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v866 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v867 = stablehlo.multiply %v865, %v866 : tensor<32x196x512xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v870 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v871 = stablehlo.add %v869, %v870 : tensor<32x196x512xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v874 = stablehlo.transpose %v873, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v877 = stablehlo.convolution(%v876, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v878 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<32x2048x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v882 = stablehlo.multiply %v881, %v881 : tensor<32x2048x14x14xf32>
    %v883 = stablehlo.multiply %v882, %v881 : tensor<32x2048x14x14xf32>
    %v884 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v885 = stablehlo.multiply %v884, %v883 : tensor<32x2048x14x14xf32>
    %v886 = stablehlo.add %v881, %v885 : tensor<32x2048x14x14xf32>
    %v887 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v888 = stablehlo.multiply %v887, %v886 : tensor<32x2048x14x14xf32>
    %v889 = stablehlo.tanh %v888 : tensor<32x2048x14x14xf32>
    %v890 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v891 = stablehlo.add %v890, %v889 : tensor<32x2048x14x14xf32>
    %v892 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v893 = stablehlo.multiply %v892, %v881 : tensor<32x2048x14x14xf32>
    %v894 = stablehlo.multiply %v893, %v891 : tensor<32x2048x14x14xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v897 = stablehlo.convolution(%v896, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v898 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v899 = stablehlo.add %v897, %v898 : tensor<32x512x14x14xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v901 = stablehlo.reshape %v900 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v902 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v903 = stablehlo.multiply %v901, %v902 : tensor<32x512x14x14xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v906 = stablehlo.reshape %v836 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v907 = stablehlo.add %v905, %v906 : tensor<32x512x14x14xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v910 = stablehlo.convolution(%v909, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v911 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v912 = stablehlo.add %v910, %v911 : tensor<32x512x14x14xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v915 = stablehlo.transpose %v914, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v919 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v920 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v921 = stablehlo.reduce(%v917 init: %v918) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v922 = stablehlo.broadcast_in_dim %v921, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v923 = stablehlo.divide %v922, %v919 : tensor<32x196x512xf32>
    %v924 = stablehlo.subtract %v917, %v923 : tensor<32x196x512xf32>
    %v925 = stablehlo.multiply %v924, %v924 : tensor<32x196x512xf32>
    %v926 = stablehlo.reduce(%v925 init: %v918) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v927 = stablehlo.broadcast_in_dim %v926, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v928 = stablehlo.divide %v927, %v919 : tensor<32x196x512xf32>
    %v929 = stablehlo.add %v928, %v920 : tensor<32x196x512xf32>
    %v930 = stablehlo.rsqrt %v929 : tensor<32x196x512xf32>
    %v931 = stablehlo.multiply %v924, %v930 : tensor<32x196x512xf32>
    %v932 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v933 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v934 = stablehlo.multiply %v931, %v932 : tensor<32x196x512xf32>
    %v935 = stablehlo.add %v934, %v933 : tensor<32x196x512xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v938 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v939 = stablehlo.multiply %v937, %v938 : tensor<32x196x512xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v942 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v943 = stablehlo.add %v941, %v942 : tensor<32x196x512xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v946 = stablehlo.transpose %v945, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v949 = stablehlo.convolution(%v948, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v950 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v951 = stablehlo.add %v949, %v950 : tensor<32x2048x14x14xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v954 = stablehlo.multiply %v953, %v953 : tensor<32x2048x14x14xf32>
    %v955 = stablehlo.multiply %v954, %v953 : tensor<32x2048x14x14xf32>
    %v956 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v957 = stablehlo.multiply %v956, %v955 : tensor<32x2048x14x14xf32>
    %v958 = stablehlo.add %v953, %v957 : tensor<32x2048x14x14xf32>
    %v959 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v960 = stablehlo.multiply %v959, %v958 : tensor<32x2048x14x14xf32>
    %v961 = stablehlo.tanh %v960 : tensor<32x2048x14x14xf32>
    %v962 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v963 = stablehlo.add %v962, %v961 : tensor<32x2048x14x14xf32>
    %v964 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v965 = stablehlo.multiply %v964, %v953 : tensor<32x2048x14x14xf32>
    %v966 = stablehlo.multiply %v965, %v963 : tensor<32x2048x14x14xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v969 = stablehlo.convolution(%v968, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v970 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v971 = stablehlo.add %v969, %v970 : tensor<32x512x14x14xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v974 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v975 = stablehlo.multiply %v973, %v974 : tensor<32x512x14x14xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v978 = stablehlo.reshape %v908 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v979 = stablehlo.add %v977, %v978 : tensor<32x512x14x14xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v982 = stablehlo.convolution(%v981, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v983 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v984 = stablehlo.add %v982, %v983 : tensor<32x512x14x14xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v987 = stablehlo.transpose %v986, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v991 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v992 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v993 = stablehlo.reduce(%v989 init: %v990) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v994 = stablehlo.broadcast_in_dim %v993, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v995 = stablehlo.divide %v994, %v991 : tensor<32x196x512xf32>
    %v996 = stablehlo.subtract %v989, %v995 : tensor<32x196x512xf32>
    %v997 = stablehlo.multiply %v996, %v996 : tensor<32x196x512xf32>
    %v998 = stablehlo.reduce(%v997 init: %v990) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v999 = stablehlo.broadcast_in_dim %v998, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1000 = stablehlo.divide %v999, %v991 : tensor<32x196x512xf32>
    %v1001 = stablehlo.add %v1000, %v992 : tensor<32x196x512xf32>
    %v1002 = stablehlo.rsqrt %v1001 : tensor<32x196x512xf32>
    %v1003 = stablehlo.multiply %v996, %v1002 : tensor<32x196x512xf32>
    %v1004 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1005 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1006 = stablehlo.multiply %v1003, %v1004 : tensor<32x196x512xf32>
    %v1007 = stablehlo.add %v1006, %v1005 : tensor<32x196x512xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1010 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1011 = stablehlo.multiply %v1009, %v1010 : tensor<32x196x512xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1014 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1015 = stablehlo.add %v1013, %v1014 : tensor<32x196x512xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1018 = stablehlo.transpose %v1017, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1021 = stablehlo.convolution(%v1020, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1022 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1023 = stablehlo.add %v1021, %v1022 : tensor<32x2048x14x14xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1025 = stablehlo.reshape %v1024 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1026 = stablehlo.multiply %v1025, %v1025 : tensor<32x2048x14x14xf32>
    %v1027 = stablehlo.multiply %v1026, %v1025 : tensor<32x2048x14x14xf32>
    %v1028 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1029 = stablehlo.multiply %v1028, %v1027 : tensor<32x2048x14x14xf32>
    %v1030 = stablehlo.add %v1025, %v1029 : tensor<32x2048x14x14xf32>
    %v1031 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1032 = stablehlo.multiply %v1031, %v1030 : tensor<32x2048x14x14xf32>
    %v1033 = stablehlo.tanh %v1032 : tensor<32x2048x14x14xf32>
    %v1034 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1035 = stablehlo.add %v1034, %v1033 : tensor<32x2048x14x14xf32>
    %v1036 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1037 = stablehlo.multiply %v1036, %v1025 : tensor<32x2048x14x14xf32>
    %v1038 = stablehlo.multiply %v1037, %v1035 : tensor<32x2048x14x14xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1041 = stablehlo.convolution(%v1040, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1042 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1043 = stablehlo.add %v1041, %v1042 : tensor<32x512x14x14xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1046 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1047 = stablehlo.multiply %v1045, %v1046 : tensor<32x512x14x14xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1050 = stablehlo.reshape %v980 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1051 = stablehlo.add %v1049, %v1050 : tensor<32x512x14x14xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1054 = stablehlo.convolution(%v1053, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1055 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1056 = stablehlo.add %v1054, %v1055 : tensor<32x512x14x14xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1059 = stablehlo.transpose %v1058, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1063 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1064 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1065 = stablehlo.reduce(%v1061 init: %v1062) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1066 = stablehlo.broadcast_in_dim %v1065, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1067 = stablehlo.divide %v1066, %v1063 : tensor<32x196x512xf32>
    %v1068 = stablehlo.subtract %v1061, %v1067 : tensor<32x196x512xf32>
    %v1069 = stablehlo.multiply %v1068, %v1068 : tensor<32x196x512xf32>
    %v1070 = stablehlo.reduce(%v1069 init: %v1062) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1071 = stablehlo.broadcast_in_dim %v1070, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1072 = stablehlo.divide %v1071, %v1063 : tensor<32x196x512xf32>
    %v1073 = stablehlo.add %v1072, %v1064 : tensor<32x196x512xf32>
    %v1074 = stablehlo.rsqrt %v1073 : tensor<32x196x512xf32>
    %v1075 = stablehlo.multiply %v1068, %v1074 : tensor<32x196x512xf32>
    %v1076 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1077 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1078 = stablehlo.multiply %v1075, %v1076 : tensor<32x196x512xf32>
    %v1079 = stablehlo.add %v1078, %v1077 : tensor<32x196x512xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1082 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1083 = stablehlo.multiply %v1081, %v1082 : tensor<32x196x512xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1086 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1087 = stablehlo.add %v1085, %v1086 : tensor<32x196x512xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1090 = stablehlo.transpose %v1089, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1093 = stablehlo.convolution(%v1092, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1094 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1095 = stablehlo.add %v1093, %v1094 : tensor<32x2048x14x14xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1098 = stablehlo.multiply %v1097, %v1097 : tensor<32x2048x14x14xf32>
    %v1099 = stablehlo.multiply %v1098, %v1097 : tensor<32x2048x14x14xf32>
    %v1100 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1101 = stablehlo.multiply %v1100, %v1099 : tensor<32x2048x14x14xf32>
    %v1102 = stablehlo.add %v1097, %v1101 : tensor<32x2048x14x14xf32>
    %v1103 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1104 = stablehlo.multiply %v1103, %v1102 : tensor<32x2048x14x14xf32>
    %v1105 = stablehlo.tanh %v1104 : tensor<32x2048x14x14xf32>
    %v1106 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1107 = stablehlo.add %v1106, %v1105 : tensor<32x2048x14x14xf32>
    %v1108 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1109 = stablehlo.multiply %v1108, %v1097 : tensor<32x2048x14x14xf32>
    %v1110 = stablehlo.multiply %v1109, %v1107 : tensor<32x2048x14x14xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1113 = stablehlo.convolution(%v1112, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1114 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1115 = stablehlo.add %v1113, %v1114 : tensor<32x512x14x14xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1117 = stablehlo.reshape %v1116 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1118 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1119 = stablehlo.multiply %v1117, %v1118 : tensor<32x512x14x14xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1122 = stablehlo.reshape %v1052 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1123 = stablehlo.add %v1121, %v1122 : tensor<32x512x14x14xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1126 = stablehlo.convolution(%v1125, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1127 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1128 = stablehlo.add %v1126, %v1127 : tensor<32x512x14x14xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1130 = stablehlo.reshape %v1129 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1131 = stablehlo.transpose %v1130, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1134 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1135 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1136 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1137 = stablehlo.reduce(%v1133 init: %v1134) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1138 = stablehlo.broadcast_in_dim %v1137, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1139 = stablehlo.divide %v1138, %v1135 : tensor<32x196x512xf32>
    %v1140 = stablehlo.subtract %v1133, %v1139 : tensor<32x196x512xf32>
    %v1141 = stablehlo.multiply %v1140, %v1140 : tensor<32x196x512xf32>
    %v1142 = stablehlo.reduce(%v1141 init: %v1134) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1143 = stablehlo.broadcast_in_dim %v1142, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1144 = stablehlo.divide %v1143, %v1135 : tensor<32x196x512xf32>
    %v1145 = stablehlo.add %v1144, %v1136 : tensor<32x196x512xf32>
    %v1146 = stablehlo.rsqrt %v1145 : tensor<32x196x512xf32>
    %v1147 = stablehlo.multiply %v1140, %v1146 : tensor<32x196x512xf32>
    %v1148 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1149 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1150 = stablehlo.multiply %v1147, %v1148 : tensor<32x196x512xf32>
    %v1151 = stablehlo.add %v1150, %v1149 : tensor<32x196x512xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1154 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1155 = stablehlo.multiply %v1153, %v1154 : tensor<32x196x512xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1157 = stablehlo.reshape %v1156 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1158 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1159 = stablehlo.add %v1157, %v1158 : tensor<32x196x512xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1162 = stablehlo.transpose %v1161, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1165 = stablehlo.convolution(%v1164, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1166 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1167 = stablehlo.add %v1165, %v1166 : tensor<32x2048x14x14xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1170 = stablehlo.multiply %v1169, %v1169 : tensor<32x2048x14x14xf32>
    %v1171 = stablehlo.multiply %v1170, %v1169 : tensor<32x2048x14x14xf32>
    %v1172 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1173 = stablehlo.multiply %v1172, %v1171 : tensor<32x2048x14x14xf32>
    %v1174 = stablehlo.add %v1169, %v1173 : tensor<32x2048x14x14xf32>
    %v1175 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1176 = stablehlo.multiply %v1175, %v1174 : tensor<32x2048x14x14xf32>
    %v1177 = stablehlo.tanh %v1176 : tensor<32x2048x14x14xf32>
    %v1178 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1179 = stablehlo.add %v1178, %v1177 : tensor<32x2048x14x14xf32>
    %v1180 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1181 = stablehlo.multiply %v1180, %v1169 : tensor<32x2048x14x14xf32>
    %v1182 = stablehlo.multiply %v1181, %v1179 : tensor<32x2048x14x14xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1185 = stablehlo.convolution(%v1184, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1186 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1187 = stablehlo.add %v1185, %v1186 : tensor<32x512x14x14xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1190 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1191 = stablehlo.multiply %v1189, %v1190 : tensor<32x512x14x14xf32>
    %v1192 = stablehlo.reshape %v1191 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1194 = stablehlo.reshape %v1124 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1195 = stablehlo.add %v1193, %v1194 : tensor<32x512x14x14xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1197 = stablehlo.reshape %v1196 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1198 = stablehlo.convolution(%v1197, %s2b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1199 = stablehlo.broadcast_in_dim %s2b9db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1200 = stablehlo.add %v1198, %v1199 : tensor<32x512x14x14xf32>
    %v1201 = stablehlo.reshape %v1200 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1202 = stablehlo.reshape %v1201 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1203 = stablehlo.transpose %v1202, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1205 = stablehlo.reshape %v1204 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1207 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1208 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1209 = stablehlo.reduce(%v1205 init: %v1206) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1210 = stablehlo.broadcast_in_dim %v1209, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1211 = stablehlo.divide %v1210, %v1207 : tensor<32x196x512xf32>
    %v1212 = stablehlo.subtract %v1205, %v1211 : tensor<32x196x512xf32>
    %v1213 = stablehlo.multiply %v1212, %v1212 : tensor<32x196x512xf32>
    %v1214 = stablehlo.reduce(%v1213 init: %v1206) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1215 = stablehlo.broadcast_in_dim %v1214, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1216 = stablehlo.divide %v1215, %v1207 : tensor<32x196x512xf32>
    %v1217 = stablehlo.add %v1216, %v1208 : tensor<32x196x512xf32>
    %v1218 = stablehlo.rsqrt %v1217 : tensor<32x196x512xf32>
    %v1219 = stablehlo.multiply %v1212, %v1218 : tensor<32x196x512xf32>
    %v1220 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1221 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1222 = stablehlo.multiply %v1219, %v1220 : tensor<32x196x512xf32>
    %v1223 = stablehlo.add %v1222, %v1221 : tensor<32x196x512xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1226 = stablehlo.broadcast_in_dim %s2b9ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1227 = stablehlo.multiply %v1225, %v1226 : tensor<32x196x512xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1230 = stablehlo.broadcast_in_dim %s2b9nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1231 = stablehlo.add %v1229, %v1230 : tensor<32x196x512xf32>
    %v1232 = stablehlo.reshape %v1231 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1234 = stablehlo.transpose %v1233, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1236 = stablehlo.reshape %v1235 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1237 = stablehlo.convolution(%v1236, %s2b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1238 = stablehlo.broadcast_in_dim %s2b9eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1239 = stablehlo.add %v1237, %v1238 : tensor<32x2048x14x14xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1242 = stablehlo.multiply %v1241, %v1241 : tensor<32x2048x14x14xf32>
    %v1243 = stablehlo.multiply %v1242, %v1241 : tensor<32x2048x14x14xf32>
    %v1244 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1245 = stablehlo.multiply %v1244, %v1243 : tensor<32x2048x14x14xf32>
    %v1246 = stablehlo.add %v1241, %v1245 : tensor<32x2048x14x14xf32>
    %v1247 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1248 = stablehlo.multiply %v1247, %v1246 : tensor<32x2048x14x14xf32>
    %v1249 = stablehlo.tanh %v1248 : tensor<32x2048x14x14xf32>
    %v1250 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1251 = stablehlo.add %v1250, %v1249 : tensor<32x2048x14x14xf32>
    %v1252 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1253 = stablehlo.multiply %v1252, %v1241 : tensor<32x2048x14x14xf32>
    %v1254 = stablehlo.multiply %v1253, %v1251 : tensor<32x2048x14x14xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1257 = stablehlo.convolution(%v1256, %s2b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1258 = stablehlo.broadcast_in_dim %s2b9pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1259 = stablehlo.add %v1257, %v1258 : tensor<32x512x14x14xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1261 = stablehlo.reshape %v1260 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1262 = stablehlo.broadcast_in_dim %s2b9lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1263 = stablehlo.multiply %v1261, %v1262 : tensor<32x512x14x14xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1266 = stablehlo.reshape %v1196 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1267 = stablehlo.add %v1265, %v1266 : tensor<32x512x14x14xf32>
    %v1268 = stablehlo.reshape %v1267 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1269 = stablehlo.reshape %v1268 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1270 = stablehlo.convolution(%v1269, %s2b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1271 = stablehlo.broadcast_in_dim %s2b10db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1272 = stablehlo.add %v1270, %v1271 : tensor<32x512x14x14xf32>
    %v1273 = stablehlo.reshape %v1272 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1275 = stablehlo.transpose %v1274, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1276 = stablehlo.reshape %v1275 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1279 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1280 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1281 = stablehlo.reduce(%v1277 init: %v1278) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1282 = stablehlo.broadcast_in_dim %v1281, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1283 = stablehlo.divide %v1282, %v1279 : tensor<32x196x512xf32>
    %v1284 = stablehlo.subtract %v1277, %v1283 : tensor<32x196x512xf32>
    %v1285 = stablehlo.multiply %v1284, %v1284 : tensor<32x196x512xf32>
    %v1286 = stablehlo.reduce(%v1285 init: %v1278) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1287 = stablehlo.broadcast_in_dim %v1286, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1288 = stablehlo.divide %v1287, %v1279 : tensor<32x196x512xf32>
    %v1289 = stablehlo.add %v1288, %v1280 : tensor<32x196x512xf32>
    %v1290 = stablehlo.rsqrt %v1289 : tensor<32x196x512xf32>
    %v1291 = stablehlo.multiply %v1284, %v1290 : tensor<32x196x512xf32>
    %v1292 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1293 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1294 = stablehlo.multiply %v1291, %v1292 : tensor<32x196x512xf32>
    %v1295 = stablehlo.add %v1294, %v1293 : tensor<32x196x512xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1297 = stablehlo.reshape %v1296 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1298 = stablehlo.broadcast_in_dim %s2b10ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1299 = stablehlo.multiply %v1297, %v1298 : tensor<32x196x512xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1302 = stablehlo.broadcast_in_dim %s2b10nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1303 = stablehlo.add %v1301, %v1302 : tensor<32x196x512xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1306 = stablehlo.transpose %v1305, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1309 = stablehlo.convolution(%v1308, %s2b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1310 = stablehlo.broadcast_in_dim %s2b10eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1311 = stablehlo.add %v1309, %v1310 : tensor<32x2048x14x14xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1314 = stablehlo.multiply %v1313, %v1313 : tensor<32x2048x14x14xf32>
    %v1315 = stablehlo.multiply %v1314, %v1313 : tensor<32x2048x14x14xf32>
    %v1316 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1317 = stablehlo.multiply %v1316, %v1315 : tensor<32x2048x14x14xf32>
    %v1318 = stablehlo.add %v1313, %v1317 : tensor<32x2048x14x14xf32>
    %v1319 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1320 = stablehlo.multiply %v1319, %v1318 : tensor<32x2048x14x14xf32>
    %v1321 = stablehlo.tanh %v1320 : tensor<32x2048x14x14xf32>
    %v1322 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1323 = stablehlo.add %v1322, %v1321 : tensor<32x2048x14x14xf32>
    %v1324 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1325 = stablehlo.multiply %v1324, %v1313 : tensor<32x2048x14x14xf32>
    %v1326 = stablehlo.multiply %v1325, %v1323 : tensor<32x2048x14x14xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1329 = stablehlo.convolution(%v1328, %s2b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1330 = stablehlo.broadcast_in_dim %s2b10pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1331 = stablehlo.add %v1329, %v1330 : tensor<32x512x14x14xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1334 = stablehlo.broadcast_in_dim %s2b10lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1335 = stablehlo.multiply %v1333, %v1334 : tensor<32x512x14x14xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1338 = stablehlo.reshape %v1268 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1339 = stablehlo.add %v1337, %v1338 : tensor<32x512x14x14xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1342 = stablehlo.convolution(%v1341, %s2b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1343 = stablehlo.broadcast_in_dim %s2b11db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1344 = stablehlo.add %v1342, %v1343 : tensor<32x512x14x14xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1347 = stablehlo.transpose %v1346, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1351 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1352 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1353 = stablehlo.reduce(%v1349 init: %v1350) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1354 = stablehlo.broadcast_in_dim %v1353, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1355 = stablehlo.divide %v1354, %v1351 : tensor<32x196x512xf32>
    %v1356 = stablehlo.subtract %v1349, %v1355 : tensor<32x196x512xf32>
    %v1357 = stablehlo.multiply %v1356, %v1356 : tensor<32x196x512xf32>
    %v1358 = stablehlo.reduce(%v1357 init: %v1350) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1359 = stablehlo.broadcast_in_dim %v1358, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1360 = stablehlo.divide %v1359, %v1351 : tensor<32x196x512xf32>
    %v1361 = stablehlo.add %v1360, %v1352 : tensor<32x196x512xf32>
    %v1362 = stablehlo.rsqrt %v1361 : tensor<32x196x512xf32>
    %v1363 = stablehlo.multiply %v1356, %v1362 : tensor<32x196x512xf32>
    %v1364 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1365 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1366 = stablehlo.multiply %v1363, %v1364 : tensor<32x196x512xf32>
    %v1367 = stablehlo.add %v1366, %v1365 : tensor<32x196x512xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1369 = stablehlo.reshape %v1368 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1370 = stablehlo.broadcast_in_dim %s2b11ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1371 = stablehlo.multiply %v1369, %v1370 : tensor<32x196x512xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1373 = stablehlo.reshape %v1372 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1374 = stablehlo.broadcast_in_dim %s2b11nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1375 = stablehlo.add %v1373, %v1374 : tensor<32x196x512xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1378 = stablehlo.transpose %v1377, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1381 = stablehlo.convolution(%v1380, %s2b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1382 = stablehlo.broadcast_in_dim %s2b11eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1383 = stablehlo.add %v1381, %v1382 : tensor<32x2048x14x14xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1386 = stablehlo.multiply %v1385, %v1385 : tensor<32x2048x14x14xf32>
    %v1387 = stablehlo.multiply %v1386, %v1385 : tensor<32x2048x14x14xf32>
    %v1388 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1389 = stablehlo.multiply %v1388, %v1387 : tensor<32x2048x14x14xf32>
    %v1390 = stablehlo.add %v1385, %v1389 : tensor<32x2048x14x14xf32>
    %v1391 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1392 = stablehlo.multiply %v1391, %v1390 : tensor<32x2048x14x14xf32>
    %v1393 = stablehlo.tanh %v1392 : tensor<32x2048x14x14xf32>
    %v1394 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1395 = stablehlo.add %v1394, %v1393 : tensor<32x2048x14x14xf32>
    %v1396 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1397 = stablehlo.multiply %v1396, %v1385 : tensor<32x2048x14x14xf32>
    %v1398 = stablehlo.multiply %v1397, %v1395 : tensor<32x2048x14x14xf32>
    %v1399 = stablehlo.reshape %v1398 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1401 = stablehlo.convolution(%v1400, %s2b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1402 = stablehlo.broadcast_in_dim %s2b11pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1403 = stablehlo.add %v1401, %v1402 : tensor<32x512x14x14xf32>
    %v1404 = stablehlo.reshape %v1403 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1405 = stablehlo.reshape %v1404 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1406 = stablehlo.broadcast_in_dim %s2b11lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1407 = stablehlo.multiply %v1405, %v1406 : tensor<32x512x14x14xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1410 = stablehlo.reshape %v1340 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1411 = stablehlo.add %v1409, %v1410 : tensor<32x512x14x14xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1414 = stablehlo.convolution(%v1413, %s2b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1415 = stablehlo.broadcast_in_dim %s2b12db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1416 = stablehlo.add %v1414, %v1415 : tensor<32x512x14x14xf32>
    %v1417 = stablehlo.reshape %v1416 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1418 = stablehlo.reshape %v1417 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1419 = stablehlo.transpose %v1418, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1423 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1424 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1425 = stablehlo.reduce(%v1421 init: %v1422) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1426 = stablehlo.broadcast_in_dim %v1425, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1427 = stablehlo.divide %v1426, %v1423 : tensor<32x196x512xf32>
    %v1428 = stablehlo.subtract %v1421, %v1427 : tensor<32x196x512xf32>
    %v1429 = stablehlo.multiply %v1428, %v1428 : tensor<32x196x512xf32>
    %v1430 = stablehlo.reduce(%v1429 init: %v1422) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1431 = stablehlo.broadcast_in_dim %v1430, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1432 = stablehlo.divide %v1431, %v1423 : tensor<32x196x512xf32>
    %v1433 = stablehlo.add %v1432, %v1424 : tensor<32x196x512xf32>
    %v1434 = stablehlo.rsqrt %v1433 : tensor<32x196x512xf32>
    %v1435 = stablehlo.multiply %v1428, %v1434 : tensor<32x196x512xf32>
    %v1436 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1437 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1438 = stablehlo.multiply %v1435, %v1436 : tensor<32x196x512xf32>
    %v1439 = stablehlo.add %v1438, %v1437 : tensor<32x196x512xf32>
    %v1440 = stablehlo.reshape %v1439 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1441 = stablehlo.reshape %v1440 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1442 = stablehlo.broadcast_in_dim %s2b12ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1443 = stablehlo.multiply %v1441, %v1442 : tensor<32x196x512xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1446 = stablehlo.broadcast_in_dim %s2b12nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1447 = stablehlo.add %v1445, %v1446 : tensor<32x196x512xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1449 = stablehlo.reshape %v1448 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1450 = stablehlo.transpose %v1449, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1453 = stablehlo.convolution(%v1452, %s2b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1454 = stablehlo.broadcast_in_dim %s2b12eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1455 = stablehlo.add %v1453, %v1454 : tensor<32x2048x14x14xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1458 = stablehlo.multiply %v1457, %v1457 : tensor<32x2048x14x14xf32>
    %v1459 = stablehlo.multiply %v1458, %v1457 : tensor<32x2048x14x14xf32>
    %v1460 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1461 = stablehlo.multiply %v1460, %v1459 : tensor<32x2048x14x14xf32>
    %v1462 = stablehlo.add %v1457, %v1461 : tensor<32x2048x14x14xf32>
    %v1463 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1464 = stablehlo.multiply %v1463, %v1462 : tensor<32x2048x14x14xf32>
    %v1465 = stablehlo.tanh %v1464 : tensor<32x2048x14x14xf32>
    %v1466 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1467 = stablehlo.add %v1466, %v1465 : tensor<32x2048x14x14xf32>
    %v1468 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1469 = stablehlo.multiply %v1468, %v1457 : tensor<32x2048x14x14xf32>
    %v1470 = stablehlo.multiply %v1469, %v1467 : tensor<32x2048x14x14xf32>
    %v1471 = stablehlo.reshape %v1470 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1473 = stablehlo.convolution(%v1472, %s2b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1474 = stablehlo.broadcast_in_dim %s2b12pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1475 = stablehlo.add %v1473, %v1474 : tensor<32x512x14x14xf32>
    %v1476 = stablehlo.reshape %v1475 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1477 = stablehlo.reshape %v1476 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1478 = stablehlo.broadcast_in_dim %s2b12lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1479 = stablehlo.multiply %v1477, %v1478 : tensor<32x512x14x14xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1481 = stablehlo.reshape %v1480 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1482 = stablehlo.reshape %v1412 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1483 = stablehlo.add %v1481, %v1482 : tensor<32x512x14x14xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1485 = stablehlo.reshape %v1484 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1486 = stablehlo.convolution(%v1485, %s2b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1487 = stablehlo.broadcast_in_dim %s2b13db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1488 = stablehlo.add %v1486, %v1487 : tensor<32x512x14x14xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1491 = stablehlo.transpose %v1490, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1495 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1496 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1497 = stablehlo.reduce(%v1493 init: %v1494) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1498 = stablehlo.broadcast_in_dim %v1497, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1499 = stablehlo.divide %v1498, %v1495 : tensor<32x196x512xf32>
    %v1500 = stablehlo.subtract %v1493, %v1499 : tensor<32x196x512xf32>
    %v1501 = stablehlo.multiply %v1500, %v1500 : tensor<32x196x512xf32>
    %v1502 = stablehlo.reduce(%v1501 init: %v1494) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1503 = stablehlo.broadcast_in_dim %v1502, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1504 = stablehlo.divide %v1503, %v1495 : tensor<32x196x512xf32>
    %v1505 = stablehlo.add %v1504, %v1496 : tensor<32x196x512xf32>
    %v1506 = stablehlo.rsqrt %v1505 : tensor<32x196x512xf32>
    %v1507 = stablehlo.multiply %v1500, %v1506 : tensor<32x196x512xf32>
    %v1508 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1509 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1510 = stablehlo.multiply %v1507, %v1508 : tensor<32x196x512xf32>
    %v1511 = stablehlo.add %v1510, %v1509 : tensor<32x196x512xf32>
    %v1512 = stablehlo.reshape %v1511 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1513 = stablehlo.reshape %v1512 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1514 = stablehlo.broadcast_in_dim %s2b13ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1515 = stablehlo.multiply %v1513, %v1514 : tensor<32x196x512xf32>
    %v1516 = stablehlo.reshape %v1515 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1517 = stablehlo.reshape %v1516 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1518 = stablehlo.broadcast_in_dim %s2b13nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1519 = stablehlo.add %v1517, %v1518 : tensor<32x196x512xf32>
    %v1520 = stablehlo.reshape %v1519 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1522 = stablehlo.transpose %v1521, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1525 = stablehlo.convolution(%v1524, %s2b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1526 = stablehlo.broadcast_in_dim %s2b13eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1527 = stablehlo.add %v1525, %v1526 : tensor<32x2048x14x14xf32>
    %v1528 = stablehlo.reshape %v1527 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1530 = stablehlo.multiply %v1529, %v1529 : tensor<32x2048x14x14xf32>
    %v1531 = stablehlo.multiply %v1530, %v1529 : tensor<32x2048x14x14xf32>
    %v1532 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1533 = stablehlo.multiply %v1532, %v1531 : tensor<32x2048x14x14xf32>
    %v1534 = stablehlo.add %v1529, %v1533 : tensor<32x2048x14x14xf32>
    %v1535 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1536 = stablehlo.multiply %v1535, %v1534 : tensor<32x2048x14x14xf32>
    %v1537 = stablehlo.tanh %v1536 : tensor<32x2048x14x14xf32>
    %v1538 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1539 = stablehlo.add %v1538, %v1537 : tensor<32x2048x14x14xf32>
    %v1540 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1541 = stablehlo.multiply %v1540, %v1529 : tensor<32x2048x14x14xf32>
    %v1542 = stablehlo.multiply %v1541, %v1539 : tensor<32x2048x14x14xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1545 = stablehlo.convolution(%v1544, %s2b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1546 = stablehlo.broadcast_in_dim %s2b13pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1547 = stablehlo.add %v1545, %v1546 : tensor<32x512x14x14xf32>
    %v1548 = stablehlo.reshape %v1547 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1550 = stablehlo.broadcast_in_dim %s2b13lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1551 = stablehlo.multiply %v1549, %v1550 : tensor<32x512x14x14xf32>
    %v1552 = stablehlo.reshape %v1551 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1553 = stablehlo.reshape %v1552 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1554 = stablehlo.reshape %v1484 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1555 = stablehlo.add %v1553, %v1554 : tensor<32x512x14x14xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1557 = stablehlo.reshape %v1556 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1558 = stablehlo.convolution(%v1557, %s2b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1559 = stablehlo.broadcast_in_dim %s2b14db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1560 = stablehlo.add %v1558, %v1559 : tensor<32x512x14x14xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1562 = stablehlo.reshape %v1561 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1563 = stablehlo.transpose %v1562, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1564 = stablehlo.reshape %v1563 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1567 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1568 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1569 = stablehlo.reduce(%v1565 init: %v1566) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1570 = stablehlo.broadcast_in_dim %v1569, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1571 = stablehlo.divide %v1570, %v1567 : tensor<32x196x512xf32>
    %v1572 = stablehlo.subtract %v1565, %v1571 : tensor<32x196x512xf32>
    %v1573 = stablehlo.multiply %v1572, %v1572 : tensor<32x196x512xf32>
    %v1574 = stablehlo.reduce(%v1573 init: %v1566) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1575 = stablehlo.broadcast_in_dim %v1574, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1576 = stablehlo.divide %v1575, %v1567 : tensor<32x196x512xf32>
    %v1577 = stablehlo.add %v1576, %v1568 : tensor<32x196x512xf32>
    %v1578 = stablehlo.rsqrt %v1577 : tensor<32x196x512xf32>
    %v1579 = stablehlo.multiply %v1572, %v1578 : tensor<32x196x512xf32>
    %v1580 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1581 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1582 = stablehlo.multiply %v1579, %v1580 : tensor<32x196x512xf32>
    %v1583 = stablehlo.add %v1582, %v1581 : tensor<32x196x512xf32>
    %v1584 = stablehlo.reshape %v1583 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1586 = stablehlo.broadcast_in_dim %s2b14ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1587 = stablehlo.multiply %v1585, %v1586 : tensor<32x196x512xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1590 = stablehlo.broadcast_in_dim %s2b14nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1591 = stablehlo.add %v1589, %v1590 : tensor<32x196x512xf32>
    %v1592 = stablehlo.reshape %v1591 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1593 = stablehlo.reshape %v1592 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1594 = stablehlo.transpose %v1593, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1595 = stablehlo.reshape %v1594 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1596 = stablehlo.reshape %v1595 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1597 = stablehlo.convolution(%v1596, %s2b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1598 = stablehlo.broadcast_in_dim %s2b14eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1599 = stablehlo.add %v1597, %v1598 : tensor<32x2048x14x14xf32>
    %v1600 = stablehlo.reshape %v1599 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1601 = stablehlo.reshape %v1600 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1602 = stablehlo.multiply %v1601, %v1601 : tensor<32x2048x14x14xf32>
    %v1603 = stablehlo.multiply %v1602, %v1601 : tensor<32x2048x14x14xf32>
    %v1604 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1605 = stablehlo.multiply %v1604, %v1603 : tensor<32x2048x14x14xf32>
    %v1606 = stablehlo.add %v1601, %v1605 : tensor<32x2048x14x14xf32>
    %v1607 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1608 = stablehlo.multiply %v1607, %v1606 : tensor<32x2048x14x14xf32>
    %v1609 = stablehlo.tanh %v1608 : tensor<32x2048x14x14xf32>
    %v1610 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1611 = stablehlo.add %v1610, %v1609 : tensor<32x2048x14x14xf32>
    %v1612 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1613 = stablehlo.multiply %v1612, %v1601 : tensor<32x2048x14x14xf32>
    %v1614 = stablehlo.multiply %v1613, %v1611 : tensor<32x2048x14x14xf32>
    %v1615 = stablehlo.reshape %v1614 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1616 = stablehlo.reshape %v1615 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1617 = stablehlo.convolution(%v1616, %s2b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1618 = stablehlo.broadcast_in_dim %s2b14pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1619 = stablehlo.add %v1617, %v1618 : tensor<32x512x14x14xf32>
    %v1620 = stablehlo.reshape %v1619 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1621 = stablehlo.reshape %v1620 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1622 = stablehlo.broadcast_in_dim %s2b14lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1623 = stablehlo.multiply %v1621, %v1622 : tensor<32x512x14x14xf32>
    %v1624 = stablehlo.reshape %v1623 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1625 = stablehlo.reshape %v1624 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1626 = stablehlo.reshape %v1556 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1627 = stablehlo.add %v1625, %v1626 : tensor<32x512x14x14xf32>
    %v1628 = stablehlo.reshape %v1627 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1629 = stablehlo.reshape %v1628 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1630 = stablehlo.convolution(%v1629, %s2b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1631 = stablehlo.broadcast_in_dim %s2b15db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1632 = stablehlo.add %v1630, %v1631 : tensor<32x512x14x14xf32>
    %v1633 = stablehlo.reshape %v1632 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1635 = stablehlo.transpose %v1634, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1636 = stablehlo.reshape %v1635 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1637 = stablehlo.reshape %v1636 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1639 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1640 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1641 = stablehlo.reduce(%v1637 init: %v1638) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1642 = stablehlo.broadcast_in_dim %v1641, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1643 = stablehlo.divide %v1642, %v1639 : tensor<32x196x512xf32>
    %v1644 = stablehlo.subtract %v1637, %v1643 : tensor<32x196x512xf32>
    %v1645 = stablehlo.multiply %v1644, %v1644 : tensor<32x196x512xf32>
    %v1646 = stablehlo.reduce(%v1645 init: %v1638) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1647 = stablehlo.broadcast_in_dim %v1646, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1648 = stablehlo.divide %v1647, %v1639 : tensor<32x196x512xf32>
    %v1649 = stablehlo.add %v1648, %v1640 : tensor<32x196x512xf32>
    %v1650 = stablehlo.rsqrt %v1649 : tensor<32x196x512xf32>
    %v1651 = stablehlo.multiply %v1644, %v1650 : tensor<32x196x512xf32>
    %v1652 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1653 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1654 = stablehlo.multiply %v1651, %v1652 : tensor<32x196x512xf32>
    %v1655 = stablehlo.add %v1654, %v1653 : tensor<32x196x512xf32>
    %v1656 = stablehlo.reshape %v1655 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1657 = stablehlo.reshape %v1656 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1658 = stablehlo.broadcast_in_dim %s2b15ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1659 = stablehlo.multiply %v1657, %v1658 : tensor<32x196x512xf32>
    %v1660 = stablehlo.reshape %v1659 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1661 = stablehlo.reshape %v1660 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1662 = stablehlo.broadcast_in_dim %s2b15nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1663 = stablehlo.add %v1661, %v1662 : tensor<32x196x512xf32>
    %v1664 = stablehlo.reshape %v1663 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1666 = stablehlo.transpose %v1665, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1667 = stablehlo.reshape %v1666 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1668 = stablehlo.reshape %v1667 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1669 = stablehlo.convolution(%v1668, %s2b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1670 = stablehlo.broadcast_in_dim %s2b15eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1671 = stablehlo.add %v1669, %v1670 : tensor<32x2048x14x14xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1674 = stablehlo.multiply %v1673, %v1673 : tensor<32x2048x14x14xf32>
    %v1675 = stablehlo.multiply %v1674, %v1673 : tensor<32x2048x14x14xf32>
    %v1676 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1677 = stablehlo.multiply %v1676, %v1675 : tensor<32x2048x14x14xf32>
    %v1678 = stablehlo.add %v1673, %v1677 : tensor<32x2048x14x14xf32>
    %v1679 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1680 = stablehlo.multiply %v1679, %v1678 : tensor<32x2048x14x14xf32>
    %v1681 = stablehlo.tanh %v1680 : tensor<32x2048x14x14xf32>
    %v1682 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1683 = stablehlo.add %v1682, %v1681 : tensor<32x2048x14x14xf32>
    %v1684 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1685 = stablehlo.multiply %v1684, %v1673 : tensor<32x2048x14x14xf32>
    %v1686 = stablehlo.multiply %v1685, %v1683 : tensor<32x2048x14x14xf32>
    %v1687 = stablehlo.reshape %v1686 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1688 = stablehlo.reshape %v1687 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1689 = stablehlo.convolution(%v1688, %s2b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1690 = stablehlo.broadcast_in_dim %s2b15pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1691 = stablehlo.add %v1689, %v1690 : tensor<32x512x14x14xf32>
    %v1692 = stablehlo.reshape %v1691 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1694 = stablehlo.broadcast_in_dim %s2b15lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1695 = stablehlo.multiply %v1693, %v1694 : tensor<32x512x14x14xf32>
    %v1696 = stablehlo.reshape %v1695 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1698 = stablehlo.reshape %v1628 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1699 = stablehlo.add %v1697, %v1698 : tensor<32x512x14x14xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1702 = stablehlo.convolution(%v1701, %s2b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1703 = stablehlo.broadcast_in_dim %s2b16db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1704 = stablehlo.add %v1702, %v1703 : tensor<32x512x14x14xf32>
    %v1705 = stablehlo.reshape %v1704 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1706 = stablehlo.reshape %v1705 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1707 = stablehlo.transpose %v1706, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1708 = stablehlo.reshape %v1707 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1709 = stablehlo.reshape %v1708 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1710 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1711 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1712 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1713 = stablehlo.reduce(%v1709 init: %v1710) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1714 = stablehlo.broadcast_in_dim %v1713, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1715 = stablehlo.divide %v1714, %v1711 : tensor<32x196x512xf32>
    %v1716 = stablehlo.subtract %v1709, %v1715 : tensor<32x196x512xf32>
    %v1717 = stablehlo.multiply %v1716, %v1716 : tensor<32x196x512xf32>
    %v1718 = stablehlo.reduce(%v1717 init: %v1710) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1719 = stablehlo.broadcast_in_dim %v1718, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1720 = stablehlo.divide %v1719, %v1711 : tensor<32x196x512xf32>
    %v1721 = stablehlo.add %v1720, %v1712 : tensor<32x196x512xf32>
    %v1722 = stablehlo.rsqrt %v1721 : tensor<32x196x512xf32>
    %v1723 = stablehlo.multiply %v1716, %v1722 : tensor<32x196x512xf32>
    %v1724 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1725 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1726 = stablehlo.multiply %v1723, %v1724 : tensor<32x196x512xf32>
    %v1727 = stablehlo.add %v1726, %v1725 : tensor<32x196x512xf32>
    %v1728 = stablehlo.reshape %v1727 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1729 = stablehlo.reshape %v1728 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1730 = stablehlo.broadcast_in_dim %s2b16ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1731 = stablehlo.multiply %v1729, %v1730 : tensor<32x196x512xf32>
    %v1732 = stablehlo.reshape %v1731 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1733 = stablehlo.reshape %v1732 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1734 = stablehlo.broadcast_in_dim %s2b16nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1735 = stablehlo.add %v1733, %v1734 : tensor<32x196x512xf32>
    %v1736 = stablehlo.reshape %v1735 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1737 = stablehlo.reshape %v1736 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1738 = stablehlo.transpose %v1737, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1739 = stablehlo.reshape %v1738 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1740 = stablehlo.reshape %v1739 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1741 = stablehlo.convolution(%v1740, %s2b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1742 = stablehlo.broadcast_in_dim %s2b16eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1743 = stablehlo.add %v1741, %v1742 : tensor<32x2048x14x14xf32>
    %v1744 = stablehlo.reshape %v1743 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1745 = stablehlo.reshape %v1744 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1746 = stablehlo.multiply %v1745, %v1745 : tensor<32x2048x14x14xf32>
    %v1747 = stablehlo.multiply %v1746, %v1745 : tensor<32x2048x14x14xf32>
    %v1748 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1749 = stablehlo.multiply %v1748, %v1747 : tensor<32x2048x14x14xf32>
    %v1750 = stablehlo.add %v1745, %v1749 : tensor<32x2048x14x14xf32>
    %v1751 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1752 = stablehlo.multiply %v1751, %v1750 : tensor<32x2048x14x14xf32>
    %v1753 = stablehlo.tanh %v1752 : tensor<32x2048x14x14xf32>
    %v1754 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1755 = stablehlo.add %v1754, %v1753 : tensor<32x2048x14x14xf32>
    %v1756 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1757 = stablehlo.multiply %v1756, %v1745 : tensor<32x2048x14x14xf32>
    %v1758 = stablehlo.multiply %v1757, %v1755 : tensor<32x2048x14x14xf32>
    %v1759 = stablehlo.reshape %v1758 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1760 = stablehlo.reshape %v1759 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1761 = stablehlo.convolution(%v1760, %s2b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1762 = stablehlo.broadcast_in_dim %s2b16pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1763 = stablehlo.add %v1761, %v1762 : tensor<32x512x14x14xf32>
    %v1764 = stablehlo.reshape %v1763 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1766 = stablehlo.broadcast_in_dim %s2b16lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1767 = stablehlo.multiply %v1765, %v1766 : tensor<32x512x14x14xf32>
    %v1768 = stablehlo.reshape %v1767 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1769 = stablehlo.reshape %v1768 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1770 = stablehlo.reshape %v1700 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1771 = stablehlo.add %v1769, %v1770 : tensor<32x512x14x14xf32>
    %v1772 = stablehlo.reshape %v1771 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1773 = stablehlo.reshape %v1772 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1774 = stablehlo.convolution(%v1773, %s2b17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1775 = stablehlo.broadcast_in_dim %s2b17db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1776 = stablehlo.add %v1774, %v1775 : tensor<32x512x14x14xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1778 = stablehlo.reshape %v1777 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1779 = stablehlo.transpose %v1778, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1783 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1784 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1785 = stablehlo.reduce(%v1781 init: %v1782) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1786 = stablehlo.broadcast_in_dim %v1785, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1787 = stablehlo.divide %v1786, %v1783 : tensor<32x196x512xf32>
    %v1788 = stablehlo.subtract %v1781, %v1787 : tensor<32x196x512xf32>
    %v1789 = stablehlo.multiply %v1788, %v1788 : tensor<32x196x512xf32>
    %v1790 = stablehlo.reduce(%v1789 init: %v1782) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1791 = stablehlo.broadcast_in_dim %v1790, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1792 = stablehlo.divide %v1791, %v1783 : tensor<32x196x512xf32>
    %v1793 = stablehlo.add %v1792, %v1784 : tensor<32x196x512xf32>
    %v1794 = stablehlo.rsqrt %v1793 : tensor<32x196x512xf32>
    %v1795 = stablehlo.multiply %v1788, %v1794 : tensor<32x196x512xf32>
    %v1796 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1797 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1798 = stablehlo.multiply %v1795, %v1796 : tensor<32x196x512xf32>
    %v1799 = stablehlo.add %v1798, %v1797 : tensor<32x196x512xf32>
    %v1800 = stablehlo.reshape %v1799 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1801 = stablehlo.reshape %v1800 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1802 = stablehlo.broadcast_in_dim %s2b17ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1803 = stablehlo.multiply %v1801, %v1802 : tensor<32x196x512xf32>
    %v1804 = stablehlo.reshape %v1803 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1805 = stablehlo.reshape %v1804 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1806 = stablehlo.broadcast_in_dim %s2b17nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1807 = stablehlo.add %v1805, %v1806 : tensor<32x196x512xf32>
    %v1808 = stablehlo.reshape %v1807 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1809 = stablehlo.reshape %v1808 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1810 = stablehlo.transpose %v1809, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1812 = stablehlo.reshape %v1811 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1813 = stablehlo.convolution(%v1812, %s2b17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1814 = stablehlo.broadcast_in_dim %s2b17eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1815 = stablehlo.add %v1813, %v1814 : tensor<32x2048x14x14xf32>
    %v1816 = stablehlo.reshape %v1815 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1817 = stablehlo.reshape %v1816 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1818 = stablehlo.multiply %v1817, %v1817 : tensor<32x2048x14x14xf32>
    %v1819 = stablehlo.multiply %v1818, %v1817 : tensor<32x2048x14x14xf32>
    %v1820 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1821 = stablehlo.multiply %v1820, %v1819 : tensor<32x2048x14x14xf32>
    %v1822 = stablehlo.add %v1817, %v1821 : tensor<32x2048x14x14xf32>
    %v1823 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1824 = stablehlo.multiply %v1823, %v1822 : tensor<32x2048x14x14xf32>
    %v1825 = stablehlo.tanh %v1824 : tensor<32x2048x14x14xf32>
    %v1826 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1827 = stablehlo.add %v1826, %v1825 : tensor<32x2048x14x14xf32>
    %v1828 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1829 = stablehlo.multiply %v1828, %v1817 : tensor<32x2048x14x14xf32>
    %v1830 = stablehlo.multiply %v1829, %v1827 : tensor<32x2048x14x14xf32>
    %v1831 = stablehlo.reshape %v1830 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1832 = stablehlo.reshape %v1831 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1833 = stablehlo.convolution(%v1832, %s2b17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1834 = stablehlo.broadcast_in_dim %s2b17pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1835 = stablehlo.add %v1833, %v1834 : tensor<32x512x14x14xf32>
    %v1836 = stablehlo.reshape %v1835 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1837 = stablehlo.reshape %v1836 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1838 = stablehlo.broadcast_in_dim %s2b17lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1839 = stablehlo.multiply %v1837, %v1838 : tensor<32x512x14x14xf32>
    %v1840 = stablehlo.reshape %v1839 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1841 = stablehlo.reshape %v1840 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1842 = stablehlo.reshape %v1772 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1843 = stablehlo.add %v1841, %v1842 : tensor<32x512x14x14xf32>
    %v1844 = stablehlo.reshape %v1843 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1845 = stablehlo.reshape %v1844 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1846 = stablehlo.convolution(%v1845, %s2b18dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1847 = stablehlo.broadcast_in_dim %s2b18db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1848 = stablehlo.add %v1846, %v1847 : tensor<32x512x14x14xf32>
    %v1849 = stablehlo.reshape %v1848 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1850 = stablehlo.reshape %v1849 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1851 = stablehlo.transpose %v1850, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1852 = stablehlo.reshape %v1851 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1855 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1856 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1857 = stablehlo.reduce(%v1853 init: %v1854) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1858 = stablehlo.broadcast_in_dim %v1857, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1859 = stablehlo.divide %v1858, %v1855 : tensor<32x196x512xf32>
    %v1860 = stablehlo.subtract %v1853, %v1859 : tensor<32x196x512xf32>
    %v1861 = stablehlo.multiply %v1860, %v1860 : tensor<32x196x512xf32>
    %v1862 = stablehlo.reduce(%v1861 init: %v1854) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1863 = stablehlo.broadcast_in_dim %v1862, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1864 = stablehlo.divide %v1863, %v1855 : tensor<32x196x512xf32>
    %v1865 = stablehlo.add %v1864, %v1856 : tensor<32x196x512xf32>
    %v1866 = stablehlo.rsqrt %v1865 : tensor<32x196x512xf32>
    %v1867 = stablehlo.multiply %v1860, %v1866 : tensor<32x196x512xf32>
    %v1868 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1869 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1870 = stablehlo.multiply %v1867, %v1868 : tensor<32x196x512xf32>
    %v1871 = stablehlo.add %v1870, %v1869 : tensor<32x196x512xf32>
    %v1872 = stablehlo.reshape %v1871 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1873 = stablehlo.reshape %v1872 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1874 = stablehlo.broadcast_in_dim %s2b18ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1875 = stablehlo.multiply %v1873, %v1874 : tensor<32x196x512xf32>
    %v1876 = stablehlo.reshape %v1875 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1877 = stablehlo.reshape %v1876 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1878 = stablehlo.broadcast_in_dim %s2b18nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1879 = stablehlo.add %v1877, %v1878 : tensor<32x196x512xf32>
    %v1880 = stablehlo.reshape %v1879 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1881 = stablehlo.reshape %v1880 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1882 = stablehlo.transpose %v1881, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1883 = stablehlo.reshape %v1882 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1884 = stablehlo.reshape %v1883 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1885 = stablehlo.convolution(%v1884, %s2b18eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1886 = stablehlo.broadcast_in_dim %s2b18eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1887 = stablehlo.add %v1885, %v1886 : tensor<32x2048x14x14xf32>
    %v1888 = stablehlo.reshape %v1887 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1889 = stablehlo.reshape %v1888 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1890 = stablehlo.multiply %v1889, %v1889 : tensor<32x2048x14x14xf32>
    %v1891 = stablehlo.multiply %v1890, %v1889 : tensor<32x2048x14x14xf32>
    %v1892 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1893 = stablehlo.multiply %v1892, %v1891 : tensor<32x2048x14x14xf32>
    %v1894 = stablehlo.add %v1889, %v1893 : tensor<32x2048x14x14xf32>
    %v1895 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1896 = stablehlo.multiply %v1895, %v1894 : tensor<32x2048x14x14xf32>
    %v1897 = stablehlo.tanh %v1896 : tensor<32x2048x14x14xf32>
    %v1898 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1899 = stablehlo.add %v1898, %v1897 : tensor<32x2048x14x14xf32>
    %v1900 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1901 = stablehlo.multiply %v1900, %v1889 : tensor<32x2048x14x14xf32>
    %v1902 = stablehlo.multiply %v1901, %v1899 : tensor<32x2048x14x14xf32>
    %v1903 = stablehlo.reshape %v1902 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1905 = stablehlo.convolution(%v1904, %s2b18pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1906 = stablehlo.broadcast_in_dim %s2b18pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1907 = stablehlo.add %v1905, %v1906 : tensor<32x512x14x14xf32>
    %v1908 = stablehlo.reshape %v1907 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1909 = stablehlo.reshape %v1908 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1910 = stablehlo.broadcast_in_dim %s2b18lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1911 = stablehlo.multiply %v1909, %v1910 : tensor<32x512x14x14xf32>
    %v1912 = stablehlo.reshape %v1911 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1914 = stablehlo.reshape %v1844 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1915 = stablehlo.add %v1913, %v1914 : tensor<32x512x14x14xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1918 = stablehlo.convolution(%v1917, %s2b19dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1919 = stablehlo.broadcast_in_dim %s2b19db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1920 = stablehlo.add %v1918, %v1919 : tensor<32x512x14x14xf32>
    %v1921 = stablehlo.reshape %v1920 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1922 = stablehlo.reshape %v1921 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1923 = stablehlo.transpose %v1922, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1925 = stablehlo.reshape %v1924 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1927 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1928 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1929 = stablehlo.reduce(%v1925 init: %v1926) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1930 = stablehlo.broadcast_in_dim %v1929, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1931 = stablehlo.divide %v1930, %v1927 : tensor<32x196x512xf32>
    %v1932 = stablehlo.subtract %v1925, %v1931 : tensor<32x196x512xf32>
    %v1933 = stablehlo.multiply %v1932, %v1932 : tensor<32x196x512xf32>
    %v1934 = stablehlo.reduce(%v1933 init: %v1926) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1935 = stablehlo.broadcast_in_dim %v1934, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1936 = stablehlo.divide %v1935, %v1927 : tensor<32x196x512xf32>
    %v1937 = stablehlo.add %v1936, %v1928 : tensor<32x196x512xf32>
    %v1938 = stablehlo.rsqrt %v1937 : tensor<32x196x512xf32>
    %v1939 = stablehlo.multiply %v1932, %v1938 : tensor<32x196x512xf32>
    %v1940 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1941 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1942 = stablehlo.multiply %v1939, %v1940 : tensor<32x196x512xf32>
    %v1943 = stablehlo.add %v1942, %v1941 : tensor<32x196x512xf32>
    %v1944 = stablehlo.reshape %v1943 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1946 = stablehlo.broadcast_in_dim %s2b19ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1947 = stablehlo.multiply %v1945, %v1946 : tensor<32x196x512xf32>
    %v1948 = stablehlo.reshape %v1947 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1950 = stablehlo.broadcast_in_dim %s2b19nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1951 = stablehlo.add %v1949, %v1950 : tensor<32x196x512xf32>
    %v1952 = stablehlo.reshape %v1951 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1953 = stablehlo.reshape %v1952 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1954 = stablehlo.transpose %v1953, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1955 = stablehlo.reshape %v1954 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1956 = stablehlo.reshape %v1955 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1957 = stablehlo.convolution(%v1956, %s2b19eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1958 = stablehlo.broadcast_in_dim %s2b19eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1959 = stablehlo.add %v1957, %v1958 : tensor<32x2048x14x14xf32>
    %v1960 = stablehlo.reshape %v1959 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1961 = stablehlo.reshape %v1960 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1962 = stablehlo.multiply %v1961, %v1961 : tensor<32x2048x14x14xf32>
    %v1963 = stablehlo.multiply %v1962, %v1961 : tensor<32x2048x14x14xf32>
    %v1964 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v1965 = stablehlo.multiply %v1964, %v1963 : tensor<32x2048x14x14xf32>
    %v1966 = stablehlo.add %v1961, %v1965 : tensor<32x2048x14x14xf32>
    %v1967 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v1968 = stablehlo.multiply %v1967, %v1966 : tensor<32x2048x14x14xf32>
    %v1969 = stablehlo.tanh %v1968 : tensor<32x2048x14x14xf32>
    %v1970 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v1971 = stablehlo.add %v1970, %v1969 : tensor<32x2048x14x14xf32>
    %v1972 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v1973 = stablehlo.multiply %v1972, %v1961 : tensor<32x2048x14x14xf32>
    %v1974 = stablehlo.multiply %v1973, %v1971 : tensor<32x2048x14x14xf32>
    %v1975 = stablehlo.reshape %v1974 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1976 = stablehlo.reshape %v1975 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1977 = stablehlo.convolution(%v1976, %s2b19pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1978 = stablehlo.broadcast_in_dim %s2b19pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1979 = stablehlo.add %v1977, %v1978 : tensor<32x512x14x14xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1981 = stablehlo.reshape %v1980 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1982 = stablehlo.broadcast_in_dim %s2b19lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1983 = stablehlo.multiply %v1981, %v1982 : tensor<32x512x14x14xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1985 = stablehlo.reshape %v1984 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1986 = stablehlo.reshape %v1916 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1987 = stablehlo.add %v1985, %v1986 : tensor<32x512x14x14xf32>
    %v1988 = stablehlo.reshape %v1987 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1989 = stablehlo.reshape %v1988 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1990 = stablehlo.convolution(%v1989, %s2b20dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1991 = stablehlo.broadcast_in_dim %s2b20db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1992 = stablehlo.add %v1990, %v1991 : tensor<32x512x14x14xf32>
    %v1993 = stablehlo.reshape %v1992 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1994 = stablehlo.reshape %v1993 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1995 = stablehlo.transpose %v1994, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1996 = stablehlo.reshape %v1995 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1997 = stablehlo.reshape %v1996 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1999 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2000 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2001 = stablehlo.reduce(%v1997 init: %v1998) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2002 = stablehlo.broadcast_in_dim %v2001, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2003 = stablehlo.divide %v2002, %v1999 : tensor<32x196x512xf32>
    %v2004 = stablehlo.subtract %v1997, %v2003 : tensor<32x196x512xf32>
    %v2005 = stablehlo.multiply %v2004, %v2004 : tensor<32x196x512xf32>
    %v2006 = stablehlo.reduce(%v2005 init: %v1998) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2007 = stablehlo.broadcast_in_dim %v2006, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2008 = stablehlo.divide %v2007, %v1999 : tensor<32x196x512xf32>
    %v2009 = stablehlo.add %v2008, %v2000 : tensor<32x196x512xf32>
    %v2010 = stablehlo.rsqrt %v2009 : tensor<32x196x512xf32>
    %v2011 = stablehlo.multiply %v2004, %v2010 : tensor<32x196x512xf32>
    %v2012 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2013 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2014 = stablehlo.multiply %v2011, %v2012 : tensor<32x196x512xf32>
    %v2015 = stablehlo.add %v2014, %v2013 : tensor<32x196x512xf32>
    %v2016 = stablehlo.reshape %v2015 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2017 = stablehlo.reshape %v2016 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2018 = stablehlo.broadcast_in_dim %s2b20ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2019 = stablehlo.multiply %v2017, %v2018 : tensor<32x196x512xf32>
    %v2020 = stablehlo.reshape %v2019 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2021 = stablehlo.reshape %v2020 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2022 = stablehlo.broadcast_in_dim %s2b20nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2023 = stablehlo.add %v2021, %v2022 : tensor<32x196x512xf32>
    %v2024 = stablehlo.reshape %v2023 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2025 = stablehlo.reshape %v2024 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2026 = stablehlo.transpose %v2025, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2027 = stablehlo.reshape %v2026 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2028 = stablehlo.reshape %v2027 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2029 = stablehlo.convolution(%v2028, %s2b20eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2030 = stablehlo.broadcast_in_dim %s2b20eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2031 = stablehlo.add %v2029, %v2030 : tensor<32x2048x14x14xf32>
    %v2032 = stablehlo.reshape %v2031 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2033 = stablehlo.reshape %v2032 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2034 = stablehlo.multiply %v2033, %v2033 : tensor<32x2048x14x14xf32>
    %v2035 = stablehlo.multiply %v2034, %v2033 : tensor<32x2048x14x14xf32>
    %v2036 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v2037 = stablehlo.multiply %v2036, %v2035 : tensor<32x2048x14x14xf32>
    %v2038 = stablehlo.add %v2033, %v2037 : tensor<32x2048x14x14xf32>
    %v2039 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v2040 = stablehlo.multiply %v2039, %v2038 : tensor<32x2048x14x14xf32>
    %v2041 = stablehlo.tanh %v2040 : tensor<32x2048x14x14xf32>
    %v2042 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v2043 = stablehlo.add %v2042, %v2041 : tensor<32x2048x14x14xf32>
    %v2044 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v2045 = stablehlo.multiply %v2044, %v2033 : tensor<32x2048x14x14xf32>
    %v2046 = stablehlo.multiply %v2045, %v2043 : tensor<32x2048x14x14xf32>
    %v2047 = stablehlo.reshape %v2046 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2048 = stablehlo.reshape %v2047 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2049 = stablehlo.convolution(%v2048, %s2b20pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2050 = stablehlo.broadcast_in_dim %s2b20pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2051 = stablehlo.add %v2049, %v2050 : tensor<32x512x14x14xf32>
    %v2052 = stablehlo.reshape %v2051 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2053 = stablehlo.reshape %v2052 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2054 = stablehlo.broadcast_in_dim %s2b20lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2055 = stablehlo.multiply %v2053, %v2054 : tensor<32x512x14x14xf32>
    %v2056 = stablehlo.reshape %v2055 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2057 = stablehlo.reshape %v2056 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2058 = stablehlo.reshape %v1988 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2059 = stablehlo.add %v2057, %v2058 : tensor<32x512x14x14xf32>
    %v2060 = stablehlo.reshape %v2059 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2061 = stablehlo.reshape %v2060 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2062 = stablehlo.convolution(%v2061, %s2b21dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2063 = stablehlo.broadcast_in_dim %s2b21db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2064 = stablehlo.add %v2062, %v2063 : tensor<32x512x14x14xf32>
    %v2065 = stablehlo.reshape %v2064 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2066 = stablehlo.reshape %v2065 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2067 = stablehlo.transpose %v2066, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2068 = stablehlo.reshape %v2067 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2069 = stablehlo.reshape %v2068 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2071 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2072 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2073 = stablehlo.reduce(%v2069 init: %v2070) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2074 = stablehlo.broadcast_in_dim %v2073, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2075 = stablehlo.divide %v2074, %v2071 : tensor<32x196x512xf32>
    %v2076 = stablehlo.subtract %v2069, %v2075 : tensor<32x196x512xf32>
    %v2077 = stablehlo.multiply %v2076, %v2076 : tensor<32x196x512xf32>
    %v2078 = stablehlo.reduce(%v2077 init: %v2070) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2079 = stablehlo.broadcast_in_dim %v2078, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2080 = stablehlo.divide %v2079, %v2071 : tensor<32x196x512xf32>
    %v2081 = stablehlo.add %v2080, %v2072 : tensor<32x196x512xf32>
    %v2082 = stablehlo.rsqrt %v2081 : tensor<32x196x512xf32>
    %v2083 = stablehlo.multiply %v2076, %v2082 : tensor<32x196x512xf32>
    %v2084 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2085 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2086 = stablehlo.multiply %v2083, %v2084 : tensor<32x196x512xf32>
    %v2087 = stablehlo.add %v2086, %v2085 : tensor<32x196x512xf32>
    %v2088 = stablehlo.reshape %v2087 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2089 = stablehlo.reshape %v2088 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2090 = stablehlo.broadcast_in_dim %s2b21ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2091 = stablehlo.multiply %v2089, %v2090 : tensor<32x196x512xf32>
    %v2092 = stablehlo.reshape %v2091 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2093 = stablehlo.reshape %v2092 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2094 = stablehlo.broadcast_in_dim %s2b21nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2095 = stablehlo.add %v2093, %v2094 : tensor<32x196x512xf32>
    %v2096 = stablehlo.reshape %v2095 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2097 = stablehlo.reshape %v2096 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2098 = stablehlo.transpose %v2097, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2099 = stablehlo.reshape %v2098 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2100 = stablehlo.reshape %v2099 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2101 = stablehlo.convolution(%v2100, %s2b21eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2102 = stablehlo.broadcast_in_dim %s2b21eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2103 = stablehlo.add %v2101, %v2102 : tensor<32x2048x14x14xf32>
    %v2104 = stablehlo.reshape %v2103 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2105 = stablehlo.reshape %v2104 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2106 = stablehlo.multiply %v2105, %v2105 : tensor<32x2048x14x14xf32>
    %v2107 = stablehlo.multiply %v2106, %v2105 : tensor<32x2048x14x14xf32>
    %v2108 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v2109 = stablehlo.multiply %v2108, %v2107 : tensor<32x2048x14x14xf32>
    %v2110 = stablehlo.add %v2105, %v2109 : tensor<32x2048x14x14xf32>
    %v2111 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v2112 = stablehlo.multiply %v2111, %v2110 : tensor<32x2048x14x14xf32>
    %v2113 = stablehlo.tanh %v2112 : tensor<32x2048x14x14xf32>
    %v2114 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v2115 = stablehlo.add %v2114, %v2113 : tensor<32x2048x14x14xf32>
    %v2116 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v2117 = stablehlo.multiply %v2116, %v2105 : tensor<32x2048x14x14xf32>
    %v2118 = stablehlo.multiply %v2117, %v2115 : tensor<32x2048x14x14xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2121 = stablehlo.convolution(%v2120, %s2b21pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2122 = stablehlo.broadcast_in_dim %s2b21pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2123 = stablehlo.add %v2121, %v2122 : tensor<32x512x14x14xf32>
    %v2124 = stablehlo.reshape %v2123 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2126 = stablehlo.broadcast_in_dim %s2b21lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2127 = stablehlo.multiply %v2125, %v2126 : tensor<32x512x14x14xf32>
    %v2128 = stablehlo.reshape %v2127 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2129 = stablehlo.reshape %v2128 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2130 = stablehlo.reshape %v2060 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2131 = stablehlo.add %v2129, %v2130 : tensor<32x512x14x14xf32>
    %v2132 = stablehlo.reshape %v2131 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2133 = stablehlo.reshape %v2132 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2134 = stablehlo.convolution(%v2133, %s2b22dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2135 = stablehlo.broadcast_in_dim %s2b22db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2136 = stablehlo.add %v2134, %v2135 : tensor<32x512x14x14xf32>
    %v2137 = stablehlo.reshape %v2136 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2138 = stablehlo.reshape %v2137 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2139 = stablehlo.transpose %v2138, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2140 = stablehlo.reshape %v2139 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2141 = stablehlo.reshape %v2140 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2143 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2144 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2145 = stablehlo.reduce(%v2141 init: %v2142) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2146 = stablehlo.broadcast_in_dim %v2145, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2147 = stablehlo.divide %v2146, %v2143 : tensor<32x196x512xf32>
    %v2148 = stablehlo.subtract %v2141, %v2147 : tensor<32x196x512xf32>
    %v2149 = stablehlo.multiply %v2148, %v2148 : tensor<32x196x512xf32>
    %v2150 = stablehlo.reduce(%v2149 init: %v2142) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2151 = stablehlo.broadcast_in_dim %v2150, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2152 = stablehlo.divide %v2151, %v2143 : tensor<32x196x512xf32>
    %v2153 = stablehlo.add %v2152, %v2144 : tensor<32x196x512xf32>
    %v2154 = stablehlo.rsqrt %v2153 : tensor<32x196x512xf32>
    %v2155 = stablehlo.multiply %v2148, %v2154 : tensor<32x196x512xf32>
    %v2156 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2157 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2158 = stablehlo.multiply %v2155, %v2156 : tensor<32x196x512xf32>
    %v2159 = stablehlo.add %v2158, %v2157 : tensor<32x196x512xf32>
    %v2160 = stablehlo.reshape %v2159 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2161 = stablehlo.reshape %v2160 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2162 = stablehlo.broadcast_in_dim %s2b22ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2163 = stablehlo.multiply %v2161, %v2162 : tensor<32x196x512xf32>
    %v2164 = stablehlo.reshape %v2163 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2166 = stablehlo.broadcast_in_dim %s2b22nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2167 = stablehlo.add %v2165, %v2166 : tensor<32x196x512xf32>
    %v2168 = stablehlo.reshape %v2167 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2169 = stablehlo.reshape %v2168 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2170 = stablehlo.transpose %v2169, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2171 = stablehlo.reshape %v2170 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2172 = stablehlo.reshape %v2171 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2173 = stablehlo.convolution(%v2172, %s2b22eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2174 = stablehlo.broadcast_in_dim %s2b22eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2175 = stablehlo.add %v2173, %v2174 : tensor<32x2048x14x14xf32>
    %v2176 = stablehlo.reshape %v2175 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2177 = stablehlo.reshape %v2176 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2178 = stablehlo.multiply %v2177, %v2177 : tensor<32x2048x14x14xf32>
    %v2179 = stablehlo.multiply %v2178, %v2177 : tensor<32x2048x14x14xf32>
    %v2180 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v2181 = stablehlo.multiply %v2180, %v2179 : tensor<32x2048x14x14xf32>
    %v2182 = stablehlo.add %v2177, %v2181 : tensor<32x2048x14x14xf32>
    %v2183 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v2184 = stablehlo.multiply %v2183, %v2182 : tensor<32x2048x14x14xf32>
    %v2185 = stablehlo.tanh %v2184 : tensor<32x2048x14x14xf32>
    %v2186 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v2187 = stablehlo.add %v2186, %v2185 : tensor<32x2048x14x14xf32>
    %v2188 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v2189 = stablehlo.multiply %v2188, %v2177 : tensor<32x2048x14x14xf32>
    %v2190 = stablehlo.multiply %v2189, %v2187 : tensor<32x2048x14x14xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2192 = stablehlo.reshape %v2191 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2193 = stablehlo.convolution(%v2192, %s2b22pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2194 = stablehlo.broadcast_in_dim %s2b22pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2195 = stablehlo.add %v2193, %v2194 : tensor<32x512x14x14xf32>
    %v2196 = stablehlo.reshape %v2195 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2197 = stablehlo.reshape %v2196 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2198 = stablehlo.broadcast_in_dim %s2b22lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2199 = stablehlo.multiply %v2197, %v2198 : tensor<32x512x14x14xf32>
    %v2200 = stablehlo.reshape %v2199 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2201 = stablehlo.reshape %v2200 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2202 = stablehlo.reshape %v2132 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2203 = stablehlo.add %v2201, %v2202 : tensor<32x512x14x14xf32>
    %v2204 = stablehlo.reshape %v2203 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2205 = stablehlo.reshape %v2204 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2206 = stablehlo.convolution(%v2205, %s2b23dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2207 = stablehlo.broadcast_in_dim %s2b23db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2208 = stablehlo.add %v2206, %v2207 : tensor<32x512x14x14xf32>
    %v2209 = stablehlo.reshape %v2208 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2210 = stablehlo.reshape %v2209 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2211 = stablehlo.transpose %v2210, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2212 = stablehlo.reshape %v2211 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2213 = stablehlo.reshape %v2212 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2215 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2216 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2217 = stablehlo.reduce(%v2213 init: %v2214) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2218 = stablehlo.broadcast_in_dim %v2217, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2219 = stablehlo.divide %v2218, %v2215 : tensor<32x196x512xf32>
    %v2220 = stablehlo.subtract %v2213, %v2219 : tensor<32x196x512xf32>
    %v2221 = stablehlo.multiply %v2220, %v2220 : tensor<32x196x512xf32>
    %v2222 = stablehlo.reduce(%v2221 init: %v2214) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2223 = stablehlo.broadcast_in_dim %v2222, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2224 = stablehlo.divide %v2223, %v2215 : tensor<32x196x512xf32>
    %v2225 = stablehlo.add %v2224, %v2216 : tensor<32x196x512xf32>
    %v2226 = stablehlo.rsqrt %v2225 : tensor<32x196x512xf32>
    %v2227 = stablehlo.multiply %v2220, %v2226 : tensor<32x196x512xf32>
    %v2228 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2229 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2230 = stablehlo.multiply %v2227, %v2228 : tensor<32x196x512xf32>
    %v2231 = stablehlo.add %v2230, %v2229 : tensor<32x196x512xf32>
    %v2232 = stablehlo.reshape %v2231 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2233 = stablehlo.reshape %v2232 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2234 = stablehlo.broadcast_in_dim %s2b23ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2235 = stablehlo.multiply %v2233, %v2234 : tensor<32x196x512xf32>
    %v2236 = stablehlo.reshape %v2235 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2237 = stablehlo.reshape %v2236 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2238 = stablehlo.broadcast_in_dim %s2b23nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2239 = stablehlo.add %v2237, %v2238 : tensor<32x196x512xf32>
    %v2240 = stablehlo.reshape %v2239 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2241 = stablehlo.reshape %v2240 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2242 = stablehlo.transpose %v2241, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2243 = stablehlo.reshape %v2242 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2244 = stablehlo.reshape %v2243 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2245 = stablehlo.convolution(%v2244, %s2b23eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2246 = stablehlo.broadcast_in_dim %s2b23eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2247 = stablehlo.add %v2245, %v2246 : tensor<32x2048x14x14xf32>
    %v2248 = stablehlo.reshape %v2247 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2249 = stablehlo.reshape %v2248 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2250 = stablehlo.multiply %v2249, %v2249 : tensor<32x2048x14x14xf32>
    %v2251 = stablehlo.multiply %v2250, %v2249 : tensor<32x2048x14x14xf32>
    %v2252 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v2253 = stablehlo.multiply %v2252, %v2251 : tensor<32x2048x14x14xf32>
    %v2254 = stablehlo.add %v2249, %v2253 : tensor<32x2048x14x14xf32>
    %v2255 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v2256 = stablehlo.multiply %v2255, %v2254 : tensor<32x2048x14x14xf32>
    %v2257 = stablehlo.tanh %v2256 : tensor<32x2048x14x14xf32>
    %v2258 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v2259 = stablehlo.add %v2258, %v2257 : tensor<32x2048x14x14xf32>
    %v2260 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v2261 = stablehlo.multiply %v2260, %v2249 : tensor<32x2048x14x14xf32>
    %v2262 = stablehlo.multiply %v2261, %v2259 : tensor<32x2048x14x14xf32>
    %v2263 = stablehlo.reshape %v2262 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2264 = stablehlo.reshape %v2263 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2265 = stablehlo.convolution(%v2264, %s2b23pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2266 = stablehlo.broadcast_in_dim %s2b23pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2267 = stablehlo.add %v2265, %v2266 : tensor<32x512x14x14xf32>
    %v2268 = stablehlo.reshape %v2267 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2269 = stablehlo.reshape %v2268 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2270 = stablehlo.broadcast_in_dim %s2b23lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2271 = stablehlo.multiply %v2269, %v2270 : tensor<32x512x14x14xf32>
    %v2272 = stablehlo.reshape %v2271 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2273 = stablehlo.reshape %v2272 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2274 = stablehlo.reshape %v2204 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2275 = stablehlo.add %v2273, %v2274 : tensor<32x512x14x14xf32>
    %v2276 = stablehlo.reshape %v2275 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2277 = stablehlo.reshape %v2276 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2278 = stablehlo.convolution(%v2277, %s2b24dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2279 = stablehlo.broadcast_in_dim %s2b24db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2280 = stablehlo.add %v2278, %v2279 : tensor<32x512x14x14xf32>
    %v2281 = stablehlo.reshape %v2280 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2282 = stablehlo.reshape %v2281 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2283 = stablehlo.transpose %v2282, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2284 = stablehlo.reshape %v2283 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2285 = stablehlo.reshape %v2284 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2287 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2288 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2289 = stablehlo.reduce(%v2285 init: %v2286) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2290 = stablehlo.broadcast_in_dim %v2289, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2291 = stablehlo.divide %v2290, %v2287 : tensor<32x196x512xf32>
    %v2292 = stablehlo.subtract %v2285, %v2291 : tensor<32x196x512xf32>
    %v2293 = stablehlo.multiply %v2292, %v2292 : tensor<32x196x512xf32>
    %v2294 = stablehlo.reduce(%v2293 init: %v2286) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2295 = stablehlo.broadcast_in_dim %v2294, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2296 = stablehlo.divide %v2295, %v2287 : tensor<32x196x512xf32>
    %v2297 = stablehlo.add %v2296, %v2288 : tensor<32x196x512xf32>
    %v2298 = stablehlo.rsqrt %v2297 : tensor<32x196x512xf32>
    %v2299 = stablehlo.multiply %v2292, %v2298 : tensor<32x196x512xf32>
    %v2300 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2301 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2302 = stablehlo.multiply %v2299, %v2300 : tensor<32x196x512xf32>
    %v2303 = stablehlo.add %v2302, %v2301 : tensor<32x196x512xf32>
    %v2304 = stablehlo.reshape %v2303 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2305 = stablehlo.reshape %v2304 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2306 = stablehlo.broadcast_in_dim %s2b24ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2307 = stablehlo.multiply %v2305, %v2306 : tensor<32x196x512xf32>
    %v2308 = stablehlo.reshape %v2307 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2309 = stablehlo.reshape %v2308 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2310 = stablehlo.broadcast_in_dim %s2b24nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2311 = stablehlo.add %v2309, %v2310 : tensor<32x196x512xf32>
    %v2312 = stablehlo.reshape %v2311 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2313 = stablehlo.reshape %v2312 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2314 = stablehlo.transpose %v2313, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2315 = stablehlo.reshape %v2314 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2316 = stablehlo.reshape %v2315 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2317 = stablehlo.convolution(%v2316, %s2b24eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2318 = stablehlo.broadcast_in_dim %s2b24eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2319 = stablehlo.add %v2317, %v2318 : tensor<32x2048x14x14xf32>
    %v2320 = stablehlo.reshape %v2319 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2321 = stablehlo.reshape %v2320 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2322 = stablehlo.multiply %v2321, %v2321 : tensor<32x2048x14x14xf32>
    %v2323 = stablehlo.multiply %v2322, %v2321 : tensor<32x2048x14x14xf32>
    %v2324 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v2325 = stablehlo.multiply %v2324, %v2323 : tensor<32x2048x14x14xf32>
    %v2326 = stablehlo.add %v2321, %v2325 : tensor<32x2048x14x14xf32>
    %v2327 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v2328 = stablehlo.multiply %v2327, %v2326 : tensor<32x2048x14x14xf32>
    %v2329 = stablehlo.tanh %v2328 : tensor<32x2048x14x14xf32>
    %v2330 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v2331 = stablehlo.add %v2330, %v2329 : tensor<32x2048x14x14xf32>
    %v2332 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v2333 = stablehlo.multiply %v2332, %v2321 : tensor<32x2048x14x14xf32>
    %v2334 = stablehlo.multiply %v2333, %v2331 : tensor<32x2048x14x14xf32>
    %v2335 = stablehlo.reshape %v2334 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2336 = stablehlo.reshape %v2335 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2337 = stablehlo.convolution(%v2336, %s2b24pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2338 = stablehlo.broadcast_in_dim %s2b24pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2339 = stablehlo.add %v2337, %v2338 : tensor<32x512x14x14xf32>
    %v2340 = stablehlo.reshape %v2339 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2342 = stablehlo.broadcast_in_dim %s2b24lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2343 = stablehlo.multiply %v2341, %v2342 : tensor<32x512x14x14xf32>
    %v2344 = stablehlo.reshape %v2343 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2345 = stablehlo.reshape %v2344 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2346 = stablehlo.reshape %v2276 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2347 = stablehlo.add %v2345, %v2346 : tensor<32x512x14x14xf32>
    %v2348 = stablehlo.reshape %v2347 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2350 = stablehlo.convolution(%v2349, %s2b25dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2351 = stablehlo.broadcast_in_dim %s2b25db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2352 = stablehlo.add %v2350, %v2351 : tensor<32x512x14x14xf32>
    %v2353 = stablehlo.reshape %v2352 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2354 = stablehlo.reshape %v2353 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2355 = stablehlo.transpose %v2354, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2356 = stablehlo.reshape %v2355 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2359 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2360 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2361 = stablehlo.reduce(%v2357 init: %v2358) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2362 = stablehlo.broadcast_in_dim %v2361, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2363 = stablehlo.divide %v2362, %v2359 : tensor<32x196x512xf32>
    %v2364 = stablehlo.subtract %v2357, %v2363 : tensor<32x196x512xf32>
    %v2365 = stablehlo.multiply %v2364, %v2364 : tensor<32x196x512xf32>
    %v2366 = stablehlo.reduce(%v2365 init: %v2358) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2367 = stablehlo.broadcast_in_dim %v2366, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2368 = stablehlo.divide %v2367, %v2359 : tensor<32x196x512xf32>
    %v2369 = stablehlo.add %v2368, %v2360 : tensor<32x196x512xf32>
    %v2370 = stablehlo.rsqrt %v2369 : tensor<32x196x512xf32>
    %v2371 = stablehlo.multiply %v2364, %v2370 : tensor<32x196x512xf32>
    %v2372 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2373 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2374 = stablehlo.multiply %v2371, %v2372 : tensor<32x196x512xf32>
    %v2375 = stablehlo.add %v2374, %v2373 : tensor<32x196x512xf32>
    %v2376 = stablehlo.reshape %v2375 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2377 = stablehlo.reshape %v2376 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2378 = stablehlo.broadcast_in_dim %s2b25ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2379 = stablehlo.multiply %v2377, %v2378 : tensor<32x196x512xf32>
    %v2380 = stablehlo.reshape %v2379 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2381 = stablehlo.reshape %v2380 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2382 = stablehlo.broadcast_in_dim %s2b25nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2383 = stablehlo.add %v2381, %v2382 : tensor<32x196x512xf32>
    %v2384 = stablehlo.reshape %v2383 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2385 = stablehlo.reshape %v2384 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2386 = stablehlo.transpose %v2385, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2387 = stablehlo.reshape %v2386 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2388 = stablehlo.reshape %v2387 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2389 = stablehlo.convolution(%v2388, %s2b25eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2390 = stablehlo.broadcast_in_dim %s2b25eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2391 = stablehlo.add %v2389, %v2390 : tensor<32x2048x14x14xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2393 = stablehlo.reshape %v2392 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2394 = stablehlo.multiply %v2393, %v2393 : tensor<32x2048x14x14xf32>
    %v2395 = stablehlo.multiply %v2394, %v2393 : tensor<32x2048x14x14xf32>
    %v2396 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v2397 = stablehlo.multiply %v2396, %v2395 : tensor<32x2048x14x14xf32>
    %v2398 = stablehlo.add %v2393, %v2397 : tensor<32x2048x14x14xf32>
    %v2399 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v2400 = stablehlo.multiply %v2399, %v2398 : tensor<32x2048x14x14xf32>
    %v2401 = stablehlo.tanh %v2400 : tensor<32x2048x14x14xf32>
    %v2402 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v2403 = stablehlo.add %v2402, %v2401 : tensor<32x2048x14x14xf32>
    %v2404 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v2405 = stablehlo.multiply %v2404, %v2393 : tensor<32x2048x14x14xf32>
    %v2406 = stablehlo.multiply %v2405, %v2403 : tensor<32x2048x14x14xf32>
    %v2407 = stablehlo.reshape %v2406 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2408 = stablehlo.reshape %v2407 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2409 = stablehlo.convolution(%v2408, %s2b25pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2410 = stablehlo.broadcast_in_dim %s2b25pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2411 = stablehlo.add %v2409, %v2410 : tensor<32x512x14x14xf32>
    %v2412 = stablehlo.reshape %v2411 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2413 = stablehlo.reshape %v2412 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2414 = stablehlo.broadcast_in_dim %s2b25lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2415 = stablehlo.multiply %v2413, %v2414 : tensor<32x512x14x14xf32>
    %v2416 = stablehlo.reshape %v2415 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2417 = stablehlo.reshape %v2416 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2418 = stablehlo.reshape %v2348 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2419 = stablehlo.add %v2417, %v2418 : tensor<32x512x14x14xf32>
    %v2420 = stablehlo.reshape %v2419 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2421 = stablehlo.reshape %v2420 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2422 = stablehlo.convolution(%v2421, %s2b26dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2423 = stablehlo.broadcast_in_dim %s2b26db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2424 = stablehlo.add %v2422, %v2423 : tensor<32x512x14x14xf32>
    %v2425 = stablehlo.reshape %v2424 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2426 = stablehlo.reshape %v2425 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2427 = stablehlo.transpose %v2426, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2428 = stablehlo.reshape %v2427 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2431 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2432 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2433 = stablehlo.reduce(%v2429 init: %v2430) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2434 = stablehlo.broadcast_in_dim %v2433, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2435 = stablehlo.divide %v2434, %v2431 : tensor<32x196x512xf32>
    %v2436 = stablehlo.subtract %v2429, %v2435 : tensor<32x196x512xf32>
    %v2437 = stablehlo.multiply %v2436, %v2436 : tensor<32x196x512xf32>
    %v2438 = stablehlo.reduce(%v2437 init: %v2430) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2439 = stablehlo.broadcast_in_dim %v2438, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2440 = stablehlo.divide %v2439, %v2431 : tensor<32x196x512xf32>
    %v2441 = stablehlo.add %v2440, %v2432 : tensor<32x196x512xf32>
    %v2442 = stablehlo.rsqrt %v2441 : tensor<32x196x512xf32>
    %v2443 = stablehlo.multiply %v2436, %v2442 : tensor<32x196x512xf32>
    %v2444 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2445 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2446 = stablehlo.multiply %v2443, %v2444 : tensor<32x196x512xf32>
    %v2447 = stablehlo.add %v2446, %v2445 : tensor<32x196x512xf32>
    %v2448 = stablehlo.reshape %v2447 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2449 = stablehlo.reshape %v2448 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2450 = stablehlo.broadcast_in_dim %s2b26ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2451 = stablehlo.multiply %v2449, %v2450 : tensor<32x196x512xf32>
    %v2452 = stablehlo.reshape %v2451 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2453 = stablehlo.reshape %v2452 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2454 = stablehlo.broadcast_in_dim %s2b26nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2455 = stablehlo.add %v2453, %v2454 : tensor<32x196x512xf32>
    %v2456 = stablehlo.reshape %v2455 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2457 = stablehlo.reshape %v2456 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2458 = stablehlo.transpose %v2457, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2459 = stablehlo.reshape %v2458 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2460 = stablehlo.reshape %v2459 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2461 = stablehlo.convolution(%v2460, %s2b26eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2462 = stablehlo.broadcast_in_dim %s2b26eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2463 = stablehlo.add %v2461, %v2462 : tensor<32x2048x14x14xf32>
    %v2464 = stablehlo.reshape %v2463 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2465 = stablehlo.reshape %v2464 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2466 = stablehlo.multiply %v2465, %v2465 : tensor<32x2048x14x14xf32>
    %v2467 = stablehlo.multiply %v2466, %v2465 : tensor<32x2048x14x14xf32>
    %v2468 = stablehlo.constant dense<0.044715> : tensor<32x2048x14x14xf32>
    %v2469 = stablehlo.multiply %v2468, %v2467 : tensor<32x2048x14x14xf32>
    %v2470 = stablehlo.add %v2465, %v2469 : tensor<32x2048x14x14xf32>
    %v2471 = stablehlo.constant dense<0.7978845608028654> : tensor<32x2048x14x14xf32>
    %v2472 = stablehlo.multiply %v2471, %v2470 : tensor<32x2048x14x14xf32>
    %v2473 = stablehlo.tanh %v2472 : tensor<32x2048x14x14xf32>
    %v2474 = stablehlo.constant dense<1.0> : tensor<32x2048x14x14xf32>
    %v2475 = stablehlo.add %v2474, %v2473 : tensor<32x2048x14x14xf32>
    %v2476 = stablehlo.constant dense<0.5> : tensor<32x2048x14x14xf32>
    %v2477 = stablehlo.multiply %v2476, %v2465 : tensor<32x2048x14x14xf32>
    %v2478 = stablehlo.multiply %v2477, %v2475 : tensor<32x2048x14x14xf32>
    %v2479 = stablehlo.reshape %v2478 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2480 = stablehlo.reshape %v2479 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2481 = stablehlo.convolution(%v2480, %s2b26pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2482 = stablehlo.broadcast_in_dim %s2b26pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2483 = stablehlo.add %v2481, %v2482 : tensor<32x512x14x14xf32>
    %v2484 = stablehlo.reshape %v2483 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2485 = stablehlo.reshape %v2484 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2486 = stablehlo.broadcast_in_dim %s2b26lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2487 = stablehlo.multiply %v2485, %v2486 : tensor<32x512x14x14xf32>
    %v2488 = stablehlo.reshape %v2487 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2489 = stablehlo.reshape %v2488 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2490 = stablehlo.reshape %v2420 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2491 = stablehlo.add %v2489, %v2490 : tensor<32x512x14x14xf32>
    %v2492 = stablehlo.reshape %v2491 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2493 = stablehlo.reshape %v2492 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2494 = stablehlo.transpose %v2493, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2495 = stablehlo.reshape %v2494 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2496 = stablehlo.reshape %v2495 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2497 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2498 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2499 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2500 = stablehlo.reduce(%v2496 init: %v2497) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2501 = stablehlo.broadcast_in_dim %v2500, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2502 = stablehlo.divide %v2501, %v2498 : tensor<32x196x512xf32>
    %v2503 = stablehlo.subtract %v2496, %v2502 : tensor<32x196x512xf32>
    %v2504 = stablehlo.multiply %v2503, %v2503 : tensor<32x196x512xf32>
    %v2505 = stablehlo.reduce(%v2504 init: %v2497) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2506 = stablehlo.broadcast_in_dim %v2505, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2507 = stablehlo.divide %v2506, %v2498 : tensor<32x196x512xf32>
    %v2508 = stablehlo.add %v2507, %v2499 : tensor<32x196x512xf32>
    %v2509 = stablehlo.rsqrt %v2508 : tensor<32x196x512xf32>
    %v2510 = stablehlo.multiply %v2503, %v2509 : tensor<32x196x512xf32>
    %v2511 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2512 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2513 = stablehlo.multiply %v2510, %v2511 : tensor<32x196x512xf32>
    %v2514 = stablehlo.add %v2513, %v2512 : tensor<32x196x512xf32>
    %v2515 = stablehlo.reshape %v2514 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2516 = stablehlo.reshape %v2515 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2517 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2518 = stablehlo.multiply %v2516, %v2517 : tensor<32x196x512xf32>
    %v2519 = stablehlo.reshape %v2518 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2520 = stablehlo.reshape %v2519 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2521 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2522 = stablehlo.add %v2520, %v2521 : tensor<32x196x512xf32>
    %v2523 = stablehlo.reshape %v2522 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2524 = stablehlo.reshape %v2523 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2525 = stablehlo.transpose %v2524, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2526 = stablehlo.reshape %v2525 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2527 = stablehlo.reshape %v2526 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2528 = stablehlo.convolution(%v2527, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<1024x512x2x2xf32>) -> tensor<32x1024x7x7xf32>
    %v2529 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2530 = stablehlo.add %v2528, %v2529 : tensor<32x1024x7x7xf32>
    %v2531 = stablehlo.reshape %v2530 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2532 = stablehlo.reshape %v2531 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2533 = stablehlo.convolution(%v2532, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x7x7xf32>) -> tensor<32x1024x7x7xf32>
    %v2534 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2535 = stablehlo.add %v2533, %v2534 : tensor<32x1024x7x7xf32>
    %v2536 = stablehlo.reshape %v2535 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2537 = stablehlo.reshape %v2536 : (tensor<32x50176xf32>) -> tensor<32x1024x49xf32>
    %v2538 = stablehlo.transpose %v2537, dims = [0, 2, 1] : (tensor<32x1024x49xf32>) -> tensor<32x49x1024xf32>
    %v2539 = stablehlo.reshape %v2538 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2540 = stablehlo.reshape %v2539 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2541 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2542 = stablehlo.constant dense<1024.0> : tensor<32x49x1024xf32>
    %v2543 = stablehlo.constant dense<1.0e-6> : tensor<32x49x1024xf32>
    %v2544 = stablehlo.reduce(%v2540 init: %v2541) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2545 = stablehlo.broadcast_in_dim %v2544, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2546 = stablehlo.divide %v2545, %v2542 : tensor<32x49x1024xf32>
    %v2547 = stablehlo.subtract %v2540, %v2546 : tensor<32x49x1024xf32>
    %v2548 = stablehlo.multiply %v2547, %v2547 : tensor<32x49x1024xf32>
    %v2549 = stablehlo.reduce(%v2548 init: %v2541) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2550 = stablehlo.broadcast_in_dim %v2549, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2551 = stablehlo.divide %v2550, %v2542 : tensor<32x49x1024xf32>
    %v2552 = stablehlo.add %v2551, %v2543 : tensor<32x49x1024xf32>
    %v2553 = stablehlo.rsqrt %v2552 : tensor<32x49x1024xf32>
    %v2554 = stablehlo.multiply %v2547, %v2553 : tensor<32x49x1024xf32>
    %v2555 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2556 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2557 = stablehlo.multiply %v2554, %v2555 : tensor<32x49x1024xf32>
    %v2558 = stablehlo.add %v2557, %v2556 : tensor<32x49x1024xf32>
    %v2559 = stablehlo.reshape %v2558 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2560 = stablehlo.reshape %v2559 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2561 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2562 = stablehlo.multiply %v2560, %v2561 : tensor<32x49x1024xf32>
    %v2563 = stablehlo.reshape %v2562 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2564 = stablehlo.reshape %v2563 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2565 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2566 = stablehlo.add %v2564, %v2565 : tensor<32x49x1024xf32>
    %v2567 = stablehlo.reshape %v2566 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2569 = stablehlo.transpose %v2568, dims = [0, 2, 1] : (tensor<32x49x1024xf32>) -> tensor<32x1024x49xf32>
    %v2570 = stablehlo.reshape %v2569 : (tensor<32x1024x49xf32>) -> tensor<32x50176xf32>
    %v2571 = stablehlo.reshape %v2570 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2572 = stablehlo.convolution(%v2571, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<4096x1024x1x1xf32>) -> tensor<32x4096x7x7xf32>
    %v2573 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<4096xf32>) -> tensor<32x4096x7x7xf32>
    %v2574 = stablehlo.add %v2572, %v2573 : tensor<32x4096x7x7xf32>
    %v2575 = stablehlo.reshape %v2574 : (tensor<32x4096x7x7xf32>) -> tensor<32x200704xf32>
    %v2576 = stablehlo.reshape %v2575 : (tensor<32x200704xf32>) -> tensor<32x4096x7x7xf32>
    %v2577 = stablehlo.multiply %v2576, %v2576 : tensor<32x4096x7x7xf32>
    %v2578 = stablehlo.multiply %v2577, %v2576 : tensor<32x4096x7x7xf32>
    %v2579 = stablehlo.constant dense<0.044715> : tensor<32x4096x7x7xf32>
    %v2580 = stablehlo.multiply %v2579, %v2578 : tensor<32x4096x7x7xf32>
    %v2581 = stablehlo.add %v2576, %v2580 : tensor<32x4096x7x7xf32>
    %v2582 = stablehlo.constant dense<0.7978845608028654> : tensor<32x4096x7x7xf32>
    %v2583 = stablehlo.multiply %v2582, %v2581 : tensor<32x4096x7x7xf32>
    %v2584 = stablehlo.tanh %v2583 : tensor<32x4096x7x7xf32>
    %v2585 = stablehlo.constant dense<1.0> : tensor<32x4096x7x7xf32>
    %v2586 = stablehlo.add %v2585, %v2584 : tensor<32x4096x7x7xf32>
    %v2587 = stablehlo.constant dense<0.5> : tensor<32x4096x7x7xf32>
    %v2588 = stablehlo.multiply %v2587, %v2576 : tensor<32x4096x7x7xf32>
    %v2589 = stablehlo.multiply %v2588, %v2586 : tensor<32x4096x7x7xf32>
    %v2590 = stablehlo.reshape %v2589 : (tensor<32x4096x7x7xf32>) -> tensor<32x200704xf32>
    %v2591 = stablehlo.reshape %v2590 : (tensor<32x200704xf32>) -> tensor<32x4096x7x7xf32>
    %v2592 = stablehlo.convolution(%v2591, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x4096x7x7xf32>, tensor<1024x4096x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v2593 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2594 = stablehlo.add %v2592, %v2593 : tensor<32x1024x7x7xf32>
    %v2595 = stablehlo.reshape %v2594 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2596 = stablehlo.reshape %v2595 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2597 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2598 = stablehlo.multiply %v2596, %v2597 : tensor<32x1024x7x7xf32>
    %v2599 = stablehlo.reshape %v2598 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2600 = stablehlo.reshape %v2599 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2601 = stablehlo.reshape %v2531 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2602 = stablehlo.add %v2600, %v2601 : tensor<32x1024x7x7xf32>
    %v2603 = stablehlo.reshape %v2602 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2604 = stablehlo.reshape %v2603 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2605 = stablehlo.convolution(%v2604, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x7x7xf32>) -> tensor<32x1024x7x7xf32>
    %v2606 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2607 = stablehlo.add %v2605, %v2606 : tensor<32x1024x7x7xf32>
    %v2608 = stablehlo.reshape %v2607 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2609 = stablehlo.reshape %v2608 : (tensor<32x50176xf32>) -> tensor<32x1024x49xf32>
    %v2610 = stablehlo.transpose %v2609, dims = [0, 2, 1] : (tensor<32x1024x49xf32>) -> tensor<32x49x1024xf32>
    %v2611 = stablehlo.reshape %v2610 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2612 = stablehlo.reshape %v2611 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2614 = stablehlo.constant dense<1024.0> : tensor<32x49x1024xf32>
    %v2615 = stablehlo.constant dense<1.0e-6> : tensor<32x49x1024xf32>
    %v2616 = stablehlo.reduce(%v2612 init: %v2613) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2617 = stablehlo.broadcast_in_dim %v2616, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2618 = stablehlo.divide %v2617, %v2614 : tensor<32x49x1024xf32>
    %v2619 = stablehlo.subtract %v2612, %v2618 : tensor<32x49x1024xf32>
    %v2620 = stablehlo.multiply %v2619, %v2619 : tensor<32x49x1024xf32>
    %v2621 = stablehlo.reduce(%v2620 init: %v2613) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2622 = stablehlo.broadcast_in_dim %v2621, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2623 = stablehlo.divide %v2622, %v2614 : tensor<32x49x1024xf32>
    %v2624 = stablehlo.add %v2623, %v2615 : tensor<32x49x1024xf32>
    %v2625 = stablehlo.rsqrt %v2624 : tensor<32x49x1024xf32>
    %v2626 = stablehlo.multiply %v2619, %v2625 : tensor<32x49x1024xf32>
    %v2627 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2628 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2629 = stablehlo.multiply %v2626, %v2627 : tensor<32x49x1024xf32>
    %v2630 = stablehlo.add %v2629, %v2628 : tensor<32x49x1024xf32>
    %v2631 = stablehlo.reshape %v2630 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2632 = stablehlo.reshape %v2631 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2633 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2634 = stablehlo.multiply %v2632, %v2633 : tensor<32x49x1024xf32>
    %v2635 = stablehlo.reshape %v2634 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2636 = stablehlo.reshape %v2635 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2637 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2638 = stablehlo.add %v2636, %v2637 : tensor<32x49x1024xf32>
    %v2639 = stablehlo.reshape %v2638 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2640 = stablehlo.reshape %v2639 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2641 = stablehlo.transpose %v2640, dims = [0, 2, 1] : (tensor<32x49x1024xf32>) -> tensor<32x1024x49xf32>
    %v2642 = stablehlo.reshape %v2641 : (tensor<32x1024x49xf32>) -> tensor<32x50176xf32>
    %v2643 = stablehlo.reshape %v2642 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2644 = stablehlo.convolution(%v2643, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<4096x1024x1x1xf32>) -> tensor<32x4096x7x7xf32>
    %v2645 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<4096xf32>) -> tensor<32x4096x7x7xf32>
    %v2646 = stablehlo.add %v2644, %v2645 : tensor<32x4096x7x7xf32>
    %v2647 = stablehlo.reshape %v2646 : (tensor<32x4096x7x7xf32>) -> tensor<32x200704xf32>
    %v2648 = stablehlo.reshape %v2647 : (tensor<32x200704xf32>) -> tensor<32x4096x7x7xf32>
    %v2649 = stablehlo.multiply %v2648, %v2648 : tensor<32x4096x7x7xf32>
    %v2650 = stablehlo.multiply %v2649, %v2648 : tensor<32x4096x7x7xf32>
    %v2651 = stablehlo.constant dense<0.044715> : tensor<32x4096x7x7xf32>
    %v2652 = stablehlo.multiply %v2651, %v2650 : tensor<32x4096x7x7xf32>
    %v2653 = stablehlo.add %v2648, %v2652 : tensor<32x4096x7x7xf32>
    %v2654 = stablehlo.constant dense<0.7978845608028654> : tensor<32x4096x7x7xf32>
    %v2655 = stablehlo.multiply %v2654, %v2653 : tensor<32x4096x7x7xf32>
    %v2656 = stablehlo.tanh %v2655 : tensor<32x4096x7x7xf32>
    %v2657 = stablehlo.constant dense<1.0> : tensor<32x4096x7x7xf32>
    %v2658 = stablehlo.add %v2657, %v2656 : tensor<32x4096x7x7xf32>
    %v2659 = stablehlo.constant dense<0.5> : tensor<32x4096x7x7xf32>
    %v2660 = stablehlo.multiply %v2659, %v2648 : tensor<32x4096x7x7xf32>
    %v2661 = stablehlo.multiply %v2660, %v2658 : tensor<32x4096x7x7xf32>
    %v2662 = stablehlo.reshape %v2661 : (tensor<32x4096x7x7xf32>) -> tensor<32x200704xf32>
    %v2663 = stablehlo.reshape %v2662 : (tensor<32x200704xf32>) -> tensor<32x4096x7x7xf32>
    %v2664 = stablehlo.convolution(%v2663, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x4096x7x7xf32>, tensor<1024x4096x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v2665 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2666 = stablehlo.add %v2664, %v2665 : tensor<32x1024x7x7xf32>
    %v2667 = stablehlo.reshape %v2666 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2668 = stablehlo.reshape %v2667 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2669 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2670 = stablehlo.multiply %v2668, %v2669 : tensor<32x1024x7x7xf32>
    %v2671 = stablehlo.reshape %v2670 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2672 = stablehlo.reshape %v2671 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2673 = stablehlo.reshape %v2603 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2674 = stablehlo.add %v2672, %v2673 : tensor<32x1024x7x7xf32>
    %v2675 = stablehlo.reshape %v2674 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2676 = stablehlo.reshape %v2675 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2677 = stablehlo.convolution(%v2676, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x7x7xf32>) -> tensor<32x1024x7x7xf32>
    %v2678 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2679 = stablehlo.add %v2677, %v2678 : tensor<32x1024x7x7xf32>
    %v2680 = stablehlo.reshape %v2679 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2681 = stablehlo.reshape %v2680 : (tensor<32x50176xf32>) -> tensor<32x1024x49xf32>
    %v2682 = stablehlo.transpose %v2681, dims = [0, 2, 1] : (tensor<32x1024x49xf32>) -> tensor<32x49x1024xf32>
    %v2683 = stablehlo.reshape %v2682 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2684 = stablehlo.reshape %v2683 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2685 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2686 = stablehlo.constant dense<1024.0> : tensor<32x49x1024xf32>
    %v2687 = stablehlo.constant dense<1.0e-6> : tensor<32x49x1024xf32>
    %v2688 = stablehlo.reduce(%v2684 init: %v2685) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2689 = stablehlo.broadcast_in_dim %v2688, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2690 = stablehlo.divide %v2689, %v2686 : tensor<32x49x1024xf32>
    %v2691 = stablehlo.subtract %v2684, %v2690 : tensor<32x49x1024xf32>
    %v2692 = stablehlo.multiply %v2691, %v2691 : tensor<32x49x1024xf32>
    %v2693 = stablehlo.reduce(%v2692 init: %v2685) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2694 = stablehlo.broadcast_in_dim %v2693, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2695 = stablehlo.divide %v2694, %v2686 : tensor<32x49x1024xf32>
    %v2696 = stablehlo.add %v2695, %v2687 : tensor<32x49x1024xf32>
    %v2697 = stablehlo.rsqrt %v2696 : tensor<32x49x1024xf32>
    %v2698 = stablehlo.multiply %v2691, %v2697 : tensor<32x49x1024xf32>
    %v2699 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2700 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2701 = stablehlo.multiply %v2698, %v2699 : tensor<32x49x1024xf32>
    %v2702 = stablehlo.add %v2701, %v2700 : tensor<32x49x1024xf32>
    %v2703 = stablehlo.reshape %v2702 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2704 = stablehlo.reshape %v2703 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2705 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2706 = stablehlo.multiply %v2704, %v2705 : tensor<32x49x1024xf32>
    %v2707 = stablehlo.reshape %v2706 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2708 = stablehlo.reshape %v2707 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2709 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2710 = stablehlo.add %v2708, %v2709 : tensor<32x49x1024xf32>
    %v2711 = stablehlo.reshape %v2710 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2712 = stablehlo.reshape %v2711 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2713 = stablehlo.transpose %v2712, dims = [0, 2, 1] : (tensor<32x49x1024xf32>) -> tensor<32x1024x49xf32>
    %v2714 = stablehlo.reshape %v2713 : (tensor<32x1024x49xf32>) -> tensor<32x50176xf32>
    %v2715 = stablehlo.reshape %v2714 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2716 = stablehlo.convolution(%v2715, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<4096x1024x1x1xf32>) -> tensor<32x4096x7x7xf32>
    %v2717 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<4096xf32>) -> tensor<32x4096x7x7xf32>
    %v2718 = stablehlo.add %v2716, %v2717 : tensor<32x4096x7x7xf32>
    %v2719 = stablehlo.reshape %v2718 : (tensor<32x4096x7x7xf32>) -> tensor<32x200704xf32>
    %v2720 = stablehlo.reshape %v2719 : (tensor<32x200704xf32>) -> tensor<32x4096x7x7xf32>
    %v2721 = stablehlo.multiply %v2720, %v2720 : tensor<32x4096x7x7xf32>
    %v2722 = stablehlo.multiply %v2721, %v2720 : tensor<32x4096x7x7xf32>
    %v2723 = stablehlo.constant dense<0.044715> : tensor<32x4096x7x7xf32>
    %v2724 = stablehlo.multiply %v2723, %v2722 : tensor<32x4096x7x7xf32>
    %v2725 = stablehlo.add %v2720, %v2724 : tensor<32x4096x7x7xf32>
    %v2726 = stablehlo.constant dense<0.7978845608028654> : tensor<32x4096x7x7xf32>
    %v2727 = stablehlo.multiply %v2726, %v2725 : tensor<32x4096x7x7xf32>
    %v2728 = stablehlo.tanh %v2727 : tensor<32x4096x7x7xf32>
    %v2729 = stablehlo.constant dense<1.0> : tensor<32x4096x7x7xf32>
    %v2730 = stablehlo.add %v2729, %v2728 : tensor<32x4096x7x7xf32>
    %v2731 = stablehlo.constant dense<0.5> : tensor<32x4096x7x7xf32>
    %v2732 = stablehlo.multiply %v2731, %v2720 : tensor<32x4096x7x7xf32>
    %v2733 = stablehlo.multiply %v2732, %v2730 : tensor<32x4096x7x7xf32>
    %v2734 = stablehlo.reshape %v2733 : (tensor<32x4096x7x7xf32>) -> tensor<32x200704xf32>
    %v2735 = stablehlo.reshape %v2734 : (tensor<32x200704xf32>) -> tensor<32x4096x7x7xf32>
    %v2736 = stablehlo.convolution(%v2735, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x4096x7x7xf32>, tensor<1024x4096x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v2737 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2738 = stablehlo.add %v2736, %v2737 : tensor<32x1024x7x7xf32>
    %v2739 = stablehlo.reshape %v2738 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2740 = stablehlo.reshape %v2739 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2741 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2742 = stablehlo.multiply %v2740, %v2741 : tensor<32x1024x7x7xf32>
    %v2743 = stablehlo.reshape %v2742 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2744 = stablehlo.reshape %v2743 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2745 = stablehlo.reshape %v2675 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2746 = stablehlo.add %v2744, %v2745 : tensor<32x1024x7x7xf32>
    %v2747 = stablehlo.reshape %v2746 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2748 = stablehlo.reshape %v2747 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2750 = stablehlo.reduce(%v2748 init: %v2749) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<32x1024xf32>
    %v2751 = stablehlo.constant dense<49.0> : tensor<32x1024xf32>
    %v2752 = stablehlo.divide %v2750, %v2751 : tensor<32x1024xf32>
    %v2753 = stablehlo.reshape %v2752 : (tensor<32x1024xf32>) -> tensor<32x1x1024xf32>
    %v2754 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2755 = stablehlo.constant dense<1024.0> : tensor<32x1x1024xf32>
    %v2756 = stablehlo.constant dense<1.0e-6> : tensor<32x1x1024xf32>
    %v2757 = stablehlo.reduce(%v2753 init: %v2754) applies stablehlo.add across dimensions = [2] : (tensor<32x1x1024xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v2758 = stablehlo.broadcast_in_dim %v2757, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x1024xf32>
    %v2759 = stablehlo.divide %v2758, %v2755 : tensor<32x1x1024xf32>
    %v2760 = stablehlo.subtract %v2753, %v2759 : tensor<32x1x1024xf32>
    %v2761 = stablehlo.multiply %v2760, %v2760 : tensor<32x1x1024xf32>
    %v2762 = stablehlo.reduce(%v2761 init: %v2754) applies stablehlo.add across dimensions = [2] : (tensor<32x1x1024xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v2763 = stablehlo.broadcast_in_dim %v2762, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x1024xf32>
    %v2764 = stablehlo.divide %v2763, %v2755 : tensor<32x1x1024xf32>
    %v2765 = stablehlo.add %v2764, %v2756 : tensor<32x1x1024xf32>
    %v2766 = stablehlo.rsqrt %v2765 : tensor<32x1x1024xf32>
    %v2767 = stablehlo.multiply %v2760, %v2766 : tensor<32x1x1024xf32>
    %v2768 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x1x1024xf32>
    %v2769 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x1x1024xf32>
    %v2770 = stablehlo.multiply %v2767, %v2768 : tensor<32x1x1024xf32>
    %v2771 = stablehlo.add %v2770, %v2769 : tensor<32x1x1024xf32>
    %v2772 = stablehlo.reshape %v2771 : (tensor<32x1x1024xf32>) -> tensor<32x1024xf32>
    %v2773 = stablehlo.reshape %v2772 : (tensor<32x1024xf32>) -> tensor<32x1x1024xf32>
    %v2774 = stablehlo.broadcast_in_dim %hng, dims = [2] : (tensor<1024xf32>) -> tensor<32x1x1024xf32>
    %v2775 = stablehlo.multiply %v2773, %v2774 : tensor<32x1x1024xf32>
    %v2776 = stablehlo.reshape %v2775 : (tensor<32x1x1024xf32>) -> tensor<32x1024xf32>
    %v2777 = stablehlo.reshape %v2776 : (tensor<32x1024xf32>) -> tensor<32x1x1024xf32>
    %v2778 = stablehlo.broadcast_in_dim %hnbt, dims = [2] : (tensor<1024xf32>) -> tensor<32x1x1024xf32>
    %v2779 = stablehlo.add %v2777, %v2778 : tensor<32x1x1024xf32>
    %v2780 = stablehlo.reshape %v2779 : (tensor<32x1x1024xf32>) -> tensor<32x1024xf32>
    %v2781 = stablehlo.dot_general %v2780, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1024xf32>, tensor<1024x1000xf32>) -> tensor<32x1000xf32>
    %v2782 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v2783 = stablehlo.add %v2781, %v2782 : tensor<32x1000xf32>
    return %v2783 : tensor<32x1000xf32>
  }
}
