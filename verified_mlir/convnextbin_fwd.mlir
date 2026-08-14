module @m {
  func.func @convnextbin_fwd(%x: tensor<32x150528xf32>, %psW: tensor<128x3x4x4xf32>, %psb: tensor<128xf32>, %psng: tensor<128xf32>, %psnbt: tensor<128xf32>, %s0b0dW: tensor<128x1x7x7xf32>, %s0b0db: tensor<128xf32>, %s0b0ng: tensor<128xf32>, %s0b0nbt: tensor<128xf32>, %s0b0eW: tensor<512x128x1x1xf32>, %s0b0eb: tensor<512xf32>, %s0b0pW: tensor<128x512x1x1xf32>, %s0b0pb: tensor<128xf32>, %s0b0lg: tensor<128xf32>, %s0b1dW: tensor<128x1x7x7xf32>, %s0b1db: tensor<128xf32>, %s0b1ng: tensor<128xf32>, %s0b1nbt: tensor<128xf32>, %s0b1eW: tensor<512x128x1x1xf32>, %s0b1eb: tensor<512xf32>, %s0b1pW: tensor<128x512x1x1xf32>, %s0b1pb: tensor<128xf32>, %s0b1lg: tensor<128xf32>, %s0b2dW: tensor<128x1x7x7xf32>, %s0b2db: tensor<128xf32>, %s0b2ng: tensor<128xf32>, %s0b2nbt: tensor<128xf32>, %s0b2eW: tensor<512x128x1x1xf32>, %s0b2eb: tensor<512xf32>, %s0b2pW: tensor<128x512x1x1xf32>, %s0b2pb: tensor<128xf32>, %s0b2lg: tensor<128xf32>, %d0ng: tensor<128xf32>, %d0nbt: tensor<128xf32>, %d0W: tensor<256x128x2x2xf32>, %d0b: tensor<256xf32>, %s1b0dW: tensor<256x1x7x7xf32>, %s1b0db: tensor<256xf32>, %s1b0ng: tensor<256xf32>, %s1b0nbt: tensor<256xf32>, %s1b0eW: tensor<1024x256x1x1xf32>, %s1b0eb: tensor<1024xf32>, %s1b0pW: tensor<256x1024x1x1xf32>, %s1b0pb: tensor<256xf32>, %s1b0lg: tensor<256xf32>, %s1b1dW: tensor<256x1x7x7xf32>, %s1b1db: tensor<256xf32>, %s1b1ng: tensor<256xf32>, %s1b1nbt: tensor<256xf32>, %s1b1eW: tensor<1024x256x1x1xf32>, %s1b1eb: tensor<1024xf32>, %s1b1pW: tensor<256x1024x1x1xf32>, %s1b1pb: tensor<256xf32>, %s1b1lg: tensor<256xf32>, %s1b2dW: tensor<256x1x7x7xf32>, %s1b2db: tensor<256xf32>, %s1b2ng: tensor<256xf32>, %s1b2nbt: tensor<256xf32>, %s1b2eW: tensor<1024x256x1x1xf32>, %s1b2eb: tensor<1024xf32>, %s1b2pW: tensor<256x1024x1x1xf32>, %s1b2pb: tensor<256xf32>, %s1b2lg: tensor<256xf32>, %d1ng: tensor<256xf32>, %d1nbt: tensor<256xf32>, %d1W: tensor<512x256x2x2xf32>, %d1b: tensor<512xf32>, %s2b0dW: tensor<512x1x7x7xf32>, %s2b0db: tensor<512xf32>, %s2b0ng: tensor<512xf32>, %s2b0nbt: tensor<512xf32>, %s2b0eW: tensor<2048x512x1x1xf32>, %s2b0eb: tensor<2048xf32>, %s2b0pW: tensor<512x2048x1x1xf32>, %s2b0pb: tensor<512xf32>, %s2b0lg: tensor<512xf32>, %s2b1dW: tensor<512x1x7x7xf32>, %s2b1db: tensor<512xf32>, %s2b1ng: tensor<512xf32>, %s2b1nbt: tensor<512xf32>, %s2b1eW: tensor<2048x512x1x1xf32>, %s2b1eb: tensor<2048xf32>, %s2b1pW: tensor<512x2048x1x1xf32>, %s2b1pb: tensor<512xf32>, %s2b1lg: tensor<512xf32>, %s2b2dW: tensor<512x1x7x7xf32>, %s2b2db: tensor<512xf32>, %s2b2ng: tensor<512xf32>, %s2b2nbt: tensor<512xf32>, %s2b2eW: tensor<2048x512x1x1xf32>, %s2b2eb: tensor<2048xf32>, %s2b2pW: tensor<512x2048x1x1xf32>, %s2b2pb: tensor<512xf32>, %s2b2lg: tensor<512xf32>, %s2b3dW: tensor<512x1x7x7xf32>, %s2b3db: tensor<512xf32>, %s2b3ng: tensor<512xf32>, %s2b3nbt: tensor<512xf32>, %s2b3eW: tensor<2048x512x1x1xf32>, %s2b3eb: tensor<2048xf32>, %s2b3pW: tensor<512x2048x1x1xf32>, %s2b3pb: tensor<512xf32>, %s2b3lg: tensor<512xf32>, %s2b4dW: tensor<512x1x7x7xf32>, %s2b4db: tensor<512xf32>, %s2b4ng: tensor<512xf32>, %s2b4nbt: tensor<512xf32>, %s2b4eW: tensor<2048x512x1x1xf32>, %s2b4eb: tensor<2048xf32>, %s2b4pW: tensor<512x2048x1x1xf32>, %s2b4pb: tensor<512xf32>, %s2b4lg: tensor<512xf32>, %s2b5dW: tensor<512x1x7x7xf32>, %s2b5db: tensor<512xf32>, %s2b5ng: tensor<512xf32>, %s2b5nbt: tensor<512xf32>, %s2b5eW: tensor<2048x512x1x1xf32>, %s2b5eb: tensor<2048xf32>, %s2b5pW: tensor<512x2048x1x1xf32>, %s2b5pb: tensor<512xf32>, %s2b5lg: tensor<512xf32>, %s2b6dW: tensor<512x1x7x7xf32>, %s2b6db: tensor<512xf32>, %s2b6ng: tensor<512xf32>, %s2b6nbt: tensor<512xf32>, %s2b6eW: tensor<2048x512x1x1xf32>, %s2b6eb: tensor<2048xf32>, %s2b6pW: tensor<512x2048x1x1xf32>, %s2b6pb: tensor<512xf32>, %s2b6lg: tensor<512xf32>, %s2b7dW: tensor<512x1x7x7xf32>, %s2b7db: tensor<512xf32>, %s2b7ng: tensor<512xf32>, %s2b7nbt: tensor<512xf32>, %s2b7eW: tensor<2048x512x1x1xf32>, %s2b7eb: tensor<2048xf32>, %s2b7pW: tensor<512x2048x1x1xf32>, %s2b7pb: tensor<512xf32>, %s2b7lg: tensor<512xf32>, %s2b8dW: tensor<512x1x7x7xf32>, %s2b8db: tensor<512xf32>, %s2b8ng: tensor<512xf32>, %s2b8nbt: tensor<512xf32>, %s2b8eW: tensor<2048x512x1x1xf32>, %s2b8eb: tensor<2048xf32>, %s2b8pW: tensor<512x2048x1x1xf32>, %s2b8pb: tensor<512xf32>, %s2b8lg: tensor<512xf32>, %s2b9dW: tensor<512x1x7x7xf32>, %s2b9db: tensor<512xf32>, %s2b9ng: tensor<512xf32>, %s2b9nbt: tensor<512xf32>, %s2b9eW: tensor<2048x512x1x1xf32>, %s2b9eb: tensor<2048xf32>, %s2b9pW: tensor<512x2048x1x1xf32>, %s2b9pb: tensor<512xf32>, %s2b9lg: tensor<512xf32>, %s2b10dW: tensor<512x1x7x7xf32>, %s2b10db: tensor<512xf32>, %s2b10ng: tensor<512xf32>, %s2b10nbt: tensor<512xf32>, %s2b10eW: tensor<2048x512x1x1xf32>, %s2b10eb: tensor<2048xf32>, %s2b10pW: tensor<512x2048x1x1xf32>, %s2b10pb: tensor<512xf32>, %s2b10lg: tensor<512xf32>, %s2b11dW: tensor<512x1x7x7xf32>, %s2b11db: tensor<512xf32>, %s2b11ng: tensor<512xf32>, %s2b11nbt: tensor<512xf32>, %s2b11eW: tensor<2048x512x1x1xf32>, %s2b11eb: tensor<2048xf32>, %s2b11pW: tensor<512x2048x1x1xf32>, %s2b11pb: tensor<512xf32>, %s2b11lg: tensor<512xf32>, %s2b12dW: tensor<512x1x7x7xf32>, %s2b12db: tensor<512xf32>, %s2b12ng: tensor<512xf32>, %s2b12nbt: tensor<512xf32>, %s2b12eW: tensor<2048x512x1x1xf32>, %s2b12eb: tensor<2048xf32>, %s2b12pW: tensor<512x2048x1x1xf32>, %s2b12pb: tensor<512xf32>, %s2b12lg: tensor<512xf32>, %s2b13dW: tensor<512x1x7x7xf32>, %s2b13db: tensor<512xf32>, %s2b13ng: tensor<512xf32>, %s2b13nbt: tensor<512xf32>, %s2b13eW: tensor<2048x512x1x1xf32>, %s2b13eb: tensor<2048xf32>, %s2b13pW: tensor<512x2048x1x1xf32>, %s2b13pb: tensor<512xf32>, %s2b13lg: tensor<512xf32>, %s2b14dW: tensor<512x1x7x7xf32>, %s2b14db: tensor<512xf32>, %s2b14ng: tensor<512xf32>, %s2b14nbt: tensor<512xf32>, %s2b14eW: tensor<2048x512x1x1xf32>, %s2b14eb: tensor<2048xf32>, %s2b14pW: tensor<512x2048x1x1xf32>, %s2b14pb: tensor<512xf32>, %s2b14lg: tensor<512xf32>, %s2b15dW: tensor<512x1x7x7xf32>, %s2b15db: tensor<512xf32>, %s2b15ng: tensor<512xf32>, %s2b15nbt: tensor<512xf32>, %s2b15eW: tensor<2048x512x1x1xf32>, %s2b15eb: tensor<2048xf32>, %s2b15pW: tensor<512x2048x1x1xf32>, %s2b15pb: tensor<512xf32>, %s2b15lg: tensor<512xf32>, %s2b16dW: tensor<512x1x7x7xf32>, %s2b16db: tensor<512xf32>, %s2b16ng: tensor<512xf32>, %s2b16nbt: tensor<512xf32>, %s2b16eW: tensor<2048x512x1x1xf32>, %s2b16eb: tensor<2048xf32>, %s2b16pW: tensor<512x2048x1x1xf32>, %s2b16pb: tensor<512xf32>, %s2b16lg: tensor<512xf32>, %s2b17dW: tensor<512x1x7x7xf32>, %s2b17db: tensor<512xf32>, %s2b17ng: tensor<512xf32>, %s2b17nbt: tensor<512xf32>, %s2b17eW: tensor<2048x512x1x1xf32>, %s2b17eb: tensor<2048xf32>, %s2b17pW: tensor<512x2048x1x1xf32>, %s2b17pb: tensor<512xf32>, %s2b17lg: tensor<512xf32>, %s2b18dW: tensor<512x1x7x7xf32>, %s2b18db: tensor<512xf32>, %s2b18ng: tensor<512xf32>, %s2b18nbt: tensor<512xf32>, %s2b18eW: tensor<2048x512x1x1xf32>, %s2b18eb: tensor<2048xf32>, %s2b18pW: tensor<512x2048x1x1xf32>, %s2b18pb: tensor<512xf32>, %s2b18lg: tensor<512xf32>, %s2b19dW: tensor<512x1x7x7xf32>, %s2b19db: tensor<512xf32>, %s2b19ng: tensor<512xf32>, %s2b19nbt: tensor<512xf32>, %s2b19eW: tensor<2048x512x1x1xf32>, %s2b19eb: tensor<2048xf32>, %s2b19pW: tensor<512x2048x1x1xf32>, %s2b19pb: tensor<512xf32>, %s2b19lg: tensor<512xf32>, %s2b20dW: tensor<512x1x7x7xf32>, %s2b20db: tensor<512xf32>, %s2b20ng: tensor<512xf32>, %s2b20nbt: tensor<512xf32>, %s2b20eW: tensor<2048x512x1x1xf32>, %s2b20eb: tensor<2048xf32>, %s2b20pW: tensor<512x2048x1x1xf32>, %s2b20pb: tensor<512xf32>, %s2b20lg: tensor<512xf32>, %s2b21dW: tensor<512x1x7x7xf32>, %s2b21db: tensor<512xf32>, %s2b21ng: tensor<512xf32>, %s2b21nbt: tensor<512xf32>, %s2b21eW: tensor<2048x512x1x1xf32>, %s2b21eb: tensor<2048xf32>, %s2b21pW: tensor<512x2048x1x1xf32>, %s2b21pb: tensor<512xf32>, %s2b21lg: tensor<512xf32>, %s2b22dW: tensor<512x1x7x7xf32>, %s2b22db: tensor<512xf32>, %s2b22ng: tensor<512xf32>, %s2b22nbt: tensor<512xf32>, %s2b22eW: tensor<2048x512x1x1xf32>, %s2b22eb: tensor<2048xf32>, %s2b22pW: tensor<512x2048x1x1xf32>, %s2b22pb: tensor<512xf32>, %s2b22lg: tensor<512xf32>, %s2b23dW: tensor<512x1x7x7xf32>, %s2b23db: tensor<512xf32>, %s2b23ng: tensor<512xf32>, %s2b23nbt: tensor<512xf32>, %s2b23eW: tensor<2048x512x1x1xf32>, %s2b23eb: tensor<2048xf32>, %s2b23pW: tensor<512x2048x1x1xf32>, %s2b23pb: tensor<512xf32>, %s2b23lg: tensor<512xf32>, %s2b24dW: tensor<512x1x7x7xf32>, %s2b24db: tensor<512xf32>, %s2b24ng: tensor<512xf32>, %s2b24nbt: tensor<512xf32>, %s2b24eW: tensor<2048x512x1x1xf32>, %s2b24eb: tensor<2048xf32>, %s2b24pW: tensor<512x2048x1x1xf32>, %s2b24pb: tensor<512xf32>, %s2b24lg: tensor<512xf32>, %s2b25dW: tensor<512x1x7x7xf32>, %s2b25db: tensor<512xf32>, %s2b25ng: tensor<512xf32>, %s2b25nbt: tensor<512xf32>, %s2b25eW: tensor<2048x512x1x1xf32>, %s2b25eb: tensor<2048xf32>, %s2b25pW: tensor<512x2048x1x1xf32>, %s2b25pb: tensor<512xf32>, %s2b25lg: tensor<512xf32>, %s2b26dW: tensor<512x1x7x7xf32>, %s2b26db: tensor<512xf32>, %s2b26ng: tensor<512xf32>, %s2b26nbt: tensor<512xf32>, %s2b26eW: tensor<2048x512x1x1xf32>, %s2b26eb: tensor<2048xf32>, %s2b26pW: tensor<512x2048x1x1xf32>, %s2b26pb: tensor<512xf32>, %s2b26lg: tensor<512xf32>, %d2ng: tensor<512xf32>, %d2nbt: tensor<512xf32>, %d2W: tensor<1024x512x2x2xf32>, %d2b: tensor<1024xf32>, %s3b0dW: tensor<1024x1x7x7xf32>, %s3b0db: tensor<1024xf32>, %s3b0ng: tensor<1024xf32>, %s3b0nbt: tensor<1024xf32>, %s3b0eW: tensor<4096x1024x1x1xf32>, %s3b0eb: tensor<4096xf32>, %s3b0pW: tensor<1024x4096x1x1xf32>, %s3b0pb: tensor<1024xf32>, %s3b0lg: tensor<1024xf32>, %s3b1dW: tensor<1024x1x7x7xf32>, %s3b1db: tensor<1024xf32>, %s3b1ng: tensor<1024xf32>, %s3b1nbt: tensor<1024xf32>, %s3b1eW: tensor<4096x1024x1x1xf32>, %s3b1eb: tensor<4096xf32>, %s3b1pW: tensor<1024x4096x1x1xf32>, %s3b1pb: tensor<1024xf32>, %s3b1lg: tensor<1024xf32>, %s3b2dW: tensor<1024x1x7x7xf32>, %s3b2db: tensor<1024xf32>, %s3b2ng: tensor<1024xf32>, %s3b2nbt: tensor<1024xf32>, %s3b2eW: tensor<4096x1024x1x1xf32>, %s3b2eb: tensor<4096xf32>, %s3b2pW: tensor<1024x4096x1x1xf32>, %s3b2pb: tensor<1024xf32>, %s3b2lg: tensor<1024xf32>, %Wd: tensor<1024x1000xf32>, %bd: tensor<1000xf32>) -> tensor<32x1000xf32> {
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
    %v83 = stablehlo.multiply %v82, %v82 : tensor<32x1605632xf32>
    %v84 = stablehlo.multiply %v83, %v82 : tensor<32x1605632xf32>
    %v85 = stablehlo.constant dense<0.044715> : tensor<32x1605632xf32>
    %v86 = stablehlo.multiply %v85, %v84 : tensor<32x1605632xf32>
    %v87 = stablehlo.add %v82, %v86 : tensor<32x1605632xf32>
    %v88 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1605632xf32>
    %v89 = stablehlo.multiply %v88, %v87 : tensor<32x1605632xf32>
    %v90 = stablehlo.tanh %v89 : tensor<32x1605632xf32>
    %v91 = stablehlo.constant dense<1.0> : tensor<32x1605632xf32>
    %v92 = stablehlo.add %v91, %v90 : tensor<32x1605632xf32>
    %v93 = stablehlo.constant dense<0.5> : tensor<32x1605632xf32>
    %v94 = stablehlo.multiply %v93, %v82 : tensor<32x1605632xf32>
    %v95 = stablehlo.multiply %v94, %v92 : tensor<32x1605632xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<32x1605632xf32>) -> tensor<32x512x56x56xf32>
    %v97 = stablehlo.convolution(%v96, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x56x56xf32>, tensor<128x512x1x1xf32>) -> tensor<32x128x56x56xf32>
    %v98 = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v99 = stablehlo.add %v97, %v98 : tensor<32x128x56x56xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v102 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v103 = stablehlo.multiply %v101, %v102 : tensor<32x128x56x56xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v105 = stablehlo.add %v104, %v38 : tensor<32x401408xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v107 = stablehlo.convolution(%v106, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x56x56xf32>, tensor<128x1x7x7xf32>) -> tensor<32x128x56x56xf32>
    %v108 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v109 = stablehlo.add %v107, %v108 : tensor<32x128x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v112 = stablehlo.transpose %v111, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v116 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v117 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v118 = stablehlo.reduce(%v114 init: %v115) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v119 = stablehlo.broadcast_in_dim %v118, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v120 = stablehlo.divide %v119, %v116 : tensor<32x3136x128xf32>
    %v121 = stablehlo.subtract %v114, %v120 : tensor<32x3136x128xf32>
    %v122 = stablehlo.multiply %v121, %v121 : tensor<32x3136x128xf32>
    %v123 = stablehlo.reduce(%v122 init: %v115) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v124 = stablehlo.broadcast_in_dim %v123, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v125 = stablehlo.divide %v124, %v116 : tensor<32x3136x128xf32>
    %v126 = stablehlo.add %v125, %v117 : tensor<32x3136x128xf32>
    %v127 = stablehlo.rsqrt %v126 : tensor<32x3136x128xf32>
    %v128 = stablehlo.multiply %v121, %v127 : tensor<32x3136x128xf32>
    %v129 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v130 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v131 = stablehlo.multiply %v128, %v129 : tensor<32x3136x128xf32>
    %v132 = stablehlo.add %v131, %v130 : tensor<32x3136x128xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v134 = stablehlo.reshape %v133 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v135 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v136 = stablehlo.multiply %v134, %v135 : tensor<32x3136x128xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v139 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v140 = stablehlo.add %v138, %v139 : tensor<32x3136x128xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v143 = stablehlo.transpose %v142, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v146 = stablehlo.convolution(%v145, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<512x128x1x1xf32>) -> tensor<32x512x56x56xf32>
    %v147 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x56x56xf32>
    %v148 = stablehlo.add %v146, %v147 : tensor<32x512x56x56xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x512x56x56xf32>) -> tensor<32x1605632xf32>
    %v150 = stablehlo.multiply %v149, %v149 : tensor<32x1605632xf32>
    %v151 = stablehlo.multiply %v150, %v149 : tensor<32x1605632xf32>
    %v152 = stablehlo.constant dense<0.044715> : tensor<32x1605632xf32>
    %v153 = stablehlo.multiply %v152, %v151 : tensor<32x1605632xf32>
    %v154 = stablehlo.add %v149, %v153 : tensor<32x1605632xf32>
    %v155 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1605632xf32>
    %v156 = stablehlo.multiply %v155, %v154 : tensor<32x1605632xf32>
    %v157 = stablehlo.tanh %v156 : tensor<32x1605632xf32>
    %v158 = stablehlo.constant dense<1.0> : tensor<32x1605632xf32>
    %v159 = stablehlo.add %v158, %v157 : tensor<32x1605632xf32>
    %v160 = stablehlo.constant dense<0.5> : tensor<32x1605632xf32>
    %v161 = stablehlo.multiply %v160, %v149 : tensor<32x1605632xf32>
    %v162 = stablehlo.multiply %v161, %v159 : tensor<32x1605632xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x1605632xf32>) -> tensor<32x512x56x56xf32>
    %v164 = stablehlo.convolution(%v163, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x56x56xf32>, tensor<128x512x1x1xf32>) -> tensor<32x128x56x56xf32>
    %v165 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v166 = stablehlo.add %v164, %v165 : tensor<32x128x56x56xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v169 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v170 = stablehlo.multiply %v168, %v169 : tensor<32x128x56x56xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v172 = stablehlo.add %v171, %v105 : tensor<32x401408xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v174 = stablehlo.convolution(%v173, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x56x56xf32>, tensor<128x1x7x7xf32>) -> tensor<32x128x56x56xf32>
    %v175 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v176 = stablehlo.add %v174, %v175 : tensor<32x128x56x56xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v179 = stablehlo.transpose %v178, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v183 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v184 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v185 = stablehlo.reduce(%v181 init: %v182) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v186 = stablehlo.broadcast_in_dim %v185, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v187 = stablehlo.divide %v186, %v183 : tensor<32x3136x128xf32>
    %v188 = stablehlo.subtract %v181, %v187 : tensor<32x3136x128xf32>
    %v189 = stablehlo.multiply %v188, %v188 : tensor<32x3136x128xf32>
    %v190 = stablehlo.reduce(%v189 init: %v182) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v191 = stablehlo.broadcast_in_dim %v190, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v192 = stablehlo.divide %v191, %v183 : tensor<32x3136x128xf32>
    %v193 = stablehlo.add %v192, %v184 : tensor<32x3136x128xf32>
    %v194 = stablehlo.rsqrt %v193 : tensor<32x3136x128xf32>
    %v195 = stablehlo.multiply %v188, %v194 : tensor<32x3136x128xf32>
    %v196 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v197 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v198 = stablehlo.multiply %v195, %v196 : tensor<32x3136x128xf32>
    %v199 = stablehlo.add %v198, %v197 : tensor<32x3136x128xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v202 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v203 = stablehlo.multiply %v201, %v202 : tensor<32x3136x128xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v206 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v207 = stablehlo.add %v205, %v206 : tensor<32x3136x128xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v210 = stablehlo.transpose %v209, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v213 = stablehlo.convolution(%v212, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<512x128x1x1xf32>) -> tensor<32x512x56x56xf32>
    %v214 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x56x56xf32>
    %v215 = stablehlo.add %v213, %v214 : tensor<32x512x56x56xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x512x56x56xf32>) -> tensor<32x1605632xf32>
    %v217 = stablehlo.multiply %v216, %v216 : tensor<32x1605632xf32>
    %v218 = stablehlo.multiply %v217, %v216 : tensor<32x1605632xf32>
    %v219 = stablehlo.constant dense<0.044715> : tensor<32x1605632xf32>
    %v220 = stablehlo.multiply %v219, %v218 : tensor<32x1605632xf32>
    %v221 = stablehlo.add %v216, %v220 : tensor<32x1605632xf32>
    %v222 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1605632xf32>
    %v223 = stablehlo.multiply %v222, %v221 : tensor<32x1605632xf32>
    %v224 = stablehlo.tanh %v223 : tensor<32x1605632xf32>
    %v225 = stablehlo.constant dense<1.0> : tensor<32x1605632xf32>
    %v226 = stablehlo.add %v225, %v224 : tensor<32x1605632xf32>
    %v227 = stablehlo.constant dense<0.5> : tensor<32x1605632xf32>
    %v228 = stablehlo.multiply %v227, %v216 : tensor<32x1605632xf32>
    %v229 = stablehlo.multiply %v228, %v226 : tensor<32x1605632xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x1605632xf32>) -> tensor<32x512x56x56xf32>
    %v231 = stablehlo.convolution(%v230, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x56x56xf32>, tensor<128x512x1x1xf32>) -> tensor<32x128x56x56xf32>
    %v232 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v233 = stablehlo.add %v231, %v232 : tensor<32x128x56x56xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v236 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v237 = stablehlo.multiply %v235, %v236 : tensor<32x128x56x56xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v239 = stablehlo.add %v238, %v172 : tensor<32x401408xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x401408xf32>) -> tensor<32x128x3136xf32>
    %v241 = stablehlo.transpose %v240, dims = [0, 2, 1] : (tensor<32x128x3136xf32>) -> tensor<32x3136x128xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v245 = stablehlo.constant dense<128.0> : tensor<32x3136x128xf32>
    %v246 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x128xf32>
    %v247 = stablehlo.reduce(%v243 init: %v244) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v248 = stablehlo.broadcast_in_dim %v247, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v249 = stablehlo.divide %v248, %v245 : tensor<32x3136x128xf32>
    %v250 = stablehlo.subtract %v243, %v249 : tensor<32x3136x128xf32>
    %v251 = stablehlo.multiply %v250, %v250 : tensor<32x3136x128xf32>
    %v252 = stablehlo.reduce(%v251 init: %v244) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x128xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v253 = stablehlo.broadcast_in_dim %v252, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x128xf32>
    %v254 = stablehlo.divide %v253, %v245 : tensor<32x3136x128xf32>
    %v255 = stablehlo.add %v254, %v246 : tensor<32x3136x128xf32>
    %v256 = stablehlo.rsqrt %v255 : tensor<32x3136x128xf32>
    %v257 = stablehlo.multiply %v250, %v256 : tensor<32x3136x128xf32>
    %v258 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v259 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x128xf32>
    %v260 = stablehlo.multiply %v257, %v258 : tensor<32x3136x128xf32>
    %v261 = stablehlo.add %v260, %v259 : tensor<32x3136x128xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v264 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v265 = stablehlo.multiply %v263, %v264 : tensor<32x3136x128xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v268 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<128xf32>) -> tensor<32x3136x128xf32>
    %v269 = stablehlo.add %v267, %v268 : tensor<32x3136x128xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<32x3136x128xf32>) -> tensor<32x401408xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x401408xf32>) -> tensor<32x3136x128xf32>
    %v272 = stablehlo.transpose %v271, dims = [0, 2, 1] : (tensor<32x3136x128xf32>) -> tensor<32x128x3136xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<32x128x3136xf32>) -> tensor<32x401408xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v275 = stablehlo.convolution(%v274, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<256x128x2x2xf32>) -> tensor<32x256x28x28xf32>
    %v276 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v277 = stablehlo.add %v275, %v276 : tensor<32x256x28x28xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v280 = stablehlo.convolution(%v279, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x28x28xf32>, tensor<256x1x7x7xf32>) -> tensor<32x256x28x28xf32>
    %v281 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v282 = stablehlo.add %v280, %v281 : tensor<32x256x28x28xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x200704xf32>) -> tensor<32x256x784xf32>
    %v285 = stablehlo.transpose %v284, dims = [0, 2, 1] : (tensor<32x256x784xf32>) -> tensor<32x784x256xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v289 = stablehlo.constant dense<256.0> : tensor<32x784x256xf32>
    %v290 = stablehlo.constant dense<1.0e-6> : tensor<32x784x256xf32>
    %v291 = stablehlo.reduce(%v287 init: %v288) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v292 = stablehlo.broadcast_in_dim %v291, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v293 = stablehlo.divide %v292, %v289 : tensor<32x784x256xf32>
    %v294 = stablehlo.subtract %v287, %v293 : tensor<32x784x256xf32>
    %v295 = stablehlo.multiply %v294, %v294 : tensor<32x784x256xf32>
    %v296 = stablehlo.reduce(%v295 init: %v288) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v297 = stablehlo.broadcast_in_dim %v296, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v298 = stablehlo.divide %v297, %v289 : tensor<32x784x256xf32>
    %v299 = stablehlo.add %v298, %v290 : tensor<32x784x256xf32>
    %v300 = stablehlo.rsqrt %v299 : tensor<32x784x256xf32>
    %v301 = stablehlo.multiply %v294, %v300 : tensor<32x784x256xf32>
    %v302 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v303 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v304 = stablehlo.multiply %v301, %v302 : tensor<32x784x256xf32>
    %v305 = stablehlo.add %v304, %v303 : tensor<32x784x256xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v308 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v309 = stablehlo.multiply %v307, %v308 : tensor<32x784x256xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v312 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v313 = stablehlo.add %v311, %v312 : tensor<32x784x256xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v316 = stablehlo.transpose %v315, dims = [0, 2, 1] : (tensor<32x784x256xf32>) -> tensor<32x256x784xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<32x256x784xf32>) -> tensor<32x200704xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v319 = stablehlo.convolution(%v318, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x28x28xf32>
    %v320 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x28x28xf32>
    %v321 = stablehlo.add %v319, %v320 : tensor<32x1024x28x28xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<32x1024x28x28xf32>) -> tensor<32x802816xf32>
    %v323 = stablehlo.multiply %v322, %v322 : tensor<32x802816xf32>
    %v324 = stablehlo.multiply %v323, %v322 : tensor<32x802816xf32>
    %v325 = stablehlo.constant dense<0.044715> : tensor<32x802816xf32>
    %v326 = stablehlo.multiply %v325, %v324 : tensor<32x802816xf32>
    %v327 = stablehlo.add %v322, %v326 : tensor<32x802816xf32>
    %v328 = stablehlo.constant dense<0.7978845608028654> : tensor<32x802816xf32>
    %v329 = stablehlo.multiply %v328, %v327 : tensor<32x802816xf32>
    %v330 = stablehlo.tanh %v329 : tensor<32x802816xf32>
    %v331 = stablehlo.constant dense<1.0> : tensor<32x802816xf32>
    %v332 = stablehlo.add %v331, %v330 : tensor<32x802816xf32>
    %v333 = stablehlo.constant dense<0.5> : tensor<32x802816xf32>
    %v334 = stablehlo.multiply %v333, %v322 : tensor<32x802816xf32>
    %v335 = stablehlo.multiply %v334, %v332 : tensor<32x802816xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x802816xf32>) -> tensor<32x1024x28x28xf32>
    %v337 = stablehlo.convolution(%v336, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x28x28xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x28x28xf32>
    %v338 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v339 = stablehlo.add %v337, %v338 : tensor<32x256x28x28xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v342 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v343 = stablehlo.multiply %v341, %v342 : tensor<32x256x28x28xf32>
    %v344 = stablehlo.reshape %v343 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v345 = stablehlo.add %v344, %v278 : tensor<32x200704xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v347 = stablehlo.convolution(%v346, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x28x28xf32>, tensor<256x1x7x7xf32>) -> tensor<32x256x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v349 = stablehlo.add %v347, %v348 : tensor<32x256x28x28xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x200704xf32>) -> tensor<32x256x784xf32>
    %v352 = stablehlo.transpose %v351, dims = [0, 2, 1] : (tensor<32x256x784xf32>) -> tensor<32x784x256xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v356 = stablehlo.constant dense<256.0> : tensor<32x784x256xf32>
    %v357 = stablehlo.constant dense<1.0e-6> : tensor<32x784x256xf32>
    %v358 = stablehlo.reduce(%v354 init: %v355) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v359 = stablehlo.broadcast_in_dim %v358, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v360 = stablehlo.divide %v359, %v356 : tensor<32x784x256xf32>
    %v361 = stablehlo.subtract %v354, %v360 : tensor<32x784x256xf32>
    %v362 = stablehlo.multiply %v361, %v361 : tensor<32x784x256xf32>
    %v363 = stablehlo.reduce(%v362 init: %v355) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v364 = stablehlo.broadcast_in_dim %v363, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v365 = stablehlo.divide %v364, %v356 : tensor<32x784x256xf32>
    %v366 = stablehlo.add %v365, %v357 : tensor<32x784x256xf32>
    %v367 = stablehlo.rsqrt %v366 : tensor<32x784x256xf32>
    %v368 = stablehlo.multiply %v361, %v367 : tensor<32x784x256xf32>
    %v369 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v370 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v371 = stablehlo.multiply %v368, %v369 : tensor<32x784x256xf32>
    %v372 = stablehlo.add %v371, %v370 : tensor<32x784x256xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v375 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v376 = stablehlo.multiply %v374, %v375 : tensor<32x784x256xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v379 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v380 = stablehlo.add %v378, %v379 : tensor<32x784x256xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v383 = stablehlo.transpose %v382, dims = [0, 2, 1] : (tensor<32x784x256xf32>) -> tensor<32x256x784xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<32x256x784xf32>) -> tensor<32x200704xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v386 = stablehlo.convolution(%v385, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x28x28xf32>
    %v387 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x28x28xf32>
    %v388 = stablehlo.add %v386, %v387 : tensor<32x1024x28x28xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x1024x28x28xf32>) -> tensor<32x802816xf32>
    %v390 = stablehlo.multiply %v389, %v389 : tensor<32x802816xf32>
    %v391 = stablehlo.multiply %v390, %v389 : tensor<32x802816xf32>
    %v392 = stablehlo.constant dense<0.044715> : tensor<32x802816xf32>
    %v393 = stablehlo.multiply %v392, %v391 : tensor<32x802816xf32>
    %v394 = stablehlo.add %v389, %v393 : tensor<32x802816xf32>
    %v395 = stablehlo.constant dense<0.7978845608028654> : tensor<32x802816xf32>
    %v396 = stablehlo.multiply %v395, %v394 : tensor<32x802816xf32>
    %v397 = stablehlo.tanh %v396 : tensor<32x802816xf32>
    %v398 = stablehlo.constant dense<1.0> : tensor<32x802816xf32>
    %v399 = stablehlo.add %v398, %v397 : tensor<32x802816xf32>
    %v400 = stablehlo.constant dense<0.5> : tensor<32x802816xf32>
    %v401 = stablehlo.multiply %v400, %v389 : tensor<32x802816xf32>
    %v402 = stablehlo.multiply %v401, %v399 : tensor<32x802816xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x802816xf32>) -> tensor<32x1024x28x28xf32>
    %v404 = stablehlo.convolution(%v403, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x28x28xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x28x28xf32>
    %v405 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v406 = stablehlo.add %v404, %v405 : tensor<32x256x28x28xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v409 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v410 = stablehlo.multiply %v408, %v409 : tensor<32x256x28x28xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v412 = stablehlo.add %v411, %v345 : tensor<32x200704xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v414 = stablehlo.convolution(%v413, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x28x28xf32>, tensor<256x1x7x7xf32>) -> tensor<32x256x28x28xf32>
    %v415 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v416 = stablehlo.add %v414, %v415 : tensor<32x256x28x28xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<32x200704xf32>) -> tensor<32x256x784xf32>
    %v419 = stablehlo.transpose %v418, dims = [0, 2, 1] : (tensor<32x256x784xf32>) -> tensor<32x784x256xf32>
    %v420 = stablehlo.reshape %v419 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v423 = stablehlo.constant dense<256.0> : tensor<32x784x256xf32>
    %v424 = stablehlo.constant dense<1.0e-6> : tensor<32x784x256xf32>
    %v425 = stablehlo.reduce(%v421 init: %v422) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v426 = stablehlo.broadcast_in_dim %v425, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v427 = stablehlo.divide %v426, %v423 : tensor<32x784x256xf32>
    %v428 = stablehlo.subtract %v421, %v427 : tensor<32x784x256xf32>
    %v429 = stablehlo.multiply %v428, %v428 : tensor<32x784x256xf32>
    %v430 = stablehlo.reduce(%v429 init: %v422) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v431 = stablehlo.broadcast_in_dim %v430, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v432 = stablehlo.divide %v431, %v423 : tensor<32x784x256xf32>
    %v433 = stablehlo.add %v432, %v424 : tensor<32x784x256xf32>
    %v434 = stablehlo.rsqrt %v433 : tensor<32x784x256xf32>
    %v435 = stablehlo.multiply %v428, %v434 : tensor<32x784x256xf32>
    %v436 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v437 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v438 = stablehlo.multiply %v435, %v436 : tensor<32x784x256xf32>
    %v439 = stablehlo.add %v438, %v437 : tensor<32x784x256xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v442 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v443 = stablehlo.multiply %v441, %v442 : tensor<32x784x256xf32>
    %v444 = stablehlo.reshape %v443 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v446 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v447 = stablehlo.add %v445, %v446 : tensor<32x784x256xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v450 = stablehlo.transpose %v449, dims = [0, 2, 1] : (tensor<32x784x256xf32>) -> tensor<32x256x784xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x256x784xf32>) -> tensor<32x200704xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v453 = stablehlo.convolution(%v452, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x28x28xf32>
    %v454 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x28x28xf32>
    %v455 = stablehlo.add %v453, %v454 : tensor<32x1024x28x28xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<32x1024x28x28xf32>) -> tensor<32x802816xf32>
    %v457 = stablehlo.multiply %v456, %v456 : tensor<32x802816xf32>
    %v458 = stablehlo.multiply %v457, %v456 : tensor<32x802816xf32>
    %v459 = stablehlo.constant dense<0.044715> : tensor<32x802816xf32>
    %v460 = stablehlo.multiply %v459, %v458 : tensor<32x802816xf32>
    %v461 = stablehlo.add %v456, %v460 : tensor<32x802816xf32>
    %v462 = stablehlo.constant dense<0.7978845608028654> : tensor<32x802816xf32>
    %v463 = stablehlo.multiply %v462, %v461 : tensor<32x802816xf32>
    %v464 = stablehlo.tanh %v463 : tensor<32x802816xf32>
    %v465 = stablehlo.constant dense<1.0> : tensor<32x802816xf32>
    %v466 = stablehlo.add %v465, %v464 : tensor<32x802816xf32>
    %v467 = stablehlo.constant dense<0.5> : tensor<32x802816xf32>
    %v468 = stablehlo.multiply %v467, %v456 : tensor<32x802816xf32>
    %v469 = stablehlo.multiply %v468, %v466 : tensor<32x802816xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x802816xf32>) -> tensor<32x1024x28x28xf32>
    %v471 = stablehlo.convolution(%v470, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x28x28xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x28x28xf32>
    %v472 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v473 = stablehlo.add %v471, %v472 : tensor<32x256x28x28xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v476 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v477 = stablehlo.multiply %v475, %v476 : tensor<32x256x28x28xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v479 = stablehlo.add %v478, %v412 : tensor<32x200704xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x200704xf32>) -> tensor<32x256x784xf32>
    %v481 = stablehlo.transpose %v480, dims = [0, 2, 1] : (tensor<32x256x784xf32>) -> tensor<32x784x256xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v485 = stablehlo.constant dense<256.0> : tensor<32x784x256xf32>
    %v486 = stablehlo.constant dense<1.0e-6> : tensor<32x784x256xf32>
    %v487 = stablehlo.reduce(%v483 init: %v484) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v488 = stablehlo.broadcast_in_dim %v487, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v489 = stablehlo.divide %v488, %v485 : tensor<32x784x256xf32>
    %v490 = stablehlo.subtract %v483, %v489 : tensor<32x784x256xf32>
    %v491 = stablehlo.multiply %v490, %v490 : tensor<32x784x256xf32>
    %v492 = stablehlo.reduce(%v491 init: %v484) applies stablehlo.add across dimensions = [2] : (tensor<32x784x256xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v493 = stablehlo.broadcast_in_dim %v492, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x256xf32>
    %v494 = stablehlo.divide %v493, %v485 : tensor<32x784x256xf32>
    %v495 = stablehlo.add %v494, %v486 : tensor<32x784x256xf32>
    %v496 = stablehlo.rsqrt %v495 : tensor<32x784x256xf32>
    %v497 = stablehlo.multiply %v490, %v496 : tensor<32x784x256xf32>
    %v498 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v499 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x256xf32>
    %v500 = stablehlo.multiply %v497, %v498 : tensor<32x784x256xf32>
    %v501 = stablehlo.add %v500, %v499 : tensor<32x784x256xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v504 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v505 = stablehlo.multiply %v503, %v504 : tensor<32x784x256xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v508 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<256xf32>) -> tensor<32x784x256xf32>
    %v509 = stablehlo.add %v507, %v508 : tensor<32x784x256xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x784x256xf32>) -> tensor<32x200704xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x200704xf32>) -> tensor<32x784x256xf32>
    %v512 = stablehlo.transpose %v511, dims = [0, 2, 1] : (tensor<32x784x256xf32>) -> tensor<32x256x784xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<32x256x784xf32>) -> tensor<32x200704xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v515 = stablehlo.convolution(%v514, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<512x256x2x2xf32>) -> tensor<32x512x14x14xf32>
    %v516 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v517 = stablehlo.add %v515, %v516 : tensor<32x512x14x14xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v520 = stablehlo.convolution(%v519, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v521 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v522 = stablehlo.add %v520, %v521 : tensor<32x512x14x14xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v524 = stablehlo.reshape %v523 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v525 = stablehlo.transpose %v524, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v527 = stablehlo.reshape %v526 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v529 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v530 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v531 = stablehlo.reduce(%v527 init: %v528) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v532 = stablehlo.broadcast_in_dim %v531, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v533 = stablehlo.divide %v532, %v529 : tensor<32x196x512xf32>
    %v534 = stablehlo.subtract %v527, %v533 : tensor<32x196x512xf32>
    %v535 = stablehlo.multiply %v534, %v534 : tensor<32x196x512xf32>
    %v536 = stablehlo.reduce(%v535 init: %v528) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v537 = stablehlo.broadcast_in_dim %v536, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v538 = stablehlo.divide %v537, %v529 : tensor<32x196x512xf32>
    %v539 = stablehlo.add %v538, %v530 : tensor<32x196x512xf32>
    %v540 = stablehlo.rsqrt %v539 : tensor<32x196x512xf32>
    %v541 = stablehlo.multiply %v534, %v540 : tensor<32x196x512xf32>
    %v542 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v543 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v544 = stablehlo.multiply %v541, %v542 : tensor<32x196x512xf32>
    %v545 = stablehlo.add %v544, %v543 : tensor<32x196x512xf32>
    %v546 = stablehlo.reshape %v545 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v548 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v549 = stablehlo.multiply %v547, %v548 : tensor<32x196x512xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v552 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v553 = stablehlo.add %v551, %v552 : tensor<32x196x512xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v556 = stablehlo.transpose %v555, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v559 = stablehlo.convolution(%v558, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v560 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v561 = stablehlo.add %v559, %v560 : tensor<32x2048x14x14xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v563 = stablehlo.multiply %v562, %v562 : tensor<32x401408xf32>
    %v564 = stablehlo.multiply %v563, %v562 : tensor<32x401408xf32>
    %v565 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v566 = stablehlo.multiply %v565, %v564 : tensor<32x401408xf32>
    %v567 = stablehlo.add %v562, %v566 : tensor<32x401408xf32>
    %v568 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v569 = stablehlo.multiply %v568, %v567 : tensor<32x401408xf32>
    %v570 = stablehlo.tanh %v569 : tensor<32x401408xf32>
    %v571 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v572 = stablehlo.add %v571, %v570 : tensor<32x401408xf32>
    %v573 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v574 = stablehlo.multiply %v573, %v562 : tensor<32x401408xf32>
    %v575 = stablehlo.multiply %v574, %v572 : tensor<32x401408xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v577 = stablehlo.convolution(%v576, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<32x512x14x14xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v582 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v583 = stablehlo.multiply %v581, %v582 : tensor<32x512x14x14xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v585 = stablehlo.add %v584, %v518 : tensor<32x100352xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v587 = stablehlo.convolution(%v586, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v588 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v589 = stablehlo.add %v587, %v588 : tensor<32x512x14x14xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v592 = stablehlo.transpose %v591, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v595 = stablehlo.constant dense<0.0> : tensor<f32>
    %v596 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v597 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v598 = stablehlo.reduce(%v594 init: %v595) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v599 = stablehlo.broadcast_in_dim %v598, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v600 = stablehlo.divide %v599, %v596 : tensor<32x196x512xf32>
    %v601 = stablehlo.subtract %v594, %v600 : tensor<32x196x512xf32>
    %v602 = stablehlo.multiply %v601, %v601 : tensor<32x196x512xf32>
    %v603 = stablehlo.reduce(%v602 init: %v595) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v604 = stablehlo.broadcast_in_dim %v603, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v605 = stablehlo.divide %v604, %v596 : tensor<32x196x512xf32>
    %v606 = stablehlo.add %v605, %v597 : tensor<32x196x512xf32>
    %v607 = stablehlo.rsqrt %v606 : tensor<32x196x512xf32>
    %v608 = stablehlo.multiply %v601, %v607 : tensor<32x196x512xf32>
    %v609 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v610 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v611 = stablehlo.multiply %v608, %v609 : tensor<32x196x512xf32>
    %v612 = stablehlo.add %v611, %v610 : tensor<32x196x512xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v614 = stablehlo.reshape %v613 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v615 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v616 = stablehlo.multiply %v614, %v615 : tensor<32x196x512xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v619 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v620 = stablehlo.add %v618, %v619 : tensor<32x196x512xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v622 = stablehlo.reshape %v621 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v623 = stablehlo.transpose %v622, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v624 = stablehlo.reshape %v623 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v626 = stablehlo.convolution(%v625, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v627 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v628 = stablehlo.add %v626, %v627 : tensor<32x2048x14x14xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v630 = stablehlo.multiply %v629, %v629 : tensor<32x401408xf32>
    %v631 = stablehlo.multiply %v630, %v629 : tensor<32x401408xf32>
    %v632 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v633 = stablehlo.multiply %v632, %v631 : tensor<32x401408xf32>
    %v634 = stablehlo.add %v629, %v633 : tensor<32x401408xf32>
    %v635 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v636 = stablehlo.multiply %v635, %v634 : tensor<32x401408xf32>
    %v637 = stablehlo.tanh %v636 : tensor<32x401408xf32>
    %v638 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v639 = stablehlo.add %v638, %v637 : tensor<32x401408xf32>
    %v640 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v641 = stablehlo.multiply %v640, %v629 : tensor<32x401408xf32>
    %v642 = stablehlo.multiply %v641, %v639 : tensor<32x401408xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v644 = stablehlo.convolution(%v643, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v645 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v646 = stablehlo.add %v644, %v645 : tensor<32x512x14x14xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v649 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v650 = stablehlo.multiply %v648, %v649 : tensor<32x512x14x14xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v652 = stablehlo.add %v651, %v585 : tensor<32x100352xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v654 = stablehlo.convolution(%v653, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v655 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v656 = stablehlo.add %v654, %v655 : tensor<32x512x14x14xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v659 = stablehlo.transpose %v658, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v663 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v664 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v665 = stablehlo.reduce(%v661 init: %v662) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v666 = stablehlo.broadcast_in_dim %v665, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v667 = stablehlo.divide %v666, %v663 : tensor<32x196x512xf32>
    %v668 = stablehlo.subtract %v661, %v667 : tensor<32x196x512xf32>
    %v669 = stablehlo.multiply %v668, %v668 : tensor<32x196x512xf32>
    %v670 = stablehlo.reduce(%v669 init: %v662) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v671 = stablehlo.broadcast_in_dim %v670, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v672 = stablehlo.divide %v671, %v663 : tensor<32x196x512xf32>
    %v673 = stablehlo.add %v672, %v664 : tensor<32x196x512xf32>
    %v674 = stablehlo.rsqrt %v673 : tensor<32x196x512xf32>
    %v675 = stablehlo.multiply %v668, %v674 : tensor<32x196x512xf32>
    %v676 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v677 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v678 = stablehlo.multiply %v675, %v676 : tensor<32x196x512xf32>
    %v679 = stablehlo.add %v678, %v677 : tensor<32x196x512xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v682 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v683 = stablehlo.multiply %v681, %v682 : tensor<32x196x512xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v686 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v687 = stablehlo.add %v685, %v686 : tensor<32x196x512xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v690 = stablehlo.transpose %v689, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v693 = stablehlo.convolution(%v692, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v694 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v695 = stablehlo.add %v693, %v694 : tensor<32x2048x14x14xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v697 = stablehlo.multiply %v696, %v696 : tensor<32x401408xf32>
    %v698 = stablehlo.multiply %v697, %v696 : tensor<32x401408xf32>
    %v699 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v700 = stablehlo.multiply %v699, %v698 : tensor<32x401408xf32>
    %v701 = stablehlo.add %v696, %v700 : tensor<32x401408xf32>
    %v702 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v703 = stablehlo.multiply %v702, %v701 : tensor<32x401408xf32>
    %v704 = stablehlo.tanh %v703 : tensor<32x401408xf32>
    %v705 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v706 = stablehlo.add %v705, %v704 : tensor<32x401408xf32>
    %v707 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v708 = stablehlo.multiply %v707, %v696 : tensor<32x401408xf32>
    %v709 = stablehlo.multiply %v708, %v706 : tensor<32x401408xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v711 = stablehlo.convolution(%v710, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v712 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v713 = stablehlo.add %v711, %v712 : tensor<32x512x14x14xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v716 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v717 = stablehlo.multiply %v715, %v716 : tensor<32x512x14x14xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v719 = stablehlo.add %v718, %v652 : tensor<32x100352xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v721 = stablehlo.convolution(%v720, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v722 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v723 = stablehlo.add %v721, %v722 : tensor<32x512x14x14xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v726 = stablehlo.transpose %v725, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v730 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v731 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v732 = stablehlo.reduce(%v728 init: %v729) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v733 = stablehlo.broadcast_in_dim %v732, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v734 = stablehlo.divide %v733, %v730 : tensor<32x196x512xf32>
    %v735 = stablehlo.subtract %v728, %v734 : tensor<32x196x512xf32>
    %v736 = stablehlo.multiply %v735, %v735 : tensor<32x196x512xf32>
    %v737 = stablehlo.reduce(%v736 init: %v729) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v738 = stablehlo.broadcast_in_dim %v737, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v739 = stablehlo.divide %v738, %v730 : tensor<32x196x512xf32>
    %v740 = stablehlo.add %v739, %v731 : tensor<32x196x512xf32>
    %v741 = stablehlo.rsqrt %v740 : tensor<32x196x512xf32>
    %v742 = stablehlo.multiply %v735, %v741 : tensor<32x196x512xf32>
    %v743 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v744 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v745 = stablehlo.multiply %v742, %v743 : tensor<32x196x512xf32>
    %v746 = stablehlo.add %v745, %v744 : tensor<32x196x512xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v749 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v750 = stablehlo.multiply %v748, %v749 : tensor<32x196x512xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v753 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v754 = stablehlo.add %v752, %v753 : tensor<32x196x512xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v757 = stablehlo.transpose %v756, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v758 = stablehlo.reshape %v757 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v759 = stablehlo.reshape %v758 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v760 = stablehlo.convolution(%v759, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v762 = stablehlo.add %v760, %v761 : tensor<32x2048x14x14xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v764 = stablehlo.multiply %v763, %v763 : tensor<32x401408xf32>
    %v765 = stablehlo.multiply %v764, %v763 : tensor<32x401408xf32>
    %v766 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v767 = stablehlo.multiply %v766, %v765 : tensor<32x401408xf32>
    %v768 = stablehlo.add %v763, %v767 : tensor<32x401408xf32>
    %v769 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v770 = stablehlo.multiply %v769, %v768 : tensor<32x401408xf32>
    %v771 = stablehlo.tanh %v770 : tensor<32x401408xf32>
    %v772 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v773 = stablehlo.add %v772, %v771 : tensor<32x401408xf32>
    %v774 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v775 = stablehlo.multiply %v774, %v763 : tensor<32x401408xf32>
    %v776 = stablehlo.multiply %v775, %v773 : tensor<32x401408xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v778 = stablehlo.convolution(%v777, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v779 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v780 = stablehlo.add %v778, %v779 : tensor<32x512x14x14xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v783 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v784 = stablehlo.multiply %v782, %v783 : tensor<32x512x14x14xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v786 = stablehlo.add %v785, %v719 : tensor<32x100352xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v788 = stablehlo.convolution(%v787, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v789 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v790 = stablehlo.add %v788, %v789 : tensor<32x512x14x14xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v793 = stablehlo.transpose %v792, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v797 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v798 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v799 = stablehlo.reduce(%v795 init: %v796) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v800 = stablehlo.broadcast_in_dim %v799, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v801 = stablehlo.divide %v800, %v797 : tensor<32x196x512xf32>
    %v802 = stablehlo.subtract %v795, %v801 : tensor<32x196x512xf32>
    %v803 = stablehlo.multiply %v802, %v802 : tensor<32x196x512xf32>
    %v804 = stablehlo.reduce(%v803 init: %v796) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v805 = stablehlo.broadcast_in_dim %v804, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v806 = stablehlo.divide %v805, %v797 : tensor<32x196x512xf32>
    %v807 = stablehlo.add %v806, %v798 : tensor<32x196x512xf32>
    %v808 = stablehlo.rsqrt %v807 : tensor<32x196x512xf32>
    %v809 = stablehlo.multiply %v802, %v808 : tensor<32x196x512xf32>
    %v810 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v811 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v812 = stablehlo.multiply %v809, %v810 : tensor<32x196x512xf32>
    %v813 = stablehlo.add %v812, %v811 : tensor<32x196x512xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v816 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v817 = stablehlo.multiply %v815, %v816 : tensor<32x196x512xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v820 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v821 = stablehlo.add %v819, %v820 : tensor<32x196x512xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v824 = stablehlo.transpose %v823, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v827 = stablehlo.convolution(%v826, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v828 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v829 = stablehlo.add %v827, %v828 : tensor<32x2048x14x14xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v831 = stablehlo.multiply %v830, %v830 : tensor<32x401408xf32>
    %v832 = stablehlo.multiply %v831, %v830 : tensor<32x401408xf32>
    %v833 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v834 = stablehlo.multiply %v833, %v832 : tensor<32x401408xf32>
    %v835 = stablehlo.add %v830, %v834 : tensor<32x401408xf32>
    %v836 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v837 = stablehlo.multiply %v836, %v835 : tensor<32x401408xf32>
    %v838 = stablehlo.tanh %v837 : tensor<32x401408xf32>
    %v839 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v840 = stablehlo.add %v839, %v838 : tensor<32x401408xf32>
    %v841 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v842 = stablehlo.multiply %v841, %v830 : tensor<32x401408xf32>
    %v843 = stablehlo.multiply %v842, %v840 : tensor<32x401408xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v845 = stablehlo.convolution(%v844, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v846 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<32x512x14x14xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v850 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v851 = stablehlo.multiply %v849, %v850 : tensor<32x512x14x14xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v853 = stablehlo.add %v852, %v786 : tensor<32x100352xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v855 = stablehlo.convolution(%v854, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v856 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v857 = stablehlo.add %v855, %v856 : tensor<32x512x14x14xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v860 = stablehlo.transpose %v859, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v864 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v865 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v866 = stablehlo.reduce(%v862 init: %v863) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v867 = stablehlo.broadcast_in_dim %v866, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v868 = stablehlo.divide %v867, %v864 : tensor<32x196x512xf32>
    %v869 = stablehlo.subtract %v862, %v868 : tensor<32x196x512xf32>
    %v870 = stablehlo.multiply %v869, %v869 : tensor<32x196x512xf32>
    %v871 = stablehlo.reduce(%v870 init: %v863) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v872 = stablehlo.broadcast_in_dim %v871, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v873 = stablehlo.divide %v872, %v864 : tensor<32x196x512xf32>
    %v874 = stablehlo.add %v873, %v865 : tensor<32x196x512xf32>
    %v875 = stablehlo.rsqrt %v874 : tensor<32x196x512xf32>
    %v876 = stablehlo.multiply %v869, %v875 : tensor<32x196x512xf32>
    %v877 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v878 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v879 = stablehlo.multiply %v876, %v877 : tensor<32x196x512xf32>
    %v880 = stablehlo.add %v879, %v878 : tensor<32x196x512xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v883 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v884 = stablehlo.multiply %v882, %v883 : tensor<32x196x512xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v887 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v888 = stablehlo.add %v886, %v887 : tensor<32x196x512xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v891 = stablehlo.transpose %v890, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v892 = stablehlo.reshape %v891 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v894 = stablehlo.convolution(%v893, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v895 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v896 = stablehlo.add %v894, %v895 : tensor<32x2048x14x14xf32>
    %v897 = stablehlo.reshape %v896 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v898 = stablehlo.multiply %v897, %v897 : tensor<32x401408xf32>
    %v899 = stablehlo.multiply %v898, %v897 : tensor<32x401408xf32>
    %v900 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v901 = stablehlo.multiply %v900, %v899 : tensor<32x401408xf32>
    %v902 = stablehlo.add %v897, %v901 : tensor<32x401408xf32>
    %v903 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v904 = stablehlo.multiply %v903, %v902 : tensor<32x401408xf32>
    %v905 = stablehlo.tanh %v904 : tensor<32x401408xf32>
    %v906 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v907 = stablehlo.add %v906, %v905 : tensor<32x401408xf32>
    %v908 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v909 = stablehlo.multiply %v908, %v897 : tensor<32x401408xf32>
    %v910 = stablehlo.multiply %v909, %v907 : tensor<32x401408xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v912 = stablehlo.convolution(%v911, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v913 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v914 = stablehlo.add %v912, %v913 : tensor<32x512x14x14xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v917 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v918 = stablehlo.multiply %v916, %v917 : tensor<32x512x14x14xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v920 = stablehlo.add %v919, %v853 : tensor<32x100352xf32>
    %v921 = stablehlo.reshape %v920 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v922 = stablehlo.convolution(%v921, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v923 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v924 = stablehlo.add %v922, %v923 : tensor<32x512x14x14xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v927 = stablehlo.transpose %v926, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v928 = stablehlo.reshape %v927 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v931 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v932 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v933 = stablehlo.reduce(%v929 init: %v930) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v934 = stablehlo.broadcast_in_dim %v933, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v935 = stablehlo.divide %v934, %v931 : tensor<32x196x512xf32>
    %v936 = stablehlo.subtract %v929, %v935 : tensor<32x196x512xf32>
    %v937 = stablehlo.multiply %v936, %v936 : tensor<32x196x512xf32>
    %v938 = stablehlo.reduce(%v937 init: %v930) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v939 = stablehlo.broadcast_in_dim %v938, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v940 = stablehlo.divide %v939, %v931 : tensor<32x196x512xf32>
    %v941 = stablehlo.add %v940, %v932 : tensor<32x196x512xf32>
    %v942 = stablehlo.rsqrt %v941 : tensor<32x196x512xf32>
    %v943 = stablehlo.multiply %v936, %v942 : tensor<32x196x512xf32>
    %v944 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v945 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v946 = stablehlo.multiply %v943, %v944 : tensor<32x196x512xf32>
    %v947 = stablehlo.add %v946, %v945 : tensor<32x196x512xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v950 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v951 = stablehlo.multiply %v949, %v950 : tensor<32x196x512xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v954 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v955 = stablehlo.add %v953, %v954 : tensor<32x196x512xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v958 = stablehlo.transpose %v957, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v961 = stablehlo.convolution(%v960, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v962 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v963 = stablehlo.add %v961, %v962 : tensor<32x2048x14x14xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v965 = stablehlo.multiply %v964, %v964 : tensor<32x401408xf32>
    %v966 = stablehlo.multiply %v965, %v964 : tensor<32x401408xf32>
    %v967 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v968 = stablehlo.multiply %v967, %v966 : tensor<32x401408xf32>
    %v969 = stablehlo.add %v964, %v968 : tensor<32x401408xf32>
    %v970 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v971 = stablehlo.multiply %v970, %v969 : tensor<32x401408xf32>
    %v972 = stablehlo.tanh %v971 : tensor<32x401408xf32>
    %v973 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v974 = stablehlo.add %v973, %v972 : tensor<32x401408xf32>
    %v975 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v976 = stablehlo.multiply %v975, %v964 : tensor<32x401408xf32>
    %v977 = stablehlo.multiply %v976, %v974 : tensor<32x401408xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v979 = stablehlo.convolution(%v978, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v980 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v981 = stablehlo.add %v979, %v980 : tensor<32x512x14x14xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v984 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v985 = stablehlo.multiply %v983, %v984 : tensor<32x512x14x14xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v987 = stablehlo.add %v986, %v920 : tensor<32x100352xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v989 = stablehlo.convolution(%v988, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v990 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v991 = stablehlo.add %v989, %v990 : tensor<32x512x14x14xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v994 = stablehlo.transpose %v993, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v998 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v999 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1000 = stablehlo.reduce(%v996 init: %v997) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1001 = stablehlo.broadcast_in_dim %v1000, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1002 = stablehlo.divide %v1001, %v998 : tensor<32x196x512xf32>
    %v1003 = stablehlo.subtract %v996, %v1002 : tensor<32x196x512xf32>
    %v1004 = stablehlo.multiply %v1003, %v1003 : tensor<32x196x512xf32>
    %v1005 = stablehlo.reduce(%v1004 init: %v997) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1006 = stablehlo.broadcast_in_dim %v1005, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1007 = stablehlo.divide %v1006, %v998 : tensor<32x196x512xf32>
    %v1008 = stablehlo.add %v1007, %v999 : tensor<32x196x512xf32>
    %v1009 = stablehlo.rsqrt %v1008 : tensor<32x196x512xf32>
    %v1010 = stablehlo.multiply %v1003, %v1009 : tensor<32x196x512xf32>
    %v1011 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1012 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1013 = stablehlo.multiply %v1010, %v1011 : tensor<32x196x512xf32>
    %v1014 = stablehlo.add %v1013, %v1012 : tensor<32x196x512xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1017 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1018 = stablehlo.multiply %v1016, %v1017 : tensor<32x196x512xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1021 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1022 = stablehlo.add %v1020, %v1021 : tensor<32x196x512xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1025 = stablehlo.transpose %v1024, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1026 = stablehlo.reshape %v1025 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1028 = stablehlo.convolution(%v1027, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1029 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1030 = stablehlo.add %v1028, %v1029 : tensor<32x2048x14x14xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1032 = stablehlo.multiply %v1031, %v1031 : tensor<32x401408xf32>
    %v1033 = stablehlo.multiply %v1032, %v1031 : tensor<32x401408xf32>
    %v1034 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1035 = stablehlo.multiply %v1034, %v1033 : tensor<32x401408xf32>
    %v1036 = stablehlo.add %v1031, %v1035 : tensor<32x401408xf32>
    %v1037 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1038 = stablehlo.multiply %v1037, %v1036 : tensor<32x401408xf32>
    %v1039 = stablehlo.tanh %v1038 : tensor<32x401408xf32>
    %v1040 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1041 = stablehlo.add %v1040, %v1039 : tensor<32x401408xf32>
    %v1042 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1043 = stablehlo.multiply %v1042, %v1031 : tensor<32x401408xf32>
    %v1044 = stablehlo.multiply %v1043, %v1041 : tensor<32x401408xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1046 = stablehlo.convolution(%v1045, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1047 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1048 = stablehlo.add %v1046, %v1047 : tensor<32x512x14x14xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1051 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1052 = stablehlo.multiply %v1050, %v1051 : tensor<32x512x14x14xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1054 = stablehlo.add %v1053, %v987 : tensor<32x100352xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1056 = stablehlo.convolution(%v1055, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1057 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1058 = stablehlo.add %v1056, %v1057 : tensor<32x512x14x14xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1061 = stablehlo.transpose %v1060, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1065 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1066 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1067 = stablehlo.reduce(%v1063 init: %v1064) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1068 = stablehlo.broadcast_in_dim %v1067, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1069 = stablehlo.divide %v1068, %v1065 : tensor<32x196x512xf32>
    %v1070 = stablehlo.subtract %v1063, %v1069 : tensor<32x196x512xf32>
    %v1071 = stablehlo.multiply %v1070, %v1070 : tensor<32x196x512xf32>
    %v1072 = stablehlo.reduce(%v1071 init: %v1064) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1073 = stablehlo.broadcast_in_dim %v1072, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1074 = stablehlo.divide %v1073, %v1065 : tensor<32x196x512xf32>
    %v1075 = stablehlo.add %v1074, %v1066 : tensor<32x196x512xf32>
    %v1076 = stablehlo.rsqrt %v1075 : tensor<32x196x512xf32>
    %v1077 = stablehlo.multiply %v1070, %v1076 : tensor<32x196x512xf32>
    %v1078 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1079 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1080 = stablehlo.multiply %v1077, %v1078 : tensor<32x196x512xf32>
    %v1081 = stablehlo.add %v1080, %v1079 : tensor<32x196x512xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1084 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1085 = stablehlo.multiply %v1083, %v1084 : tensor<32x196x512xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1088 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1089 = stablehlo.add %v1087, %v1088 : tensor<32x196x512xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1092 = stablehlo.transpose %v1091, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1095 = stablehlo.convolution(%v1094, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1096 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1097 = stablehlo.add %v1095, %v1096 : tensor<32x2048x14x14xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1099 = stablehlo.multiply %v1098, %v1098 : tensor<32x401408xf32>
    %v1100 = stablehlo.multiply %v1099, %v1098 : tensor<32x401408xf32>
    %v1101 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1102 = stablehlo.multiply %v1101, %v1100 : tensor<32x401408xf32>
    %v1103 = stablehlo.add %v1098, %v1102 : tensor<32x401408xf32>
    %v1104 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1105 = stablehlo.multiply %v1104, %v1103 : tensor<32x401408xf32>
    %v1106 = stablehlo.tanh %v1105 : tensor<32x401408xf32>
    %v1107 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1108 = stablehlo.add %v1107, %v1106 : tensor<32x401408xf32>
    %v1109 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1110 = stablehlo.multiply %v1109, %v1098 : tensor<32x401408xf32>
    %v1111 = stablehlo.multiply %v1110, %v1108 : tensor<32x401408xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1113 = stablehlo.convolution(%v1112, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1114 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1115 = stablehlo.add %v1113, %v1114 : tensor<32x512x14x14xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1117 = stablehlo.reshape %v1116 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1118 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1119 = stablehlo.multiply %v1117, %v1118 : tensor<32x512x14x14xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1121 = stablehlo.add %v1120, %v1054 : tensor<32x100352xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1123 = stablehlo.convolution(%v1122, %s2b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1124 = stablehlo.broadcast_in_dim %s2b9db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1125 = stablehlo.add %v1123, %v1124 : tensor<32x512x14x14xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1128 = stablehlo.transpose %v1127, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1130 = stablehlo.reshape %v1129 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1132 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1133 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1134 = stablehlo.reduce(%v1130 init: %v1131) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1135 = stablehlo.broadcast_in_dim %v1134, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1136 = stablehlo.divide %v1135, %v1132 : tensor<32x196x512xf32>
    %v1137 = stablehlo.subtract %v1130, %v1136 : tensor<32x196x512xf32>
    %v1138 = stablehlo.multiply %v1137, %v1137 : tensor<32x196x512xf32>
    %v1139 = stablehlo.reduce(%v1138 init: %v1131) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1140 = stablehlo.broadcast_in_dim %v1139, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1141 = stablehlo.divide %v1140, %v1132 : tensor<32x196x512xf32>
    %v1142 = stablehlo.add %v1141, %v1133 : tensor<32x196x512xf32>
    %v1143 = stablehlo.rsqrt %v1142 : tensor<32x196x512xf32>
    %v1144 = stablehlo.multiply %v1137, %v1143 : tensor<32x196x512xf32>
    %v1145 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1146 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1147 = stablehlo.multiply %v1144, %v1145 : tensor<32x196x512xf32>
    %v1148 = stablehlo.add %v1147, %v1146 : tensor<32x196x512xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1151 = stablehlo.broadcast_in_dim %s2b9ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1152 = stablehlo.multiply %v1150, %v1151 : tensor<32x196x512xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1154 = stablehlo.reshape %v1153 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1155 = stablehlo.broadcast_in_dim %s2b9nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1156 = stablehlo.add %v1154, %v1155 : tensor<32x196x512xf32>
    %v1157 = stablehlo.reshape %v1156 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1158 = stablehlo.reshape %v1157 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1159 = stablehlo.transpose %v1158, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1162 = stablehlo.convolution(%v1161, %s2b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1163 = stablehlo.broadcast_in_dim %s2b9eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1164 = stablehlo.add %v1162, %v1163 : tensor<32x2048x14x14xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1166 = stablehlo.multiply %v1165, %v1165 : tensor<32x401408xf32>
    %v1167 = stablehlo.multiply %v1166, %v1165 : tensor<32x401408xf32>
    %v1168 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1169 = stablehlo.multiply %v1168, %v1167 : tensor<32x401408xf32>
    %v1170 = stablehlo.add %v1165, %v1169 : tensor<32x401408xf32>
    %v1171 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1172 = stablehlo.multiply %v1171, %v1170 : tensor<32x401408xf32>
    %v1173 = stablehlo.tanh %v1172 : tensor<32x401408xf32>
    %v1174 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1175 = stablehlo.add %v1174, %v1173 : tensor<32x401408xf32>
    %v1176 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1177 = stablehlo.multiply %v1176, %v1165 : tensor<32x401408xf32>
    %v1178 = stablehlo.multiply %v1177, %v1175 : tensor<32x401408xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1180 = stablehlo.convolution(%v1179, %s2b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1181 = stablehlo.broadcast_in_dim %s2b9pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1182 = stablehlo.add %v1180, %v1181 : tensor<32x512x14x14xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1185 = stablehlo.broadcast_in_dim %s2b9lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1186 = stablehlo.multiply %v1184, %v1185 : tensor<32x512x14x14xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1188 = stablehlo.add %v1187, %v1121 : tensor<32x100352xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1190 = stablehlo.convolution(%v1189, %s2b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1191 = stablehlo.broadcast_in_dim %s2b10db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1192 = stablehlo.add %v1190, %v1191 : tensor<32x512x14x14xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1195 = stablehlo.transpose %v1194, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1197 = stablehlo.reshape %v1196 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1199 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1200 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1201 = stablehlo.reduce(%v1197 init: %v1198) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1202 = stablehlo.broadcast_in_dim %v1201, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1203 = stablehlo.divide %v1202, %v1199 : tensor<32x196x512xf32>
    %v1204 = stablehlo.subtract %v1197, %v1203 : tensor<32x196x512xf32>
    %v1205 = stablehlo.multiply %v1204, %v1204 : tensor<32x196x512xf32>
    %v1206 = stablehlo.reduce(%v1205 init: %v1198) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1207 = stablehlo.broadcast_in_dim %v1206, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1208 = stablehlo.divide %v1207, %v1199 : tensor<32x196x512xf32>
    %v1209 = stablehlo.add %v1208, %v1200 : tensor<32x196x512xf32>
    %v1210 = stablehlo.rsqrt %v1209 : tensor<32x196x512xf32>
    %v1211 = stablehlo.multiply %v1204, %v1210 : tensor<32x196x512xf32>
    %v1212 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1213 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1214 = stablehlo.multiply %v1211, %v1212 : tensor<32x196x512xf32>
    %v1215 = stablehlo.add %v1214, %v1213 : tensor<32x196x512xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1218 = stablehlo.broadcast_in_dim %s2b10ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1219 = stablehlo.multiply %v1217, %v1218 : tensor<32x196x512xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1222 = stablehlo.broadcast_in_dim %s2b10nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1223 = stablehlo.add %v1221, %v1222 : tensor<32x196x512xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1226 = stablehlo.transpose %v1225, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1229 = stablehlo.convolution(%v1228, %s2b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1230 = stablehlo.broadcast_in_dim %s2b10eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1231 = stablehlo.add %v1229, %v1230 : tensor<32x2048x14x14xf32>
    %v1232 = stablehlo.reshape %v1231 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1233 = stablehlo.multiply %v1232, %v1232 : tensor<32x401408xf32>
    %v1234 = stablehlo.multiply %v1233, %v1232 : tensor<32x401408xf32>
    %v1235 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1236 = stablehlo.multiply %v1235, %v1234 : tensor<32x401408xf32>
    %v1237 = stablehlo.add %v1232, %v1236 : tensor<32x401408xf32>
    %v1238 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1239 = stablehlo.multiply %v1238, %v1237 : tensor<32x401408xf32>
    %v1240 = stablehlo.tanh %v1239 : tensor<32x401408xf32>
    %v1241 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1242 = stablehlo.add %v1241, %v1240 : tensor<32x401408xf32>
    %v1243 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1244 = stablehlo.multiply %v1243, %v1232 : tensor<32x401408xf32>
    %v1245 = stablehlo.multiply %v1244, %v1242 : tensor<32x401408xf32>
    %v1246 = stablehlo.reshape %v1245 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1247 = stablehlo.convolution(%v1246, %s2b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1248 = stablehlo.broadcast_in_dim %s2b10pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1249 = stablehlo.add %v1247, %v1248 : tensor<32x512x14x14xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1251 = stablehlo.reshape %v1250 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1252 = stablehlo.broadcast_in_dim %s2b10lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1253 = stablehlo.multiply %v1251, %v1252 : tensor<32x512x14x14xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1255 = stablehlo.add %v1254, %v1188 : tensor<32x100352xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1257 = stablehlo.convolution(%v1256, %s2b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1258 = stablehlo.broadcast_in_dim %s2b11db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1259 = stablehlo.add %v1257, %v1258 : tensor<32x512x14x14xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1261 = stablehlo.reshape %v1260 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1262 = stablehlo.transpose %v1261, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1266 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1267 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1268 = stablehlo.reduce(%v1264 init: %v1265) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1269 = stablehlo.broadcast_in_dim %v1268, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1270 = stablehlo.divide %v1269, %v1266 : tensor<32x196x512xf32>
    %v1271 = stablehlo.subtract %v1264, %v1270 : tensor<32x196x512xf32>
    %v1272 = stablehlo.multiply %v1271, %v1271 : tensor<32x196x512xf32>
    %v1273 = stablehlo.reduce(%v1272 init: %v1265) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1274 = stablehlo.broadcast_in_dim %v1273, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1275 = stablehlo.divide %v1274, %v1266 : tensor<32x196x512xf32>
    %v1276 = stablehlo.add %v1275, %v1267 : tensor<32x196x512xf32>
    %v1277 = stablehlo.rsqrt %v1276 : tensor<32x196x512xf32>
    %v1278 = stablehlo.multiply %v1271, %v1277 : tensor<32x196x512xf32>
    %v1279 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1280 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1281 = stablehlo.multiply %v1278, %v1279 : tensor<32x196x512xf32>
    %v1282 = stablehlo.add %v1281, %v1280 : tensor<32x196x512xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1285 = stablehlo.broadcast_in_dim %s2b11ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1286 = stablehlo.multiply %v1284, %v1285 : tensor<32x196x512xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1289 = stablehlo.broadcast_in_dim %s2b11nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1290 = stablehlo.add %v1288, %v1289 : tensor<32x196x512xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1293 = stablehlo.transpose %v1292, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1296 = stablehlo.convolution(%v1295, %s2b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1297 = stablehlo.broadcast_in_dim %s2b11eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1298 = stablehlo.add %v1296, %v1297 : tensor<32x2048x14x14xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1300 = stablehlo.multiply %v1299, %v1299 : tensor<32x401408xf32>
    %v1301 = stablehlo.multiply %v1300, %v1299 : tensor<32x401408xf32>
    %v1302 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1303 = stablehlo.multiply %v1302, %v1301 : tensor<32x401408xf32>
    %v1304 = stablehlo.add %v1299, %v1303 : tensor<32x401408xf32>
    %v1305 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1306 = stablehlo.multiply %v1305, %v1304 : tensor<32x401408xf32>
    %v1307 = stablehlo.tanh %v1306 : tensor<32x401408xf32>
    %v1308 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1309 = stablehlo.add %v1308, %v1307 : tensor<32x401408xf32>
    %v1310 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1311 = stablehlo.multiply %v1310, %v1299 : tensor<32x401408xf32>
    %v1312 = stablehlo.multiply %v1311, %v1309 : tensor<32x401408xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1314 = stablehlo.convolution(%v1313, %s2b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1315 = stablehlo.broadcast_in_dim %s2b11pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1316 = stablehlo.add %v1314, %v1315 : tensor<32x512x14x14xf32>
    %v1317 = stablehlo.reshape %v1316 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1319 = stablehlo.broadcast_in_dim %s2b11lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1320 = stablehlo.multiply %v1318, %v1319 : tensor<32x512x14x14xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1322 = stablehlo.add %v1321, %v1255 : tensor<32x100352xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1324 = stablehlo.convolution(%v1323, %s2b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1325 = stablehlo.broadcast_in_dim %s2b12db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1326 = stablehlo.add %v1324, %v1325 : tensor<32x512x14x14xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1329 = stablehlo.transpose %v1328, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1333 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1334 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1335 = stablehlo.reduce(%v1331 init: %v1332) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1336 = stablehlo.broadcast_in_dim %v1335, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1337 = stablehlo.divide %v1336, %v1333 : tensor<32x196x512xf32>
    %v1338 = stablehlo.subtract %v1331, %v1337 : tensor<32x196x512xf32>
    %v1339 = stablehlo.multiply %v1338, %v1338 : tensor<32x196x512xf32>
    %v1340 = stablehlo.reduce(%v1339 init: %v1332) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1341 = stablehlo.broadcast_in_dim %v1340, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1342 = stablehlo.divide %v1341, %v1333 : tensor<32x196x512xf32>
    %v1343 = stablehlo.add %v1342, %v1334 : tensor<32x196x512xf32>
    %v1344 = stablehlo.rsqrt %v1343 : tensor<32x196x512xf32>
    %v1345 = stablehlo.multiply %v1338, %v1344 : tensor<32x196x512xf32>
    %v1346 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1347 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1348 = stablehlo.multiply %v1345, %v1346 : tensor<32x196x512xf32>
    %v1349 = stablehlo.add %v1348, %v1347 : tensor<32x196x512xf32>
    %v1350 = stablehlo.reshape %v1349 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1352 = stablehlo.broadcast_in_dim %s2b12ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1353 = stablehlo.multiply %v1351, %v1352 : tensor<32x196x512xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1356 = stablehlo.broadcast_in_dim %s2b12nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1357 = stablehlo.add %v1355, %v1356 : tensor<32x196x512xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1360 = stablehlo.transpose %v1359, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1363 = stablehlo.convolution(%v1362, %s2b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1364 = stablehlo.broadcast_in_dim %s2b12eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1365 = stablehlo.add %v1363, %v1364 : tensor<32x2048x14x14xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1367 = stablehlo.multiply %v1366, %v1366 : tensor<32x401408xf32>
    %v1368 = stablehlo.multiply %v1367, %v1366 : tensor<32x401408xf32>
    %v1369 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1370 = stablehlo.multiply %v1369, %v1368 : tensor<32x401408xf32>
    %v1371 = stablehlo.add %v1366, %v1370 : tensor<32x401408xf32>
    %v1372 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1373 = stablehlo.multiply %v1372, %v1371 : tensor<32x401408xf32>
    %v1374 = stablehlo.tanh %v1373 : tensor<32x401408xf32>
    %v1375 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1376 = stablehlo.add %v1375, %v1374 : tensor<32x401408xf32>
    %v1377 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1378 = stablehlo.multiply %v1377, %v1366 : tensor<32x401408xf32>
    %v1379 = stablehlo.multiply %v1378, %v1376 : tensor<32x401408xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1381 = stablehlo.convolution(%v1380, %s2b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1382 = stablehlo.broadcast_in_dim %s2b12pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1383 = stablehlo.add %v1381, %v1382 : tensor<32x512x14x14xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1386 = stablehlo.broadcast_in_dim %s2b12lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1387 = stablehlo.multiply %v1385, %v1386 : tensor<32x512x14x14xf32>
    %v1388 = stablehlo.reshape %v1387 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1389 = stablehlo.add %v1388, %v1322 : tensor<32x100352xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1391 = stablehlo.convolution(%v1390, %s2b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1392 = stablehlo.broadcast_in_dim %s2b13db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1393 = stablehlo.add %v1391, %v1392 : tensor<32x512x14x14xf32>
    %v1394 = stablehlo.reshape %v1393 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1395 = stablehlo.reshape %v1394 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1396 = stablehlo.transpose %v1395, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1400 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1401 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1402 = stablehlo.reduce(%v1398 init: %v1399) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1403 = stablehlo.broadcast_in_dim %v1402, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1404 = stablehlo.divide %v1403, %v1400 : tensor<32x196x512xf32>
    %v1405 = stablehlo.subtract %v1398, %v1404 : tensor<32x196x512xf32>
    %v1406 = stablehlo.multiply %v1405, %v1405 : tensor<32x196x512xf32>
    %v1407 = stablehlo.reduce(%v1406 init: %v1399) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1408 = stablehlo.broadcast_in_dim %v1407, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1409 = stablehlo.divide %v1408, %v1400 : tensor<32x196x512xf32>
    %v1410 = stablehlo.add %v1409, %v1401 : tensor<32x196x512xf32>
    %v1411 = stablehlo.rsqrt %v1410 : tensor<32x196x512xf32>
    %v1412 = stablehlo.multiply %v1405, %v1411 : tensor<32x196x512xf32>
    %v1413 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1414 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1415 = stablehlo.multiply %v1412, %v1413 : tensor<32x196x512xf32>
    %v1416 = stablehlo.add %v1415, %v1414 : tensor<32x196x512xf32>
    %v1417 = stablehlo.reshape %v1416 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1418 = stablehlo.reshape %v1417 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1419 = stablehlo.broadcast_in_dim %s2b13ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1420 = stablehlo.multiply %v1418, %v1419 : tensor<32x196x512xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1422 = stablehlo.reshape %v1421 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1423 = stablehlo.broadcast_in_dim %s2b13nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1424 = stablehlo.add %v1422, %v1423 : tensor<32x196x512xf32>
    %v1425 = stablehlo.reshape %v1424 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1426 = stablehlo.reshape %v1425 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1427 = stablehlo.transpose %v1426, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1430 = stablehlo.convolution(%v1429, %s2b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1431 = stablehlo.broadcast_in_dim %s2b13eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1432 = stablehlo.add %v1430, %v1431 : tensor<32x2048x14x14xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1434 = stablehlo.multiply %v1433, %v1433 : tensor<32x401408xf32>
    %v1435 = stablehlo.multiply %v1434, %v1433 : tensor<32x401408xf32>
    %v1436 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1437 = stablehlo.multiply %v1436, %v1435 : tensor<32x401408xf32>
    %v1438 = stablehlo.add %v1433, %v1437 : tensor<32x401408xf32>
    %v1439 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1440 = stablehlo.multiply %v1439, %v1438 : tensor<32x401408xf32>
    %v1441 = stablehlo.tanh %v1440 : tensor<32x401408xf32>
    %v1442 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1443 = stablehlo.add %v1442, %v1441 : tensor<32x401408xf32>
    %v1444 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1445 = stablehlo.multiply %v1444, %v1433 : tensor<32x401408xf32>
    %v1446 = stablehlo.multiply %v1445, %v1443 : tensor<32x401408xf32>
    %v1447 = stablehlo.reshape %v1446 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1448 = stablehlo.convolution(%v1447, %s2b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1449 = stablehlo.broadcast_in_dim %s2b13pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1450 = stablehlo.add %v1448, %v1449 : tensor<32x512x14x14xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1453 = stablehlo.broadcast_in_dim %s2b13lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1454 = stablehlo.multiply %v1452, %v1453 : tensor<32x512x14x14xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1456 = stablehlo.add %v1455, %v1389 : tensor<32x100352xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1458 = stablehlo.convolution(%v1457, %s2b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1459 = stablehlo.broadcast_in_dim %s2b14db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1460 = stablehlo.add %v1458, %v1459 : tensor<32x512x14x14xf32>
    %v1461 = stablehlo.reshape %v1460 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1463 = stablehlo.transpose %v1462, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1465 = stablehlo.reshape %v1464 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1466 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1467 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1468 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1469 = stablehlo.reduce(%v1465 init: %v1466) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1470 = stablehlo.broadcast_in_dim %v1469, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1471 = stablehlo.divide %v1470, %v1467 : tensor<32x196x512xf32>
    %v1472 = stablehlo.subtract %v1465, %v1471 : tensor<32x196x512xf32>
    %v1473 = stablehlo.multiply %v1472, %v1472 : tensor<32x196x512xf32>
    %v1474 = stablehlo.reduce(%v1473 init: %v1466) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1475 = stablehlo.broadcast_in_dim %v1474, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1476 = stablehlo.divide %v1475, %v1467 : tensor<32x196x512xf32>
    %v1477 = stablehlo.add %v1476, %v1468 : tensor<32x196x512xf32>
    %v1478 = stablehlo.rsqrt %v1477 : tensor<32x196x512xf32>
    %v1479 = stablehlo.multiply %v1472, %v1478 : tensor<32x196x512xf32>
    %v1480 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1481 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1482 = stablehlo.multiply %v1479, %v1480 : tensor<32x196x512xf32>
    %v1483 = stablehlo.add %v1482, %v1481 : tensor<32x196x512xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1485 = stablehlo.reshape %v1484 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1486 = stablehlo.broadcast_in_dim %s2b14ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1487 = stablehlo.multiply %v1485, %v1486 : tensor<32x196x512xf32>
    %v1488 = stablehlo.reshape %v1487 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1490 = stablehlo.broadcast_in_dim %s2b14nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1491 = stablehlo.add %v1489, %v1490 : tensor<32x196x512xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1494 = stablehlo.transpose %v1493, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1495 = stablehlo.reshape %v1494 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1496 = stablehlo.reshape %v1495 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1497 = stablehlo.convolution(%v1496, %s2b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1498 = stablehlo.broadcast_in_dim %s2b14eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1499 = stablehlo.add %v1497, %v1498 : tensor<32x2048x14x14xf32>
    %v1500 = stablehlo.reshape %v1499 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1501 = stablehlo.multiply %v1500, %v1500 : tensor<32x401408xf32>
    %v1502 = stablehlo.multiply %v1501, %v1500 : tensor<32x401408xf32>
    %v1503 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1504 = stablehlo.multiply %v1503, %v1502 : tensor<32x401408xf32>
    %v1505 = stablehlo.add %v1500, %v1504 : tensor<32x401408xf32>
    %v1506 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1507 = stablehlo.multiply %v1506, %v1505 : tensor<32x401408xf32>
    %v1508 = stablehlo.tanh %v1507 : tensor<32x401408xf32>
    %v1509 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1510 = stablehlo.add %v1509, %v1508 : tensor<32x401408xf32>
    %v1511 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1512 = stablehlo.multiply %v1511, %v1500 : tensor<32x401408xf32>
    %v1513 = stablehlo.multiply %v1512, %v1510 : tensor<32x401408xf32>
    %v1514 = stablehlo.reshape %v1513 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1515 = stablehlo.convolution(%v1514, %s2b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1516 = stablehlo.broadcast_in_dim %s2b14pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1517 = stablehlo.add %v1515, %v1516 : tensor<32x512x14x14xf32>
    %v1518 = stablehlo.reshape %v1517 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1520 = stablehlo.broadcast_in_dim %s2b14lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1521 = stablehlo.multiply %v1519, %v1520 : tensor<32x512x14x14xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1523 = stablehlo.add %v1522, %v1456 : tensor<32x100352xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1525 = stablehlo.convolution(%v1524, %s2b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1526 = stablehlo.broadcast_in_dim %s2b15db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1527 = stablehlo.add %v1525, %v1526 : tensor<32x512x14x14xf32>
    %v1528 = stablehlo.reshape %v1527 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1530 = stablehlo.transpose %v1529, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1532 = stablehlo.reshape %v1531 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1534 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1535 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1536 = stablehlo.reduce(%v1532 init: %v1533) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1537 = stablehlo.broadcast_in_dim %v1536, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1538 = stablehlo.divide %v1537, %v1534 : tensor<32x196x512xf32>
    %v1539 = stablehlo.subtract %v1532, %v1538 : tensor<32x196x512xf32>
    %v1540 = stablehlo.multiply %v1539, %v1539 : tensor<32x196x512xf32>
    %v1541 = stablehlo.reduce(%v1540 init: %v1533) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1542 = stablehlo.broadcast_in_dim %v1541, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1543 = stablehlo.divide %v1542, %v1534 : tensor<32x196x512xf32>
    %v1544 = stablehlo.add %v1543, %v1535 : tensor<32x196x512xf32>
    %v1545 = stablehlo.rsqrt %v1544 : tensor<32x196x512xf32>
    %v1546 = stablehlo.multiply %v1539, %v1545 : tensor<32x196x512xf32>
    %v1547 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1548 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1549 = stablehlo.multiply %v1546, %v1547 : tensor<32x196x512xf32>
    %v1550 = stablehlo.add %v1549, %v1548 : tensor<32x196x512xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1552 = stablehlo.reshape %v1551 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1553 = stablehlo.broadcast_in_dim %s2b15ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1554 = stablehlo.multiply %v1552, %v1553 : tensor<32x196x512xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1557 = stablehlo.broadcast_in_dim %s2b15nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1558 = stablehlo.add %v1556, %v1557 : tensor<32x196x512xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1560 = stablehlo.reshape %v1559 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1561 = stablehlo.transpose %v1560, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1562 = stablehlo.reshape %v1561 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1564 = stablehlo.convolution(%v1563, %s2b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1565 = stablehlo.broadcast_in_dim %s2b15eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1566 = stablehlo.add %v1564, %v1565 : tensor<32x2048x14x14xf32>
    %v1567 = stablehlo.reshape %v1566 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1568 = stablehlo.multiply %v1567, %v1567 : tensor<32x401408xf32>
    %v1569 = stablehlo.multiply %v1568, %v1567 : tensor<32x401408xf32>
    %v1570 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1571 = stablehlo.multiply %v1570, %v1569 : tensor<32x401408xf32>
    %v1572 = stablehlo.add %v1567, %v1571 : tensor<32x401408xf32>
    %v1573 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1574 = stablehlo.multiply %v1573, %v1572 : tensor<32x401408xf32>
    %v1575 = stablehlo.tanh %v1574 : tensor<32x401408xf32>
    %v1576 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1577 = stablehlo.add %v1576, %v1575 : tensor<32x401408xf32>
    %v1578 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1579 = stablehlo.multiply %v1578, %v1567 : tensor<32x401408xf32>
    %v1580 = stablehlo.multiply %v1579, %v1577 : tensor<32x401408xf32>
    %v1581 = stablehlo.reshape %v1580 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1582 = stablehlo.convolution(%v1581, %s2b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1583 = stablehlo.broadcast_in_dim %s2b15pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1584 = stablehlo.add %v1582, %v1583 : tensor<32x512x14x14xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1586 = stablehlo.reshape %v1585 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1587 = stablehlo.broadcast_in_dim %s2b15lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1588 = stablehlo.multiply %v1586, %v1587 : tensor<32x512x14x14xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1590 = stablehlo.add %v1589, %v1523 : tensor<32x100352xf32>
    %v1591 = stablehlo.reshape %v1590 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1592 = stablehlo.convolution(%v1591, %s2b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1593 = stablehlo.broadcast_in_dim %s2b16db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1594 = stablehlo.add %v1592, %v1593 : tensor<32x512x14x14xf32>
    %v1595 = stablehlo.reshape %v1594 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1596 = stablehlo.reshape %v1595 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1597 = stablehlo.transpose %v1596, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1598 = stablehlo.reshape %v1597 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1599 = stablehlo.reshape %v1598 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1600 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1601 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1602 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1603 = stablehlo.reduce(%v1599 init: %v1600) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1604 = stablehlo.broadcast_in_dim %v1603, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1605 = stablehlo.divide %v1604, %v1601 : tensor<32x196x512xf32>
    %v1606 = stablehlo.subtract %v1599, %v1605 : tensor<32x196x512xf32>
    %v1607 = stablehlo.multiply %v1606, %v1606 : tensor<32x196x512xf32>
    %v1608 = stablehlo.reduce(%v1607 init: %v1600) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1609 = stablehlo.broadcast_in_dim %v1608, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1610 = stablehlo.divide %v1609, %v1601 : tensor<32x196x512xf32>
    %v1611 = stablehlo.add %v1610, %v1602 : tensor<32x196x512xf32>
    %v1612 = stablehlo.rsqrt %v1611 : tensor<32x196x512xf32>
    %v1613 = stablehlo.multiply %v1606, %v1612 : tensor<32x196x512xf32>
    %v1614 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1615 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1616 = stablehlo.multiply %v1613, %v1614 : tensor<32x196x512xf32>
    %v1617 = stablehlo.add %v1616, %v1615 : tensor<32x196x512xf32>
    %v1618 = stablehlo.reshape %v1617 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1619 = stablehlo.reshape %v1618 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1620 = stablehlo.broadcast_in_dim %s2b16ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1621 = stablehlo.multiply %v1619, %v1620 : tensor<32x196x512xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1623 = stablehlo.reshape %v1622 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1624 = stablehlo.broadcast_in_dim %s2b16nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1625 = stablehlo.add %v1623, %v1624 : tensor<32x196x512xf32>
    %v1626 = stablehlo.reshape %v1625 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1627 = stablehlo.reshape %v1626 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1628 = stablehlo.transpose %v1627, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1629 = stablehlo.reshape %v1628 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1630 = stablehlo.reshape %v1629 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1631 = stablehlo.convolution(%v1630, %s2b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1632 = stablehlo.broadcast_in_dim %s2b16eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1633 = stablehlo.add %v1631, %v1632 : tensor<32x2048x14x14xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1635 = stablehlo.multiply %v1634, %v1634 : tensor<32x401408xf32>
    %v1636 = stablehlo.multiply %v1635, %v1634 : tensor<32x401408xf32>
    %v1637 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1638 = stablehlo.multiply %v1637, %v1636 : tensor<32x401408xf32>
    %v1639 = stablehlo.add %v1634, %v1638 : tensor<32x401408xf32>
    %v1640 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1641 = stablehlo.multiply %v1640, %v1639 : tensor<32x401408xf32>
    %v1642 = stablehlo.tanh %v1641 : tensor<32x401408xf32>
    %v1643 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1644 = stablehlo.add %v1643, %v1642 : tensor<32x401408xf32>
    %v1645 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1646 = stablehlo.multiply %v1645, %v1634 : tensor<32x401408xf32>
    %v1647 = stablehlo.multiply %v1646, %v1644 : tensor<32x401408xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1649 = stablehlo.convolution(%v1648, %s2b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1650 = stablehlo.broadcast_in_dim %s2b16pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1651 = stablehlo.add %v1649, %v1650 : tensor<32x512x14x14xf32>
    %v1652 = stablehlo.reshape %v1651 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1653 = stablehlo.reshape %v1652 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1654 = stablehlo.broadcast_in_dim %s2b16lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1655 = stablehlo.multiply %v1653, %v1654 : tensor<32x512x14x14xf32>
    %v1656 = stablehlo.reshape %v1655 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1657 = stablehlo.add %v1656, %v1590 : tensor<32x100352xf32>
    %v1658 = stablehlo.reshape %v1657 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1659 = stablehlo.convolution(%v1658, %s2b17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1660 = stablehlo.broadcast_in_dim %s2b17db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1661 = stablehlo.add %v1659, %v1660 : tensor<32x512x14x14xf32>
    %v1662 = stablehlo.reshape %v1661 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1664 = stablehlo.transpose %v1663, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1667 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1668 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1669 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1670 = stablehlo.reduce(%v1666 init: %v1667) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1671 = stablehlo.broadcast_in_dim %v1670, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1672 = stablehlo.divide %v1671, %v1668 : tensor<32x196x512xf32>
    %v1673 = stablehlo.subtract %v1666, %v1672 : tensor<32x196x512xf32>
    %v1674 = stablehlo.multiply %v1673, %v1673 : tensor<32x196x512xf32>
    %v1675 = stablehlo.reduce(%v1674 init: %v1667) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1676 = stablehlo.broadcast_in_dim %v1675, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1677 = stablehlo.divide %v1676, %v1668 : tensor<32x196x512xf32>
    %v1678 = stablehlo.add %v1677, %v1669 : tensor<32x196x512xf32>
    %v1679 = stablehlo.rsqrt %v1678 : tensor<32x196x512xf32>
    %v1680 = stablehlo.multiply %v1673, %v1679 : tensor<32x196x512xf32>
    %v1681 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1682 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1683 = stablehlo.multiply %v1680, %v1681 : tensor<32x196x512xf32>
    %v1684 = stablehlo.add %v1683, %v1682 : tensor<32x196x512xf32>
    %v1685 = stablehlo.reshape %v1684 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1686 = stablehlo.reshape %v1685 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1687 = stablehlo.broadcast_in_dim %s2b17ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1688 = stablehlo.multiply %v1686, %v1687 : tensor<32x196x512xf32>
    %v1689 = stablehlo.reshape %v1688 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1690 = stablehlo.reshape %v1689 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1691 = stablehlo.broadcast_in_dim %s2b17nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1692 = stablehlo.add %v1690, %v1691 : tensor<32x196x512xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1694 = stablehlo.reshape %v1693 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1695 = stablehlo.transpose %v1694, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1696 = stablehlo.reshape %v1695 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1698 = stablehlo.convolution(%v1697, %s2b17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1699 = stablehlo.broadcast_in_dim %s2b17eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1700 = stablehlo.add %v1698, %v1699 : tensor<32x2048x14x14xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1702 = stablehlo.multiply %v1701, %v1701 : tensor<32x401408xf32>
    %v1703 = stablehlo.multiply %v1702, %v1701 : tensor<32x401408xf32>
    %v1704 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1705 = stablehlo.multiply %v1704, %v1703 : tensor<32x401408xf32>
    %v1706 = stablehlo.add %v1701, %v1705 : tensor<32x401408xf32>
    %v1707 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1708 = stablehlo.multiply %v1707, %v1706 : tensor<32x401408xf32>
    %v1709 = stablehlo.tanh %v1708 : tensor<32x401408xf32>
    %v1710 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1711 = stablehlo.add %v1710, %v1709 : tensor<32x401408xf32>
    %v1712 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1713 = stablehlo.multiply %v1712, %v1701 : tensor<32x401408xf32>
    %v1714 = stablehlo.multiply %v1713, %v1711 : tensor<32x401408xf32>
    %v1715 = stablehlo.reshape %v1714 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1716 = stablehlo.convolution(%v1715, %s2b17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1717 = stablehlo.broadcast_in_dim %s2b17pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1718 = stablehlo.add %v1716, %v1717 : tensor<32x512x14x14xf32>
    %v1719 = stablehlo.reshape %v1718 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1720 = stablehlo.reshape %v1719 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1721 = stablehlo.broadcast_in_dim %s2b17lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1722 = stablehlo.multiply %v1720, %v1721 : tensor<32x512x14x14xf32>
    %v1723 = stablehlo.reshape %v1722 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1724 = stablehlo.add %v1723, %v1657 : tensor<32x100352xf32>
    %v1725 = stablehlo.reshape %v1724 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1726 = stablehlo.convolution(%v1725, %s2b18dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1727 = stablehlo.broadcast_in_dim %s2b18db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1728 = stablehlo.add %v1726, %v1727 : tensor<32x512x14x14xf32>
    %v1729 = stablehlo.reshape %v1728 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1730 = stablehlo.reshape %v1729 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1731 = stablehlo.transpose %v1730, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1732 = stablehlo.reshape %v1731 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1733 = stablehlo.reshape %v1732 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1735 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1736 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1737 = stablehlo.reduce(%v1733 init: %v1734) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1738 = stablehlo.broadcast_in_dim %v1737, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1739 = stablehlo.divide %v1738, %v1735 : tensor<32x196x512xf32>
    %v1740 = stablehlo.subtract %v1733, %v1739 : tensor<32x196x512xf32>
    %v1741 = stablehlo.multiply %v1740, %v1740 : tensor<32x196x512xf32>
    %v1742 = stablehlo.reduce(%v1741 init: %v1734) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1743 = stablehlo.broadcast_in_dim %v1742, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1744 = stablehlo.divide %v1743, %v1735 : tensor<32x196x512xf32>
    %v1745 = stablehlo.add %v1744, %v1736 : tensor<32x196x512xf32>
    %v1746 = stablehlo.rsqrt %v1745 : tensor<32x196x512xf32>
    %v1747 = stablehlo.multiply %v1740, %v1746 : tensor<32x196x512xf32>
    %v1748 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1749 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1750 = stablehlo.multiply %v1747, %v1748 : tensor<32x196x512xf32>
    %v1751 = stablehlo.add %v1750, %v1749 : tensor<32x196x512xf32>
    %v1752 = stablehlo.reshape %v1751 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1754 = stablehlo.broadcast_in_dim %s2b18ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1755 = stablehlo.multiply %v1753, %v1754 : tensor<32x196x512xf32>
    %v1756 = stablehlo.reshape %v1755 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1757 = stablehlo.reshape %v1756 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1758 = stablehlo.broadcast_in_dim %s2b18nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1759 = stablehlo.add %v1757, %v1758 : tensor<32x196x512xf32>
    %v1760 = stablehlo.reshape %v1759 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1761 = stablehlo.reshape %v1760 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1762 = stablehlo.transpose %v1761, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1763 = stablehlo.reshape %v1762 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1764 = stablehlo.reshape %v1763 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1765 = stablehlo.convolution(%v1764, %s2b18eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1766 = stablehlo.broadcast_in_dim %s2b18eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1767 = stablehlo.add %v1765, %v1766 : tensor<32x2048x14x14xf32>
    %v1768 = stablehlo.reshape %v1767 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1769 = stablehlo.multiply %v1768, %v1768 : tensor<32x401408xf32>
    %v1770 = stablehlo.multiply %v1769, %v1768 : tensor<32x401408xf32>
    %v1771 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1772 = stablehlo.multiply %v1771, %v1770 : tensor<32x401408xf32>
    %v1773 = stablehlo.add %v1768, %v1772 : tensor<32x401408xf32>
    %v1774 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1775 = stablehlo.multiply %v1774, %v1773 : tensor<32x401408xf32>
    %v1776 = stablehlo.tanh %v1775 : tensor<32x401408xf32>
    %v1777 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1778 = stablehlo.add %v1777, %v1776 : tensor<32x401408xf32>
    %v1779 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1780 = stablehlo.multiply %v1779, %v1768 : tensor<32x401408xf32>
    %v1781 = stablehlo.multiply %v1780, %v1778 : tensor<32x401408xf32>
    %v1782 = stablehlo.reshape %v1781 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1783 = stablehlo.convolution(%v1782, %s2b18pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1784 = stablehlo.broadcast_in_dim %s2b18pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1785 = stablehlo.add %v1783, %v1784 : tensor<32x512x14x14xf32>
    %v1786 = stablehlo.reshape %v1785 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1787 = stablehlo.reshape %v1786 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1788 = stablehlo.broadcast_in_dim %s2b18lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1789 = stablehlo.multiply %v1787, %v1788 : tensor<32x512x14x14xf32>
    %v1790 = stablehlo.reshape %v1789 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1791 = stablehlo.add %v1790, %v1724 : tensor<32x100352xf32>
    %v1792 = stablehlo.reshape %v1791 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1793 = stablehlo.convolution(%v1792, %s2b19dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1794 = stablehlo.broadcast_in_dim %s2b19db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1795 = stablehlo.add %v1793, %v1794 : tensor<32x512x14x14xf32>
    %v1796 = stablehlo.reshape %v1795 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1798 = stablehlo.transpose %v1797, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1799 = stablehlo.reshape %v1798 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1800 = stablehlo.reshape %v1799 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1802 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1803 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1804 = stablehlo.reduce(%v1800 init: %v1801) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1805 = stablehlo.broadcast_in_dim %v1804, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1806 = stablehlo.divide %v1805, %v1802 : tensor<32x196x512xf32>
    %v1807 = stablehlo.subtract %v1800, %v1806 : tensor<32x196x512xf32>
    %v1808 = stablehlo.multiply %v1807, %v1807 : tensor<32x196x512xf32>
    %v1809 = stablehlo.reduce(%v1808 init: %v1801) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1810 = stablehlo.broadcast_in_dim %v1809, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1811 = stablehlo.divide %v1810, %v1802 : tensor<32x196x512xf32>
    %v1812 = stablehlo.add %v1811, %v1803 : tensor<32x196x512xf32>
    %v1813 = stablehlo.rsqrt %v1812 : tensor<32x196x512xf32>
    %v1814 = stablehlo.multiply %v1807, %v1813 : tensor<32x196x512xf32>
    %v1815 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1816 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1817 = stablehlo.multiply %v1814, %v1815 : tensor<32x196x512xf32>
    %v1818 = stablehlo.add %v1817, %v1816 : tensor<32x196x512xf32>
    %v1819 = stablehlo.reshape %v1818 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1820 = stablehlo.reshape %v1819 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1821 = stablehlo.broadcast_in_dim %s2b19ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1822 = stablehlo.multiply %v1820, %v1821 : tensor<32x196x512xf32>
    %v1823 = stablehlo.reshape %v1822 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1824 = stablehlo.reshape %v1823 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1825 = stablehlo.broadcast_in_dim %s2b19nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1826 = stablehlo.add %v1824, %v1825 : tensor<32x196x512xf32>
    %v1827 = stablehlo.reshape %v1826 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1828 = stablehlo.reshape %v1827 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1829 = stablehlo.transpose %v1828, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1830 = stablehlo.reshape %v1829 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1831 = stablehlo.reshape %v1830 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1832 = stablehlo.convolution(%v1831, %s2b19eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1833 = stablehlo.broadcast_in_dim %s2b19eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1834 = stablehlo.add %v1832, %v1833 : tensor<32x2048x14x14xf32>
    %v1835 = stablehlo.reshape %v1834 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1836 = stablehlo.multiply %v1835, %v1835 : tensor<32x401408xf32>
    %v1837 = stablehlo.multiply %v1836, %v1835 : tensor<32x401408xf32>
    %v1838 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1839 = stablehlo.multiply %v1838, %v1837 : tensor<32x401408xf32>
    %v1840 = stablehlo.add %v1835, %v1839 : tensor<32x401408xf32>
    %v1841 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1842 = stablehlo.multiply %v1841, %v1840 : tensor<32x401408xf32>
    %v1843 = stablehlo.tanh %v1842 : tensor<32x401408xf32>
    %v1844 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1845 = stablehlo.add %v1844, %v1843 : tensor<32x401408xf32>
    %v1846 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1847 = stablehlo.multiply %v1846, %v1835 : tensor<32x401408xf32>
    %v1848 = stablehlo.multiply %v1847, %v1845 : tensor<32x401408xf32>
    %v1849 = stablehlo.reshape %v1848 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1850 = stablehlo.convolution(%v1849, %s2b19pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1851 = stablehlo.broadcast_in_dim %s2b19pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1852 = stablehlo.add %v1850, %v1851 : tensor<32x512x14x14xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1854 = stablehlo.reshape %v1853 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1855 = stablehlo.broadcast_in_dim %s2b19lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1856 = stablehlo.multiply %v1854, %v1855 : tensor<32x512x14x14xf32>
    %v1857 = stablehlo.reshape %v1856 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1858 = stablehlo.add %v1857, %v1791 : tensor<32x100352xf32>
    %v1859 = stablehlo.reshape %v1858 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1860 = stablehlo.convolution(%v1859, %s2b20dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1861 = stablehlo.broadcast_in_dim %s2b20db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1862 = stablehlo.add %v1860, %v1861 : tensor<32x512x14x14xf32>
    %v1863 = stablehlo.reshape %v1862 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1864 = stablehlo.reshape %v1863 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1865 = stablehlo.transpose %v1864, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1866 = stablehlo.reshape %v1865 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1867 = stablehlo.reshape %v1866 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1869 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1870 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1871 = stablehlo.reduce(%v1867 init: %v1868) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1872 = stablehlo.broadcast_in_dim %v1871, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1873 = stablehlo.divide %v1872, %v1869 : tensor<32x196x512xf32>
    %v1874 = stablehlo.subtract %v1867, %v1873 : tensor<32x196x512xf32>
    %v1875 = stablehlo.multiply %v1874, %v1874 : tensor<32x196x512xf32>
    %v1876 = stablehlo.reduce(%v1875 init: %v1868) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1877 = stablehlo.broadcast_in_dim %v1876, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1878 = stablehlo.divide %v1877, %v1869 : tensor<32x196x512xf32>
    %v1879 = stablehlo.add %v1878, %v1870 : tensor<32x196x512xf32>
    %v1880 = stablehlo.rsqrt %v1879 : tensor<32x196x512xf32>
    %v1881 = stablehlo.multiply %v1874, %v1880 : tensor<32x196x512xf32>
    %v1882 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1883 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1884 = stablehlo.multiply %v1881, %v1882 : tensor<32x196x512xf32>
    %v1885 = stablehlo.add %v1884, %v1883 : tensor<32x196x512xf32>
    %v1886 = stablehlo.reshape %v1885 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1887 = stablehlo.reshape %v1886 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1888 = stablehlo.broadcast_in_dim %s2b20ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1889 = stablehlo.multiply %v1887, %v1888 : tensor<32x196x512xf32>
    %v1890 = stablehlo.reshape %v1889 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1891 = stablehlo.reshape %v1890 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1892 = stablehlo.broadcast_in_dim %s2b20nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1893 = stablehlo.add %v1891, %v1892 : tensor<32x196x512xf32>
    %v1894 = stablehlo.reshape %v1893 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1895 = stablehlo.reshape %v1894 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1896 = stablehlo.transpose %v1895, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1897 = stablehlo.reshape %v1896 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1898 = stablehlo.reshape %v1897 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1899 = stablehlo.convolution(%v1898, %s2b20eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1900 = stablehlo.broadcast_in_dim %s2b20eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1901 = stablehlo.add %v1899, %v1900 : tensor<32x2048x14x14xf32>
    %v1902 = stablehlo.reshape %v1901 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1903 = stablehlo.multiply %v1902, %v1902 : tensor<32x401408xf32>
    %v1904 = stablehlo.multiply %v1903, %v1902 : tensor<32x401408xf32>
    %v1905 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1906 = stablehlo.multiply %v1905, %v1904 : tensor<32x401408xf32>
    %v1907 = stablehlo.add %v1902, %v1906 : tensor<32x401408xf32>
    %v1908 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1909 = stablehlo.multiply %v1908, %v1907 : tensor<32x401408xf32>
    %v1910 = stablehlo.tanh %v1909 : tensor<32x401408xf32>
    %v1911 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1912 = stablehlo.add %v1911, %v1910 : tensor<32x401408xf32>
    %v1913 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1914 = stablehlo.multiply %v1913, %v1902 : tensor<32x401408xf32>
    %v1915 = stablehlo.multiply %v1914, %v1912 : tensor<32x401408xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1917 = stablehlo.convolution(%v1916, %s2b20pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1918 = stablehlo.broadcast_in_dim %s2b20pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1919 = stablehlo.add %v1917, %v1918 : tensor<32x512x14x14xf32>
    %v1920 = stablehlo.reshape %v1919 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1921 = stablehlo.reshape %v1920 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1922 = stablehlo.broadcast_in_dim %s2b20lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1923 = stablehlo.multiply %v1921, %v1922 : tensor<32x512x14x14xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1925 = stablehlo.add %v1924, %v1858 : tensor<32x100352xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1927 = stablehlo.convolution(%v1926, %s2b21dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1928 = stablehlo.broadcast_in_dim %s2b21db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1929 = stablehlo.add %v1927, %v1928 : tensor<32x512x14x14xf32>
    %v1930 = stablehlo.reshape %v1929 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1931 = stablehlo.reshape %v1930 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1932 = stablehlo.transpose %v1931, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v1933 = stablehlo.reshape %v1932 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1934 = stablehlo.reshape %v1933 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1936 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v1937 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v1938 = stablehlo.reduce(%v1934 init: %v1935) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1939 = stablehlo.broadcast_in_dim %v1938, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1940 = stablehlo.divide %v1939, %v1936 : tensor<32x196x512xf32>
    %v1941 = stablehlo.subtract %v1934, %v1940 : tensor<32x196x512xf32>
    %v1942 = stablehlo.multiply %v1941, %v1941 : tensor<32x196x512xf32>
    %v1943 = stablehlo.reduce(%v1942 init: %v1935) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1944 = stablehlo.broadcast_in_dim %v1943, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v1945 = stablehlo.divide %v1944, %v1936 : tensor<32x196x512xf32>
    %v1946 = stablehlo.add %v1945, %v1937 : tensor<32x196x512xf32>
    %v1947 = stablehlo.rsqrt %v1946 : tensor<32x196x512xf32>
    %v1948 = stablehlo.multiply %v1941, %v1947 : tensor<32x196x512xf32>
    %v1949 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1950 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v1951 = stablehlo.multiply %v1948, %v1949 : tensor<32x196x512xf32>
    %v1952 = stablehlo.add %v1951, %v1950 : tensor<32x196x512xf32>
    %v1953 = stablehlo.reshape %v1952 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1954 = stablehlo.reshape %v1953 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1955 = stablehlo.broadcast_in_dim %s2b21ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1956 = stablehlo.multiply %v1954, %v1955 : tensor<32x196x512xf32>
    %v1957 = stablehlo.reshape %v1956 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1958 = stablehlo.reshape %v1957 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1959 = stablehlo.broadcast_in_dim %s2b21nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v1960 = stablehlo.add %v1958, %v1959 : tensor<32x196x512xf32>
    %v1961 = stablehlo.reshape %v1960 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v1962 = stablehlo.reshape %v1961 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v1963 = stablehlo.transpose %v1962, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v1964 = stablehlo.reshape %v1963 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v1965 = stablehlo.reshape %v1964 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1966 = stablehlo.convolution(%v1965, %s2b21eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v1967 = stablehlo.broadcast_in_dim %s2b21eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v1968 = stablehlo.add %v1966, %v1967 : tensor<32x2048x14x14xf32>
    %v1969 = stablehlo.reshape %v1968 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v1970 = stablehlo.multiply %v1969, %v1969 : tensor<32x401408xf32>
    %v1971 = stablehlo.multiply %v1970, %v1969 : tensor<32x401408xf32>
    %v1972 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v1973 = stablehlo.multiply %v1972, %v1971 : tensor<32x401408xf32>
    %v1974 = stablehlo.add %v1969, %v1973 : tensor<32x401408xf32>
    %v1975 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v1976 = stablehlo.multiply %v1975, %v1974 : tensor<32x401408xf32>
    %v1977 = stablehlo.tanh %v1976 : tensor<32x401408xf32>
    %v1978 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v1979 = stablehlo.add %v1978, %v1977 : tensor<32x401408xf32>
    %v1980 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v1981 = stablehlo.multiply %v1980, %v1969 : tensor<32x401408xf32>
    %v1982 = stablehlo.multiply %v1981, %v1979 : tensor<32x401408xf32>
    %v1983 = stablehlo.reshape %v1982 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v1984 = stablehlo.convolution(%v1983, %s2b21pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1985 = stablehlo.broadcast_in_dim %s2b21pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1986 = stablehlo.add %v1984, %v1985 : tensor<32x512x14x14xf32>
    %v1987 = stablehlo.reshape %v1986 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1988 = stablehlo.reshape %v1987 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1989 = stablehlo.broadcast_in_dim %s2b21lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1990 = stablehlo.multiply %v1988, %v1989 : tensor<32x512x14x14xf32>
    %v1991 = stablehlo.reshape %v1990 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1992 = stablehlo.add %v1991, %v1925 : tensor<32x100352xf32>
    %v1993 = stablehlo.reshape %v1992 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1994 = stablehlo.convolution(%v1993, %s2b22dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v1995 = stablehlo.broadcast_in_dim %s2b22db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1996 = stablehlo.add %v1994, %v1995 : tensor<32x512x14x14xf32>
    %v1997 = stablehlo.reshape %v1996 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1998 = stablehlo.reshape %v1997 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v1999 = stablehlo.transpose %v1998, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2000 = stablehlo.reshape %v1999 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2001 = stablehlo.reshape %v2000 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2003 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2004 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2005 = stablehlo.reduce(%v2001 init: %v2002) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2006 = stablehlo.broadcast_in_dim %v2005, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2007 = stablehlo.divide %v2006, %v2003 : tensor<32x196x512xf32>
    %v2008 = stablehlo.subtract %v2001, %v2007 : tensor<32x196x512xf32>
    %v2009 = stablehlo.multiply %v2008, %v2008 : tensor<32x196x512xf32>
    %v2010 = stablehlo.reduce(%v2009 init: %v2002) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2011 = stablehlo.broadcast_in_dim %v2010, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2012 = stablehlo.divide %v2011, %v2003 : tensor<32x196x512xf32>
    %v2013 = stablehlo.add %v2012, %v2004 : tensor<32x196x512xf32>
    %v2014 = stablehlo.rsqrt %v2013 : tensor<32x196x512xf32>
    %v2015 = stablehlo.multiply %v2008, %v2014 : tensor<32x196x512xf32>
    %v2016 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2017 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2018 = stablehlo.multiply %v2015, %v2016 : tensor<32x196x512xf32>
    %v2019 = stablehlo.add %v2018, %v2017 : tensor<32x196x512xf32>
    %v2020 = stablehlo.reshape %v2019 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2021 = stablehlo.reshape %v2020 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2022 = stablehlo.broadcast_in_dim %s2b22ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2023 = stablehlo.multiply %v2021, %v2022 : tensor<32x196x512xf32>
    %v2024 = stablehlo.reshape %v2023 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2025 = stablehlo.reshape %v2024 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2026 = stablehlo.broadcast_in_dim %s2b22nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2027 = stablehlo.add %v2025, %v2026 : tensor<32x196x512xf32>
    %v2028 = stablehlo.reshape %v2027 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2029 = stablehlo.reshape %v2028 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2030 = stablehlo.transpose %v2029, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2031 = stablehlo.reshape %v2030 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2032 = stablehlo.reshape %v2031 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2033 = stablehlo.convolution(%v2032, %s2b22eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2034 = stablehlo.broadcast_in_dim %s2b22eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2035 = stablehlo.add %v2033, %v2034 : tensor<32x2048x14x14xf32>
    %v2036 = stablehlo.reshape %v2035 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2037 = stablehlo.multiply %v2036, %v2036 : tensor<32x401408xf32>
    %v2038 = stablehlo.multiply %v2037, %v2036 : tensor<32x401408xf32>
    %v2039 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v2040 = stablehlo.multiply %v2039, %v2038 : tensor<32x401408xf32>
    %v2041 = stablehlo.add %v2036, %v2040 : tensor<32x401408xf32>
    %v2042 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v2043 = stablehlo.multiply %v2042, %v2041 : tensor<32x401408xf32>
    %v2044 = stablehlo.tanh %v2043 : tensor<32x401408xf32>
    %v2045 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v2046 = stablehlo.add %v2045, %v2044 : tensor<32x401408xf32>
    %v2047 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v2048 = stablehlo.multiply %v2047, %v2036 : tensor<32x401408xf32>
    %v2049 = stablehlo.multiply %v2048, %v2046 : tensor<32x401408xf32>
    %v2050 = stablehlo.reshape %v2049 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2051 = stablehlo.convolution(%v2050, %s2b22pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2052 = stablehlo.broadcast_in_dim %s2b22pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2053 = stablehlo.add %v2051, %v2052 : tensor<32x512x14x14xf32>
    %v2054 = stablehlo.reshape %v2053 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2055 = stablehlo.reshape %v2054 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2056 = stablehlo.broadcast_in_dim %s2b22lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2057 = stablehlo.multiply %v2055, %v2056 : tensor<32x512x14x14xf32>
    %v2058 = stablehlo.reshape %v2057 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2059 = stablehlo.add %v2058, %v1992 : tensor<32x100352xf32>
    %v2060 = stablehlo.reshape %v2059 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2061 = stablehlo.convolution(%v2060, %s2b23dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2062 = stablehlo.broadcast_in_dim %s2b23db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2063 = stablehlo.add %v2061, %v2062 : tensor<32x512x14x14xf32>
    %v2064 = stablehlo.reshape %v2063 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2065 = stablehlo.reshape %v2064 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2066 = stablehlo.transpose %v2065, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2067 = stablehlo.reshape %v2066 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2068 = stablehlo.reshape %v2067 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2069 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2070 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2071 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2072 = stablehlo.reduce(%v2068 init: %v2069) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2073 = stablehlo.broadcast_in_dim %v2072, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2074 = stablehlo.divide %v2073, %v2070 : tensor<32x196x512xf32>
    %v2075 = stablehlo.subtract %v2068, %v2074 : tensor<32x196x512xf32>
    %v2076 = stablehlo.multiply %v2075, %v2075 : tensor<32x196x512xf32>
    %v2077 = stablehlo.reduce(%v2076 init: %v2069) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2078 = stablehlo.broadcast_in_dim %v2077, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2079 = stablehlo.divide %v2078, %v2070 : tensor<32x196x512xf32>
    %v2080 = stablehlo.add %v2079, %v2071 : tensor<32x196x512xf32>
    %v2081 = stablehlo.rsqrt %v2080 : tensor<32x196x512xf32>
    %v2082 = stablehlo.multiply %v2075, %v2081 : tensor<32x196x512xf32>
    %v2083 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2084 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2085 = stablehlo.multiply %v2082, %v2083 : tensor<32x196x512xf32>
    %v2086 = stablehlo.add %v2085, %v2084 : tensor<32x196x512xf32>
    %v2087 = stablehlo.reshape %v2086 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2088 = stablehlo.reshape %v2087 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2089 = stablehlo.broadcast_in_dim %s2b23ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2090 = stablehlo.multiply %v2088, %v2089 : tensor<32x196x512xf32>
    %v2091 = stablehlo.reshape %v2090 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2092 = stablehlo.reshape %v2091 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2093 = stablehlo.broadcast_in_dim %s2b23nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2094 = stablehlo.add %v2092, %v2093 : tensor<32x196x512xf32>
    %v2095 = stablehlo.reshape %v2094 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2096 = stablehlo.reshape %v2095 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2097 = stablehlo.transpose %v2096, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2098 = stablehlo.reshape %v2097 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2099 = stablehlo.reshape %v2098 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2100 = stablehlo.convolution(%v2099, %s2b23eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2101 = stablehlo.broadcast_in_dim %s2b23eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2102 = stablehlo.add %v2100, %v2101 : tensor<32x2048x14x14xf32>
    %v2103 = stablehlo.reshape %v2102 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2104 = stablehlo.multiply %v2103, %v2103 : tensor<32x401408xf32>
    %v2105 = stablehlo.multiply %v2104, %v2103 : tensor<32x401408xf32>
    %v2106 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v2107 = stablehlo.multiply %v2106, %v2105 : tensor<32x401408xf32>
    %v2108 = stablehlo.add %v2103, %v2107 : tensor<32x401408xf32>
    %v2109 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v2110 = stablehlo.multiply %v2109, %v2108 : tensor<32x401408xf32>
    %v2111 = stablehlo.tanh %v2110 : tensor<32x401408xf32>
    %v2112 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v2113 = stablehlo.add %v2112, %v2111 : tensor<32x401408xf32>
    %v2114 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v2115 = stablehlo.multiply %v2114, %v2103 : tensor<32x401408xf32>
    %v2116 = stablehlo.multiply %v2115, %v2113 : tensor<32x401408xf32>
    %v2117 = stablehlo.reshape %v2116 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2118 = stablehlo.convolution(%v2117, %s2b23pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2119 = stablehlo.broadcast_in_dim %s2b23pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2120 = stablehlo.add %v2118, %v2119 : tensor<32x512x14x14xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2122 = stablehlo.reshape %v2121 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2123 = stablehlo.broadcast_in_dim %s2b23lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2124 = stablehlo.multiply %v2122, %v2123 : tensor<32x512x14x14xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2126 = stablehlo.add %v2125, %v2059 : tensor<32x100352xf32>
    %v2127 = stablehlo.reshape %v2126 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2128 = stablehlo.convolution(%v2127, %s2b24dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2129 = stablehlo.broadcast_in_dim %s2b24db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2130 = stablehlo.add %v2128, %v2129 : tensor<32x512x14x14xf32>
    %v2131 = stablehlo.reshape %v2130 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2132 = stablehlo.reshape %v2131 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2133 = stablehlo.transpose %v2132, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2134 = stablehlo.reshape %v2133 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2135 = stablehlo.reshape %v2134 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2137 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2138 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2139 = stablehlo.reduce(%v2135 init: %v2136) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2140 = stablehlo.broadcast_in_dim %v2139, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2141 = stablehlo.divide %v2140, %v2137 : tensor<32x196x512xf32>
    %v2142 = stablehlo.subtract %v2135, %v2141 : tensor<32x196x512xf32>
    %v2143 = stablehlo.multiply %v2142, %v2142 : tensor<32x196x512xf32>
    %v2144 = stablehlo.reduce(%v2143 init: %v2136) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2145 = stablehlo.broadcast_in_dim %v2144, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2146 = stablehlo.divide %v2145, %v2137 : tensor<32x196x512xf32>
    %v2147 = stablehlo.add %v2146, %v2138 : tensor<32x196x512xf32>
    %v2148 = stablehlo.rsqrt %v2147 : tensor<32x196x512xf32>
    %v2149 = stablehlo.multiply %v2142, %v2148 : tensor<32x196x512xf32>
    %v2150 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2151 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2152 = stablehlo.multiply %v2149, %v2150 : tensor<32x196x512xf32>
    %v2153 = stablehlo.add %v2152, %v2151 : tensor<32x196x512xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2155 = stablehlo.reshape %v2154 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2156 = stablehlo.broadcast_in_dim %s2b24ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2157 = stablehlo.multiply %v2155, %v2156 : tensor<32x196x512xf32>
    %v2158 = stablehlo.reshape %v2157 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2159 = stablehlo.reshape %v2158 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2160 = stablehlo.broadcast_in_dim %s2b24nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2161 = stablehlo.add %v2159, %v2160 : tensor<32x196x512xf32>
    %v2162 = stablehlo.reshape %v2161 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2163 = stablehlo.reshape %v2162 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2164 = stablehlo.transpose %v2163, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2166 = stablehlo.reshape %v2165 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2167 = stablehlo.convolution(%v2166, %s2b24eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2168 = stablehlo.broadcast_in_dim %s2b24eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2169 = stablehlo.add %v2167, %v2168 : tensor<32x2048x14x14xf32>
    %v2170 = stablehlo.reshape %v2169 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2171 = stablehlo.multiply %v2170, %v2170 : tensor<32x401408xf32>
    %v2172 = stablehlo.multiply %v2171, %v2170 : tensor<32x401408xf32>
    %v2173 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v2174 = stablehlo.multiply %v2173, %v2172 : tensor<32x401408xf32>
    %v2175 = stablehlo.add %v2170, %v2174 : tensor<32x401408xf32>
    %v2176 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v2177 = stablehlo.multiply %v2176, %v2175 : tensor<32x401408xf32>
    %v2178 = stablehlo.tanh %v2177 : tensor<32x401408xf32>
    %v2179 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v2180 = stablehlo.add %v2179, %v2178 : tensor<32x401408xf32>
    %v2181 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v2182 = stablehlo.multiply %v2181, %v2170 : tensor<32x401408xf32>
    %v2183 = stablehlo.multiply %v2182, %v2180 : tensor<32x401408xf32>
    %v2184 = stablehlo.reshape %v2183 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2185 = stablehlo.convolution(%v2184, %s2b24pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2186 = stablehlo.broadcast_in_dim %s2b24pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2187 = stablehlo.add %v2185, %v2186 : tensor<32x512x14x14xf32>
    %v2188 = stablehlo.reshape %v2187 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2189 = stablehlo.reshape %v2188 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2190 = stablehlo.broadcast_in_dim %s2b24lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2191 = stablehlo.multiply %v2189, %v2190 : tensor<32x512x14x14xf32>
    %v2192 = stablehlo.reshape %v2191 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2193 = stablehlo.add %v2192, %v2126 : tensor<32x100352xf32>
    %v2194 = stablehlo.reshape %v2193 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2195 = stablehlo.convolution(%v2194, %s2b25dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2196 = stablehlo.broadcast_in_dim %s2b25db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2197 = stablehlo.add %v2195, %v2196 : tensor<32x512x14x14xf32>
    %v2198 = stablehlo.reshape %v2197 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2199 = stablehlo.reshape %v2198 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2200 = stablehlo.transpose %v2199, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2201 = stablehlo.reshape %v2200 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2202 = stablehlo.reshape %v2201 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2204 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2205 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2206 = stablehlo.reduce(%v2202 init: %v2203) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2207 = stablehlo.broadcast_in_dim %v2206, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2208 = stablehlo.divide %v2207, %v2204 : tensor<32x196x512xf32>
    %v2209 = stablehlo.subtract %v2202, %v2208 : tensor<32x196x512xf32>
    %v2210 = stablehlo.multiply %v2209, %v2209 : tensor<32x196x512xf32>
    %v2211 = stablehlo.reduce(%v2210 init: %v2203) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2212 = stablehlo.broadcast_in_dim %v2211, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2213 = stablehlo.divide %v2212, %v2204 : tensor<32x196x512xf32>
    %v2214 = stablehlo.add %v2213, %v2205 : tensor<32x196x512xf32>
    %v2215 = stablehlo.rsqrt %v2214 : tensor<32x196x512xf32>
    %v2216 = stablehlo.multiply %v2209, %v2215 : tensor<32x196x512xf32>
    %v2217 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2218 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2219 = stablehlo.multiply %v2216, %v2217 : tensor<32x196x512xf32>
    %v2220 = stablehlo.add %v2219, %v2218 : tensor<32x196x512xf32>
    %v2221 = stablehlo.reshape %v2220 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2222 = stablehlo.reshape %v2221 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2223 = stablehlo.broadcast_in_dim %s2b25ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2224 = stablehlo.multiply %v2222, %v2223 : tensor<32x196x512xf32>
    %v2225 = stablehlo.reshape %v2224 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2226 = stablehlo.reshape %v2225 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2227 = stablehlo.broadcast_in_dim %s2b25nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2228 = stablehlo.add %v2226, %v2227 : tensor<32x196x512xf32>
    %v2229 = stablehlo.reshape %v2228 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2230 = stablehlo.reshape %v2229 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2231 = stablehlo.transpose %v2230, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2232 = stablehlo.reshape %v2231 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2233 = stablehlo.reshape %v2232 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2234 = stablehlo.convolution(%v2233, %s2b25eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2235 = stablehlo.broadcast_in_dim %s2b25eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2236 = stablehlo.add %v2234, %v2235 : tensor<32x2048x14x14xf32>
    %v2237 = stablehlo.reshape %v2236 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2238 = stablehlo.multiply %v2237, %v2237 : tensor<32x401408xf32>
    %v2239 = stablehlo.multiply %v2238, %v2237 : tensor<32x401408xf32>
    %v2240 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v2241 = stablehlo.multiply %v2240, %v2239 : tensor<32x401408xf32>
    %v2242 = stablehlo.add %v2237, %v2241 : tensor<32x401408xf32>
    %v2243 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v2244 = stablehlo.multiply %v2243, %v2242 : tensor<32x401408xf32>
    %v2245 = stablehlo.tanh %v2244 : tensor<32x401408xf32>
    %v2246 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v2247 = stablehlo.add %v2246, %v2245 : tensor<32x401408xf32>
    %v2248 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v2249 = stablehlo.multiply %v2248, %v2237 : tensor<32x401408xf32>
    %v2250 = stablehlo.multiply %v2249, %v2247 : tensor<32x401408xf32>
    %v2251 = stablehlo.reshape %v2250 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2252 = stablehlo.convolution(%v2251, %s2b25pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2253 = stablehlo.broadcast_in_dim %s2b25pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2254 = stablehlo.add %v2252, %v2253 : tensor<32x512x14x14xf32>
    %v2255 = stablehlo.reshape %v2254 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2256 = stablehlo.reshape %v2255 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2257 = stablehlo.broadcast_in_dim %s2b25lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2258 = stablehlo.multiply %v2256, %v2257 : tensor<32x512x14x14xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2260 = stablehlo.add %v2259, %v2193 : tensor<32x100352xf32>
    %v2261 = stablehlo.reshape %v2260 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2262 = stablehlo.convolution(%v2261, %s2b26dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x14x14xf32>, tensor<512x1x7x7xf32>) -> tensor<32x512x14x14xf32>
    %v2263 = stablehlo.broadcast_in_dim %s2b26db, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2264 = stablehlo.add %v2262, %v2263 : tensor<32x512x14x14xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2266 = stablehlo.reshape %v2265 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2267 = stablehlo.transpose %v2266, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2268 = stablehlo.reshape %v2267 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2269 = stablehlo.reshape %v2268 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2271 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2272 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2273 = stablehlo.reduce(%v2269 init: %v2270) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2274 = stablehlo.broadcast_in_dim %v2273, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2275 = stablehlo.divide %v2274, %v2271 : tensor<32x196x512xf32>
    %v2276 = stablehlo.subtract %v2269, %v2275 : tensor<32x196x512xf32>
    %v2277 = stablehlo.multiply %v2276, %v2276 : tensor<32x196x512xf32>
    %v2278 = stablehlo.reduce(%v2277 init: %v2270) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2279 = stablehlo.broadcast_in_dim %v2278, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2280 = stablehlo.divide %v2279, %v2271 : tensor<32x196x512xf32>
    %v2281 = stablehlo.add %v2280, %v2272 : tensor<32x196x512xf32>
    %v2282 = stablehlo.rsqrt %v2281 : tensor<32x196x512xf32>
    %v2283 = stablehlo.multiply %v2276, %v2282 : tensor<32x196x512xf32>
    %v2284 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2285 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2286 = stablehlo.multiply %v2283, %v2284 : tensor<32x196x512xf32>
    %v2287 = stablehlo.add %v2286, %v2285 : tensor<32x196x512xf32>
    %v2288 = stablehlo.reshape %v2287 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2289 = stablehlo.reshape %v2288 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2290 = stablehlo.broadcast_in_dim %s2b26ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2291 = stablehlo.multiply %v2289, %v2290 : tensor<32x196x512xf32>
    %v2292 = stablehlo.reshape %v2291 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2293 = stablehlo.reshape %v2292 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2294 = stablehlo.broadcast_in_dim %s2b26nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2295 = stablehlo.add %v2293, %v2294 : tensor<32x196x512xf32>
    %v2296 = stablehlo.reshape %v2295 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2297 = stablehlo.reshape %v2296 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2298 = stablehlo.transpose %v2297, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2299 = stablehlo.reshape %v2298 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2300 = stablehlo.reshape %v2299 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2301 = stablehlo.convolution(%v2300, %s2b26eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x14x14xf32>
    %v2302 = stablehlo.broadcast_in_dim %s2b26eb, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x14x14xf32>
    %v2303 = stablehlo.add %v2301, %v2302 : tensor<32x2048x14x14xf32>
    %v2304 = stablehlo.reshape %v2303 : (tensor<32x2048x14x14xf32>) -> tensor<32x401408xf32>
    %v2305 = stablehlo.multiply %v2304, %v2304 : tensor<32x401408xf32>
    %v2306 = stablehlo.multiply %v2305, %v2304 : tensor<32x401408xf32>
    %v2307 = stablehlo.constant dense<0.044715> : tensor<32x401408xf32>
    %v2308 = stablehlo.multiply %v2307, %v2306 : tensor<32x401408xf32>
    %v2309 = stablehlo.add %v2304, %v2308 : tensor<32x401408xf32>
    %v2310 = stablehlo.constant dense<0.7978845608028654> : tensor<32x401408xf32>
    %v2311 = stablehlo.multiply %v2310, %v2309 : tensor<32x401408xf32>
    %v2312 = stablehlo.tanh %v2311 : tensor<32x401408xf32>
    %v2313 = stablehlo.constant dense<1.0> : tensor<32x401408xf32>
    %v2314 = stablehlo.add %v2313, %v2312 : tensor<32x401408xf32>
    %v2315 = stablehlo.constant dense<0.5> : tensor<32x401408xf32>
    %v2316 = stablehlo.multiply %v2315, %v2304 : tensor<32x401408xf32>
    %v2317 = stablehlo.multiply %v2316, %v2314 : tensor<32x401408xf32>
    %v2318 = stablehlo.reshape %v2317 : (tensor<32x401408xf32>) -> tensor<32x2048x14x14xf32>
    %v2319 = stablehlo.convolution(%v2318, %s2b26pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x14x14xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v2320 = stablehlo.broadcast_in_dim %s2b26pb, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2321 = stablehlo.add %v2319, %v2320 : tensor<32x512x14x14xf32>
    %v2322 = stablehlo.reshape %v2321 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2323 = stablehlo.reshape %v2322 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2324 = stablehlo.broadcast_in_dim %s2b26lg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v2325 = stablehlo.multiply %v2323, %v2324 : tensor<32x512x14x14xf32>
    %v2326 = stablehlo.reshape %v2325 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v2327 = stablehlo.add %v2326, %v2260 : tensor<32x100352xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<32x100352xf32>) -> tensor<32x512x196xf32>
    %v2329 = stablehlo.transpose %v2328, dims = [0, 2, 1] : (tensor<32x512x196xf32>) -> tensor<32x196x512xf32>
    %v2330 = stablehlo.reshape %v2329 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2331 = stablehlo.reshape %v2330 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2333 = stablehlo.constant dense<512.0> : tensor<32x196x512xf32>
    %v2334 = stablehlo.constant dense<1.0e-6> : tensor<32x196x512xf32>
    %v2335 = stablehlo.reduce(%v2331 init: %v2332) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2336 = stablehlo.broadcast_in_dim %v2335, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2337 = stablehlo.divide %v2336, %v2333 : tensor<32x196x512xf32>
    %v2338 = stablehlo.subtract %v2331, %v2337 : tensor<32x196x512xf32>
    %v2339 = stablehlo.multiply %v2338, %v2338 : tensor<32x196x512xf32>
    %v2340 = stablehlo.reduce(%v2339 init: %v2332) applies stablehlo.add across dimensions = [2] : (tensor<32x196x512xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2341 = stablehlo.broadcast_in_dim %v2340, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x512xf32>
    %v2342 = stablehlo.divide %v2341, %v2333 : tensor<32x196x512xf32>
    %v2343 = stablehlo.add %v2342, %v2334 : tensor<32x196x512xf32>
    %v2344 = stablehlo.rsqrt %v2343 : tensor<32x196x512xf32>
    %v2345 = stablehlo.multiply %v2338, %v2344 : tensor<32x196x512xf32>
    %v2346 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2347 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x512xf32>
    %v2348 = stablehlo.multiply %v2345, %v2346 : tensor<32x196x512xf32>
    %v2349 = stablehlo.add %v2348, %v2347 : tensor<32x196x512xf32>
    %v2350 = stablehlo.reshape %v2349 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2351 = stablehlo.reshape %v2350 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2352 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2353 = stablehlo.multiply %v2351, %v2352 : tensor<32x196x512xf32>
    %v2354 = stablehlo.reshape %v2353 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2355 = stablehlo.reshape %v2354 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2356 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<512xf32>) -> tensor<32x196x512xf32>
    %v2357 = stablehlo.add %v2355, %v2356 : tensor<32x196x512xf32>
    %v2358 = stablehlo.reshape %v2357 : (tensor<32x196x512xf32>) -> tensor<32x100352xf32>
    %v2359 = stablehlo.reshape %v2358 : (tensor<32x100352xf32>) -> tensor<32x196x512xf32>
    %v2360 = stablehlo.transpose %v2359, dims = [0, 2, 1] : (tensor<32x196x512xf32>) -> tensor<32x512x196xf32>
    %v2361 = stablehlo.reshape %v2360 : (tensor<32x512x196xf32>) -> tensor<32x100352xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v2363 = stablehlo.convolution(%v2362, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<1024x512x2x2xf32>) -> tensor<32x1024x7x7xf32>
    %v2364 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2365 = stablehlo.add %v2363, %v2364 : tensor<32x1024x7x7xf32>
    %v2366 = stablehlo.reshape %v2365 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2367 = stablehlo.reshape %v2366 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2368 = stablehlo.convolution(%v2367, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x7x7xf32>) -> tensor<32x1024x7x7xf32>
    %v2369 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2370 = stablehlo.add %v2368, %v2369 : tensor<32x1024x7x7xf32>
    %v2371 = stablehlo.reshape %v2370 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2372 = stablehlo.reshape %v2371 : (tensor<32x50176xf32>) -> tensor<32x1024x49xf32>
    %v2373 = stablehlo.transpose %v2372, dims = [0, 2, 1] : (tensor<32x1024x49xf32>) -> tensor<32x49x1024xf32>
    %v2374 = stablehlo.reshape %v2373 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2375 = stablehlo.reshape %v2374 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2377 = stablehlo.constant dense<1024.0> : tensor<32x49x1024xf32>
    %v2378 = stablehlo.constant dense<1.0e-6> : tensor<32x49x1024xf32>
    %v2379 = stablehlo.reduce(%v2375 init: %v2376) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2380 = stablehlo.broadcast_in_dim %v2379, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2381 = stablehlo.divide %v2380, %v2377 : tensor<32x49x1024xf32>
    %v2382 = stablehlo.subtract %v2375, %v2381 : tensor<32x49x1024xf32>
    %v2383 = stablehlo.multiply %v2382, %v2382 : tensor<32x49x1024xf32>
    %v2384 = stablehlo.reduce(%v2383 init: %v2376) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2385 = stablehlo.broadcast_in_dim %v2384, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2386 = stablehlo.divide %v2385, %v2377 : tensor<32x49x1024xf32>
    %v2387 = stablehlo.add %v2386, %v2378 : tensor<32x49x1024xf32>
    %v2388 = stablehlo.rsqrt %v2387 : tensor<32x49x1024xf32>
    %v2389 = stablehlo.multiply %v2382, %v2388 : tensor<32x49x1024xf32>
    %v2390 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2391 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2392 = stablehlo.multiply %v2389, %v2390 : tensor<32x49x1024xf32>
    %v2393 = stablehlo.add %v2392, %v2391 : tensor<32x49x1024xf32>
    %v2394 = stablehlo.reshape %v2393 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2396 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2397 = stablehlo.multiply %v2395, %v2396 : tensor<32x49x1024xf32>
    %v2398 = stablehlo.reshape %v2397 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2399 = stablehlo.reshape %v2398 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2400 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2401 = stablehlo.add %v2399, %v2400 : tensor<32x49x1024xf32>
    %v2402 = stablehlo.reshape %v2401 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2403 = stablehlo.reshape %v2402 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2404 = stablehlo.transpose %v2403, dims = [0, 2, 1] : (tensor<32x49x1024xf32>) -> tensor<32x1024x49xf32>
    %v2405 = stablehlo.reshape %v2404 : (tensor<32x1024x49xf32>) -> tensor<32x50176xf32>
    %v2406 = stablehlo.reshape %v2405 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2407 = stablehlo.convolution(%v2406, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<4096x1024x1x1xf32>) -> tensor<32x4096x7x7xf32>
    %v2408 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<4096xf32>) -> tensor<32x4096x7x7xf32>
    %v2409 = stablehlo.add %v2407, %v2408 : tensor<32x4096x7x7xf32>
    %v2410 = stablehlo.reshape %v2409 : (tensor<32x4096x7x7xf32>) -> tensor<32x200704xf32>
    %v2411 = stablehlo.multiply %v2410, %v2410 : tensor<32x200704xf32>
    %v2412 = stablehlo.multiply %v2411, %v2410 : tensor<32x200704xf32>
    %v2413 = stablehlo.constant dense<0.044715> : tensor<32x200704xf32>
    %v2414 = stablehlo.multiply %v2413, %v2412 : tensor<32x200704xf32>
    %v2415 = stablehlo.add %v2410, %v2414 : tensor<32x200704xf32>
    %v2416 = stablehlo.constant dense<0.7978845608028654> : tensor<32x200704xf32>
    %v2417 = stablehlo.multiply %v2416, %v2415 : tensor<32x200704xf32>
    %v2418 = stablehlo.tanh %v2417 : tensor<32x200704xf32>
    %v2419 = stablehlo.constant dense<1.0> : tensor<32x200704xf32>
    %v2420 = stablehlo.add %v2419, %v2418 : tensor<32x200704xf32>
    %v2421 = stablehlo.constant dense<0.5> : tensor<32x200704xf32>
    %v2422 = stablehlo.multiply %v2421, %v2410 : tensor<32x200704xf32>
    %v2423 = stablehlo.multiply %v2422, %v2420 : tensor<32x200704xf32>
    %v2424 = stablehlo.reshape %v2423 : (tensor<32x200704xf32>) -> tensor<32x4096x7x7xf32>
    %v2425 = stablehlo.convolution(%v2424, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x4096x7x7xf32>, tensor<1024x4096x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v2426 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2427 = stablehlo.add %v2425, %v2426 : tensor<32x1024x7x7xf32>
    %v2428 = stablehlo.reshape %v2427 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2430 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2431 = stablehlo.multiply %v2429, %v2430 : tensor<32x1024x7x7xf32>
    %v2432 = stablehlo.reshape %v2431 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2433 = stablehlo.add %v2432, %v2366 : tensor<32x50176xf32>
    %v2434 = stablehlo.reshape %v2433 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2435 = stablehlo.convolution(%v2434, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x7x7xf32>) -> tensor<32x1024x7x7xf32>
    %v2436 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2437 = stablehlo.add %v2435, %v2436 : tensor<32x1024x7x7xf32>
    %v2438 = stablehlo.reshape %v2437 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2439 = stablehlo.reshape %v2438 : (tensor<32x50176xf32>) -> tensor<32x1024x49xf32>
    %v2440 = stablehlo.transpose %v2439, dims = [0, 2, 1] : (tensor<32x1024x49xf32>) -> tensor<32x49x1024xf32>
    %v2441 = stablehlo.reshape %v2440 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2442 = stablehlo.reshape %v2441 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2444 = stablehlo.constant dense<1024.0> : tensor<32x49x1024xf32>
    %v2445 = stablehlo.constant dense<1.0e-6> : tensor<32x49x1024xf32>
    %v2446 = stablehlo.reduce(%v2442 init: %v2443) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2447 = stablehlo.broadcast_in_dim %v2446, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2448 = stablehlo.divide %v2447, %v2444 : tensor<32x49x1024xf32>
    %v2449 = stablehlo.subtract %v2442, %v2448 : tensor<32x49x1024xf32>
    %v2450 = stablehlo.multiply %v2449, %v2449 : tensor<32x49x1024xf32>
    %v2451 = stablehlo.reduce(%v2450 init: %v2443) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2452 = stablehlo.broadcast_in_dim %v2451, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2453 = stablehlo.divide %v2452, %v2444 : tensor<32x49x1024xf32>
    %v2454 = stablehlo.add %v2453, %v2445 : tensor<32x49x1024xf32>
    %v2455 = stablehlo.rsqrt %v2454 : tensor<32x49x1024xf32>
    %v2456 = stablehlo.multiply %v2449, %v2455 : tensor<32x49x1024xf32>
    %v2457 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2458 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2459 = stablehlo.multiply %v2456, %v2457 : tensor<32x49x1024xf32>
    %v2460 = stablehlo.add %v2459, %v2458 : tensor<32x49x1024xf32>
    %v2461 = stablehlo.reshape %v2460 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2462 = stablehlo.reshape %v2461 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2463 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2464 = stablehlo.multiply %v2462, %v2463 : tensor<32x49x1024xf32>
    %v2465 = stablehlo.reshape %v2464 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2466 = stablehlo.reshape %v2465 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2467 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2468 = stablehlo.add %v2466, %v2467 : tensor<32x49x1024xf32>
    %v2469 = stablehlo.reshape %v2468 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2470 = stablehlo.reshape %v2469 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2471 = stablehlo.transpose %v2470, dims = [0, 2, 1] : (tensor<32x49x1024xf32>) -> tensor<32x1024x49xf32>
    %v2472 = stablehlo.reshape %v2471 : (tensor<32x1024x49xf32>) -> tensor<32x50176xf32>
    %v2473 = stablehlo.reshape %v2472 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2474 = stablehlo.convolution(%v2473, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<4096x1024x1x1xf32>) -> tensor<32x4096x7x7xf32>
    %v2475 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<4096xf32>) -> tensor<32x4096x7x7xf32>
    %v2476 = stablehlo.add %v2474, %v2475 : tensor<32x4096x7x7xf32>
    %v2477 = stablehlo.reshape %v2476 : (tensor<32x4096x7x7xf32>) -> tensor<32x200704xf32>
    %v2478 = stablehlo.multiply %v2477, %v2477 : tensor<32x200704xf32>
    %v2479 = stablehlo.multiply %v2478, %v2477 : tensor<32x200704xf32>
    %v2480 = stablehlo.constant dense<0.044715> : tensor<32x200704xf32>
    %v2481 = stablehlo.multiply %v2480, %v2479 : tensor<32x200704xf32>
    %v2482 = stablehlo.add %v2477, %v2481 : tensor<32x200704xf32>
    %v2483 = stablehlo.constant dense<0.7978845608028654> : tensor<32x200704xf32>
    %v2484 = stablehlo.multiply %v2483, %v2482 : tensor<32x200704xf32>
    %v2485 = stablehlo.tanh %v2484 : tensor<32x200704xf32>
    %v2486 = stablehlo.constant dense<1.0> : tensor<32x200704xf32>
    %v2487 = stablehlo.add %v2486, %v2485 : tensor<32x200704xf32>
    %v2488 = stablehlo.constant dense<0.5> : tensor<32x200704xf32>
    %v2489 = stablehlo.multiply %v2488, %v2477 : tensor<32x200704xf32>
    %v2490 = stablehlo.multiply %v2489, %v2487 : tensor<32x200704xf32>
    %v2491 = stablehlo.reshape %v2490 : (tensor<32x200704xf32>) -> tensor<32x4096x7x7xf32>
    %v2492 = stablehlo.convolution(%v2491, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x4096x7x7xf32>, tensor<1024x4096x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v2493 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2494 = stablehlo.add %v2492, %v2493 : tensor<32x1024x7x7xf32>
    %v2495 = stablehlo.reshape %v2494 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2496 = stablehlo.reshape %v2495 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2497 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2498 = stablehlo.multiply %v2496, %v2497 : tensor<32x1024x7x7xf32>
    %v2499 = stablehlo.reshape %v2498 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2500 = stablehlo.add %v2499, %v2433 : tensor<32x50176xf32>
    %v2501 = stablehlo.reshape %v2500 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2502 = stablehlo.convolution(%v2501, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x7x7xf32>) -> tensor<32x1024x7x7xf32>
    %v2503 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2504 = stablehlo.add %v2502, %v2503 : tensor<32x1024x7x7xf32>
    %v2505 = stablehlo.reshape %v2504 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2506 = stablehlo.reshape %v2505 : (tensor<32x50176xf32>) -> tensor<32x1024x49xf32>
    %v2507 = stablehlo.transpose %v2506, dims = [0, 2, 1] : (tensor<32x1024x49xf32>) -> tensor<32x49x1024xf32>
    %v2508 = stablehlo.reshape %v2507 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2509 = stablehlo.reshape %v2508 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2510 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2511 = stablehlo.constant dense<1024.0> : tensor<32x49x1024xf32>
    %v2512 = stablehlo.constant dense<1.0e-6> : tensor<32x49x1024xf32>
    %v2513 = stablehlo.reduce(%v2509 init: %v2510) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2514 = stablehlo.broadcast_in_dim %v2513, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2515 = stablehlo.divide %v2514, %v2511 : tensor<32x49x1024xf32>
    %v2516 = stablehlo.subtract %v2509, %v2515 : tensor<32x49x1024xf32>
    %v2517 = stablehlo.multiply %v2516, %v2516 : tensor<32x49x1024xf32>
    %v2518 = stablehlo.reduce(%v2517 init: %v2510) applies stablehlo.add across dimensions = [2] : (tensor<32x49x1024xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v2519 = stablehlo.broadcast_in_dim %v2518, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x1024xf32>
    %v2520 = stablehlo.divide %v2519, %v2511 : tensor<32x49x1024xf32>
    %v2521 = stablehlo.add %v2520, %v2512 : tensor<32x49x1024xf32>
    %v2522 = stablehlo.rsqrt %v2521 : tensor<32x49x1024xf32>
    %v2523 = stablehlo.multiply %v2516, %v2522 : tensor<32x49x1024xf32>
    %v2524 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2525 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x1024xf32>
    %v2526 = stablehlo.multiply %v2523, %v2524 : tensor<32x49x1024xf32>
    %v2527 = stablehlo.add %v2526, %v2525 : tensor<32x49x1024xf32>
    %v2528 = stablehlo.reshape %v2527 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2529 = stablehlo.reshape %v2528 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2530 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2531 = stablehlo.multiply %v2529, %v2530 : tensor<32x49x1024xf32>
    %v2532 = stablehlo.reshape %v2531 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2533 = stablehlo.reshape %v2532 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2534 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<1024xf32>) -> tensor<32x49x1024xf32>
    %v2535 = stablehlo.add %v2533, %v2534 : tensor<32x49x1024xf32>
    %v2536 = stablehlo.reshape %v2535 : (tensor<32x49x1024xf32>) -> tensor<32x50176xf32>
    %v2537 = stablehlo.reshape %v2536 : (tensor<32x50176xf32>) -> tensor<32x49x1024xf32>
    %v2538 = stablehlo.transpose %v2537, dims = [0, 2, 1] : (tensor<32x49x1024xf32>) -> tensor<32x1024x49xf32>
    %v2539 = stablehlo.reshape %v2538 : (tensor<32x1024x49xf32>) -> tensor<32x50176xf32>
    %v2540 = stablehlo.reshape %v2539 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2541 = stablehlo.convolution(%v2540, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<4096x1024x1x1xf32>) -> tensor<32x4096x7x7xf32>
    %v2542 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<4096xf32>) -> tensor<32x4096x7x7xf32>
    %v2543 = stablehlo.add %v2541, %v2542 : tensor<32x4096x7x7xf32>
    %v2544 = stablehlo.reshape %v2543 : (tensor<32x4096x7x7xf32>) -> tensor<32x200704xf32>
    %v2545 = stablehlo.multiply %v2544, %v2544 : tensor<32x200704xf32>
    %v2546 = stablehlo.multiply %v2545, %v2544 : tensor<32x200704xf32>
    %v2547 = stablehlo.constant dense<0.044715> : tensor<32x200704xf32>
    %v2548 = stablehlo.multiply %v2547, %v2546 : tensor<32x200704xf32>
    %v2549 = stablehlo.add %v2544, %v2548 : tensor<32x200704xf32>
    %v2550 = stablehlo.constant dense<0.7978845608028654> : tensor<32x200704xf32>
    %v2551 = stablehlo.multiply %v2550, %v2549 : tensor<32x200704xf32>
    %v2552 = stablehlo.tanh %v2551 : tensor<32x200704xf32>
    %v2553 = stablehlo.constant dense<1.0> : tensor<32x200704xf32>
    %v2554 = stablehlo.add %v2553, %v2552 : tensor<32x200704xf32>
    %v2555 = stablehlo.constant dense<0.5> : tensor<32x200704xf32>
    %v2556 = stablehlo.multiply %v2555, %v2544 : tensor<32x200704xf32>
    %v2557 = stablehlo.multiply %v2556, %v2554 : tensor<32x200704xf32>
    %v2558 = stablehlo.reshape %v2557 : (tensor<32x200704xf32>) -> tensor<32x4096x7x7xf32>
    %v2559 = stablehlo.convolution(%v2558, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x4096x7x7xf32>, tensor<1024x4096x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v2560 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2561 = stablehlo.add %v2559, %v2560 : tensor<32x1024x7x7xf32>
    %v2562 = stablehlo.reshape %v2561 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2563 = stablehlo.reshape %v2562 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2564 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2565 = stablehlo.multiply %v2563, %v2564 : tensor<32x1024x7x7xf32>
    %v2566 = stablehlo.reshape %v2565 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2567 = stablehlo.add %v2566, %v2500 : tensor<32x50176xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2570 = stablehlo.reduce(%v2568 init: %v2569) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<32x1024xf32>
    %v2571 = stablehlo.constant dense<49.0> : tensor<32x1024xf32>
    %v2572 = stablehlo.divide %v2570, %v2571 : tensor<32x1024xf32>
    %v2573 = stablehlo.dot_general %v2572, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1024xf32>, tensor<1024x1000xf32>) -> tensor<32x1000xf32>
    %v2574 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v2575 = stablehlo.add %v2573, %v2574 : tensor<32x1000xf32>
    return %v2575 : tensor<32x1000xf32>
  }
}
