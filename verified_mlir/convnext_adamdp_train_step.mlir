module @m {
  func.func @convnext_adamdp_train_step(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<f32>, %s0b0nbt: tensor<f32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<f32>, %s0b1nbt: tensor<f32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<f32>, %s0b2nbt: tensor<f32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<f32>, %d0nbt: tensor<f32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<f32>, %s1b0nbt: tensor<f32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<f32>, %s1b1nbt: tensor<f32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<f32>, %s1b2nbt: tensor<f32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<f32>, %d1nbt: tensor<f32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<f32>, %s2b0nbt: tensor<f32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<f32>, %s2b1nbt: tensor<f32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<f32>, %s2b2nbt: tensor<f32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<f32>, %s2b3nbt: tensor<f32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<f32>, %s2b4nbt: tensor<f32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<f32>, %s2b5nbt: tensor<f32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<f32>, %s2b6nbt: tensor<f32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<f32>, %s2b7nbt: tensor<f32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<f32>, %s2b8nbt: tensor<f32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<f32>, %d2nbt: tensor<f32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<f32>, %s3b0nbt: tensor<f32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<f32>, %s3b1nbt: tensor<f32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<f32>, %s3b2nbt: tensor<f32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<f32>, %hnbt: tensor<f32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>, %psWm: tensor<96x3x4x4xf32>, %psbm: tensor<96xf32>, %s0b0dWm: tensor<96x1x7x7xf32>, %s0b0dbm: tensor<96xf32>, %s0b0ngm: tensor<f32>, %s0b0nbtm: tensor<f32>, %s0b0eWm: tensor<384x96x1x1xf32>, %s0b0ebm: tensor<384xf32>, %s0b0pWm: tensor<96x384x1x1xf32>, %s0b0pbm: tensor<96xf32>, %s0b0lgm: tensor<96xf32>, %s0b1dWm: tensor<96x1x7x7xf32>, %s0b1dbm: tensor<96xf32>, %s0b1ngm: tensor<f32>, %s0b1nbtm: tensor<f32>, %s0b1eWm: tensor<384x96x1x1xf32>, %s0b1ebm: tensor<384xf32>, %s0b1pWm: tensor<96x384x1x1xf32>, %s0b1pbm: tensor<96xf32>, %s0b1lgm: tensor<96xf32>, %s0b2dWm: tensor<96x1x7x7xf32>, %s0b2dbm: tensor<96xf32>, %s0b2ngm: tensor<f32>, %s0b2nbtm: tensor<f32>, %s0b2eWm: tensor<384x96x1x1xf32>, %s0b2ebm: tensor<384xf32>, %s0b2pWm: tensor<96x384x1x1xf32>, %s0b2pbm: tensor<96xf32>, %s0b2lgm: tensor<96xf32>, %d0ngm: tensor<f32>, %d0nbtm: tensor<f32>, %d0Wm: tensor<192x96x2x2xf32>, %d0bm: tensor<192xf32>, %s1b0dWm: tensor<192x1x7x7xf32>, %s1b0dbm: tensor<192xf32>, %s1b0ngm: tensor<f32>, %s1b0nbtm: tensor<f32>, %s1b0eWm: tensor<768x192x1x1xf32>, %s1b0ebm: tensor<768xf32>, %s1b0pWm: tensor<192x768x1x1xf32>, %s1b0pbm: tensor<192xf32>, %s1b0lgm: tensor<192xf32>, %s1b1dWm: tensor<192x1x7x7xf32>, %s1b1dbm: tensor<192xf32>, %s1b1ngm: tensor<f32>, %s1b1nbtm: tensor<f32>, %s1b1eWm: tensor<768x192x1x1xf32>, %s1b1ebm: tensor<768xf32>, %s1b1pWm: tensor<192x768x1x1xf32>, %s1b1pbm: tensor<192xf32>, %s1b1lgm: tensor<192xf32>, %s1b2dWm: tensor<192x1x7x7xf32>, %s1b2dbm: tensor<192xf32>, %s1b2ngm: tensor<f32>, %s1b2nbtm: tensor<f32>, %s1b2eWm: tensor<768x192x1x1xf32>, %s1b2ebm: tensor<768xf32>, %s1b2pWm: tensor<192x768x1x1xf32>, %s1b2pbm: tensor<192xf32>, %s1b2lgm: tensor<192xf32>, %d1ngm: tensor<f32>, %d1nbtm: tensor<f32>, %d1Wm: tensor<384x192x2x2xf32>, %d1bm: tensor<384xf32>, %s2b0dWm: tensor<384x1x7x7xf32>, %s2b0dbm: tensor<384xf32>, %s2b0ngm: tensor<f32>, %s2b0nbtm: tensor<f32>, %s2b0eWm: tensor<1536x384x1x1xf32>, %s2b0ebm: tensor<1536xf32>, %s2b0pWm: tensor<384x1536x1x1xf32>, %s2b0pbm: tensor<384xf32>, %s2b0lgm: tensor<384xf32>, %s2b1dWm: tensor<384x1x7x7xf32>, %s2b1dbm: tensor<384xf32>, %s2b1ngm: tensor<f32>, %s2b1nbtm: tensor<f32>, %s2b1eWm: tensor<1536x384x1x1xf32>, %s2b1ebm: tensor<1536xf32>, %s2b1pWm: tensor<384x1536x1x1xf32>, %s2b1pbm: tensor<384xf32>, %s2b1lgm: tensor<384xf32>, %s2b2dWm: tensor<384x1x7x7xf32>, %s2b2dbm: tensor<384xf32>, %s2b2ngm: tensor<f32>, %s2b2nbtm: tensor<f32>, %s2b2eWm: tensor<1536x384x1x1xf32>, %s2b2ebm: tensor<1536xf32>, %s2b2pWm: tensor<384x1536x1x1xf32>, %s2b2pbm: tensor<384xf32>, %s2b2lgm: tensor<384xf32>, %s2b3dWm: tensor<384x1x7x7xf32>, %s2b3dbm: tensor<384xf32>, %s2b3ngm: tensor<f32>, %s2b3nbtm: tensor<f32>, %s2b3eWm: tensor<1536x384x1x1xf32>, %s2b3ebm: tensor<1536xf32>, %s2b3pWm: tensor<384x1536x1x1xf32>, %s2b3pbm: tensor<384xf32>, %s2b3lgm: tensor<384xf32>, %s2b4dWm: tensor<384x1x7x7xf32>, %s2b4dbm: tensor<384xf32>, %s2b4ngm: tensor<f32>, %s2b4nbtm: tensor<f32>, %s2b4eWm: tensor<1536x384x1x1xf32>, %s2b4ebm: tensor<1536xf32>, %s2b4pWm: tensor<384x1536x1x1xf32>, %s2b4pbm: tensor<384xf32>, %s2b4lgm: tensor<384xf32>, %s2b5dWm: tensor<384x1x7x7xf32>, %s2b5dbm: tensor<384xf32>, %s2b5ngm: tensor<f32>, %s2b5nbtm: tensor<f32>, %s2b5eWm: tensor<1536x384x1x1xf32>, %s2b5ebm: tensor<1536xf32>, %s2b5pWm: tensor<384x1536x1x1xf32>, %s2b5pbm: tensor<384xf32>, %s2b5lgm: tensor<384xf32>, %s2b6dWm: tensor<384x1x7x7xf32>, %s2b6dbm: tensor<384xf32>, %s2b6ngm: tensor<f32>, %s2b6nbtm: tensor<f32>, %s2b6eWm: tensor<1536x384x1x1xf32>, %s2b6ebm: tensor<1536xf32>, %s2b6pWm: tensor<384x1536x1x1xf32>, %s2b6pbm: tensor<384xf32>, %s2b6lgm: tensor<384xf32>, %s2b7dWm: tensor<384x1x7x7xf32>, %s2b7dbm: tensor<384xf32>, %s2b7ngm: tensor<f32>, %s2b7nbtm: tensor<f32>, %s2b7eWm: tensor<1536x384x1x1xf32>, %s2b7ebm: tensor<1536xf32>, %s2b7pWm: tensor<384x1536x1x1xf32>, %s2b7pbm: tensor<384xf32>, %s2b7lgm: tensor<384xf32>, %s2b8dWm: tensor<384x1x7x7xf32>, %s2b8dbm: tensor<384xf32>, %s2b8ngm: tensor<f32>, %s2b8nbtm: tensor<f32>, %s2b8eWm: tensor<1536x384x1x1xf32>, %s2b8ebm: tensor<1536xf32>, %s2b8pWm: tensor<384x1536x1x1xf32>, %s2b8pbm: tensor<384xf32>, %s2b8lgm: tensor<384xf32>, %d2ngm: tensor<f32>, %d2nbtm: tensor<f32>, %d2Wm: tensor<768x384x2x2xf32>, %d2bm: tensor<768xf32>, %s3b0dWm: tensor<768x1x7x7xf32>, %s3b0dbm: tensor<768xf32>, %s3b0ngm: tensor<f32>, %s3b0nbtm: tensor<f32>, %s3b0eWm: tensor<3072x768x1x1xf32>, %s3b0ebm: tensor<3072xf32>, %s3b0pWm: tensor<768x3072x1x1xf32>, %s3b0pbm: tensor<768xf32>, %s3b0lgm: tensor<768xf32>, %s3b1dWm: tensor<768x1x7x7xf32>, %s3b1dbm: tensor<768xf32>, %s3b1ngm: tensor<f32>, %s3b1nbtm: tensor<f32>, %s3b1eWm: tensor<3072x768x1x1xf32>, %s3b1ebm: tensor<3072xf32>, %s3b1pWm: tensor<768x3072x1x1xf32>, %s3b1pbm: tensor<768xf32>, %s3b1lgm: tensor<768xf32>, %s3b2dWm: tensor<768x1x7x7xf32>, %s3b2dbm: tensor<768xf32>, %s3b2ngm: tensor<f32>, %s3b2nbtm: tensor<f32>, %s3b2eWm: tensor<3072x768x1x1xf32>, %s3b2ebm: tensor<3072xf32>, %s3b2pWm: tensor<768x3072x1x1xf32>, %s3b2pbm: tensor<768xf32>, %s3b2lgm: tensor<768xf32>, %hngm: tensor<f32>, %hnbtm: tensor<f32>, %Wdm: tensor<768x10xf32>, %bdm: tensor<10xf32>, %psWv: tensor<96x3x4x4xf32>, %psbv: tensor<96xf32>, %s0b0dWv: tensor<96x1x7x7xf32>, %s0b0dbv: tensor<96xf32>, %s0b0ngv: tensor<f32>, %s0b0nbtv: tensor<f32>, %s0b0eWv: tensor<384x96x1x1xf32>, %s0b0ebv: tensor<384xf32>, %s0b0pWv: tensor<96x384x1x1xf32>, %s0b0pbv: tensor<96xf32>, %s0b0lgv: tensor<96xf32>, %s0b1dWv: tensor<96x1x7x7xf32>, %s0b1dbv: tensor<96xf32>, %s0b1ngv: tensor<f32>, %s0b1nbtv: tensor<f32>, %s0b1eWv: tensor<384x96x1x1xf32>, %s0b1ebv: tensor<384xf32>, %s0b1pWv: tensor<96x384x1x1xf32>, %s0b1pbv: tensor<96xf32>, %s0b1lgv: tensor<96xf32>, %s0b2dWv: tensor<96x1x7x7xf32>, %s0b2dbv: tensor<96xf32>, %s0b2ngv: tensor<f32>, %s0b2nbtv: tensor<f32>, %s0b2eWv: tensor<384x96x1x1xf32>, %s0b2ebv: tensor<384xf32>, %s0b2pWv: tensor<96x384x1x1xf32>, %s0b2pbv: tensor<96xf32>, %s0b2lgv: tensor<96xf32>, %d0ngv: tensor<f32>, %d0nbtv: tensor<f32>, %d0Wv: tensor<192x96x2x2xf32>, %d0bv: tensor<192xf32>, %s1b0dWv: tensor<192x1x7x7xf32>, %s1b0dbv: tensor<192xf32>, %s1b0ngv: tensor<f32>, %s1b0nbtv: tensor<f32>, %s1b0eWv: tensor<768x192x1x1xf32>, %s1b0ebv: tensor<768xf32>, %s1b0pWv: tensor<192x768x1x1xf32>, %s1b0pbv: tensor<192xf32>, %s1b0lgv: tensor<192xf32>, %s1b1dWv: tensor<192x1x7x7xf32>, %s1b1dbv: tensor<192xf32>, %s1b1ngv: tensor<f32>, %s1b1nbtv: tensor<f32>, %s1b1eWv: tensor<768x192x1x1xf32>, %s1b1ebv: tensor<768xf32>, %s1b1pWv: tensor<192x768x1x1xf32>, %s1b1pbv: tensor<192xf32>, %s1b1lgv: tensor<192xf32>, %s1b2dWv: tensor<192x1x7x7xf32>, %s1b2dbv: tensor<192xf32>, %s1b2ngv: tensor<f32>, %s1b2nbtv: tensor<f32>, %s1b2eWv: tensor<768x192x1x1xf32>, %s1b2ebv: tensor<768xf32>, %s1b2pWv: tensor<192x768x1x1xf32>, %s1b2pbv: tensor<192xf32>, %s1b2lgv: tensor<192xf32>, %d1ngv: tensor<f32>, %d1nbtv: tensor<f32>, %d1Wv: tensor<384x192x2x2xf32>, %d1bv: tensor<384xf32>, %s2b0dWv: tensor<384x1x7x7xf32>, %s2b0dbv: tensor<384xf32>, %s2b0ngv: tensor<f32>, %s2b0nbtv: tensor<f32>, %s2b0eWv: tensor<1536x384x1x1xf32>, %s2b0ebv: tensor<1536xf32>, %s2b0pWv: tensor<384x1536x1x1xf32>, %s2b0pbv: tensor<384xf32>, %s2b0lgv: tensor<384xf32>, %s2b1dWv: tensor<384x1x7x7xf32>, %s2b1dbv: tensor<384xf32>, %s2b1ngv: tensor<f32>, %s2b1nbtv: tensor<f32>, %s2b1eWv: tensor<1536x384x1x1xf32>, %s2b1ebv: tensor<1536xf32>, %s2b1pWv: tensor<384x1536x1x1xf32>, %s2b1pbv: tensor<384xf32>, %s2b1lgv: tensor<384xf32>, %s2b2dWv: tensor<384x1x7x7xf32>, %s2b2dbv: tensor<384xf32>, %s2b2ngv: tensor<f32>, %s2b2nbtv: tensor<f32>, %s2b2eWv: tensor<1536x384x1x1xf32>, %s2b2ebv: tensor<1536xf32>, %s2b2pWv: tensor<384x1536x1x1xf32>, %s2b2pbv: tensor<384xf32>, %s2b2lgv: tensor<384xf32>, %s2b3dWv: tensor<384x1x7x7xf32>, %s2b3dbv: tensor<384xf32>, %s2b3ngv: tensor<f32>, %s2b3nbtv: tensor<f32>, %s2b3eWv: tensor<1536x384x1x1xf32>, %s2b3ebv: tensor<1536xf32>, %s2b3pWv: tensor<384x1536x1x1xf32>, %s2b3pbv: tensor<384xf32>, %s2b3lgv: tensor<384xf32>, %s2b4dWv: tensor<384x1x7x7xf32>, %s2b4dbv: tensor<384xf32>, %s2b4ngv: tensor<f32>, %s2b4nbtv: tensor<f32>, %s2b4eWv: tensor<1536x384x1x1xf32>, %s2b4ebv: tensor<1536xf32>, %s2b4pWv: tensor<384x1536x1x1xf32>, %s2b4pbv: tensor<384xf32>, %s2b4lgv: tensor<384xf32>, %s2b5dWv: tensor<384x1x7x7xf32>, %s2b5dbv: tensor<384xf32>, %s2b5ngv: tensor<f32>, %s2b5nbtv: tensor<f32>, %s2b5eWv: tensor<1536x384x1x1xf32>, %s2b5ebv: tensor<1536xf32>, %s2b5pWv: tensor<384x1536x1x1xf32>, %s2b5pbv: tensor<384xf32>, %s2b5lgv: tensor<384xf32>, %s2b6dWv: tensor<384x1x7x7xf32>, %s2b6dbv: tensor<384xf32>, %s2b6ngv: tensor<f32>, %s2b6nbtv: tensor<f32>, %s2b6eWv: tensor<1536x384x1x1xf32>, %s2b6ebv: tensor<1536xf32>, %s2b6pWv: tensor<384x1536x1x1xf32>, %s2b6pbv: tensor<384xf32>, %s2b6lgv: tensor<384xf32>, %s2b7dWv: tensor<384x1x7x7xf32>, %s2b7dbv: tensor<384xf32>, %s2b7ngv: tensor<f32>, %s2b7nbtv: tensor<f32>, %s2b7eWv: tensor<1536x384x1x1xf32>, %s2b7ebv: tensor<1536xf32>, %s2b7pWv: tensor<384x1536x1x1xf32>, %s2b7pbv: tensor<384xf32>, %s2b7lgv: tensor<384xf32>, %s2b8dWv: tensor<384x1x7x7xf32>, %s2b8dbv: tensor<384xf32>, %s2b8ngv: tensor<f32>, %s2b8nbtv: tensor<f32>, %s2b8eWv: tensor<1536x384x1x1xf32>, %s2b8ebv: tensor<1536xf32>, %s2b8pWv: tensor<384x1536x1x1xf32>, %s2b8pbv: tensor<384xf32>, %s2b8lgv: tensor<384xf32>, %d2ngv: tensor<f32>, %d2nbtv: tensor<f32>, %d2Wv: tensor<768x384x2x2xf32>, %d2bv: tensor<768xf32>, %s3b0dWv: tensor<768x1x7x7xf32>, %s3b0dbv: tensor<768xf32>, %s3b0ngv: tensor<f32>, %s3b0nbtv: tensor<f32>, %s3b0eWv: tensor<3072x768x1x1xf32>, %s3b0ebv: tensor<3072xf32>, %s3b0pWv: tensor<768x3072x1x1xf32>, %s3b0pbv: tensor<768xf32>, %s3b0lgv: tensor<768xf32>, %s3b1dWv: tensor<768x1x7x7xf32>, %s3b1dbv: tensor<768xf32>, %s3b1ngv: tensor<f32>, %s3b1nbtv: tensor<f32>, %s3b1eWv: tensor<3072x768x1x1xf32>, %s3b1ebv: tensor<3072xf32>, %s3b1pWv: tensor<768x3072x1x1xf32>, %s3b1pbv: tensor<768xf32>, %s3b1lgv: tensor<768xf32>, %s3b2dWv: tensor<768x1x7x7xf32>, %s3b2dbv: tensor<768xf32>, %s3b2ngv: tensor<f32>, %s3b2nbtv: tensor<f32>, %s3b2eWv: tensor<3072x768x1x1xf32>, %s3b2ebv: tensor<3072xf32>, %s3b2pWv: tensor<768x3072x1x1xf32>, %s3b2pbv: tensor<768xf32>, %s3b2lgv: tensor<768xf32>, %hngv: tensor<f32>, %hnbtv: tensor<f32>, %Wdv: tensor<768x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<32x10xf32>) -> (tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %bsc = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    // ── ConvNeXt-T AdamW train step, DATA-PARALLEL over 2 replicas ──
    // Every line is pretty(verified AST node) EXCEPT the per-parameter `%arsum*`
    // all_reduce / `%armean*` blocks: those are a TRUSTED CARVE-OUT (handoff §5), emitted
    // text outside the faithfulness theorems. Each replica evaluates the same tied graph
    // at the batch it was rendered for; the collective averages that function's gradients
    // over disjoint equal batches. Unlike the BN nets, ConvNeXt normalises with LayerNorm
    // — within one example, never across the batch — so N x b IS 1 x (N.b) here and the
    // §10.3b caveat does not apply.
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
    %v1024 = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %v1025 = stablehlo.multiply %onehot, %v1024 : tensor<32x10xf32>
    %v1026 = stablehlo.add %v1023, %v1025 : tensor<32x10xf32>
    %v1027 = stablehlo.constant dense<-0.010000> : tensor<32x10xf32>
    %v1028 = stablehlo.add %v1026, %v1027 : tensor<32x10xf32>
    %v1029 = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %v1030 = stablehlo.divide %v1028, %v1029 : tensor<32x10xf32>
    %v1031 = stablehlo.dot_general %v1030, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<768x10xf32>) -> tensor<32x768xf32>
    %v1032 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1033 = stablehlo.constant dense<768.0> : tensor<32x768xf32>
    %v1034 = stablehlo.constant dense<1.0e-6> : tensor<32x768xf32>
    %v1035 = stablehlo.reduce(%v996 init: %v1032) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1036 = stablehlo.broadcast_in_dim %v1035, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1037 = stablehlo.divide %v1036, %v1033 : tensor<32x768xf32>
    %v1038 = stablehlo.subtract %v996, %v1037 : tensor<32x768xf32>
    %v1039 = stablehlo.multiply %v1038, %v1038 : tensor<32x768xf32>
    %v1040 = stablehlo.reduce(%v1039 init: %v1032) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1041 = stablehlo.broadcast_in_dim %v1040, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1042 = stablehlo.divide %v1041, %v1033 : tensor<32x768xf32>
    %v1043 = stablehlo.add %v1042, %v1034 : tensor<32x768xf32>
    %v1044 = stablehlo.rsqrt %v1043 : tensor<32x768xf32>
    %v1045 = stablehlo.multiply %v1038, %v1044 : tensor<32x768xf32>
    %v1046 = stablehlo.broadcast_in_dim %hng, dims = [] : (tensor<f32>) -> tensor<32x768xf32>
    %v1047 = stablehlo.multiply %v1046, %v1031 : tensor<32x768xf32>
    %v1048 = stablehlo.reduce(%v1047 init: %v1032) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1049 = stablehlo.broadcast_in_dim %v1048, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1050 = stablehlo.multiply %v1045, %v1047 : tensor<32x768xf32>
    %v1051 = stablehlo.reduce(%v1050 init: %v1032) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1052 = stablehlo.broadcast_in_dim %v1051, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1053 = stablehlo.multiply %v1047, %v1033 : tensor<32x768xf32>
    %v1054 = stablehlo.subtract %v1053, %v1049 : tensor<32x768xf32>
    %v1055 = stablehlo.multiply %v1045, %v1052 : tensor<32x768xf32>
    %v1056 = stablehlo.subtract %v1054, %v1055 : tensor<32x768xf32>
    %v1057 = stablehlo.divide %v1044, %v1033 : tensor<32x768xf32>
    %v1058 = stablehlo.multiply %v1057, %v1056 : tensor<32x768xf32>
    %v1059 = stablehlo.dot_general %v1014, %v1030, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<32x10xf32>) -> tensor<768x10xf32>
    %v1060 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1061 = stablehlo.reduce(%v1030 init: %v1060) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1063 = stablehlo.constant dense<768.0> : tensor<32x768xf32>
    %v1064 = stablehlo.constant dense<1.0e-6> : tensor<32x768xf32>
    %v1065 = stablehlo.reduce(%v996 init: %v1062) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1066 = stablehlo.broadcast_in_dim %v1065, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1067 = stablehlo.divide %v1066, %v1063 : tensor<32x768xf32>
    %v1068 = stablehlo.subtract %v996, %v1067 : tensor<32x768xf32>
    %v1069 = stablehlo.multiply %v1068, %v1068 : tensor<32x768xf32>
    %v1070 = stablehlo.reduce(%v1069 init: %v1062) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %v1071 = stablehlo.broadcast_in_dim %v1070, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %v1072 = stablehlo.divide %v1071, %v1063 : tensor<32x768xf32>
    %v1073 = stablehlo.add %v1072, %v1064 : tensor<32x768xf32>
    %v1074 = stablehlo.rsqrt %v1073 : tensor<32x768xf32>
    %v1075 = stablehlo.multiply %v1068, %v1074 : tensor<32x768xf32>
    %v1076 = stablehlo.multiply %v1031, %v1075 : tensor<32x768xf32>
    %v1077 = stablehlo.reduce(%v1076 init: %v1062) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<f32>
    %v1078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1079 = stablehlo.reduce(%v1031 init: %v1078) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<f32>
    %dgi = stablehlo.reshape %v1058 : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : (tensor<32x768x1x1xf32>) -> tensor<32x768x7x7xf32>
    %dgn = stablehlo.constant dense<49.0> : tensor<32x768x7x7xf32>
    %dgd = stablehlo.divide %dgb, %dgn : tensor<32x768x7x7xf32>
    %dgapf = stablehlo.reshape %dgd : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1080 = stablehlo.reshape %dgapf : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1081 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1082 = stablehlo.multiply %v1080, %v1081 : tensor<32x768x7x7xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1085 = stablehlo.transpose %s3b2pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1086 = stablehlo.reverse %v1085, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1087 = stablehlo.convolution(%v1084, %v1086)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1089 = stablehlo.multiply %v968, %v968 : tensor<32x150528xf32>
    %v1090 = stablehlo.multiply %v1089, %v968 : tensor<32x150528xf32>
    %v1091 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1092 = stablehlo.multiply %v1091, %v1090 : tensor<32x150528xf32>
    %v1093 = stablehlo.add %v968, %v1092 : tensor<32x150528xf32>
    %v1094 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1095 = stablehlo.multiply %v1094, %v1093 : tensor<32x150528xf32>
    %v1096 = stablehlo.tanh %v1095 : tensor<32x150528xf32>
    %v1097 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1098 = stablehlo.add %v1097, %v1096 : tensor<32x150528xf32>
    %v1099 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1100 = stablehlo.multiply %v1099, %v1098 : tensor<32x150528xf32>
    %v1101 = stablehlo.multiply %v1096, %v1096 : tensor<32x150528xf32>
    %v1102 = stablehlo.subtract %v1097, %v1101 : tensor<32x150528xf32>
    %v1103 = stablehlo.multiply %v1099, %v968 : tensor<32x150528xf32>
    %v1104 = stablehlo.multiply %v1103, %v1102 : tensor<32x150528xf32>
    %v1105 = stablehlo.constant dense<0.134145> : tensor<32x150528xf32>
    %v1106 = stablehlo.multiply %v1105, %v1089 : tensor<32x150528xf32>
    %v1107 = stablehlo.add %v1097, %v1106 : tensor<32x150528xf32>
    %v1108 = stablehlo.multiply %v1094, %v1107 : tensor<32x150528xf32>
    %v1109 = stablehlo.multiply %v1104, %v1108 : tensor<32x150528xf32>
    %v1110 = stablehlo.add %v1100, %v1109 : tensor<32x150528xf32>
    %v1111 = stablehlo.multiply %v1088, %v1110 : tensor<32x150528xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1113 = stablehlo.transpose %s3b2eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1114 = stablehlo.reverse %v1113, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1115 = stablehlo.convolution(%v1112, %v1114)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1118 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v1119 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v1120 = stablehlo.reduce(%v945 init: %v1117) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1121 = stablehlo.broadcast_in_dim %v1120, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1122 = stablehlo.divide %v1121, %v1118 : tensor<32x37632xf32>
    %v1123 = stablehlo.subtract %v945, %v1122 : tensor<32x37632xf32>
    %v1124 = stablehlo.multiply %v1123, %v1123 : tensor<32x37632xf32>
    %v1125 = stablehlo.reduce(%v1124 init: %v1117) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1126 = stablehlo.broadcast_in_dim %v1125, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1127 = stablehlo.divide %v1126, %v1118 : tensor<32x37632xf32>
    %v1128 = stablehlo.add %v1127, %v1119 : tensor<32x37632xf32>
    %v1129 = stablehlo.rsqrt %v1128 : tensor<32x37632xf32>
    %v1130 = stablehlo.multiply %v1123, %v1129 : tensor<32x37632xf32>
    %v1131 = stablehlo.broadcast_in_dim %s3b2ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v1132 = stablehlo.multiply %v1131, %v1116 : tensor<32x37632xf32>
    %v1133 = stablehlo.reduce(%v1132 init: %v1117) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1134 = stablehlo.broadcast_in_dim %v1133, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1135 = stablehlo.multiply %v1130, %v1132 : tensor<32x37632xf32>
    %v1136 = stablehlo.reduce(%v1135 init: %v1117) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1137 = stablehlo.broadcast_in_dim %v1136, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1138 = stablehlo.multiply %v1132, %v1118 : tensor<32x37632xf32>
    %v1139 = stablehlo.subtract %v1138, %v1134 : tensor<32x37632xf32>
    %v1140 = stablehlo.multiply %v1130, %v1137 : tensor<32x37632xf32>
    %v1141 = stablehlo.subtract %v1139, %v1140 : tensor<32x37632xf32>
    %v1142 = stablehlo.divide %v1129, %v1118 : tensor<32x37632xf32>
    %v1143 = stablehlo.multiply %v1142, %v1141 : tensor<32x37632xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1145 = stablehlo.reverse %s3b2dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1146 = stablehlo.convolution(%v1144, %v1145)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1147 = stablehlo.reshape %v1146 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1148 = stablehlo.add %v1147, %dgapf : tensor<32x37632xf32>
    %v1149 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1150 = stablehlo.reshape %v986 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1151 = stablehlo.reshape %dgapf : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1152 = stablehlo.multiply %v1150, %v1151 : tensor<32x768x7x7xf32>
    %v1153 = stablehlo.reduce(%v1152 init: %v1149) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1154 = stablehlo.reshape %v981 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1155 = stablehlo.reshape %v1083 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1156 = stablehlo.transpose %v1154, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1157 = stablehlo.transpose %v1155, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1158 = stablehlo.convolution(%v1156, %v1157)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1159 = stablehlo.transpose %v1158, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1160 = stablehlo.reshape %v1083 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1162 = stablehlo.reduce(%v1160 init: %v1161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1163 = stablehlo.reshape %v963 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1164 = stablehlo.reshape %v1111 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1165 = stablehlo.transpose %v1163, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1166 = stablehlo.transpose %v1164, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1167 = stablehlo.convolution(%v1165, %v1166)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1168 = stablehlo.transpose %v1167, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1169 = stablehlo.reshape %v1111 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1171 = stablehlo.reduce(%v1169 init: %v1170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1173 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v1174 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v1175 = stablehlo.reduce(%v945 init: %v1172) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1176 = stablehlo.broadcast_in_dim %v1175, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1177 = stablehlo.divide %v1176, %v1173 : tensor<32x37632xf32>
    %v1178 = stablehlo.subtract %v945, %v1177 : tensor<32x37632xf32>
    %v1179 = stablehlo.multiply %v1178, %v1178 : tensor<32x37632xf32>
    %v1180 = stablehlo.reduce(%v1179 init: %v1172) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1181 = stablehlo.broadcast_in_dim %v1180, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1182 = stablehlo.divide %v1181, %v1173 : tensor<32x37632xf32>
    %v1183 = stablehlo.add %v1182, %v1174 : tensor<32x37632xf32>
    %v1184 = stablehlo.rsqrt %v1183 : tensor<32x37632xf32>
    %v1185 = stablehlo.multiply %v1178, %v1184 : tensor<32x37632xf32>
    %v1186 = stablehlo.multiply %v1116, %v1185 : tensor<32x37632xf32>
    %v1187 = stablehlo.reduce(%v1186 init: %v1172) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %v1188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1189 = stablehlo.reduce(%v1116 init: %v1188) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %v1190 = stablehlo.reshape %v940 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1191 = stablehlo.reshape %v1143 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1192 = stablehlo.transpose %v1190, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1193 = stablehlo.transpose %v1191, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1194 = stablehlo.convolution(%v1192, %v1193)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1196 = stablehlo.reshape %v1143 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1198 = stablehlo.reduce(%v1196 init: %v1197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1199 = stablehlo.reshape %v1148 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1200 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1201 = stablehlo.multiply %v1199, %v1200 : tensor<32x768x7x7xf32>
    %v1202 = stablehlo.reshape %v1201 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1204 = stablehlo.transpose %s3b1pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1205 = stablehlo.reverse %v1204, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1206 = stablehlo.convolution(%v1203, %v1205)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1208 = stablehlo.multiply %v917, %v917 : tensor<32x150528xf32>
    %v1209 = stablehlo.multiply %v1208, %v917 : tensor<32x150528xf32>
    %v1210 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1211 = stablehlo.multiply %v1210, %v1209 : tensor<32x150528xf32>
    %v1212 = stablehlo.add %v917, %v1211 : tensor<32x150528xf32>
    %v1213 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1214 = stablehlo.multiply %v1213, %v1212 : tensor<32x150528xf32>
    %v1215 = stablehlo.tanh %v1214 : tensor<32x150528xf32>
    %v1216 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1217 = stablehlo.add %v1216, %v1215 : tensor<32x150528xf32>
    %v1218 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1219 = stablehlo.multiply %v1218, %v1217 : tensor<32x150528xf32>
    %v1220 = stablehlo.multiply %v1215, %v1215 : tensor<32x150528xf32>
    %v1221 = stablehlo.subtract %v1216, %v1220 : tensor<32x150528xf32>
    %v1222 = stablehlo.multiply %v1218, %v917 : tensor<32x150528xf32>
    %v1223 = stablehlo.multiply %v1222, %v1221 : tensor<32x150528xf32>
    %v1224 = stablehlo.constant dense<0.134145> : tensor<32x150528xf32>
    %v1225 = stablehlo.multiply %v1224, %v1208 : tensor<32x150528xf32>
    %v1226 = stablehlo.add %v1216, %v1225 : tensor<32x150528xf32>
    %v1227 = stablehlo.multiply %v1213, %v1226 : tensor<32x150528xf32>
    %v1228 = stablehlo.multiply %v1223, %v1227 : tensor<32x150528xf32>
    %v1229 = stablehlo.add %v1219, %v1228 : tensor<32x150528xf32>
    %v1230 = stablehlo.multiply %v1207, %v1229 : tensor<32x150528xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1232 = stablehlo.transpose %s3b1eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1233 = stablehlo.reverse %v1232, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1234 = stablehlo.convolution(%v1231, %v1233)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1236 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1237 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v1238 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v1239 = stablehlo.reduce(%v894 init: %v1236) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1240 = stablehlo.broadcast_in_dim %v1239, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1241 = stablehlo.divide %v1240, %v1237 : tensor<32x37632xf32>
    %v1242 = stablehlo.subtract %v894, %v1241 : tensor<32x37632xf32>
    %v1243 = stablehlo.multiply %v1242, %v1242 : tensor<32x37632xf32>
    %v1244 = stablehlo.reduce(%v1243 init: %v1236) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1245 = stablehlo.broadcast_in_dim %v1244, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1246 = stablehlo.divide %v1245, %v1237 : tensor<32x37632xf32>
    %v1247 = stablehlo.add %v1246, %v1238 : tensor<32x37632xf32>
    %v1248 = stablehlo.rsqrt %v1247 : tensor<32x37632xf32>
    %v1249 = stablehlo.multiply %v1242, %v1248 : tensor<32x37632xf32>
    %v1250 = stablehlo.broadcast_in_dim %s3b1ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v1251 = stablehlo.multiply %v1250, %v1235 : tensor<32x37632xf32>
    %v1252 = stablehlo.reduce(%v1251 init: %v1236) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1253 = stablehlo.broadcast_in_dim %v1252, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1254 = stablehlo.multiply %v1249, %v1251 : tensor<32x37632xf32>
    %v1255 = stablehlo.reduce(%v1254 init: %v1236) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1256 = stablehlo.broadcast_in_dim %v1255, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1257 = stablehlo.multiply %v1251, %v1237 : tensor<32x37632xf32>
    %v1258 = stablehlo.subtract %v1257, %v1253 : tensor<32x37632xf32>
    %v1259 = stablehlo.multiply %v1249, %v1256 : tensor<32x37632xf32>
    %v1260 = stablehlo.subtract %v1258, %v1259 : tensor<32x37632xf32>
    %v1261 = stablehlo.divide %v1248, %v1237 : tensor<32x37632xf32>
    %v1262 = stablehlo.multiply %v1261, %v1260 : tensor<32x37632xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1264 = stablehlo.reverse %s3b1dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1265 = stablehlo.convolution(%v1263, %v1264)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1266 = stablehlo.reshape %v1265 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1267 = stablehlo.add %v1266, %v1148 : tensor<32x37632xf32>
    %v1268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1269 = stablehlo.reshape %v935 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1270 = stablehlo.reshape %v1148 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1271 = stablehlo.multiply %v1269, %v1270 : tensor<32x768x7x7xf32>
    %v1272 = stablehlo.reduce(%v1271 init: %v1268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1273 = stablehlo.reshape %v930 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1274 = stablehlo.reshape %v1202 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1275 = stablehlo.transpose %v1273, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1276 = stablehlo.transpose %v1274, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1277 = stablehlo.convolution(%v1275, %v1276)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1278 = stablehlo.transpose %v1277, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1279 = stablehlo.reshape %v1202 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1281 = stablehlo.reduce(%v1279 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1282 = stablehlo.reshape %v912 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1283 = stablehlo.reshape %v1230 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1284 = stablehlo.transpose %v1282, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1285 = stablehlo.transpose %v1283, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1286 = stablehlo.convolution(%v1284, %v1285)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1287 = stablehlo.transpose %v1286, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1288 = stablehlo.reshape %v1230 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1290 = stablehlo.reduce(%v1288 init: %v1289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1292 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v1293 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v1294 = stablehlo.reduce(%v894 init: %v1291) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1295 = stablehlo.broadcast_in_dim %v1294, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1296 = stablehlo.divide %v1295, %v1292 : tensor<32x37632xf32>
    %v1297 = stablehlo.subtract %v894, %v1296 : tensor<32x37632xf32>
    %v1298 = stablehlo.multiply %v1297, %v1297 : tensor<32x37632xf32>
    %v1299 = stablehlo.reduce(%v1298 init: %v1291) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1300 = stablehlo.broadcast_in_dim %v1299, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1301 = stablehlo.divide %v1300, %v1292 : tensor<32x37632xf32>
    %v1302 = stablehlo.add %v1301, %v1293 : tensor<32x37632xf32>
    %v1303 = stablehlo.rsqrt %v1302 : tensor<32x37632xf32>
    %v1304 = stablehlo.multiply %v1297, %v1303 : tensor<32x37632xf32>
    %v1305 = stablehlo.multiply %v1235, %v1304 : tensor<32x37632xf32>
    %v1306 = stablehlo.reduce(%v1305 init: %v1291) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %v1307 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1308 = stablehlo.reduce(%v1235 init: %v1307) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %v1309 = stablehlo.reshape %v889 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1310 = stablehlo.reshape %v1262 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1311 = stablehlo.transpose %v1309, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1312 = stablehlo.transpose %v1310, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1313 = stablehlo.convolution(%v1311, %v1312)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1315 = stablehlo.reshape %v1262 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1317 = stablehlo.reduce(%v1315 init: %v1316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1318 = stablehlo.reshape %v1267 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1319 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1320 = stablehlo.multiply %v1318, %v1319 : tensor<32x768x7x7xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1323 = stablehlo.transpose %s3b0pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1324 = stablehlo.reverse %v1323, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1325 = stablehlo.convolution(%v1322, %v1324)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1327 = stablehlo.multiply %v866, %v866 : tensor<32x150528xf32>
    %v1328 = stablehlo.multiply %v1327, %v866 : tensor<32x150528xf32>
    %v1329 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1330 = stablehlo.multiply %v1329, %v1328 : tensor<32x150528xf32>
    %v1331 = stablehlo.add %v866, %v1330 : tensor<32x150528xf32>
    %v1332 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1333 = stablehlo.multiply %v1332, %v1331 : tensor<32x150528xf32>
    %v1334 = stablehlo.tanh %v1333 : tensor<32x150528xf32>
    %v1335 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1336 = stablehlo.add %v1335, %v1334 : tensor<32x150528xf32>
    %v1337 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1338 = stablehlo.multiply %v1337, %v1336 : tensor<32x150528xf32>
    %v1339 = stablehlo.multiply %v1334, %v1334 : tensor<32x150528xf32>
    %v1340 = stablehlo.subtract %v1335, %v1339 : tensor<32x150528xf32>
    %v1341 = stablehlo.multiply %v1337, %v866 : tensor<32x150528xf32>
    %v1342 = stablehlo.multiply %v1341, %v1340 : tensor<32x150528xf32>
    %v1343 = stablehlo.constant dense<0.134145> : tensor<32x150528xf32>
    %v1344 = stablehlo.multiply %v1343, %v1327 : tensor<32x150528xf32>
    %v1345 = stablehlo.add %v1335, %v1344 : tensor<32x150528xf32>
    %v1346 = stablehlo.multiply %v1332, %v1345 : tensor<32x150528xf32>
    %v1347 = stablehlo.multiply %v1342, %v1346 : tensor<32x150528xf32>
    %v1348 = stablehlo.add %v1338, %v1347 : tensor<32x150528xf32>
    %v1349 = stablehlo.multiply %v1326, %v1348 : tensor<32x150528xf32>
    %v1350 = stablehlo.reshape %v1349 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1351 = stablehlo.transpose %s3b0eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1352 = stablehlo.reverse %v1351, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1353 = stablehlo.convolution(%v1350, %v1352)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1356 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v1357 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v1358 = stablehlo.reduce(%v843 init: %v1355) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1359 = stablehlo.broadcast_in_dim %v1358, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1360 = stablehlo.divide %v1359, %v1356 : tensor<32x37632xf32>
    %v1361 = stablehlo.subtract %v843, %v1360 : tensor<32x37632xf32>
    %v1362 = stablehlo.multiply %v1361, %v1361 : tensor<32x37632xf32>
    %v1363 = stablehlo.reduce(%v1362 init: %v1355) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1364 = stablehlo.broadcast_in_dim %v1363, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1365 = stablehlo.divide %v1364, %v1356 : tensor<32x37632xf32>
    %v1366 = stablehlo.add %v1365, %v1357 : tensor<32x37632xf32>
    %v1367 = stablehlo.rsqrt %v1366 : tensor<32x37632xf32>
    %v1368 = stablehlo.multiply %v1361, %v1367 : tensor<32x37632xf32>
    %v1369 = stablehlo.broadcast_in_dim %s3b0ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %v1370 = stablehlo.multiply %v1369, %v1354 : tensor<32x37632xf32>
    %v1371 = stablehlo.reduce(%v1370 init: %v1355) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1372 = stablehlo.broadcast_in_dim %v1371, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1373 = stablehlo.multiply %v1368, %v1370 : tensor<32x37632xf32>
    %v1374 = stablehlo.reduce(%v1373 init: %v1355) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1375 = stablehlo.broadcast_in_dim %v1374, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1376 = stablehlo.multiply %v1370, %v1356 : tensor<32x37632xf32>
    %v1377 = stablehlo.subtract %v1376, %v1372 : tensor<32x37632xf32>
    %v1378 = stablehlo.multiply %v1368, %v1375 : tensor<32x37632xf32>
    %v1379 = stablehlo.subtract %v1377, %v1378 : tensor<32x37632xf32>
    %v1380 = stablehlo.divide %v1367, %v1356 : tensor<32x37632xf32>
    %v1381 = stablehlo.multiply %v1380, %v1379 : tensor<32x37632xf32>
    %v1382 = stablehlo.reshape %v1381 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1383 = stablehlo.reverse %s3b0dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1384 = stablehlo.convolution(%v1382, %v1383)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1386 = stablehlo.add %v1385, %v1267 : tensor<32x37632xf32>
    %v1387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1388 = stablehlo.reshape %v884 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1389 = stablehlo.reshape %v1267 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1390 = stablehlo.multiply %v1388, %v1389 : tensor<32x768x7x7xf32>
    %v1391 = stablehlo.reduce(%v1390 init: %v1387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1392 = stablehlo.reshape %v879 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1393 = stablehlo.reshape %v1321 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1394 = stablehlo.transpose %v1392, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1395 = stablehlo.transpose %v1393, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1396 = stablehlo.convolution(%v1394, %v1395)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1397 = stablehlo.transpose %v1396, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1398 = stablehlo.reshape %v1321 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1400 = stablehlo.reduce(%v1398 init: %v1399) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1401 = stablehlo.reshape %v861 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1402 = stablehlo.reshape %v1349 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1403 = stablehlo.transpose %v1401, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1404 = stablehlo.transpose %v1402, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1405 = stablehlo.convolution(%v1403, %v1404)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1406 = stablehlo.transpose %v1405, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1407 = stablehlo.reshape %v1349 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1408 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1409 = stablehlo.reduce(%v1407 init: %v1408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1411 = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %v1412 = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %v1413 = stablehlo.reduce(%v843 init: %v1410) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1414 = stablehlo.broadcast_in_dim %v1413, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1415 = stablehlo.divide %v1414, %v1411 : tensor<32x37632xf32>
    %v1416 = stablehlo.subtract %v843, %v1415 : tensor<32x37632xf32>
    %v1417 = stablehlo.multiply %v1416, %v1416 : tensor<32x37632xf32>
    %v1418 = stablehlo.reduce(%v1417 init: %v1410) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %v1419 = stablehlo.broadcast_in_dim %v1418, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1420 = stablehlo.divide %v1419, %v1411 : tensor<32x37632xf32>
    %v1421 = stablehlo.add %v1420, %v1412 : tensor<32x37632xf32>
    %v1422 = stablehlo.rsqrt %v1421 : tensor<32x37632xf32>
    %v1423 = stablehlo.multiply %v1416, %v1422 : tensor<32x37632xf32>
    %v1424 = stablehlo.multiply %v1354, %v1423 : tensor<32x37632xf32>
    %v1425 = stablehlo.reduce(%v1424 init: %v1410) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %v1426 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1427 = stablehlo.reduce(%v1354 init: %v1426) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %v1428 = stablehlo.reshape %v838 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1429 = stablehlo.reshape %v1381 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1430 = stablehlo.transpose %v1428, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1431 = stablehlo.transpose %v1429, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1432 = stablehlo.convolution(%v1430, %v1431)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1434 = stablehlo.reshape %v1381 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1435 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1436 = stablehlo.reduce(%v1434 init: %v1435) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1437 = stablehlo.reshape %v1386 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1439 = stablehlo.pad %v1437, %v1438, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x14x14xf32>
    %v1440 = stablehlo.transpose %d2W, dims = [1, 0, 2, 3] : (tensor<768x384x2x2xf32>) -> tensor<384x768x2x2xf32>
    %v1441 = stablehlo.reverse %v1440, dims = [2, 3] : tensor<384x768x2x2xf32>
    %v1442 = stablehlo.convolution(%v1439, %v1441)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x14x14xf32>, tensor<384x768x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1445 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1446 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1447 = stablehlo.reduce(%v815 init: %v1444) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1448 = stablehlo.broadcast_in_dim %v1447, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1449 = stablehlo.divide %v1448, %v1445 : tensor<32x75264xf32>
    %v1450 = stablehlo.subtract %v815, %v1449 : tensor<32x75264xf32>
    %v1451 = stablehlo.multiply %v1450, %v1450 : tensor<32x75264xf32>
    %v1452 = stablehlo.reduce(%v1451 init: %v1444) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1453 = stablehlo.broadcast_in_dim %v1452, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1454 = stablehlo.divide %v1453, %v1445 : tensor<32x75264xf32>
    %v1455 = stablehlo.add %v1454, %v1446 : tensor<32x75264xf32>
    %v1456 = stablehlo.rsqrt %v1455 : tensor<32x75264xf32>
    %v1457 = stablehlo.multiply %v1450, %v1456 : tensor<32x75264xf32>
    %v1458 = stablehlo.broadcast_in_dim %d2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1459 = stablehlo.multiply %v1458, %v1443 : tensor<32x75264xf32>
    %v1460 = stablehlo.reduce(%v1459 init: %v1444) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1461 = stablehlo.broadcast_in_dim %v1460, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1462 = stablehlo.multiply %v1457, %v1459 : tensor<32x75264xf32>
    %v1463 = stablehlo.reduce(%v1462 init: %v1444) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1464 = stablehlo.broadcast_in_dim %v1463, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1465 = stablehlo.multiply %v1459, %v1445 : tensor<32x75264xf32>
    %v1466 = stablehlo.subtract %v1465, %v1461 : tensor<32x75264xf32>
    %v1467 = stablehlo.multiply %v1457, %v1464 : tensor<32x75264xf32>
    %v1468 = stablehlo.subtract %v1466, %v1467 : tensor<32x75264xf32>
    %v1469 = stablehlo.divide %v1456, %v1445 : tensor<32x75264xf32>
    %v1470 = stablehlo.multiply %v1469, %v1468 : tensor<32x75264xf32>
    %v1471 = stablehlo.reshape %v1386 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1472 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1473 = stablehlo.reduce(%v1471 init: %v1472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1475 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1476 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1477 = stablehlo.reduce(%v815 init: %v1474) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1478 = stablehlo.broadcast_in_dim %v1477, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1479 = stablehlo.divide %v1478, %v1475 : tensor<32x75264xf32>
    %v1480 = stablehlo.subtract %v815, %v1479 : tensor<32x75264xf32>
    %v1481 = stablehlo.multiply %v1480, %v1480 : tensor<32x75264xf32>
    %v1482 = stablehlo.reduce(%v1481 init: %v1474) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1483 = stablehlo.broadcast_in_dim %v1482, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1484 = stablehlo.divide %v1483, %v1475 : tensor<32x75264xf32>
    %v1485 = stablehlo.add %v1484, %v1476 : tensor<32x75264xf32>
    %v1486 = stablehlo.rsqrt %v1485 : tensor<32x75264xf32>
    %v1487 = stablehlo.multiply %v1480, %v1486 : tensor<32x75264xf32>
    %v1488 = stablehlo.multiply %v1443, %v1487 : tensor<32x75264xf32>
    %v1489 = stablehlo.reduce(%v1488 init: %v1474) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1491 = stablehlo.reduce(%v1443 init: %v1490) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1492 = stablehlo.reshape %v833 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1493 = stablehlo.reshape %v1386 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1495 = stablehlo.pad %v1493, %v1494, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x13x13xf32>
    %v1496 = stablehlo.transpose %v1492, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1497 = stablehlo.transpose %v1495, dims = [1, 0, 2, 3] : (tensor<32x768x13x13xf32>) -> tensor<768x32x13x13xf32>
    %v1498 = stablehlo.convolution(%v1496, %v1497)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<768x32x13x13xf32>) -> tensor<384x768x2x2xf32>
    %v1499 = stablehlo.transpose %v1498, dims = [1, 0, 2, 3] : (tensor<384x768x2x2xf32>) -> tensor<768x384x2x2xf32>
    %v1500 = stablehlo.reshape %v1470 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1501 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1502 = stablehlo.multiply %v1500, %v1501 : tensor<32x384x14x14xf32>
    %v1503 = stablehlo.reshape %v1502 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1504 = stablehlo.reshape %v1503 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1505 = stablehlo.transpose %s2b8pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1506 = stablehlo.reverse %v1505, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1507 = stablehlo.convolution(%v1504, %v1506)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1508 = stablehlo.reshape %v1507 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1509 = stablehlo.multiply %v792, %v792 : tensor<32x301056xf32>
    %v1510 = stablehlo.multiply %v1509, %v792 : tensor<32x301056xf32>
    %v1511 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1512 = stablehlo.multiply %v1511, %v1510 : tensor<32x301056xf32>
    %v1513 = stablehlo.add %v792, %v1512 : tensor<32x301056xf32>
    %v1514 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1515 = stablehlo.multiply %v1514, %v1513 : tensor<32x301056xf32>
    %v1516 = stablehlo.tanh %v1515 : tensor<32x301056xf32>
    %v1517 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1518 = stablehlo.add %v1517, %v1516 : tensor<32x301056xf32>
    %v1519 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1520 = stablehlo.multiply %v1519, %v1518 : tensor<32x301056xf32>
    %v1521 = stablehlo.multiply %v1516, %v1516 : tensor<32x301056xf32>
    %v1522 = stablehlo.subtract %v1517, %v1521 : tensor<32x301056xf32>
    %v1523 = stablehlo.multiply %v1519, %v792 : tensor<32x301056xf32>
    %v1524 = stablehlo.multiply %v1523, %v1522 : tensor<32x301056xf32>
    %v1525 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1526 = stablehlo.multiply %v1525, %v1509 : tensor<32x301056xf32>
    %v1527 = stablehlo.add %v1517, %v1526 : tensor<32x301056xf32>
    %v1528 = stablehlo.multiply %v1514, %v1527 : tensor<32x301056xf32>
    %v1529 = stablehlo.multiply %v1524, %v1528 : tensor<32x301056xf32>
    %v1530 = stablehlo.add %v1520, %v1529 : tensor<32x301056xf32>
    %v1531 = stablehlo.multiply %v1508, %v1530 : tensor<32x301056xf32>
    %v1532 = stablehlo.reshape %v1531 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1533 = stablehlo.transpose %s2b8eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1534 = stablehlo.reverse %v1533, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1535 = stablehlo.convolution(%v1532, %v1534)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1536 = stablehlo.reshape %v1535 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1538 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1539 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1540 = stablehlo.reduce(%v769 init: %v1537) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1541 = stablehlo.broadcast_in_dim %v1540, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1542 = stablehlo.divide %v1541, %v1538 : tensor<32x75264xf32>
    %v1543 = stablehlo.subtract %v769, %v1542 : tensor<32x75264xf32>
    %v1544 = stablehlo.multiply %v1543, %v1543 : tensor<32x75264xf32>
    %v1545 = stablehlo.reduce(%v1544 init: %v1537) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1546 = stablehlo.broadcast_in_dim %v1545, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1547 = stablehlo.divide %v1546, %v1538 : tensor<32x75264xf32>
    %v1548 = stablehlo.add %v1547, %v1539 : tensor<32x75264xf32>
    %v1549 = stablehlo.rsqrt %v1548 : tensor<32x75264xf32>
    %v1550 = stablehlo.multiply %v1543, %v1549 : tensor<32x75264xf32>
    %v1551 = stablehlo.broadcast_in_dim %s2b8ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1552 = stablehlo.multiply %v1551, %v1536 : tensor<32x75264xf32>
    %v1553 = stablehlo.reduce(%v1552 init: %v1537) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1554 = stablehlo.broadcast_in_dim %v1553, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1555 = stablehlo.multiply %v1550, %v1552 : tensor<32x75264xf32>
    %v1556 = stablehlo.reduce(%v1555 init: %v1537) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1557 = stablehlo.broadcast_in_dim %v1556, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1558 = stablehlo.multiply %v1552, %v1538 : tensor<32x75264xf32>
    %v1559 = stablehlo.subtract %v1558, %v1554 : tensor<32x75264xf32>
    %v1560 = stablehlo.multiply %v1550, %v1557 : tensor<32x75264xf32>
    %v1561 = stablehlo.subtract %v1559, %v1560 : tensor<32x75264xf32>
    %v1562 = stablehlo.divide %v1549, %v1538 : tensor<32x75264xf32>
    %v1563 = stablehlo.multiply %v1562, %v1561 : tensor<32x75264xf32>
    %v1564 = stablehlo.reshape %v1563 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1565 = stablehlo.reverse %s2b8dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1566 = stablehlo.convolution(%v1564, %v1565)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1567 = stablehlo.reshape %v1566 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1568 = stablehlo.add %v1567, %v1470 : tensor<32x75264xf32>
    %v1569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1570 = stablehlo.reshape %v810 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1571 = stablehlo.reshape %v1470 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1572 = stablehlo.multiply %v1570, %v1571 : tensor<32x384x14x14xf32>
    %v1573 = stablehlo.reduce(%v1572 init: %v1569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1574 = stablehlo.reshape %v805 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1575 = stablehlo.reshape %v1503 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1576 = stablehlo.transpose %v1574, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1577 = stablehlo.transpose %v1575, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1578 = stablehlo.convolution(%v1576, %v1577)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1579 = stablehlo.transpose %v1578, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1580 = stablehlo.reshape %v1503 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1582 = stablehlo.reduce(%v1580 init: %v1581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1583 = stablehlo.reshape %v787 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1584 = stablehlo.reshape %v1531 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1585 = stablehlo.transpose %v1583, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1586 = stablehlo.transpose %v1584, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1587 = stablehlo.convolution(%v1585, %v1586)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1588 = stablehlo.transpose %v1587, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1589 = stablehlo.reshape %v1531 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1591 = stablehlo.reduce(%v1589 init: %v1590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1593 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1594 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1595 = stablehlo.reduce(%v769 init: %v1592) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1596 = stablehlo.broadcast_in_dim %v1595, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1597 = stablehlo.divide %v1596, %v1593 : tensor<32x75264xf32>
    %v1598 = stablehlo.subtract %v769, %v1597 : tensor<32x75264xf32>
    %v1599 = stablehlo.multiply %v1598, %v1598 : tensor<32x75264xf32>
    %v1600 = stablehlo.reduce(%v1599 init: %v1592) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1601 = stablehlo.broadcast_in_dim %v1600, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1602 = stablehlo.divide %v1601, %v1593 : tensor<32x75264xf32>
    %v1603 = stablehlo.add %v1602, %v1594 : tensor<32x75264xf32>
    %v1604 = stablehlo.rsqrt %v1603 : tensor<32x75264xf32>
    %v1605 = stablehlo.multiply %v1598, %v1604 : tensor<32x75264xf32>
    %v1606 = stablehlo.multiply %v1536, %v1605 : tensor<32x75264xf32>
    %v1607 = stablehlo.reduce(%v1606 init: %v1592) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1609 = stablehlo.reduce(%v1536 init: %v1608) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1610 = stablehlo.reshape %v764 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1611 = stablehlo.reshape %v1563 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1612 = stablehlo.transpose %v1610, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1613 = stablehlo.transpose %v1611, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1614 = stablehlo.convolution(%v1612, %v1613)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v1615 = stablehlo.reshape %v1614 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v1616 = stablehlo.reshape %v1563 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1618 = stablehlo.reduce(%v1616 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1619 = stablehlo.reshape %v1568 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1620 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1621 = stablehlo.multiply %v1619, %v1620 : tensor<32x384x14x14xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1623 = stablehlo.reshape %v1622 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1624 = stablehlo.transpose %s2b7pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1625 = stablehlo.reverse %v1624, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1626 = stablehlo.convolution(%v1623, %v1625)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1627 = stablehlo.reshape %v1626 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1628 = stablehlo.multiply %v741, %v741 : tensor<32x301056xf32>
    %v1629 = stablehlo.multiply %v1628, %v741 : tensor<32x301056xf32>
    %v1630 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1631 = stablehlo.multiply %v1630, %v1629 : tensor<32x301056xf32>
    %v1632 = stablehlo.add %v741, %v1631 : tensor<32x301056xf32>
    %v1633 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1634 = stablehlo.multiply %v1633, %v1632 : tensor<32x301056xf32>
    %v1635 = stablehlo.tanh %v1634 : tensor<32x301056xf32>
    %v1636 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1637 = stablehlo.add %v1636, %v1635 : tensor<32x301056xf32>
    %v1638 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1639 = stablehlo.multiply %v1638, %v1637 : tensor<32x301056xf32>
    %v1640 = stablehlo.multiply %v1635, %v1635 : tensor<32x301056xf32>
    %v1641 = stablehlo.subtract %v1636, %v1640 : tensor<32x301056xf32>
    %v1642 = stablehlo.multiply %v1638, %v741 : tensor<32x301056xf32>
    %v1643 = stablehlo.multiply %v1642, %v1641 : tensor<32x301056xf32>
    %v1644 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1645 = stablehlo.multiply %v1644, %v1628 : tensor<32x301056xf32>
    %v1646 = stablehlo.add %v1636, %v1645 : tensor<32x301056xf32>
    %v1647 = stablehlo.multiply %v1633, %v1646 : tensor<32x301056xf32>
    %v1648 = stablehlo.multiply %v1643, %v1647 : tensor<32x301056xf32>
    %v1649 = stablehlo.add %v1639, %v1648 : tensor<32x301056xf32>
    %v1650 = stablehlo.multiply %v1627, %v1649 : tensor<32x301056xf32>
    %v1651 = stablehlo.reshape %v1650 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1652 = stablehlo.transpose %s2b7eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1653 = stablehlo.reverse %v1652, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1654 = stablehlo.convolution(%v1651, %v1653)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1655 = stablehlo.reshape %v1654 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1657 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1658 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1659 = stablehlo.reduce(%v718 init: %v1656) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1660 = stablehlo.broadcast_in_dim %v1659, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1661 = stablehlo.divide %v1660, %v1657 : tensor<32x75264xf32>
    %v1662 = stablehlo.subtract %v718, %v1661 : tensor<32x75264xf32>
    %v1663 = stablehlo.multiply %v1662, %v1662 : tensor<32x75264xf32>
    %v1664 = stablehlo.reduce(%v1663 init: %v1656) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1665 = stablehlo.broadcast_in_dim %v1664, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1666 = stablehlo.divide %v1665, %v1657 : tensor<32x75264xf32>
    %v1667 = stablehlo.add %v1666, %v1658 : tensor<32x75264xf32>
    %v1668 = stablehlo.rsqrt %v1667 : tensor<32x75264xf32>
    %v1669 = stablehlo.multiply %v1662, %v1668 : tensor<32x75264xf32>
    %v1670 = stablehlo.broadcast_in_dim %s2b7ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1671 = stablehlo.multiply %v1670, %v1655 : tensor<32x75264xf32>
    %v1672 = stablehlo.reduce(%v1671 init: %v1656) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1673 = stablehlo.broadcast_in_dim %v1672, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1674 = stablehlo.multiply %v1669, %v1671 : tensor<32x75264xf32>
    %v1675 = stablehlo.reduce(%v1674 init: %v1656) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1676 = stablehlo.broadcast_in_dim %v1675, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1677 = stablehlo.multiply %v1671, %v1657 : tensor<32x75264xf32>
    %v1678 = stablehlo.subtract %v1677, %v1673 : tensor<32x75264xf32>
    %v1679 = stablehlo.multiply %v1669, %v1676 : tensor<32x75264xf32>
    %v1680 = stablehlo.subtract %v1678, %v1679 : tensor<32x75264xf32>
    %v1681 = stablehlo.divide %v1668, %v1657 : tensor<32x75264xf32>
    %v1682 = stablehlo.multiply %v1681, %v1680 : tensor<32x75264xf32>
    %v1683 = stablehlo.reshape %v1682 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1684 = stablehlo.reverse %s2b7dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1685 = stablehlo.convolution(%v1683, %v1684)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1686 = stablehlo.reshape %v1685 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1687 = stablehlo.add %v1686, %v1568 : tensor<32x75264xf32>
    %v1688 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1689 = stablehlo.reshape %v759 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1690 = stablehlo.reshape %v1568 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1691 = stablehlo.multiply %v1689, %v1690 : tensor<32x384x14x14xf32>
    %v1692 = stablehlo.reduce(%v1691 init: %v1688) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1693 = stablehlo.reshape %v754 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1694 = stablehlo.reshape %v1622 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1695 = stablehlo.transpose %v1693, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1696 = stablehlo.transpose %v1694, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1697 = stablehlo.convolution(%v1695, %v1696)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1698 = stablehlo.transpose %v1697, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1699 = stablehlo.reshape %v1622 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1700 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1701 = stablehlo.reduce(%v1699 init: %v1700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1702 = stablehlo.reshape %v736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1703 = stablehlo.reshape %v1650 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1704 = stablehlo.transpose %v1702, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1705 = stablehlo.transpose %v1703, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1706 = stablehlo.convolution(%v1704, %v1705)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1707 = stablehlo.transpose %v1706, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1708 = stablehlo.reshape %v1650 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1710 = stablehlo.reduce(%v1708 init: %v1709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1712 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1713 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1714 = stablehlo.reduce(%v718 init: %v1711) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1715 = stablehlo.broadcast_in_dim %v1714, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1716 = stablehlo.divide %v1715, %v1712 : tensor<32x75264xf32>
    %v1717 = stablehlo.subtract %v718, %v1716 : tensor<32x75264xf32>
    %v1718 = stablehlo.multiply %v1717, %v1717 : tensor<32x75264xf32>
    %v1719 = stablehlo.reduce(%v1718 init: %v1711) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1720 = stablehlo.broadcast_in_dim %v1719, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1721 = stablehlo.divide %v1720, %v1712 : tensor<32x75264xf32>
    %v1722 = stablehlo.add %v1721, %v1713 : tensor<32x75264xf32>
    %v1723 = stablehlo.rsqrt %v1722 : tensor<32x75264xf32>
    %v1724 = stablehlo.multiply %v1717, %v1723 : tensor<32x75264xf32>
    %v1725 = stablehlo.multiply %v1655, %v1724 : tensor<32x75264xf32>
    %v1726 = stablehlo.reduce(%v1725 init: %v1711) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1727 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1728 = stablehlo.reduce(%v1655 init: %v1727) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1729 = stablehlo.reshape %v713 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1730 = stablehlo.reshape %v1682 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1731 = stablehlo.transpose %v1729, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1732 = stablehlo.transpose %v1730, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1733 = stablehlo.convolution(%v1731, %v1732)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v1734 = stablehlo.reshape %v1733 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v1735 = stablehlo.reshape %v1682 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1736 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1737 = stablehlo.reduce(%v1735 init: %v1736) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1738 = stablehlo.reshape %v1687 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1739 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1740 = stablehlo.multiply %v1738, %v1739 : tensor<32x384x14x14xf32>
    %v1741 = stablehlo.reshape %v1740 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1742 = stablehlo.reshape %v1741 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1743 = stablehlo.transpose %s2b6pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1744 = stablehlo.reverse %v1743, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1745 = stablehlo.convolution(%v1742, %v1744)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1746 = stablehlo.reshape %v1745 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1747 = stablehlo.multiply %v690, %v690 : tensor<32x301056xf32>
    %v1748 = stablehlo.multiply %v1747, %v690 : tensor<32x301056xf32>
    %v1749 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1750 = stablehlo.multiply %v1749, %v1748 : tensor<32x301056xf32>
    %v1751 = stablehlo.add %v690, %v1750 : tensor<32x301056xf32>
    %v1752 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1753 = stablehlo.multiply %v1752, %v1751 : tensor<32x301056xf32>
    %v1754 = stablehlo.tanh %v1753 : tensor<32x301056xf32>
    %v1755 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1756 = stablehlo.add %v1755, %v1754 : tensor<32x301056xf32>
    %v1757 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1758 = stablehlo.multiply %v1757, %v1756 : tensor<32x301056xf32>
    %v1759 = stablehlo.multiply %v1754, %v1754 : tensor<32x301056xf32>
    %v1760 = stablehlo.subtract %v1755, %v1759 : tensor<32x301056xf32>
    %v1761 = stablehlo.multiply %v1757, %v690 : tensor<32x301056xf32>
    %v1762 = stablehlo.multiply %v1761, %v1760 : tensor<32x301056xf32>
    %v1763 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1764 = stablehlo.multiply %v1763, %v1747 : tensor<32x301056xf32>
    %v1765 = stablehlo.add %v1755, %v1764 : tensor<32x301056xf32>
    %v1766 = stablehlo.multiply %v1752, %v1765 : tensor<32x301056xf32>
    %v1767 = stablehlo.multiply %v1762, %v1766 : tensor<32x301056xf32>
    %v1768 = stablehlo.add %v1758, %v1767 : tensor<32x301056xf32>
    %v1769 = stablehlo.multiply %v1746, %v1768 : tensor<32x301056xf32>
    %v1770 = stablehlo.reshape %v1769 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1771 = stablehlo.transpose %s2b6eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1772 = stablehlo.reverse %v1771, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1773 = stablehlo.convolution(%v1770, %v1772)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1774 = stablehlo.reshape %v1773 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1775 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1776 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1777 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1778 = stablehlo.reduce(%v667 init: %v1775) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1779 = stablehlo.broadcast_in_dim %v1778, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1780 = stablehlo.divide %v1779, %v1776 : tensor<32x75264xf32>
    %v1781 = stablehlo.subtract %v667, %v1780 : tensor<32x75264xf32>
    %v1782 = stablehlo.multiply %v1781, %v1781 : tensor<32x75264xf32>
    %v1783 = stablehlo.reduce(%v1782 init: %v1775) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1784 = stablehlo.broadcast_in_dim %v1783, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1785 = stablehlo.divide %v1784, %v1776 : tensor<32x75264xf32>
    %v1786 = stablehlo.add %v1785, %v1777 : tensor<32x75264xf32>
    %v1787 = stablehlo.rsqrt %v1786 : tensor<32x75264xf32>
    %v1788 = stablehlo.multiply %v1781, %v1787 : tensor<32x75264xf32>
    %v1789 = stablehlo.broadcast_in_dim %s2b6ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1790 = stablehlo.multiply %v1789, %v1774 : tensor<32x75264xf32>
    %v1791 = stablehlo.reduce(%v1790 init: %v1775) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1792 = stablehlo.broadcast_in_dim %v1791, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1793 = stablehlo.multiply %v1788, %v1790 : tensor<32x75264xf32>
    %v1794 = stablehlo.reduce(%v1793 init: %v1775) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1795 = stablehlo.broadcast_in_dim %v1794, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1796 = stablehlo.multiply %v1790, %v1776 : tensor<32x75264xf32>
    %v1797 = stablehlo.subtract %v1796, %v1792 : tensor<32x75264xf32>
    %v1798 = stablehlo.multiply %v1788, %v1795 : tensor<32x75264xf32>
    %v1799 = stablehlo.subtract %v1797, %v1798 : tensor<32x75264xf32>
    %v1800 = stablehlo.divide %v1787, %v1776 : tensor<32x75264xf32>
    %v1801 = stablehlo.multiply %v1800, %v1799 : tensor<32x75264xf32>
    %v1802 = stablehlo.reshape %v1801 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1803 = stablehlo.reverse %s2b6dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1804 = stablehlo.convolution(%v1802, %v1803)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1805 = stablehlo.reshape %v1804 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1806 = stablehlo.add %v1805, %v1687 : tensor<32x75264xf32>
    %v1807 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1808 = stablehlo.reshape %v708 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1809 = stablehlo.reshape %v1687 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1810 = stablehlo.multiply %v1808, %v1809 : tensor<32x384x14x14xf32>
    %v1811 = stablehlo.reduce(%v1810 init: %v1807) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1812 = stablehlo.reshape %v703 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1813 = stablehlo.reshape %v1741 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1814 = stablehlo.transpose %v1812, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1815 = stablehlo.transpose %v1813, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1816 = stablehlo.convolution(%v1814, %v1815)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1817 = stablehlo.transpose %v1816, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1818 = stablehlo.reshape %v1741 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1819 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1820 = stablehlo.reduce(%v1818 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1821 = stablehlo.reshape %v685 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1822 = stablehlo.reshape %v1769 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1823 = stablehlo.transpose %v1821, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1824 = stablehlo.transpose %v1822, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1825 = stablehlo.convolution(%v1823, %v1824)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1826 = stablehlo.transpose %v1825, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1827 = stablehlo.reshape %v1769 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1829 = stablehlo.reduce(%v1827 init: %v1828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1831 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1832 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1833 = stablehlo.reduce(%v667 init: %v1830) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1834 = stablehlo.broadcast_in_dim %v1833, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1835 = stablehlo.divide %v1834, %v1831 : tensor<32x75264xf32>
    %v1836 = stablehlo.subtract %v667, %v1835 : tensor<32x75264xf32>
    %v1837 = stablehlo.multiply %v1836, %v1836 : tensor<32x75264xf32>
    %v1838 = stablehlo.reduce(%v1837 init: %v1830) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1839 = stablehlo.broadcast_in_dim %v1838, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1840 = stablehlo.divide %v1839, %v1831 : tensor<32x75264xf32>
    %v1841 = stablehlo.add %v1840, %v1832 : tensor<32x75264xf32>
    %v1842 = stablehlo.rsqrt %v1841 : tensor<32x75264xf32>
    %v1843 = stablehlo.multiply %v1836, %v1842 : tensor<32x75264xf32>
    %v1844 = stablehlo.multiply %v1774, %v1843 : tensor<32x75264xf32>
    %v1845 = stablehlo.reduce(%v1844 init: %v1830) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1847 = stablehlo.reduce(%v1774 init: %v1846) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1848 = stablehlo.reshape %v662 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1849 = stablehlo.reshape %v1801 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1850 = stablehlo.transpose %v1848, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1851 = stablehlo.transpose %v1849, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1852 = stablehlo.convolution(%v1850, %v1851)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v1854 = stablehlo.reshape %v1801 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1856 = stablehlo.reduce(%v1854 init: %v1855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1857 = stablehlo.reshape %v1806 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1858 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1859 = stablehlo.multiply %v1857, %v1858 : tensor<32x384x14x14xf32>
    %v1860 = stablehlo.reshape %v1859 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1861 = stablehlo.reshape %v1860 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1862 = stablehlo.transpose %s2b5pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1863 = stablehlo.reverse %v1862, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1864 = stablehlo.convolution(%v1861, %v1863)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1865 = stablehlo.reshape %v1864 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1866 = stablehlo.multiply %v639, %v639 : tensor<32x301056xf32>
    %v1867 = stablehlo.multiply %v1866, %v639 : tensor<32x301056xf32>
    %v1868 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1869 = stablehlo.multiply %v1868, %v1867 : tensor<32x301056xf32>
    %v1870 = stablehlo.add %v639, %v1869 : tensor<32x301056xf32>
    %v1871 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1872 = stablehlo.multiply %v1871, %v1870 : tensor<32x301056xf32>
    %v1873 = stablehlo.tanh %v1872 : tensor<32x301056xf32>
    %v1874 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1875 = stablehlo.add %v1874, %v1873 : tensor<32x301056xf32>
    %v1876 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1877 = stablehlo.multiply %v1876, %v1875 : tensor<32x301056xf32>
    %v1878 = stablehlo.multiply %v1873, %v1873 : tensor<32x301056xf32>
    %v1879 = stablehlo.subtract %v1874, %v1878 : tensor<32x301056xf32>
    %v1880 = stablehlo.multiply %v1876, %v639 : tensor<32x301056xf32>
    %v1881 = stablehlo.multiply %v1880, %v1879 : tensor<32x301056xf32>
    %v1882 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1883 = stablehlo.multiply %v1882, %v1866 : tensor<32x301056xf32>
    %v1884 = stablehlo.add %v1874, %v1883 : tensor<32x301056xf32>
    %v1885 = stablehlo.multiply %v1871, %v1884 : tensor<32x301056xf32>
    %v1886 = stablehlo.multiply %v1881, %v1885 : tensor<32x301056xf32>
    %v1887 = stablehlo.add %v1877, %v1886 : tensor<32x301056xf32>
    %v1888 = stablehlo.multiply %v1865, %v1887 : tensor<32x301056xf32>
    %v1889 = stablehlo.reshape %v1888 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1890 = stablehlo.transpose %s2b5eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1891 = stablehlo.reverse %v1890, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1892 = stablehlo.convolution(%v1889, %v1891)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1893 = stablehlo.reshape %v1892 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1895 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1896 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1897 = stablehlo.reduce(%v616 init: %v1894) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1898 = stablehlo.broadcast_in_dim %v1897, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1899 = stablehlo.divide %v1898, %v1895 : tensor<32x75264xf32>
    %v1900 = stablehlo.subtract %v616, %v1899 : tensor<32x75264xf32>
    %v1901 = stablehlo.multiply %v1900, %v1900 : tensor<32x75264xf32>
    %v1902 = stablehlo.reduce(%v1901 init: %v1894) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1903 = stablehlo.broadcast_in_dim %v1902, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1904 = stablehlo.divide %v1903, %v1895 : tensor<32x75264xf32>
    %v1905 = stablehlo.add %v1904, %v1896 : tensor<32x75264xf32>
    %v1906 = stablehlo.rsqrt %v1905 : tensor<32x75264xf32>
    %v1907 = stablehlo.multiply %v1900, %v1906 : tensor<32x75264xf32>
    %v1908 = stablehlo.broadcast_in_dim %s2b5ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1909 = stablehlo.multiply %v1908, %v1893 : tensor<32x75264xf32>
    %v1910 = stablehlo.reduce(%v1909 init: %v1894) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1911 = stablehlo.broadcast_in_dim %v1910, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1912 = stablehlo.multiply %v1907, %v1909 : tensor<32x75264xf32>
    %v1913 = stablehlo.reduce(%v1912 init: %v1894) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1914 = stablehlo.broadcast_in_dim %v1913, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1915 = stablehlo.multiply %v1909, %v1895 : tensor<32x75264xf32>
    %v1916 = stablehlo.subtract %v1915, %v1911 : tensor<32x75264xf32>
    %v1917 = stablehlo.multiply %v1907, %v1914 : tensor<32x75264xf32>
    %v1918 = stablehlo.subtract %v1916, %v1917 : tensor<32x75264xf32>
    %v1919 = stablehlo.divide %v1906, %v1895 : tensor<32x75264xf32>
    %v1920 = stablehlo.multiply %v1919, %v1918 : tensor<32x75264xf32>
    %v1921 = stablehlo.reshape %v1920 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1922 = stablehlo.reverse %s2b5dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1923 = stablehlo.convolution(%v1921, %v1922)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1925 = stablehlo.add %v1924, %v1806 : tensor<32x75264xf32>
    %v1926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1927 = stablehlo.reshape %v657 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1928 = stablehlo.reshape %v1806 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1929 = stablehlo.multiply %v1927, %v1928 : tensor<32x384x14x14xf32>
    %v1930 = stablehlo.reduce(%v1929 init: %v1926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1931 = stablehlo.reshape %v652 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1932 = stablehlo.reshape %v1860 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1933 = stablehlo.transpose %v1931, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1934 = stablehlo.transpose %v1932, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1935 = stablehlo.convolution(%v1933, %v1934)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1936 = stablehlo.transpose %v1935, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1937 = stablehlo.reshape %v1860 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1939 = stablehlo.reduce(%v1937 init: %v1938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1940 = stablehlo.reshape %v634 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1941 = stablehlo.reshape %v1888 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1942 = stablehlo.transpose %v1940, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1943 = stablehlo.transpose %v1941, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1944 = stablehlo.convolution(%v1942, %v1943)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1945 = stablehlo.transpose %v1944, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1946 = stablehlo.reshape %v1888 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1948 = stablehlo.reduce(%v1946 init: %v1947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1950 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1951 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1952 = stablehlo.reduce(%v616 init: %v1949) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1953 = stablehlo.broadcast_in_dim %v1952, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1954 = stablehlo.divide %v1953, %v1950 : tensor<32x75264xf32>
    %v1955 = stablehlo.subtract %v616, %v1954 : tensor<32x75264xf32>
    %v1956 = stablehlo.multiply %v1955, %v1955 : tensor<32x75264xf32>
    %v1957 = stablehlo.reduce(%v1956 init: %v1949) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1958 = stablehlo.broadcast_in_dim %v1957, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1959 = stablehlo.divide %v1958, %v1950 : tensor<32x75264xf32>
    %v1960 = stablehlo.add %v1959, %v1951 : tensor<32x75264xf32>
    %v1961 = stablehlo.rsqrt %v1960 : tensor<32x75264xf32>
    %v1962 = stablehlo.multiply %v1955, %v1961 : tensor<32x75264xf32>
    %v1963 = stablehlo.multiply %v1893, %v1962 : tensor<32x75264xf32>
    %v1964 = stablehlo.reduce(%v1963 init: %v1949) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1966 = stablehlo.reduce(%v1893 init: %v1965) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1967 = stablehlo.reshape %v611 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1968 = stablehlo.reshape %v1920 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1969 = stablehlo.transpose %v1967, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1970 = stablehlo.transpose %v1968, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1971 = stablehlo.convolution(%v1969, %v1970)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v1972 = stablehlo.reshape %v1971 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v1973 = stablehlo.reshape %v1920 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1974 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1975 = stablehlo.reduce(%v1973 init: %v1974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1976 = stablehlo.reshape %v1925 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1977 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1978 = stablehlo.multiply %v1976, %v1977 : tensor<32x384x14x14xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1981 = stablehlo.transpose %s2b4pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1982 = stablehlo.reverse %v1981, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1983 = stablehlo.convolution(%v1980, %v1982)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1985 = stablehlo.multiply %v588, %v588 : tensor<32x301056xf32>
    %v1986 = stablehlo.multiply %v1985, %v588 : tensor<32x301056xf32>
    %v1987 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1988 = stablehlo.multiply %v1987, %v1986 : tensor<32x301056xf32>
    %v1989 = stablehlo.add %v588, %v1988 : tensor<32x301056xf32>
    %v1990 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1991 = stablehlo.multiply %v1990, %v1989 : tensor<32x301056xf32>
    %v1992 = stablehlo.tanh %v1991 : tensor<32x301056xf32>
    %v1993 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1994 = stablehlo.add %v1993, %v1992 : tensor<32x301056xf32>
    %v1995 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1996 = stablehlo.multiply %v1995, %v1994 : tensor<32x301056xf32>
    %v1997 = stablehlo.multiply %v1992, %v1992 : tensor<32x301056xf32>
    %v1998 = stablehlo.subtract %v1993, %v1997 : tensor<32x301056xf32>
    %v1999 = stablehlo.multiply %v1995, %v588 : tensor<32x301056xf32>
    %v2000 = stablehlo.multiply %v1999, %v1998 : tensor<32x301056xf32>
    %v2001 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2002 = stablehlo.multiply %v2001, %v1985 : tensor<32x301056xf32>
    %v2003 = stablehlo.add %v1993, %v2002 : tensor<32x301056xf32>
    %v2004 = stablehlo.multiply %v1990, %v2003 : tensor<32x301056xf32>
    %v2005 = stablehlo.multiply %v2000, %v2004 : tensor<32x301056xf32>
    %v2006 = stablehlo.add %v1996, %v2005 : tensor<32x301056xf32>
    %v2007 = stablehlo.multiply %v1984, %v2006 : tensor<32x301056xf32>
    %v2008 = stablehlo.reshape %v2007 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2009 = stablehlo.transpose %s2b4eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2010 = stablehlo.reverse %v2009, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2011 = stablehlo.convolution(%v2008, %v2010)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2012 = stablehlo.reshape %v2011 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2014 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2015 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2016 = stablehlo.reduce(%v565 init: %v2013) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2017 = stablehlo.broadcast_in_dim %v2016, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2018 = stablehlo.divide %v2017, %v2014 : tensor<32x75264xf32>
    %v2019 = stablehlo.subtract %v565, %v2018 : tensor<32x75264xf32>
    %v2020 = stablehlo.multiply %v2019, %v2019 : tensor<32x75264xf32>
    %v2021 = stablehlo.reduce(%v2020 init: %v2013) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2022 = stablehlo.broadcast_in_dim %v2021, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2023 = stablehlo.divide %v2022, %v2014 : tensor<32x75264xf32>
    %v2024 = stablehlo.add %v2023, %v2015 : tensor<32x75264xf32>
    %v2025 = stablehlo.rsqrt %v2024 : tensor<32x75264xf32>
    %v2026 = stablehlo.multiply %v2019, %v2025 : tensor<32x75264xf32>
    %v2027 = stablehlo.broadcast_in_dim %s2b4ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2028 = stablehlo.multiply %v2027, %v2012 : tensor<32x75264xf32>
    %v2029 = stablehlo.reduce(%v2028 init: %v2013) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2030 = stablehlo.broadcast_in_dim %v2029, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2031 = stablehlo.multiply %v2026, %v2028 : tensor<32x75264xf32>
    %v2032 = stablehlo.reduce(%v2031 init: %v2013) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2033 = stablehlo.broadcast_in_dim %v2032, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2034 = stablehlo.multiply %v2028, %v2014 : tensor<32x75264xf32>
    %v2035 = stablehlo.subtract %v2034, %v2030 : tensor<32x75264xf32>
    %v2036 = stablehlo.multiply %v2026, %v2033 : tensor<32x75264xf32>
    %v2037 = stablehlo.subtract %v2035, %v2036 : tensor<32x75264xf32>
    %v2038 = stablehlo.divide %v2025, %v2014 : tensor<32x75264xf32>
    %v2039 = stablehlo.multiply %v2038, %v2037 : tensor<32x75264xf32>
    %v2040 = stablehlo.reshape %v2039 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2041 = stablehlo.reverse %s2b4dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2042 = stablehlo.convolution(%v2040, %v2041)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2044 = stablehlo.add %v2043, %v1925 : tensor<32x75264xf32>
    %v2045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2046 = stablehlo.reshape %v606 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2047 = stablehlo.reshape %v1925 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2048 = stablehlo.multiply %v2046, %v2047 : tensor<32x384x14x14xf32>
    %v2049 = stablehlo.reduce(%v2048 init: %v2045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2050 = stablehlo.reshape %v601 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2051 = stablehlo.reshape %v1979 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2052 = stablehlo.transpose %v2050, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2053 = stablehlo.transpose %v2051, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2054 = stablehlo.convolution(%v2052, %v2053)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2055 = stablehlo.transpose %v2054, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2056 = stablehlo.reshape %v1979 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2057 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2058 = stablehlo.reduce(%v2056 init: %v2057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2059 = stablehlo.reshape %v583 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2060 = stablehlo.reshape %v2007 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2061 = stablehlo.transpose %v2059, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2062 = stablehlo.transpose %v2060, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2063 = stablehlo.convolution(%v2061, %v2062)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2064 = stablehlo.transpose %v2063, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2065 = stablehlo.reshape %v2007 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2067 = stablehlo.reduce(%v2065 init: %v2066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2069 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2070 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2071 = stablehlo.reduce(%v565 init: %v2068) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2072 = stablehlo.broadcast_in_dim %v2071, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2073 = stablehlo.divide %v2072, %v2069 : tensor<32x75264xf32>
    %v2074 = stablehlo.subtract %v565, %v2073 : tensor<32x75264xf32>
    %v2075 = stablehlo.multiply %v2074, %v2074 : tensor<32x75264xf32>
    %v2076 = stablehlo.reduce(%v2075 init: %v2068) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2077 = stablehlo.broadcast_in_dim %v2076, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2078 = stablehlo.divide %v2077, %v2069 : tensor<32x75264xf32>
    %v2079 = stablehlo.add %v2078, %v2070 : tensor<32x75264xf32>
    %v2080 = stablehlo.rsqrt %v2079 : tensor<32x75264xf32>
    %v2081 = stablehlo.multiply %v2074, %v2080 : tensor<32x75264xf32>
    %v2082 = stablehlo.multiply %v2012, %v2081 : tensor<32x75264xf32>
    %v2083 = stablehlo.reduce(%v2082 init: %v2068) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2084 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2085 = stablehlo.reduce(%v2012 init: %v2084) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2086 = stablehlo.reshape %v560 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2087 = stablehlo.reshape %v2039 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2088 = stablehlo.transpose %v2086, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2089 = stablehlo.transpose %v2087, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2090 = stablehlo.convolution(%v2088, %v2089)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2091 = stablehlo.reshape %v2090 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2092 = stablehlo.reshape %v2039 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2094 = stablehlo.reduce(%v2092 init: %v2093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2095 = stablehlo.reshape %v2044 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2096 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2097 = stablehlo.multiply %v2095, %v2096 : tensor<32x384x14x14xf32>
    %v2098 = stablehlo.reshape %v2097 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2099 = stablehlo.reshape %v2098 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2100 = stablehlo.transpose %s2b3pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2101 = stablehlo.reverse %v2100, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2102 = stablehlo.convolution(%v2099, %v2101)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2103 = stablehlo.reshape %v2102 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2104 = stablehlo.multiply %v537, %v537 : tensor<32x301056xf32>
    %v2105 = stablehlo.multiply %v2104, %v537 : tensor<32x301056xf32>
    %v2106 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2107 = stablehlo.multiply %v2106, %v2105 : tensor<32x301056xf32>
    %v2108 = stablehlo.add %v537, %v2107 : tensor<32x301056xf32>
    %v2109 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2110 = stablehlo.multiply %v2109, %v2108 : tensor<32x301056xf32>
    %v2111 = stablehlo.tanh %v2110 : tensor<32x301056xf32>
    %v2112 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2113 = stablehlo.add %v2112, %v2111 : tensor<32x301056xf32>
    %v2114 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2115 = stablehlo.multiply %v2114, %v2113 : tensor<32x301056xf32>
    %v2116 = stablehlo.multiply %v2111, %v2111 : tensor<32x301056xf32>
    %v2117 = stablehlo.subtract %v2112, %v2116 : tensor<32x301056xf32>
    %v2118 = stablehlo.multiply %v2114, %v537 : tensor<32x301056xf32>
    %v2119 = stablehlo.multiply %v2118, %v2117 : tensor<32x301056xf32>
    %v2120 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2121 = stablehlo.multiply %v2120, %v2104 : tensor<32x301056xf32>
    %v2122 = stablehlo.add %v2112, %v2121 : tensor<32x301056xf32>
    %v2123 = stablehlo.multiply %v2109, %v2122 : tensor<32x301056xf32>
    %v2124 = stablehlo.multiply %v2119, %v2123 : tensor<32x301056xf32>
    %v2125 = stablehlo.add %v2115, %v2124 : tensor<32x301056xf32>
    %v2126 = stablehlo.multiply %v2103, %v2125 : tensor<32x301056xf32>
    %v2127 = stablehlo.reshape %v2126 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2128 = stablehlo.transpose %s2b3eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2129 = stablehlo.reverse %v2128, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2130 = stablehlo.convolution(%v2127, %v2129)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2131 = stablehlo.reshape %v2130 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2132 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2133 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2134 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2135 = stablehlo.reduce(%v514 init: %v2132) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2136 = stablehlo.broadcast_in_dim %v2135, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2137 = stablehlo.divide %v2136, %v2133 : tensor<32x75264xf32>
    %v2138 = stablehlo.subtract %v514, %v2137 : tensor<32x75264xf32>
    %v2139 = stablehlo.multiply %v2138, %v2138 : tensor<32x75264xf32>
    %v2140 = stablehlo.reduce(%v2139 init: %v2132) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2141 = stablehlo.broadcast_in_dim %v2140, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2142 = stablehlo.divide %v2141, %v2133 : tensor<32x75264xf32>
    %v2143 = stablehlo.add %v2142, %v2134 : tensor<32x75264xf32>
    %v2144 = stablehlo.rsqrt %v2143 : tensor<32x75264xf32>
    %v2145 = stablehlo.multiply %v2138, %v2144 : tensor<32x75264xf32>
    %v2146 = stablehlo.broadcast_in_dim %s2b3ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2147 = stablehlo.multiply %v2146, %v2131 : tensor<32x75264xf32>
    %v2148 = stablehlo.reduce(%v2147 init: %v2132) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2149 = stablehlo.broadcast_in_dim %v2148, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2150 = stablehlo.multiply %v2145, %v2147 : tensor<32x75264xf32>
    %v2151 = stablehlo.reduce(%v2150 init: %v2132) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2152 = stablehlo.broadcast_in_dim %v2151, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2153 = stablehlo.multiply %v2147, %v2133 : tensor<32x75264xf32>
    %v2154 = stablehlo.subtract %v2153, %v2149 : tensor<32x75264xf32>
    %v2155 = stablehlo.multiply %v2145, %v2152 : tensor<32x75264xf32>
    %v2156 = stablehlo.subtract %v2154, %v2155 : tensor<32x75264xf32>
    %v2157 = stablehlo.divide %v2144, %v2133 : tensor<32x75264xf32>
    %v2158 = stablehlo.multiply %v2157, %v2156 : tensor<32x75264xf32>
    %v2159 = stablehlo.reshape %v2158 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2160 = stablehlo.reverse %s2b3dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2161 = stablehlo.convolution(%v2159, %v2160)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2162 = stablehlo.reshape %v2161 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2163 = stablehlo.add %v2162, %v2044 : tensor<32x75264xf32>
    %v2164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2165 = stablehlo.reshape %v555 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2166 = stablehlo.reshape %v2044 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2167 = stablehlo.multiply %v2165, %v2166 : tensor<32x384x14x14xf32>
    %v2168 = stablehlo.reduce(%v2167 init: %v2164) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2169 = stablehlo.reshape %v550 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2170 = stablehlo.reshape %v2098 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2171 = stablehlo.transpose %v2169, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2172 = stablehlo.transpose %v2170, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2173 = stablehlo.convolution(%v2171, %v2172)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2174 = stablehlo.transpose %v2173, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2175 = stablehlo.reshape %v2098 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2177 = stablehlo.reduce(%v2175 init: %v2176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2178 = stablehlo.reshape %v532 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2179 = stablehlo.reshape %v2126 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2180 = stablehlo.transpose %v2178, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2181 = stablehlo.transpose %v2179, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2182 = stablehlo.convolution(%v2180, %v2181)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2183 = stablehlo.transpose %v2182, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2184 = stablehlo.reshape %v2126 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2186 = stablehlo.reduce(%v2184 init: %v2185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2187 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2188 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2189 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2190 = stablehlo.reduce(%v514 init: %v2187) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2191 = stablehlo.broadcast_in_dim %v2190, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2192 = stablehlo.divide %v2191, %v2188 : tensor<32x75264xf32>
    %v2193 = stablehlo.subtract %v514, %v2192 : tensor<32x75264xf32>
    %v2194 = stablehlo.multiply %v2193, %v2193 : tensor<32x75264xf32>
    %v2195 = stablehlo.reduce(%v2194 init: %v2187) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2196 = stablehlo.broadcast_in_dim %v2195, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2197 = stablehlo.divide %v2196, %v2188 : tensor<32x75264xf32>
    %v2198 = stablehlo.add %v2197, %v2189 : tensor<32x75264xf32>
    %v2199 = stablehlo.rsqrt %v2198 : tensor<32x75264xf32>
    %v2200 = stablehlo.multiply %v2193, %v2199 : tensor<32x75264xf32>
    %v2201 = stablehlo.multiply %v2131, %v2200 : tensor<32x75264xf32>
    %v2202 = stablehlo.reduce(%v2201 init: %v2187) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2204 = stablehlo.reduce(%v2131 init: %v2203) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2205 = stablehlo.reshape %v509 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2206 = stablehlo.reshape %v2158 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2207 = stablehlo.transpose %v2205, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2208 = stablehlo.transpose %v2206, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2209 = stablehlo.convolution(%v2207, %v2208)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2210 = stablehlo.reshape %v2209 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2211 = stablehlo.reshape %v2158 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2213 = stablehlo.reduce(%v2211 init: %v2212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2214 = stablehlo.reshape %v2163 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2215 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2216 = stablehlo.multiply %v2214, %v2215 : tensor<32x384x14x14xf32>
    %v2217 = stablehlo.reshape %v2216 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2218 = stablehlo.reshape %v2217 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2219 = stablehlo.transpose %s2b2pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2220 = stablehlo.reverse %v2219, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2221 = stablehlo.convolution(%v2218, %v2220)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2222 = stablehlo.reshape %v2221 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2223 = stablehlo.multiply %v486, %v486 : tensor<32x301056xf32>
    %v2224 = stablehlo.multiply %v2223, %v486 : tensor<32x301056xf32>
    %v2225 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2226 = stablehlo.multiply %v2225, %v2224 : tensor<32x301056xf32>
    %v2227 = stablehlo.add %v486, %v2226 : tensor<32x301056xf32>
    %v2228 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2229 = stablehlo.multiply %v2228, %v2227 : tensor<32x301056xf32>
    %v2230 = stablehlo.tanh %v2229 : tensor<32x301056xf32>
    %v2231 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2232 = stablehlo.add %v2231, %v2230 : tensor<32x301056xf32>
    %v2233 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2234 = stablehlo.multiply %v2233, %v2232 : tensor<32x301056xf32>
    %v2235 = stablehlo.multiply %v2230, %v2230 : tensor<32x301056xf32>
    %v2236 = stablehlo.subtract %v2231, %v2235 : tensor<32x301056xf32>
    %v2237 = stablehlo.multiply %v2233, %v486 : tensor<32x301056xf32>
    %v2238 = stablehlo.multiply %v2237, %v2236 : tensor<32x301056xf32>
    %v2239 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2240 = stablehlo.multiply %v2239, %v2223 : tensor<32x301056xf32>
    %v2241 = stablehlo.add %v2231, %v2240 : tensor<32x301056xf32>
    %v2242 = stablehlo.multiply %v2228, %v2241 : tensor<32x301056xf32>
    %v2243 = stablehlo.multiply %v2238, %v2242 : tensor<32x301056xf32>
    %v2244 = stablehlo.add %v2234, %v2243 : tensor<32x301056xf32>
    %v2245 = stablehlo.multiply %v2222, %v2244 : tensor<32x301056xf32>
    %v2246 = stablehlo.reshape %v2245 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2247 = stablehlo.transpose %s2b2eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2248 = stablehlo.reverse %v2247, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2249 = stablehlo.convolution(%v2246, %v2248)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2250 = stablehlo.reshape %v2249 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2252 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2253 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2254 = stablehlo.reduce(%v463 init: %v2251) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2255 = stablehlo.broadcast_in_dim %v2254, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2256 = stablehlo.divide %v2255, %v2252 : tensor<32x75264xf32>
    %v2257 = stablehlo.subtract %v463, %v2256 : tensor<32x75264xf32>
    %v2258 = stablehlo.multiply %v2257, %v2257 : tensor<32x75264xf32>
    %v2259 = stablehlo.reduce(%v2258 init: %v2251) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2260 = stablehlo.broadcast_in_dim %v2259, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2261 = stablehlo.divide %v2260, %v2252 : tensor<32x75264xf32>
    %v2262 = stablehlo.add %v2261, %v2253 : tensor<32x75264xf32>
    %v2263 = stablehlo.rsqrt %v2262 : tensor<32x75264xf32>
    %v2264 = stablehlo.multiply %v2257, %v2263 : tensor<32x75264xf32>
    %v2265 = stablehlo.broadcast_in_dim %s2b2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2266 = stablehlo.multiply %v2265, %v2250 : tensor<32x75264xf32>
    %v2267 = stablehlo.reduce(%v2266 init: %v2251) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2268 = stablehlo.broadcast_in_dim %v2267, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2269 = stablehlo.multiply %v2264, %v2266 : tensor<32x75264xf32>
    %v2270 = stablehlo.reduce(%v2269 init: %v2251) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2271 = stablehlo.broadcast_in_dim %v2270, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2272 = stablehlo.multiply %v2266, %v2252 : tensor<32x75264xf32>
    %v2273 = stablehlo.subtract %v2272, %v2268 : tensor<32x75264xf32>
    %v2274 = stablehlo.multiply %v2264, %v2271 : tensor<32x75264xf32>
    %v2275 = stablehlo.subtract %v2273, %v2274 : tensor<32x75264xf32>
    %v2276 = stablehlo.divide %v2263, %v2252 : tensor<32x75264xf32>
    %v2277 = stablehlo.multiply %v2276, %v2275 : tensor<32x75264xf32>
    %v2278 = stablehlo.reshape %v2277 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2279 = stablehlo.reverse %s2b2dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2280 = stablehlo.convolution(%v2278, %v2279)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2281 = stablehlo.reshape %v2280 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2282 = stablehlo.add %v2281, %v2163 : tensor<32x75264xf32>
    %v2283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2284 = stablehlo.reshape %v504 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2285 = stablehlo.reshape %v2163 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2286 = stablehlo.multiply %v2284, %v2285 : tensor<32x384x14x14xf32>
    %v2287 = stablehlo.reduce(%v2286 init: %v2283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2288 = stablehlo.reshape %v499 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2289 = stablehlo.reshape %v2217 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2290 = stablehlo.transpose %v2288, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2291 = stablehlo.transpose %v2289, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2292 = stablehlo.convolution(%v2290, %v2291)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2293 = stablehlo.transpose %v2292, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2294 = stablehlo.reshape %v2217 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2295 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2296 = stablehlo.reduce(%v2294 init: %v2295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2297 = stablehlo.reshape %v481 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2298 = stablehlo.reshape %v2245 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2299 = stablehlo.transpose %v2297, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2300 = stablehlo.transpose %v2298, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2301 = stablehlo.convolution(%v2299, %v2300)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2302 = stablehlo.transpose %v2301, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2303 = stablehlo.reshape %v2245 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2305 = stablehlo.reduce(%v2303 init: %v2304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2307 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2308 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2309 = stablehlo.reduce(%v463 init: %v2306) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2310 = stablehlo.broadcast_in_dim %v2309, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2311 = stablehlo.divide %v2310, %v2307 : tensor<32x75264xf32>
    %v2312 = stablehlo.subtract %v463, %v2311 : tensor<32x75264xf32>
    %v2313 = stablehlo.multiply %v2312, %v2312 : tensor<32x75264xf32>
    %v2314 = stablehlo.reduce(%v2313 init: %v2306) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2315 = stablehlo.broadcast_in_dim %v2314, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2316 = stablehlo.divide %v2315, %v2307 : tensor<32x75264xf32>
    %v2317 = stablehlo.add %v2316, %v2308 : tensor<32x75264xf32>
    %v2318 = stablehlo.rsqrt %v2317 : tensor<32x75264xf32>
    %v2319 = stablehlo.multiply %v2312, %v2318 : tensor<32x75264xf32>
    %v2320 = stablehlo.multiply %v2250, %v2319 : tensor<32x75264xf32>
    %v2321 = stablehlo.reduce(%v2320 init: %v2306) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2322 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2323 = stablehlo.reduce(%v2250 init: %v2322) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2324 = stablehlo.reshape %v458 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2325 = stablehlo.reshape %v2277 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2326 = stablehlo.transpose %v2324, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2327 = stablehlo.transpose %v2325, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2328 = stablehlo.convolution(%v2326, %v2327)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2329 = stablehlo.reshape %v2328 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2330 = stablehlo.reshape %v2277 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2332 = stablehlo.reduce(%v2330 init: %v2331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2333 = stablehlo.reshape %v2282 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2334 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2335 = stablehlo.multiply %v2333, %v2334 : tensor<32x384x14x14xf32>
    %v2336 = stablehlo.reshape %v2335 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2337 = stablehlo.reshape %v2336 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2338 = stablehlo.transpose %s2b1pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2339 = stablehlo.reverse %v2338, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2340 = stablehlo.convolution(%v2337, %v2339)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2342 = stablehlo.multiply %v435, %v435 : tensor<32x301056xf32>
    %v2343 = stablehlo.multiply %v2342, %v435 : tensor<32x301056xf32>
    %v2344 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2345 = stablehlo.multiply %v2344, %v2343 : tensor<32x301056xf32>
    %v2346 = stablehlo.add %v435, %v2345 : tensor<32x301056xf32>
    %v2347 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2348 = stablehlo.multiply %v2347, %v2346 : tensor<32x301056xf32>
    %v2349 = stablehlo.tanh %v2348 : tensor<32x301056xf32>
    %v2350 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2351 = stablehlo.add %v2350, %v2349 : tensor<32x301056xf32>
    %v2352 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2353 = stablehlo.multiply %v2352, %v2351 : tensor<32x301056xf32>
    %v2354 = stablehlo.multiply %v2349, %v2349 : tensor<32x301056xf32>
    %v2355 = stablehlo.subtract %v2350, %v2354 : tensor<32x301056xf32>
    %v2356 = stablehlo.multiply %v2352, %v435 : tensor<32x301056xf32>
    %v2357 = stablehlo.multiply %v2356, %v2355 : tensor<32x301056xf32>
    %v2358 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2359 = stablehlo.multiply %v2358, %v2342 : tensor<32x301056xf32>
    %v2360 = stablehlo.add %v2350, %v2359 : tensor<32x301056xf32>
    %v2361 = stablehlo.multiply %v2347, %v2360 : tensor<32x301056xf32>
    %v2362 = stablehlo.multiply %v2357, %v2361 : tensor<32x301056xf32>
    %v2363 = stablehlo.add %v2353, %v2362 : tensor<32x301056xf32>
    %v2364 = stablehlo.multiply %v2341, %v2363 : tensor<32x301056xf32>
    %v2365 = stablehlo.reshape %v2364 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2366 = stablehlo.transpose %s2b1eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2367 = stablehlo.reverse %v2366, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2368 = stablehlo.convolution(%v2365, %v2367)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2369 = stablehlo.reshape %v2368 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2370 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2371 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2372 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2373 = stablehlo.reduce(%v412 init: %v2370) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2374 = stablehlo.broadcast_in_dim %v2373, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2375 = stablehlo.divide %v2374, %v2371 : tensor<32x75264xf32>
    %v2376 = stablehlo.subtract %v412, %v2375 : tensor<32x75264xf32>
    %v2377 = stablehlo.multiply %v2376, %v2376 : tensor<32x75264xf32>
    %v2378 = stablehlo.reduce(%v2377 init: %v2370) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2379 = stablehlo.broadcast_in_dim %v2378, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2380 = stablehlo.divide %v2379, %v2371 : tensor<32x75264xf32>
    %v2381 = stablehlo.add %v2380, %v2372 : tensor<32x75264xf32>
    %v2382 = stablehlo.rsqrt %v2381 : tensor<32x75264xf32>
    %v2383 = stablehlo.multiply %v2376, %v2382 : tensor<32x75264xf32>
    %v2384 = stablehlo.broadcast_in_dim %s2b1ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2385 = stablehlo.multiply %v2384, %v2369 : tensor<32x75264xf32>
    %v2386 = stablehlo.reduce(%v2385 init: %v2370) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2387 = stablehlo.broadcast_in_dim %v2386, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2388 = stablehlo.multiply %v2383, %v2385 : tensor<32x75264xf32>
    %v2389 = stablehlo.reduce(%v2388 init: %v2370) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2390 = stablehlo.broadcast_in_dim %v2389, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2391 = stablehlo.multiply %v2385, %v2371 : tensor<32x75264xf32>
    %v2392 = stablehlo.subtract %v2391, %v2387 : tensor<32x75264xf32>
    %v2393 = stablehlo.multiply %v2383, %v2390 : tensor<32x75264xf32>
    %v2394 = stablehlo.subtract %v2392, %v2393 : tensor<32x75264xf32>
    %v2395 = stablehlo.divide %v2382, %v2371 : tensor<32x75264xf32>
    %v2396 = stablehlo.multiply %v2395, %v2394 : tensor<32x75264xf32>
    %v2397 = stablehlo.reshape %v2396 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2398 = stablehlo.reverse %s2b1dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2399 = stablehlo.convolution(%v2397, %v2398)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2400 = stablehlo.reshape %v2399 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2401 = stablehlo.add %v2400, %v2282 : tensor<32x75264xf32>
    %v2402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2403 = stablehlo.reshape %v453 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2404 = stablehlo.reshape %v2282 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2405 = stablehlo.multiply %v2403, %v2404 : tensor<32x384x14x14xf32>
    %v2406 = stablehlo.reduce(%v2405 init: %v2402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2407 = stablehlo.reshape %v448 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2408 = stablehlo.reshape %v2336 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2409 = stablehlo.transpose %v2407, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2410 = stablehlo.transpose %v2408, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2411 = stablehlo.convolution(%v2409, %v2410)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2412 = stablehlo.transpose %v2411, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2413 = stablehlo.reshape %v2336 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2415 = stablehlo.reduce(%v2413 init: %v2414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2416 = stablehlo.reshape %v430 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2417 = stablehlo.reshape %v2364 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2418 = stablehlo.transpose %v2416, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2419 = stablehlo.transpose %v2417, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2420 = stablehlo.convolution(%v2418, %v2419)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2421 = stablehlo.transpose %v2420, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2422 = stablehlo.reshape %v2364 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2424 = stablehlo.reduce(%v2422 init: %v2423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2426 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2427 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2428 = stablehlo.reduce(%v412 init: %v2425) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2429 = stablehlo.broadcast_in_dim %v2428, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2430 = stablehlo.divide %v2429, %v2426 : tensor<32x75264xf32>
    %v2431 = stablehlo.subtract %v412, %v2430 : tensor<32x75264xf32>
    %v2432 = stablehlo.multiply %v2431, %v2431 : tensor<32x75264xf32>
    %v2433 = stablehlo.reduce(%v2432 init: %v2425) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2434 = stablehlo.broadcast_in_dim %v2433, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2435 = stablehlo.divide %v2434, %v2426 : tensor<32x75264xf32>
    %v2436 = stablehlo.add %v2435, %v2427 : tensor<32x75264xf32>
    %v2437 = stablehlo.rsqrt %v2436 : tensor<32x75264xf32>
    %v2438 = stablehlo.multiply %v2431, %v2437 : tensor<32x75264xf32>
    %v2439 = stablehlo.multiply %v2369, %v2438 : tensor<32x75264xf32>
    %v2440 = stablehlo.reduce(%v2439 init: %v2425) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2441 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2442 = stablehlo.reduce(%v2369 init: %v2441) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2443 = stablehlo.reshape %v407 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2444 = stablehlo.reshape %v2396 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2445 = stablehlo.transpose %v2443, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2446 = stablehlo.transpose %v2444, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2447 = stablehlo.convolution(%v2445, %v2446)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2448 = stablehlo.reshape %v2447 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2449 = stablehlo.reshape %v2396 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2451 = stablehlo.reduce(%v2449 init: %v2450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2452 = stablehlo.reshape %v2401 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2453 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2454 = stablehlo.multiply %v2452, %v2453 : tensor<32x384x14x14xf32>
    %v2455 = stablehlo.reshape %v2454 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2456 = stablehlo.reshape %v2455 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2457 = stablehlo.transpose %s2b0pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2458 = stablehlo.reverse %v2457, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2459 = stablehlo.convolution(%v2456, %v2458)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2460 = stablehlo.reshape %v2459 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2461 = stablehlo.multiply %v384, %v384 : tensor<32x301056xf32>
    %v2462 = stablehlo.multiply %v2461, %v384 : tensor<32x301056xf32>
    %v2463 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2464 = stablehlo.multiply %v2463, %v2462 : tensor<32x301056xf32>
    %v2465 = stablehlo.add %v384, %v2464 : tensor<32x301056xf32>
    %v2466 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2467 = stablehlo.multiply %v2466, %v2465 : tensor<32x301056xf32>
    %v2468 = stablehlo.tanh %v2467 : tensor<32x301056xf32>
    %v2469 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2470 = stablehlo.add %v2469, %v2468 : tensor<32x301056xf32>
    %v2471 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2472 = stablehlo.multiply %v2471, %v2470 : tensor<32x301056xf32>
    %v2473 = stablehlo.multiply %v2468, %v2468 : tensor<32x301056xf32>
    %v2474 = stablehlo.subtract %v2469, %v2473 : tensor<32x301056xf32>
    %v2475 = stablehlo.multiply %v2471, %v384 : tensor<32x301056xf32>
    %v2476 = stablehlo.multiply %v2475, %v2474 : tensor<32x301056xf32>
    %v2477 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2478 = stablehlo.multiply %v2477, %v2461 : tensor<32x301056xf32>
    %v2479 = stablehlo.add %v2469, %v2478 : tensor<32x301056xf32>
    %v2480 = stablehlo.multiply %v2466, %v2479 : tensor<32x301056xf32>
    %v2481 = stablehlo.multiply %v2476, %v2480 : tensor<32x301056xf32>
    %v2482 = stablehlo.add %v2472, %v2481 : tensor<32x301056xf32>
    %v2483 = stablehlo.multiply %v2460, %v2482 : tensor<32x301056xf32>
    %v2484 = stablehlo.reshape %v2483 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2485 = stablehlo.transpose %s2b0eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2486 = stablehlo.reverse %v2485, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2487 = stablehlo.convolution(%v2484, %v2486)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2488 = stablehlo.reshape %v2487 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2490 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2491 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2492 = stablehlo.reduce(%v361 init: %v2489) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2493 = stablehlo.broadcast_in_dim %v2492, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2494 = stablehlo.divide %v2493, %v2490 : tensor<32x75264xf32>
    %v2495 = stablehlo.subtract %v361, %v2494 : tensor<32x75264xf32>
    %v2496 = stablehlo.multiply %v2495, %v2495 : tensor<32x75264xf32>
    %v2497 = stablehlo.reduce(%v2496 init: %v2489) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2498 = stablehlo.broadcast_in_dim %v2497, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2499 = stablehlo.divide %v2498, %v2490 : tensor<32x75264xf32>
    %v2500 = stablehlo.add %v2499, %v2491 : tensor<32x75264xf32>
    %v2501 = stablehlo.rsqrt %v2500 : tensor<32x75264xf32>
    %v2502 = stablehlo.multiply %v2495, %v2501 : tensor<32x75264xf32>
    %v2503 = stablehlo.broadcast_in_dim %s2b0ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2504 = stablehlo.multiply %v2503, %v2488 : tensor<32x75264xf32>
    %v2505 = stablehlo.reduce(%v2504 init: %v2489) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2506 = stablehlo.broadcast_in_dim %v2505, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2507 = stablehlo.multiply %v2502, %v2504 : tensor<32x75264xf32>
    %v2508 = stablehlo.reduce(%v2507 init: %v2489) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2509 = stablehlo.broadcast_in_dim %v2508, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2510 = stablehlo.multiply %v2504, %v2490 : tensor<32x75264xf32>
    %v2511 = stablehlo.subtract %v2510, %v2506 : tensor<32x75264xf32>
    %v2512 = stablehlo.multiply %v2502, %v2509 : tensor<32x75264xf32>
    %v2513 = stablehlo.subtract %v2511, %v2512 : tensor<32x75264xf32>
    %v2514 = stablehlo.divide %v2501, %v2490 : tensor<32x75264xf32>
    %v2515 = stablehlo.multiply %v2514, %v2513 : tensor<32x75264xf32>
    %v2516 = stablehlo.reshape %v2515 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2517 = stablehlo.reverse %s2b0dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2518 = stablehlo.convolution(%v2516, %v2517)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2519 = stablehlo.reshape %v2518 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2520 = stablehlo.add %v2519, %v2401 : tensor<32x75264xf32>
    %v2521 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2522 = stablehlo.reshape %v402 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2523 = stablehlo.reshape %v2401 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2524 = stablehlo.multiply %v2522, %v2523 : tensor<32x384x14x14xf32>
    %v2525 = stablehlo.reduce(%v2524 init: %v2521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2526 = stablehlo.reshape %v397 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2527 = stablehlo.reshape %v2455 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2528 = stablehlo.transpose %v2526, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2529 = stablehlo.transpose %v2527, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2530 = stablehlo.convolution(%v2528, %v2529)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2531 = stablehlo.transpose %v2530, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2532 = stablehlo.reshape %v2455 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2534 = stablehlo.reduce(%v2532 init: %v2533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2535 = stablehlo.reshape %v379 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2536 = stablehlo.reshape %v2483 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2537 = stablehlo.transpose %v2535, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2538 = stablehlo.transpose %v2536, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2539 = stablehlo.convolution(%v2537, %v2538)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2540 = stablehlo.transpose %v2539, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2541 = stablehlo.reshape %v2483 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2543 = stablehlo.reduce(%v2541 init: %v2542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2544 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2545 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2546 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2547 = stablehlo.reduce(%v361 init: %v2544) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2548 = stablehlo.broadcast_in_dim %v2547, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2549 = stablehlo.divide %v2548, %v2545 : tensor<32x75264xf32>
    %v2550 = stablehlo.subtract %v361, %v2549 : tensor<32x75264xf32>
    %v2551 = stablehlo.multiply %v2550, %v2550 : tensor<32x75264xf32>
    %v2552 = stablehlo.reduce(%v2551 init: %v2544) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2553 = stablehlo.broadcast_in_dim %v2552, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2554 = stablehlo.divide %v2553, %v2545 : tensor<32x75264xf32>
    %v2555 = stablehlo.add %v2554, %v2546 : tensor<32x75264xf32>
    %v2556 = stablehlo.rsqrt %v2555 : tensor<32x75264xf32>
    %v2557 = stablehlo.multiply %v2550, %v2556 : tensor<32x75264xf32>
    %v2558 = stablehlo.multiply %v2488, %v2557 : tensor<32x75264xf32>
    %v2559 = stablehlo.reduce(%v2558 init: %v2544) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2560 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2561 = stablehlo.reduce(%v2488 init: %v2560) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2562 = stablehlo.reshape %v356 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2563 = stablehlo.reshape %v2515 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2564 = stablehlo.transpose %v2562, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2565 = stablehlo.transpose %v2563, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2566 = stablehlo.convolution(%v2564, %v2565)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2567 = stablehlo.reshape %v2566 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2568 = stablehlo.reshape %v2515 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2570 = stablehlo.reduce(%v2568 init: %v2569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2571 = stablehlo.reshape %v2520 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2572 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2573 = stablehlo.pad %v2571, %v2572, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x28x28xf32>
    %v2574 = stablehlo.transpose %d1W, dims = [1, 0, 2, 3] : (tensor<384x192x2x2xf32>) -> tensor<192x384x2x2xf32>
    %v2575 = stablehlo.reverse %v2574, dims = [2, 3] : tensor<192x384x2x2xf32>
    %v2576 = stablehlo.convolution(%v2573, %v2575)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x28x28xf32>, tensor<192x384x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v2577 = stablehlo.reshape %v2576 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2578 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2579 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2580 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2581 = stablehlo.reduce(%v333 init: %v2578) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2582 = stablehlo.broadcast_in_dim %v2581, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2583 = stablehlo.divide %v2582, %v2579 : tensor<32x150528xf32>
    %v2584 = stablehlo.subtract %v333, %v2583 : tensor<32x150528xf32>
    %v2585 = stablehlo.multiply %v2584, %v2584 : tensor<32x150528xf32>
    %v2586 = stablehlo.reduce(%v2585 init: %v2578) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2587 = stablehlo.broadcast_in_dim %v2586, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2588 = stablehlo.divide %v2587, %v2579 : tensor<32x150528xf32>
    %v2589 = stablehlo.add %v2588, %v2580 : tensor<32x150528xf32>
    %v2590 = stablehlo.rsqrt %v2589 : tensor<32x150528xf32>
    %v2591 = stablehlo.multiply %v2584, %v2590 : tensor<32x150528xf32>
    %v2592 = stablehlo.broadcast_in_dim %d1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v2593 = stablehlo.multiply %v2592, %v2577 : tensor<32x150528xf32>
    %v2594 = stablehlo.reduce(%v2593 init: %v2578) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2595 = stablehlo.broadcast_in_dim %v2594, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2596 = stablehlo.multiply %v2591, %v2593 : tensor<32x150528xf32>
    %v2597 = stablehlo.reduce(%v2596 init: %v2578) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2598 = stablehlo.broadcast_in_dim %v2597, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2599 = stablehlo.multiply %v2593, %v2579 : tensor<32x150528xf32>
    %v2600 = stablehlo.subtract %v2599, %v2595 : tensor<32x150528xf32>
    %v2601 = stablehlo.multiply %v2591, %v2598 : tensor<32x150528xf32>
    %v2602 = stablehlo.subtract %v2600, %v2601 : tensor<32x150528xf32>
    %v2603 = stablehlo.divide %v2590, %v2579 : tensor<32x150528xf32>
    %v2604 = stablehlo.multiply %v2603, %v2602 : tensor<32x150528xf32>
    %v2605 = stablehlo.reshape %v2520 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2607 = stablehlo.reduce(%v2605 init: %v2606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2609 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2610 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2611 = stablehlo.reduce(%v333 init: %v2608) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2612 = stablehlo.broadcast_in_dim %v2611, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2613 = stablehlo.divide %v2612, %v2609 : tensor<32x150528xf32>
    %v2614 = stablehlo.subtract %v333, %v2613 : tensor<32x150528xf32>
    %v2615 = stablehlo.multiply %v2614, %v2614 : tensor<32x150528xf32>
    %v2616 = stablehlo.reduce(%v2615 init: %v2608) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2617 = stablehlo.broadcast_in_dim %v2616, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2618 = stablehlo.divide %v2617, %v2609 : tensor<32x150528xf32>
    %v2619 = stablehlo.add %v2618, %v2610 : tensor<32x150528xf32>
    %v2620 = stablehlo.rsqrt %v2619 : tensor<32x150528xf32>
    %v2621 = stablehlo.multiply %v2614, %v2620 : tensor<32x150528xf32>
    %v2622 = stablehlo.multiply %v2577, %v2621 : tensor<32x150528xf32>
    %v2623 = stablehlo.reduce(%v2622 init: %v2608) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2625 = stablehlo.reduce(%v2577 init: %v2624) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2626 = stablehlo.reshape %v351 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2627 = stablehlo.reshape %v2520 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2628 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2629 = stablehlo.pad %v2627, %v2628, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x27x27xf32>
    %v2630 = stablehlo.transpose %v2626, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2631 = stablehlo.transpose %v2629, dims = [1, 0, 2, 3] : (tensor<32x384x27x27xf32>) -> tensor<384x32x27x27xf32>
    %v2632 = stablehlo.convolution(%v2630, %v2631)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<384x32x27x27xf32>) -> tensor<192x384x2x2xf32>
    %v2633 = stablehlo.transpose %v2632, dims = [1, 0, 2, 3] : (tensor<192x384x2x2xf32>) -> tensor<384x192x2x2xf32>
    %v2634 = stablehlo.reshape %v2604 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2635 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v2636 = stablehlo.multiply %v2634, %v2635 : tensor<32x192x28x28xf32>
    %v2637 = stablehlo.reshape %v2636 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2638 = stablehlo.reshape %v2637 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2639 = stablehlo.transpose %s1b2pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2640 = stablehlo.reverse %v2639, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v2641 = stablehlo.convolution(%v2638, %v2640)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v2642 = stablehlo.reshape %v2641 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v2643 = stablehlo.multiply %v310, %v310 : tensor<32x602112xf32>
    %v2644 = stablehlo.multiply %v2643, %v310 : tensor<32x602112xf32>
    %v2645 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v2646 = stablehlo.multiply %v2645, %v2644 : tensor<32x602112xf32>
    %v2647 = stablehlo.add %v310, %v2646 : tensor<32x602112xf32>
    %v2648 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v2649 = stablehlo.multiply %v2648, %v2647 : tensor<32x602112xf32>
    %v2650 = stablehlo.tanh %v2649 : tensor<32x602112xf32>
    %v2651 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v2652 = stablehlo.add %v2651, %v2650 : tensor<32x602112xf32>
    %v2653 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v2654 = stablehlo.multiply %v2653, %v2652 : tensor<32x602112xf32>
    %v2655 = stablehlo.multiply %v2650, %v2650 : tensor<32x602112xf32>
    %v2656 = stablehlo.subtract %v2651, %v2655 : tensor<32x602112xf32>
    %v2657 = stablehlo.multiply %v2653, %v310 : tensor<32x602112xf32>
    %v2658 = stablehlo.multiply %v2657, %v2656 : tensor<32x602112xf32>
    %v2659 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v2660 = stablehlo.multiply %v2659, %v2643 : tensor<32x602112xf32>
    %v2661 = stablehlo.add %v2651, %v2660 : tensor<32x602112xf32>
    %v2662 = stablehlo.multiply %v2648, %v2661 : tensor<32x602112xf32>
    %v2663 = stablehlo.multiply %v2658, %v2662 : tensor<32x602112xf32>
    %v2664 = stablehlo.add %v2654, %v2663 : tensor<32x602112xf32>
    %v2665 = stablehlo.multiply %v2642, %v2664 : tensor<32x602112xf32>
    %v2666 = stablehlo.reshape %v2665 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2667 = stablehlo.transpose %s1b2eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2668 = stablehlo.reverse %v2667, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v2669 = stablehlo.convolution(%v2666, %v2668)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v2670 = stablehlo.reshape %v2669 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2672 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2673 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2674 = stablehlo.reduce(%v287 init: %v2671) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2675 = stablehlo.broadcast_in_dim %v2674, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2676 = stablehlo.divide %v2675, %v2672 : tensor<32x150528xf32>
    %v2677 = stablehlo.subtract %v287, %v2676 : tensor<32x150528xf32>
    %v2678 = stablehlo.multiply %v2677, %v2677 : tensor<32x150528xf32>
    %v2679 = stablehlo.reduce(%v2678 init: %v2671) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2680 = stablehlo.broadcast_in_dim %v2679, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2681 = stablehlo.divide %v2680, %v2672 : tensor<32x150528xf32>
    %v2682 = stablehlo.add %v2681, %v2673 : tensor<32x150528xf32>
    %v2683 = stablehlo.rsqrt %v2682 : tensor<32x150528xf32>
    %v2684 = stablehlo.multiply %v2677, %v2683 : tensor<32x150528xf32>
    %v2685 = stablehlo.broadcast_in_dim %s1b2ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v2686 = stablehlo.multiply %v2685, %v2670 : tensor<32x150528xf32>
    %v2687 = stablehlo.reduce(%v2686 init: %v2671) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2688 = stablehlo.broadcast_in_dim %v2687, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2689 = stablehlo.multiply %v2684, %v2686 : tensor<32x150528xf32>
    %v2690 = stablehlo.reduce(%v2689 init: %v2671) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2691 = stablehlo.broadcast_in_dim %v2690, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2692 = stablehlo.multiply %v2686, %v2672 : tensor<32x150528xf32>
    %v2693 = stablehlo.subtract %v2692, %v2688 : tensor<32x150528xf32>
    %v2694 = stablehlo.multiply %v2684, %v2691 : tensor<32x150528xf32>
    %v2695 = stablehlo.subtract %v2693, %v2694 : tensor<32x150528xf32>
    %v2696 = stablehlo.divide %v2683, %v2672 : tensor<32x150528xf32>
    %v2697 = stablehlo.multiply %v2696, %v2695 : tensor<32x150528xf32>
    %v2698 = stablehlo.reshape %v2697 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2699 = stablehlo.reverse %s1b2dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v2700 = stablehlo.convolution(%v2698, %v2699)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v2701 = stablehlo.reshape %v2700 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2702 = stablehlo.add %v2701, %v2604 : tensor<32x150528xf32>
    %v2703 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2704 = stablehlo.reshape %v328 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2705 = stablehlo.reshape %v2604 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2706 = stablehlo.multiply %v2704, %v2705 : tensor<32x192x28x28xf32>
    %v2707 = stablehlo.reduce(%v2706 init: %v2703) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2708 = stablehlo.reshape %v323 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2709 = stablehlo.reshape %v2637 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2710 = stablehlo.transpose %v2708, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v2711 = stablehlo.transpose %v2709, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2712 = stablehlo.convolution(%v2710, %v2711)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v2713 = stablehlo.transpose %v2712, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2714 = stablehlo.reshape %v2637 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2716 = stablehlo.reduce(%v2714 init: %v2715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2717 = stablehlo.reshape %v305 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2718 = stablehlo.reshape %v2665 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2719 = stablehlo.transpose %v2717, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2720 = stablehlo.transpose %v2718, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v2721 = stablehlo.convolution(%v2719, %v2720)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v2722 = stablehlo.transpose %v2721, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2723 = stablehlo.reshape %v2665 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2725 = stablehlo.reduce(%v2723 init: %v2724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v2726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2727 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2728 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2729 = stablehlo.reduce(%v287 init: %v2726) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2730 = stablehlo.broadcast_in_dim %v2729, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2731 = stablehlo.divide %v2730, %v2727 : tensor<32x150528xf32>
    %v2732 = stablehlo.subtract %v287, %v2731 : tensor<32x150528xf32>
    %v2733 = stablehlo.multiply %v2732, %v2732 : tensor<32x150528xf32>
    %v2734 = stablehlo.reduce(%v2733 init: %v2726) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2735 = stablehlo.broadcast_in_dim %v2734, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2736 = stablehlo.divide %v2735, %v2727 : tensor<32x150528xf32>
    %v2737 = stablehlo.add %v2736, %v2728 : tensor<32x150528xf32>
    %v2738 = stablehlo.rsqrt %v2737 : tensor<32x150528xf32>
    %v2739 = stablehlo.multiply %v2732, %v2738 : tensor<32x150528xf32>
    %v2740 = stablehlo.multiply %v2670, %v2739 : tensor<32x150528xf32>
    %v2741 = stablehlo.reduce(%v2740 init: %v2726) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2743 = stablehlo.reduce(%v2670 init: %v2742) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2744 = stablehlo.reshape %v282 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2745 = stablehlo.reshape %v2697 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2746 = stablehlo.transpose %v2744, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2747 = stablehlo.transpose %v2745, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2748 = stablehlo.convolution(%v2746, %v2747)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v2749 = stablehlo.reshape %v2748 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v2750 = stablehlo.reshape %v2697 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2752 = stablehlo.reduce(%v2750 init: %v2751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2753 = stablehlo.reshape %v2702 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2754 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v2755 = stablehlo.multiply %v2753, %v2754 : tensor<32x192x28x28xf32>
    %v2756 = stablehlo.reshape %v2755 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2757 = stablehlo.reshape %v2756 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2758 = stablehlo.transpose %s1b1pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2759 = stablehlo.reverse %v2758, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v2760 = stablehlo.convolution(%v2757, %v2759)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v2761 = stablehlo.reshape %v2760 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v2762 = stablehlo.multiply %v259, %v259 : tensor<32x602112xf32>
    %v2763 = stablehlo.multiply %v2762, %v259 : tensor<32x602112xf32>
    %v2764 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v2765 = stablehlo.multiply %v2764, %v2763 : tensor<32x602112xf32>
    %v2766 = stablehlo.add %v259, %v2765 : tensor<32x602112xf32>
    %v2767 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v2768 = stablehlo.multiply %v2767, %v2766 : tensor<32x602112xf32>
    %v2769 = stablehlo.tanh %v2768 : tensor<32x602112xf32>
    %v2770 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v2771 = stablehlo.add %v2770, %v2769 : tensor<32x602112xf32>
    %v2772 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v2773 = stablehlo.multiply %v2772, %v2771 : tensor<32x602112xf32>
    %v2774 = stablehlo.multiply %v2769, %v2769 : tensor<32x602112xf32>
    %v2775 = stablehlo.subtract %v2770, %v2774 : tensor<32x602112xf32>
    %v2776 = stablehlo.multiply %v2772, %v259 : tensor<32x602112xf32>
    %v2777 = stablehlo.multiply %v2776, %v2775 : tensor<32x602112xf32>
    %v2778 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v2779 = stablehlo.multiply %v2778, %v2762 : tensor<32x602112xf32>
    %v2780 = stablehlo.add %v2770, %v2779 : tensor<32x602112xf32>
    %v2781 = stablehlo.multiply %v2767, %v2780 : tensor<32x602112xf32>
    %v2782 = stablehlo.multiply %v2777, %v2781 : tensor<32x602112xf32>
    %v2783 = stablehlo.add %v2773, %v2782 : tensor<32x602112xf32>
    %v2784 = stablehlo.multiply %v2761, %v2783 : tensor<32x602112xf32>
    %v2785 = stablehlo.reshape %v2784 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2786 = stablehlo.transpose %s1b1eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2787 = stablehlo.reverse %v2786, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v2788 = stablehlo.convolution(%v2785, %v2787)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v2789 = stablehlo.reshape %v2788 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2791 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2792 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2793 = stablehlo.reduce(%v236 init: %v2790) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2794 = stablehlo.broadcast_in_dim %v2793, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2795 = stablehlo.divide %v2794, %v2791 : tensor<32x150528xf32>
    %v2796 = stablehlo.subtract %v236, %v2795 : tensor<32x150528xf32>
    %v2797 = stablehlo.multiply %v2796, %v2796 : tensor<32x150528xf32>
    %v2798 = stablehlo.reduce(%v2797 init: %v2790) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2799 = stablehlo.broadcast_in_dim %v2798, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2800 = stablehlo.divide %v2799, %v2791 : tensor<32x150528xf32>
    %v2801 = stablehlo.add %v2800, %v2792 : tensor<32x150528xf32>
    %v2802 = stablehlo.rsqrt %v2801 : tensor<32x150528xf32>
    %v2803 = stablehlo.multiply %v2796, %v2802 : tensor<32x150528xf32>
    %v2804 = stablehlo.broadcast_in_dim %s1b1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v2805 = stablehlo.multiply %v2804, %v2789 : tensor<32x150528xf32>
    %v2806 = stablehlo.reduce(%v2805 init: %v2790) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2807 = stablehlo.broadcast_in_dim %v2806, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2808 = stablehlo.multiply %v2803, %v2805 : tensor<32x150528xf32>
    %v2809 = stablehlo.reduce(%v2808 init: %v2790) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2810 = stablehlo.broadcast_in_dim %v2809, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2811 = stablehlo.multiply %v2805, %v2791 : tensor<32x150528xf32>
    %v2812 = stablehlo.subtract %v2811, %v2807 : tensor<32x150528xf32>
    %v2813 = stablehlo.multiply %v2803, %v2810 : tensor<32x150528xf32>
    %v2814 = stablehlo.subtract %v2812, %v2813 : tensor<32x150528xf32>
    %v2815 = stablehlo.divide %v2802, %v2791 : tensor<32x150528xf32>
    %v2816 = stablehlo.multiply %v2815, %v2814 : tensor<32x150528xf32>
    %v2817 = stablehlo.reshape %v2816 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2818 = stablehlo.reverse %s1b1dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v2819 = stablehlo.convolution(%v2817, %v2818)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v2820 = stablehlo.reshape %v2819 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2821 = stablehlo.add %v2820, %v2702 : tensor<32x150528xf32>
    %v2822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2823 = stablehlo.reshape %v277 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2824 = stablehlo.reshape %v2702 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2825 = stablehlo.multiply %v2823, %v2824 : tensor<32x192x28x28xf32>
    %v2826 = stablehlo.reduce(%v2825 init: %v2822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2827 = stablehlo.reshape %v272 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2828 = stablehlo.reshape %v2756 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2829 = stablehlo.transpose %v2827, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v2830 = stablehlo.transpose %v2828, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2831 = stablehlo.convolution(%v2829, %v2830)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v2832 = stablehlo.transpose %v2831, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2833 = stablehlo.reshape %v2756 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2835 = stablehlo.reduce(%v2833 init: %v2834) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2836 = stablehlo.reshape %v254 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2837 = stablehlo.reshape %v2784 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2838 = stablehlo.transpose %v2836, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2839 = stablehlo.transpose %v2837, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v2840 = stablehlo.convolution(%v2838, %v2839)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v2841 = stablehlo.transpose %v2840, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2842 = stablehlo.reshape %v2784 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2844 = stablehlo.reduce(%v2842 init: %v2843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v2845 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2846 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2847 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2848 = stablehlo.reduce(%v236 init: %v2845) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2849 = stablehlo.broadcast_in_dim %v2848, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2850 = stablehlo.divide %v2849, %v2846 : tensor<32x150528xf32>
    %v2851 = stablehlo.subtract %v236, %v2850 : tensor<32x150528xf32>
    %v2852 = stablehlo.multiply %v2851, %v2851 : tensor<32x150528xf32>
    %v2853 = stablehlo.reduce(%v2852 init: %v2845) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2854 = stablehlo.broadcast_in_dim %v2853, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2855 = stablehlo.divide %v2854, %v2846 : tensor<32x150528xf32>
    %v2856 = stablehlo.add %v2855, %v2847 : tensor<32x150528xf32>
    %v2857 = stablehlo.rsqrt %v2856 : tensor<32x150528xf32>
    %v2858 = stablehlo.multiply %v2851, %v2857 : tensor<32x150528xf32>
    %v2859 = stablehlo.multiply %v2789, %v2858 : tensor<32x150528xf32>
    %v2860 = stablehlo.reduce(%v2859 init: %v2845) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2862 = stablehlo.reduce(%v2789 init: %v2861) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2863 = stablehlo.reshape %v231 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2864 = stablehlo.reshape %v2816 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2865 = stablehlo.transpose %v2863, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2866 = stablehlo.transpose %v2864, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2867 = stablehlo.convolution(%v2865, %v2866)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v2868 = stablehlo.reshape %v2867 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v2869 = stablehlo.reshape %v2816 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2870 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2871 = stablehlo.reduce(%v2869 init: %v2870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2872 = stablehlo.reshape %v2821 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2873 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v2874 = stablehlo.multiply %v2872, %v2873 : tensor<32x192x28x28xf32>
    %v2875 = stablehlo.reshape %v2874 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2876 = stablehlo.reshape %v2875 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2877 = stablehlo.transpose %s1b0pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2878 = stablehlo.reverse %v2877, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v2879 = stablehlo.convolution(%v2876, %v2878)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v2880 = stablehlo.reshape %v2879 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v2881 = stablehlo.multiply %v208, %v208 : tensor<32x602112xf32>
    %v2882 = stablehlo.multiply %v2881, %v208 : tensor<32x602112xf32>
    %v2883 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v2884 = stablehlo.multiply %v2883, %v2882 : tensor<32x602112xf32>
    %v2885 = stablehlo.add %v208, %v2884 : tensor<32x602112xf32>
    %v2886 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v2887 = stablehlo.multiply %v2886, %v2885 : tensor<32x602112xf32>
    %v2888 = stablehlo.tanh %v2887 : tensor<32x602112xf32>
    %v2889 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v2890 = stablehlo.add %v2889, %v2888 : tensor<32x602112xf32>
    %v2891 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v2892 = stablehlo.multiply %v2891, %v2890 : tensor<32x602112xf32>
    %v2893 = stablehlo.multiply %v2888, %v2888 : tensor<32x602112xf32>
    %v2894 = stablehlo.subtract %v2889, %v2893 : tensor<32x602112xf32>
    %v2895 = stablehlo.multiply %v2891, %v208 : tensor<32x602112xf32>
    %v2896 = stablehlo.multiply %v2895, %v2894 : tensor<32x602112xf32>
    %v2897 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v2898 = stablehlo.multiply %v2897, %v2881 : tensor<32x602112xf32>
    %v2899 = stablehlo.add %v2889, %v2898 : tensor<32x602112xf32>
    %v2900 = stablehlo.multiply %v2886, %v2899 : tensor<32x602112xf32>
    %v2901 = stablehlo.multiply %v2896, %v2900 : tensor<32x602112xf32>
    %v2902 = stablehlo.add %v2892, %v2901 : tensor<32x602112xf32>
    %v2903 = stablehlo.multiply %v2880, %v2902 : tensor<32x602112xf32>
    %v2904 = stablehlo.reshape %v2903 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2905 = stablehlo.transpose %s1b0eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2906 = stablehlo.reverse %v2905, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v2907 = stablehlo.convolution(%v2904, %v2906)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v2908 = stablehlo.reshape %v2907 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2910 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2911 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2912 = stablehlo.reduce(%v185 init: %v2909) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2913 = stablehlo.broadcast_in_dim %v2912, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2914 = stablehlo.divide %v2913, %v2910 : tensor<32x150528xf32>
    %v2915 = stablehlo.subtract %v185, %v2914 : tensor<32x150528xf32>
    %v2916 = stablehlo.multiply %v2915, %v2915 : tensor<32x150528xf32>
    %v2917 = stablehlo.reduce(%v2916 init: %v2909) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2918 = stablehlo.broadcast_in_dim %v2917, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2919 = stablehlo.divide %v2918, %v2910 : tensor<32x150528xf32>
    %v2920 = stablehlo.add %v2919, %v2911 : tensor<32x150528xf32>
    %v2921 = stablehlo.rsqrt %v2920 : tensor<32x150528xf32>
    %v2922 = stablehlo.multiply %v2915, %v2921 : tensor<32x150528xf32>
    %v2923 = stablehlo.broadcast_in_dim %s1b0ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v2924 = stablehlo.multiply %v2923, %v2908 : tensor<32x150528xf32>
    %v2925 = stablehlo.reduce(%v2924 init: %v2909) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2926 = stablehlo.broadcast_in_dim %v2925, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2927 = stablehlo.multiply %v2922, %v2924 : tensor<32x150528xf32>
    %v2928 = stablehlo.reduce(%v2927 init: %v2909) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2929 = stablehlo.broadcast_in_dim %v2928, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2930 = stablehlo.multiply %v2924, %v2910 : tensor<32x150528xf32>
    %v2931 = stablehlo.subtract %v2930, %v2926 : tensor<32x150528xf32>
    %v2932 = stablehlo.multiply %v2922, %v2929 : tensor<32x150528xf32>
    %v2933 = stablehlo.subtract %v2931, %v2932 : tensor<32x150528xf32>
    %v2934 = stablehlo.divide %v2921, %v2910 : tensor<32x150528xf32>
    %v2935 = stablehlo.multiply %v2934, %v2933 : tensor<32x150528xf32>
    %v2936 = stablehlo.reshape %v2935 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2937 = stablehlo.reverse %s1b0dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v2938 = stablehlo.convolution(%v2936, %v2937)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v2939 = stablehlo.reshape %v2938 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2940 = stablehlo.add %v2939, %v2821 : tensor<32x150528xf32>
    %v2941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2942 = stablehlo.reshape %v226 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2943 = stablehlo.reshape %v2821 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2944 = stablehlo.multiply %v2942, %v2943 : tensor<32x192x28x28xf32>
    %v2945 = stablehlo.reduce(%v2944 init: %v2941) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2946 = stablehlo.reshape %v221 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2947 = stablehlo.reshape %v2875 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2948 = stablehlo.transpose %v2946, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v2949 = stablehlo.transpose %v2947, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2950 = stablehlo.convolution(%v2948, %v2949)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v2951 = stablehlo.transpose %v2950, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2952 = stablehlo.reshape %v2875 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2953 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2954 = stablehlo.reduce(%v2952 init: %v2953) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2955 = stablehlo.reshape %v203 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2956 = stablehlo.reshape %v2903 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2957 = stablehlo.transpose %v2955, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2958 = stablehlo.transpose %v2956, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v2959 = stablehlo.convolution(%v2957, %v2958)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v2960 = stablehlo.transpose %v2959, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2961 = stablehlo.reshape %v2903 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2963 = stablehlo.reduce(%v2961 init: %v2962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v2964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2965 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2966 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2967 = stablehlo.reduce(%v185 init: %v2964) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2968 = stablehlo.broadcast_in_dim %v2967, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2969 = stablehlo.divide %v2968, %v2965 : tensor<32x150528xf32>
    %v2970 = stablehlo.subtract %v185, %v2969 : tensor<32x150528xf32>
    %v2971 = stablehlo.multiply %v2970, %v2970 : tensor<32x150528xf32>
    %v2972 = stablehlo.reduce(%v2971 init: %v2964) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2973 = stablehlo.broadcast_in_dim %v2972, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2974 = stablehlo.divide %v2973, %v2965 : tensor<32x150528xf32>
    %v2975 = stablehlo.add %v2974, %v2966 : tensor<32x150528xf32>
    %v2976 = stablehlo.rsqrt %v2975 : tensor<32x150528xf32>
    %v2977 = stablehlo.multiply %v2970, %v2976 : tensor<32x150528xf32>
    %v2978 = stablehlo.multiply %v2908, %v2977 : tensor<32x150528xf32>
    %v2979 = stablehlo.reduce(%v2978 init: %v2964) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2981 = stablehlo.reduce(%v2908 init: %v2980) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2982 = stablehlo.reshape %v180 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2983 = stablehlo.reshape %v2935 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2984 = stablehlo.transpose %v2982, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2985 = stablehlo.transpose %v2983, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2986 = stablehlo.convolution(%v2984, %v2985)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v2987 = stablehlo.reshape %v2986 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v2988 = stablehlo.reshape %v2935 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2990 = stablehlo.reduce(%v2988 init: %v2989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2991 = stablehlo.reshape %v2940 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2993 = stablehlo.pad %v2991, %v2992, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x56x56xf32>
    %v2994 = stablehlo.transpose %d0W, dims = [1, 0, 2, 3] : (tensor<192x96x2x2xf32>) -> tensor<96x192x2x2xf32>
    %v2995 = stablehlo.reverse %v2994, dims = [2, 3] : tensor<96x192x2x2xf32>
    %v2996 = stablehlo.convolution(%v2993, %v2995)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x56x56xf32>, tensor<96x192x2x2xf32>) -> tensor<32x96x56x56xf32>
    %v2997 = stablehlo.reshape %v2996 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v2998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2999 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3000 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3001 = stablehlo.reduce(%v157 init: %v2998) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3002 = stablehlo.broadcast_in_dim %v3001, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3003 = stablehlo.divide %v3002, %v2999 : tensor<32x301056xf32>
    %v3004 = stablehlo.subtract %v157, %v3003 : tensor<32x301056xf32>
    %v3005 = stablehlo.multiply %v3004, %v3004 : tensor<32x301056xf32>
    %v3006 = stablehlo.reduce(%v3005 init: %v2998) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3007 = stablehlo.broadcast_in_dim %v3006, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3008 = stablehlo.divide %v3007, %v2999 : tensor<32x301056xf32>
    %v3009 = stablehlo.add %v3008, %v3000 : tensor<32x301056xf32>
    %v3010 = stablehlo.rsqrt %v3009 : tensor<32x301056xf32>
    %v3011 = stablehlo.multiply %v3004, %v3010 : tensor<32x301056xf32>
    %v3012 = stablehlo.broadcast_in_dim %d0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3013 = stablehlo.multiply %v3012, %v2997 : tensor<32x301056xf32>
    %v3014 = stablehlo.reduce(%v3013 init: %v2998) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3015 = stablehlo.broadcast_in_dim %v3014, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3016 = stablehlo.multiply %v3011, %v3013 : tensor<32x301056xf32>
    %v3017 = stablehlo.reduce(%v3016 init: %v2998) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3018 = stablehlo.broadcast_in_dim %v3017, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3019 = stablehlo.multiply %v3013, %v2999 : tensor<32x301056xf32>
    %v3020 = stablehlo.subtract %v3019, %v3015 : tensor<32x301056xf32>
    %v3021 = stablehlo.multiply %v3011, %v3018 : tensor<32x301056xf32>
    %v3022 = stablehlo.subtract %v3020, %v3021 : tensor<32x301056xf32>
    %v3023 = stablehlo.divide %v3010, %v2999 : tensor<32x301056xf32>
    %v3024 = stablehlo.multiply %v3023, %v3022 : tensor<32x301056xf32>
    %v3025 = stablehlo.reshape %v2940 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3026 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3027 = stablehlo.reduce(%v3025 init: %v3026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3028 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3029 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3030 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3031 = stablehlo.reduce(%v157 init: %v3028) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3032 = stablehlo.broadcast_in_dim %v3031, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3033 = stablehlo.divide %v3032, %v3029 : tensor<32x301056xf32>
    %v3034 = stablehlo.subtract %v157, %v3033 : tensor<32x301056xf32>
    %v3035 = stablehlo.multiply %v3034, %v3034 : tensor<32x301056xf32>
    %v3036 = stablehlo.reduce(%v3035 init: %v3028) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3037 = stablehlo.broadcast_in_dim %v3036, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3038 = stablehlo.divide %v3037, %v3029 : tensor<32x301056xf32>
    %v3039 = stablehlo.add %v3038, %v3030 : tensor<32x301056xf32>
    %v3040 = stablehlo.rsqrt %v3039 : tensor<32x301056xf32>
    %v3041 = stablehlo.multiply %v3034, %v3040 : tensor<32x301056xf32>
    %v3042 = stablehlo.multiply %v2997, %v3041 : tensor<32x301056xf32>
    %v3043 = stablehlo.reduce(%v3042 init: %v3028) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3045 = stablehlo.reduce(%v2997 init: %v3044) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3046 = stablehlo.reshape %v175 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3047 = stablehlo.reshape %v2940 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3049 = stablehlo.pad %v3047, %v3048, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x55x55xf32>
    %v3050 = stablehlo.transpose %v3046, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3051 = stablehlo.transpose %v3049, dims = [1, 0, 2, 3] : (tensor<32x192x55x55xf32>) -> tensor<192x32x55x55xf32>
    %v3052 = stablehlo.convolution(%v3050, %v3051)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<192x32x55x55xf32>) -> tensor<96x192x2x2xf32>
    %v3053 = stablehlo.transpose %v3052, dims = [1, 0, 2, 3] : (tensor<96x192x2x2xf32>) -> tensor<192x96x2x2xf32>
    %v3054 = stablehlo.reshape %v3024 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3055 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3056 = stablehlo.multiply %v3054, %v3055 : tensor<32x96x56x56xf32>
    %v3057 = stablehlo.reshape %v3056 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3058 = stablehlo.reshape %v3057 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3059 = stablehlo.transpose %s0b2pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3060 = stablehlo.reverse %v3059, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3061 = stablehlo.convolution(%v3058, %v3060)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3062 = stablehlo.reshape %v3061 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3063 = stablehlo.multiply %v134, %v134 : tensor<32x1204224xf32>
    %v3064 = stablehlo.multiply %v3063, %v134 : tensor<32x1204224xf32>
    %v3065 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3066 = stablehlo.multiply %v3065, %v3064 : tensor<32x1204224xf32>
    %v3067 = stablehlo.add %v134, %v3066 : tensor<32x1204224xf32>
    %v3068 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3069 = stablehlo.multiply %v3068, %v3067 : tensor<32x1204224xf32>
    %v3070 = stablehlo.tanh %v3069 : tensor<32x1204224xf32>
    %v3071 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3072 = stablehlo.add %v3071, %v3070 : tensor<32x1204224xf32>
    %v3073 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3074 = stablehlo.multiply %v3073, %v3072 : tensor<32x1204224xf32>
    %v3075 = stablehlo.multiply %v3070, %v3070 : tensor<32x1204224xf32>
    %v3076 = stablehlo.subtract %v3071, %v3075 : tensor<32x1204224xf32>
    %v3077 = stablehlo.multiply %v3073, %v134 : tensor<32x1204224xf32>
    %v3078 = stablehlo.multiply %v3077, %v3076 : tensor<32x1204224xf32>
    %v3079 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3080 = stablehlo.multiply %v3079, %v3063 : tensor<32x1204224xf32>
    %v3081 = stablehlo.add %v3071, %v3080 : tensor<32x1204224xf32>
    %v3082 = stablehlo.multiply %v3068, %v3081 : tensor<32x1204224xf32>
    %v3083 = stablehlo.multiply %v3078, %v3082 : tensor<32x1204224xf32>
    %v3084 = stablehlo.add %v3074, %v3083 : tensor<32x1204224xf32>
    %v3085 = stablehlo.multiply %v3062, %v3084 : tensor<32x1204224xf32>
    %v3086 = stablehlo.reshape %v3085 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3087 = stablehlo.transpose %s0b2eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3088 = stablehlo.reverse %v3087, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3089 = stablehlo.convolution(%v3086, %v3088)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3090 = stablehlo.reshape %v3089 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3092 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3093 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3094 = stablehlo.reduce(%v111 init: %v3091) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3095 = stablehlo.broadcast_in_dim %v3094, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3096 = stablehlo.divide %v3095, %v3092 : tensor<32x301056xf32>
    %v3097 = stablehlo.subtract %v111, %v3096 : tensor<32x301056xf32>
    %v3098 = stablehlo.multiply %v3097, %v3097 : tensor<32x301056xf32>
    %v3099 = stablehlo.reduce(%v3098 init: %v3091) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3100 = stablehlo.broadcast_in_dim %v3099, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3101 = stablehlo.divide %v3100, %v3092 : tensor<32x301056xf32>
    %v3102 = stablehlo.add %v3101, %v3093 : tensor<32x301056xf32>
    %v3103 = stablehlo.rsqrt %v3102 : tensor<32x301056xf32>
    %v3104 = stablehlo.multiply %v3097, %v3103 : tensor<32x301056xf32>
    %v3105 = stablehlo.broadcast_in_dim %s0b2ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3106 = stablehlo.multiply %v3105, %v3090 : tensor<32x301056xf32>
    %v3107 = stablehlo.reduce(%v3106 init: %v3091) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3108 = stablehlo.broadcast_in_dim %v3107, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3109 = stablehlo.multiply %v3104, %v3106 : tensor<32x301056xf32>
    %v3110 = stablehlo.reduce(%v3109 init: %v3091) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3111 = stablehlo.broadcast_in_dim %v3110, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3112 = stablehlo.multiply %v3106, %v3092 : tensor<32x301056xf32>
    %v3113 = stablehlo.subtract %v3112, %v3108 : tensor<32x301056xf32>
    %v3114 = stablehlo.multiply %v3104, %v3111 : tensor<32x301056xf32>
    %v3115 = stablehlo.subtract %v3113, %v3114 : tensor<32x301056xf32>
    %v3116 = stablehlo.divide %v3103, %v3092 : tensor<32x301056xf32>
    %v3117 = stablehlo.multiply %v3116, %v3115 : tensor<32x301056xf32>
    %v3118 = stablehlo.reshape %v3117 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3119 = stablehlo.reverse %s0b2dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v3120 = stablehlo.convolution(%v3118, %v3119)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v3121 = stablehlo.reshape %v3120 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3122 = stablehlo.add %v3121, %v3024 : tensor<32x301056xf32>
    %v3123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3124 = stablehlo.reshape %v152 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3125 = stablehlo.reshape %v3024 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3126 = stablehlo.multiply %v3124, %v3125 : tensor<32x96x56x56xf32>
    %v3127 = stablehlo.reduce(%v3126 init: %v3123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3128 = stablehlo.reshape %v147 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3129 = stablehlo.reshape %v3057 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3130 = stablehlo.transpose %v3128, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3131 = stablehlo.transpose %v3129, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3132 = stablehlo.convolution(%v3130, %v3131)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v3133 = stablehlo.transpose %v3132, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3134 = stablehlo.reshape %v3057 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3136 = stablehlo.reduce(%v3134 init: %v3135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3137 = stablehlo.reshape %v129 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3138 = stablehlo.reshape %v3085 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3139 = stablehlo.transpose %v3137, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3140 = stablehlo.transpose %v3138, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3141 = stablehlo.convolution(%v3139, %v3140)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v3142 = stablehlo.transpose %v3141, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3143 = stablehlo.reshape %v3085 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3145 = stablehlo.reduce(%v3143 init: %v3144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v3146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3147 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3148 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3149 = stablehlo.reduce(%v111 init: %v3146) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3150 = stablehlo.broadcast_in_dim %v3149, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3151 = stablehlo.divide %v3150, %v3147 : tensor<32x301056xf32>
    %v3152 = stablehlo.subtract %v111, %v3151 : tensor<32x301056xf32>
    %v3153 = stablehlo.multiply %v3152, %v3152 : tensor<32x301056xf32>
    %v3154 = stablehlo.reduce(%v3153 init: %v3146) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3155 = stablehlo.broadcast_in_dim %v3154, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3156 = stablehlo.divide %v3155, %v3147 : tensor<32x301056xf32>
    %v3157 = stablehlo.add %v3156, %v3148 : tensor<32x301056xf32>
    %v3158 = stablehlo.rsqrt %v3157 : tensor<32x301056xf32>
    %v3159 = stablehlo.multiply %v3152, %v3158 : tensor<32x301056xf32>
    %v3160 = stablehlo.multiply %v3090, %v3159 : tensor<32x301056xf32>
    %v3161 = stablehlo.reduce(%v3160 init: %v3146) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3163 = stablehlo.reduce(%v3090 init: %v3162) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3164 = stablehlo.reshape %v106 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3165 = stablehlo.reshape %v3117 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3166 = stablehlo.transpose %v3164, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3167 = stablehlo.transpose %v3165, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3168 = stablehlo.convolution(%v3166, %v3167)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v3169 = stablehlo.reshape %v3168 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v3170 = stablehlo.reshape %v3117 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3172 = stablehlo.reduce(%v3170 init: %v3171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3173 = stablehlo.reshape %v3122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3174 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3175 = stablehlo.multiply %v3173, %v3174 : tensor<32x96x56x56xf32>
    %v3176 = stablehlo.reshape %v3175 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3177 = stablehlo.reshape %v3176 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3178 = stablehlo.transpose %s0b1pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3179 = stablehlo.reverse %v3178, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3180 = stablehlo.convolution(%v3177, %v3179)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3181 = stablehlo.reshape %v3180 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3182 = stablehlo.multiply %v83, %v83 : tensor<32x1204224xf32>
    %v3183 = stablehlo.multiply %v3182, %v83 : tensor<32x1204224xf32>
    %v3184 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3185 = stablehlo.multiply %v3184, %v3183 : tensor<32x1204224xf32>
    %v3186 = stablehlo.add %v83, %v3185 : tensor<32x1204224xf32>
    %v3187 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3188 = stablehlo.multiply %v3187, %v3186 : tensor<32x1204224xf32>
    %v3189 = stablehlo.tanh %v3188 : tensor<32x1204224xf32>
    %v3190 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3191 = stablehlo.add %v3190, %v3189 : tensor<32x1204224xf32>
    %v3192 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3193 = stablehlo.multiply %v3192, %v3191 : tensor<32x1204224xf32>
    %v3194 = stablehlo.multiply %v3189, %v3189 : tensor<32x1204224xf32>
    %v3195 = stablehlo.subtract %v3190, %v3194 : tensor<32x1204224xf32>
    %v3196 = stablehlo.multiply %v3192, %v83 : tensor<32x1204224xf32>
    %v3197 = stablehlo.multiply %v3196, %v3195 : tensor<32x1204224xf32>
    %v3198 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3199 = stablehlo.multiply %v3198, %v3182 : tensor<32x1204224xf32>
    %v3200 = stablehlo.add %v3190, %v3199 : tensor<32x1204224xf32>
    %v3201 = stablehlo.multiply %v3187, %v3200 : tensor<32x1204224xf32>
    %v3202 = stablehlo.multiply %v3197, %v3201 : tensor<32x1204224xf32>
    %v3203 = stablehlo.add %v3193, %v3202 : tensor<32x1204224xf32>
    %v3204 = stablehlo.multiply %v3181, %v3203 : tensor<32x1204224xf32>
    %v3205 = stablehlo.reshape %v3204 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3206 = stablehlo.transpose %s0b1eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3207 = stablehlo.reverse %v3206, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3208 = stablehlo.convolution(%v3205, %v3207)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3209 = stablehlo.reshape %v3208 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3211 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3212 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3213 = stablehlo.reduce(%v60 init: %v3210) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3214 = stablehlo.broadcast_in_dim %v3213, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3215 = stablehlo.divide %v3214, %v3211 : tensor<32x301056xf32>
    %v3216 = stablehlo.subtract %v60, %v3215 : tensor<32x301056xf32>
    %v3217 = stablehlo.multiply %v3216, %v3216 : tensor<32x301056xf32>
    %v3218 = stablehlo.reduce(%v3217 init: %v3210) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3219 = stablehlo.broadcast_in_dim %v3218, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3220 = stablehlo.divide %v3219, %v3211 : tensor<32x301056xf32>
    %v3221 = stablehlo.add %v3220, %v3212 : tensor<32x301056xf32>
    %v3222 = stablehlo.rsqrt %v3221 : tensor<32x301056xf32>
    %v3223 = stablehlo.multiply %v3216, %v3222 : tensor<32x301056xf32>
    %v3224 = stablehlo.broadcast_in_dim %s0b1ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3225 = stablehlo.multiply %v3224, %v3209 : tensor<32x301056xf32>
    %v3226 = stablehlo.reduce(%v3225 init: %v3210) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3227 = stablehlo.broadcast_in_dim %v3226, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3228 = stablehlo.multiply %v3223, %v3225 : tensor<32x301056xf32>
    %v3229 = stablehlo.reduce(%v3228 init: %v3210) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3230 = stablehlo.broadcast_in_dim %v3229, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3231 = stablehlo.multiply %v3225, %v3211 : tensor<32x301056xf32>
    %v3232 = stablehlo.subtract %v3231, %v3227 : tensor<32x301056xf32>
    %v3233 = stablehlo.multiply %v3223, %v3230 : tensor<32x301056xf32>
    %v3234 = stablehlo.subtract %v3232, %v3233 : tensor<32x301056xf32>
    %v3235 = stablehlo.divide %v3222, %v3211 : tensor<32x301056xf32>
    %v3236 = stablehlo.multiply %v3235, %v3234 : tensor<32x301056xf32>
    %v3237 = stablehlo.reshape %v3236 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3238 = stablehlo.reverse %s0b1dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v3239 = stablehlo.convolution(%v3237, %v3238)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v3240 = stablehlo.reshape %v3239 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3241 = stablehlo.add %v3240, %v3122 : tensor<32x301056xf32>
    %v3242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3243 = stablehlo.reshape %v101 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3244 = stablehlo.reshape %v3122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3245 = stablehlo.multiply %v3243, %v3244 : tensor<32x96x56x56xf32>
    %v3246 = stablehlo.reduce(%v3245 init: %v3242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3247 = stablehlo.reshape %v96 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3248 = stablehlo.reshape %v3176 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3249 = stablehlo.transpose %v3247, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3250 = stablehlo.transpose %v3248, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3251 = stablehlo.convolution(%v3249, %v3250)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v3252 = stablehlo.transpose %v3251, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3253 = stablehlo.reshape %v3176 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3255 = stablehlo.reduce(%v3253 init: %v3254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3256 = stablehlo.reshape %v78 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3257 = stablehlo.reshape %v3204 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3258 = stablehlo.transpose %v3256, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3259 = stablehlo.transpose %v3257, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3260 = stablehlo.convolution(%v3258, %v3259)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v3261 = stablehlo.transpose %v3260, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3262 = stablehlo.reshape %v3204 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3264 = stablehlo.reduce(%v3262 init: %v3263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v3265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3266 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3267 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3268 = stablehlo.reduce(%v60 init: %v3265) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3269 = stablehlo.broadcast_in_dim %v3268, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3270 = stablehlo.divide %v3269, %v3266 : tensor<32x301056xf32>
    %v3271 = stablehlo.subtract %v60, %v3270 : tensor<32x301056xf32>
    %v3272 = stablehlo.multiply %v3271, %v3271 : tensor<32x301056xf32>
    %v3273 = stablehlo.reduce(%v3272 init: %v3265) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3274 = stablehlo.broadcast_in_dim %v3273, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3275 = stablehlo.divide %v3274, %v3266 : tensor<32x301056xf32>
    %v3276 = stablehlo.add %v3275, %v3267 : tensor<32x301056xf32>
    %v3277 = stablehlo.rsqrt %v3276 : tensor<32x301056xf32>
    %v3278 = stablehlo.multiply %v3271, %v3277 : tensor<32x301056xf32>
    %v3279 = stablehlo.multiply %v3209, %v3278 : tensor<32x301056xf32>
    %v3280 = stablehlo.reduce(%v3279 init: %v3265) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3282 = stablehlo.reduce(%v3209 init: %v3281) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3283 = stablehlo.reshape %v55 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3284 = stablehlo.reshape %v3236 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3285 = stablehlo.transpose %v3283, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3286 = stablehlo.transpose %v3284, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3287 = stablehlo.convolution(%v3285, %v3286)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v3288 = stablehlo.reshape %v3287 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v3289 = stablehlo.reshape %v3236 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3291 = stablehlo.reduce(%v3289 init: %v3290) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3292 = stablehlo.reshape %v3241 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3293 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3294 = stablehlo.multiply %v3292, %v3293 : tensor<32x96x56x56xf32>
    %v3295 = stablehlo.reshape %v3294 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3296 = stablehlo.reshape %v3295 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3297 = stablehlo.transpose %s0b0pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3298 = stablehlo.reverse %v3297, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3299 = stablehlo.convolution(%v3296, %v3298)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3300 = stablehlo.reshape %v3299 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3301 = stablehlo.multiply %v32, %v32 : tensor<32x1204224xf32>
    %v3302 = stablehlo.multiply %v3301, %v32 : tensor<32x1204224xf32>
    %v3303 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3304 = stablehlo.multiply %v3303, %v3302 : tensor<32x1204224xf32>
    %v3305 = stablehlo.add %v32, %v3304 : tensor<32x1204224xf32>
    %v3306 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3307 = stablehlo.multiply %v3306, %v3305 : tensor<32x1204224xf32>
    %v3308 = stablehlo.tanh %v3307 : tensor<32x1204224xf32>
    %v3309 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3310 = stablehlo.add %v3309, %v3308 : tensor<32x1204224xf32>
    %v3311 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3312 = stablehlo.multiply %v3311, %v3310 : tensor<32x1204224xf32>
    %v3313 = stablehlo.multiply %v3308, %v3308 : tensor<32x1204224xf32>
    %v3314 = stablehlo.subtract %v3309, %v3313 : tensor<32x1204224xf32>
    %v3315 = stablehlo.multiply %v3311, %v32 : tensor<32x1204224xf32>
    %v3316 = stablehlo.multiply %v3315, %v3314 : tensor<32x1204224xf32>
    %v3317 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3318 = stablehlo.multiply %v3317, %v3301 : tensor<32x1204224xf32>
    %v3319 = stablehlo.add %v3309, %v3318 : tensor<32x1204224xf32>
    %v3320 = stablehlo.multiply %v3306, %v3319 : tensor<32x1204224xf32>
    %v3321 = stablehlo.multiply %v3316, %v3320 : tensor<32x1204224xf32>
    %v3322 = stablehlo.add %v3312, %v3321 : tensor<32x1204224xf32>
    %v3323 = stablehlo.multiply %v3300, %v3322 : tensor<32x1204224xf32>
    %v3324 = stablehlo.reshape %v3323 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3325 = stablehlo.transpose %s0b0eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3326 = stablehlo.reverse %v3325, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3327 = stablehlo.convolution(%v3324, %v3326)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3328 = stablehlo.reshape %v3327 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3330 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3331 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3332 = stablehlo.reduce(%v9 init: %v3329) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3333 = stablehlo.broadcast_in_dim %v3332, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3334 = stablehlo.divide %v3333, %v3330 : tensor<32x301056xf32>
    %v3335 = stablehlo.subtract %v9, %v3334 : tensor<32x301056xf32>
    %v3336 = stablehlo.multiply %v3335, %v3335 : tensor<32x301056xf32>
    %v3337 = stablehlo.reduce(%v3336 init: %v3329) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3338 = stablehlo.broadcast_in_dim %v3337, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3339 = stablehlo.divide %v3338, %v3330 : tensor<32x301056xf32>
    %v3340 = stablehlo.add %v3339, %v3331 : tensor<32x301056xf32>
    %v3341 = stablehlo.rsqrt %v3340 : tensor<32x301056xf32>
    %v3342 = stablehlo.multiply %v3335, %v3341 : tensor<32x301056xf32>
    %v3343 = stablehlo.broadcast_in_dim %s0b0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3344 = stablehlo.multiply %v3343, %v3328 : tensor<32x301056xf32>
    %v3345 = stablehlo.reduce(%v3344 init: %v3329) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3346 = stablehlo.broadcast_in_dim %v3345, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3347 = stablehlo.multiply %v3342, %v3344 : tensor<32x301056xf32>
    %v3348 = stablehlo.reduce(%v3347 init: %v3329) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3349 = stablehlo.broadcast_in_dim %v3348, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3350 = stablehlo.multiply %v3344, %v3330 : tensor<32x301056xf32>
    %v3351 = stablehlo.subtract %v3350, %v3346 : tensor<32x301056xf32>
    %v3352 = stablehlo.multiply %v3342, %v3349 : tensor<32x301056xf32>
    %v3353 = stablehlo.subtract %v3351, %v3352 : tensor<32x301056xf32>
    %v3354 = stablehlo.divide %v3341, %v3330 : tensor<32x301056xf32>
    %v3355 = stablehlo.multiply %v3354, %v3353 : tensor<32x301056xf32>
    %v3356 = stablehlo.reshape %v3355 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3357 = stablehlo.reverse %s0b0dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v3358 = stablehlo.convolution(%v3356, %v3357)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v3359 = stablehlo.reshape %v3358 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3360 = stablehlo.add %v3359, %v3241 : tensor<32x301056xf32>
    %v3361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3362 = stablehlo.reshape %v50 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3363 = stablehlo.reshape %v3241 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3364 = stablehlo.multiply %v3362, %v3363 : tensor<32x96x56x56xf32>
    %v3365 = stablehlo.reduce(%v3364 init: %v3361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3366 = stablehlo.reshape %v45 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3367 = stablehlo.reshape %v3295 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3368 = stablehlo.transpose %v3366, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3369 = stablehlo.transpose %v3367, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3370 = stablehlo.convolution(%v3368, %v3369)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v3371 = stablehlo.transpose %v3370, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3372 = stablehlo.reshape %v3295 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3374 = stablehlo.reduce(%v3372 init: %v3373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3375 = stablehlo.reshape %v27 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3376 = stablehlo.reshape %v3323 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3377 = stablehlo.transpose %v3375, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3378 = stablehlo.transpose %v3376, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3379 = stablehlo.convolution(%v3377, %v3378)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v3380 = stablehlo.transpose %v3379, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3381 = stablehlo.reshape %v3323 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3383 = stablehlo.reduce(%v3381 init: %v3382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v3384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3385 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3386 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3387 = stablehlo.reduce(%v9 init: %v3384) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3388 = stablehlo.broadcast_in_dim %v3387, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3389 = stablehlo.divide %v3388, %v3385 : tensor<32x301056xf32>
    %v3390 = stablehlo.subtract %v9, %v3389 : tensor<32x301056xf32>
    %v3391 = stablehlo.multiply %v3390, %v3390 : tensor<32x301056xf32>
    %v3392 = stablehlo.reduce(%v3391 init: %v3384) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3393 = stablehlo.broadcast_in_dim %v3392, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3394 = stablehlo.divide %v3393, %v3385 : tensor<32x301056xf32>
    %v3395 = stablehlo.add %v3394, %v3386 : tensor<32x301056xf32>
    %v3396 = stablehlo.rsqrt %v3395 : tensor<32x301056xf32>
    %v3397 = stablehlo.multiply %v3390, %v3396 : tensor<32x301056xf32>
    %v3398 = stablehlo.multiply %v3328, %v3397 : tensor<32x301056xf32>
    %v3399 = stablehlo.reduce(%v3398 init: %v3384) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3401 = stablehlo.reduce(%v3328 init: %v3400) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3402 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3403 = stablehlo.reshape %v3355 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3404 = stablehlo.transpose %v3402, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3405 = stablehlo.transpose %v3403, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3406 = stablehlo.convolution(%v3404, %v3405)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v3407 = stablehlo.reshape %v3406 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v3408 = stablehlo.reshape %v3355 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3410 = stablehlo.reduce(%v3408 init: %v3409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3414 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v3415 = stablehlo.reshape %v3360 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3417 = stablehlo.pad %v3415, %v3416, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x221x221xf32>
    %v3418 = stablehlo.transpose %v3414, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v3419 = stablehlo.transpose %v3417, dims = [1, 0, 2, 3] : (tensor<32x96x221x221xf32>) -> tensor<96x32x221x221xf32>
    %v3420 = stablehlo.convolution(%v3418, %v3419)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<96x32x221x221xf32>) -> tensor<3x96x4x4xf32>
    %v3421 = stablehlo.transpose %v3420, dims = [1, 0, 2, 3] : (tensor<3x96x4x4xf32>) -> tensor<96x3x4x4xf32>
    %v3411 = stablehlo.reshape %v3360 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3413 = stablehlo.reduce(%v3411 init: %v3412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %arsumpsW = "stablehlo.all_reduce"(%v3421) ({
    ^bb0(%arapsW: tensor<f32>, %arbpsW: tensor<f32>):
      %araddpsW = stablehlo.add %arapsW, %arbpsW : tensor<f32>
      stablehlo.return %araddpsW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96x3x4x4xf32>) -> tensor<96x3x4x4xf32>
    %arnpsW = stablehlo.constant dense<2.0> : tensor<96x3x4x4xf32>
    %armeanpsW = stablehlo.divide %arsumpsW, %arnpsW : tensor<96x3x4x4xf32>
    %v3422 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3423 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3424 = stablehlo.multiply %v3422, %psWm : tensor<96x3x4x4xf32>
    %v3425 = stablehlo.multiply %v3423, %armeanpsW : tensor<96x3x4x4xf32>
    %v3426 = stablehlo.add %v3424, %v3425 : tensor<96x3x4x4xf32>
    %v3427 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3428 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3429 = stablehlo.multiply %v3427, %psWv : tensor<96x3x4x4xf32>
    %v3430 = stablehlo.multiply %armeanpsW, %armeanpsW : tensor<96x3x4x4xf32>
    %v3431 = stablehlo.multiply %v3428, %v3430 : tensor<96x3x4x4xf32>
    %v3432 = stablehlo.add %v3429, %v3431 : tensor<96x3x4x4xf32>
    %v3433 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3434 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3435 = stablehlo.multiply %v3433, %psWm : tensor<96x3x4x4xf32>
    %v3436 = stablehlo.multiply %v3434, %armeanpsW : tensor<96x3x4x4xf32>
    %v3437 = stablehlo.add %v3435, %v3436 : tensor<96x3x4x4xf32>
    %v3438 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3439 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3440 = stablehlo.multiply %v3438, %psWv : tensor<96x3x4x4xf32>
    %v3441 = stablehlo.multiply %armeanpsW, %armeanpsW : tensor<96x3x4x4xf32>
    %v3442 = stablehlo.multiply %v3439, %v3441 : tensor<96x3x4x4xf32>
    %v3443 = stablehlo.add %v3440, %v3442 : tensor<96x3x4x4xf32>
    %v3444 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3445 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3446 = stablehlo.divide %v3437, %v3444 : tensor<96x3x4x4xf32>
    %v3447 = stablehlo.divide %v3443, %v3445 : tensor<96x3x4x4xf32>
    %v3448 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3449 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3450 = stablehlo.sqrt %v3447 : tensor<96x3x4x4xf32>
    %v3451 = stablehlo.add %v3450, %v3449 : tensor<96x3x4x4xf32>
    %v3452 = stablehlo.divide %v3446, %v3451 : tensor<96x3x4x4xf32>
    %v3453 = stablehlo.multiply %v3448, %v3452 : tensor<96x3x4x4xf32>
    %v3454 = stablehlo.subtract %psW, %v3453 : tensor<96x3x4x4xf32>
    %v3455 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3456 = stablehlo.multiply %v3455, %v3448 : tensor<96x3x4x4xf32>
    %v3457 = stablehlo.multiply %v3456, %psW : tensor<96x3x4x4xf32>
    %v3458 = stablehlo.subtract %v3454, %v3457 : tensor<96x3x4x4xf32>
    %arsumpsb = "stablehlo.all_reduce"(%v3413) ({
    ^bb0(%arapsb: tensor<f32>, %arbpsb: tensor<f32>):
      %araddpsb = stablehlo.add %arapsb, %arbpsb : tensor<f32>
      stablehlo.return %araddpsb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96xf32>) -> tensor<96xf32>
    %arnpsb = stablehlo.constant dense<2.0> : tensor<96xf32>
    %armeanpsb = stablehlo.divide %arsumpsb, %arnpsb : tensor<96xf32>
    %v3459 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3460 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3461 = stablehlo.multiply %v3459, %psbm : tensor<96xf32>
    %v3462 = stablehlo.multiply %v3460, %armeanpsb : tensor<96xf32>
    %v3463 = stablehlo.add %v3461, %v3462 : tensor<96xf32>
    %v3464 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3465 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3466 = stablehlo.multiply %v3464, %psbv : tensor<96xf32>
    %v3467 = stablehlo.multiply %armeanpsb, %armeanpsb : tensor<96xf32>
    %v3468 = stablehlo.multiply %v3465, %v3467 : tensor<96xf32>
    %v3469 = stablehlo.add %v3466, %v3468 : tensor<96xf32>
    %v3470 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3471 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3472 = stablehlo.multiply %v3470, %psbm : tensor<96xf32>
    %v3473 = stablehlo.multiply %v3471, %armeanpsb : tensor<96xf32>
    %v3474 = stablehlo.add %v3472, %v3473 : tensor<96xf32>
    %v3475 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3476 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3477 = stablehlo.multiply %v3475, %psbv : tensor<96xf32>
    %v3478 = stablehlo.multiply %armeanpsb, %armeanpsb : tensor<96xf32>
    %v3479 = stablehlo.multiply %v3476, %v3478 : tensor<96xf32>
    %v3480 = stablehlo.add %v3477, %v3479 : tensor<96xf32>
    %v3481 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3482 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3483 = stablehlo.divide %v3474, %v3481 : tensor<96xf32>
    %v3484 = stablehlo.divide %v3480, %v3482 : tensor<96xf32>
    %v3485 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3486 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3487 = stablehlo.sqrt %v3484 : tensor<96xf32>
    %v3488 = stablehlo.add %v3487, %v3486 : tensor<96xf32>
    %v3489 = stablehlo.divide %v3483, %v3488 : tensor<96xf32>
    %v3490 = stablehlo.multiply %v3485, %v3489 : tensor<96xf32>
    %v3491 = stablehlo.subtract %psb, %v3490 : tensor<96xf32>
    %v3492 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3493 = stablehlo.multiply %v3492, %v3485 : tensor<96xf32>
    %v3494 = stablehlo.multiply %v3493, %psb : tensor<96xf32>
    %v3495 = stablehlo.subtract %v3491, %v3494 : tensor<96xf32>
    %arsums0b0dW = "stablehlo.all_reduce"(%v3407) ({
    ^bb0(%aras0b0dW: tensor<f32>, %arbs0b0dW: tensor<f32>):
      %aradds0b0dW = stablehlo.add %aras0b0dW, %arbs0b0dW : tensor<f32>
      stablehlo.return %aradds0b0dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96x1x7x7xf32>) -> tensor<96x1x7x7xf32>
    %arns0b0dW = stablehlo.constant dense<2.0> : tensor<96x1x7x7xf32>
    %armeans0b0dW = stablehlo.divide %arsums0b0dW, %arns0b0dW : tensor<96x1x7x7xf32>
    %v3496 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3497 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3498 = stablehlo.multiply %v3496, %s0b0dWm : tensor<96x1x7x7xf32>
    %v3499 = stablehlo.multiply %v3497, %armeans0b0dW : tensor<96x1x7x7xf32>
    %v3500 = stablehlo.add %v3498, %v3499 : tensor<96x1x7x7xf32>
    %v3501 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3502 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3503 = stablehlo.multiply %v3501, %s0b0dWv : tensor<96x1x7x7xf32>
    %v3504 = stablehlo.multiply %armeans0b0dW, %armeans0b0dW : tensor<96x1x7x7xf32>
    %v3505 = stablehlo.multiply %v3502, %v3504 : tensor<96x1x7x7xf32>
    %v3506 = stablehlo.add %v3503, %v3505 : tensor<96x1x7x7xf32>
    %v3507 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3508 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3509 = stablehlo.multiply %v3507, %s0b0dWm : tensor<96x1x7x7xf32>
    %v3510 = stablehlo.multiply %v3508, %armeans0b0dW : tensor<96x1x7x7xf32>
    %v3511 = stablehlo.add %v3509, %v3510 : tensor<96x1x7x7xf32>
    %v3512 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3513 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3514 = stablehlo.multiply %v3512, %s0b0dWv : tensor<96x1x7x7xf32>
    %v3515 = stablehlo.multiply %armeans0b0dW, %armeans0b0dW : tensor<96x1x7x7xf32>
    %v3516 = stablehlo.multiply %v3513, %v3515 : tensor<96x1x7x7xf32>
    %v3517 = stablehlo.add %v3514, %v3516 : tensor<96x1x7x7xf32>
    %v3518 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3519 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3520 = stablehlo.divide %v3511, %v3518 : tensor<96x1x7x7xf32>
    %v3521 = stablehlo.divide %v3517, %v3519 : tensor<96x1x7x7xf32>
    %v3522 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3523 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3524 = stablehlo.sqrt %v3521 : tensor<96x1x7x7xf32>
    %v3525 = stablehlo.add %v3524, %v3523 : tensor<96x1x7x7xf32>
    %v3526 = stablehlo.divide %v3520, %v3525 : tensor<96x1x7x7xf32>
    %v3527 = stablehlo.multiply %v3522, %v3526 : tensor<96x1x7x7xf32>
    %v3528 = stablehlo.subtract %s0b0dW, %v3527 : tensor<96x1x7x7xf32>
    %v3529 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3530 = stablehlo.multiply %v3529, %v3522 : tensor<96x1x7x7xf32>
    %v3531 = stablehlo.multiply %v3530, %s0b0dW : tensor<96x1x7x7xf32>
    %v3532 = stablehlo.subtract %v3528, %v3531 : tensor<96x1x7x7xf32>
    %arsums0b0db = "stablehlo.all_reduce"(%v3410) ({
    ^bb0(%aras0b0db: tensor<f32>, %arbs0b0db: tensor<f32>):
      %aradds0b0db = stablehlo.add %aras0b0db, %arbs0b0db : tensor<f32>
      stablehlo.return %aradds0b0db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96xf32>) -> tensor<96xf32>
    %arns0b0db = stablehlo.constant dense<2.0> : tensor<96xf32>
    %armeans0b0db = stablehlo.divide %arsums0b0db, %arns0b0db : tensor<96xf32>
    %v3533 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3534 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3535 = stablehlo.multiply %v3533, %s0b0dbm : tensor<96xf32>
    %v3536 = stablehlo.multiply %v3534, %armeans0b0db : tensor<96xf32>
    %v3537 = stablehlo.add %v3535, %v3536 : tensor<96xf32>
    %v3538 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3539 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3540 = stablehlo.multiply %v3538, %s0b0dbv : tensor<96xf32>
    %v3541 = stablehlo.multiply %armeans0b0db, %armeans0b0db : tensor<96xf32>
    %v3542 = stablehlo.multiply %v3539, %v3541 : tensor<96xf32>
    %v3543 = stablehlo.add %v3540, %v3542 : tensor<96xf32>
    %v3544 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3545 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3546 = stablehlo.multiply %v3544, %s0b0dbm : tensor<96xf32>
    %v3547 = stablehlo.multiply %v3545, %armeans0b0db : tensor<96xf32>
    %v3548 = stablehlo.add %v3546, %v3547 : tensor<96xf32>
    %v3549 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3550 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3551 = stablehlo.multiply %v3549, %s0b0dbv : tensor<96xf32>
    %v3552 = stablehlo.multiply %armeans0b0db, %armeans0b0db : tensor<96xf32>
    %v3553 = stablehlo.multiply %v3550, %v3552 : tensor<96xf32>
    %v3554 = stablehlo.add %v3551, %v3553 : tensor<96xf32>
    %v3555 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3556 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3557 = stablehlo.divide %v3548, %v3555 : tensor<96xf32>
    %v3558 = stablehlo.divide %v3554, %v3556 : tensor<96xf32>
    %v3559 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3560 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3561 = stablehlo.sqrt %v3558 : tensor<96xf32>
    %v3562 = stablehlo.add %v3561, %v3560 : tensor<96xf32>
    %v3563 = stablehlo.divide %v3557, %v3562 : tensor<96xf32>
    %v3564 = stablehlo.multiply %v3559, %v3563 : tensor<96xf32>
    %v3565 = stablehlo.subtract %s0b0db, %v3564 : tensor<96xf32>
    %v3566 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3567 = stablehlo.multiply %v3566, %v3559 : tensor<96xf32>
    %v3568 = stablehlo.multiply %v3567, %s0b0db : tensor<96xf32>
    %v3569 = stablehlo.subtract %v3565, %v3568 : tensor<96xf32>
    %arsums0b0ng = "stablehlo.all_reduce"(%v3399) ({
    ^bb0(%aras0b0ng: tensor<f32>, %arbs0b0ng: tensor<f32>):
      %aradds0b0ng = stablehlo.add %aras0b0ng, %arbs0b0ng : tensor<f32>
      stablehlo.return %aradds0b0ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns0b0ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans0b0ng = stablehlo.divide %arsums0b0ng, %arns0b0ng : tensor<f32>
    %v3570 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3571 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3572 = stablehlo.multiply %v3570, %s0b0ngm : tensor<f32>
    %v3573 = stablehlo.multiply %v3571, %armeans0b0ng : tensor<f32>
    %v3574 = stablehlo.add %v3572, %v3573 : tensor<f32>
    %v3575 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3576 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3577 = stablehlo.multiply %v3575, %s0b0ngv : tensor<f32>
    %v3578 = stablehlo.multiply %armeans0b0ng, %armeans0b0ng : tensor<f32>
    %v3579 = stablehlo.multiply %v3576, %v3578 : tensor<f32>
    %v3580 = stablehlo.add %v3577, %v3579 : tensor<f32>
    %v3581 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3582 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3583 = stablehlo.multiply %v3581, %s0b0ngm : tensor<f32>
    %v3584 = stablehlo.multiply %v3582, %armeans0b0ng : tensor<f32>
    %v3585 = stablehlo.add %v3583, %v3584 : tensor<f32>
    %v3586 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3587 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3588 = stablehlo.multiply %v3586, %s0b0ngv : tensor<f32>
    %v3589 = stablehlo.multiply %armeans0b0ng, %armeans0b0ng : tensor<f32>
    %v3590 = stablehlo.multiply %v3587, %v3589 : tensor<f32>
    %v3591 = stablehlo.add %v3588, %v3590 : tensor<f32>
    %v3592 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3593 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3594 = stablehlo.divide %v3585, %v3592 : tensor<f32>
    %v3595 = stablehlo.divide %v3591, %v3593 : tensor<f32>
    %v3596 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3597 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3598 = stablehlo.sqrt %v3595 : tensor<f32>
    %v3599 = stablehlo.add %v3598, %v3597 : tensor<f32>
    %v3600 = stablehlo.divide %v3594, %v3599 : tensor<f32>
    %v3601 = stablehlo.multiply %v3596, %v3600 : tensor<f32>
    %v3602 = stablehlo.subtract %s0b0ng, %v3601 : tensor<f32>
    %v3603 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3604 = stablehlo.multiply %v3603, %v3596 : tensor<f32>
    %v3605 = stablehlo.multiply %v3604, %s0b0ng : tensor<f32>
    %v3606 = stablehlo.subtract %v3602, %v3605 : tensor<f32>
    %arsums0b0nbt = "stablehlo.all_reduce"(%v3401) ({
    ^bb0(%aras0b0nbt: tensor<f32>, %arbs0b0nbt: tensor<f32>):
      %aradds0b0nbt = stablehlo.add %aras0b0nbt, %arbs0b0nbt : tensor<f32>
      stablehlo.return %aradds0b0nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns0b0nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans0b0nbt = stablehlo.divide %arsums0b0nbt, %arns0b0nbt : tensor<f32>
    %v3607 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3608 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3609 = stablehlo.multiply %v3607, %s0b0nbtm : tensor<f32>
    %v3610 = stablehlo.multiply %v3608, %armeans0b0nbt : tensor<f32>
    %v3611 = stablehlo.add %v3609, %v3610 : tensor<f32>
    %v3612 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3613 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3614 = stablehlo.multiply %v3612, %s0b0nbtv : tensor<f32>
    %v3615 = stablehlo.multiply %armeans0b0nbt, %armeans0b0nbt : tensor<f32>
    %v3616 = stablehlo.multiply %v3613, %v3615 : tensor<f32>
    %v3617 = stablehlo.add %v3614, %v3616 : tensor<f32>
    %v3618 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3619 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3620 = stablehlo.multiply %v3618, %s0b0nbtm : tensor<f32>
    %v3621 = stablehlo.multiply %v3619, %armeans0b0nbt : tensor<f32>
    %v3622 = stablehlo.add %v3620, %v3621 : tensor<f32>
    %v3623 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3624 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3625 = stablehlo.multiply %v3623, %s0b0nbtv : tensor<f32>
    %v3626 = stablehlo.multiply %armeans0b0nbt, %armeans0b0nbt : tensor<f32>
    %v3627 = stablehlo.multiply %v3624, %v3626 : tensor<f32>
    %v3628 = stablehlo.add %v3625, %v3627 : tensor<f32>
    %v3629 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3630 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3631 = stablehlo.divide %v3622, %v3629 : tensor<f32>
    %v3632 = stablehlo.divide %v3628, %v3630 : tensor<f32>
    %v3633 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3634 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3635 = stablehlo.sqrt %v3632 : tensor<f32>
    %v3636 = stablehlo.add %v3635, %v3634 : tensor<f32>
    %v3637 = stablehlo.divide %v3631, %v3636 : tensor<f32>
    %v3638 = stablehlo.multiply %v3633, %v3637 : tensor<f32>
    %v3639 = stablehlo.subtract %s0b0nbt, %v3638 : tensor<f32>
    %v3640 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3641 = stablehlo.multiply %v3640, %v3633 : tensor<f32>
    %v3642 = stablehlo.multiply %v3641, %s0b0nbt : tensor<f32>
    %v3643 = stablehlo.subtract %v3639, %v3642 : tensor<f32>
    %arsums0b0eW = "stablehlo.all_reduce"(%v3380) ({
    ^bb0(%aras0b0eW: tensor<f32>, %arbs0b0eW: tensor<f32>):
      %aradds0b0eW = stablehlo.add %aras0b0eW, %arbs0b0eW : tensor<f32>
      stablehlo.return %aradds0b0eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x96x1x1xf32>) -> tensor<384x96x1x1xf32>
    %arns0b0eW = stablehlo.constant dense<2.0> : tensor<384x96x1x1xf32>
    %armeans0b0eW = stablehlo.divide %arsums0b0eW, %arns0b0eW : tensor<384x96x1x1xf32>
    %v3644 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3645 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3646 = stablehlo.multiply %v3644, %s0b0eWm : tensor<384x96x1x1xf32>
    %v3647 = stablehlo.multiply %v3645, %armeans0b0eW : tensor<384x96x1x1xf32>
    %v3648 = stablehlo.add %v3646, %v3647 : tensor<384x96x1x1xf32>
    %v3649 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3650 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3651 = stablehlo.multiply %v3649, %s0b0eWv : tensor<384x96x1x1xf32>
    %v3652 = stablehlo.multiply %armeans0b0eW, %armeans0b0eW : tensor<384x96x1x1xf32>
    %v3653 = stablehlo.multiply %v3650, %v3652 : tensor<384x96x1x1xf32>
    %v3654 = stablehlo.add %v3651, %v3653 : tensor<384x96x1x1xf32>
    %v3655 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3656 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3657 = stablehlo.multiply %v3655, %s0b0eWm : tensor<384x96x1x1xf32>
    %v3658 = stablehlo.multiply %v3656, %armeans0b0eW : tensor<384x96x1x1xf32>
    %v3659 = stablehlo.add %v3657, %v3658 : tensor<384x96x1x1xf32>
    %v3660 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3661 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3662 = stablehlo.multiply %v3660, %s0b0eWv : tensor<384x96x1x1xf32>
    %v3663 = stablehlo.multiply %armeans0b0eW, %armeans0b0eW : tensor<384x96x1x1xf32>
    %v3664 = stablehlo.multiply %v3661, %v3663 : tensor<384x96x1x1xf32>
    %v3665 = stablehlo.add %v3662, %v3664 : tensor<384x96x1x1xf32>
    %v3666 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3667 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3668 = stablehlo.divide %v3659, %v3666 : tensor<384x96x1x1xf32>
    %v3669 = stablehlo.divide %v3665, %v3667 : tensor<384x96x1x1xf32>
    %v3670 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3671 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3672 = stablehlo.sqrt %v3669 : tensor<384x96x1x1xf32>
    %v3673 = stablehlo.add %v3672, %v3671 : tensor<384x96x1x1xf32>
    %v3674 = stablehlo.divide %v3668, %v3673 : tensor<384x96x1x1xf32>
    %v3675 = stablehlo.multiply %v3670, %v3674 : tensor<384x96x1x1xf32>
    %v3676 = stablehlo.subtract %s0b0eW, %v3675 : tensor<384x96x1x1xf32>
    %v3677 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3678 = stablehlo.multiply %v3677, %v3670 : tensor<384x96x1x1xf32>
    %v3679 = stablehlo.multiply %v3678, %s0b0eW : tensor<384x96x1x1xf32>
    %v3680 = stablehlo.subtract %v3676, %v3679 : tensor<384x96x1x1xf32>
    %arsums0b0eb = "stablehlo.all_reduce"(%v3383) ({
    ^bb0(%aras0b0eb: tensor<f32>, %arbs0b0eb: tensor<f32>):
      %aradds0b0eb = stablehlo.add %aras0b0eb, %arbs0b0eb : tensor<f32>
      stablehlo.return %aradds0b0eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns0b0eb = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans0b0eb = stablehlo.divide %arsums0b0eb, %arns0b0eb : tensor<384xf32>
    %v3681 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3682 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3683 = stablehlo.multiply %v3681, %s0b0ebm : tensor<384xf32>
    %v3684 = stablehlo.multiply %v3682, %armeans0b0eb : tensor<384xf32>
    %v3685 = stablehlo.add %v3683, %v3684 : tensor<384xf32>
    %v3686 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3687 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3688 = stablehlo.multiply %v3686, %s0b0ebv : tensor<384xf32>
    %v3689 = stablehlo.multiply %armeans0b0eb, %armeans0b0eb : tensor<384xf32>
    %v3690 = stablehlo.multiply %v3687, %v3689 : tensor<384xf32>
    %v3691 = stablehlo.add %v3688, %v3690 : tensor<384xf32>
    %v3692 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3693 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3694 = stablehlo.multiply %v3692, %s0b0ebm : tensor<384xf32>
    %v3695 = stablehlo.multiply %v3693, %armeans0b0eb : tensor<384xf32>
    %v3696 = stablehlo.add %v3694, %v3695 : tensor<384xf32>
    %v3697 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3698 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3699 = stablehlo.multiply %v3697, %s0b0ebv : tensor<384xf32>
    %v3700 = stablehlo.multiply %armeans0b0eb, %armeans0b0eb : tensor<384xf32>
    %v3701 = stablehlo.multiply %v3698, %v3700 : tensor<384xf32>
    %v3702 = stablehlo.add %v3699, %v3701 : tensor<384xf32>
    %v3703 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3704 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3705 = stablehlo.divide %v3696, %v3703 : tensor<384xf32>
    %v3706 = stablehlo.divide %v3702, %v3704 : tensor<384xf32>
    %v3707 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3708 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3709 = stablehlo.sqrt %v3706 : tensor<384xf32>
    %v3710 = stablehlo.add %v3709, %v3708 : tensor<384xf32>
    %v3711 = stablehlo.divide %v3705, %v3710 : tensor<384xf32>
    %v3712 = stablehlo.multiply %v3707, %v3711 : tensor<384xf32>
    %v3713 = stablehlo.subtract %s0b0eb, %v3712 : tensor<384xf32>
    %v3714 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3715 = stablehlo.multiply %v3714, %v3707 : tensor<384xf32>
    %v3716 = stablehlo.multiply %v3715, %s0b0eb : tensor<384xf32>
    %v3717 = stablehlo.subtract %v3713, %v3716 : tensor<384xf32>
    %arsums0b0pW = "stablehlo.all_reduce"(%v3371) ({
    ^bb0(%aras0b0pW: tensor<f32>, %arbs0b0pW: tensor<f32>):
      %aradds0b0pW = stablehlo.add %aras0b0pW, %arbs0b0pW : tensor<f32>
      stablehlo.return %aradds0b0pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96x384x1x1xf32>) -> tensor<96x384x1x1xf32>
    %arns0b0pW = stablehlo.constant dense<2.0> : tensor<96x384x1x1xf32>
    %armeans0b0pW = stablehlo.divide %arsums0b0pW, %arns0b0pW : tensor<96x384x1x1xf32>
    %v3718 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3719 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3720 = stablehlo.multiply %v3718, %s0b0pWm : tensor<96x384x1x1xf32>
    %v3721 = stablehlo.multiply %v3719, %armeans0b0pW : tensor<96x384x1x1xf32>
    %v3722 = stablehlo.add %v3720, %v3721 : tensor<96x384x1x1xf32>
    %v3723 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3724 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3725 = stablehlo.multiply %v3723, %s0b0pWv : tensor<96x384x1x1xf32>
    %v3726 = stablehlo.multiply %armeans0b0pW, %armeans0b0pW : tensor<96x384x1x1xf32>
    %v3727 = stablehlo.multiply %v3724, %v3726 : tensor<96x384x1x1xf32>
    %v3728 = stablehlo.add %v3725, %v3727 : tensor<96x384x1x1xf32>
    %v3729 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3730 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3731 = stablehlo.multiply %v3729, %s0b0pWm : tensor<96x384x1x1xf32>
    %v3732 = stablehlo.multiply %v3730, %armeans0b0pW : tensor<96x384x1x1xf32>
    %v3733 = stablehlo.add %v3731, %v3732 : tensor<96x384x1x1xf32>
    %v3734 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3735 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3736 = stablehlo.multiply %v3734, %s0b0pWv : tensor<96x384x1x1xf32>
    %v3737 = stablehlo.multiply %armeans0b0pW, %armeans0b0pW : tensor<96x384x1x1xf32>
    %v3738 = stablehlo.multiply %v3735, %v3737 : tensor<96x384x1x1xf32>
    %v3739 = stablehlo.add %v3736, %v3738 : tensor<96x384x1x1xf32>
    %v3740 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3741 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3742 = stablehlo.divide %v3733, %v3740 : tensor<96x384x1x1xf32>
    %v3743 = stablehlo.divide %v3739, %v3741 : tensor<96x384x1x1xf32>
    %v3744 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3745 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3746 = stablehlo.sqrt %v3743 : tensor<96x384x1x1xf32>
    %v3747 = stablehlo.add %v3746, %v3745 : tensor<96x384x1x1xf32>
    %v3748 = stablehlo.divide %v3742, %v3747 : tensor<96x384x1x1xf32>
    %v3749 = stablehlo.multiply %v3744, %v3748 : tensor<96x384x1x1xf32>
    %v3750 = stablehlo.subtract %s0b0pW, %v3749 : tensor<96x384x1x1xf32>
    %v3751 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3752 = stablehlo.multiply %v3751, %v3744 : tensor<96x384x1x1xf32>
    %v3753 = stablehlo.multiply %v3752, %s0b0pW : tensor<96x384x1x1xf32>
    %v3754 = stablehlo.subtract %v3750, %v3753 : tensor<96x384x1x1xf32>
    %arsums0b0pb = "stablehlo.all_reduce"(%v3374) ({
    ^bb0(%aras0b0pb: tensor<f32>, %arbs0b0pb: tensor<f32>):
      %aradds0b0pb = stablehlo.add %aras0b0pb, %arbs0b0pb : tensor<f32>
      stablehlo.return %aradds0b0pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96xf32>) -> tensor<96xf32>
    %arns0b0pb = stablehlo.constant dense<2.0> : tensor<96xf32>
    %armeans0b0pb = stablehlo.divide %arsums0b0pb, %arns0b0pb : tensor<96xf32>
    %v3755 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3756 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3757 = stablehlo.multiply %v3755, %s0b0pbm : tensor<96xf32>
    %v3758 = stablehlo.multiply %v3756, %armeans0b0pb : tensor<96xf32>
    %v3759 = stablehlo.add %v3757, %v3758 : tensor<96xf32>
    %v3760 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3761 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3762 = stablehlo.multiply %v3760, %s0b0pbv : tensor<96xf32>
    %v3763 = stablehlo.multiply %armeans0b0pb, %armeans0b0pb : tensor<96xf32>
    %v3764 = stablehlo.multiply %v3761, %v3763 : tensor<96xf32>
    %v3765 = stablehlo.add %v3762, %v3764 : tensor<96xf32>
    %v3766 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3767 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3768 = stablehlo.multiply %v3766, %s0b0pbm : tensor<96xf32>
    %v3769 = stablehlo.multiply %v3767, %armeans0b0pb : tensor<96xf32>
    %v3770 = stablehlo.add %v3768, %v3769 : tensor<96xf32>
    %v3771 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3772 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3773 = stablehlo.multiply %v3771, %s0b0pbv : tensor<96xf32>
    %v3774 = stablehlo.multiply %armeans0b0pb, %armeans0b0pb : tensor<96xf32>
    %v3775 = stablehlo.multiply %v3772, %v3774 : tensor<96xf32>
    %v3776 = stablehlo.add %v3773, %v3775 : tensor<96xf32>
    %v3777 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3778 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3779 = stablehlo.divide %v3770, %v3777 : tensor<96xf32>
    %v3780 = stablehlo.divide %v3776, %v3778 : tensor<96xf32>
    %v3781 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3782 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3783 = stablehlo.sqrt %v3780 : tensor<96xf32>
    %v3784 = stablehlo.add %v3783, %v3782 : tensor<96xf32>
    %v3785 = stablehlo.divide %v3779, %v3784 : tensor<96xf32>
    %v3786 = stablehlo.multiply %v3781, %v3785 : tensor<96xf32>
    %v3787 = stablehlo.subtract %s0b0pb, %v3786 : tensor<96xf32>
    %v3788 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3789 = stablehlo.multiply %v3788, %v3781 : tensor<96xf32>
    %v3790 = stablehlo.multiply %v3789, %s0b0pb : tensor<96xf32>
    %v3791 = stablehlo.subtract %v3787, %v3790 : tensor<96xf32>
    %arsums0b0lg = "stablehlo.all_reduce"(%v3365) ({
    ^bb0(%aras0b0lg: tensor<f32>, %arbs0b0lg: tensor<f32>):
      %aradds0b0lg = stablehlo.add %aras0b0lg, %arbs0b0lg : tensor<f32>
      stablehlo.return %aradds0b0lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96xf32>) -> tensor<96xf32>
    %arns0b0lg = stablehlo.constant dense<2.0> : tensor<96xf32>
    %armeans0b0lg = stablehlo.divide %arsums0b0lg, %arns0b0lg : tensor<96xf32>
    %v3792 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3793 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3794 = stablehlo.multiply %v3792, %s0b0lgm : tensor<96xf32>
    %v3795 = stablehlo.multiply %v3793, %armeans0b0lg : tensor<96xf32>
    %v3796 = stablehlo.add %v3794, %v3795 : tensor<96xf32>
    %v3797 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3798 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3799 = stablehlo.multiply %v3797, %s0b0lgv : tensor<96xf32>
    %v3800 = stablehlo.multiply %armeans0b0lg, %armeans0b0lg : tensor<96xf32>
    %v3801 = stablehlo.multiply %v3798, %v3800 : tensor<96xf32>
    %v3802 = stablehlo.add %v3799, %v3801 : tensor<96xf32>
    %v3803 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3804 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3805 = stablehlo.multiply %v3803, %s0b0lgm : tensor<96xf32>
    %v3806 = stablehlo.multiply %v3804, %armeans0b0lg : tensor<96xf32>
    %v3807 = stablehlo.add %v3805, %v3806 : tensor<96xf32>
    %v3808 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3809 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3810 = stablehlo.multiply %v3808, %s0b0lgv : tensor<96xf32>
    %v3811 = stablehlo.multiply %armeans0b0lg, %armeans0b0lg : tensor<96xf32>
    %v3812 = stablehlo.multiply %v3809, %v3811 : tensor<96xf32>
    %v3813 = stablehlo.add %v3810, %v3812 : tensor<96xf32>
    %v3814 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3815 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3816 = stablehlo.divide %v3807, %v3814 : tensor<96xf32>
    %v3817 = stablehlo.divide %v3813, %v3815 : tensor<96xf32>
    %v3818 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3819 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3820 = stablehlo.sqrt %v3817 : tensor<96xf32>
    %v3821 = stablehlo.add %v3820, %v3819 : tensor<96xf32>
    %v3822 = stablehlo.divide %v3816, %v3821 : tensor<96xf32>
    %v3823 = stablehlo.multiply %v3818, %v3822 : tensor<96xf32>
    %v3824 = stablehlo.subtract %s0b0lg, %v3823 : tensor<96xf32>
    %v3825 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3826 = stablehlo.multiply %v3825, %v3818 : tensor<96xf32>
    %v3827 = stablehlo.multiply %v3826, %s0b0lg : tensor<96xf32>
    %v3828 = stablehlo.subtract %v3824, %v3827 : tensor<96xf32>
    %arsums0b1dW = "stablehlo.all_reduce"(%v3288) ({
    ^bb0(%aras0b1dW: tensor<f32>, %arbs0b1dW: tensor<f32>):
      %aradds0b1dW = stablehlo.add %aras0b1dW, %arbs0b1dW : tensor<f32>
      stablehlo.return %aradds0b1dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96x1x7x7xf32>) -> tensor<96x1x7x7xf32>
    %arns0b1dW = stablehlo.constant dense<2.0> : tensor<96x1x7x7xf32>
    %armeans0b1dW = stablehlo.divide %arsums0b1dW, %arns0b1dW : tensor<96x1x7x7xf32>
    %v3829 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3830 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3831 = stablehlo.multiply %v3829, %s0b1dWm : tensor<96x1x7x7xf32>
    %v3832 = stablehlo.multiply %v3830, %armeans0b1dW : tensor<96x1x7x7xf32>
    %v3833 = stablehlo.add %v3831, %v3832 : tensor<96x1x7x7xf32>
    %v3834 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3835 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3836 = stablehlo.multiply %v3834, %s0b1dWv : tensor<96x1x7x7xf32>
    %v3837 = stablehlo.multiply %armeans0b1dW, %armeans0b1dW : tensor<96x1x7x7xf32>
    %v3838 = stablehlo.multiply %v3835, %v3837 : tensor<96x1x7x7xf32>
    %v3839 = stablehlo.add %v3836, %v3838 : tensor<96x1x7x7xf32>
    %v3840 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3841 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3842 = stablehlo.multiply %v3840, %s0b1dWm : tensor<96x1x7x7xf32>
    %v3843 = stablehlo.multiply %v3841, %armeans0b1dW : tensor<96x1x7x7xf32>
    %v3844 = stablehlo.add %v3842, %v3843 : tensor<96x1x7x7xf32>
    %v3845 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3846 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3847 = stablehlo.multiply %v3845, %s0b1dWv : tensor<96x1x7x7xf32>
    %v3848 = stablehlo.multiply %armeans0b1dW, %armeans0b1dW : tensor<96x1x7x7xf32>
    %v3849 = stablehlo.multiply %v3846, %v3848 : tensor<96x1x7x7xf32>
    %v3850 = stablehlo.add %v3847, %v3849 : tensor<96x1x7x7xf32>
    %v3851 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3852 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3853 = stablehlo.divide %v3844, %v3851 : tensor<96x1x7x7xf32>
    %v3854 = stablehlo.divide %v3850, %v3852 : tensor<96x1x7x7xf32>
    %v3855 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3856 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3857 = stablehlo.sqrt %v3854 : tensor<96x1x7x7xf32>
    %v3858 = stablehlo.add %v3857, %v3856 : tensor<96x1x7x7xf32>
    %v3859 = stablehlo.divide %v3853, %v3858 : tensor<96x1x7x7xf32>
    %v3860 = stablehlo.multiply %v3855, %v3859 : tensor<96x1x7x7xf32>
    %v3861 = stablehlo.subtract %s0b1dW, %v3860 : tensor<96x1x7x7xf32>
    %v3862 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3863 = stablehlo.multiply %v3862, %v3855 : tensor<96x1x7x7xf32>
    %v3864 = stablehlo.multiply %v3863, %s0b1dW : tensor<96x1x7x7xf32>
    %v3865 = stablehlo.subtract %v3861, %v3864 : tensor<96x1x7x7xf32>
    %arsums0b1db = "stablehlo.all_reduce"(%v3291) ({
    ^bb0(%aras0b1db: tensor<f32>, %arbs0b1db: tensor<f32>):
      %aradds0b1db = stablehlo.add %aras0b1db, %arbs0b1db : tensor<f32>
      stablehlo.return %aradds0b1db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96xf32>) -> tensor<96xf32>
    %arns0b1db = stablehlo.constant dense<2.0> : tensor<96xf32>
    %armeans0b1db = stablehlo.divide %arsums0b1db, %arns0b1db : tensor<96xf32>
    %v3866 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3867 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3868 = stablehlo.multiply %v3866, %s0b1dbm : tensor<96xf32>
    %v3869 = stablehlo.multiply %v3867, %armeans0b1db : tensor<96xf32>
    %v3870 = stablehlo.add %v3868, %v3869 : tensor<96xf32>
    %v3871 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3872 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3873 = stablehlo.multiply %v3871, %s0b1dbv : tensor<96xf32>
    %v3874 = stablehlo.multiply %armeans0b1db, %armeans0b1db : tensor<96xf32>
    %v3875 = stablehlo.multiply %v3872, %v3874 : tensor<96xf32>
    %v3876 = stablehlo.add %v3873, %v3875 : tensor<96xf32>
    %v3877 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3878 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3879 = stablehlo.multiply %v3877, %s0b1dbm : tensor<96xf32>
    %v3880 = stablehlo.multiply %v3878, %armeans0b1db : tensor<96xf32>
    %v3881 = stablehlo.add %v3879, %v3880 : tensor<96xf32>
    %v3882 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3883 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3884 = stablehlo.multiply %v3882, %s0b1dbv : tensor<96xf32>
    %v3885 = stablehlo.multiply %armeans0b1db, %armeans0b1db : tensor<96xf32>
    %v3886 = stablehlo.multiply %v3883, %v3885 : tensor<96xf32>
    %v3887 = stablehlo.add %v3884, %v3886 : tensor<96xf32>
    %v3888 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3889 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3890 = stablehlo.divide %v3881, %v3888 : tensor<96xf32>
    %v3891 = stablehlo.divide %v3887, %v3889 : tensor<96xf32>
    %v3892 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3893 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3894 = stablehlo.sqrt %v3891 : tensor<96xf32>
    %v3895 = stablehlo.add %v3894, %v3893 : tensor<96xf32>
    %v3896 = stablehlo.divide %v3890, %v3895 : tensor<96xf32>
    %v3897 = stablehlo.multiply %v3892, %v3896 : tensor<96xf32>
    %v3898 = stablehlo.subtract %s0b1db, %v3897 : tensor<96xf32>
    %v3899 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3900 = stablehlo.multiply %v3899, %v3892 : tensor<96xf32>
    %v3901 = stablehlo.multiply %v3900, %s0b1db : tensor<96xf32>
    %v3902 = stablehlo.subtract %v3898, %v3901 : tensor<96xf32>
    %arsums0b1ng = "stablehlo.all_reduce"(%v3280) ({
    ^bb0(%aras0b1ng: tensor<f32>, %arbs0b1ng: tensor<f32>):
      %aradds0b1ng = stablehlo.add %aras0b1ng, %arbs0b1ng : tensor<f32>
      stablehlo.return %aradds0b1ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns0b1ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans0b1ng = stablehlo.divide %arsums0b1ng, %arns0b1ng : tensor<f32>
    %v3903 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3904 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3905 = stablehlo.multiply %v3903, %s0b1ngm : tensor<f32>
    %v3906 = stablehlo.multiply %v3904, %armeans0b1ng : tensor<f32>
    %v3907 = stablehlo.add %v3905, %v3906 : tensor<f32>
    %v3908 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3909 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3910 = stablehlo.multiply %v3908, %s0b1ngv : tensor<f32>
    %v3911 = stablehlo.multiply %armeans0b1ng, %armeans0b1ng : tensor<f32>
    %v3912 = stablehlo.multiply %v3909, %v3911 : tensor<f32>
    %v3913 = stablehlo.add %v3910, %v3912 : tensor<f32>
    %v3914 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3915 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3916 = stablehlo.multiply %v3914, %s0b1ngm : tensor<f32>
    %v3917 = stablehlo.multiply %v3915, %armeans0b1ng : tensor<f32>
    %v3918 = stablehlo.add %v3916, %v3917 : tensor<f32>
    %v3919 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3920 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3921 = stablehlo.multiply %v3919, %s0b1ngv : tensor<f32>
    %v3922 = stablehlo.multiply %armeans0b1ng, %armeans0b1ng : tensor<f32>
    %v3923 = stablehlo.multiply %v3920, %v3922 : tensor<f32>
    %v3924 = stablehlo.add %v3921, %v3923 : tensor<f32>
    %v3925 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3926 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3927 = stablehlo.divide %v3918, %v3925 : tensor<f32>
    %v3928 = stablehlo.divide %v3924, %v3926 : tensor<f32>
    %v3929 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3930 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3931 = stablehlo.sqrt %v3928 : tensor<f32>
    %v3932 = stablehlo.add %v3931, %v3930 : tensor<f32>
    %v3933 = stablehlo.divide %v3927, %v3932 : tensor<f32>
    %v3934 = stablehlo.multiply %v3929, %v3933 : tensor<f32>
    %v3935 = stablehlo.subtract %s0b1ng, %v3934 : tensor<f32>
    %v3936 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3937 = stablehlo.multiply %v3936, %v3929 : tensor<f32>
    %v3938 = stablehlo.multiply %v3937, %s0b1ng : tensor<f32>
    %v3939 = stablehlo.subtract %v3935, %v3938 : tensor<f32>
    %arsums0b1nbt = "stablehlo.all_reduce"(%v3282) ({
    ^bb0(%aras0b1nbt: tensor<f32>, %arbs0b1nbt: tensor<f32>):
      %aradds0b1nbt = stablehlo.add %aras0b1nbt, %arbs0b1nbt : tensor<f32>
      stablehlo.return %aradds0b1nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns0b1nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans0b1nbt = stablehlo.divide %arsums0b1nbt, %arns0b1nbt : tensor<f32>
    %v3940 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3941 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3942 = stablehlo.multiply %v3940, %s0b1nbtm : tensor<f32>
    %v3943 = stablehlo.multiply %v3941, %armeans0b1nbt : tensor<f32>
    %v3944 = stablehlo.add %v3942, %v3943 : tensor<f32>
    %v3945 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3946 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3947 = stablehlo.multiply %v3945, %s0b1nbtv : tensor<f32>
    %v3948 = stablehlo.multiply %armeans0b1nbt, %armeans0b1nbt : tensor<f32>
    %v3949 = stablehlo.multiply %v3946, %v3948 : tensor<f32>
    %v3950 = stablehlo.add %v3947, %v3949 : tensor<f32>
    %v3951 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3952 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3953 = stablehlo.multiply %v3951, %s0b1nbtm : tensor<f32>
    %v3954 = stablehlo.multiply %v3952, %armeans0b1nbt : tensor<f32>
    %v3955 = stablehlo.add %v3953, %v3954 : tensor<f32>
    %v3956 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3957 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3958 = stablehlo.multiply %v3956, %s0b1nbtv : tensor<f32>
    %v3959 = stablehlo.multiply %armeans0b1nbt, %armeans0b1nbt : tensor<f32>
    %v3960 = stablehlo.multiply %v3957, %v3959 : tensor<f32>
    %v3961 = stablehlo.add %v3958, %v3960 : tensor<f32>
    %v3962 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3963 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3964 = stablehlo.divide %v3955, %v3962 : tensor<f32>
    %v3965 = stablehlo.divide %v3961, %v3963 : tensor<f32>
    %v3966 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3967 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3968 = stablehlo.sqrt %v3965 : tensor<f32>
    %v3969 = stablehlo.add %v3968, %v3967 : tensor<f32>
    %v3970 = stablehlo.divide %v3964, %v3969 : tensor<f32>
    %v3971 = stablehlo.multiply %v3966, %v3970 : tensor<f32>
    %v3972 = stablehlo.subtract %s0b1nbt, %v3971 : tensor<f32>
    %v3973 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3974 = stablehlo.multiply %v3973, %v3966 : tensor<f32>
    %v3975 = stablehlo.multiply %v3974, %s0b1nbt : tensor<f32>
    %v3976 = stablehlo.subtract %v3972, %v3975 : tensor<f32>
    %arsums0b1eW = "stablehlo.all_reduce"(%v3261) ({
    ^bb0(%aras0b1eW: tensor<f32>, %arbs0b1eW: tensor<f32>):
      %aradds0b1eW = stablehlo.add %aras0b1eW, %arbs0b1eW : tensor<f32>
      stablehlo.return %aradds0b1eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x96x1x1xf32>) -> tensor<384x96x1x1xf32>
    %arns0b1eW = stablehlo.constant dense<2.0> : tensor<384x96x1x1xf32>
    %armeans0b1eW = stablehlo.divide %arsums0b1eW, %arns0b1eW : tensor<384x96x1x1xf32>
    %v3977 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3978 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3979 = stablehlo.multiply %v3977, %s0b1eWm : tensor<384x96x1x1xf32>
    %v3980 = stablehlo.multiply %v3978, %armeans0b1eW : tensor<384x96x1x1xf32>
    %v3981 = stablehlo.add %v3979, %v3980 : tensor<384x96x1x1xf32>
    %v3982 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3983 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3984 = stablehlo.multiply %v3982, %s0b1eWv : tensor<384x96x1x1xf32>
    %v3985 = stablehlo.multiply %armeans0b1eW, %armeans0b1eW : tensor<384x96x1x1xf32>
    %v3986 = stablehlo.multiply %v3983, %v3985 : tensor<384x96x1x1xf32>
    %v3987 = stablehlo.add %v3984, %v3986 : tensor<384x96x1x1xf32>
    %v3988 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3989 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3990 = stablehlo.multiply %v3988, %s0b1eWm : tensor<384x96x1x1xf32>
    %v3991 = stablehlo.multiply %v3989, %armeans0b1eW : tensor<384x96x1x1xf32>
    %v3992 = stablehlo.add %v3990, %v3991 : tensor<384x96x1x1xf32>
    %v3993 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3994 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3995 = stablehlo.multiply %v3993, %s0b1eWv : tensor<384x96x1x1xf32>
    %v3996 = stablehlo.multiply %armeans0b1eW, %armeans0b1eW : tensor<384x96x1x1xf32>
    %v3997 = stablehlo.multiply %v3994, %v3996 : tensor<384x96x1x1xf32>
    %v3998 = stablehlo.add %v3995, %v3997 : tensor<384x96x1x1xf32>
    %v3999 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4000 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4001 = stablehlo.divide %v3992, %v3999 : tensor<384x96x1x1xf32>
    %v4002 = stablehlo.divide %v3998, %v4000 : tensor<384x96x1x1xf32>
    %v4003 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4004 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4005 = stablehlo.sqrt %v4002 : tensor<384x96x1x1xf32>
    %v4006 = stablehlo.add %v4005, %v4004 : tensor<384x96x1x1xf32>
    %v4007 = stablehlo.divide %v4001, %v4006 : tensor<384x96x1x1xf32>
    %v4008 = stablehlo.multiply %v4003, %v4007 : tensor<384x96x1x1xf32>
    %v4009 = stablehlo.subtract %s0b1eW, %v4008 : tensor<384x96x1x1xf32>
    %v4010 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4011 = stablehlo.multiply %v4010, %v4003 : tensor<384x96x1x1xf32>
    %v4012 = stablehlo.multiply %v4011, %s0b1eW : tensor<384x96x1x1xf32>
    %v4013 = stablehlo.subtract %v4009, %v4012 : tensor<384x96x1x1xf32>
    %arsums0b1eb = "stablehlo.all_reduce"(%v3264) ({
    ^bb0(%aras0b1eb: tensor<f32>, %arbs0b1eb: tensor<f32>):
      %aradds0b1eb = stablehlo.add %aras0b1eb, %arbs0b1eb : tensor<f32>
      stablehlo.return %aradds0b1eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns0b1eb = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans0b1eb = stablehlo.divide %arsums0b1eb, %arns0b1eb : tensor<384xf32>
    %v4014 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4015 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4016 = stablehlo.multiply %v4014, %s0b1ebm : tensor<384xf32>
    %v4017 = stablehlo.multiply %v4015, %armeans0b1eb : tensor<384xf32>
    %v4018 = stablehlo.add %v4016, %v4017 : tensor<384xf32>
    %v4019 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4020 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4021 = stablehlo.multiply %v4019, %s0b1ebv : tensor<384xf32>
    %v4022 = stablehlo.multiply %armeans0b1eb, %armeans0b1eb : tensor<384xf32>
    %v4023 = stablehlo.multiply %v4020, %v4022 : tensor<384xf32>
    %v4024 = stablehlo.add %v4021, %v4023 : tensor<384xf32>
    %v4025 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4026 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4027 = stablehlo.multiply %v4025, %s0b1ebm : tensor<384xf32>
    %v4028 = stablehlo.multiply %v4026, %armeans0b1eb : tensor<384xf32>
    %v4029 = stablehlo.add %v4027, %v4028 : tensor<384xf32>
    %v4030 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4031 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4032 = stablehlo.multiply %v4030, %s0b1ebv : tensor<384xf32>
    %v4033 = stablehlo.multiply %armeans0b1eb, %armeans0b1eb : tensor<384xf32>
    %v4034 = stablehlo.multiply %v4031, %v4033 : tensor<384xf32>
    %v4035 = stablehlo.add %v4032, %v4034 : tensor<384xf32>
    %v4036 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4037 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4038 = stablehlo.divide %v4029, %v4036 : tensor<384xf32>
    %v4039 = stablehlo.divide %v4035, %v4037 : tensor<384xf32>
    %v4040 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4041 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4042 = stablehlo.sqrt %v4039 : tensor<384xf32>
    %v4043 = stablehlo.add %v4042, %v4041 : tensor<384xf32>
    %v4044 = stablehlo.divide %v4038, %v4043 : tensor<384xf32>
    %v4045 = stablehlo.multiply %v4040, %v4044 : tensor<384xf32>
    %v4046 = stablehlo.subtract %s0b1eb, %v4045 : tensor<384xf32>
    %v4047 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4048 = stablehlo.multiply %v4047, %v4040 : tensor<384xf32>
    %v4049 = stablehlo.multiply %v4048, %s0b1eb : tensor<384xf32>
    %v4050 = stablehlo.subtract %v4046, %v4049 : tensor<384xf32>
    %arsums0b1pW = "stablehlo.all_reduce"(%v3252) ({
    ^bb0(%aras0b1pW: tensor<f32>, %arbs0b1pW: tensor<f32>):
      %aradds0b1pW = stablehlo.add %aras0b1pW, %arbs0b1pW : tensor<f32>
      stablehlo.return %aradds0b1pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96x384x1x1xf32>) -> tensor<96x384x1x1xf32>
    %arns0b1pW = stablehlo.constant dense<2.0> : tensor<96x384x1x1xf32>
    %armeans0b1pW = stablehlo.divide %arsums0b1pW, %arns0b1pW : tensor<96x384x1x1xf32>
    %v4051 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4052 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4053 = stablehlo.multiply %v4051, %s0b1pWm : tensor<96x384x1x1xf32>
    %v4054 = stablehlo.multiply %v4052, %armeans0b1pW : tensor<96x384x1x1xf32>
    %v4055 = stablehlo.add %v4053, %v4054 : tensor<96x384x1x1xf32>
    %v4056 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4057 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4058 = stablehlo.multiply %v4056, %s0b1pWv : tensor<96x384x1x1xf32>
    %v4059 = stablehlo.multiply %armeans0b1pW, %armeans0b1pW : tensor<96x384x1x1xf32>
    %v4060 = stablehlo.multiply %v4057, %v4059 : tensor<96x384x1x1xf32>
    %v4061 = stablehlo.add %v4058, %v4060 : tensor<96x384x1x1xf32>
    %v4062 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4063 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4064 = stablehlo.multiply %v4062, %s0b1pWm : tensor<96x384x1x1xf32>
    %v4065 = stablehlo.multiply %v4063, %armeans0b1pW : tensor<96x384x1x1xf32>
    %v4066 = stablehlo.add %v4064, %v4065 : tensor<96x384x1x1xf32>
    %v4067 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4068 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4069 = stablehlo.multiply %v4067, %s0b1pWv : tensor<96x384x1x1xf32>
    %v4070 = stablehlo.multiply %armeans0b1pW, %armeans0b1pW : tensor<96x384x1x1xf32>
    %v4071 = stablehlo.multiply %v4068, %v4070 : tensor<96x384x1x1xf32>
    %v4072 = stablehlo.add %v4069, %v4071 : tensor<96x384x1x1xf32>
    %v4073 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4074 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4075 = stablehlo.divide %v4066, %v4073 : tensor<96x384x1x1xf32>
    %v4076 = stablehlo.divide %v4072, %v4074 : tensor<96x384x1x1xf32>
    %v4077 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4078 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4079 = stablehlo.sqrt %v4076 : tensor<96x384x1x1xf32>
    %v4080 = stablehlo.add %v4079, %v4078 : tensor<96x384x1x1xf32>
    %v4081 = stablehlo.divide %v4075, %v4080 : tensor<96x384x1x1xf32>
    %v4082 = stablehlo.multiply %v4077, %v4081 : tensor<96x384x1x1xf32>
    %v4083 = stablehlo.subtract %s0b1pW, %v4082 : tensor<96x384x1x1xf32>
    %v4084 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4085 = stablehlo.multiply %v4084, %v4077 : tensor<96x384x1x1xf32>
    %v4086 = stablehlo.multiply %v4085, %s0b1pW : tensor<96x384x1x1xf32>
    %v4087 = stablehlo.subtract %v4083, %v4086 : tensor<96x384x1x1xf32>
    %arsums0b1pb = "stablehlo.all_reduce"(%v3255) ({
    ^bb0(%aras0b1pb: tensor<f32>, %arbs0b1pb: tensor<f32>):
      %aradds0b1pb = stablehlo.add %aras0b1pb, %arbs0b1pb : tensor<f32>
      stablehlo.return %aradds0b1pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96xf32>) -> tensor<96xf32>
    %arns0b1pb = stablehlo.constant dense<2.0> : tensor<96xf32>
    %armeans0b1pb = stablehlo.divide %arsums0b1pb, %arns0b1pb : tensor<96xf32>
    %v4088 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4089 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4090 = stablehlo.multiply %v4088, %s0b1pbm : tensor<96xf32>
    %v4091 = stablehlo.multiply %v4089, %armeans0b1pb : tensor<96xf32>
    %v4092 = stablehlo.add %v4090, %v4091 : tensor<96xf32>
    %v4093 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4094 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4095 = stablehlo.multiply %v4093, %s0b1pbv : tensor<96xf32>
    %v4096 = stablehlo.multiply %armeans0b1pb, %armeans0b1pb : tensor<96xf32>
    %v4097 = stablehlo.multiply %v4094, %v4096 : tensor<96xf32>
    %v4098 = stablehlo.add %v4095, %v4097 : tensor<96xf32>
    %v4099 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4100 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4101 = stablehlo.multiply %v4099, %s0b1pbm : tensor<96xf32>
    %v4102 = stablehlo.multiply %v4100, %armeans0b1pb : tensor<96xf32>
    %v4103 = stablehlo.add %v4101, %v4102 : tensor<96xf32>
    %v4104 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4105 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4106 = stablehlo.multiply %v4104, %s0b1pbv : tensor<96xf32>
    %v4107 = stablehlo.multiply %armeans0b1pb, %armeans0b1pb : tensor<96xf32>
    %v4108 = stablehlo.multiply %v4105, %v4107 : tensor<96xf32>
    %v4109 = stablehlo.add %v4106, %v4108 : tensor<96xf32>
    %v4110 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4111 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4112 = stablehlo.divide %v4103, %v4110 : tensor<96xf32>
    %v4113 = stablehlo.divide %v4109, %v4111 : tensor<96xf32>
    %v4114 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4115 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4116 = stablehlo.sqrt %v4113 : tensor<96xf32>
    %v4117 = stablehlo.add %v4116, %v4115 : tensor<96xf32>
    %v4118 = stablehlo.divide %v4112, %v4117 : tensor<96xf32>
    %v4119 = stablehlo.multiply %v4114, %v4118 : tensor<96xf32>
    %v4120 = stablehlo.subtract %s0b1pb, %v4119 : tensor<96xf32>
    %v4121 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4122 = stablehlo.multiply %v4121, %v4114 : tensor<96xf32>
    %v4123 = stablehlo.multiply %v4122, %s0b1pb : tensor<96xf32>
    %v4124 = stablehlo.subtract %v4120, %v4123 : tensor<96xf32>
    %arsums0b1lg = "stablehlo.all_reduce"(%v3246) ({
    ^bb0(%aras0b1lg: tensor<f32>, %arbs0b1lg: tensor<f32>):
      %aradds0b1lg = stablehlo.add %aras0b1lg, %arbs0b1lg : tensor<f32>
      stablehlo.return %aradds0b1lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96xf32>) -> tensor<96xf32>
    %arns0b1lg = stablehlo.constant dense<2.0> : tensor<96xf32>
    %armeans0b1lg = stablehlo.divide %arsums0b1lg, %arns0b1lg : tensor<96xf32>
    %v4125 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4126 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4127 = stablehlo.multiply %v4125, %s0b1lgm : tensor<96xf32>
    %v4128 = stablehlo.multiply %v4126, %armeans0b1lg : tensor<96xf32>
    %v4129 = stablehlo.add %v4127, %v4128 : tensor<96xf32>
    %v4130 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4131 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4132 = stablehlo.multiply %v4130, %s0b1lgv : tensor<96xf32>
    %v4133 = stablehlo.multiply %armeans0b1lg, %armeans0b1lg : tensor<96xf32>
    %v4134 = stablehlo.multiply %v4131, %v4133 : tensor<96xf32>
    %v4135 = stablehlo.add %v4132, %v4134 : tensor<96xf32>
    %v4136 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4137 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4138 = stablehlo.multiply %v4136, %s0b1lgm : tensor<96xf32>
    %v4139 = stablehlo.multiply %v4137, %armeans0b1lg : tensor<96xf32>
    %v4140 = stablehlo.add %v4138, %v4139 : tensor<96xf32>
    %v4141 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4142 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4143 = stablehlo.multiply %v4141, %s0b1lgv : tensor<96xf32>
    %v4144 = stablehlo.multiply %armeans0b1lg, %armeans0b1lg : tensor<96xf32>
    %v4145 = stablehlo.multiply %v4142, %v4144 : tensor<96xf32>
    %v4146 = stablehlo.add %v4143, %v4145 : tensor<96xf32>
    %v4147 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4148 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4149 = stablehlo.divide %v4140, %v4147 : tensor<96xf32>
    %v4150 = stablehlo.divide %v4146, %v4148 : tensor<96xf32>
    %v4151 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4152 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4153 = stablehlo.sqrt %v4150 : tensor<96xf32>
    %v4154 = stablehlo.add %v4153, %v4152 : tensor<96xf32>
    %v4155 = stablehlo.divide %v4149, %v4154 : tensor<96xf32>
    %v4156 = stablehlo.multiply %v4151, %v4155 : tensor<96xf32>
    %v4157 = stablehlo.subtract %s0b1lg, %v4156 : tensor<96xf32>
    %v4158 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4159 = stablehlo.multiply %v4158, %v4151 : tensor<96xf32>
    %v4160 = stablehlo.multiply %v4159, %s0b1lg : tensor<96xf32>
    %v4161 = stablehlo.subtract %v4157, %v4160 : tensor<96xf32>
    %arsums0b2dW = "stablehlo.all_reduce"(%v3169) ({
    ^bb0(%aras0b2dW: tensor<f32>, %arbs0b2dW: tensor<f32>):
      %aradds0b2dW = stablehlo.add %aras0b2dW, %arbs0b2dW : tensor<f32>
      stablehlo.return %aradds0b2dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96x1x7x7xf32>) -> tensor<96x1x7x7xf32>
    %arns0b2dW = stablehlo.constant dense<2.0> : tensor<96x1x7x7xf32>
    %armeans0b2dW = stablehlo.divide %arsums0b2dW, %arns0b2dW : tensor<96x1x7x7xf32>
    %v4162 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4163 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4164 = stablehlo.multiply %v4162, %s0b2dWm : tensor<96x1x7x7xf32>
    %v4165 = stablehlo.multiply %v4163, %armeans0b2dW : tensor<96x1x7x7xf32>
    %v4166 = stablehlo.add %v4164, %v4165 : tensor<96x1x7x7xf32>
    %v4167 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4168 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4169 = stablehlo.multiply %v4167, %s0b2dWv : tensor<96x1x7x7xf32>
    %v4170 = stablehlo.multiply %armeans0b2dW, %armeans0b2dW : tensor<96x1x7x7xf32>
    %v4171 = stablehlo.multiply %v4168, %v4170 : tensor<96x1x7x7xf32>
    %v4172 = stablehlo.add %v4169, %v4171 : tensor<96x1x7x7xf32>
    %v4173 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4174 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4175 = stablehlo.multiply %v4173, %s0b2dWm : tensor<96x1x7x7xf32>
    %v4176 = stablehlo.multiply %v4174, %armeans0b2dW : tensor<96x1x7x7xf32>
    %v4177 = stablehlo.add %v4175, %v4176 : tensor<96x1x7x7xf32>
    %v4178 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4179 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4180 = stablehlo.multiply %v4178, %s0b2dWv : tensor<96x1x7x7xf32>
    %v4181 = stablehlo.multiply %armeans0b2dW, %armeans0b2dW : tensor<96x1x7x7xf32>
    %v4182 = stablehlo.multiply %v4179, %v4181 : tensor<96x1x7x7xf32>
    %v4183 = stablehlo.add %v4180, %v4182 : tensor<96x1x7x7xf32>
    %v4184 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4185 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4186 = stablehlo.divide %v4177, %v4184 : tensor<96x1x7x7xf32>
    %v4187 = stablehlo.divide %v4183, %v4185 : tensor<96x1x7x7xf32>
    %v4188 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4189 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4190 = stablehlo.sqrt %v4187 : tensor<96x1x7x7xf32>
    %v4191 = stablehlo.add %v4190, %v4189 : tensor<96x1x7x7xf32>
    %v4192 = stablehlo.divide %v4186, %v4191 : tensor<96x1x7x7xf32>
    %v4193 = stablehlo.multiply %v4188, %v4192 : tensor<96x1x7x7xf32>
    %v4194 = stablehlo.subtract %s0b2dW, %v4193 : tensor<96x1x7x7xf32>
    %v4195 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4196 = stablehlo.multiply %v4195, %v4188 : tensor<96x1x7x7xf32>
    %v4197 = stablehlo.multiply %v4196, %s0b2dW : tensor<96x1x7x7xf32>
    %v4198 = stablehlo.subtract %v4194, %v4197 : tensor<96x1x7x7xf32>
    %arsums0b2db = "stablehlo.all_reduce"(%v3172) ({
    ^bb0(%aras0b2db: tensor<f32>, %arbs0b2db: tensor<f32>):
      %aradds0b2db = stablehlo.add %aras0b2db, %arbs0b2db : tensor<f32>
      stablehlo.return %aradds0b2db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96xf32>) -> tensor<96xf32>
    %arns0b2db = stablehlo.constant dense<2.0> : tensor<96xf32>
    %armeans0b2db = stablehlo.divide %arsums0b2db, %arns0b2db : tensor<96xf32>
    %v4199 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4200 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4201 = stablehlo.multiply %v4199, %s0b2dbm : tensor<96xf32>
    %v4202 = stablehlo.multiply %v4200, %armeans0b2db : tensor<96xf32>
    %v4203 = stablehlo.add %v4201, %v4202 : tensor<96xf32>
    %v4204 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4205 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4206 = stablehlo.multiply %v4204, %s0b2dbv : tensor<96xf32>
    %v4207 = stablehlo.multiply %armeans0b2db, %armeans0b2db : tensor<96xf32>
    %v4208 = stablehlo.multiply %v4205, %v4207 : tensor<96xf32>
    %v4209 = stablehlo.add %v4206, %v4208 : tensor<96xf32>
    %v4210 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4211 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4212 = stablehlo.multiply %v4210, %s0b2dbm : tensor<96xf32>
    %v4213 = stablehlo.multiply %v4211, %armeans0b2db : tensor<96xf32>
    %v4214 = stablehlo.add %v4212, %v4213 : tensor<96xf32>
    %v4215 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4216 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4217 = stablehlo.multiply %v4215, %s0b2dbv : tensor<96xf32>
    %v4218 = stablehlo.multiply %armeans0b2db, %armeans0b2db : tensor<96xf32>
    %v4219 = stablehlo.multiply %v4216, %v4218 : tensor<96xf32>
    %v4220 = stablehlo.add %v4217, %v4219 : tensor<96xf32>
    %v4221 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4222 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4223 = stablehlo.divide %v4214, %v4221 : tensor<96xf32>
    %v4224 = stablehlo.divide %v4220, %v4222 : tensor<96xf32>
    %v4225 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4226 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4227 = stablehlo.sqrt %v4224 : tensor<96xf32>
    %v4228 = stablehlo.add %v4227, %v4226 : tensor<96xf32>
    %v4229 = stablehlo.divide %v4223, %v4228 : tensor<96xf32>
    %v4230 = stablehlo.multiply %v4225, %v4229 : tensor<96xf32>
    %v4231 = stablehlo.subtract %s0b2db, %v4230 : tensor<96xf32>
    %v4232 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4233 = stablehlo.multiply %v4232, %v4225 : tensor<96xf32>
    %v4234 = stablehlo.multiply %v4233, %s0b2db : tensor<96xf32>
    %v4235 = stablehlo.subtract %v4231, %v4234 : tensor<96xf32>
    %arsums0b2ng = "stablehlo.all_reduce"(%v3161) ({
    ^bb0(%aras0b2ng: tensor<f32>, %arbs0b2ng: tensor<f32>):
      %aradds0b2ng = stablehlo.add %aras0b2ng, %arbs0b2ng : tensor<f32>
      stablehlo.return %aradds0b2ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns0b2ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans0b2ng = stablehlo.divide %arsums0b2ng, %arns0b2ng : tensor<f32>
    %v4236 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4237 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4238 = stablehlo.multiply %v4236, %s0b2ngm : tensor<f32>
    %v4239 = stablehlo.multiply %v4237, %armeans0b2ng : tensor<f32>
    %v4240 = stablehlo.add %v4238, %v4239 : tensor<f32>
    %v4241 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4242 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4243 = stablehlo.multiply %v4241, %s0b2ngv : tensor<f32>
    %v4244 = stablehlo.multiply %armeans0b2ng, %armeans0b2ng : tensor<f32>
    %v4245 = stablehlo.multiply %v4242, %v4244 : tensor<f32>
    %v4246 = stablehlo.add %v4243, %v4245 : tensor<f32>
    %v4247 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4248 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4249 = stablehlo.multiply %v4247, %s0b2ngm : tensor<f32>
    %v4250 = stablehlo.multiply %v4248, %armeans0b2ng : tensor<f32>
    %v4251 = stablehlo.add %v4249, %v4250 : tensor<f32>
    %v4252 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4253 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4254 = stablehlo.multiply %v4252, %s0b2ngv : tensor<f32>
    %v4255 = stablehlo.multiply %armeans0b2ng, %armeans0b2ng : tensor<f32>
    %v4256 = stablehlo.multiply %v4253, %v4255 : tensor<f32>
    %v4257 = stablehlo.add %v4254, %v4256 : tensor<f32>
    %v4258 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4259 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4260 = stablehlo.divide %v4251, %v4258 : tensor<f32>
    %v4261 = stablehlo.divide %v4257, %v4259 : tensor<f32>
    %v4262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4263 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4264 = stablehlo.sqrt %v4261 : tensor<f32>
    %v4265 = stablehlo.add %v4264, %v4263 : tensor<f32>
    %v4266 = stablehlo.divide %v4260, %v4265 : tensor<f32>
    %v4267 = stablehlo.multiply %v4262, %v4266 : tensor<f32>
    %v4268 = stablehlo.subtract %s0b2ng, %v4267 : tensor<f32>
    %v4269 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4270 = stablehlo.multiply %v4269, %v4262 : tensor<f32>
    %v4271 = stablehlo.multiply %v4270, %s0b2ng : tensor<f32>
    %v4272 = stablehlo.subtract %v4268, %v4271 : tensor<f32>
    %arsums0b2nbt = "stablehlo.all_reduce"(%v3163) ({
    ^bb0(%aras0b2nbt: tensor<f32>, %arbs0b2nbt: tensor<f32>):
      %aradds0b2nbt = stablehlo.add %aras0b2nbt, %arbs0b2nbt : tensor<f32>
      stablehlo.return %aradds0b2nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns0b2nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans0b2nbt = stablehlo.divide %arsums0b2nbt, %arns0b2nbt : tensor<f32>
    %v4273 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4274 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4275 = stablehlo.multiply %v4273, %s0b2nbtm : tensor<f32>
    %v4276 = stablehlo.multiply %v4274, %armeans0b2nbt : tensor<f32>
    %v4277 = stablehlo.add %v4275, %v4276 : tensor<f32>
    %v4278 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4279 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4280 = stablehlo.multiply %v4278, %s0b2nbtv : tensor<f32>
    %v4281 = stablehlo.multiply %armeans0b2nbt, %armeans0b2nbt : tensor<f32>
    %v4282 = stablehlo.multiply %v4279, %v4281 : tensor<f32>
    %v4283 = stablehlo.add %v4280, %v4282 : tensor<f32>
    %v4284 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4285 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4286 = stablehlo.multiply %v4284, %s0b2nbtm : tensor<f32>
    %v4287 = stablehlo.multiply %v4285, %armeans0b2nbt : tensor<f32>
    %v4288 = stablehlo.add %v4286, %v4287 : tensor<f32>
    %v4289 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4290 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4291 = stablehlo.multiply %v4289, %s0b2nbtv : tensor<f32>
    %v4292 = stablehlo.multiply %armeans0b2nbt, %armeans0b2nbt : tensor<f32>
    %v4293 = stablehlo.multiply %v4290, %v4292 : tensor<f32>
    %v4294 = stablehlo.add %v4291, %v4293 : tensor<f32>
    %v4295 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4296 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4297 = stablehlo.divide %v4288, %v4295 : tensor<f32>
    %v4298 = stablehlo.divide %v4294, %v4296 : tensor<f32>
    %v4299 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4300 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4301 = stablehlo.sqrt %v4298 : tensor<f32>
    %v4302 = stablehlo.add %v4301, %v4300 : tensor<f32>
    %v4303 = stablehlo.divide %v4297, %v4302 : tensor<f32>
    %v4304 = stablehlo.multiply %v4299, %v4303 : tensor<f32>
    %v4305 = stablehlo.subtract %s0b2nbt, %v4304 : tensor<f32>
    %v4306 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4307 = stablehlo.multiply %v4306, %v4299 : tensor<f32>
    %v4308 = stablehlo.multiply %v4307, %s0b2nbt : tensor<f32>
    %v4309 = stablehlo.subtract %v4305, %v4308 : tensor<f32>
    %arsums0b2eW = "stablehlo.all_reduce"(%v3142) ({
    ^bb0(%aras0b2eW: tensor<f32>, %arbs0b2eW: tensor<f32>):
      %aradds0b2eW = stablehlo.add %aras0b2eW, %arbs0b2eW : tensor<f32>
      stablehlo.return %aradds0b2eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x96x1x1xf32>) -> tensor<384x96x1x1xf32>
    %arns0b2eW = stablehlo.constant dense<2.0> : tensor<384x96x1x1xf32>
    %armeans0b2eW = stablehlo.divide %arsums0b2eW, %arns0b2eW : tensor<384x96x1x1xf32>
    %v4310 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4311 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4312 = stablehlo.multiply %v4310, %s0b2eWm : tensor<384x96x1x1xf32>
    %v4313 = stablehlo.multiply %v4311, %armeans0b2eW : tensor<384x96x1x1xf32>
    %v4314 = stablehlo.add %v4312, %v4313 : tensor<384x96x1x1xf32>
    %v4315 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4316 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4317 = stablehlo.multiply %v4315, %s0b2eWv : tensor<384x96x1x1xf32>
    %v4318 = stablehlo.multiply %armeans0b2eW, %armeans0b2eW : tensor<384x96x1x1xf32>
    %v4319 = stablehlo.multiply %v4316, %v4318 : tensor<384x96x1x1xf32>
    %v4320 = stablehlo.add %v4317, %v4319 : tensor<384x96x1x1xf32>
    %v4321 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4322 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4323 = stablehlo.multiply %v4321, %s0b2eWm : tensor<384x96x1x1xf32>
    %v4324 = stablehlo.multiply %v4322, %armeans0b2eW : tensor<384x96x1x1xf32>
    %v4325 = stablehlo.add %v4323, %v4324 : tensor<384x96x1x1xf32>
    %v4326 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4327 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4328 = stablehlo.multiply %v4326, %s0b2eWv : tensor<384x96x1x1xf32>
    %v4329 = stablehlo.multiply %armeans0b2eW, %armeans0b2eW : tensor<384x96x1x1xf32>
    %v4330 = stablehlo.multiply %v4327, %v4329 : tensor<384x96x1x1xf32>
    %v4331 = stablehlo.add %v4328, %v4330 : tensor<384x96x1x1xf32>
    %v4332 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4333 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4334 = stablehlo.divide %v4325, %v4332 : tensor<384x96x1x1xf32>
    %v4335 = stablehlo.divide %v4331, %v4333 : tensor<384x96x1x1xf32>
    %v4336 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4337 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4338 = stablehlo.sqrt %v4335 : tensor<384x96x1x1xf32>
    %v4339 = stablehlo.add %v4338, %v4337 : tensor<384x96x1x1xf32>
    %v4340 = stablehlo.divide %v4334, %v4339 : tensor<384x96x1x1xf32>
    %v4341 = stablehlo.multiply %v4336, %v4340 : tensor<384x96x1x1xf32>
    %v4342 = stablehlo.subtract %s0b2eW, %v4341 : tensor<384x96x1x1xf32>
    %v4343 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4344 = stablehlo.multiply %v4343, %v4336 : tensor<384x96x1x1xf32>
    %v4345 = stablehlo.multiply %v4344, %s0b2eW : tensor<384x96x1x1xf32>
    %v4346 = stablehlo.subtract %v4342, %v4345 : tensor<384x96x1x1xf32>
    %arsums0b2eb = "stablehlo.all_reduce"(%v3145) ({
    ^bb0(%aras0b2eb: tensor<f32>, %arbs0b2eb: tensor<f32>):
      %aradds0b2eb = stablehlo.add %aras0b2eb, %arbs0b2eb : tensor<f32>
      stablehlo.return %aradds0b2eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns0b2eb = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans0b2eb = stablehlo.divide %arsums0b2eb, %arns0b2eb : tensor<384xf32>
    %v4347 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4348 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4349 = stablehlo.multiply %v4347, %s0b2ebm : tensor<384xf32>
    %v4350 = stablehlo.multiply %v4348, %armeans0b2eb : tensor<384xf32>
    %v4351 = stablehlo.add %v4349, %v4350 : tensor<384xf32>
    %v4352 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4353 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4354 = stablehlo.multiply %v4352, %s0b2ebv : tensor<384xf32>
    %v4355 = stablehlo.multiply %armeans0b2eb, %armeans0b2eb : tensor<384xf32>
    %v4356 = stablehlo.multiply %v4353, %v4355 : tensor<384xf32>
    %v4357 = stablehlo.add %v4354, %v4356 : tensor<384xf32>
    %v4358 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4359 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4360 = stablehlo.multiply %v4358, %s0b2ebm : tensor<384xf32>
    %v4361 = stablehlo.multiply %v4359, %armeans0b2eb : tensor<384xf32>
    %v4362 = stablehlo.add %v4360, %v4361 : tensor<384xf32>
    %v4363 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4364 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4365 = stablehlo.multiply %v4363, %s0b2ebv : tensor<384xf32>
    %v4366 = stablehlo.multiply %armeans0b2eb, %armeans0b2eb : tensor<384xf32>
    %v4367 = stablehlo.multiply %v4364, %v4366 : tensor<384xf32>
    %v4368 = stablehlo.add %v4365, %v4367 : tensor<384xf32>
    %v4369 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4370 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4371 = stablehlo.divide %v4362, %v4369 : tensor<384xf32>
    %v4372 = stablehlo.divide %v4368, %v4370 : tensor<384xf32>
    %v4373 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4374 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4375 = stablehlo.sqrt %v4372 : tensor<384xf32>
    %v4376 = stablehlo.add %v4375, %v4374 : tensor<384xf32>
    %v4377 = stablehlo.divide %v4371, %v4376 : tensor<384xf32>
    %v4378 = stablehlo.multiply %v4373, %v4377 : tensor<384xf32>
    %v4379 = stablehlo.subtract %s0b2eb, %v4378 : tensor<384xf32>
    %v4380 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4381 = stablehlo.multiply %v4380, %v4373 : tensor<384xf32>
    %v4382 = stablehlo.multiply %v4381, %s0b2eb : tensor<384xf32>
    %v4383 = stablehlo.subtract %v4379, %v4382 : tensor<384xf32>
    %arsums0b2pW = "stablehlo.all_reduce"(%v3133) ({
    ^bb0(%aras0b2pW: tensor<f32>, %arbs0b2pW: tensor<f32>):
      %aradds0b2pW = stablehlo.add %aras0b2pW, %arbs0b2pW : tensor<f32>
      stablehlo.return %aradds0b2pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96x384x1x1xf32>) -> tensor<96x384x1x1xf32>
    %arns0b2pW = stablehlo.constant dense<2.0> : tensor<96x384x1x1xf32>
    %armeans0b2pW = stablehlo.divide %arsums0b2pW, %arns0b2pW : tensor<96x384x1x1xf32>
    %v4384 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4385 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4386 = stablehlo.multiply %v4384, %s0b2pWm : tensor<96x384x1x1xf32>
    %v4387 = stablehlo.multiply %v4385, %armeans0b2pW : tensor<96x384x1x1xf32>
    %v4388 = stablehlo.add %v4386, %v4387 : tensor<96x384x1x1xf32>
    %v4389 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4390 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4391 = stablehlo.multiply %v4389, %s0b2pWv : tensor<96x384x1x1xf32>
    %v4392 = stablehlo.multiply %armeans0b2pW, %armeans0b2pW : tensor<96x384x1x1xf32>
    %v4393 = stablehlo.multiply %v4390, %v4392 : tensor<96x384x1x1xf32>
    %v4394 = stablehlo.add %v4391, %v4393 : tensor<96x384x1x1xf32>
    %v4395 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4396 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4397 = stablehlo.multiply %v4395, %s0b2pWm : tensor<96x384x1x1xf32>
    %v4398 = stablehlo.multiply %v4396, %armeans0b2pW : tensor<96x384x1x1xf32>
    %v4399 = stablehlo.add %v4397, %v4398 : tensor<96x384x1x1xf32>
    %v4400 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4401 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4402 = stablehlo.multiply %v4400, %s0b2pWv : tensor<96x384x1x1xf32>
    %v4403 = stablehlo.multiply %armeans0b2pW, %armeans0b2pW : tensor<96x384x1x1xf32>
    %v4404 = stablehlo.multiply %v4401, %v4403 : tensor<96x384x1x1xf32>
    %v4405 = stablehlo.add %v4402, %v4404 : tensor<96x384x1x1xf32>
    %v4406 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4407 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4408 = stablehlo.divide %v4399, %v4406 : tensor<96x384x1x1xf32>
    %v4409 = stablehlo.divide %v4405, %v4407 : tensor<96x384x1x1xf32>
    %v4410 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4411 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4412 = stablehlo.sqrt %v4409 : tensor<96x384x1x1xf32>
    %v4413 = stablehlo.add %v4412, %v4411 : tensor<96x384x1x1xf32>
    %v4414 = stablehlo.divide %v4408, %v4413 : tensor<96x384x1x1xf32>
    %v4415 = stablehlo.multiply %v4410, %v4414 : tensor<96x384x1x1xf32>
    %v4416 = stablehlo.subtract %s0b2pW, %v4415 : tensor<96x384x1x1xf32>
    %v4417 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4418 = stablehlo.multiply %v4417, %v4410 : tensor<96x384x1x1xf32>
    %v4419 = stablehlo.multiply %v4418, %s0b2pW : tensor<96x384x1x1xf32>
    %v4420 = stablehlo.subtract %v4416, %v4419 : tensor<96x384x1x1xf32>
    %arsums0b2pb = "stablehlo.all_reduce"(%v3136) ({
    ^bb0(%aras0b2pb: tensor<f32>, %arbs0b2pb: tensor<f32>):
      %aradds0b2pb = stablehlo.add %aras0b2pb, %arbs0b2pb : tensor<f32>
      stablehlo.return %aradds0b2pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96xf32>) -> tensor<96xf32>
    %arns0b2pb = stablehlo.constant dense<2.0> : tensor<96xf32>
    %armeans0b2pb = stablehlo.divide %arsums0b2pb, %arns0b2pb : tensor<96xf32>
    %v4421 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4422 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4423 = stablehlo.multiply %v4421, %s0b2pbm : tensor<96xf32>
    %v4424 = stablehlo.multiply %v4422, %armeans0b2pb : tensor<96xf32>
    %v4425 = stablehlo.add %v4423, %v4424 : tensor<96xf32>
    %v4426 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4427 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4428 = stablehlo.multiply %v4426, %s0b2pbv : tensor<96xf32>
    %v4429 = stablehlo.multiply %armeans0b2pb, %armeans0b2pb : tensor<96xf32>
    %v4430 = stablehlo.multiply %v4427, %v4429 : tensor<96xf32>
    %v4431 = stablehlo.add %v4428, %v4430 : tensor<96xf32>
    %v4432 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4433 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4434 = stablehlo.multiply %v4432, %s0b2pbm : tensor<96xf32>
    %v4435 = stablehlo.multiply %v4433, %armeans0b2pb : tensor<96xf32>
    %v4436 = stablehlo.add %v4434, %v4435 : tensor<96xf32>
    %v4437 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4438 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4439 = stablehlo.multiply %v4437, %s0b2pbv : tensor<96xf32>
    %v4440 = stablehlo.multiply %armeans0b2pb, %armeans0b2pb : tensor<96xf32>
    %v4441 = stablehlo.multiply %v4438, %v4440 : tensor<96xf32>
    %v4442 = stablehlo.add %v4439, %v4441 : tensor<96xf32>
    %v4443 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4444 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4445 = stablehlo.divide %v4436, %v4443 : tensor<96xf32>
    %v4446 = stablehlo.divide %v4442, %v4444 : tensor<96xf32>
    %v4447 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4448 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4449 = stablehlo.sqrt %v4446 : tensor<96xf32>
    %v4450 = stablehlo.add %v4449, %v4448 : tensor<96xf32>
    %v4451 = stablehlo.divide %v4445, %v4450 : tensor<96xf32>
    %v4452 = stablehlo.multiply %v4447, %v4451 : tensor<96xf32>
    %v4453 = stablehlo.subtract %s0b2pb, %v4452 : tensor<96xf32>
    %v4454 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4455 = stablehlo.multiply %v4454, %v4447 : tensor<96xf32>
    %v4456 = stablehlo.multiply %v4455, %s0b2pb : tensor<96xf32>
    %v4457 = stablehlo.subtract %v4453, %v4456 : tensor<96xf32>
    %arsums0b2lg = "stablehlo.all_reduce"(%v3127) ({
    ^bb0(%aras0b2lg: tensor<f32>, %arbs0b2lg: tensor<f32>):
      %aradds0b2lg = stablehlo.add %aras0b2lg, %arbs0b2lg : tensor<f32>
      stablehlo.return %aradds0b2lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<96xf32>) -> tensor<96xf32>
    %arns0b2lg = stablehlo.constant dense<2.0> : tensor<96xf32>
    %armeans0b2lg = stablehlo.divide %arsums0b2lg, %arns0b2lg : tensor<96xf32>
    %v4458 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4459 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4460 = stablehlo.multiply %v4458, %s0b2lgm : tensor<96xf32>
    %v4461 = stablehlo.multiply %v4459, %armeans0b2lg : tensor<96xf32>
    %v4462 = stablehlo.add %v4460, %v4461 : tensor<96xf32>
    %v4463 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4464 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4465 = stablehlo.multiply %v4463, %s0b2lgv : tensor<96xf32>
    %v4466 = stablehlo.multiply %armeans0b2lg, %armeans0b2lg : tensor<96xf32>
    %v4467 = stablehlo.multiply %v4464, %v4466 : tensor<96xf32>
    %v4468 = stablehlo.add %v4465, %v4467 : tensor<96xf32>
    %v4469 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4470 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4471 = stablehlo.multiply %v4469, %s0b2lgm : tensor<96xf32>
    %v4472 = stablehlo.multiply %v4470, %armeans0b2lg : tensor<96xf32>
    %v4473 = stablehlo.add %v4471, %v4472 : tensor<96xf32>
    %v4474 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4475 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4476 = stablehlo.multiply %v4474, %s0b2lgv : tensor<96xf32>
    %v4477 = stablehlo.multiply %armeans0b2lg, %armeans0b2lg : tensor<96xf32>
    %v4478 = stablehlo.multiply %v4475, %v4477 : tensor<96xf32>
    %v4479 = stablehlo.add %v4476, %v4478 : tensor<96xf32>
    %v4480 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4481 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4482 = stablehlo.divide %v4473, %v4480 : tensor<96xf32>
    %v4483 = stablehlo.divide %v4479, %v4481 : tensor<96xf32>
    %v4484 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4485 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4486 = stablehlo.sqrt %v4483 : tensor<96xf32>
    %v4487 = stablehlo.add %v4486, %v4485 : tensor<96xf32>
    %v4488 = stablehlo.divide %v4482, %v4487 : tensor<96xf32>
    %v4489 = stablehlo.multiply %v4484, %v4488 : tensor<96xf32>
    %v4490 = stablehlo.subtract %s0b2lg, %v4489 : tensor<96xf32>
    %v4491 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4492 = stablehlo.multiply %v4491, %v4484 : tensor<96xf32>
    %v4493 = stablehlo.multiply %v4492, %s0b2lg : tensor<96xf32>
    %v4494 = stablehlo.subtract %v4490, %v4493 : tensor<96xf32>
    %arsumd0ng = "stablehlo.all_reduce"(%v3043) ({
    ^bb0(%arad0ng: tensor<f32>, %arbd0ng: tensor<f32>):
      %araddd0ng = stablehlo.add %arad0ng, %arbd0ng : tensor<f32>
      stablehlo.return %araddd0ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arnd0ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeand0ng = stablehlo.divide %arsumd0ng, %arnd0ng : tensor<f32>
    %v4495 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4496 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4497 = stablehlo.multiply %v4495, %d0ngm : tensor<f32>
    %v4498 = stablehlo.multiply %v4496, %armeand0ng : tensor<f32>
    %v4499 = stablehlo.add %v4497, %v4498 : tensor<f32>
    %v4500 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4501 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4502 = stablehlo.multiply %v4500, %d0ngv : tensor<f32>
    %v4503 = stablehlo.multiply %armeand0ng, %armeand0ng : tensor<f32>
    %v4504 = stablehlo.multiply %v4501, %v4503 : tensor<f32>
    %v4505 = stablehlo.add %v4502, %v4504 : tensor<f32>
    %v4506 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4507 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4508 = stablehlo.multiply %v4506, %d0ngm : tensor<f32>
    %v4509 = stablehlo.multiply %v4507, %armeand0ng : tensor<f32>
    %v4510 = stablehlo.add %v4508, %v4509 : tensor<f32>
    %v4511 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4512 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4513 = stablehlo.multiply %v4511, %d0ngv : tensor<f32>
    %v4514 = stablehlo.multiply %armeand0ng, %armeand0ng : tensor<f32>
    %v4515 = stablehlo.multiply %v4512, %v4514 : tensor<f32>
    %v4516 = stablehlo.add %v4513, %v4515 : tensor<f32>
    %v4517 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4518 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4519 = stablehlo.divide %v4510, %v4517 : tensor<f32>
    %v4520 = stablehlo.divide %v4516, %v4518 : tensor<f32>
    %v4521 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4522 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4523 = stablehlo.sqrt %v4520 : tensor<f32>
    %v4524 = stablehlo.add %v4523, %v4522 : tensor<f32>
    %v4525 = stablehlo.divide %v4519, %v4524 : tensor<f32>
    %v4526 = stablehlo.multiply %v4521, %v4525 : tensor<f32>
    %v4527 = stablehlo.subtract %d0ng, %v4526 : tensor<f32>
    %v4528 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4529 = stablehlo.multiply %v4528, %v4521 : tensor<f32>
    %v4530 = stablehlo.multiply %v4529, %d0ng : tensor<f32>
    %v4531 = stablehlo.subtract %v4527, %v4530 : tensor<f32>
    %arsumd0nbt = "stablehlo.all_reduce"(%v3045) ({
    ^bb0(%arad0nbt: tensor<f32>, %arbd0nbt: tensor<f32>):
      %araddd0nbt = stablehlo.add %arad0nbt, %arbd0nbt : tensor<f32>
      stablehlo.return %araddd0nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arnd0nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeand0nbt = stablehlo.divide %arsumd0nbt, %arnd0nbt : tensor<f32>
    %v4532 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4533 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4534 = stablehlo.multiply %v4532, %d0nbtm : tensor<f32>
    %v4535 = stablehlo.multiply %v4533, %armeand0nbt : tensor<f32>
    %v4536 = stablehlo.add %v4534, %v4535 : tensor<f32>
    %v4537 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4538 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4539 = stablehlo.multiply %v4537, %d0nbtv : tensor<f32>
    %v4540 = stablehlo.multiply %armeand0nbt, %armeand0nbt : tensor<f32>
    %v4541 = stablehlo.multiply %v4538, %v4540 : tensor<f32>
    %v4542 = stablehlo.add %v4539, %v4541 : tensor<f32>
    %v4543 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4544 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4545 = stablehlo.multiply %v4543, %d0nbtm : tensor<f32>
    %v4546 = stablehlo.multiply %v4544, %armeand0nbt : tensor<f32>
    %v4547 = stablehlo.add %v4545, %v4546 : tensor<f32>
    %v4548 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4549 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4550 = stablehlo.multiply %v4548, %d0nbtv : tensor<f32>
    %v4551 = stablehlo.multiply %armeand0nbt, %armeand0nbt : tensor<f32>
    %v4552 = stablehlo.multiply %v4549, %v4551 : tensor<f32>
    %v4553 = stablehlo.add %v4550, %v4552 : tensor<f32>
    %v4554 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4555 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4556 = stablehlo.divide %v4547, %v4554 : tensor<f32>
    %v4557 = stablehlo.divide %v4553, %v4555 : tensor<f32>
    %v4558 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4559 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4560 = stablehlo.sqrt %v4557 : tensor<f32>
    %v4561 = stablehlo.add %v4560, %v4559 : tensor<f32>
    %v4562 = stablehlo.divide %v4556, %v4561 : tensor<f32>
    %v4563 = stablehlo.multiply %v4558, %v4562 : tensor<f32>
    %v4564 = stablehlo.subtract %d0nbt, %v4563 : tensor<f32>
    %v4565 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4566 = stablehlo.multiply %v4565, %v4558 : tensor<f32>
    %v4567 = stablehlo.multiply %v4566, %d0nbt : tensor<f32>
    %v4568 = stablehlo.subtract %v4564, %v4567 : tensor<f32>
    %arsumd0W = "stablehlo.all_reduce"(%v3053) ({
    ^bb0(%arad0W: tensor<f32>, %arbd0W: tensor<f32>):
      %araddd0W = stablehlo.add %arad0W, %arbd0W : tensor<f32>
      stablehlo.return %araddd0W : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192x96x2x2xf32>) -> tensor<192x96x2x2xf32>
    %arnd0W = stablehlo.constant dense<2.0> : tensor<192x96x2x2xf32>
    %armeand0W = stablehlo.divide %arsumd0W, %arnd0W : tensor<192x96x2x2xf32>
    %v4569 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4570 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4571 = stablehlo.multiply %v4569, %d0Wm : tensor<192x96x2x2xf32>
    %v4572 = stablehlo.multiply %v4570, %armeand0W : tensor<192x96x2x2xf32>
    %v4573 = stablehlo.add %v4571, %v4572 : tensor<192x96x2x2xf32>
    %v4574 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4575 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4576 = stablehlo.multiply %v4574, %d0Wv : tensor<192x96x2x2xf32>
    %v4577 = stablehlo.multiply %armeand0W, %armeand0W : tensor<192x96x2x2xf32>
    %v4578 = stablehlo.multiply %v4575, %v4577 : tensor<192x96x2x2xf32>
    %v4579 = stablehlo.add %v4576, %v4578 : tensor<192x96x2x2xf32>
    %v4580 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4581 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4582 = stablehlo.multiply %v4580, %d0Wm : tensor<192x96x2x2xf32>
    %v4583 = stablehlo.multiply %v4581, %armeand0W : tensor<192x96x2x2xf32>
    %v4584 = stablehlo.add %v4582, %v4583 : tensor<192x96x2x2xf32>
    %v4585 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4586 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4587 = stablehlo.multiply %v4585, %d0Wv : tensor<192x96x2x2xf32>
    %v4588 = stablehlo.multiply %armeand0W, %armeand0W : tensor<192x96x2x2xf32>
    %v4589 = stablehlo.multiply %v4586, %v4588 : tensor<192x96x2x2xf32>
    %v4590 = stablehlo.add %v4587, %v4589 : tensor<192x96x2x2xf32>
    %v4591 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4592 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4593 = stablehlo.divide %v4584, %v4591 : tensor<192x96x2x2xf32>
    %v4594 = stablehlo.divide %v4590, %v4592 : tensor<192x96x2x2xf32>
    %v4595 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4596 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4597 = stablehlo.sqrt %v4594 : tensor<192x96x2x2xf32>
    %v4598 = stablehlo.add %v4597, %v4596 : tensor<192x96x2x2xf32>
    %v4599 = stablehlo.divide %v4593, %v4598 : tensor<192x96x2x2xf32>
    %v4600 = stablehlo.multiply %v4595, %v4599 : tensor<192x96x2x2xf32>
    %v4601 = stablehlo.subtract %d0W, %v4600 : tensor<192x96x2x2xf32>
    %v4602 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4603 = stablehlo.multiply %v4602, %v4595 : tensor<192x96x2x2xf32>
    %v4604 = stablehlo.multiply %v4603, %d0W : tensor<192x96x2x2xf32>
    %v4605 = stablehlo.subtract %v4601, %v4604 : tensor<192x96x2x2xf32>
    %arsumd0b = "stablehlo.all_reduce"(%v3027) ({
    ^bb0(%arad0b: tensor<f32>, %arbd0b: tensor<f32>):
      %araddd0b = stablehlo.add %arad0b, %arbd0b : tensor<f32>
      stablehlo.return %araddd0b : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192xf32>) -> tensor<192xf32>
    %arnd0b = stablehlo.constant dense<2.0> : tensor<192xf32>
    %armeand0b = stablehlo.divide %arsumd0b, %arnd0b : tensor<192xf32>
    %v4606 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4607 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4608 = stablehlo.multiply %v4606, %d0bm : tensor<192xf32>
    %v4609 = stablehlo.multiply %v4607, %armeand0b : tensor<192xf32>
    %v4610 = stablehlo.add %v4608, %v4609 : tensor<192xf32>
    %v4611 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4612 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4613 = stablehlo.multiply %v4611, %d0bv : tensor<192xf32>
    %v4614 = stablehlo.multiply %armeand0b, %armeand0b : tensor<192xf32>
    %v4615 = stablehlo.multiply %v4612, %v4614 : tensor<192xf32>
    %v4616 = stablehlo.add %v4613, %v4615 : tensor<192xf32>
    %v4617 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4618 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4619 = stablehlo.multiply %v4617, %d0bm : tensor<192xf32>
    %v4620 = stablehlo.multiply %v4618, %armeand0b : tensor<192xf32>
    %v4621 = stablehlo.add %v4619, %v4620 : tensor<192xf32>
    %v4622 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4623 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4624 = stablehlo.multiply %v4622, %d0bv : tensor<192xf32>
    %v4625 = stablehlo.multiply %armeand0b, %armeand0b : tensor<192xf32>
    %v4626 = stablehlo.multiply %v4623, %v4625 : tensor<192xf32>
    %v4627 = stablehlo.add %v4624, %v4626 : tensor<192xf32>
    %v4628 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4629 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4630 = stablehlo.divide %v4621, %v4628 : tensor<192xf32>
    %v4631 = stablehlo.divide %v4627, %v4629 : tensor<192xf32>
    %v4632 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4633 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4634 = stablehlo.sqrt %v4631 : tensor<192xf32>
    %v4635 = stablehlo.add %v4634, %v4633 : tensor<192xf32>
    %v4636 = stablehlo.divide %v4630, %v4635 : tensor<192xf32>
    %v4637 = stablehlo.multiply %v4632, %v4636 : tensor<192xf32>
    %v4638 = stablehlo.subtract %d0b, %v4637 : tensor<192xf32>
    %v4639 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4640 = stablehlo.multiply %v4639, %v4632 : tensor<192xf32>
    %v4641 = stablehlo.multiply %v4640, %d0b : tensor<192xf32>
    %v4642 = stablehlo.subtract %v4638, %v4641 : tensor<192xf32>
    %arsums1b0dW = "stablehlo.all_reduce"(%v2987) ({
    ^bb0(%aras1b0dW: tensor<f32>, %arbs1b0dW: tensor<f32>):
      %aradds1b0dW = stablehlo.add %aras1b0dW, %arbs1b0dW : tensor<f32>
      stablehlo.return %aradds1b0dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192x1x7x7xf32>) -> tensor<192x1x7x7xf32>
    %arns1b0dW = stablehlo.constant dense<2.0> : tensor<192x1x7x7xf32>
    %armeans1b0dW = stablehlo.divide %arsums1b0dW, %arns1b0dW : tensor<192x1x7x7xf32>
    %v4643 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4644 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4645 = stablehlo.multiply %v4643, %s1b0dWm : tensor<192x1x7x7xf32>
    %v4646 = stablehlo.multiply %v4644, %armeans1b0dW : tensor<192x1x7x7xf32>
    %v4647 = stablehlo.add %v4645, %v4646 : tensor<192x1x7x7xf32>
    %v4648 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4649 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4650 = stablehlo.multiply %v4648, %s1b0dWv : tensor<192x1x7x7xf32>
    %v4651 = stablehlo.multiply %armeans1b0dW, %armeans1b0dW : tensor<192x1x7x7xf32>
    %v4652 = stablehlo.multiply %v4649, %v4651 : tensor<192x1x7x7xf32>
    %v4653 = stablehlo.add %v4650, %v4652 : tensor<192x1x7x7xf32>
    %v4654 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4655 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4656 = stablehlo.multiply %v4654, %s1b0dWm : tensor<192x1x7x7xf32>
    %v4657 = stablehlo.multiply %v4655, %armeans1b0dW : tensor<192x1x7x7xf32>
    %v4658 = stablehlo.add %v4656, %v4657 : tensor<192x1x7x7xf32>
    %v4659 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4660 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4661 = stablehlo.multiply %v4659, %s1b0dWv : tensor<192x1x7x7xf32>
    %v4662 = stablehlo.multiply %armeans1b0dW, %armeans1b0dW : tensor<192x1x7x7xf32>
    %v4663 = stablehlo.multiply %v4660, %v4662 : tensor<192x1x7x7xf32>
    %v4664 = stablehlo.add %v4661, %v4663 : tensor<192x1x7x7xf32>
    %v4665 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4666 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4667 = stablehlo.divide %v4658, %v4665 : tensor<192x1x7x7xf32>
    %v4668 = stablehlo.divide %v4664, %v4666 : tensor<192x1x7x7xf32>
    %v4669 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4670 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4671 = stablehlo.sqrt %v4668 : tensor<192x1x7x7xf32>
    %v4672 = stablehlo.add %v4671, %v4670 : tensor<192x1x7x7xf32>
    %v4673 = stablehlo.divide %v4667, %v4672 : tensor<192x1x7x7xf32>
    %v4674 = stablehlo.multiply %v4669, %v4673 : tensor<192x1x7x7xf32>
    %v4675 = stablehlo.subtract %s1b0dW, %v4674 : tensor<192x1x7x7xf32>
    %v4676 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4677 = stablehlo.multiply %v4676, %v4669 : tensor<192x1x7x7xf32>
    %v4678 = stablehlo.multiply %v4677, %s1b0dW : tensor<192x1x7x7xf32>
    %v4679 = stablehlo.subtract %v4675, %v4678 : tensor<192x1x7x7xf32>
    %arsums1b0db = "stablehlo.all_reduce"(%v2990) ({
    ^bb0(%aras1b0db: tensor<f32>, %arbs1b0db: tensor<f32>):
      %aradds1b0db = stablehlo.add %aras1b0db, %arbs1b0db : tensor<f32>
      stablehlo.return %aradds1b0db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192xf32>) -> tensor<192xf32>
    %arns1b0db = stablehlo.constant dense<2.0> : tensor<192xf32>
    %armeans1b0db = stablehlo.divide %arsums1b0db, %arns1b0db : tensor<192xf32>
    %v4680 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4681 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4682 = stablehlo.multiply %v4680, %s1b0dbm : tensor<192xf32>
    %v4683 = stablehlo.multiply %v4681, %armeans1b0db : tensor<192xf32>
    %v4684 = stablehlo.add %v4682, %v4683 : tensor<192xf32>
    %v4685 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4686 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4687 = stablehlo.multiply %v4685, %s1b0dbv : tensor<192xf32>
    %v4688 = stablehlo.multiply %armeans1b0db, %armeans1b0db : tensor<192xf32>
    %v4689 = stablehlo.multiply %v4686, %v4688 : tensor<192xf32>
    %v4690 = stablehlo.add %v4687, %v4689 : tensor<192xf32>
    %v4691 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4692 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4693 = stablehlo.multiply %v4691, %s1b0dbm : tensor<192xf32>
    %v4694 = stablehlo.multiply %v4692, %armeans1b0db : tensor<192xf32>
    %v4695 = stablehlo.add %v4693, %v4694 : tensor<192xf32>
    %v4696 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4697 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4698 = stablehlo.multiply %v4696, %s1b0dbv : tensor<192xf32>
    %v4699 = stablehlo.multiply %armeans1b0db, %armeans1b0db : tensor<192xf32>
    %v4700 = stablehlo.multiply %v4697, %v4699 : tensor<192xf32>
    %v4701 = stablehlo.add %v4698, %v4700 : tensor<192xf32>
    %v4702 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4703 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4704 = stablehlo.divide %v4695, %v4702 : tensor<192xf32>
    %v4705 = stablehlo.divide %v4701, %v4703 : tensor<192xf32>
    %v4706 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4707 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4708 = stablehlo.sqrt %v4705 : tensor<192xf32>
    %v4709 = stablehlo.add %v4708, %v4707 : tensor<192xf32>
    %v4710 = stablehlo.divide %v4704, %v4709 : tensor<192xf32>
    %v4711 = stablehlo.multiply %v4706, %v4710 : tensor<192xf32>
    %v4712 = stablehlo.subtract %s1b0db, %v4711 : tensor<192xf32>
    %v4713 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4714 = stablehlo.multiply %v4713, %v4706 : tensor<192xf32>
    %v4715 = stablehlo.multiply %v4714, %s1b0db : tensor<192xf32>
    %v4716 = stablehlo.subtract %v4712, %v4715 : tensor<192xf32>
    %arsums1b0ng = "stablehlo.all_reduce"(%v2979) ({
    ^bb0(%aras1b0ng: tensor<f32>, %arbs1b0ng: tensor<f32>):
      %aradds1b0ng = stablehlo.add %aras1b0ng, %arbs1b0ng : tensor<f32>
      stablehlo.return %aradds1b0ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns1b0ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans1b0ng = stablehlo.divide %arsums1b0ng, %arns1b0ng : tensor<f32>
    %v4717 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4718 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4719 = stablehlo.multiply %v4717, %s1b0ngm : tensor<f32>
    %v4720 = stablehlo.multiply %v4718, %armeans1b0ng : tensor<f32>
    %v4721 = stablehlo.add %v4719, %v4720 : tensor<f32>
    %v4722 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4723 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4724 = stablehlo.multiply %v4722, %s1b0ngv : tensor<f32>
    %v4725 = stablehlo.multiply %armeans1b0ng, %armeans1b0ng : tensor<f32>
    %v4726 = stablehlo.multiply %v4723, %v4725 : tensor<f32>
    %v4727 = stablehlo.add %v4724, %v4726 : tensor<f32>
    %v4728 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4729 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4730 = stablehlo.multiply %v4728, %s1b0ngm : tensor<f32>
    %v4731 = stablehlo.multiply %v4729, %armeans1b0ng : tensor<f32>
    %v4732 = stablehlo.add %v4730, %v4731 : tensor<f32>
    %v4733 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4734 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4735 = stablehlo.multiply %v4733, %s1b0ngv : tensor<f32>
    %v4736 = stablehlo.multiply %armeans1b0ng, %armeans1b0ng : tensor<f32>
    %v4737 = stablehlo.multiply %v4734, %v4736 : tensor<f32>
    %v4738 = stablehlo.add %v4735, %v4737 : tensor<f32>
    %v4739 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4740 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4741 = stablehlo.divide %v4732, %v4739 : tensor<f32>
    %v4742 = stablehlo.divide %v4738, %v4740 : tensor<f32>
    %v4743 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4744 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4745 = stablehlo.sqrt %v4742 : tensor<f32>
    %v4746 = stablehlo.add %v4745, %v4744 : tensor<f32>
    %v4747 = stablehlo.divide %v4741, %v4746 : tensor<f32>
    %v4748 = stablehlo.multiply %v4743, %v4747 : tensor<f32>
    %v4749 = stablehlo.subtract %s1b0ng, %v4748 : tensor<f32>
    %v4750 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4751 = stablehlo.multiply %v4750, %v4743 : tensor<f32>
    %v4752 = stablehlo.multiply %v4751, %s1b0ng : tensor<f32>
    %v4753 = stablehlo.subtract %v4749, %v4752 : tensor<f32>
    %arsums1b0nbt = "stablehlo.all_reduce"(%v2981) ({
    ^bb0(%aras1b0nbt: tensor<f32>, %arbs1b0nbt: tensor<f32>):
      %aradds1b0nbt = stablehlo.add %aras1b0nbt, %arbs1b0nbt : tensor<f32>
      stablehlo.return %aradds1b0nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns1b0nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans1b0nbt = stablehlo.divide %arsums1b0nbt, %arns1b0nbt : tensor<f32>
    %v4754 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4755 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4756 = stablehlo.multiply %v4754, %s1b0nbtm : tensor<f32>
    %v4757 = stablehlo.multiply %v4755, %armeans1b0nbt : tensor<f32>
    %v4758 = stablehlo.add %v4756, %v4757 : tensor<f32>
    %v4759 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4760 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4761 = stablehlo.multiply %v4759, %s1b0nbtv : tensor<f32>
    %v4762 = stablehlo.multiply %armeans1b0nbt, %armeans1b0nbt : tensor<f32>
    %v4763 = stablehlo.multiply %v4760, %v4762 : tensor<f32>
    %v4764 = stablehlo.add %v4761, %v4763 : tensor<f32>
    %v4765 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4766 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4767 = stablehlo.multiply %v4765, %s1b0nbtm : tensor<f32>
    %v4768 = stablehlo.multiply %v4766, %armeans1b0nbt : tensor<f32>
    %v4769 = stablehlo.add %v4767, %v4768 : tensor<f32>
    %v4770 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4771 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4772 = stablehlo.multiply %v4770, %s1b0nbtv : tensor<f32>
    %v4773 = stablehlo.multiply %armeans1b0nbt, %armeans1b0nbt : tensor<f32>
    %v4774 = stablehlo.multiply %v4771, %v4773 : tensor<f32>
    %v4775 = stablehlo.add %v4772, %v4774 : tensor<f32>
    %v4776 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4777 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4778 = stablehlo.divide %v4769, %v4776 : tensor<f32>
    %v4779 = stablehlo.divide %v4775, %v4777 : tensor<f32>
    %v4780 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4781 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4782 = stablehlo.sqrt %v4779 : tensor<f32>
    %v4783 = stablehlo.add %v4782, %v4781 : tensor<f32>
    %v4784 = stablehlo.divide %v4778, %v4783 : tensor<f32>
    %v4785 = stablehlo.multiply %v4780, %v4784 : tensor<f32>
    %v4786 = stablehlo.subtract %s1b0nbt, %v4785 : tensor<f32>
    %v4787 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4788 = stablehlo.multiply %v4787, %v4780 : tensor<f32>
    %v4789 = stablehlo.multiply %v4788, %s1b0nbt : tensor<f32>
    %v4790 = stablehlo.subtract %v4786, %v4789 : tensor<f32>
    %arsums1b0eW = "stablehlo.all_reduce"(%v2960) ({
    ^bb0(%aras1b0eW: tensor<f32>, %arbs1b0eW: tensor<f32>):
      %aradds1b0eW = stablehlo.add %aras1b0eW, %arbs1b0eW : tensor<f32>
      stablehlo.return %aradds1b0eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768x192x1x1xf32>) -> tensor<768x192x1x1xf32>
    %arns1b0eW = stablehlo.constant dense<2.0> : tensor<768x192x1x1xf32>
    %armeans1b0eW = stablehlo.divide %arsums1b0eW, %arns1b0eW : tensor<768x192x1x1xf32>
    %v4791 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4792 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4793 = stablehlo.multiply %v4791, %s1b0eWm : tensor<768x192x1x1xf32>
    %v4794 = stablehlo.multiply %v4792, %armeans1b0eW : tensor<768x192x1x1xf32>
    %v4795 = stablehlo.add %v4793, %v4794 : tensor<768x192x1x1xf32>
    %v4796 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4797 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4798 = stablehlo.multiply %v4796, %s1b0eWv : tensor<768x192x1x1xf32>
    %v4799 = stablehlo.multiply %armeans1b0eW, %armeans1b0eW : tensor<768x192x1x1xf32>
    %v4800 = stablehlo.multiply %v4797, %v4799 : tensor<768x192x1x1xf32>
    %v4801 = stablehlo.add %v4798, %v4800 : tensor<768x192x1x1xf32>
    %v4802 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4803 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4804 = stablehlo.multiply %v4802, %s1b0eWm : tensor<768x192x1x1xf32>
    %v4805 = stablehlo.multiply %v4803, %armeans1b0eW : tensor<768x192x1x1xf32>
    %v4806 = stablehlo.add %v4804, %v4805 : tensor<768x192x1x1xf32>
    %v4807 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4808 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4809 = stablehlo.multiply %v4807, %s1b0eWv : tensor<768x192x1x1xf32>
    %v4810 = stablehlo.multiply %armeans1b0eW, %armeans1b0eW : tensor<768x192x1x1xf32>
    %v4811 = stablehlo.multiply %v4808, %v4810 : tensor<768x192x1x1xf32>
    %v4812 = stablehlo.add %v4809, %v4811 : tensor<768x192x1x1xf32>
    %v4813 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4814 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4815 = stablehlo.divide %v4806, %v4813 : tensor<768x192x1x1xf32>
    %v4816 = stablehlo.divide %v4812, %v4814 : tensor<768x192x1x1xf32>
    %v4817 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4818 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4819 = stablehlo.sqrt %v4816 : tensor<768x192x1x1xf32>
    %v4820 = stablehlo.add %v4819, %v4818 : tensor<768x192x1x1xf32>
    %v4821 = stablehlo.divide %v4815, %v4820 : tensor<768x192x1x1xf32>
    %v4822 = stablehlo.multiply %v4817, %v4821 : tensor<768x192x1x1xf32>
    %v4823 = stablehlo.subtract %s1b0eW, %v4822 : tensor<768x192x1x1xf32>
    %v4824 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4825 = stablehlo.multiply %v4824, %v4817 : tensor<768x192x1x1xf32>
    %v4826 = stablehlo.multiply %v4825, %s1b0eW : tensor<768x192x1x1xf32>
    %v4827 = stablehlo.subtract %v4823, %v4826 : tensor<768x192x1x1xf32>
    %arsums1b0eb = "stablehlo.all_reduce"(%v2963) ({
    ^bb0(%aras1b0eb: tensor<f32>, %arbs1b0eb: tensor<f32>):
      %aradds1b0eb = stablehlo.add %aras1b0eb, %arbs1b0eb : tensor<f32>
      stablehlo.return %aradds1b0eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arns1b0eb = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeans1b0eb = stablehlo.divide %arsums1b0eb, %arns1b0eb : tensor<768xf32>
    %v4828 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4829 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4830 = stablehlo.multiply %v4828, %s1b0ebm : tensor<768xf32>
    %v4831 = stablehlo.multiply %v4829, %armeans1b0eb : tensor<768xf32>
    %v4832 = stablehlo.add %v4830, %v4831 : tensor<768xf32>
    %v4833 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4834 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4835 = stablehlo.multiply %v4833, %s1b0ebv : tensor<768xf32>
    %v4836 = stablehlo.multiply %armeans1b0eb, %armeans1b0eb : tensor<768xf32>
    %v4837 = stablehlo.multiply %v4834, %v4836 : tensor<768xf32>
    %v4838 = stablehlo.add %v4835, %v4837 : tensor<768xf32>
    %v4839 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4840 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4841 = stablehlo.multiply %v4839, %s1b0ebm : tensor<768xf32>
    %v4842 = stablehlo.multiply %v4840, %armeans1b0eb : tensor<768xf32>
    %v4843 = stablehlo.add %v4841, %v4842 : tensor<768xf32>
    %v4844 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4845 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4846 = stablehlo.multiply %v4844, %s1b0ebv : tensor<768xf32>
    %v4847 = stablehlo.multiply %armeans1b0eb, %armeans1b0eb : tensor<768xf32>
    %v4848 = stablehlo.multiply %v4845, %v4847 : tensor<768xf32>
    %v4849 = stablehlo.add %v4846, %v4848 : tensor<768xf32>
    %v4850 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4851 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4852 = stablehlo.divide %v4843, %v4850 : tensor<768xf32>
    %v4853 = stablehlo.divide %v4849, %v4851 : tensor<768xf32>
    %v4854 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4855 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4856 = stablehlo.sqrt %v4853 : tensor<768xf32>
    %v4857 = stablehlo.add %v4856, %v4855 : tensor<768xf32>
    %v4858 = stablehlo.divide %v4852, %v4857 : tensor<768xf32>
    %v4859 = stablehlo.multiply %v4854, %v4858 : tensor<768xf32>
    %v4860 = stablehlo.subtract %s1b0eb, %v4859 : tensor<768xf32>
    %v4861 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4862 = stablehlo.multiply %v4861, %v4854 : tensor<768xf32>
    %v4863 = stablehlo.multiply %v4862, %s1b0eb : tensor<768xf32>
    %v4864 = stablehlo.subtract %v4860, %v4863 : tensor<768xf32>
    %arsums1b0pW = "stablehlo.all_reduce"(%v2951) ({
    ^bb0(%aras1b0pW: tensor<f32>, %arbs1b0pW: tensor<f32>):
      %aradds1b0pW = stablehlo.add %aras1b0pW, %arbs1b0pW : tensor<f32>
      stablehlo.return %aradds1b0pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192x768x1x1xf32>) -> tensor<192x768x1x1xf32>
    %arns1b0pW = stablehlo.constant dense<2.0> : tensor<192x768x1x1xf32>
    %armeans1b0pW = stablehlo.divide %arsums1b0pW, %arns1b0pW : tensor<192x768x1x1xf32>
    %v4865 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4866 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4867 = stablehlo.multiply %v4865, %s1b0pWm : tensor<192x768x1x1xf32>
    %v4868 = stablehlo.multiply %v4866, %armeans1b0pW : tensor<192x768x1x1xf32>
    %v4869 = stablehlo.add %v4867, %v4868 : tensor<192x768x1x1xf32>
    %v4870 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4871 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4872 = stablehlo.multiply %v4870, %s1b0pWv : tensor<192x768x1x1xf32>
    %v4873 = stablehlo.multiply %armeans1b0pW, %armeans1b0pW : tensor<192x768x1x1xf32>
    %v4874 = stablehlo.multiply %v4871, %v4873 : tensor<192x768x1x1xf32>
    %v4875 = stablehlo.add %v4872, %v4874 : tensor<192x768x1x1xf32>
    %v4876 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4877 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4878 = stablehlo.multiply %v4876, %s1b0pWm : tensor<192x768x1x1xf32>
    %v4879 = stablehlo.multiply %v4877, %armeans1b0pW : tensor<192x768x1x1xf32>
    %v4880 = stablehlo.add %v4878, %v4879 : tensor<192x768x1x1xf32>
    %v4881 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4882 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4883 = stablehlo.multiply %v4881, %s1b0pWv : tensor<192x768x1x1xf32>
    %v4884 = stablehlo.multiply %armeans1b0pW, %armeans1b0pW : tensor<192x768x1x1xf32>
    %v4885 = stablehlo.multiply %v4882, %v4884 : tensor<192x768x1x1xf32>
    %v4886 = stablehlo.add %v4883, %v4885 : tensor<192x768x1x1xf32>
    %v4887 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4888 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4889 = stablehlo.divide %v4880, %v4887 : tensor<192x768x1x1xf32>
    %v4890 = stablehlo.divide %v4886, %v4888 : tensor<192x768x1x1xf32>
    %v4891 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4892 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4893 = stablehlo.sqrt %v4890 : tensor<192x768x1x1xf32>
    %v4894 = stablehlo.add %v4893, %v4892 : tensor<192x768x1x1xf32>
    %v4895 = stablehlo.divide %v4889, %v4894 : tensor<192x768x1x1xf32>
    %v4896 = stablehlo.multiply %v4891, %v4895 : tensor<192x768x1x1xf32>
    %v4897 = stablehlo.subtract %s1b0pW, %v4896 : tensor<192x768x1x1xf32>
    %v4898 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4899 = stablehlo.multiply %v4898, %v4891 : tensor<192x768x1x1xf32>
    %v4900 = stablehlo.multiply %v4899, %s1b0pW : tensor<192x768x1x1xf32>
    %v4901 = stablehlo.subtract %v4897, %v4900 : tensor<192x768x1x1xf32>
    %arsums1b0pb = "stablehlo.all_reduce"(%v2954) ({
    ^bb0(%aras1b0pb: tensor<f32>, %arbs1b0pb: tensor<f32>):
      %aradds1b0pb = stablehlo.add %aras1b0pb, %arbs1b0pb : tensor<f32>
      stablehlo.return %aradds1b0pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192xf32>) -> tensor<192xf32>
    %arns1b0pb = stablehlo.constant dense<2.0> : tensor<192xf32>
    %armeans1b0pb = stablehlo.divide %arsums1b0pb, %arns1b0pb : tensor<192xf32>
    %v4902 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4903 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4904 = stablehlo.multiply %v4902, %s1b0pbm : tensor<192xf32>
    %v4905 = stablehlo.multiply %v4903, %armeans1b0pb : tensor<192xf32>
    %v4906 = stablehlo.add %v4904, %v4905 : tensor<192xf32>
    %v4907 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4908 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4909 = stablehlo.multiply %v4907, %s1b0pbv : tensor<192xf32>
    %v4910 = stablehlo.multiply %armeans1b0pb, %armeans1b0pb : tensor<192xf32>
    %v4911 = stablehlo.multiply %v4908, %v4910 : tensor<192xf32>
    %v4912 = stablehlo.add %v4909, %v4911 : tensor<192xf32>
    %v4913 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4914 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4915 = stablehlo.multiply %v4913, %s1b0pbm : tensor<192xf32>
    %v4916 = stablehlo.multiply %v4914, %armeans1b0pb : tensor<192xf32>
    %v4917 = stablehlo.add %v4915, %v4916 : tensor<192xf32>
    %v4918 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4919 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4920 = stablehlo.multiply %v4918, %s1b0pbv : tensor<192xf32>
    %v4921 = stablehlo.multiply %armeans1b0pb, %armeans1b0pb : tensor<192xf32>
    %v4922 = stablehlo.multiply %v4919, %v4921 : tensor<192xf32>
    %v4923 = stablehlo.add %v4920, %v4922 : tensor<192xf32>
    %v4924 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4925 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4926 = stablehlo.divide %v4917, %v4924 : tensor<192xf32>
    %v4927 = stablehlo.divide %v4923, %v4925 : tensor<192xf32>
    %v4928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4929 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4930 = stablehlo.sqrt %v4927 : tensor<192xf32>
    %v4931 = stablehlo.add %v4930, %v4929 : tensor<192xf32>
    %v4932 = stablehlo.divide %v4926, %v4931 : tensor<192xf32>
    %v4933 = stablehlo.multiply %v4928, %v4932 : tensor<192xf32>
    %v4934 = stablehlo.subtract %s1b0pb, %v4933 : tensor<192xf32>
    %v4935 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4936 = stablehlo.multiply %v4935, %v4928 : tensor<192xf32>
    %v4937 = stablehlo.multiply %v4936, %s1b0pb : tensor<192xf32>
    %v4938 = stablehlo.subtract %v4934, %v4937 : tensor<192xf32>
    %arsums1b0lg = "stablehlo.all_reduce"(%v2945) ({
    ^bb0(%aras1b0lg: tensor<f32>, %arbs1b0lg: tensor<f32>):
      %aradds1b0lg = stablehlo.add %aras1b0lg, %arbs1b0lg : tensor<f32>
      stablehlo.return %aradds1b0lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192xf32>) -> tensor<192xf32>
    %arns1b0lg = stablehlo.constant dense<2.0> : tensor<192xf32>
    %armeans1b0lg = stablehlo.divide %arsums1b0lg, %arns1b0lg : tensor<192xf32>
    %v4939 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4940 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4941 = stablehlo.multiply %v4939, %s1b0lgm : tensor<192xf32>
    %v4942 = stablehlo.multiply %v4940, %armeans1b0lg : tensor<192xf32>
    %v4943 = stablehlo.add %v4941, %v4942 : tensor<192xf32>
    %v4944 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4945 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4946 = stablehlo.multiply %v4944, %s1b0lgv : tensor<192xf32>
    %v4947 = stablehlo.multiply %armeans1b0lg, %armeans1b0lg : tensor<192xf32>
    %v4948 = stablehlo.multiply %v4945, %v4947 : tensor<192xf32>
    %v4949 = stablehlo.add %v4946, %v4948 : tensor<192xf32>
    %v4950 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4951 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4952 = stablehlo.multiply %v4950, %s1b0lgm : tensor<192xf32>
    %v4953 = stablehlo.multiply %v4951, %armeans1b0lg : tensor<192xf32>
    %v4954 = stablehlo.add %v4952, %v4953 : tensor<192xf32>
    %v4955 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4956 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4957 = stablehlo.multiply %v4955, %s1b0lgv : tensor<192xf32>
    %v4958 = stablehlo.multiply %armeans1b0lg, %armeans1b0lg : tensor<192xf32>
    %v4959 = stablehlo.multiply %v4956, %v4958 : tensor<192xf32>
    %v4960 = stablehlo.add %v4957, %v4959 : tensor<192xf32>
    %v4961 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4962 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4963 = stablehlo.divide %v4954, %v4961 : tensor<192xf32>
    %v4964 = stablehlo.divide %v4960, %v4962 : tensor<192xf32>
    %v4965 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4966 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4967 = stablehlo.sqrt %v4964 : tensor<192xf32>
    %v4968 = stablehlo.add %v4967, %v4966 : tensor<192xf32>
    %v4969 = stablehlo.divide %v4963, %v4968 : tensor<192xf32>
    %v4970 = stablehlo.multiply %v4965, %v4969 : tensor<192xf32>
    %v4971 = stablehlo.subtract %s1b0lg, %v4970 : tensor<192xf32>
    %v4972 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4973 = stablehlo.multiply %v4972, %v4965 : tensor<192xf32>
    %v4974 = stablehlo.multiply %v4973, %s1b0lg : tensor<192xf32>
    %v4975 = stablehlo.subtract %v4971, %v4974 : tensor<192xf32>
    %arsums1b1dW = "stablehlo.all_reduce"(%v2868) ({
    ^bb0(%aras1b1dW: tensor<f32>, %arbs1b1dW: tensor<f32>):
      %aradds1b1dW = stablehlo.add %aras1b1dW, %arbs1b1dW : tensor<f32>
      stablehlo.return %aradds1b1dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192x1x7x7xf32>) -> tensor<192x1x7x7xf32>
    %arns1b1dW = stablehlo.constant dense<2.0> : tensor<192x1x7x7xf32>
    %armeans1b1dW = stablehlo.divide %arsums1b1dW, %arns1b1dW : tensor<192x1x7x7xf32>
    %v4976 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4977 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4978 = stablehlo.multiply %v4976, %s1b1dWm : tensor<192x1x7x7xf32>
    %v4979 = stablehlo.multiply %v4977, %armeans1b1dW : tensor<192x1x7x7xf32>
    %v4980 = stablehlo.add %v4978, %v4979 : tensor<192x1x7x7xf32>
    %v4981 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4982 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4983 = stablehlo.multiply %v4981, %s1b1dWv : tensor<192x1x7x7xf32>
    %v4984 = stablehlo.multiply %armeans1b1dW, %armeans1b1dW : tensor<192x1x7x7xf32>
    %v4985 = stablehlo.multiply %v4982, %v4984 : tensor<192x1x7x7xf32>
    %v4986 = stablehlo.add %v4983, %v4985 : tensor<192x1x7x7xf32>
    %v4987 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4988 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4989 = stablehlo.multiply %v4987, %s1b1dWm : tensor<192x1x7x7xf32>
    %v4990 = stablehlo.multiply %v4988, %armeans1b1dW : tensor<192x1x7x7xf32>
    %v4991 = stablehlo.add %v4989, %v4990 : tensor<192x1x7x7xf32>
    %v4992 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4993 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4994 = stablehlo.multiply %v4992, %s1b1dWv : tensor<192x1x7x7xf32>
    %v4995 = stablehlo.multiply %armeans1b1dW, %armeans1b1dW : tensor<192x1x7x7xf32>
    %v4996 = stablehlo.multiply %v4993, %v4995 : tensor<192x1x7x7xf32>
    %v4997 = stablehlo.add %v4994, %v4996 : tensor<192x1x7x7xf32>
    %v4998 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4999 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5000 = stablehlo.divide %v4991, %v4998 : tensor<192x1x7x7xf32>
    %v5001 = stablehlo.divide %v4997, %v4999 : tensor<192x1x7x7xf32>
    %v5002 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5003 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5004 = stablehlo.sqrt %v5001 : tensor<192x1x7x7xf32>
    %v5005 = stablehlo.add %v5004, %v5003 : tensor<192x1x7x7xf32>
    %v5006 = stablehlo.divide %v5000, %v5005 : tensor<192x1x7x7xf32>
    %v5007 = stablehlo.multiply %v5002, %v5006 : tensor<192x1x7x7xf32>
    %v5008 = stablehlo.subtract %s1b1dW, %v5007 : tensor<192x1x7x7xf32>
    %v5009 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5010 = stablehlo.multiply %v5009, %v5002 : tensor<192x1x7x7xf32>
    %v5011 = stablehlo.multiply %v5010, %s1b1dW : tensor<192x1x7x7xf32>
    %v5012 = stablehlo.subtract %v5008, %v5011 : tensor<192x1x7x7xf32>
    %arsums1b1db = "stablehlo.all_reduce"(%v2871) ({
    ^bb0(%aras1b1db: tensor<f32>, %arbs1b1db: tensor<f32>):
      %aradds1b1db = stablehlo.add %aras1b1db, %arbs1b1db : tensor<f32>
      stablehlo.return %aradds1b1db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192xf32>) -> tensor<192xf32>
    %arns1b1db = stablehlo.constant dense<2.0> : tensor<192xf32>
    %armeans1b1db = stablehlo.divide %arsums1b1db, %arns1b1db : tensor<192xf32>
    %v5013 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5014 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5015 = stablehlo.multiply %v5013, %s1b1dbm : tensor<192xf32>
    %v5016 = stablehlo.multiply %v5014, %armeans1b1db : tensor<192xf32>
    %v5017 = stablehlo.add %v5015, %v5016 : tensor<192xf32>
    %v5018 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5019 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5020 = stablehlo.multiply %v5018, %s1b1dbv : tensor<192xf32>
    %v5021 = stablehlo.multiply %armeans1b1db, %armeans1b1db : tensor<192xf32>
    %v5022 = stablehlo.multiply %v5019, %v5021 : tensor<192xf32>
    %v5023 = stablehlo.add %v5020, %v5022 : tensor<192xf32>
    %v5024 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5025 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5026 = stablehlo.multiply %v5024, %s1b1dbm : tensor<192xf32>
    %v5027 = stablehlo.multiply %v5025, %armeans1b1db : tensor<192xf32>
    %v5028 = stablehlo.add %v5026, %v5027 : tensor<192xf32>
    %v5029 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5030 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5031 = stablehlo.multiply %v5029, %s1b1dbv : tensor<192xf32>
    %v5032 = stablehlo.multiply %armeans1b1db, %armeans1b1db : tensor<192xf32>
    %v5033 = stablehlo.multiply %v5030, %v5032 : tensor<192xf32>
    %v5034 = stablehlo.add %v5031, %v5033 : tensor<192xf32>
    %v5035 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5036 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5037 = stablehlo.divide %v5028, %v5035 : tensor<192xf32>
    %v5038 = stablehlo.divide %v5034, %v5036 : tensor<192xf32>
    %v5039 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5040 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5041 = stablehlo.sqrt %v5038 : tensor<192xf32>
    %v5042 = stablehlo.add %v5041, %v5040 : tensor<192xf32>
    %v5043 = stablehlo.divide %v5037, %v5042 : tensor<192xf32>
    %v5044 = stablehlo.multiply %v5039, %v5043 : tensor<192xf32>
    %v5045 = stablehlo.subtract %s1b1db, %v5044 : tensor<192xf32>
    %v5046 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5047 = stablehlo.multiply %v5046, %v5039 : tensor<192xf32>
    %v5048 = stablehlo.multiply %v5047, %s1b1db : tensor<192xf32>
    %v5049 = stablehlo.subtract %v5045, %v5048 : tensor<192xf32>
    %arsums1b1ng = "stablehlo.all_reduce"(%v2860) ({
    ^bb0(%aras1b1ng: tensor<f32>, %arbs1b1ng: tensor<f32>):
      %aradds1b1ng = stablehlo.add %aras1b1ng, %arbs1b1ng : tensor<f32>
      stablehlo.return %aradds1b1ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns1b1ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans1b1ng = stablehlo.divide %arsums1b1ng, %arns1b1ng : tensor<f32>
    %v5050 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5051 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5052 = stablehlo.multiply %v5050, %s1b1ngm : tensor<f32>
    %v5053 = stablehlo.multiply %v5051, %armeans1b1ng : tensor<f32>
    %v5054 = stablehlo.add %v5052, %v5053 : tensor<f32>
    %v5055 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5056 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5057 = stablehlo.multiply %v5055, %s1b1ngv : tensor<f32>
    %v5058 = stablehlo.multiply %armeans1b1ng, %armeans1b1ng : tensor<f32>
    %v5059 = stablehlo.multiply %v5056, %v5058 : tensor<f32>
    %v5060 = stablehlo.add %v5057, %v5059 : tensor<f32>
    %v5061 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5062 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5063 = stablehlo.multiply %v5061, %s1b1ngm : tensor<f32>
    %v5064 = stablehlo.multiply %v5062, %armeans1b1ng : tensor<f32>
    %v5065 = stablehlo.add %v5063, %v5064 : tensor<f32>
    %v5066 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5067 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5068 = stablehlo.multiply %v5066, %s1b1ngv : tensor<f32>
    %v5069 = stablehlo.multiply %armeans1b1ng, %armeans1b1ng : tensor<f32>
    %v5070 = stablehlo.multiply %v5067, %v5069 : tensor<f32>
    %v5071 = stablehlo.add %v5068, %v5070 : tensor<f32>
    %v5072 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5073 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5074 = stablehlo.divide %v5065, %v5072 : tensor<f32>
    %v5075 = stablehlo.divide %v5071, %v5073 : tensor<f32>
    %v5076 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5077 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5078 = stablehlo.sqrt %v5075 : tensor<f32>
    %v5079 = stablehlo.add %v5078, %v5077 : tensor<f32>
    %v5080 = stablehlo.divide %v5074, %v5079 : tensor<f32>
    %v5081 = stablehlo.multiply %v5076, %v5080 : tensor<f32>
    %v5082 = stablehlo.subtract %s1b1ng, %v5081 : tensor<f32>
    %v5083 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5084 = stablehlo.multiply %v5083, %v5076 : tensor<f32>
    %v5085 = stablehlo.multiply %v5084, %s1b1ng : tensor<f32>
    %v5086 = stablehlo.subtract %v5082, %v5085 : tensor<f32>
    %arsums1b1nbt = "stablehlo.all_reduce"(%v2862) ({
    ^bb0(%aras1b1nbt: tensor<f32>, %arbs1b1nbt: tensor<f32>):
      %aradds1b1nbt = stablehlo.add %aras1b1nbt, %arbs1b1nbt : tensor<f32>
      stablehlo.return %aradds1b1nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns1b1nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans1b1nbt = stablehlo.divide %arsums1b1nbt, %arns1b1nbt : tensor<f32>
    %v5087 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5088 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5089 = stablehlo.multiply %v5087, %s1b1nbtm : tensor<f32>
    %v5090 = stablehlo.multiply %v5088, %armeans1b1nbt : tensor<f32>
    %v5091 = stablehlo.add %v5089, %v5090 : tensor<f32>
    %v5092 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5093 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5094 = stablehlo.multiply %v5092, %s1b1nbtv : tensor<f32>
    %v5095 = stablehlo.multiply %armeans1b1nbt, %armeans1b1nbt : tensor<f32>
    %v5096 = stablehlo.multiply %v5093, %v5095 : tensor<f32>
    %v5097 = stablehlo.add %v5094, %v5096 : tensor<f32>
    %v5098 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5099 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5100 = stablehlo.multiply %v5098, %s1b1nbtm : tensor<f32>
    %v5101 = stablehlo.multiply %v5099, %armeans1b1nbt : tensor<f32>
    %v5102 = stablehlo.add %v5100, %v5101 : tensor<f32>
    %v5103 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5104 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5105 = stablehlo.multiply %v5103, %s1b1nbtv : tensor<f32>
    %v5106 = stablehlo.multiply %armeans1b1nbt, %armeans1b1nbt : tensor<f32>
    %v5107 = stablehlo.multiply %v5104, %v5106 : tensor<f32>
    %v5108 = stablehlo.add %v5105, %v5107 : tensor<f32>
    %v5109 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5110 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5111 = stablehlo.divide %v5102, %v5109 : tensor<f32>
    %v5112 = stablehlo.divide %v5108, %v5110 : tensor<f32>
    %v5113 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5114 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5115 = stablehlo.sqrt %v5112 : tensor<f32>
    %v5116 = stablehlo.add %v5115, %v5114 : tensor<f32>
    %v5117 = stablehlo.divide %v5111, %v5116 : tensor<f32>
    %v5118 = stablehlo.multiply %v5113, %v5117 : tensor<f32>
    %v5119 = stablehlo.subtract %s1b1nbt, %v5118 : tensor<f32>
    %v5120 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5121 = stablehlo.multiply %v5120, %v5113 : tensor<f32>
    %v5122 = stablehlo.multiply %v5121, %s1b1nbt : tensor<f32>
    %v5123 = stablehlo.subtract %v5119, %v5122 : tensor<f32>
    %arsums1b1eW = "stablehlo.all_reduce"(%v2841) ({
    ^bb0(%aras1b1eW: tensor<f32>, %arbs1b1eW: tensor<f32>):
      %aradds1b1eW = stablehlo.add %aras1b1eW, %arbs1b1eW : tensor<f32>
      stablehlo.return %aradds1b1eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768x192x1x1xf32>) -> tensor<768x192x1x1xf32>
    %arns1b1eW = stablehlo.constant dense<2.0> : tensor<768x192x1x1xf32>
    %armeans1b1eW = stablehlo.divide %arsums1b1eW, %arns1b1eW : tensor<768x192x1x1xf32>
    %v5124 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5125 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5126 = stablehlo.multiply %v5124, %s1b1eWm : tensor<768x192x1x1xf32>
    %v5127 = stablehlo.multiply %v5125, %armeans1b1eW : tensor<768x192x1x1xf32>
    %v5128 = stablehlo.add %v5126, %v5127 : tensor<768x192x1x1xf32>
    %v5129 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5130 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5131 = stablehlo.multiply %v5129, %s1b1eWv : tensor<768x192x1x1xf32>
    %v5132 = stablehlo.multiply %armeans1b1eW, %armeans1b1eW : tensor<768x192x1x1xf32>
    %v5133 = stablehlo.multiply %v5130, %v5132 : tensor<768x192x1x1xf32>
    %v5134 = stablehlo.add %v5131, %v5133 : tensor<768x192x1x1xf32>
    %v5135 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5136 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5137 = stablehlo.multiply %v5135, %s1b1eWm : tensor<768x192x1x1xf32>
    %v5138 = stablehlo.multiply %v5136, %armeans1b1eW : tensor<768x192x1x1xf32>
    %v5139 = stablehlo.add %v5137, %v5138 : tensor<768x192x1x1xf32>
    %v5140 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5141 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5142 = stablehlo.multiply %v5140, %s1b1eWv : tensor<768x192x1x1xf32>
    %v5143 = stablehlo.multiply %armeans1b1eW, %armeans1b1eW : tensor<768x192x1x1xf32>
    %v5144 = stablehlo.multiply %v5141, %v5143 : tensor<768x192x1x1xf32>
    %v5145 = stablehlo.add %v5142, %v5144 : tensor<768x192x1x1xf32>
    %v5146 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5147 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5148 = stablehlo.divide %v5139, %v5146 : tensor<768x192x1x1xf32>
    %v5149 = stablehlo.divide %v5145, %v5147 : tensor<768x192x1x1xf32>
    %v5150 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5151 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5152 = stablehlo.sqrt %v5149 : tensor<768x192x1x1xf32>
    %v5153 = stablehlo.add %v5152, %v5151 : tensor<768x192x1x1xf32>
    %v5154 = stablehlo.divide %v5148, %v5153 : tensor<768x192x1x1xf32>
    %v5155 = stablehlo.multiply %v5150, %v5154 : tensor<768x192x1x1xf32>
    %v5156 = stablehlo.subtract %s1b1eW, %v5155 : tensor<768x192x1x1xf32>
    %v5157 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5158 = stablehlo.multiply %v5157, %v5150 : tensor<768x192x1x1xf32>
    %v5159 = stablehlo.multiply %v5158, %s1b1eW : tensor<768x192x1x1xf32>
    %v5160 = stablehlo.subtract %v5156, %v5159 : tensor<768x192x1x1xf32>
    %arsums1b1eb = "stablehlo.all_reduce"(%v2844) ({
    ^bb0(%aras1b1eb: tensor<f32>, %arbs1b1eb: tensor<f32>):
      %aradds1b1eb = stablehlo.add %aras1b1eb, %arbs1b1eb : tensor<f32>
      stablehlo.return %aradds1b1eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arns1b1eb = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeans1b1eb = stablehlo.divide %arsums1b1eb, %arns1b1eb : tensor<768xf32>
    %v5161 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5162 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5163 = stablehlo.multiply %v5161, %s1b1ebm : tensor<768xf32>
    %v5164 = stablehlo.multiply %v5162, %armeans1b1eb : tensor<768xf32>
    %v5165 = stablehlo.add %v5163, %v5164 : tensor<768xf32>
    %v5166 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5167 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5168 = stablehlo.multiply %v5166, %s1b1ebv : tensor<768xf32>
    %v5169 = stablehlo.multiply %armeans1b1eb, %armeans1b1eb : tensor<768xf32>
    %v5170 = stablehlo.multiply %v5167, %v5169 : tensor<768xf32>
    %v5171 = stablehlo.add %v5168, %v5170 : tensor<768xf32>
    %v5172 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5173 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5174 = stablehlo.multiply %v5172, %s1b1ebm : tensor<768xf32>
    %v5175 = stablehlo.multiply %v5173, %armeans1b1eb : tensor<768xf32>
    %v5176 = stablehlo.add %v5174, %v5175 : tensor<768xf32>
    %v5177 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5178 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5179 = stablehlo.multiply %v5177, %s1b1ebv : tensor<768xf32>
    %v5180 = stablehlo.multiply %armeans1b1eb, %armeans1b1eb : tensor<768xf32>
    %v5181 = stablehlo.multiply %v5178, %v5180 : tensor<768xf32>
    %v5182 = stablehlo.add %v5179, %v5181 : tensor<768xf32>
    %v5183 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5184 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5185 = stablehlo.divide %v5176, %v5183 : tensor<768xf32>
    %v5186 = stablehlo.divide %v5182, %v5184 : tensor<768xf32>
    %v5187 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5188 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5189 = stablehlo.sqrt %v5186 : tensor<768xf32>
    %v5190 = stablehlo.add %v5189, %v5188 : tensor<768xf32>
    %v5191 = stablehlo.divide %v5185, %v5190 : tensor<768xf32>
    %v5192 = stablehlo.multiply %v5187, %v5191 : tensor<768xf32>
    %v5193 = stablehlo.subtract %s1b1eb, %v5192 : tensor<768xf32>
    %v5194 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5195 = stablehlo.multiply %v5194, %v5187 : tensor<768xf32>
    %v5196 = stablehlo.multiply %v5195, %s1b1eb : tensor<768xf32>
    %v5197 = stablehlo.subtract %v5193, %v5196 : tensor<768xf32>
    %arsums1b1pW = "stablehlo.all_reduce"(%v2832) ({
    ^bb0(%aras1b1pW: tensor<f32>, %arbs1b1pW: tensor<f32>):
      %aradds1b1pW = stablehlo.add %aras1b1pW, %arbs1b1pW : tensor<f32>
      stablehlo.return %aradds1b1pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192x768x1x1xf32>) -> tensor<192x768x1x1xf32>
    %arns1b1pW = stablehlo.constant dense<2.0> : tensor<192x768x1x1xf32>
    %armeans1b1pW = stablehlo.divide %arsums1b1pW, %arns1b1pW : tensor<192x768x1x1xf32>
    %v5198 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5199 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5200 = stablehlo.multiply %v5198, %s1b1pWm : tensor<192x768x1x1xf32>
    %v5201 = stablehlo.multiply %v5199, %armeans1b1pW : tensor<192x768x1x1xf32>
    %v5202 = stablehlo.add %v5200, %v5201 : tensor<192x768x1x1xf32>
    %v5203 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5204 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5205 = stablehlo.multiply %v5203, %s1b1pWv : tensor<192x768x1x1xf32>
    %v5206 = stablehlo.multiply %armeans1b1pW, %armeans1b1pW : tensor<192x768x1x1xf32>
    %v5207 = stablehlo.multiply %v5204, %v5206 : tensor<192x768x1x1xf32>
    %v5208 = stablehlo.add %v5205, %v5207 : tensor<192x768x1x1xf32>
    %v5209 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5210 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5211 = stablehlo.multiply %v5209, %s1b1pWm : tensor<192x768x1x1xf32>
    %v5212 = stablehlo.multiply %v5210, %armeans1b1pW : tensor<192x768x1x1xf32>
    %v5213 = stablehlo.add %v5211, %v5212 : tensor<192x768x1x1xf32>
    %v5214 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5215 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5216 = stablehlo.multiply %v5214, %s1b1pWv : tensor<192x768x1x1xf32>
    %v5217 = stablehlo.multiply %armeans1b1pW, %armeans1b1pW : tensor<192x768x1x1xf32>
    %v5218 = stablehlo.multiply %v5215, %v5217 : tensor<192x768x1x1xf32>
    %v5219 = stablehlo.add %v5216, %v5218 : tensor<192x768x1x1xf32>
    %v5220 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5221 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5222 = stablehlo.divide %v5213, %v5220 : tensor<192x768x1x1xf32>
    %v5223 = stablehlo.divide %v5219, %v5221 : tensor<192x768x1x1xf32>
    %v5224 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5225 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5226 = stablehlo.sqrt %v5223 : tensor<192x768x1x1xf32>
    %v5227 = stablehlo.add %v5226, %v5225 : tensor<192x768x1x1xf32>
    %v5228 = stablehlo.divide %v5222, %v5227 : tensor<192x768x1x1xf32>
    %v5229 = stablehlo.multiply %v5224, %v5228 : tensor<192x768x1x1xf32>
    %v5230 = stablehlo.subtract %s1b1pW, %v5229 : tensor<192x768x1x1xf32>
    %v5231 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5232 = stablehlo.multiply %v5231, %v5224 : tensor<192x768x1x1xf32>
    %v5233 = stablehlo.multiply %v5232, %s1b1pW : tensor<192x768x1x1xf32>
    %v5234 = stablehlo.subtract %v5230, %v5233 : tensor<192x768x1x1xf32>
    %arsums1b1pb = "stablehlo.all_reduce"(%v2835) ({
    ^bb0(%aras1b1pb: tensor<f32>, %arbs1b1pb: tensor<f32>):
      %aradds1b1pb = stablehlo.add %aras1b1pb, %arbs1b1pb : tensor<f32>
      stablehlo.return %aradds1b1pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192xf32>) -> tensor<192xf32>
    %arns1b1pb = stablehlo.constant dense<2.0> : tensor<192xf32>
    %armeans1b1pb = stablehlo.divide %arsums1b1pb, %arns1b1pb : tensor<192xf32>
    %v5235 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5236 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5237 = stablehlo.multiply %v5235, %s1b1pbm : tensor<192xf32>
    %v5238 = stablehlo.multiply %v5236, %armeans1b1pb : tensor<192xf32>
    %v5239 = stablehlo.add %v5237, %v5238 : tensor<192xf32>
    %v5240 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5241 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5242 = stablehlo.multiply %v5240, %s1b1pbv : tensor<192xf32>
    %v5243 = stablehlo.multiply %armeans1b1pb, %armeans1b1pb : tensor<192xf32>
    %v5244 = stablehlo.multiply %v5241, %v5243 : tensor<192xf32>
    %v5245 = stablehlo.add %v5242, %v5244 : tensor<192xf32>
    %v5246 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5247 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5248 = stablehlo.multiply %v5246, %s1b1pbm : tensor<192xf32>
    %v5249 = stablehlo.multiply %v5247, %armeans1b1pb : tensor<192xf32>
    %v5250 = stablehlo.add %v5248, %v5249 : tensor<192xf32>
    %v5251 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5252 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5253 = stablehlo.multiply %v5251, %s1b1pbv : tensor<192xf32>
    %v5254 = stablehlo.multiply %armeans1b1pb, %armeans1b1pb : tensor<192xf32>
    %v5255 = stablehlo.multiply %v5252, %v5254 : tensor<192xf32>
    %v5256 = stablehlo.add %v5253, %v5255 : tensor<192xf32>
    %v5257 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5258 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5259 = stablehlo.divide %v5250, %v5257 : tensor<192xf32>
    %v5260 = stablehlo.divide %v5256, %v5258 : tensor<192xf32>
    %v5261 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5262 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5263 = stablehlo.sqrt %v5260 : tensor<192xf32>
    %v5264 = stablehlo.add %v5263, %v5262 : tensor<192xf32>
    %v5265 = stablehlo.divide %v5259, %v5264 : tensor<192xf32>
    %v5266 = stablehlo.multiply %v5261, %v5265 : tensor<192xf32>
    %v5267 = stablehlo.subtract %s1b1pb, %v5266 : tensor<192xf32>
    %v5268 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5269 = stablehlo.multiply %v5268, %v5261 : tensor<192xf32>
    %v5270 = stablehlo.multiply %v5269, %s1b1pb : tensor<192xf32>
    %v5271 = stablehlo.subtract %v5267, %v5270 : tensor<192xf32>
    %arsums1b1lg = "stablehlo.all_reduce"(%v2826) ({
    ^bb0(%aras1b1lg: tensor<f32>, %arbs1b1lg: tensor<f32>):
      %aradds1b1lg = stablehlo.add %aras1b1lg, %arbs1b1lg : tensor<f32>
      stablehlo.return %aradds1b1lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192xf32>) -> tensor<192xf32>
    %arns1b1lg = stablehlo.constant dense<2.0> : tensor<192xf32>
    %armeans1b1lg = stablehlo.divide %arsums1b1lg, %arns1b1lg : tensor<192xf32>
    %v5272 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5273 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5274 = stablehlo.multiply %v5272, %s1b1lgm : tensor<192xf32>
    %v5275 = stablehlo.multiply %v5273, %armeans1b1lg : tensor<192xf32>
    %v5276 = stablehlo.add %v5274, %v5275 : tensor<192xf32>
    %v5277 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5278 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5279 = stablehlo.multiply %v5277, %s1b1lgv : tensor<192xf32>
    %v5280 = stablehlo.multiply %armeans1b1lg, %armeans1b1lg : tensor<192xf32>
    %v5281 = stablehlo.multiply %v5278, %v5280 : tensor<192xf32>
    %v5282 = stablehlo.add %v5279, %v5281 : tensor<192xf32>
    %v5283 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5284 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5285 = stablehlo.multiply %v5283, %s1b1lgm : tensor<192xf32>
    %v5286 = stablehlo.multiply %v5284, %armeans1b1lg : tensor<192xf32>
    %v5287 = stablehlo.add %v5285, %v5286 : tensor<192xf32>
    %v5288 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5289 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5290 = stablehlo.multiply %v5288, %s1b1lgv : tensor<192xf32>
    %v5291 = stablehlo.multiply %armeans1b1lg, %armeans1b1lg : tensor<192xf32>
    %v5292 = stablehlo.multiply %v5289, %v5291 : tensor<192xf32>
    %v5293 = stablehlo.add %v5290, %v5292 : tensor<192xf32>
    %v5294 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5295 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5296 = stablehlo.divide %v5287, %v5294 : tensor<192xf32>
    %v5297 = stablehlo.divide %v5293, %v5295 : tensor<192xf32>
    %v5298 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5299 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5300 = stablehlo.sqrt %v5297 : tensor<192xf32>
    %v5301 = stablehlo.add %v5300, %v5299 : tensor<192xf32>
    %v5302 = stablehlo.divide %v5296, %v5301 : tensor<192xf32>
    %v5303 = stablehlo.multiply %v5298, %v5302 : tensor<192xf32>
    %v5304 = stablehlo.subtract %s1b1lg, %v5303 : tensor<192xf32>
    %v5305 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5306 = stablehlo.multiply %v5305, %v5298 : tensor<192xf32>
    %v5307 = stablehlo.multiply %v5306, %s1b1lg : tensor<192xf32>
    %v5308 = stablehlo.subtract %v5304, %v5307 : tensor<192xf32>
    %arsums1b2dW = "stablehlo.all_reduce"(%v2749) ({
    ^bb0(%aras1b2dW: tensor<f32>, %arbs1b2dW: tensor<f32>):
      %aradds1b2dW = stablehlo.add %aras1b2dW, %arbs1b2dW : tensor<f32>
      stablehlo.return %aradds1b2dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192x1x7x7xf32>) -> tensor<192x1x7x7xf32>
    %arns1b2dW = stablehlo.constant dense<2.0> : tensor<192x1x7x7xf32>
    %armeans1b2dW = stablehlo.divide %arsums1b2dW, %arns1b2dW : tensor<192x1x7x7xf32>
    %v5309 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5310 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5311 = stablehlo.multiply %v5309, %s1b2dWm : tensor<192x1x7x7xf32>
    %v5312 = stablehlo.multiply %v5310, %armeans1b2dW : tensor<192x1x7x7xf32>
    %v5313 = stablehlo.add %v5311, %v5312 : tensor<192x1x7x7xf32>
    %v5314 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5315 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5316 = stablehlo.multiply %v5314, %s1b2dWv : tensor<192x1x7x7xf32>
    %v5317 = stablehlo.multiply %armeans1b2dW, %armeans1b2dW : tensor<192x1x7x7xf32>
    %v5318 = stablehlo.multiply %v5315, %v5317 : tensor<192x1x7x7xf32>
    %v5319 = stablehlo.add %v5316, %v5318 : tensor<192x1x7x7xf32>
    %v5320 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5321 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5322 = stablehlo.multiply %v5320, %s1b2dWm : tensor<192x1x7x7xf32>
    %v5323 = stablehlo.multiply %v5321, %armeans1b2dW : tensor<192x1x7x7xf32>
    %v5324 = stablehlo.add %v5322, %v5323 : tensor<192x1x7x7xf32>
    %v5325 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5326 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5327 = stablehlo.multiply %v5325, %s1b2dWv : tensor<192x1x7x7xf32>
    %v5328 = stablehlo.multiply %armeans1b2dW, %armeans1b2dW : tensor<192x1x7x7xf32>
    %v5329 = stablehlo.multiply %v5326, %v5328 : tensor<192x1x7x7xf32>
    %v5330 = stablehlo.add %v5327, %v5329 : tensor<192x1x7x7xf32>
    %v5331 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5332 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5333 = stablehlo.divide %v5324, %v5331 : tensor<192x1x7x7xf32>
    %v5334 = stablehlo.divide %v5330, %v5332 : tensor<192x1x7x7xf32>
    %v5335 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5336 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5337 = stablehlo.sqrt %v5334 : tensor<192x1x7x7xf32>
    %v5338 = stablehlo.add %v5337, %v5336 : tensor<192x1x7x7xf32>
    %v5339 = stablehlo.divide %v5333, %v5338 : tensor<192x1x7x7xf32>
    %v5340 = stablehlo.multiply %v5335, %v5339 : tensor<192x1x7x7xf32>
    %v5341 = stablehlo.subtract %s1b2dW, %v5340 : tensor<192x1x7x7xf32>
    %v5342 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5343 = stablehlo.multiply %v5342, %v5335 : tensor<192x1x7x7xf32>
    %v5344 = stablehlo.multiply %v5343, %s1b2dW : tensor<192x1x7x7xf32>
    %v5345 = stablehlo.subtract %v5341, %v5344 : tensor<192x1x7x7xf32>
    %arsums1b2db = "stablehlo.all_reduce"(%v2752) ({
    ^bb0(%aras1b2db: tensor<f32>, %arbs1b2db: tensor<f32>):
      %aradds1b2db = stablehlo.add %aras1b2db, %arbs1b2db : tensor<f32>
      stablehlo.return %aradds1b2db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192xf32>) -> tensor<192xf32>
    %arns1b2db = stablehlo.constant dense<2.0> : tensor<192xf32>
    %armeans1b2db = stablehlo.divide %arsums1b2db, %arns1b2db : tensor<192xf32>
    %v5346 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5347 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5348 = stablehlo.multiply %v5346, %s1b2dbm : tensor<192xf32>
    %v5349 = stablehlo.multiply %v5347, %armeans1b2db : tensor<192xf32>
    %v5350 = stablehlo.add %v5348, %v5349 : tensor<192xf32>
    %v5351 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5352 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5353 = stablehlo.multiply %v5351, %s1b2dbv : tensor<192xf32>
    %v5354 = stablehlo.multiply %armeans1b2db, %armeans1b2db : tensor<192xf32>
    %v5355 = stablehlo.multiply %v5352, %v5354 : tensor<192xf32>
    %v5356 = stablehlo.add %v5353, %v5355 : tensor<192xf32>
    %v5357 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5358 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5359 = stablehlo.multiply %v5357, %s1b2dbm : tensor<192xf32>
    %v5360 = stablehlo.multiply %v5358, %armeans1b2db : tensor<192xf32>
    %v5361 = stablehlo.add %v5359, %v5360 : tensor<192xf32>
    %v5362 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5363 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5364 = stablehlo.multiply %v5362, %s1b2dbv : tensor<192xf32>
    %v5365 = stablehlo.multiply %armeans1b2db, %armeans1b2db : tensor<192xf32>
    %v5366 = stablehlo.multiply %v5363, %v5365 : tensor<192xf32>
    %v5367 = stablehlo.add %v5364, %v5366 : tensor<192xf32>
    %v5368 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5369 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5370 = stablehlo.divide %v5361, %v5368 : tensor<192xf32>
    %v5371 = stablehlo.divide %v5367, %v5369 : tensor<192xf32>
    %v5372 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5373 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5374 = stablehlo.sqrt %v5371 : tensor<192xf32>
    %v5375 = stablehlo.add %v5374, %v5373 : tensor<192xf32>
    %v5376 = stablehlo.divide %v5370, %v5375 : tensor<192xf32>
    %v5377 = stablehlo.multiply %v5372, %v5376 : tensor<192xf32>
    %v5378 = stablehlo.subtract %s1b2db, %v5377 : tensor<192xf32>
    %v5379 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5380 = stablehlo.multiply %v5379, %v5372 : tensor<192xf32>
    %v5381 = stablehlo.multiply %v5380, %s1b2db : tensor<192xf32>
    %v5382 = stablehlo.subtract %v5378, %v5381 : tensor<192xf32>
    %arsums1b2ng = "stablehlo.all_reduce"(%v2741) ({
    ^bb0(%aras1b2ng: tensor<f32>, %arbs1b2ng: tensor<f32>):
      %aradds1b2ng = stablehlo.add %aras1b2ng, %arbs1b2ng : tensor<f32>
      stablehlo.return %aradds1b2ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns1b2ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans1b2ng = stablehlo.divide %arsums1b2ng, %arns1b2ng : tensor<f32>
    %v5383 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5384 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5385 = stablehlo.multiply %v5383, %s1b2ngm : tensor<f32>
    %v5386 = stablehlo.multiply %v5384, %armeans1b2ng : tensor<f32>
    %v5387 = stablehlo.add %v5385, %v5386 : tensor<f32>
    %v5388 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5389 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5390 = stablehlo.multiply %v5388, %s1b2ngv : tensor<f32>
    %v5391 = stablehlo.multiply %armeans1b2ng, %armeans1b2ng : tensor<f32>
    %v5392 = stablehlo.multiply %v5389, %v5391 : tensor<f32>
    %v5393 = stablehlo.add %v5390, %v5392 : tensor<f32>
    %v5394 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5395 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5396 = stablehlo.multiply %v5394, %s1b2ngm : tensor<f32>
    %v5397 = stablehlo.multiply %v5395, %armeans1b2ng : tensor<f32>
    %v5398 = stablehlo.add %v5396, %v5397 : tensor<f32>
    %v5399 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5400 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5401 = stablehlo.multiply %v5399, %s1b2ngv : tensor<f32>
    %v5402 = stablehlo.multiply %armeans1b2ng, %armeans1b2ng : tensor<f32>
    %v5403 = stablehlo.multiply %v5400, %v5402 : tensor<f32>
    %v5404 = stablehlo.add %v5401, %v5403 : tensor<f32>
    %v5405 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5406 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5407 = stablehlo.divide %v5398, %v5405 : tensor<f32>
    %v5408 = stablehlo.divide %v5404, %v5406 : tensor<f32>
    %v5409 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5410 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5411 = stablehlo.sqrt %v5408 : tensor<f32>
    %v5412 = stablehlo.add %v5411, %v5410 : tensor<f32>
    %v5413 = stablehlo.divide %v5407, %v5412 : tensor<f32>
    %v5414 = stablehlo.multiply %v5409, %v5413 : tensor<f32>
    %v5415 = stablehlo.subtract %s1b2ng, %v5414 : tensor<f32>
    %v5416 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5417 = stablehlo.multiply %v5416, %v5409 : tensor<f32>
    %v5418 = stablehlo.multiply %v5417, %s1b2ng : tensor<f32>
    %v5419 = stablehlo.subtract %v5415, %v5418 : tensor<f32>
    %arsums1b2nbt = "stablehlo.all_reduce"(%v2743) ({
    ^bb0(%aras1b2nbt: tensor<f32>, %arbs1b2nbt: tensor<f32>):
      %aradds1b2nbt = stablehlo.add %aras1b2nbt, %arbs1b2nbt : tensor<f32>
      stablehlo.return %aradds1b2nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns1b2nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans1b2nbt = stablehlo.divide %arsums1b2nbt, %arns1b2nbt : tensor<f32>
    %v5420 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5421 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5422 = stablehlo.multiply %v5420, %s1b2nbtm : tensor<f32>
    %v5423 = stablehlo.multiply %v5421, %armeans1b2nbt : tensor<f32>
    %v5424 = stablehlo.add %v5422, %v5423 : tensor<f32>
    %v5425 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5426 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5427 = stablehlo.multiply %v5425, %s1b2nbtv : tensor<f32>
    %v5428 = stablehlo.multiply %armeans1b2nbt, %armeans1b2nbt : tensor<f32>
    %v5429 = stablehlo.multiply %v5426, %v5428 : tensor<f32>
    %v5430 = stablehlo.add %v5427, %v5429 : tensor<f32>
    %v5431 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5432 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5433 = stablehlo.multiply %v5431, %s1b2nbtm : tensor<f32>
    %v5434 = stablehlo.multiply %v5432, %armeans1b2nbt : tensor<f32>
    %v5435 = stablehlo.add %v5433, %v5434 : tensor<f32>
    %v5436 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5437 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5438 = stablehlo.multiply %v5436, %s1b2nbtv : tensor<f32>
    %v5439 = stablehlo.multiply %armeans1b2nbt, %armeans1b2nbt : tensor<f32>
    %v5440 = stablehlo.multiply %v5437, %v5439 : tensor<f32>
    %v5441 = stablehlo.add %v5438, %v5440 : tensor<f32>
    %v5442 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5443 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5444 = stablehlo.divide %v5435, %v5442 : tensor<f32>
    %v5445 = stablehlo.divide %v5441, %v5443 : tensor<f32>
    %v5446 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5447 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5448 = stablehlo.sqrt %v5445 : tensor<f32>
    %v5449 = stablehlo.add %v5448, %v5447 : tensor<f32>
    %v5450 = stablehlo.divide %v5444, %v5449 : tensor<f32>
    %v5451 = stablehlo.multiply %v5446, %v5450 : tensor<f32>
    %v5452 = stablehlo.subtract %s1b2nbt, %v5451 : tensor<f32>
    %v5453 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5454 = stablehlo.multiply %v5453, %v5446 : tensor<f32>
    %v5455 = stablehlo.multiply %v5454, %s1b2nbt : tensor<f32>
    %v5456 = stablehlo.subtract %v5452, %v5455 : tensor<f32>
    %arsums1b2eW = "stablehlo.all_reduce"(%v2722) ({
    ^bb0(%aras1b2eW: tensor<f32>, %arbs1b2eW: tensor<f32>):
      %aradds1b2eW = stablehlo.add %aras1b2eW, %arbs1b2eW : tensor<f32>
      stablehlo.return %aradds1b2eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768x192x1x1xf32>) -> tensor<768x192x1x1xf32>
    %arns1b2eW = stablehlo.constant dense<2.0> : tensor<768x192x1x1xf32>
    %armeans1b2eW = stablehlo.divide %arsums1b2eW, %arns1b2eW : tensor<768x192x1x1xf32>
    %v5457 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5458 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5459 = stablehlo.multiply %v5457, %s1b2eWm : tensor<768x192x1x1xf32>
    %v5460 = stablehlo.multiply %v5458, %armeans1b2eW : tensor<768x192x1x1xf32>
    %v5461 = stablehlo.add %v5459, %v5460 : tensor<768x192x1x1xf32>
    %v5462 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5463 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5464 = stablehlo.multiply %v5462, %s1b2eWv : tensor<768x192x1x1xf32>
    %v5465 = stablehlo.multiply %armeans1b2eW, %armeans1b2eW : tensor<768x192x1x1xf32>
    %v5466 = stablehlo.multiply %v5463, %v5465 : tensor<768x192x1x1xf32>
    %v5467 = stablehlo.add %v5464, %v5466 : tensor<768x192x1x1xf32>
    %v5468 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5469 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5470 = stablehlo.multiply %v5468, %s1b2eWm : tensor<768x192x1x1xf32>
    %v5471 = stablehlo.multiply %v5469, %armeans1b2eW : tensor<768x192x1x1xf32>
    %v5472 = stablehlo.add %v5470, %v5471 : tensor<768x192x1x1xf32>
    %v5473 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5474 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5475 = stablehlo.multiply %v5473, %s1b2eWv : tensor<768x192x1x1xf32>
    %v5476 = stablehlo.multiply %armeans1b2eW, %armeans1b2eW : tensor<768x192x1x1xf32>
    %v5477 = stablehlo.multiply %v5474, %v5476 : tensor<768x192x1x1xf32>
    %v5478 = stablehlo.add %v5475, %v5477 : tensor<768x192x1x1xf32>
    %v5479 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5480 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5481 = stablehlo.divide %v5472, %v5479 : tensor<768x192x1x1xf32>
    %v5482 = stablehlo.divide %v5478, %v5480 : tensor<768x192x1x1xf32>
    %v5483 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5484 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5485 = stablehlo.sqrt %v5482 : tensor<768x192x1x1xf32>
    %v5486 = stablehlo.add %v5485, %v5484 : tensor<768x192x1x1xf32>
    %v5487 = stablehlo.divide %v5481, %v5486 : tensor<768x192x1x1xf32>
    %v5488 = stablehlo.multiply %v5483, %v5487 : tensor<768x192x1x1xf32>
    %v5489 = stablehlo.subtract %s1b2eW, %v5488 : tensor<768x192x1x1xf32>
    %v5490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5491 = stablehlo.multiply %v5490, %v5483 : tensor<768x192x1x1xf32>
    %v5492 = stablehlo.multiply %v5491, %s1b2eW : tensor<768x192x1x1xf32>
    %v5493 = stablehlo.subtract %v5489, %v5492 : tensor<768x192x1x1xf32>
    %arsums1b2eb = "stablehlo.all_reduce"(%v2725) ({
    ^bb0(%aras1b2eb: tensor<f32>, %arbs1b2eb: tensor<f32>):
      %aradds1b2eb = stablehlo.add %aras1b2eb, %arbs1b2eb : tensor<f32>
      stablehlo.return %aradds1b2eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arns1b2eb = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeans1b2eb = stablehlo.divide %arsums1b2eb, %arns1b2eb : tensor<768xf32>
    %v5494 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5495 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5496 = stablehlo.multiply %v5494, %s1b2ebm : tensor<768xf32>
    %v5497 = stablehlo.multiply %v5495, %armeans1b2eb : tensor<768xf32>
    %v5498 = stablehlo.add %v5496, %v5497 : tensor<768xf32>
    %v5499 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5500 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5501 = stablehlo.multiply %v5499, %s1b2ebv : tensor<768xf32>
    %v5502 = stablehlo.multiply %armeans1b2eb, %armeans1b2eb : tensor<768xf32>
    %v5503 = stablehlo.multiply %v5500, %v5502 : tensor<768xf32>
    %v5504 = stablehlo.add %v5501, %v5503 : tensor<768xf32>
    %v5505 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5506 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5507 = stablehlo.multiply %v5505, %s1b2ebm : tensor<768xf32>
    %v5508 = stablehlo.multiply %v5506, %armeans1b2eb : tensor<768xf32>
    %v5509 = stablehlo.add %v5507, %v5508 : tensor<768xf32>
    %v5510 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5511 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5512 = stablehlo.multiply %v5510, %s1b2ebv : tensor<768xf32>
    %v5513 = stablehlo.multiply %armeans1b2eb, %armeans1b2eb : tensor<768xf32>
    %v5514 = stablehlo.multiply %v5511, %v5513 : tensor<768xf32>
    %v5515 = stablehlo.add %v5512, %v5514 : tensor<768xf32>
    %v5516 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5517 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5518 = stablehlo.divide %v5509, %v5516 : tensor<768xf32>
    %v5519 = stablehlo.divide %v5515, %v5517 : tensor<768xf32>
    %v5520 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5521 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5522 = stablehlo.sqrt %v5519 : tensor<768xf32>
    %v5523 = stablehlo.add %v5522, %v5521 : tensor<768xf32>
    %v5524 = stablehlo.divide %v5518, %v5523 : tensor<768xf32>
    %v5525 = stablehlo.multiply %v5520, %v5524 : tensor<768xf32>
    %v5526 = stablehlo.subtract %s1b2eb, %v5525 : tensor<768xf32>
    %v5527 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5528 = stablehlo.multiply %v5527, %v5520 : tensor<768xf32>
    %v5529 = stablehlo.multiply %v5528, %s1b2eb : tensor<768xf32>
    %v5530 = stablehlo.subtract %v5526, %v5529 : tensor<768xf32>
    %arsums1b2pW = "stablehlo.all_reduce"(%v2713) ({
    ^bb0(%aras1b2pW: tensor<f32>, %arbs1b2pW: tensor<f32>):
      %aradds1b2pW = stablehlo.add %aras1b2pW, %arbs1b2pW : tensor<f32>
      stablehlo.return %aradds1b2pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192x768x1x1xf32>) -> tensor<192x768x1x1xf32>
    %arns1b2pW = stablehlo.constant dense<2.0> : tensor<192x768x1x1xf32>
    %armeans1b2pW = stablehlo.divide %arsums1b2pW, %arns1b2pW : tensor<192x768x1x1xf32>
    %v5531 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5532 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5533 = stablehlo.multiply %v5531, %s1b2pWm : tensor<192x768x1x1xf32>
    %v5534 = stablehlo.multiply %v5532, %armeans1b2pW : tensor<192x768x1x1xf32>
    %v5535 = stablehlo.add %v5533, %v5534 : tensor<192x768x1x1xf32>
    %v5536 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5537 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5538 = stablehlo.multiply %v5536, %s1b2pWv : tensor<192x768x1x1xf32>
    %v5539 = stablehlo.multiply %armeans1b2pW, %armeans1b2pW : tensor<192x768x1x1xf32>
    %v5540 = stablehlo.multiply %v5537, %v5539 : tensor<192x768x1x1xf32>
    %v5541 = stablehlo.add %v5538, %v5540 : tensor<192x768x1x1xf32>
    %v5542 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5543 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5544 = stablehlo.multiply %v5542, %s1b2pWm : tensor<192x768x1x1xf32>
    %v5545 = stablehlo.multiply %v5543, %armeans1b2pW : tensor<192x768x1x1xf32>
    %v5546 = stablehlo.add %v5544, %v5545 : tensor<192x768x1x1xf32>
    %v5547 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5548 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5549 = stablehlo.multiply %v5547, %s1b2pWv : tensor<192x768x1x1xf32>
    %v5550 = stablehlo.multiply %armeans1b2pW, %armeans1b2pW : tensor<192x768x1x1xf32>
    %v5551 = stablehlo.multiply %v5548, %v5550 : tensor<192x768x1x1xf32>
    %v5552 = stablehlo.add %v5549, %v5551 : tensor<192x768x1x1xf32>
    %v5553 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5554 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5555 = stablehlo.divide %v5546, %v5553 : tensor<192x768x1x1xf32>
    %v5556 = stablehlo.divide %v5552, %v5554 : tensor<192x768x1x1xf32>
    %v5557 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5558 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5559 = stablehlo.sqrt %v5556 : tensor<192x768x1x1xf32>
    %v5560 = stablehlo.add %v5559, %v5558 : tensor<192x768x1x1xf32>
    %v5561 = stablehlo.divide %v5555, %v5560 : tensor<192x768x1x1xf32>
    %v5562 = stablehlo.multiply %v5557, %v5561 : tensor<192x768x1x1xf32>
    %v5563 = stablehlo.subtract %s1b2pW, %v5562 : tensor<192x768x1x1xf32>
    %v5564 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5565 = stablehlo.multiply %v5564, %v5557 : tensor<192x768x1x1xf32>
    %v5566 = stablehlo.multiply %v5565, %s1b2pW : tensor<192x768x1x1xf32>
    %v5567 = stablehlo.subtract %v5563, %v5566 : tensor<192x768x1x1xf32>
    %arsums1b2pb = "stablehlo.all_reduce"(%v2716) ({
    ^bb0(%aras1b2pb: tensor<f32>, %arbs1b2pb: tensor<f32>):
      %aradds1b2pb = stablehlo.add %aras1b2pb, %arbs1b2pb : tensor<f32>
      stablehlo.return %aradds1b2pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192xf32>) -> tensor<192xf32>
    %arns1b2pb = stablehlo.constant dense<2.0> : tensor<192xf32>
    %armeans1b2pb = stablehlo.divide %arsums1b2pb, %arns1b2pb : tensor<192xf32>
    %v5568 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5569 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5570 = stablehlo.multiply %v5568, %s1b2pbm : tensor<192xf32>
    %v5571 = stablehlo.multiply %v5569, %armeans1b2pb : tensor<192xf32>
    %v5572 = stablehlo.add %v5570, %v5571 : tensor<192xf32>
    %v5573 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5574 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5575 = stablehlo.multiply %v5573, %s1b2pbv : tensor<192xf32>
    %v5576 = stablehlo.multiply %armeans1b2pb, %armeans1b2pb : tensor<192xf32>
    %v5577 = stablehlo.multiply %v5574, %v5576 : tensor<192xf32>
    %v5578 = stablehlo.add %v5575, %v5577 : tensor<192xf32>
    %v5579 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5580 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5581 = stablehlo.multiply %v5579, %s1b2pbm : tensor<192xf32>
    %v5582 = stablehlo.multiply %v5580, %armeans1b2pb : tensor<192xf32>
    %v5583 = stablehlo.add %v5581, %v5582 : tensor<192xf32>
    %v5584 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5585 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5586 = stablehlo.multiply %v5584, %s1b2pbv : tensor<192xf32>
    %v5587 = stablehlo.multiply %armeans1b2pb, %armeans1b2pb : tensor<192xf32>
    %v5588 = stablehlo.multiply %v5585, %v5587 : tensor<192xf32>
    %v5589 = stablehlo.add %v5586, %v5588 : tensor<192xf32>
    %v5590 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5591 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5592 = stablehlo.divide %v5583, %v5590 : tensor<192xf32>
    %v5593 = stablehlo.divide %v5589, %v5591 : tensor<192xf32>
    %v5594 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5595 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5596 = stablehlo.sqrt %v5593 : tensor<192xf32>
    %v5597 = stablehlo.add %v5596, %v5595 : tensor<192xf32>
    %v5598 = stablehlo.divide %v5592, %v5597 : tensor<192xf32>
    %v5599 = stablehlo.multiply %v5594, %v5598 : tensor<192xf32>
    %v5600 = stablehlo.subtract %s1b2pb, %v5599 : tensor<192xf32>
    %v5601 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5602 = stablehlo.multiply %v5601, %v5594 : tensor<192xf32>
    %v5603 = stablehlo.multiply %v5602, %s1b2pb : tensor<192xf32>
    %v5604 = stablehlo.subtract %v5600, %v5603 : tensor<192xf32>
    %arsums1b2lg = "stablehlo.all_reduce"(%v2707) ({
    ^bb0(%aras1b2lg: tensor<f32>, %arbs1b2lg: tensor<f32>):
      %aradds1b2lg = stablehlo.add %aras1b2lg, %arbs1b2lg : tensor<f32>
      stablehlo.return %aradds1b2lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<192xf32>) -> tensor<192xf32>
    %arns1b2lg = stablehlo.constant dense<2.0> : tensor<192xf32>
    %armeans1b2lg = stablehlo.divide %arsums1b2lg, %arns1b2lg : tensor<192xf32>
    %v5605 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5606 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5607 = stablehlo.multiply %v5605, %s1b2lgm : tensor<192xf32>
    %v5608 = stablehlo.multiply %v5606, %armeans1b2lg : tensor<192xf32>
    %v5609 = stablehlo.add %v5607, %v5608 : tensor<192xf32>
    %v5610 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5611 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5612 = stablehlo.multiply %v5610, %s1b2lgv : tensor<192xf32>
    %v5613 = stablehlo.multiply %armeans1b2lg, %armeans1b2lg : tensor<192xf32>
    %v5614 = stablehlo.multiply %v5611, %v5613 : tensor<192xf32>
    %v5615 = stablehlo.add %v5612, %v5614 : tensor<192xf32>
    %v5616 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5617 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5618 = stablehlo.multiply %v5616, %s1b2lgm : tensor<192xf32>
    %v5619 = stablehlo.multiply %v5617, %armeans1b2lg : tensor<192xf32>
    %v5620 = stablehlo.add %v5618, %v5619 : tensor<192xf32>
    %v5621 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5622 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5623 = stablehlo.multiply %v5621, %s1b2lgv : tensor<192xf32>
    %v5624 = stablehlo.multiply %armeans1b2lg, %armeans1b2lg : tensor<192xf32>
    %v5625 = stablehlo.multiply %v5622, %v5624 : tensor<192xf32>
    %v5626 = stablehlo.add %v5623, %v5625 : tensor<192xf32>
    %v5627 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5628 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5629 = stablehlo.divide %v5620, %v5627 : tensor<192xf32>
    %v5630 = stablehlo.divide %v5626, %v5628 : tensor<192xf32>
    %v5631 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5632 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5633 = stablehlo.sqrt %v5630 : tensor<192xf32>
    %v5634 = stablehlo.add %v5633, %v5632 : tensor<192xf32>
    %v5635 = stablehlo.divide %v5629, %v5634 : tensor<192xf32>
    %v5636 = stablehlo.multiply %v5631, %v5635 : tensor<192xf32>
    %v5637 = stablehlo.subtract %s1b2lg, %v5636 : tensor<192xf32>
    %v5638 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5639 = stablehlo.multiply %v5638, %v5631 : tensor<192xf32>
    %v5640 = stablehlo.multiply %v5639, %s1b2lg : tensor<192xf32>
    %v5641 = stablehlo.subtract %v5637, %v5640 : tensor<192xf32>
    %arsumd1ng = "stablehlo.all_reduce"(%v2623) ({
    ^bb0(%arad1ng: tensor<f32>, %arbd1ng: tensor<f32>):
      %araddd1ng = stablehlo.add %arad1ng, %arbd1ng : tensor<f32>
      stablehlo.return %araddd1ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arnd1ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeand1ng = stablehlo.divide %arsumd1ng, %arnd1ng : tensor<f32>
    %v5642 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5643 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5644 = stablehlo.multiply %v5642, %d1ngm : tensor<f32>
    %v5645 = stablehlo.multiply %v5643, %armeand1ng : tensor<f32>
    %v5646 = stablehlo.add %v5644, %v5645 : tensor<f32>
    %v5647 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5648 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5649 = stablehlo.multiply %v5647, %d1ngv : tensor<f32>
    %v5650 = stablehlo.multiply %armeand1ng, %armeand1ng : tensor<f32>
    %v5651 = stablehlo.multiply %v5648, %v5650 : tensor<f32>
    %v5652 = stablehlo.add %v5649, %v5651 : tensor<f32>
    %v5653 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5654 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5655 = stablehlo.multiply %v5653, %d1ngm : tensor<f32>
    %v5656 = stablehlo.multiply %v5654, %armeand1ng : tensor<f32>
    %v5657 = stablehlo.add %v5655, %v5656 : tensor<f32>
    %v5658 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5659 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5660 = stablehlo.multiply %v5658, %d1ngv : tensor<f32>
    %v5661 = stablehlo.multiply %armeand1ng, %armeand1ng : tensor<f32>
    %v5662 = stablehlo.multiply %v5659, %v5661 : tensor<f32>
    %v5663 = stablehlo.add %v5660, %v5662 : tensor<f32>
    %v5664 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5665 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5666 = stablehlo.divide %v5657, %v5664 : tensor<f32>
    %v5667 = stablehlo.divide %v5663, %v5665 : tensor<f32>
    %v5668 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5669 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5670 = stablehlo.sqrt %v5667 : tensor<f32>
    %v5671 = stablehlo.add %v5670, %v5669 : tensor<f32>
    %v5672 = stablehlo.divide %v5666, %v5671 : tensor<f32>
    %v5673 = stablehlo.multiply %v5668, %v5672 : tensor<f32>
    %v5674 = stablehlo.subtract %d1ng, %v5673 : tensor<f32>
    %v5675 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5676 = stablehlo.multiply %v5675, %v5668 : tensor<f32>
    %v5677 = stablehlo.multiply %v5676, %d1ng : tensor<f32>
    %v5678 = stablehlo.subtract %v5674, %v5677 : tensor<f32>
    %arsumd1nbt = "stablehlo.all_reduce"(%v2625) ({
    ^bb0(%arad1nbt: tensor<f32>, %arbd1nbt: tensor<f32>):
      %araddd1nbt = stablehlo.add %arad1nbt, %arbd1nbt : tensor<f32>
      stablehlo.return %araddd1nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arnd1nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeand1nbt = stablehlo.divide %arsumd1nbt, %arnd1nbt : tensor<f32>
    %v5679 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5680 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5681 = stablehlo.multiply %v5679, %d1nbtm : tensor<f32>
    %v5682 = stablehlo.multiply %v5680, %armeand1nbt : tensor<f32>
    %v5683 = stablehlo.add %v5681, %v5682 : tensor<f32>
    %v5684 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5685 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5686 = stablehlo.multiply %v5684, %d1nbtv : tensor<f32>
    %v5687 = stablehlo.multiply %armeand1nbt, %armeand1nbt : tensor<f32>
    %v5688 = stablehlo.multiply %v5685, %v5687 : tensor<f32>
    %v5689 = stablehlo.add %v5686, %v5688 : tensor<f32>
    %v5690 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5691 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5692 = stablehlo.multiply %v5690, %d1nbtm : tensor<f32>
    %v5693 = stablehlo.multiply %v5691, %armeand1nbt : tensor<f32>
    %v5694 = stablehlo.add %v5692, %v5693 : tensor<f32>
    %v5695 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5696 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5697 = stablehlo.multiply %v5695, %d1nbtv : tensor<f32>
    %v5698 = stablehlo.multiply %armeand1nbt, %armeand1nbt : tensor<f32>
    %v5699 = stablehlo.multiply %v5696, %v5698 : tensor<f32>
    %v5700 = stablehlo.add %v5697, %v5699 : tensor<f32>
    %v5701 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5702 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5703 = stablehlo.divide %v5694, %v5701 : tensor<f32>
    %v5704 = stablehlo.divide %v5700, %v5702 : tensor<f32>
    %v5705 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5706 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5707 = stablehlo.sqrt %v5704 : tensor<f32>
    %v5708 = stablehlo.add %v5707, %v5706 : tensor<f32>
    %v5709 = stablehlo.divide %v5703, %v5708 : tensor<f32>
    %v5710 = stablehlo.multiply %v5705, %v5709 : tensor<f32>
    %v5711 = stablehlo.subtract %d1nbt, %v5710 : tensor<f32>
    %v5712 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5713 = stablehlo.multiply %v5712, %v5705 : tensor<f32>
    %v5714 = stablehlo.multiply %v5713, %d1nbt : tensor<f32>
    %v5715 = stablehlo.subtract %v5711, %v5714 : tensor<f32>
    %arsumd1W = "stablehlo.all_reduce"(%v2633) ({
    ^bb0(%arad1W: tensor<f32>, %arbd1W: tensor<f32>):
      %araddd1W = stablehlo.add %arad1W, %arbd1W : tensor<f32>
      stablehlo.return %araddd1W : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x192x2x2xf32>) -> tensor<384x192x2x2xf32>
    %arnd1W = stablehlo.constant dense<2.0> : tensor<384x192x2x2xf32>
    %armeand1W = stablehlo.divide %arsumd1W, %arnd1W : tensor<384x192x2x2xf32>
    %v5716 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5717 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5718 = stablehlo.multiply %v5716, %d1Wm : tensor<384x192x2x2xf32>
    %v5719 = stablehlo.multiply %v5717, %armeand1W : tensor<384x192x2x2xf32>
    %v5720 = stablehlo.add %v5718, %v5719 : tensor<384x192x2x2xf32>
    %v5721 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5722 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5723 = stablehlo.multiply %v5721, %d1Wv : tensor<384x192x2x2xf32>
    %v5724 = stablehlo.multiply %armeand1W, %armeand1W : tensor<384x192x2x2xf32>
    %v5725 = stablehlo.multiply %v5722, %v5724 : tensor<384x192x2x2xf32>
    %v5726 = stablehlo.add %v5723, %v5725 : tensor<384x192x2x2xf32>
    %v5727 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5728 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5729 = stablehlo.multiply %v5727, %d1Wm : tensor<384x192x2x2xf32>
    %v5730 = stablehlo.multiply %v5728, %armeand1W : tensor<384x192x2x2xf32>
    %v5731 = stablehlo.add %v5729, %v5730 : tensor<384x192x2x2xf32>
    %v5732 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5733 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5734 = stablehlo.multiply %v5732, %d1Wv : tensor<384x192x2x2xf32>
    %v5735 = stablehlo.multiply %armeand1W, %armeand1W : tensor<384x192x2x2xf32>
    %v5736 = stablehlo.multiply %v5733, %v5735 : tensor<384x192x2x2xf32>
    %v5737 = stablehlo.add %v5734, %v5736 : tensor<384x192x2x2xf32>
    %v5738 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5739 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5740 = stablehlo.divide %v5731, %v5738 : tensor<384x192x2x2xf32>
    %v5741 = stablehlo.divide %v5737, %v5739 : tensor<384x192x2x2xf32>
    %v5742 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5743 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5744 = stablehlo.sqrt %v5741 : tensor<384x192x2x2xf32>
    %v5745 = stablehlo.add %v5744, %v5743 : tensor<384x192x2x2xf32>
    %v5746 = stablehlo.divide %v5740, %v5745 : tensor<384x192x2x2xf32>
    %v5747 = stablehlo.multiply %v5742, %v5746 : tensor<384x192x2x2xf32>
    %v5748 = stablehlo.subtract %d1W, %v5747 : tensor<384x192x2x2xf32>
    %v5749 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5750 = stablehlo.multiply %v5749, %v5742 : tensor<384x192x2x2xf32>
    %v5751 = stablehlo.multiply %v5750, %d1W : tensor<384x192x2x2xf32>
    %v5752 = stablehlo.subtract %v5748, %v5751 : tensor<384x192x2x2xf32>
    %arsumd1b = "stablehlo.all_reduce"(%v2607) ({
    ^bb0(%arad1b: tensor<f32>, %arbd1b: tensor<f32>):
      %araddd1b = stablehlo.add %arad1b, %arbd1b : tensor<f32>
      stablehlo.return %araddd1b : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arnd1b = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeand1b = stablehlo.divide %arsumd1b, %arnd1b : tensor<384xf32>
    %v5753 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5754 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5755 = stablehlo.multiply %v5753, %d1bm : tensor<384xf32>
    %v5756 = stablehlo.multiply %v5754, %armeand1b : tensor<384xf32>
    %v5757 = stablehlo.add %v5755, %v5756 : tensor<384xf32>
    %v5758 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5759 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5760 = stablehlo.multiply %v5758, %d1bv : tensor<384xf32>
    %v5761 = stablehlo.multiply %armeand1b, %armeand1b : tensor<384xf32>
    %v5762 = stablehlo.multiply %v5759, %v5761 : tensor<384xf32>
    %v5763 = stablehlo.add %v5760, %v5762 : tensor<384xf32>
    %v5764 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5765 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5766 = stablehlo.multiply %v5764, %d1bm : tensor<384xf32>
    %v5767 = stablehlo.multiply %v5765, %armeand1b : tensor<384xf32>
    %v5768 = stablehlo.add %v5766, %v5767 : tensor<384xf32>
    %v5769 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5770 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5771 = stablehlo.multiply %v5769, %d1bv : tensor<384xf32>
    %v5772 = stablehlo.multiply %armeand1b, %armeand1b : tensor<384xf32>
    %v5773 = stablehlo.multiply %v5770, %v5772 : tensor<384xf32>
    %v5774 = stablehlo.add %v5771, %v5773 : tensor<384xf32>
    %v5775 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5776 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5777 = stablehlo.divide %v5768, %v5775 : tensor<384xf32>
    %v5778 = stablehlo.divide %v5774, %v5776 : tensor<384xf32>
    %v5779 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5780 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5781 = stablehlo.sqrt %v5778 : tensor<384xf32>
    %v5782 = stablehlo.add %v5781, %v5780 : tensor<384xf32>
    %v5783 = stablehlo.divide %v5777, %v5782 : tensor<384xf32>
    %v5784 = stablehlo.multiply %v5779, %v5783 : tensor<384xf32>
    %v5785 = stablehlo.subtract %d1b, %v5784 : tensor<384xf32>
    %v5786 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5787 = stablehlo.multiply %v5786, %v5779 : tensor<384xf32>
    %v5788 = stablehlo.multiply %v5787, %d1b : tensor<384xf32>
    %v5789 = stablehlo.subtract %v5785, %v5788 : tensor<384xf32>
    %arsums2b0dW = "stablehlo.all_reduce"(%v2567) ({
    ^bb0(%aras2b0dW: tensor<f32>, %arbs2b0dW: tensor<f32>):
      %aradds2b0dW = stablehlo.add %aras2b0dW, %arbs2b0dW : tensor<f32>
      stablehlo.return %aradds2b0dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1x7x7xf32>) -> tensor<384x1x7x7xf32>
    %arns2b0dW = stablehlo.constant dense<2.0> : tensor<384x1x7x7xf32>
    %armeans2b0dW = stablehlo.divide %arsums2b0dW, %arns2b0dW : tensor<384x1x7x7xf32>
    %v5790 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5791 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5792 = stablehlo.multiply %v5790, %s2b0dWm : tensor<384x1x7x7xf32>
    %v5793 = stablehlo.multiply %v5791, %armeans2b0dW : tensor<384x1x7x7xf32>
    %v5794 = stablehlo.add %v5792, %v5793 : tensor<384x1x7x7xf32>
    %v5795 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5796 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5797 = stablehlo.multiply %v5795, %s2b0dWv : tensor<384x1x7x7xf32>
    %v5798 = stablehlo.multiply %armeans2b0dW, %armeans2b0dW : tensor<384x1x7x7xf32>
    %v5799 = stablehlo.multiply %v5796, %v5798 : tensor<384x1x7x7xf32>
    %v5800 = stablehlo.add %v5797, %v5799 : tensor<384x1x7x7xf32>
    %v5801 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5802 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5803 = stablehlo.multiply %v5801, %s2b0dWm : tensor<384x1x7x7xf32>
    %v5804 = stablehlo.multiply %v5802, %armeans2b0dW : tensor<384x1x7x7xf32>
    %v5805 = stablehlo.add %v5803, %v5804 : tensor<384x1x7x7xf32>
    %v5806 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5807 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5808 = stablehlo.multiply %v5806, %s2b0dWv : tensor<384x1x7x7xf32>
    %v5809 = stablehlo.multiply %armeans2b0dW, %armeans2b0dW : tensor<384x1x7x7xf32>
    %v5810 = stablehlo.multiply %v5807, %v5809 : tensor<384x1x7x7xf32>
    %v5811 = stablehlo.add %v5808, %v5810 : tensor<384x1x7x7xf32>
    %v5812 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5813 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5814 = stablehlo.divide %v5805, %v5812 : tensor<384x1x7x7xf32>
    %v5815 = stablehlo.divide %v5811, %v5813 : tensor<384x1x7x7xf32>
    %v5816 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5817 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5818 = stablehlo.sqrt %v5815 : tensor<384x1x7x7xf32>
    %v5819 = stablehlo.add %v5818, %v5817 : tensor<384x1x7x7xf32>
    %v5820 = stablehlo.divide %v5814, %v5819 : tensor<384x1x7x7xf32>
    %v5821 = stablehlo.multiply %v5816, %v5820 : tensor<384x1x7x7xf32>
    %v5822 = stablehlo.subtract %s2b0dW, %v5821 : tensor<384x1x7x7xf32>
    %v5823 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5824 = stablehlo.multiply %v5823, %v5816 : tensor<384x1x7x7xf32>
    %v5825 = stablehlo.multiply %v5824, %s2b0dW : tensor<384x1x7x7xf32>
    %v5826 = stablehlo.subtract %v5822, %v5825 : tensor<384x1x7x7xf32>
    %arsums2b0db = "stablehlo.all_reduce"(%v2570) ({
    ^bb0(%aras2b0db: tensor<f32>, %arbs2b0db: tensor<f32>):
      %aradds2b0db = stablehlo.add %aras2b0db, %arbs2b0db : tensor<f32>
      stablehlo.return %aradds2b0db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b0db = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b0db = stablehlo.divide %arsums2b0db, %arns2b0db : tensor<384xf32>
    %v5827 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5828 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5829 = stablehlo.multiply %v5827, %s2b0dbm : tensor<384xf32>
    %v5830 = stablehlo.multiply %v5828, %armeans2b0db : tensor<384xf32>
    %v5831 = stablehlo.add %v5829, %v5830 : tensor<384xf32>
    %v5832 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5833 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5834 = stablehlo.multiply %v5832, %s2b0dbv : tensor<384xf32>
    %v5835 = stablehlo.multiply %armeans2b0db, %armeans2b0db : tensor<384xf32>
    %v5836 = stablehlo.multiply %v5833, %v5835 : tensor<384xf32>
    %v5837 = stablehlo.add %v5834, %v5836 : tensor<384xf32>
    %v5838 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5839 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5840 = stablehlo.multiply %v5838, %s2b0dbm : tensor<384xf32>
    %v5841 = stablehlo.multiply %v5839, %armeans2b0db : tensor<384xf32>
    %v5842 = stablehlo.add %v5840, %v5841 : tensor<384xf32>
    %v5843 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5844 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5845 = stablehlo.multiply %v5843, %s2b0dbv : tensor<384xf32>
    %v5846 = stablehlo.multiply %armeans2b0db, %armeans2b0db : tensor<384xf32>
    %v5847 = stablehlo.multiply %v5844, %v5846 : tensor<384xf32>
    %v5848 = stablehlo.add %v5845, %v5847 : tensor<384xf32>
    %v5849 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5850 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5851 = stablehlo.divide %v5842, %v5849 : tensor<384xf32>
    %v5852 = stablehlo.divide %v5848, %v5850 : tensor<384xf32>
    %v5853 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5854 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5855 = stablehlo.sqrt %v5852 : tensor<384xf32>
    %v5856 = stablehlo.add %v5855, %v5854 : tensor<384xf32>
    %v5857 = stablehlo.divide %v5851, %v5856 : tensor<384xf32>
    %v5858 = stablehlo.multiply %v5853, %v5857 : tensor<384xf32>
    %v5859 = stablehlo.subtract %s2b0db, %v5858 : tensor<384xf32>
    %v5860 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5861 = stablehlo.multiply %v5860, %v5853 : tensor<384xf32>
    %v5862 = stablehlo.multiply %v5861, %s2b0db : tensor<384xf32>
    %v5863 = stablehlo.subtract %v5859, %v5862 : tensor<384xf32>
    %arsums2b0ng = "stablehlo.all_reduce"(%v2559) ({
    ^bb0(%aras2b0ng: tensor<f32>, %arbs2b0ng: tensor<f32>):
      %aradds2b0ng = stablehlo.add %aras2b0ng, %arbs2b0ng : tensor<f32>
      stablehlo.return %aradds2b0ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b0ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b0ng = stablehlo.divide %arsums2b0ng, %arns2b0ng : tensor<f32>
    %v5864 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5865 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5866 = stablehlo.multiply %v5864, %s2b0ngm : tensor<f32>
    %v5867 = stablehlo.multiply %v5865, %armeans2b0ng : tensor<f32>
    %v5868 = stablehlo.add %v5866, %v5867 : tensor<f32>
    %v5869 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5870 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5871 = stablehlo.multiply %v5869, %s2b0ngv : tensor<f32>
    %v5872 = stablehlo.multiply %armeans2b0ng, %armeans2b0ng : tensor<f32>
    %v5873 = stablehlo.multiply %v5870, %v5872 : tensor<f32>
    %v5874 = stablehlo.add %v5871, %v5873 : tensor<f32>
    %v5875 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5876 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5877 = stablehlo.multiply %v5875, %s2b0ngm : tensor<f32>
    %v5878 = stablehlo.multiply %v5876, %armeans2b0ng : tensor<f32>
    %v5879 = stablehlo.add %v5877, %v5878 : tensor<f32>
    %v5880 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5881 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5882 = stablehlo.multiply %v5880, %s2b0ngv : tensor<f32>
    %v5883 = stablehlo.multiply %armeans2b0ng, %armeans2b0ng : tensor<f32>
    %v5884 = stablehlo.multiply %v5881, %v5883 : tensor<f32>
    %v5885 = stablehlo.add %v5882, %v5884 : tensor<f32>
    %v5886 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5887 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5888 = stablehlo.divide %v5879, %v5886 : tensor<f32>
    %v5889 = stablehlo.divide %v5885, %v5887 : tensor<f32>
    %v5890 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5891 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5892 = stablehlo.sqrt %v5889 : tensor<f32>
    %v5893 = stablehlo.add %v5892, %v5891 : tensor<f32>
    %v5894 = stablehlo.divide %v5888, %v5893 : tensor<f32>
    %v5895 = stablehlo.multiply %v5890, %v5894 : tensor<f32>
    %v5896 = stablehlo.subtract %s2b0ng, %v5895 : tensor<f32>
    %v5897 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5898 = stablehlo.multiply %v5897, %v5890 : tensor<f32>
    %v5899 = stablehlo.multiply %v5898, %s2b0ng : tensor<f32>
    %v5900 = stablehlo.subtract %v5896, %v5899 : tensor<f32>
    %arsums2b0nbt = "stablehlo.all_reduce"(%v2561) ({
    ^bb0(%aras2b0nbt: tensor<f32>, %arbs2b0nbt: tensor<f32>):
      %aradds2b0nbt = stablehlo.add %aras2b0nbt, %arbs2b0nbt : tensor<f32>
      stablehlo.return %aradds2b0nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b0nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b0nbt = stablehlo.divide %arsums2b0nbt, %arns2b0nbt : tensor<f32>
    %v5901 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5902 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5903 = stablehlo.multiply %v5901, %s2b0nbtm : tensor<f32>
    %v5904 = stablehlo.multiply %v5902, %armeans2b0nbt : tensor<f32>
    %v5905 = stablehlo.add %v5903, %v5904 : tensor<f32>
    %v5906 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5907 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5908 = stablehlo.multiply %v5906, %s2b0nbtv : tensor<f32>
    %v5909 = stablehlo.multiply %armeans2b0nbt, %armeans2b0nbt : tensor<f32>
    %v5910 = stablehlo.multiply %v5907, %v5909 : tensor<f32>
    %v5911 = stablehlo.add %v5908, %v5910 : tensor<f32>
    %v5912 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5913 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5914 = stablehlo.multiply %v5912, %s2b0nbtm : tensor<f32>
    %v5915 = stablehlo.multiply %v5913, %armeans2b0nbt : tensor<f32>
    %v5916 = stablehlo.add %v5914, %v5915 : tensor<f32>
    %v5917 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5918 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5919 = stablehlo.multiply %v5917, %s2b0nbtv : tensor<f32>
    %v5920 = stablehlo.multiply %armeans2b0nbt, %armeans2b0nbt : tensor<f32>
    %v5921 = stablehlo.multiply %v5918, %v5920 : tensor<f32>
    %v5922 = stablehlo.add %v5919, %v5921 : tensor<f32>
    %v5923 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5924 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5925 = stablehlo.divide %v5916, %v5923 : tensor<f32>
    %v5926 = stablehlo.divide %v5922, %v5924 : tensor<f32>
    %v5927 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5928 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5929 = stablehlo.sqrt %v5926 : tensor<f32>
    %v5930 = stablehlo.add %v5929, %v5928 : tensor<f32>
    %v5931 = stablehlo.divide %v5925, %v5930 : tensor<f32>
    %v5932 = stablehlo.multiply %v5927, %v5931 : tensor<f32>
    %v5933 = stablehlo.subtract %s2b0nbt, %v5932 : tensor<f32>
    %v5934 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5935 = stablehlo.multiply %v5934, %v5927 : tensor<f32>
    %v5936 = stablehlo.multiply %v5935, %s2b0nbt : tensor<f32>
    %v5937 = stablehlo.subtract %v5933, %v5936 : tensor<f32>
    %arsums2b0eW = "stablehlo.all_reduce"(%v2540) ({
    ^bb0(%aras2b0eW: tensor<f32>, %arbs2b0eW: tensor<f32>):
      %aradds2b0eW = stablehlo.add %aras2b0eW, %arbs2b0eW : tensor<f32>
      stablehlo.return %aradds2b0eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536x384x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %arns2b0eW = stablehlo.constant dense<2.0> : tensor<1536x384x1x1xf32>
    %armeans2b0eW = stablehlo.divide %arsums2b0eW, %arns2b0eW : tensor<1536x384x1x1xf32>
    %v5938 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5939 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5940 = stablehlo.multiply %v5938, %s2b0eWm : tensor<1536x384x1x1xf32>
    %v5941 = stablehlo.multiply %v5939, %armeans2b0eW : tensor<1536x384x1x1xf32>
    %v5942 = stablehlo.add %v5940, %v5941 : tensor<1536x384x1x1xf32>
    %v5943 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5944 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5945 = stablehlo.multiply %v5943, %s2b0eWv : tensor<1536x384x1x1xf32>
    %v5946 = stablehlo.multiply %armeans2b0eW, %armeans2b0eW : tensor<1536x384x1x1xf32>
    %v5947 = stablehlo.multiply %v5944, %v5946 : tensor<1536x384x1x1xf32>
    %v5948 = stablehlo.add %v5945, %v5947 : tensor<1536x384x1x1xf32>
    %v5949 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5950 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5951 = stablehlo.multiply %v5949, %s2b0eWm : tensor<1536x384x1x1xf32>
    %v5952 = stablehlo.multiply %v5950, %armeans2b0eW : tensor<1536x384x1x1xf32>
    %v5953 = stablehlo.add %v5951, %v5952 : tensor<1536x384x1x1xf32>
    %v5954 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5955 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5956 = stablehlo.multiply %v5954, %s2b0eWv : tensor<1536x384x1x1xf32>
    %v5957 = stablehlo.multiply %armeans2b0eW, %armeans2b0eW : tensor<1536x384x1x1xf32>
    %v5958 = stablehlo.multiply %v5955, %v5957 : tensor<1536x384x1x1xf32>
    %v5959 = stablehlo.add %v5956, %v5958 : tensor<1536x384x1x1xf32>
    %v5960 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5961 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5962 = stablehlo.divide %v5953, %v5960 : tensor<1536x384x1x1xf32>
    %v5963 = stablehlo.divide %v5959, %v5961 : tensor<1536x384x1x1xf32>
    %v5964 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5965 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5966 = stablehlo.sqrt %v5963 : tensor<1536x384x1x1xf32>
    %v5967 = stablehlo.add %v5966, %v5965 : tensor<1536x384x1x1xf32>
    %v5968 = stablehlo.divide %v5962, %v5967 : tensor<1536x384x1x1xf32>
    %v5969 = stablehlo.multiply %v5964, %v5968 : tensor<1536x384x1x1xf32>
    %v5970 = stablehlo.subtract %s2b0eW, %v5969 : tensor<1536x384x1x1xf32>
    %v5971 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5972 = stablehlo.multiply %v5971, %v5964 : tensor<1536x384x1x1xf32>
    %v5973 = stablehlo.multiply %v5972, %s2b0eW : tensor<1536x384x1x1xf32>
    %v5974 = stablehlo.subtract %v5970, %v5973 : tensor<1536x384x1x1xf32>
    %arsums2b0eb = "stablehlo.all_reduce"(%v2543) ({
    ^bb0(%aras2b0eb: tensor<f32>, %arbs2b0eb: tensor<f32>):
      %aradds2b0eb = stablehlo.add %aras2b0eb, %arbs2b0eb : tensor<f32>
      stablehlo.return %aradds2b0eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536xf32>) -> tensor<1536xf32>
    %arns2b0eb = stablehlo.constant dense<2.0> : tensor<1536xf32>
    %armeans2b0eb = stablehlo.divide %arsums2b0eb, %arns2b0eb : tensor<1536xf32>
    %v5975 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5976 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5977 = stablehlo.multiply %v5975, %s2b0ebm : tensor<1536xf32>
    %v5978 = stablehlo.multiply %v5976, %armeans2b0eb : tensor<1536xf32>
    %v5979 = stablehlo.add %v5977, %v5978 : tensor<1536xf32>
    %v5980 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5981 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5982 = stablehlo.multiply %v5980, %s2b0ebv : tensor<1536xf32>
    %v5983 = stablehlo.multiply %armeans2b0eb, %armeans2b0eb : tensor<1536xf32>
    %v5984 = stablehlo.multiply %v5981, %v5983 : tensor<1536xf32>
    %v5985 = stablehlo.add %v5982, %v5984 : tensor<1536xf32>
    %v5986 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5987 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5988 = stablehlo.multiply %v5986, %s2b0ebm : tensor<1536xf32>
    %v5989 = stablehlo.multiply %v5987, %armeans2b0eb : tensor<1536xf32>
    %v5990 = stablehlo.add %v5988, %v5989 : tensor<1536xf32>
    %v5991 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5992 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5993 = stablehlo.multiply %v5991, %s2b0ebv : tensor<1536xf32>
    %v5994 = stablehlo.multiply %armeans2b0eb, %armeans2b0eb : tensor<1536xf32>
    %v5995 = stablehlo.multiply %v5992, %v5994 : tensor<1536xf32>
    %v5996 = stablehlo.add %v5993, %v5995 : tensor<1536xf32>
    %v5997 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5998 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5999 = stablehlo.divide %v5990, %v5997 : tensor<1536xf32>
    %v6000 = stablehlo.divide %v5996, %v5998 : tensor<1536xf32>
    %v6001 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6002 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6003 = stablehlo.sqrt %v6000 : tensor<1536xf32>
    %v6004 = stablehlo.add %v6003, %v6002 : tensor<1536xf32>
    %v6005 = stablehlo.divide %v5999, %v6004 : tensor<1536xf32>
    %v6006 = stablehlo.multiply %v6001, %v6005 : tensor<1536xf32>
    %v6007 = stablehlo.subtract %s2b0eb, %v6006 : tensor<1536xf32>
    %v6008 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6009 = stablehlo.multiply %v6008, %v6001 : tensor<1536xf32>
    %v6010 = stablehlo.multiply %v6009, %s2b0eb : tensor<1536xf32>
    %v6011 = stablehlo.subtract %v6007, %v6010 : tensor<1536xf32>
    %arsums2b0pW = "stablehlo.all_reduce"(%v2531) ({
    ^bb0(%aras2b0pW: tensor<f32>, %arbs2b0pW: tensor<f32>):
      %aradds2b0pW = stablehlo.add %aras2b0pW, %arbs2b0pW : tensor<f32>
      stablehlo.return %aradds2b0pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1536x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %arns2b0pW = stablehlo.constant dense<2.0> : tensor<384x1536x1x1xf32>
    %armeans2b0pW = stablehlo.divide %arsums2b0pW, %arns2b0pW : tensor<384x1536x1x1xf32>
    %v6012 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6013 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6014 = stablehlo.multiply %v6012, %s2b0pWm : tensor<384x1536x1x1xf32>
    %v6015 = stablehlo.multiply %v6013, %armeans2b0pW : tensor<384x1536x1x1xf32>
    %v6016 = stablehlo.add %v6014, %v6015 : tensor<384x1536x1x1xf32>
    %v6017 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6018 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6019 = stablehlo.multiply %v6017, %s2b0pWv : tensor<384x1536x1x1xf32>
    %v6020 = stablehlo.multiply %armeans2b0pW, %armeans2b0pW : tensor<384x1536x1x1xf32>
    %v6021 = stablehlo.multiply %v6018, %v6020 : tensor<384x1536x1x1xf32>
    %v6022 = stablehlo.add %v6019, %v6021 : tensor<384x1536x1x1xf32>
    %v6023 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6024 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6025 = stablehlo.multiply %v6023, %s2b0pWm : tensor<384x1536x1x1xf32>
    %v6026 = stablehlo.multiply %v6024, %armeans2b0pW : tensor<384x1536x1x1xf32>
    %v6027 = stablehlo.add %v6025, %v6026 : tensor<384x1536x1x1xf32>
    %v6028 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6029 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6030 = stablehlo.multiply %v6028, %s2b0pWv : tensor<384x1536x1x1xf32>
    %v6031 = stablehlo.multiply %armeans2b0pW, %armeans2b0pW : tensor<384x1536x1x1xf32>
    %v6032 = stablehlo.multiply %v6029, %v6031 : tensor<384x1536x1x1xf32>
    %v6033 = stablehlo.add %v6030, %v6032 : tensor<384x1536x1x1xf32>
    %v6034 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6035 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6036 = stablehlo.divide %v6027, %v6034 : tensor<384x1536x1x1xf32>
    %v6037 = stablehlo.divide %v6033, %v6035 : tensor<384x1536x1x1xf32>
    %v6038 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6039 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6040 = stablehlo.sqrt %v6037 : tensor<384x1536x1x1xf32>
    %v6041 = stablehlo.add %v6040, %v6039 : tensor<384x1536x1x1xf32>
    %v6042 = stablehlo.divide %v6036, %v6041 : tensor<384x1536x1x1xf32>
    %v6043 = stablehlo.multiply %v6038, %v6042 : tensor<384x1536x1x1xf32>
    %v6044 = stablehlo.subtract %s2b0pW, %v6043 : tensor<384x1536x1x1xf32>
    %v6045 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6046 = stablehlo.multiply %v6045, %v6038 : tensor<384x1536x1x1xf32>
    %v6047 = stablehlo.multiply %v6046, %s2b0pW : tensor<384x1536x1x1xf32>
    %v6048 = stablehlo.subtract %v6044, %v6047 : tensor<384x1536x1x1xf32>
    %arsums2b0pb = "stablehlo.all_reduce"(%v2534) ({
    ^bb0(%aras2b0pb: tensor<f32>, %arbs2b0pb: tensor<f32>):
      %aradds2b0pb = stablehlo.add %aras2b0pb, %arbs2b0pb : tensor<f32>
      stablehlo.return %aradds2b0pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b0pb = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b0pb = stablehlo.divide %arsums2b0pb, %arns2b0pb : tensor<384xf32>
    %v6049 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6050 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6051 = stablehlo.multiply %v6049, %s2b0pbm : tensor<384xf32>
    %v6052 = stablehlo.multiply %v6050, %armeans2b0pb : tensor<384xf32>
    %v6053 = stablehlo.add %v6051, %v6052 : tensor<384xf32>
    %v6054 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6055 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6056 = stablehlo.multiply %v6054, %s2b0pbv : tensor<384xf32>
    %v6057 = stablehlo.multiply %armeans2b0pb, %armeans2b0pb : tensor<384xf32>
    %v6058 = stablehlo.multiply %v6055, %v6057 : tensor<384xf32>
    %v6059 = stablehlo.add %v6056, %v6058 : tensor<384xf32>
    %v6060 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6061 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6062 = stablehlo.multiply %v6060, %s2b0pbm : tensor<384xf32>
    %v6063 = stablehlo.multiply %v6061, %armeans2b0pb : tensor<384xf32>
    %v6064 = stablehlo.add %v6062, %v6063 : tensor<384xf32>
    %v6065 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6066 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6067 = stablehlo.multiply %v6065, %s2b0pbv : tensor<384xf32>
    %v6068 = stablehlo.multiply %armeans2b0pb, %armeans2b0pb : tensor<384xf32>
    %v6069 = stablehlo.multiply %v6066, %v6068 : tensor<384xf32>
    %v6070 = stablehlo.add %v6067, %v6069 : tensor<384xf32>
    %v6071 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6072 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6073 = stablehlo.divide %v6064, %v6071 : tensor<384xf32>
    %v6074 = stablehlo.divide %v6070, %v6072 : tensor<384xf32>
    %v6075 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6076 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6077 = stablehlo.sqrt %v6074 : tensor<384xf32>
    %v6078 = stablehlo.add %v6077, %v6076 : tensor<384xf32>
    %v6079 = stablehlo.divide %v6073, %v6078 : tensor<384xf32>
    %v6080 = stablehlo.multiply %v6075, %v6079 : tensor<384xf32>
    %v6081 = stablehlo.subtract %s2b0pb, %v6080 : tensor<384xf32>
    %v6082 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6083 = stablehlo.multiply %v6082, %v6075 : tensor<384xf32>
    %v6084 = stablehlo.multiply %v6083, %s2b0pb : tensor<384xf32>
    %v6085 = stablehlo.subtract %v6081, %v6084 : tensor<384xf32>
    %arsums2b0lg = "stablehlo.all_reduce"(%v2525) ({
    ^bb0(%aras2b0lg: tensor<f32>, %arbs2b0lg: tensor<f32>):
      %aradds2b0lg = stablehlo.add %aras2b0lg, %arbs2b0lg : tensor<f32>
      stablehlo.return %aradds2b0lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b0lg = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b0lg = stablehlo.divide %arsums2b0lg, %arns2b0lg : tensor<384xf32>
    %v6086 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6087 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6088 = stablehlo.multiply %v6086, %s2b0lgm : tensor<384xf32>
    %v6089 = stablehlo.multiply %v6087, %armeans2b0lg : tensor<384xf32>
    %v6090 = stablehlo.add %v6088, %v6089 : tensor<384xf32>
    %v6091 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6092 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6093 = stablehlo.multiply %v6091, %s2b0lgv : tensor<384xf32>
    %v6094 = stablehlo.multiply %armeans2b0lg, %armeans2b0lg : tensor<384xf32>
    %v6095 = stablehlo.multiply %v6092, %v6094 : tensor<384xf32>
    %v6096 = stablehlo.add %v6093, %v6095 : tensor<384xf32>
    %v6097 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6098 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6099 = stablehlo.multiply %v6097, %s2b0lgm : tensor<384xf32>
    %v6100 = stablehlo.multiply %v6098, %armeans2b0lg : tensor<384xf32>
    %v6101 = stablehlo.add %v6099, %v6100 : tensor<384xf32>
    %v6102 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6103 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6104 = stablehlo.multiply %v6102, %s2b0lgv : tensor<384xf32>
    %v6105 = stablehlo.multiply %armeans2b0lg, %armeans2b0lg : tensor<384xf32>
    %v6106 = stablehlo.multiply %v6103, %v6105 : tensor<384xf32>
    %v6107 = stablehlo.add %v6104, %v6106 : tensor<384xf32>
    %v6108 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6109 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6110 = stablehlo.divide %v6101, %v6108 : tensor<384xf32>
    %v6111 = stablehlo.divide %v6107, %v6109 : tensor<384xf32>
    %v6112 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6113 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6114 = stablehlo.sqrt %v6111 : tensor<384xf32>
    %v6115 = stablehlo.add %v6114, %v6113 : tensor<384xf32>
    %v6116 = stablehlo.divide %v6110, %v6115 : tensor<384xf32>
    %v6117 = stablehlo.multiply %v6112, %v6116 : tensor<384xf32>
    %v6118 = stablehlo.subtract %s2b0lg, %v6117 : tensor<384xf32>
    %v6119 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6120 = stablehlo.multiply %v6119, %v6112 : tensor<384xf32>
    %v6121 = stablehlo.multiply %v6120, %s2b0lg : tensor<384xf32>
    %v6122 = stablehlo.subtract %v6118, %v6121 : tensor<384xf32>
    %arsums2b1dW = "stablehlo.all_reduce"(%v2448) ({
    ^bb0(%aras2b1dW: tensor<f32>, %arbs2b1dW: tensor<f32>):
      %aradds2b1dW = stablehlo.add %aras2b1dW, %arbs2b1dW : tensor<f32>
      stablehlo.return %aradds2b1dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1x7x7xf32>) -> tensor<384x1x7x7xf32>
    %arns2b1dW = stablehlo.constant dense<2.0> : tensor<384x1x7x7xf32>
    %armeans2b1dW = stablehlo.divide %arsums2b1dW, %arns2b1dW : tensor<384x1x7x7xf32>
    %v6123 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6124 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6125 = stablehlo.multiply %v6123, %s2b1dWm : tensor<384x1x7x7xf32>
    %v6126 = stablehlo.multiply %v6124, %armeans2b1dW : tensor<384x1x7x7xf32>
    %v6127 = stablehlo.add %v6125, %v6126 : tensor<384x1x7x7xf32>
    %v6128 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6129 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6130 = stablehlo.multiply %v6128, %s2b1dWv : tensor<384x1x7x7xf32>
    %v6131 = stablehlo.multiply %armeans2b1dW, %armeans2b1dW : tensor<384x1x7x7xf32>
    %v6132 = stablehlo.multiply %v6129, %v6131 : tensor<384x1x7x7xf32>
    %v6133 = stablehlo.add %v6130, %v6132 : tensor<384x1x7x7xf32>
    %v6134 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6135 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6136 = stablehlo.multiply %v6134, %s2b1dWm : tensor<384x1x7x7xf32>
    %v6137 = stablehlo.multiply %v6135, %armeans2b1dW : tensor<384x1x7x7xf32>
    %v6138 = stablehlo.add %v6136, %v6137 : tensor<384x1x7x7xf32>
    %v6139 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6140 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6141 = stablehlo.multiply %v6139, %s2b1dWv : tensor<384x1x7x7xf32>
    %v6142 = stablehlo.multiply %armeans2b1dW, %armeans2b1dW : tensor<384x1x7x7xf32>
    %v6143 = stablehlo.multiply %v6140, %v6142 : tensor<384x1x7x7xf32>
    %v6144 = stablehlo.add %v6141, %v6143 : tensor<384x1x7x7xf32>
    %v6145 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6146 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6147 = stablehlo.divide %v6138, %v6145 : tensor<384x1x7x7xf32>
    %v6148 = stablehlo.divide %v6144, %v6146 : tensor<384x1x7x7xf32>
    %v6149 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6150 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6151 = stablehlo.sqrt %v6148 : tensor<384x1x7x7xf32>
    %v6152 = stablehlo.add %v6151, %v6150 : tensor<384x1x7x7xf32>
    %v6153 = stablehlo.divide %v6147, %v6152 : tensor<384x1x7x7xf32>
    %v6154 = stablehlo.multiply %v6149, %v6153 : tensor<384x1x7x7xf32>
    %v6155 = stablehlo.subtract %s2b1dW, %v6154 : tensor<384x1x7x7xf32>
    %v6156 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6157 = stablehlo.multiply %v6156, %v6149 : tensor<384x1x7x7xf32>
    %v6158 = stablehlo.multiply %v6157, %s2b1dW : tensor<384x1x7x7xf32>
    %v6159 = stablehlo.subtract %v6155, %v6158 : tensor<384x1x7x7xf32>
    %arsums2b1db = "stablehlo.all_reduce"(%v2451) ({
    ^bb0(%aras2b1db: tensor<f32>, %arbs2b1db: tensor<f32>):
      %aradds2b1db = stablehlo.add %aras2b1db, %arbs2b1db : tensor<f32>
      stablehlo.return %aradds2b1db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b1db = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b1db = stablehlo.divide %arsums2b1db, %arns2b1db : tensor<384xf32>
    %v6160 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6161 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6162 = stablehlo.multiply %v6160, %s2b1dbm : tensor<384xf32>
    %v6163 = stablehlo.multiply %v6161, %armeans2b1db : tensor<384xf32>
    %v6164 = stablehlo.add %v6162, %v6163 : tensor<384xf32>
    %v6165 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6166 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6167 = stablehlo.multiply %v6165, %s2b1dbv : tensor<384xf32>
    %v6168 = stablehlo.multiply %armeans2b1db, %armeans2b1db : tensor<384xf32>
    %v6169 = stablehlo.multiply %v6166, %v6168 : tensor<384xf32>
    %v6170 = stablehlo.add %v6167, %v6169 : tensor<384xf32>
    %v6171 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6172 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6173 = stablehlo.multiply %v6171, %s2b1dbm : tensor<384xf32>
    %v6174 = stablehlo.multiply %v6172, %armeans2b1db : tensor<384xf32>
    %v6175 = stablehlo.add %v6173, %v6174 : tensor<384xf32>
    %v6176 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6177 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6178 = stablehlo.multiply %v6176, %s2b1dbv : tensor<384xf32>
    %v6179 = stablehlo.multiply %armeans2b1db, %armeans2b1db : tensor<384xf32>
    %v6180 = stablehlo.multiply %v6177, %v6179 : tensor<384xf32>
    %v6181 = stablehlo.add %v6178, %v6180 : tensor<384xf32>
    %v6182 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6183 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6184 = stablehlo.divide %v6175, %v6182 : tensor<384xf32>
    %v6185 = stablehlo.divide %v6181, %v6183 : tensor<384xf32>
    %v6186 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6187 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6188 = stablehlo.sqrt %v6185 : tensor<384xf32>
    %v6189 = stablehlo.add %v6188, %v6187 : tensor<384xf32>
    %v6190 = stablehlo.divide %v6184, %v6189 : tensor<384xf32>
    %v6191 = stablehlo.multiply %v6186, %v6190 : tensor<384xf32>
    %v6192 = stablehlo.subtract %s2b1db, %v6191 : tensor<384xf32>
    %v6193 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6194 = stablehlo.multiply %v6193, %v6186 : tensor<384xf32>
    %v6195 = stablehlo.multiply %v6194, %s2b1db : tensor<384xf32>
    %v6196 = stablehlo.subtract %v6192, %v6195 : tensor<384xf32>
    %arsums2b1ng = "stablehlo.all_reduce"(%v2440) ({
    ^bb0(%aras2b1ng: tensor<f32>, %arbs2b1ng: tensor<f32>):
      %aradds2b1ng = stablehlo.add %aras2b1ng, %arbs2b1ng : tensor<f32>
      stablehlo.return %aradds2b1ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b1ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b1ng = stablehlo.divide %arsums2b1ng, %arns2b1ng : tensor<f32>
    %v6197 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6198 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6199 = stablehlo.multiply %v6197, %s2b1ngm : tensor<f32>
    %v6200 = stablehlo.multiply %v6198, %armeans2b1ng : tensor<f32>
    %v6201 = stablehlo.add %v6199, %v6200 : tensor<f32>
    %v6202 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6203 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6204 = stablehlo.multiply %v6202, %s2b1ngv : tensor<f32>
    %v6205 = stablehlo.multiply %armeans2b1ng, %armeans2b1ng : tensor<f32>
    %v6206 = stablehlo.multiply %v6203, %v6205 : tensor<f32>
    %v6207 = stablehlo.add %v6204, %v6206 : tensor<f32>
    %v6208 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6209 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6210 = stablehlo.multiply %v6208, %s2b1ngm : tensor<f32>
    %v6211 = stablehlo.multiply %v6209, %armeans2b1ng : tensor<f32>
    %v6212 = stablehlo.add %v6210, %v6211 : tensor<f32>
    %v6213 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6214 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6215 = stablehlo.multiply %v6213, %s2b1ngv : tensor<f32>
    %v6216 = stablehlo.multiply %armeans2b1ng, %armeans2b1ng : tensor<f32>
    %v6217 = stablehlo.multiply %v6214, %v6216 : tensor<f32>
    %v6218 = stablehlo.add %v6215, %v6217 : tensor<f32>
    %v6219 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6220 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6221 = stablehlo.divide %v6212, %v6219 : tensor<f32>
    %v6222 = stablehlo.divide %v6218, %v6220 : tensor<f32>
    %v6223 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6224 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6225 = stablehlo.sqrt %v6222 : tensor<f32>
    %v6226 = stablehlo.add %v6225, %v6224 : tensor<f32>
    %v6227 = stablehlo.divide %v6221, %v6226 : tensor<f32>
    %v6228 = stablehlo.multiply %v6223, %v6227 : tensor<f32>
    %v6229 = stablehlo.subtract %s2b1ng, %v6228 : tensor<f32>
    %v6230 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6231 = stablehlo.multiply %v6230, %v6223 : tensor<f32>
    %v6232 = stablehlo.multiply %v6231, %s2b1ng : tensor<f32>
    %v6233 = stablehlo.subtract %v6229, %v6232 : tensor<f32>
    %arsums2b1nbt = "stablehlo.all_reduce"(%v2442) ({
    ^bb0(%aras2b1nbt: tensor<f32>, %arbs2b1nbt: tensor<f32>):
      %aradds2b1nbt = stablehlo.add %aras2b1nbt, %arbs2b1nbt : tensor<f32>
      stablehlo.return %aradds2b1nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b1nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b1nbt = stablehlo.divide %arsums2b1nbt, %arns2b1nbt : tensor<f32>
    %v6234 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6235 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6236 = stablehlo.multiply %v6234, %s2b1nbtm : tensor<f32>
    %v6237 = stablehlo.multiply %v6235, %armeans2b1nbt : tensor<f32>
    %v6238 = stablehlo.add %v6236, %v6237 : tensor<f32>
    %v6239 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6240 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6241 = stablehlo.multiply %v6239, %s2b1nbtv : tensor<f32>
    %v6242 = stablehlo.multiply %armeans2b1nbt, %armeans2b1nbt : tensor<f32>
    %v6243 = stablehlo.multiply %v6240, %v6242 : tensor<f32>
    %v6244 = stablehlo.add %v6241, %v6243 : tensor<f32>
    %v6245 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6246 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6247 = stablehlo.multiply %v6245, %s2b1nbtm : tensor<f32>
    %v6248 = stablehlo.multiply %v6246, %armeans2b1nbt : tensor<f32>
    %v6249 = stablehlo.add %v6247, %v6248 : tensor<f32>
    %v6250 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6251 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6252 = stablehlo.multiply %v6250, %s2b1nbtv : tensor<f32>
    %v6253 = stablehlo.multiply %armeans2b1nbt, %armeans2b1nbt : tensor<f32>
    %v6254 = stablehlo.multiply %v6251, %v6253 : tensor<f32>
    %v6255 = stablehlo.add %v6252, %v6254 : tensor<f32>
    %v6256 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6257 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6258 = stablehlo.divide %v6249, %v6256 : tensor<f32>
    %v6259 = stablehlo.divide %v6255, %v6257 : tensor<f32>
    %v6260 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6261 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6262 = stablehlo.sqrt %v6259 : tensor<f32>
    %v6263 = stablehlo.add %v6262, %v6261 : tensor<f32>
    %v6264 = stablehlo.divide %v6258, %v6263 : tensor<f32>
    %v6265 = stablehlo.multiply %v6260, %v6264 : tensor<f32>
    %v6266 = stablehlo.subtract %s2b1nbt, %v6265 : tensor<f32>
    %v6267 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6268 = stablehlo.multiply %v6267, %v6260 : tensor<f32>
    %v6269 = stablehlo.multiply %v6268, %s2b1nbt : tensor<f32>
    %v6270 = stablehlo.subtract %v6266, %v6269 : tensor<f32>
    %arsums2b1eW = "stablehlo.all_reduce"(%v2421) ({
    ^bb0(%aras2b1eW: tensor<f32>, %arbs2b1eW: tensor<f32>):
      %aradds2b1eW = stablehlo.add %aras2b1eW, %arbs2b1eW : tensor<f32>
      stablehlo.return %aradds2b1eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536x384x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %arns2b1eW = stablehlo.constant dense<2.0> : tensor<1536x384x1x1xf32>
    %armeans2b1eW = stablehlo.divide %arsums2b1eW, %arns2b1eW : tensor<1536x384x1x1xf32>
    %v6271 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6272 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6273 = stablehlo.multiply %v6271, %s2b1eWm : tensor<1536x384x1x1xf32>
    %v6274 = stablehlo.multiply %v6272, %armeans2b1eW : tensor<1536x384x1x1xf32>
    %v6275 = stablehlo.add %v6273, %v6274 : tensor<1536x384x1x1xf32>
    %v6276 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6277 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6278 = stablehlo.multiply %v6276, %s2b1eWv : tensor<1536x384x1x1xf32>
    %v6279 = stablehlo.multiply %armeans2b1eW, %armeans2b1eW : tensor<1536x384x1x1xf32>
    %v6280 = stablehlo.multiply %v6277, %v6279 : tensor<1536x384x1x1xf32>
    %v6281 = stablehlo.add %v6278, %v6280 : tensor<1536x384x1x1xf32>
    %v6282 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6283 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6284 = stablehlo.multiply %v6282, %s2b1eWm : tensor<1536x384x1x1xf32>
    %v6285 = stablehlo.multiply %v6283, %armeans2b1eW : tensor<1536x384x1x1xf32>
    %v6286 = stablehlo.add %v6284, %v6285 : tensor<1536x384x1x1xf32>
    %v6287 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6288 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6289 = stablehlo.multiply %v6287, %s2b1eWv : tensor<1536x384x1x1xf32>
    %v6290 = stablehlo.multiply %armeans2b1eW, %armeans2b1eW : tensor<1536x384x1x1xf32>
    %v6291 = stablehlo.multiply %v6288, %v6290 : tensor<1536x384x1x1xf32>
    %v6292 = stablehlo.add %v6289, %v6291 : tensor<1536x384x1x1xf32>
    %v6293 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6294 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6295 = stablehlo.divide %v6286, %v6293 : tensor<1536x384x1x1xf32>
    %v6296 = stablehlo.divide %v6292, %v6294 : tensor<1536x384x1x1xf32>
    %v6297 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6298 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6299 = stablehlo.sqrt %v6296 : tensor<1536x384x1x1xf32>
    %v6300 = stablehlo.add %v6299, %v6298 : tensor<1536x384x1x1xf32>
    %v6301 = stablehlo.divide %v6295, %v6300 : tensor<1536x384x1x1xf32>
    %v6302 = stablehlo.multiply %v6297, %v6301 : tensor<1536x384x1x1xf32>
    %v6303 = stablehlo.subtract %s2b1eW, %v6302 : tensor<1536x384x1x1xf32>
    %v6304 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6305 = stablehlo.multiply %v6304, %v6297 : tensor<1536x384x1x1xf32>
    %v6306 = stablehlo.multiply %v6305, %s2b1eW : tensor<1536x384x1x1xf32>
    %v6307 = stablehlo.subtract %v6303, %v6306 : tensor<1536x384x1x1xf32>
    %arsums2b1eb = "stablehlo.all_reduce"(%v2424) ({
    ^bb0(%aras2b1eb: tensor<f32>, %arbs2b1eb: tensor<f32>):
      %aradds2b1eb = stablehlo.add %aras2b1eb, %arbs2b1eb : tensor<f32>
      stablehlo.return %aradds2b1eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536xf32>) -> tensor<1536xf32>
    %arns2b1eb = stablehlo.constant dense<2.0> : tensor<1536xf32>
    %armeans2b1eb = stablehlo.divide %arsums2b1eb, %arns2b1eb : tensor<1536xf32>
    %v6308 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6309 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6310 = stablehlo.multiply %v6308, %s2b1ebm : tensor<1536xf32>
    %v6311 = stablehlo.multiply %v6309, %armeans2b1eb : tensor<1536xf32>
    %v6312 = stablehlo.add %v6310, %v6311 : tensor<1536xf32>
    %v6313 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6314 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6315 = stablehlo.multiply %v6313, %s2b1ebv : tensor<1536xf32>
    %v6316 = stablehlo.multiply %armeans2b1eb, %armeans2b1eb : tensor<1536xf32>
    %v6317 = stablehlo.multiply %v6314, %v6316 : tensor<1536xf32>
    %v6318 = stablehlo.add %v6315, %v6317 : tensor<1536xf32>
    %v6319 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6320 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6321 = stablehlo.multiply %v6319, %s2b1ebm : tensor<1536xf32>
    %v6322 = stablehlo.multiply %v6320, %armeans2b1eb : tensor<1536xf32>
    %v6323 = stablehlo.add %v6321, %v6322 : tensor<1536xf32>
    %v6324 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6325 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6326 = stablehlo.multiply %v6324, %s2b1ebv : tensor<1536xf32>
    %v6327 = stablehlo.multiply %armeans2b1eb, %armeans2b1eb : tensor<1536xf32>
    %v6328 = stablehlo.multiply %v6325, %v6327 : tensor<1536xf32>
    %v6329 = stablehlo.add %v6326, %v6328 : tensor<1536xf32>
    %v6330 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6331 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6332 = stablehlo.divide %v6323, %v6330 : tensor<1536xf32>
    %v6333 = stablehlo.divide %v6329, %v6331 : tensor<1536xf32>
    %v6334 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6335 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6336 = stablehlo.sqrt %v6333 : tensor<1536xf32>
    %v6337 = stablehlo.add %v6336, %v6335 : tensor<1536xf32>
    %v6338 = stablehlo.divide %v6332, %v6337 : tensor<1536xf32>
    %v6339 = stablehlo.multiply %v6334, %v6338 : tensor<1536xf32>
    %v6340 = stablehlo.subtract %s2b1eb, %v6339 : tensor<1536xf32>
    %v6341 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6342 = stablehlo.multiply %v6341, %v6334 : tensor<1536xf32>
    %v6343 = stablehlo.multiply %v6342, %s2b1eb : tensor<1536xf32>
    %v6344 = stablehlo.subtract %v6340, %v6343 : tensor<1536xf32>
    %arsums2b1pW = "stablehlo.all_reduce"(%v2412) ({
    ^bb0(%aras2b1pW: tensor<f32>, %arbs2b1pW: tensor<f32>):
      %aradds2b1pW = stablehlo.add %aras2b1pW, %arbs2b1pW : tensor<f32>
      stablehlo.return %aradds2b1pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1536x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %arns2b1pW = stablehlo.constant dense<2.0> : tensor<384x1536x1x1xf32>
    %armeans2b1pW = stablehlo.divide %arsums2b1pW, %arns2b1pW : tensor<384x1536x1x1xf32>
    %v6345 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6346 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6347 = stablehlo.multiply %v6345, %s2b1pWm : tensor<384x1536x1x1xf32>
    %v6348 = stablehlo.multiply %v6346, %armeans2b1pW : tensor<384x1536x1x1xf32>
    %v6349 = stablehlo.add %v6347, %v6348 : tensor<384x1536x1x1xf32>
    %v6350 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6351 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6352 = stablehlo.multiply %v6350, %s2b1pWv : tensor<384x1536x1x1xf32>
    %v6353 = stablehlo.multiply %armeans2b1pW, %armeans2b1pW : tensor<384x1536x1x1xf32>
    %v6354 = stablehlo.multiply %v6351, %v6353 : tensor<384x1536x1x1xf32>
    %v6355 = stablehlo.add %v6352, %v6354 : tensor<384x1536x1x1xf32>
    %v6356 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6357 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6358 = stablehlo.multiply %v6356, %s2b1pWm : tensor<384x1536x1x1xf32>
    %v6359 = stablehlo.multiply %v6357, %armeans2b1pW : tensor<384x1536x1x1xf32>
    %v6360 = stablehlo.add %v6358, %v6359 : tensor<384x1536x1x1xf32>
    %v6361 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6362 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6363 = stablehlo.multiply %v6361, %s2b1pWv : tensor<384x1536x1x1xf32>
    %v6364 = stablehlo.multiply %armeans2b1pW, %armeans2b1pW : tensor<384x1536x1x1xf32>
    %v6365 = stablehlo.multiply %v6362, %v6364 : tensor<384x1536x1x1xf32>
    %v6366 = stablehlo.add %v6363, %v6365 : tensor<384x1536x1x1xf32>
    %v6367 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6368 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6369 = stablehlo.divide %v6360, %v6367 : tensor<384x1536x1x1xf32>
    %v6370 = stablehlo.divide %v6366, %v6368 : tensor<384x1536x1x1xf32>
    %v6371 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6372 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6373 = stablehlo.sqrt %v6370 : tensor<384x1536x1x1xf32>
    %v6374 = stablehlo.add %v6373, %v6372 : tensor<384x1536x1x1xf32>
    %v6375 = stablehlo.divide %v6369, %v6374 : tensor<384x1536x1x1xf32>
    %v6376 = stablehlo.multiply %v6371, %v6375 : tensor<384x1536x1x1xf32>
    %v6377 = stablehlo.subtract %s2b1pW, %v6376 : tensor<384x1536x1x1xf32>
    %v6378 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6379 = stablehlo.multiply %v6378, %v6371 : tensor<384x1536x1x1xf32>
    %v6380 = stablehlo.multiply %v6379, %s2b1pW : tensor<384x1536x1x1xf32>
    %v6381 = stablehlo.subtract %v6377, %v6380 : tensor<384x1536x1x1xf32>
    %arsums2b1pb = "stablehlo.all_reduce"(%v2415) ({
    ^bb0(%aras2b1pb: tensor<f32>, %arbs2b1pb: tensor<f32>):
      %aradds2b1pb = stablehlo.add %aras2b1pb, %arbs2b1pb : tensor<f32>
      stablehlo.return %aradds2b1pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b1pb = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b1pb = stablehlo.divide %arsums2b1pb, %arns2b1pb : tensor<384xf32>
    %v6382 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6383 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6384 = stablehlo.multiply %v6382, %s2b1pbm : tensor<384xf32>
    %v6385 = stablehlo.multiply %v6383, %armeans2b1pb : tensor<384xf32>
    %v6386 = stablehlo.add %v6384, %v6385 : tensor<384xf32>
    %v6387 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6388 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6389 = stablehlo.multiply %v6387, %s2b1pbv : tensor<384xf32>
    %v6390 = stablehlo.multiply %armeans2b1pb, %armeans2b1pb : tensor<384xf32>
    %v6391 = stablehlo.multiply %v6388, %v6390 : tensor<384xf32>
    %v6392 = stablehlo.add %v6389, %v6391 : tensor<384xf32>
    %v6393 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6394 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6395 = stablehlo.multiply %v6393, %s2b1pbm : tensor<384xf32>
    %v6396 = stablehlo.multiply %v6394, %armeans2b1pb : tensor<384xf32>
    %v6397 = stablehlo.add %v6395, %v6396 : tensor<384xf32>
    %v6398 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6399 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6400 = stablehlo.multiply %v6398, %s2b1pbv : tensor<384xf32>
    %v6401 = stablehlo.multiply %armeans2b1pb, %armeans2b1pb : tensor<384xf32>
    %v6402 = stablehlo.multiply %v6399, %v6401 : tensor<384xf32>
    %v6403 = stablehlo.add %v6400, %v6402 : tensor<384xf32>
    %v6404 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6405 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6406 = stablehlo.divide %v6397, %v6404 : tensor<384xf32>
    %v6407 = stablehlo.divide %v6403, %v6405 : tensor<384xf32>
    %v6408 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6409 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6410 = stablehlo.sqrt %v6407 : tensor<384xf32>
    %v6411 = stablehlo.add %v6410, %v6409 : tensor<384xf32>
    %v6412 = stablehlo.divide %v6406, %v6411 : tensor<384xf32>
    %v6413 = stablehlo.multiply %v6408, %v6412 : tensor<384xf32>
    %v6414 = stablehlo.subtract %s2b1pb, %v6413 : tensor<384xf32>
    %v6415 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6416 = stablehlo.multiply %v6415, %v6408 : tensor<384xf32>
    %v6417 = stablehlo.multiply %v6416, %s2b1pb : tensor<384xf32>
    %v6418 = stablehlo.subtract %v6414, %v6417 : tensor<384xf32>
    %arsums2b1lg = "stablehlo.all_reduce"(%v2406) ({
    ^bb0(%aras2b1lg: tensor<f32>, %arbs2b1lg: tensor<f32>):
      %aradds2b1lg = stablehlo.add %aras2b1lg, %arbs2b1lg : tensor<f32>
      stablehlo.return %aradds2b1lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b1lg = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b1lg = stablehlo.divide %arsums2b1lg, %arns2b1lg : tensor<384xf32>
    %v6419 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6420 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6421 = stablehlo.multiply %v6419, %s2b1lgm : tensor<384xf32>
    %v6422 = stablehlo.multiply %v6420, %armeans2b1lg : tensor<384xf32>
    %v6423 = stablehlo.add %v6421, %v6422 : tensor<384xf32>
    %v6424 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6425 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6426 = stablehlo.multiply %v6424, %s2b1lgv : tensor<384xf32>
    %v6427 = stablehlo.multiply %armeans2b1lg, %armeans2b1lg : tensor<384xf32>
    %v6428 = stablehlo.multiply %v6425, %v6427 : tensor<384xf32>
    %v6429 = stablehlo.add %v6426, %v6428 : tensor<384xf32>
    %v6430 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6431 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6432 = stablehlo.multiply %v6430, %s2b1lgm : tensor<384xf32>
    %v6433 = stablehlo.multiply %v6431, %armeans2b1lg : tensor<384xf32>
    %v6434 = stablehlo.add %v6432, %v6433 : tensor<384xf32>
    %v6435 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6436 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6437 = stablehlo.multiply %v6435, %s2b1lgv : tensor<384xf32>
    %v6438 = stablehlo.multiply %armeans2b1lg, %armeans2b1lg : tensor<384xf32>
    %v6439 = stablehlo.multiply %v6436, %v6438 : tensor<384xf32>
    %v6440 = stablehlo.add %v6437, %v6439 : tensor<384xf32>
    %v6441 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6442 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6443 = stablehlo.divide %v6434, %v6441 : tensor<384xf32>
    %v6444 = stablehlo.divide %v6440, %v6442 : tensor<384xf32>
    %v6445 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6446 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6447 = stablehlo.sqrt %v6444 : tensor<384xf32>
    %v6448 = stablehlo.add %v6447, %v6446 : tensor<384xf32>
    %v6449 = stablehlo.divide %v6443, %v6448 : tensor<384xf32>
    %v6450 = stablehlo.multiply %v6445, %v6449 : tensor<384xf32>
    %v6451 = stablehlo.subtract %s2b1lg, %v6450 : tensor<384xf32>
    %v6452 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6453 = stablehlo.multiply %v6452, %v6445 : tensor<384xf32>
    %v6454 = stablehlo.multiply %v6453, %s2b1lg : tensor<384xf32>
    %v6455 = stablehlo.subtract %v6451, %v6454 : tensor<384xf32>
    %arsums2b2dW = "stablehlo.all_reduce"(%v2329) ({
    ^bb0(%aras2b2dW: tensor<f32>, %arbs2b2dW: tensor<f32>):
      %aradds2b2dW = stablehlo.add %aras2b2dW, %arbs2b2dW : tensor<f32>
      stablehlo.return %aradds2b2dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1x7x7xf32>) -> tensor<384x1x7x7xf32>
    %arns2b2dW = stablehlo.constant dense<2.0> : tensor<384x1x7x7xf32>
    %armeans2b2dW = stablehlo.divide %arsums2b2dW, %arns2b2dW : tensor<384x1x7x7xf32>
    %v6456 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6457 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6458 = stablehlo.multiply %v6456, %s2b2dWm : tensor<384x1x7x7xf32>
    %v6459 = stablehlo.multiply %v6457, %armeans2b2dW : tensor<384x1x7x7xf32>
    %v6460 = stablehlo.add %v6458, %v6459 : tensor<384x1x7x7xf32>
    %v6461 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6462 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6463 = stablehlo.multiply %v6461, %s2b2dWv : tensor<384x1x7x7xf32>
    %v6464 = stablehlo.multiply %armeans2b2dW, %armeans2b2dW : tensor<384x1x7x7xf32>
    %v6465 = stablehlo.multiply %v6462, %v6464 : tensor<384x1x7x7xf32>
    %v6466 = stablehlo.add %v6463, %v6465 : tensor<384x1x7x7xf32>
    %v6467 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6468 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6469 = stablehlo.multiply %v6467, %s2b2dWm : tensor<384x1x7x7xf32>
    %v6470 = stablehlo.multiply %v6468, %armeans2b2dW : tensor<384x1x7x7xf32>
    %v6471 = stablehlo.add %v6469, %v6470 : tensor<384x1x7x7xf32>
    %v6472 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6473 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6474 = stablehlo.multiply %v6472, %s2b2dWv : tensor<384x1x7x7xf32>
    %v6475 = stablehlo.multiply %armeans2b2dW, %armeans2b2dW : tensor<384x1x7x7xf32>
    %v6476 = stablehlo.multiply %v6473, %v6475 : tensor<384x1x7x7xf32>
    %v6477 = stablehlo.add %v6474, %v6476 : tensor<384x1x7x7xf32>
    %v6478 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6479 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6480 = stablehlo.divide %v6471, %v6478 : tensor<384x1x7x7xf32>
    %v6481 = stablehlo.divide %v6477, %v6479 : tensor<384x1x7x7xf32>
    %v6482 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6483 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6484 = stablehlo.sqrt %v6481 : tensor<384x1x7x7xf32>
    %v6485 = stablehlo.add %v6484, %v6483 : tensor<384x1x7x7xf32>
    %v6486 = stablehlo.divide %v6480, %v6485 : tensor<384x1x7x7xf32>
    %v6487 = stablehlo.multiply %v6482, %v6486 : tensor<384x1x7x7xf32>
    %v6488 = stablehlo.subtract %s2b2dW, %v6487 : tensor<384x1x7x7xf32>
    %v6489 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6490 = stablehlo.multiply %v6489, %v6482 : tensor<384x1x7x7xf32>
    %v6491 = stablehlo.multiply %v6490, %s2b2dW : tensor<384x1x7x7xf32>
    %v6492 = stablehlo.subtract %v6488, %v6491 : tensor<384x1x7x7xf32>
    %arsums2b2db = "stablehlo.all_reduce"(%v2332) ({
    ^bb0(%aras2b2db: tensor<f32>, %arbs2b2db: tensor<f32>):
      %aradds2b2db = stablehlo.add %aras2b2db, %arbs2b2db : tensor<f32>
      stablehlo.return %aradds2b2db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b2db = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b2db = stablehlo.divide %arsums2b2db, %arns2b2db : tensor<384xf32>
    %v6493 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6494 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6495 = stablehlo.multiply %v6493, %s2b2dbm : tensor<384xf32>
    %v6496 = stablehlo.multiply %v6494, %armeans2b2db : tensor<384xf32>
    %v6497 = stablehlo.add %v6495, %v6496 : tensor<384xf32>
    %v6498 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6499 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6500 = stablehlo.multiply %v6498, %s2b2dbv : tensor<384xf32>
    %v6501 = stablehlo.multiply %armeans2b2db, %armeans2b2db : tensor<384xf32>
    %v6502 = stablehlo.multiply %v6499, %v6501 : tensor<384xf32>
    %v6503 = stablehlo.add %v6500, %v6502 : tensor<384xf32>
    %v6504 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6505 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6506 = stablehlo.multiply %v6504, %s2b2dbm : tensor<384xf32>
    %v6507 = stablehlo.multiply %v6505, %armeans2b2db : tensor<384xf32>
    %v6508 = stablehlo.add %v6506, %v6507 : tensor<384xf32>
    %v6509 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6510 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6511 = stablehlo.multiply %v6509, %s2b2dbv : tensor<384xf32>
    %v6512 = stablehlo.multiply %armeans2b2db, %armeans2b2db : tensor<384xf32>
    %v6513 = stablehlo.multiply %v6510, %v6512 : tensor<384xf32>
    %v6514 = stablehlo.add %v6511, %v6513 : tensor<384xf32>
    %v6515 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6516 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6517 = stablehlo.divide %v6508, %v6515 : tensor<384xf32>
    %v6518 = stablehlo.divide %v6514, %v6516 : tensor<384xf32>
    %v6519 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6520 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6521 = stablehlo.sqrt %v6518 : tensor<384xf32>
    %v6522 = stablehlo.add %v6521, %v6520 : tensor<384xf32>
    %v6523 = stablehlo.divide %v6517, %v6522 : tensor<384xf32>
    %v6524 = stablehlo.multiply %v6519, %v6523 : tensor<384xf32>
    %v6525 = stablehlo.subtract %s2b2db, %v6524 : tensor<384xf32>
    %v6526 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6527 = stablehlo.multiply %v6526, %v6519 : tensor<384xf32>
    %v6528 = stablehlo.multiply %v6527, %s2b2db : tensor<384xf32>
    %v6529 = stablehlo.subtract %v6525, %v6528 : tensor<384xf32>
    %arsums2b2ng = "stablehlo.all_reduce"(%v2321) ({
    ^bb0(%aras2b2ng: tensor<f32>, %arbs2b2ng: tensor<f32>):
      %aradds2b2ng = stablehlo.add %aras2b2ng, %arbs2b2ng : tensor<f32>
      stablehlo.return %aradds2b2ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b2ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b2ng = stablehlo.divide %arsums2b2ng, %arns2b2ng : tensor<f32>
    %v6530 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6531 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6532 = stablehlo.multiply %v6530, %s2b2ngm : tensor<f32>
    %v6533 = stablehlo.multiply %v6531, %armeans2b2ng : tensor<f32>
    %v6534 = stablehlo.add %v6532, %v6533 : tensor<f32>
    %v6535 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6536 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6537 = stablehlo.multiply %v6535, %s2b2ngv : tensor<f32>
    %v6538 = stablehlo.multiply %armeans2b2ng, %armeans2b2ng : tensor<f32>
    %v6539 = stablehlo.multiply %v6536, %v6538 : tensor<f32>
    %v6540 = stablehlo.add %v6537, %v6539 : tensor<f32>
    %v6541 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6542 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6543 = stablehlo.multiply %v6541, %s2b2ngm : tensor<f32>
    %v6544 = stablehlo.multiply %v6542, %armeans2b2ng : tensor<f32>
    %v6545 = stablehlo.add %v6543, %v6544 : tensor<f32>
    %v6546 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6547 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6548 = stablehlo.multiply %v6546, %s2b2ngv : tensor<f32>
    %v6549 = stablehlo.multiply %armeans2b2ng, %armeans2b2ng : tensor<f32>
    %v6550 = stablehlo.multiply %v6547, %v6549 : tensor<f32>
    %v6551 = stablehlo.add %v6548, %v6550 : tensor<f32>
    %v6552 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6553 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6554 = stablehlo.divide %v6545, %v6552 : tensor<f32>
    %v6555 = stablehlo.divide %v6551, %v6553 : tensor<f32>
    %v6556 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6557 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6558 = stablehlo.sqrt %v6555 : tensor<f32>
    %v6559 = stablehlo.add %v6558, %v6557 : tensor<f32>
    %v6560 = stablehlo.divide %v6554, %v6559 : tensor<f32>
    %v6561 = stablehlo.multiply %v6556, %v6560 : tensor<f32>
    %v6562 = stablehlo.subtract %s2b2ng, %v6561 : tensor<f32>
    %v6563 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6564 = stablehlo.multiply %v6563, %v6556 : tensor<f32>
    %v6565 = stablehlo.multiply %v6564, %s2b2ng : tensor<f32>
    %v6566 = stablehlo.subtract %v6562, %v6565 : tensor<f32>
    %arsums2b2nbt = "stablehlo.all_reduce"(%v2323) ({
    ^bb0(%aras2b2nbt: tensor<f32>, %arbs2b2nbt: tensor<f32>):
      %aradds2b2nbt = stablehlo.add %aras2b2nbt, %arbs2b2nbt : tensor<f32>
      stablehlo.return %aradds2b2nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b2nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b2nbt = stablehlo.divide %arsums2b2nbt, %arns2b2nbt : tensor<f32>
    %v6567 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6568 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6569 = stablehlo.multiply %v6567, %s2b2nbtm : tensor<f32>
    %v6570 = stablehlo.multiply %v6568, %armeans2b2nbt : tensor<f32>
    %v6571 = stablehlo.add %v6569, %v6570 : tensor<f32>
    %v6572 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6573 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6574 = stablehlo.multiply %v6572, %s2b2nbtv : tensor<f32>
    %v6575 = stablehlo.multiply %armeans2b2nbt, %armeans2b2nbt : tensor<f32>
    %v6576 = stablehlo.multiply %v6573, %v6575 : tensor<f32>
    %v6577 = stablehlo.add %v6574, %v6576 : tensor<f32>
    %v6578 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6579 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6580 = stablehlo.multiply %v6578, %s2b2nbtm : tensor<f32>
    %v6581 = stablehlo.multiply %v6579, %armeans2b2nbt : tensor<f32>
    %v6582 = stablehlo.add %v6580, %v6581 : tensor<f32>
    %v6583 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6584 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6585 = stablehlo.multiply %v6583, %s2b2nbtv : tensor<f32>
    %v6586 = stablehlo.multiply %armeans2b2nbt, %armeans2b2nbt : tensor<f32>
    %v6587 = stablehlo.multiply %v6584, %v6586 : tensor<f32>
    %v6588 = stablehlo.add %v6585, %v6587 : tensor<f32>
    %v6589 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6590 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6591 = stablehlo.divide %v6582, %v6589 : tensor<f32>
    %v6592 = stablehlo.divide %v6588, %v6590 : tensor<f32>
    %v6593 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6594 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6595 = stablehlo.sqrt %v6592 : tensor<f32>
    %v6596 = stablehlo.add %v6595, %v6594 : tensor<f32>
    %v6597 = stablehlo.divide %v6591, %v6596 : tensor<f32>
    %v6598 = stablehlo.multiply %v6593, %v6597 : tensor<f32>
    %v6599 = stablehlo.subtract %s2b2nbt, %v6598 : tensor<f32>
    %v6600 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6601 = stablehlo.multiply %v6600, %v6593 : tensor<f32>
    %v6602 = stablehlo.multiply %v6601, %s2b2nbt : tensor<f32>
    %v6603 = stablehlo.subtract %v6599, %v6602 : tensor<f32>
    %arsums2b2eW = "stablehlo.all_reduce"(%v2302) ({
    ^bb0(%aras2b2eW: tensor<f32>, %arbs2b2eW: tensor<f32>):
      %aradds2b2eW = stablehlo.add %aras2b2eW, %arbs2b2eW : tensor<f32>
      stablehlo.return %aradds2b2eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536x384x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %arns2b2eW = stablehlo.constant dense<2.0> : tensor<1536x384x1x1xf32>
    %armeans2b2eW = stablehlo.divide %arsums2b2eW, %arns2b2eW : tensor<1536x384x1x1xf32>
    %v6604 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6605 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6606 = stablehlo.multiply %v6604, %s2b2eWm : tensor<1536x384x1x1xf32>
    %v6607 = stablehlo.multiply %v6605, %armeans2b2eW : tensor<1536x384x1x1xf32>
    %v6608 = stablehlo.add %v6606, %v6607 : tensor<1536x384x1x1xf32>
    %v6609 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6610 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6611 = stablehlo.multiply %v6609, %s2b2eWv : tensor<1536x384x1x1xf32>
    %v6612 = stablehlo.multiply %armeans2b2eW, %armeans2b2eW : tensor<1536x384x1x1xf32>
    %v6613 = stablehlo.multiply %v6610, %v6612 : tensor<1536x384x1x1xf32>
    %v6614 = stablehlo.add %v6611, %v6613 : tensor<1536x384x1x1xf32>
    %v6615 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6616 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6617 = stablehlo.multiply %v6615, %s2b2eWm : tensor<1536x384x1x1xf32>
    %v6618 = stablehlo.multiply %v6616, %armeans2b2eW : tensor<1536x384x1x1xf32>
    %v6619 = stablehlo.add %v6617, %v6618 : tensor<1536x384x1x1xf32>
    %v6620 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6621 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6622 = stablehlo.multiply %v6620, %s2b2eWv : tensor<1536x384x1x1xf32>
    %v6623 = stablehlo.multiply %armeans2b2eW, %armeans2b2eW : tensor<1536x384x1x1xf32>
    %v6624 = stablehlo.multiply %v6621, %v6623 : tensor<1536x384x1x1xf32>
    %v6625 = stablehlo.add %v6622, %v6624 : tensor<1536x384x1x1xf32>
    %v6626 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6627 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6628 = stablehlo.divide %v6619, %v6626 : tensor<1536x384x1x1xf32>
    %v6629 = stablehlo.divide %v6625, %v6627 : tensor<1536x384x1x1xf32>
    %v6630 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6631 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6632 = stablehlo.sqrt %v6629 : tensor<1536x384x1x1xf32>
    %v6633 = stablehlo.add %v6632, %v6631 : tensor<1536x384x1x1xf32>
    %v6634 = stablehlo.divide %v6628, %v6633 : tensor<1536x384x1x1xf32>
    %v6635 = stablehlo.multiply %v6630, %v6634 : tensor<1536x384x1x1xf32>
    %v6636 = stablehlo.subtract %s2b2eW, %v6635 : tensor<1536x384x1x1xf32>
    %v6637 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6638 = stablehlo.multiply %v6637, %v6630 : tensor<1536x384x1x1xf32>
    %v6639 = stablehlo.multiply %v6638, %s2b2eW : tensor<1536x384x1x1xf32>
    %v6640 = stablehlo.subtract %v6636, %v6639 : tensor<1536x384x1x1xf32>
    %arsums2b2eb = "stablehlo.all_reduce"(%v2305) ({
    ^bb0(%aras2b2eb: tensor<f32>, %arbs2b2eb: tensor<f32>):
      %aradds2b2eb = stablehlo.add %aras2b2eb, %arbs2b2eb : tensor<f32>
      stablehlo.return %aradds2b2eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536xf32>) -> tensor<1536xf32>
    %arns2b2eb = stablehlo.constant dense<2.0> : tensor<1536xf32>
    %armeans2b2eb = stablehlo.divide %arsums2b2eb, %arns2b2eb : tensor<1536xf32>
    %v6641 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6642 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6643 = stablehlo.multiply %v6641, %s2b2ebm : tensor<1536xf32>
    %v6644 = stablehlo.multiply %v6642, %armeans2b2eb : tensor<1536xf32>
    %v6645 = stablehlo.add %v6643, %v6644 : tensor<1536xf32>
    %v6646 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6647 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6648 = stablehlo.multiply %v6646, %s2b2ebv : tensor<1536xf32>
    %v6649 = stablehlo.multiply %armeans2b2eb, %armeans2b2eb : tensor<1536xf32>
    %v6650 = stablehlo.multiply %v6647, %v6649 : tensor<1536xf32>
    %v6651 = stablehlo.add %v6648, %v6650 : tensor<1536xf32>
    %v6652 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6653 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6654 = stablehlo.multiply %v6652, %s2b2ebm : tensor<1536xf32>
    %v6655 = stablehlo.multiply %v6653, %armeans2b2eb : tensor<1536xf32>
    %v6656 = stablehlo.add %v6654, %v6655 : tensor<1536xf32>
    %v6657 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6658 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6659 = stablehlo.multiply %v6657, %s2b2ebv : tensor<1536xf32>
    %v6660 = stablehlo.multiply %armeans2b2eb, %armeans2b2eb : tensor<1536xf32>
    %v6661 = stablehlo.multiply %v6658, %v6660 : tensor<1536xf32>
    %v6662 = stablehlo.add %v6659, %v6661 : tensor<1536xf32>
    %v6663 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6664 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6665 = stablehlo.divide %v6656, %v6663 : tensor<1536xf32>
    %v6666 = stablehlo.divide %v6662, %v6664 : tensor<1536xf32>
    %v6667 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6668 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6669 = stablehlo.sqrt %v6666 : tensor<1536xf32>
    %v6670 = stablehlo.add %v6669, %v6668 : tensor<1536xf32>
    %v6671 = stablehlo.divide %v6665, %v6670 : tensor<1536xf32>
    %v6672 = stablehlo.multiply %v6667, %v6671 : tensor<1536xf32>
    %v6673 = stablehlo.subtract %s2b2eb, %v6672 : tensor<1536xf32>
    %v6674 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6675 = stablehlo.multiply %v6674, %v6667 : tensor<1536xf32>
    %v6676 = stablehlo.multiply %v6675, %s2b2eb : tensor<1536xf32>
    %v6677 = stablehlo.subtract %v6673, %v6676 : tensor<1536xf32>
    %arsums2b2pW = "stablehlo.all_reduce"(%v2293) ({
    ^bb0(%aras2b2pW: tensor<f32>, %arbs2b2pW: tensor<f32>):
      %aradds2b2pW = stablehlo.add %aras2b2pW, %arbs2b2pW : tensor<f32>
      stablehlo.return %aradds2b2pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1536x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %arns2b2pW = stablehlo.constant dense<2.0> : tensor<384x1536x1x1xf32>
    %armeans2b2pW = stablehlo.divide %arsums2b2pW, %arns2b2pW : tensor<384x1536x1x1xf32>
    %v6678 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6679 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6680 = stablehlo.multiply %v6678, %s2b2pWm : tensor<384x1536x1x1xf32>
    %v6681 = stablehlo.multiply %v6679, %armeans2b2pW : tensor<384x1536x1x1xf32>
    %v6682 = stablehlo.add %v6680, %v6681 : tensor<384x1536x1x1xf32>
    %v6683 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6684 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6685 = stablehlo.multiply %v6683, %s2b2pWv : tensor<384x1536x1x1xf32>
    %v6686 = stablehlo.multiply %armeans2b2pW, %armeans2b2pW : tensor<384x1536x1x1xf32>
    %v6687 = stablehlo.multiply %v6684, %v6686 : tensor<384x1536x1x1xf32>
    %v6688 = stablehlo.add %v6685, %v6687 : tensor<384x1536x1x1xf32>
    %v6689 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6690 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6691 = stablehlo.multiply %v6689, %s2b2pWm : tensor<384x1536x1x1xf32>
    %v6692 = stablehlo.multiply %v6690, %armeans2b2pW : tensor<384x1536x1x1xf32>
    %v6693 = stablehlo.add %v6691, %v6692 : tensor<384x1536x1x1xf32>
    %v6694 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6695 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6696 = stablehlo.multiply %v6694, %s2b2pWv : tensor<384x1536x1x1xf32>
    %v6697 = stablehlo.multiply %armeans2b2pW, %armeans2b2pW : tensor<384x1536x1x1xf32>
    %v6698 = stablehlo.multiply %v6695, %v6697 : tensor<384x1536x1x1xf32>
    %v6699 = stablehlo.add %v6696, %v6698 : tensor<384x1536x1x1xf32>
    %v6700 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6701 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6702 = stablehlo.divide %v6693, %v6700 : tensor<384x1536x1x1xf32>
    %v6703 = stablehlo.divide %v6699, %v6701 : tensor<384x1536x1x1xf32>
    %v6704 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6705 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6706 = stablehlo.sqrt %v6703 : tensor<384x1536x1x1xf32>
    %v6707 = stablehlo.add %v6706, %v6705 : tensor<384x1536x1x1xf32>
    %v6708 = stablehlo.divide %v6702, %v6707 : tensor<384x1536x1x1xf32>
    %v6709 = stablehlo.multiply %v6704, %v6708 : tensor<384x1536x1x1xf32>
    %v6710 = stablehlo.subtract %s2b2pW, %v6709 : tensor<384x1536x1x1xf32>
    %v6711 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6712 = stablehlo.multiply %v6711, %v6704 : tensor<384x1536x1x1xf32>
    %v6713 = stablehlo.multiply %v6712, %s2b2pW : tensor<384x1536x1x1xf32>
    %v6714 = stablehlo.subtract %v6710, %v6713 : tensor<384x1536x1x1xf32>
    %arsums2b2pb = "stablehlo.all_reduce"(%v2296) ({
    ^bb0(%aras2b2pb: tensor<f32>, %arbs2b2pb: tensor<f32>):
      %aradds2b2pb = stablehlo.add %aras2b2pb, %arbs2b2pb : tensor<f32>
      stablehlo.return %aradds2b2pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b2pb = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b2pb = stablehlo.divide %arsums2b2pb, %arns2b2pb : tensor<384xf32>
    %v6715 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6716 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6717 = stablehlo.multiply %v6715, %s2b2pbm : tensor<384xf32>
    %v6718 = stablehlo.multiply %v6716, %armeans2b2pb : tensor<384xf32>
    %v6719 = stablehlo.add %v6717, %v6718 : tensor<384xf32>
    %v6720 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6721 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6722 = stablehlo.multiply %v6720, %s2b2pbv : tensor<384xf32>
    %v6723 = stablehlo.multiply %armeans2b2pb, %armeans2b2pb : tensor<384xf32>
    %v6724 = stablehlo.multiply %v6721, %v6723 : tensor<384xf32>
    %v6725 = stablehlo.add %v6722, %v6724 : tensor<384xf32>
    %v6726 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6727 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6728 = stablehlo.multiply %v6726, %s2b2pbm : tensor<384xf32>
    %v6729 = stablehlo.multiply %v6727, %armeans2b2pb : tensor<384xf32>
    %v6730 = stablehlo.add %v6728, %v6729 : tensor<384xf32>
    %v6731 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6732 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6733 = stablehlo.multiply %v6731, %s2b2pbv : tensor<384xf32>
    %v6734 = stablehlo.multiply %armeans2b2pb, %armeans2b2pb : tensor<384xf32>
    %v6735 = stablehlo.multiply %v6732, %v6734 : tensor<384xf32>
    %v6736 = stablehlo.add %v6733, %v6735 : tensor<384xf32>
    %v6737 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6738 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6739 = stablehlo.divide %v6730, %v6737 : tensor<384xf32>
    %v6740 = stablehlo.divide %v6736, %v6738 : tensor<384xf32>
    %v6741 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6742 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6743 = stablehlo.sqrt %v6740 : tensor<384xf32>
    %v6744 = stablehlo.add %v6743, %v6742 : tensor<384xf32>
    %v6745 = stablehlo.divide %v6739, %v6744 : tensor<384xf32>
    %v6746 = stablehlo.multiply %v6741, %v6745 : tensor<384xf32>
    %v6747 = stablehlo.subtract %s2b2pb, %v6746 : tensor<384xf32>
    %v6748 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6749 = stablehlo.multiply %v6748, %v6741 : tensor<384xf32>
    %v6750 = stablehlo.multiply %v6749, %s2b2pb : tensor<384xf32>
    %v6751 = stablehlo.subtract %v6747, %v6750 : tensor<384xf32>
    %arsums2b2lg = "stablehlo.all_reduce"(%v2287) ({
    ^bb0(%aras2b2lg: tensor<f32>, %arbs2b2lg: tensor<f32>):
      %aradds2b2lg = stablehlo.add %aras2b2lg, %arbs2b2lg : tensor<f32>
      stablehlo.return %aradds2b2lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b2lg = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b2lg = stablehlo.divide %arsums2b2lg, %arns2b2lg : tensor<384xf32>
    %v6752 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6753 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6754 = stablehlo.multiply %v6752, %s2b2lgm : tensor<384xf32>
    %v6755 = stablehlo.multiply %v6753, %armeans2b2lg : tensor<384xf32>
    %v6756 = stablehlo.add %v6754, %v6755 : tensor<384xf32>
    %v6757 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6758 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6759 = stablehlo.multiply %v6757, %s2b2lgv : tensor<384xf32>
    %v6760 = stablehlo.multiply %armeans2b2lg, %armeans2b2lg : tensor<384xf32>
    %v6761 = stablehlo.multiply %v6758, %v6760 : tensor<384xf32>
    %v6762 = stablehlo.add %v6759, %v6761 : tensor<384xf32>
    %v6763 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6764 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6765 = stablehlo.multiply %v6763, %s2b2lgm : tensor<384xf32>
    %v6766 = stablehlo.multiply %v6764, %armeans2b2lg : tensor<384xf32>
    %v6767 = stablehlo.add %v6765, %v6766 : tensor<384xf32>
    %v6768 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6769 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6770 = stablehlo.multiply %v6768, %s2b2lgv : tensor<384xf32>
    %v6771 = stablehlo.multiply %armeans2b2lg, %armeans2b2lg : tensor<384xf32>
    %v6772 = stablehlo.multiply %v6769, %v6771 : tensor<384xf32>
    %v6773 = stablehlo.add %v6770, %v6772 : tensor<384xf32>
    %v6774 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6775 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6776 = stablehlo.divide %v6767, %v6774 : tensor<384xf32>
    %v6777 = stablehlo.divide %v6773, %v6775 : tensor<384xf32>
    %v6778 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6779 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6780 = stablehlo.sqrt %v6777 : tensor<384xf32>
    %v6781 = stablehlo.add %v6780, %v6779 : tensor<384xf32>
    %v6782 = stablehlo.divide %v6776, %v6781 : tensor<384xf32>
    %v6783 = stablehlo.multiply %v6778, %v6782 : tensor<384xf32>
    %v6784 = stablehlo.subtract %s2b2lg, %v6783 : tensor<384xf32>
    %v6785 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6786 = stablehlo.multiply %v6785, %v6778 : tensor<384xf32>
    %v6787 = stablehlo.multiply %v6786, %s2b2lg : tensor<384xf32>
    %v6788 = stablehlo.subtract %v6784, %v6787 : tensor<384xf32>
    %arsums2b3dW = "stablehlo.all_reduce"(%v2210) ({
    ^bb0(%aras2b3dW: tensor<f32>, %arbs2b3dW: tensor<f32>):
      %aradds2b3dW = stablehlo.add %aras2b3dW, %arbs2b3dW : tensor<f32>
      stablehlo.return %aradds2b3dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1x7x7xf32>) -> tensor<384x1x7x7xf32>
    %arns2b3dW = stablehlo.constant dense<2.0> : tensor<384x1x7x7xf32>
    %armeans2b3dW = stablehlo.divide %arsums2b3dW, %arns2b3dW : tensor<384x1x7x7xf32>
    %v6789 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6790 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6791 = stablehlo.multiply %v6789, %s2b3dWm : tensor<384x1x7x7xf32>
    %v6792 = stablehlo.multiply %v6790, %armeans2b3dW : tensor<384x1x7x7xf32>
    %v6793 = stablehlo.add %v6791, %v6792 : tensor<384x1x7x7xf32>
    %v6794 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6795 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6796 = stablehlo.multiply %v6794, %s2b3dWv : tensor<384x1x7x7xf32>
    %v6797 = stablehlo.multiply %armeans2b3dW, %armeans2b3dW : tensor<384x1x7x7xf32>
    %v6798 = stablehlo.multiply %v6795, %v6797 : tensor<384x1x7x7xf32>
    %v6799 = stablehlo.add %v6796, %v6798 : tensor<384x1x7x7xf32>
    %v6800 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6801 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6802 = stablehlo.multiply %v6800, %s2b3dWm : tensor<384x1x7x7xf32>
    %v6803 = stablehlo.multiply %v6801, %armeans2b3dW : tensor<384x1x7x7xf32>
    %v6804 = stablehlo.add %v6802, %v6803 : tensor<384x1x7x7xf32>
    %v6805 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6806 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6807 = stablehlo.multiply %v6805, %s2b3dWv : tensor<384x1x7x7xf32>
    %v6808 = stablehlo.multiply %armeans2b3dW, %armeans2b3dW : tensor<384x1x7x7xf32>
    %v6809 = stablehlo.multiply %v6806, %v6808 : tensor<384x1x7x7xf32>
    %v6810 = stablehlo.add %v6807, %v6809 : tensor<384x1x7x7xf32>
    %v6811 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6812 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6813 = stablehlo.divide %v6804, %v6811 : tensor<384x1x7x7xf32>
    %v6814 = stablehlo.divide %v6810, %v6812 : tensor<384x1x7x7xf32>
    %v6815 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6816 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6817 = stablehlo.sqrt %v6814 : tensor<384x1x7x7xf32>
    %v6818 = stablehlo.add %v6817, %v6816 : tensor<384x1x7x7xf32>
    %v6819 = stablehlo.divide %v6813, %v6818 : tensor<384x1x7x7xf32>
    %v6820 = stablehlo.multiply %v6815, %v6819 : tensor<384x1x7x7xf32>
    %v6821 = stablehlo.subtract %s2b3dW, %v6820 : tensor<384x1x7x7xf32>
    %v6822 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6823 = stablehlo.multiply %v6822, %v6815 : tensor<384x1x7x7xf32>
    %v6824 = stablehlo.multiply %v6823, %s2b3dW : tensor<384x1x7x7xf32>
    %v6825 = stablehlo.subtract %v6821, %v6824 : tensor<384x1x7x7xf32>
    %arsums2b3db = "stablehlo.all_reduce"(%v2213) ({
    ^bb0(%aras2b3db: tensor<f32>, %arbs2b3db: tensor<f32>):
      %aradds2b3db = stablehlo.add %aras2b3db, %arbs2b3db : tensor<f32>
      stablehlo.return %aradds2b3db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b3db = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b3db = stablehlo.divide %arsums2b3db, %arns2b3db : tensor<384xf32>
    %v6826 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6827 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6828 = stablehlo.multiply %v6826, %s2b3dbm : tensor<384xf32>
    %v6829 = stablehlo.multiply %v6827, %armeans2b3db : tensor<384xf32>
    %v6830 = stablehlo.add %v6828, %v6829 : tensor<384xf32>
    %v6831 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6832 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6833 = stablehlo.multiply %v6831, %s2b3dbv : tensor<384xf32>
    %v6834 = stablehlo.multiply %armeans2b3db, %armeans2b3db : tensor<384xf32>
    %v6835 = stablehlo.multiply %v6832, %v6834 : tensor<384xf32>
    %v6836 = stablehlo.add %v6833, %v6835 : tensor<384xf32>
    %v6837 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6838 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6839 = stablehlo.multiply %v6837, %s2b3dbm : tensor<384xf32>
    %v6840 = stablehlo.multiply %v6838, %armeans2b3db : tensor<384xf32>
    %v6841 = stablehlo.add %v6839, %v6840 : tensor<384xf32>
    %v6842 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6843 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6844 = stablehlo.multiply %v6842, %s2b3dbv : tensor<384xf32>
    %v6845 = stablehlo.multiply %armeans2b3db, %armeans2b3db : tensor<384xf32>
    %v6846 = stablehlo.multiply %v6843, %v6845 : tensor<384xf32>
    %v6847 = stablehlo.add %v6844, %v6846 : tensor<384xf32>
    %v6848 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6849 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6850 = stablehlo.divide %v6841, %v6848 : tensor<384xf32>
    %v6851 = stablehlo.divide %v6847, %v6849 : tensor<384xf32>
    %v6852 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6853 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6854 = stablehlo.sqrt %v6851 : tensor<384xf32>
    %v6855 = stablehlo.add %v6854, %v6853 : tensor<384xf32>
    %v6856 = stablehlo.divide %v6850, %v6855 : tensor<384xf32>
    %v6857 = stablehlo.multiply %v6852, %v6856 : tensor<384xf32>
    %v6858 = stablehlo.subtract %s2b3db, %v6857 : tensor<384xf32>
    %v6859 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6860 = stablehlo.multiply %v6859, %v6852 : tensor<384xf32>
    %v6861 = stablehlo.multiply %v6860, %s2b3db : tensor<384xf32>
    %v6862 = stablehlo.subtract %v6858, %v6861 : tensor<384xf32>
    %arsums2b3ng = "stablehlo.all_reduce"(%v2202) ({
    ^bb0(%aras2b3ng: tensor<f32>, %arbs2b3ng: tensor<f32>):
      %aradds2b3ng = stablehlo.add %aras2b3ng, %arbs2b3ng : tensor<f32>
      stablehlo.return %aradds2b3ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b3ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b3ng = stablehlo.divide %arsums2b3ng, %arns2b3ng : tensor<f32>
    %v6863 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6864 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6865 = stablehlo.multiply %v6863, %s2b3ngm : tensor<f32>
    %v6866 = stablehlo.multiply %v6864, %armeans2b3ng : tensor<f32>
    %v6867 = stablehlo.add %v6865, %v6866 : tensor<f32>
    %v6868 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6869 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6870 = stablehlo.multiply %v6868, %s2b3ngv : tensor<f32>
    %v6871 = stablehlo.multiply %armeans2b3ng, %armeans2b3ng : tensor<f32>
    %v6872 = stablehlo.multiply %v6869, %v6871 : tensor<f32>
    %v6873 = stablehlo.add %v6870, %v6872 : tensor<f32>
    %v6874 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6875 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6876 = stablehlo.multiply %v6874, %s2b3ngm : tensor<f32>
    %v6877 = stablehlo.multiply %v6875, %armeans2b3ng : tensor<f32>
    %v6878 = stablehlo.add %v6876, %v6877 : tensor<f32>
    %v6879 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6880 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6881 = stablehlo.multiply %v6879, %s2b3ngv : tensor<f32>
    %v6882 = stablehlo.multiply %armeans2b3ng, %armeans2b3ng : tensor<f32>
    %v6883 = stablehlo.multiply %v6880, %v6882 : tensor<f32>
    %v6884 = stablehlo.add %v6881, %v6883 : tensor<f32>
    %v6885 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6886 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6887 = stablehlo.divide %v6878, %v6885 : tensor<f32>
    %v6888 = stablehlo.divide %v6884, %v6886 : tensor<f32>
    %v6889 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6890 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6891 = stablehlo.sqrt %v6888 : tensor<f32>
    %v6892 = stablehlo.add %v6891, %v6890 : tensor<f32>
    %v6893 = stablehlo.divide %v6887, %v6892 : tensor<f32>
    %v6894 = stablehlo.multiply %v6889, %v6893 : tensor<f32>
    %v6895 = stablehlo.subtract %s2b3ng, %v6894 : tensor<f32>
    %v6896 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6897 = stablehlo.multiply %v6896, %v6889 : tensor<f32>
    %v6898 = stablehlo.multiply %v6897, %s2b3ng : tensor<f32>
    %v6899 = stablehlo.subtract %v6895, %v6898 : tensor<f32>
    %arsums2b3nbt = "stablehlo.all_reduce"(%v2204) ({
    ^bb0(%aras2b3nbt: tensor<f32>, %arbs2b3nbt: tensor<f32>):
      %aradds2b3nbt = stablehlo.add %aras2b3nbt, %arbs2b3nbt : tensor<f32>
      stablehlo.return %aradds2b3nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b3nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b3nbt = stablehlo.divide %arsums2b3nbt, %arns2b3nbt : tensor<f32>
    %v6900 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6901 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6902 = stablehlo.multiply %v6900, %s2b3nbtm : tensor<f32>
    %v6903 = stablehlo.multiply %v6901, %armeans2b3nbt : tensor<f32>
    %v6904 = stablehlo.add %v6902, %v6903 : tensor<f32>
    %v6905 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6906 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6907 = stablehlo.multiply %v6905, %s2b3nbtv : tensor<f32>
    %v6908 = stablehlo.multiply %armeans2b3nbt, %armeans2b3nbt : tensor<f32>
    %v6909 = stablehlo.multiply %v6906, %v6908 : tensor<f32>
    %v6910 = stablehlo.add %v6907, %v6909 : tensor<f32>
    %v6911 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6912 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6913 = stablehlo.multiply %v6911, %s2b3nbtm : tensor<f32>
    %v6914 = stablehlo.multiply %v6912, %armeans2b3nbt : tensor<f32>
    %v6915 = stablehlo.add %v6913, %v6914 : tensor<f32>
    %v6916 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6917 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6918 = stablehlo.multiply %v6916, %s2b3nbtv : tensor<f32>
    %v6919 = stablehlo.multiply %armeans2b3nbt, %armeans2b3nbt : tensor<f32>
    %v6920 = stablehlo.multiply %v6917, %v6919 : tensor<f32>
    %v6921 = stablehlo.add %v6918, %v6920 : tensor<f32>
    %v6922 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6923 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6924 = stablehlo.divide %v6915, %v6922 : tensor<f32>
    %v6925 = stablehlo.divide %v6921, %v6923 : tensor<f32>
    %v6926 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6927 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6928 = stablehlo.sqrt %v6925 : tensor<f32>
    %v6929 = stablehlo.add %v6928, %v6927 : tensor<f32>
    %v6930 = stablehlo.divide %v6924, %v6929 : tensor<f32>
    %v6931 = stablehlo.multiply %v6926, %v6930 : tensor<f32>
    %v6932 = stablehlo.subtract %s2b3nbt, %v6931 : tensor<f32>
    %v6933 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6934 = stablehlo.multiply %v6933, %v6926 : tensor<f32>
    %v6935 = stablehlo.multiply %v6934, %s2b3nbt : tensor<f32>
    %v6936 = stablehlo.subtract %v6932, %v6935 : tensor<f32>
    %arsums2b3eW = "stablehlo.all_reduce"(%v2183) ({
    ^bb0(%aras2b3eW: tensor<f32>, %arbs2b3eW: tensor<f32>):
      %aradds2b3eW = stablehlo.add %aras2b3eW, %arbs2b3eW : tensor<f32>
      stablehlo.return %aradds2b3eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536x384x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %arns2b3eW = stablehlo.constant dense<2.0> : tensor<1536x384x1x1xf32>
    %armeans2b3eW = stablehlo.divide %arsums2b3eW, %arns2b3eW : tensor<1536x384x1x1xf32>
    %v6937 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6938 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6939 = stablehlo.multiply %v6937, %s2b3eWm : tensor<1536x384x1x1xf32>
    %v6940 = stablehlo.multiply %v6938, %armeans2b3eW : tensor<1536x384x1x1xf32>
    %v6941 = stablehlo.add %v6939, %v6940 : tensor<1536x384x1x1xf32>
    %v6942 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6943 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6944 = stablehlo.multiply %v6942, %s2b3eWv : tensor<1536x384x1x1xf32>
    %v6945 = stablehlo.multiply %armeans2b3eW, %armeans2b3eW : tensor<1536x384x1x1xf32>
    %v6946 = stablehlo.multiply %v6943, %v6945 : tensor<1536x384x1x1xf32>
    %v6947 = stablehlo.add %v6944, %v6946 : tensor<1536x384x1x1xf32>
    %v6948 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6949 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6950 = stablehlo.multiply %v6948, %s2b3eWm : tensor<1536x384x1x1xf32>
    %v6951 = stablehlo.multiply %v6949, %armeans2b3eW : tensor<1536x384x1x1xf32>
    %v6952 = stablehlo.add %v6950, %v6951 : tensor<1536x384x1x1xf32>
    %v6953 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6954 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6955 = stablehlo.multiply %v6953, %s2b3eWv : tensor<1536x384x1x1xf32>
    %v6956 = stablehlo.multiply %armeans2b3eW, %armeans2b3eW : tensor<1536x384x1x1xf32>
    %v6957 = stablehlo.multiply %v6954, %v6956 : tensor<1536x384x1x1xf32>
    %v6958 = stablehlo.add %v6955, %v6957 : tensor<1536x384x1x1xf32>
    %v6959 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6960 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6961 = stablehlo.divide %v6952, %v6959 : tensor<1536x384x1x1xf32>
    %v6962 = stablehlo.divide %v6958, %v6960 : tensor<1536x384x1x1xf32>
    %v6963 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6964 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6965 = stablehlo.sqrt %v6962 : tensor<1536x384x1x1xf32>
    %v6966 = stablehlo.add %v6965, %v6964 : tensor<1536x384x1x1xf32>
    %v6967 = stablehlo.divide %v6961, %v6966 : tensor<1536x384x1x1xf32>
    %v6968 = stablehlo.multiply %v6963, %v6967 : tensor<1536x384x1x1xf32>
    %v6969 = stablehlo.subtract %s2b3eW, %v6968 : tensor<1536x384x1x1xf32>
    %v6970 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6971 = stablehlo.multiply %v6970, %v6963 : tensor<1536x384x1x1xf32>
    %v6972 = stablehlo.multiply %v6971, %s2b3eW : tensor<1536x384x1x1xf32>
    %v6973 = stablehlo.subtract %v6969, %v6972 : tensor<1536x384x1x1xf32>
    %arsums2b3eb = "stablehlo.all_reduce"(%v2186) ({
    ^bb0(%aras2b3eb: tensor<f32>, %arbs2b3eb: tensor<f32>):
      %aradds2b3eb = stablehlo.add %aras2b3eb, %arbs2b3eb : tensor<f32>
      stablehlo.return %aradds2b3eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536xf32>) -> tensor<1536xf32>
    %arns2b3eb = stablehlo.constant dense<2.0> : tensor<1536xf32>
    %armeans2b3eb = stablehlo.divide %arsums2b3eb, %arns2b3eb : tensor<1536xf32>
    %v6974 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6975 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6976 = stablehlo.multiply %v6974, %s2b3ebm : tensor<1536xf32>
    %v6977 = stablehlo.multiply %v6975, %armeans2b3eb : tensor<1536xf32>
    %v6978 = stablehlo.add %v6976, %v6977 : tensor<1536xf32>
    %v6979 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6980 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6981 = stablehlo.multiply %v6979, %s2b3ebv : tensor<1536xf32>
    %v6982 = stablehlo.multiply %armeans2b3eb, %armeans2b3eb : tensor<1536xf32>
    %v6983 = stablehlo.multiply %v6980, %v6982 : tensor<1536xf32>
    %v6984 = stablehlo.add %v6981, %v6983 : tensor<1536xf32>
    %v6985 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6986 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6987 = stablehlo.multiply %v6985, %s2b3ebm : tensor<1536xf32>
    %v6988 = stablehlo.multiply %v6986, %armeans2b3eb : tensor<1536xf32>
    %v6989 = stablehlo.add %v6987, %v6988 : tensor<1536xf32>
    %v6990 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6991 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6992 = stablehlo.multiply %v6990, %s2b3ebv : tensor<1536xf32>
    %v6993 = stablehlo.multiply %armeans2b3eb, %armeans2b3eb : tensor<1536xf32>
    %v6994 = stablehlo.multiply %v6991, %v6993 : tensor<1536xf32>
    %v6995 = stablehlo.add %v6992, %v6994 : tensor<1536xf32>
    %v6996 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6997 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6998 = stablehlo.divide %v6989, %v6996 : tensor<1536xf32>
    %v6999 = stablehlo.divide %v6995, %v6997 : tensor<1536xf32>
    %v7000 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7001 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7002 = stablehlo.sqrt %v6999 : tensor<1536xf32>
    %v7003 = stablehlo.add %v7002, %v7001 : tensor<1536xf32>
    %v7004 = stablehlo.divide %v6998, %v7003 : tensor<1536xf32>
    %v7005 = stablehlo.multiply %v7000, %v7004 : tensor<1536xf32>
    %v7006 = stablehlo.subtract %s2b3eb, %v7005 : tensor<1536xf32>
    %v7007 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7008 = stablehlo.multiply %v7007, %v7000 : tensor<1536xf32>
    %v7009 = stablehlo.multiply %v7008, %s2b3eb : tensor<1536xf32>
    %v7010 = stablehlo.subtract %v7006, %v7009 : tensor<1536xf32>
    %arsums2b3pW = "stablehlo.all_reduce"(%v2174) ({
    ^bb0(%aras2b3pW: tensor<f32>, %arbs2b3pW: tensor<f32>):
      %aradds2b3pW = stablehlo.add %aras2b3pW, %arbs2b3pW : tensor<f32>
      stablehlo.return %aradds2b3pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1536x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %arns2b3pW = stablehlo.constant dense<2.0> : tensor<384x1536x1x1xf32>
    %armeans2b3pW = stablehlo.divide %arsums2b3pW, %arns2b3pW : tensor<384x1536x1x1xf32>
    %v7011 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7012 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7013 = stablehlo.multiply %v7011, %s2b3pWm : tensor<384x1536x1x1xf32>
    %v7014 = stablehlo.multiply %v7012, %armeans2b3pW : tensor<384x1536x1x1xf32>
    %v7015 = stablehlo.add %v7013, %v7014 : tensor<384x1536x1x1xf32>
    %v7016 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7017 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7018 = stablehlo.multiply %v7016, %s2b3pWv : tensor<384x1536x1x1xf32>
    %v7019 = stablehlo.multiply %armeans2b3pW, %armeans2b3pW : tensor<384x1536x1x1xf32>
    %v7020 = stablehlo.multiply %v7017, %v7019 : tensor<384x1536x1x1xf32>
    %v7021 = stablehlo.add %v7018, %v7020 : tensor<384x1536x1x1xf32>
    %v7022 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7023 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7024 = stablehlo.multiply %v7022, %s2b3pWm : tensor<384x1536x1x1xf32>
    %v7025 = stablehlo.multiply %v7023, %armeans2b3pW : tensor<384x1536x1x1xf32>
    %v7026 = stablehlo.add %v7024, %v7025 : tensor<384x1536x1x1xf32>
    %v7027 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7028 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7029 = stablehlo.multiply %v7027, %s2b3pWv : tensor<384x1536x1x1xf32>
    %v7030 = stablehlo.multiply %armeans2b3pW, %armeans2b3pW : tensor<384x1536x1x1xf32>
    %v7031 = stablehlo.multiply %v7028, %v7030 : tensor<384x1536x1x1xf32>
    %v7032 = stablehlo.add %v7029, %v7031 : tensor<384x1536x1x1xf32>
    %v7033 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7034 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7035 = stablehlo.divide %v7026, %v7033 : tensor<384x1536x1x1xf32>
    %v7036 = stablehlo.divide %v7032, %v7034 : tensor<384x1536x1x1xf32>
    %v7037 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7038 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7039 = stablehlo.sqrt %v7036 : tensor<384x1536x1x1xf32>
    %v7040 = stablehlo.add %v7039, %v7038 : tensor<384x1536x1x1xf32>
    %v7041 = stablehlo.divide %v7035, %v7040 : tensor<384x1536x1x1xf32>
    %v7042 = stablehlo.multiply %v7037, %v7041 : tensor<384x1536x1x1xf32>
    %v7043 = stablehlo.subtract %s2b3pW, %v7042 : tensor<384x1536x1x1xf32>
    %v7044 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7045 = stablehlo.multiply %v7044, %v7037 : tensor<384x1536x1x1xf32>
    %v7046 = stablehlo.multiply %v7045, %s2b3pW : tensor<384x1536x1x1xf32>
    %v7047 = stablehlo.subtract %v7043, %v7046 : tensor<384x1536x1x1xf32>
    %arsums2b3pb = "stablehlo.all_reduce"(%v2177) ({
    ^bb0(%aras2b3pb: tensor<f32>, %arbs2b3pb: tensor<f32>):
      %aradds2b3pb = stablehlo.add %aras2b3pb, %arbs2b3pb : tensor<f32>
      stablehlo.return %aradds2b3pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b3pb = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b3pb = stablehlo.divide %arsums2b3pb, %arns2b3pb : tensor<384xf32>
    %v7048 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7049 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7050 = stablehlo.multiply %v7048, %s2b3pbm : tensor<384xf32>
    %v7051 = stablehlo.multiply %v7049, %armeans2b3pb : tensor<384xf32>
    %v7052 = stablehlo.add %v7050, %v7051 : tensor<384xf32>
    %v7053 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7054 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7055 = stablehlo.multiply %v7053, %s2b3pbv : tensor<384xf32>
    %v7056 = stablehlo.multiply %armeans2b3pb, %armeans2b3pb : tensor<384xf32>
    %v7057 = stablehlo.multiply %v7054, %v7056 : tensor<384xf32>
    %v7058 = stablehlo.add %v7055, %v7057 : tensor<384xf32>
    %v7059 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7060 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7061 = stablehlo.multiply %v7059, %s2b3pbm : tensor<384xf32>
    %v7062 = stablehlo.multiply %v7060, %armeans2b3pb : tensor<384xf32>
    %v7063 = stablehlo.add %v7061, %v7062 : tensor<384xf32>
    %v7064 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7065 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7066 = stablehlo.multiply %v7064, %s2b3pbv : tensor<384xf32>
    %v7067 = stablehlo.multiply %armeans2b3pb, %armeans2b3pb : tensor<384xf32>
    %v7068 = stablehlo.multiply %v7065, %v7067 : tensor<384xf32>
    %v7069 = stablehlo.add %v7066, %v7068 : tensor<384xf32>
    %v7070 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7071 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7072 = stablehlo.divide %v7063, %v7070 : tensor<384xf32>
    %v7073 = stablehlo.divide %v7069, %v7071 : tensor<384xf32>
    %v7074 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7075 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7076 = stablehlo.sqrt %v7073 : tensor<384xf32>
    %v7077 = stablehlo.add %v7076, %v7075 : tensor<384xf32>
    %v7078 = stablehlo.divide %v7072, %v7077 : tensor<384xf32>
    %v7079 = stablehlo.multiply %v7074, %v7078 : tensor<384xf32>
    %v7080 = stablehlo.subtract %s2b3pb, %v7079 : tensor<384xf32>
    %v7081 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7082 = stablehlo.multiply %v7081, %v7074 : tensor<384xf32>
    %v7083 = stablehlo.multiply %v7082, %s2b3pb : tensor<384xf32>
    %v7084 = stablehlo.subtract %v7080, %v7083 : tensor<384xf32>
    %arsums2b3lg = "stablehlo.all_reduce"(%v2168) ({
    ^bb0(%aras2b3lg: tensor<f32>, %arbs2b3lg: tensor<f32>):
      %aradds2b3lg = stablehlo.add %aras2b3lg, %arbs2b3lg : tensor<f32>
      stablehlo.return %aradds2b3lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b3lg = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b3lg = stablehlo.divide %arsums2b3lg, %arns2b3lg : tensor<384xf32>
    %v7085 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7086 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7087 = stablehlo.multiply %v7085, %s2b3lgm : tensor<384xf32>
    %v7088 = stablehlo.multiply %v7086, %armeans2b3lg : tensor<384xf32>
    %v7089 = stablehlo.add %v7087, %v7088 : tensor<384xf32>
    %v7090 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7091 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7092 = stablehlo.multiply %v7090, %s2b3lgv : tensor<384xf32>
    %v7093 = stablehlo.multiply %armeans2b3lg, %armeans2b3lg : tensor<384xf32>
    %v7094 = stablehlo.multiply %v7091, %v7093 : tensor<384xf32>
    %v7095 = stablehlo.add %v7092, %v7094 : tensor<384xf32>
    %v7096 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7097 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7098 = stablehlo.multiply %v7096, %s2b3lgm : tensor<384xf32>
    %v7099 = stablehlo.multiply %v7097, %armeans2b3lg : tensor<384xf32>
    %v7100 = stablehlo.add %v7098, %v7099 : tensor<384xf32>
    %v7101 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7102 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7103 = stablehlo.multiply %v7101, %s2b3lgv : tensor<384xf32>
    %v7104 = stablehlo.multiply %armeans2b3lg, %armeans2b3lg : tensor<384xf32>
    %v7105 = stablehlo.multiply %v7102, %v7104 : tensor<384xf32>
    %v7106 = stablehlo.add %v7103, %v7105 : tensor<384xf32>
    %v7107 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7108 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7109 = stablehlo.divide %v7100, %v7107 : tensor<384xf32>
    %v7110 = stablehlo.divide %v7106, %v7108 : tensor<384xf32>
    %v7111 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7112 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7113 = stablehlo.sqrt %v7110 : tensor<384xf32>
    %v7114 = stablehlo.add %v7113, %v7112 : tensor<384xf32>
    %v7115 = stablehlo.divide %v7109, %v7114 : tensor<384xf32>
    %v7116 = stablehlo.multiply %v7111, %v7115 : tensor<384xf32>
    %v7117 = stablehlo.subtract %s2b3lg, %v7116 : tensor<384xf32>
    %v7118 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7119 = stablehlo.multiply %v7118, %v7111 : tensor<384xf32>
    %v7120 = stablehlo.multiply %v7119, %s2b3lg : tensor<384xf32>
    %v7121 = stablehlo.subtract %v7117, %v7120 : tensor<384xf32>
    %arsums2b4dW = "stablehlo.all_reduce"(%v2091) ({
    ^bb0(%aras2b4dW: tensor<f32>, %arbs2b4dW: tensor<f32>):
      %aradds2b4dW = stablehlo.add %aras2b4dW, %arbs2b4dW : tensor<f32>
      stablehlo.return %aradds2b4dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1x7x7xf32>) -> tensor<384x1x7x7xf32>
    %arns2b4dW = stablehlo.constant dense<2.0> : tensor<384x1x7x7xf32>
    %armeans2b4dW = stablehlo.divide %arsums2b4dW, %arns2b4dW : tensor<384x1x7x7xf32>
    %v7122 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7123 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7124 = stablehlo.multiply %v7122, %s2b4dWm : tensor<384x1x7x7xf32>
    %v7125 = stablehlo.multiply %v7123, %armeans2b4dW : tensor<384x1x7x7xf32>
    %v7126 = stablehlo.add %v7124, %v7125 : tensor<384x1x7x7xf32>
    %v7127 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7128 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7129 = stablehlo.multiply %v7127, %s2b4dWv : tensor<384x1x7x7xf32>
    %v7130 = stablehlo.multiply %armeans2b4dW, %armeans2b4dW : tensor<384x1x7x7xf32>
    %v7131 = stablehlo.multiply %v7128, %v7130 : tensor<384x1x7x7xf32>
    %v7132 = stablehlo.add %v7129, %v7131 : tensor<384x1x7x7xf32>
    %v7133 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7134 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7135 = stablehlo.multiply %v7133, %s2b4dWm : tensor<384x1x7x7xf32>
    %v7136 = stablehlo.multiply %v7134, %armeans2b4dW : tensor<384x1x7x7xf32>
    %v7137 = stablehlo.add %v7135, %v7136 : tensor<384x1x7x7xf32>
    %v7138 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7139 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7140 = stablehlo.multiply %v7138, %s2b4dWv : tensor<384x1x7x7xf32>
    %v7141 = stablehlo.multiply %armeans2b4dW, %armeans2b4dW : tensor<384x1x7x7xf32>
    %v7142 = stablehlo.multiply %v7139, %v7141 : tensor<384x1x7x7xf32>
    %v7143 = stablehlo.add %v7140, %v7142 : tensor<384x1x7x7xf32>
    %v7144 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7145 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7146 = stablehlo.divide %v7137, %v7144 : tensor<384x1x7x7xf32>
    %v7147 = stablehlo.divide %v7143, %v7145 : tensor<384x1x7x7xf32>
    %v7148 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7149 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7150 = stablehlo.sqrt %v7147 : tensor<384x1x7x7xf32>
    %v7151 = stablehlo.add %v7150, %v7149 : tensor<384x1x7x7xf32>
    %v7152 = stablehlo.divide %v7146, %v7151 : tensor<384x1x7x7xf32>
    %v7153 = stablehlo.multiply %v7148, %v7152 : tensor<384x1x7x7xf32>
    %v7154 = stablehlo.subtract %s2b4dW, %v7153 : tensor<384x1x7x7xf32>
    %v7155 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7156 = stablehlo.multiply %v7155, %v7148 : tensor<384x1x7x7xf32>
    %v7157 = stablehlo.multiply %v7156, %s2b4dW : tensor<384x1x7x7xf32>
    %v7158 = stablehlo.subtract %v7154, %v7157 : tensor<384x1x7x7xf32>
    %arsums2b4db = "stablehlo.all_reduce"(%v2094) ({
    ^bb0(%aras2b4db: tensor<f32>, %arbs2b4db: tensor<f32>):
      %aradds2b4db = stablehlo.add %aras2b4db, %arbs2b4db : tensor<f32>
      stablehlo.return %aradds2b4db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b4db = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b4db = stablehlo.divide %arsums2b4db, %arns2b4db : tensor<384xf32>
    %v7159 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7160 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7161 = stablehlo.multiply %v7159, %s2b4dbm : tensor<384xf32>
    %v7162 = stablehlo.multiply %v7160, %armeans2b4db : tensor<384xf32>
    %v7163 = stablehlo.add %v7161, %v7162 : tensor<384xf32>
    %v7164 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7165 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7166 = stablehlo.multiply %v7164, %s2b4dbv : tensor<384xf32>
    %v7167 = stablehlo.multiply %armeans2b4db, %armeans2b4db : tensor<384xf32>
    %v7168 = stablehlo.multiply %v7165, %v7167 : tensor<384xf32>
    %v7169 = stablehlo.add %v7166, %v7168 : tensor<384xf32>
    %v7170 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7171 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7172 = stablehlo.multiply %v7170, %s2b4dbm : tensor<384xf32>
    %v7173 = stablehlo.multiply %v7171, %armeans2b4db : tensor<384xf32>
    %v7174 = stablehlo.add %v7172, %v7173 : tensor<384xf32>
    %v7175 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7176 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7177 = stablehlo.multiply %v7175, %s2b4dbv : tensor<384xf32>
    %v7178 = stablehlo.multiply %armeans2b4db, %armeans2b4db : tensor<384xf32>
    %v7179 = stablehlo.multiply %v7176, %v7178 : tensor<384xf32>
    %v7180 = stablehlo.add %v7177, %v7179 : tensor<384xf32>
    %v7181 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7182 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7183 = stablehlo.divide %v7174, %v7181 : tensor<384xf32>
    %v7184 = stablehlo.divide %v7180, %v7182 : tensor<384xf32>
    %v7185 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7186 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7187 = stablehlo.sqrt %v7184 : tensor<384xf32>
    %v7188 = stablehlo.add %v7187, %v7186 : tensor<384xf32>
    %v7189 = stablehlo.divide %v7183, %v7188 : tensor<384xf32>
    %v7190 = stablehlo.multiply %v7185, %v7189 : tensor<384xf32>
    %v7191 = stablehlo.subtract %s2b4db, %v7190 : tensor<384xf32>
    %v7192 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7193 = stablehlo.multiply %v7192, %v7185 : tensor<384xf32>
    %v7194 = stablehlo.multiply %v7193, %s2b4db : tensor<384xf32>
    %v7195 = stablehlo.subtract %v7191, %v7194 : tensor<384xf32>
    %arsums2b4ng = "stablehlo.all_reduce"(%v2083) ({
    ^bb0(%aras2b4ng: tensor<f32>, %arbs2b4ng: tensor<f32>):
      %aradds2b4ng = stablehlo.add %aras2b4ng, %arbs2b4ng : tensor<f32>
      stablehlo.return %aradds2b4ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b4ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b4ng = stablehlo.divide %arsums2b4ng, %arns2b4ng : tensor<f32>
    %v7196 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7197 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7198 = stablehlo.multiply %v7196, %s2b4ngm : tensor<f32>
    %v7199 = stablehlo.multiply %v7197, %armeans2b4ng : tensor<f32>
    %v7200 = stablehlo.add %v7198, %v7199 : tensor<f32>
    %v7201 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7202 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7203 = stablehlo.multiply %v7201, %s2b4ngv : tensor<f32>
    %v7204 = stablehlo.multiply %armeans2b4ng, %armeans2b4ng : tensor<f32>
    %v7205 = stablehlo.multiply %v7202, %v7204 : tensor<f32>
    %v7206 = stablehlo.add %v7203, %v7205 : tensor<f32>
    %v7207 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7208 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7209 = stablehlo.multiply %v7207, %s2b4ngm : tensor<f32>
    %v7210 = stablehlo.multiply %v7208, %armeans2b4ng : tensor<f32>
    %v7211 = stablehlo.add %v7209, %v7210 : tensor<f32>
    %v7212 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7213 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7214 = stablehlo.multiply %v7212, %s2b4ngv : tensor<f32>
    %v7215 = stablehlo.multiply %armeans2b4ng, %armeans2b4ng : tensor<f32>
    %v7216 = stablehlo.multiply %v7213, %v7215 : tensor<f32>
    %v7217 = stablehlo.add %v7214, %v7216 : tensor<f32>
    %v7218 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7219 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7220 = stablehlo.divide %v7211, %v7218 : tensor<f32>
    %v7221 = stablehlo.divide %v7217, %v7219 : tensor<f32>
    %v7222 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7223 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7224 = stablehlo.sqrt %v7221 : tensor<f32>
    %v7225 = stablehlo.add %v7224, %v7223 : tensor<f32>
    %v7226 = stablehlo.divide %v7220, %v7225 : tensor<f32>
    %v7227 = stablehlo.multiply %v7222, %v7226 : tensor<f32>
    %v7228 = stablehlo.subtract %s2b4ng, %v7227 : tensor<f32>
    %v7229 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7230 = stablehlo.multiply %v7229, %v7222 : tensor<f32>
    %v7231 = stablehlo.multiply %v7230, %s2b4ng : tensor<f32>
    %v7232 = stablehlo.subtract %v7228, %v7231 : tensor<f32>
    %arsums2b4nbt = "stablehlo.all_reduce"(%v2085) ({
    ^bb0(%aras2b4nbt: tensor<f32>, %arbs2b4nbt: tensor<f32>):
      %aradds2b4nbt = stablehlo.add %aras2b4nbt, %arbs2b4nbt : tensor<f32>
      stablehlo.return %aradds2b4nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b4nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b4nbt = stablehlo.divide %arsums2b4nbt, %arns2b4nbt : tensor<f32>
    %v7233 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7234 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7235 = stablehlo.multiply %v7233, %s2b4nbtm : tensor<f32>
    %v7236 = stablehlo.multiply %v7234, %armeans2b4nbt : tensor<f32>
    %v7237 = stablehlo.add %v7235, %v7236 : tensor<f32>
    %v7238 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7239 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7240 = stablehlo.multiply %v7238, %s2b4nbtv : tensor<f32>
    %v7241 = stablehlo.multiply %armeans2b4nbt, %armeans2b4nbt : tensor<f32>
    %v7242 = stablehlo.multiply %v7239, %v7241 : tensor<f32>
    %v7243 = stablehlo.add %v7240, %v7242 : tensor<f32>
    %v7244 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7245 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7246 = stablehlo.multiply %v7244, %s2b4nbtm : tensor<f32>
    %v7247 = stablehlo.multiply %v7245, %armeans2b4nbt : tensor<f32>
    %v7248 = stablehlo.add %v7246, %v7247 : tensor<f32>
    %v7249 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7250 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7251 = stablehlo.multiply %v7249, %s2b4nbtv : tensor<f32>
    %v7252 = stablehlo.multiply %armeans2b4nbt, %armeans2b4nbt : tensor<f32>
    %v7253 = stablehlo.multiply %v7250, %v7252 : tensor<f32>
    %v7254 = stablehlo.add %v7251, %v7253 : tensor<f32>
    %v7255 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7256 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7257 = stablehlo.divide %v7248, %v7255 : tensor<f32>
    %v7258 = stablehlo.divide %v7254, %v7256 : tensor<f32>
    %v7259 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7260 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7261 = stablehlo.sqrt %v7258 : tensor<f32>
    %v7262 = stablehlo.add %v7261, %v7260 : tensor<f32>
    %v7263 = stablehlo.divide %v7257, %v7262 : tensor<f32>
    %v7264 = stablehlo.multiply %v7259, %v7263 : tensor<f32>
    %v7265 = stablehlo.subtract %s2b4nbt, %v7264 : tensor<f32>
    %v7266 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7267 = stablehlo.multiply %v7266, %v7259 : tensor<f32>
    %v7268 = stablehlo.multiply %v7267, %s2b4nbt : tensor<f32>
    %v7269 = stablehlo.subtract %v7265, %v7268 : tensor<f32>
    %arsums2b4eW = "stablehlo.all_reduce"(%v2064) ({
    ^bb0(%aras2b4eW: tensor<f32>, %arbs2b4eW: tensor<f32>):
      %aradds2b4eW = stablehlo.add %aras2b4eW, %arbs2b4eW : tensor<f32>
      stablehlo.return %aradds2b4eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536x384x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %arns2b4eW = stablehlo.constant dense<2.0> : tensor<1536x384x1x1xf32>
    %armeans2b4eW = stablehlo.divide %arsums2b4eW, %arns2b4eW : tensor<1536x384x1x1xf32>
    %v7270 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7271 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7272 = stablehlo.multiply %v7270, %s2b4eWm : tensor<1536x384x1x1xf32>
    %v7273 = stablehlo.multiply %v7271, %armeans2b4eW : tensor<1536x384x1x1xf32>
    %v7274 = stablehlo.add %v7272, %v7273 : tensor<1536x384x1x1xf32>
    %v7275 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7276 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7277 = stablehlo.multiply %v7275, %s2b4eWv : tensor<1536x384x1x1xf32>
    %v7278 = stablehlo.multiply %armeans2b4eW, %armeans2b4eW : tensor<1536x384x1x1xf32>
    %v7279 = stablehlo.multiply %v7276, %v7278 : tensor<1536x384x1x1xf32>
    %v7280 = stablehlo.add %v7277, %v7279 : tensor<1536x384x1x1xf32>
    %v7281 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7282 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7283 = stablehlo.multiply %v7281, %s2b4eWm : tensor<1536x384x1x1xf32>
    %v7284 = stablehlo.multiply %v7282, %armeans2b4eW : tensor<1536x384x1x1xf32>
    %v7285 = stablehlo.add %v7283, %v7284 : tensor<1536x384x1x1xf32>
    %v7286 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7287 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7288 = stablehlo.multiply %v7286, %s2b4eWv : tensor<1536x384x1x1xf32>
    %v7289 = stablehlo.multiply %armeans2b4eW, %armeans2b4eW : tensor<1536x384x1x1xf32>
    %v7290 = stablehlo.multiply %v7287, %v7289 : tensor<1536x384x1x1xf32>
    %v7291 = stablehlo.add %v7288, %v7290 : tensor<1536x384x1x1xf32>
    %v7292 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7293 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7294 = stablehlo.divide %v7285, %v7292 : tensor<1536x384x1x1xf32>
    %v7295 = stablehlo.divide %v7291, %v7293 : tensor<1536x384x1x1xf32>
    %v7296 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7297 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7298 = stablehlo.sqrt %v7295 : tensor<1536x384x1x1xf32>
    %v7299 = stablehlo.add %v7298, %v7297 : tensor<1536x384x1x1xf32>
    %v7300 = stablehlo.divide %v7294, %v7299 : tensor<1536x384x1x1xf32>
    %v7301 = stablehlo.multiply %v7296, %v7300 : tensor<1536x384x1x1xf32>
    %v7302 = stablehlo.subtract %s2b4eW, %v7301 : tensor<1536x384x1x1xf32>
    %v7303 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7304 = stablehlo.multiply %v7303, %v7296 : tensor<1536x384x1x1xf32>
    %v7305 = stablehlo.multiply %v7304, %s2b4eW : tensor<1536x384x1x1xf32>
    %v7306 = stablehlo.subtract %v7302, %v7305 : tensor<1536x384x1x1xf32>
    %arsums2b4eb = "stablehlo.all_reduce"(%v2067) ({
    ^bb0(%aras2b4eb: tensor<f32>, %arbs2b4eb: tensor<f32>):
      %aradds2b4eb = stablehlo.add %aras2b4eb, %arbs2b4eb : tensor<f32>
      stablehlo.return %aradds2b4eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536xf32>) -> tensor<1536xf32>
    %arns2b4eb = stablehlo.constant dense<2.0> : tensor<1536xf32>
    %armeans2b4eb = stablehlo.divide %arsums2b4eb, %arns2b4eb : tensor<1536xf32>
    %v7307 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7308 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7309 = stablehlo.multiply %v7307, %s2b4ebm : tensor<1536xf32>
    %v7310 = stablehlo.multiply %v7308, %armeans2b4eb : tensor<1536xf32>
    %v7311 = stablehlo.add %v7309, %v7310 : tensor<1536xf32>
    %v7312 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7313 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7314 = stablehlo.multiply %v7312, %s2b4ebv : tensor<1536xf32>
    %v7315 = stablehlo.multiply %armeans2b4eb, %armeans2b4eb : tensor<1536xf32>
    %v7316 = stablehlo.multiply %v7313, %v7315 : tensor<1536xf32>
    %v7317 = stablehlo.add %v7314, %v7316 : tensor<1536xf32>
    %v7318 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7319 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7320 = stablehlo.multiply %v7318, %s2b4ebm : tensor<1536xf32>
    %v7321 = stablehlo.multiply %v7319, %armeans2b4eb : tensor<1536xf32>
    %v7322 = stablehlo.add %v7320, %v7321 : tensor<1536xf32>
    %v7323 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7324 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7325 = stablehlo.multiply %v7323, %s2b4ebv : tensor<1536xf32>
    %v7326 = stablehlo.multiply %armeans2b4eb, %armeans2b4eb : tensor<1536xf32>
    %v7327 = stablehlo.multiply %v7324, %v7326 : tensor<1536xf32>
    %v7328 = stablehlo.add %v7325, %v7327 : tensor<1536xf32>
    %v7329 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7330 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7331 = stablehlo.divide %v7322, %v7329 : tensor<1536xf32>
    %v7332 = stablehlo.divide %v7328, %v7330 : tensor<1536xf32>
    %v7333 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7334 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7335 = stablehlo.sqrt %v7332 : tensor<1536xf32>
    %v7336 = stablehlo.add %v7335, %v7334 : tensor<1536xf32>
    %v7337 = stablehlo.divide %v7331, %v7336 : tensor<1536xf32>
    %v7338 = stablehlo.multiply %v7333, %v7337 : tensor<1536xf32>
    %v7339 = stablehlo.subtract %s2b4eb, %v7338 : tensor<1536xf32>
    %v7340 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7341 = stablehlo.multiply %v7340, %v7333 : tensor<1536xf32>
    %v7342 = stablehlo.multiply %v7341, %s2b4eb : tensor<1536xf32>
    %v7343 = stablehlo.subtract %v7339, %v7342 : tensor<1536xf32>
    %arsums2b4pW = "stablehlo.all_reduce"(%v2055) ({
    ^bb0(%aras2b4pW: tensor<f32>, %arbs2b4pW: tensor<f32>):
      %aradds2b4pW = stablehlo.add %aras2b4pW, %arbs2b4pW : tensor<f32>
      stablehlo.return %aradds2b4pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1536x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %arns2b4pW = stablehlo.constant dense<2.0> : tensor<384x1536x1x1xf32>
    %armeans2b4pW = stablehlo.divide %arsums2b4pW, %arns2b4pW : tensor<384x1536x1x1xf32>
    %v7344 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7345 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7346 = stablehlo.multiply %v7344, %s2b4pWm : tensor<384x1536x1x1xf32>
    %v7347 = stablehlo.multiply %v7345, %armeans2b4pW : tensor<384x1536x1x1xf32>
    %v7348 = stablehlo.add %v7346, %v7347 : tensor<384x1536x1x1xf32>
    %v7349 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7350 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7351 = stablehlo.multiply %v7349, %s2b4pWv : tensor<384x1536x1x1xf32>
    %v7352 = stablehlo.multiply %armeans2b4pW, %armeans2b4pW : tensor<384x1536x1x1xf32>
    %v7353 = stablehlo.multiply %v7350, %v7352 : tensor<384x1536x1x1xf32>
    %v7354 = stablehlo.add %v7351, %v7353 : tensor<384x1536x1x1xf32>
    %v7355 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7356 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7357 = stablehlo.multiply %v7355, %s2b4pWm : tensor<384x1536x1x1xf32>
    %v7358 = stablehlo.multiply %v7356, %armeans2b4pW : tensor<384x1536x1x1xf32>
    %v7359 = stablehlo.add %v7357, %v7358 : tensor<384x1536x1x1xf32>
    %v7360 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7361 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7362 = stablehlo.multiply %v7360, %s2b4pWv : tensor<384x1536x1x1xf32>
    %v7363 = stablehlo.multiply %armeans2b4pW, %armeans2b4pW : tensor<384x1536x1x1xf32>
    %v7364 = stablehlo.multiply %v7361, %v7363 : tensor<384x1536x1x1xf32>
    %v7365 = stablehlo.add %v7362, %v7364 : tensor<384x1536x1x1xf32>
    %v7366 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7367 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7368 = stablehlo.divide %v7359, %v7366 : tensor<384x1536x1x1xf32>
    %v7369 = stablehlo.divide %v7365, %v7367 : tensor<384x1536x1x1xf32>
    %v7370 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7371 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7372 = stablehlo.sqrt %v7369 : tensor<384x1536x1x1xf32>
    %v7373 = stablehlo.add %v7372, %v7371 : tensor<384x1536x1x1xf32>
    %v7374 = stablehlo.divide %v7368, %v7373 : tensor<384x1536x1x1xf32>
    %v7375 = stablehlo.multiply %v7370, %v7374 : tensor<384x1536x1x1xf32>
    %v7376 = stablehlo.subtract %s2b4pW, %v7375 : tensor<384x1536x1x1xf32>
    %v7377 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7378 = stablehlo.multiply %v7377, %v7370 : tensor<384x1536x1x1xf32>
    %v7379 = stablehlo.multiply %v7378, %s2b4pW : tensor<384x1536x1x1xf32>
    %v7380 = stablehlo.subtract %v7376, %v7379 : tensor<384x1536x1x1xf32>
    %arsums2b4pb = "stablehlo.all_reduce"(%v2058) ({
    ^bb0(%aras2b4pb: tensor<f32>, %arbs2b4pb: tensor<f32>):
      %aradds2b4pb = stablehlo.add %aras2b4pb, %arbs2b4pb : tensor<f32>
      stablehlo.return %aradds2b4pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b4pb = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b4pb = stablehlo.divide %arsums2b4pb, %arns2b4pb : tensor<384xf32>
    %v7381 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7382 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7383 = stablehlo.multiply %v7381, %s2b4pbm : tensor<384xf32>
    %v7384 = stablehlo.multiply %v7382, %armeans2b4pb : tensor<384xf32>
    %v7385 = stablehlo.add %v7383, %v7384 : tensor<384xf32>
    %v7386 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7387 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7388 = stablehlo.multiply %v7386, %s2b4pbv : tensor<384xf32>
    %v7389 = stablehlo.multiply %armeans2b4pb, %armeans2b4pb : tensor<384xf32>
    %v7390 = stablehlo.multiply %v7387, %v7389 : tensor<384xf32>
    %v7391 = stablehlo.add %v7388, %v7390 : tensor<384xf32>
    %v7392 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7393 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7394 = stablehlo.multiply %v7392, %s2b4pbm : tensor<384xf32>
    %v7395 = stablehlo.multiply %v7393, %armeans2b4pb : tensor<384xf32>
    %v7396 = stablehlo.add %v7394, %v7395 : tensor<384xf32>
    %v7397 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7398 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7399 = stablehlo.multiply %v7397, %s2b4pbv : tensor<384xf32>
    %v7400 = stablehlo.multiply %armeans2b4pb, %armeans2b4pb : tensor<384xf32>
    %v7401 = stablehlo.multiply %v7398, %v7400 : tensor<384xf32>
    %v7402 = stablehlo.add %v7399, %v7401 : tensor<384xf32>
    %v7403 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7404 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7405 = stablehlo.divide %v7396, %v7403 : tensor<384xf32>
    %v7406 = stablehlo.divide %v7402, %v7404 : tensor<384xf32>
    %v7407 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7408 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7409 = stablehlo.sqrt %v7406 : tensor<384xf32>
    %v7410 = stablehlo.add %v7409, %v7408 : tensor<384xf32>
    %v7411 = stablehlo.divide %v7405, %v7410 : tensor<384xf32>
    %v7412 = stablehlo.multiply %v7407, %v7411 : tensor<384xf32>
    %v7413 = stablehlo.subtract %s2b4pb, %v7412 : tensor<384xf32>
    %v7414 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7415 = stablehlo.multiply %v7414, %v7407 : tensor<384xf32>
    %v7416 = stablehlo.multiply %v7415, %s2b4pb : tensor<384xf32>
    %v7417 = stablehlo.subtract %v7413, %v7416 : tensor<384xf32>
    %arsums2b4lg = "stablehlo.all_reduce"(%v2049) ({
    ^bb0(%aras2b4lg: tensor<f32>, %arbs2b4lg: tensor<f32>):
      %aradds2b4lg = stablehlo.add %aras2b4lg, %arbs2b4lg : tensor<f32>
      stablehlo.return %aradds2b4lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b4lg = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b4lg = stablehlo.divide %arsums2b4lg, %arns2b4lg : tensor<384xf32>
    %v7418 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7419 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7420 = stablehlo.multiply %v7418, %s2b4lgm : tensor<384xf32>
    %v7421 = stablehlo.multiply %v7419, %armeans2b4lg : tensor<384xf32>
    %v7422 = stablehlo.add %v7420, %v7421 : tensor<384xf32>
    %v7423 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7424 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7425 = stablehlo.multiply %v7423, %s2b4lgv : tensor<384xf32>
    %v7426 = stablehlo.multiply %armeans2b4lg, %armeans2b4lg : tensor<384xf32>
    %v7427 = stablehlo.multiply %v7424, %v7426 : tensor<384xf32>
    %v7428 = stablehlo.add %v7425, %v7427 : tensor<384xf32>
    %v7429 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7430 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7431 = stablehlo.multiply %v7429, %s2b4lgm : tensor<384xf32>
    %v7432 = stablehlo.multiply %v7430, %armeans2b4lg : tensor<384xf32>
    %v7433 = stablehlo.add %v7431, %v7432 : tensor<384xf32>
    %v7434 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7435 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7436 = stablehlo.multiply %v7434, %s2b4lgv : tensor<384xf32>
    %v7437 = stablehlo.multiply %armeans2b4lg, %armeans2b4lg : tensor<384xf32>
    %v7438 = stablehlo.multiply %v7435, %v7437 : tensor<384xf32>
    %v7439 = stablehlo.add %v7436, %v7438 : tensor<384xf32>
    %v7440 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7441 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7442 = stablehlo.divide %v7433, %v7440 : tensor<384xf32>
    %v7443 = stablehlo.divide %v7439, %v7441 : tensor<384xf32>
    %v7444 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7445 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7446 = stablehlo.sqrt %v7443 : tensor<384xf32>
    %v7447 = stablehlo.add %v7446, %v7445 : tensor<384xf32>
    %v7448 = stablehlo.divide %v7442, %v7447 : tensor<384xf32>
    %v7449 = stablehlo.multiply %v7444, %v7448 : tensor<384xf32>
    %v7450 = stablehlo.subtract %s2b4lg, %v7449 : tensor<384xf32>
    %v7451 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7452 = stablehlo.multiply %v7451, %v7444 : tensor<384xf32>
    %v7453 = stablehlo.multiply %v7452, %s2b4lg : tensor<384xf32>
    %v7454 = stablehlo.subtract %v7450, %v7453 : tensor<384xf32>
    %arsums2b5dW = "stablehlo.all_reduce"(%v1972) ({
    ^bb0(%aras2b5dW: tensor<f32>, %arbs2b5dW: tensor<f32>):
      %aradds2b5dW = stablehlo.add %aras2b5dW, %arbs2b5dW : tensor<f32>
      stablehlo.return %aradds2b5dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1x7x7xf32>) -> tensor<384x1x7x7xf32>
    %arns2b5dW = stablehlo.constant dense<2.0> : tensor<384x1x7x7xf32>
    %armeans2b5dW = stablehlo.divide %arsums2b5dW, %arns2b5dW : tensor<384x1x7x7xf32>
    %v7455 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7456 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7457 = stablehlo.multiply %v7455, %s2b5dWm : tensor<384x1x7x7xf32>
    %v7458 = stablehlo.multiply %v7456, %armeans2b5dW : tensor<384x1x7x7xf32>
    %v7459 = stablehlo.add %v7457, %v7458 : tensor<384x1x7x7xf32>
    %v7460 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7461 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7462 = stablehlo.multiply %v7460, %s2b5dWv : tensor<384x1x7x7xf32>
    %v7463 = stablehlo.multiply %armeans2b5dW, %armeans2b5dW : tensor<384x1x7x7xf32>
    %v7464 = stablehlo.multiply %v7461, %v7463 : tensor<384x1x7x7xf32>
    %v7465 = stablehlo.add %v7462, %v7464 : tensor<384x1x7x7xf32>
    %v7466 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7467 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7468 = stablehlo.multiply %v7466, %s2b5dWm : tensor<384x1x7x7xf32>
    %v7469 = stablehlo.multiply %v7467, %armeans2b5dW : tensor<384x1x7x7xf32>
    %v7470 = stablehlo.add %v7468, %v7469 : tensor<384x1x7x7xf32>
    %v7471 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7472 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7473 = stablehlo.multiply %v7471, %s2b5dWv : tensor<384x1x7x7xf32>
    %v7474 = stablehlo.multiply %armeans2b5dW, %armeans2b5dW : tensor<384x1x7x7xf32>
    %v7475 = stablehlo.multiply %v7472, %v7474 : tensor<384x1x7x7xf32>
    %v7476 = stablehlo.add %v7473, %v7475 : tensor<384x1x7x7xf32>
    %v7477 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7478 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7479 = stablehlo.divide %v7470, %v7477 : tensor<384x1x7x7xf32>
    %v7480 = stablehlo.divide %v7476, %v7478 : tensor<384x1x7x7xf32>
    %v7481 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7482 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7483 = stablehlo.sqrt %v7480 : tensor<384x1x7x7xf32>
    %v7484 = stablehlo.add %v7483, %v7482 : tensor<384x1x7x7xf32>
    %v7485 = stablehlo.divide %v7479, %v7484 : tensor<384x1x7x7xf32>
    %v7486 = stablehlo.multiply %v7481, %v7485 : tensor<384x1x7x7xf32>
    %v7487 = stablehlo.subtract %s2b5dW, %v7486 : tensor<384x1x7x7xf32>
    %v7488 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7489 = stablehlo.multiply %v7488, %v7481 : tensor<384x1x7x7xf32>
    %v7490 = stablehlo.multiply %v7489, %s2b5dW : tensor<384x1x7x7xf32>
    %v7491 = stablehlo.subtract %v7487, %v7490 : tensor<384x1x7x7xf32>
    %arsums2b5db = "stablehlo.all_reduce"(%v1975) ({
    ^bb0(%aras2b5db: tensor<f32>, %arbs2b5db: tensor<f32>):
      %aradds2b5db = stablehlo.add %aras2b5db, %arbs2b5db : tensor<f32>
      stablehlo.return %aradds2b5db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b5db = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b5db = stablehlo.divide %arsums2b5db, %arns2b5db : tensor<384xf32>
    %v7492 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7493 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7494 = stablehlo.multiply %v7492, %s2b5dbm : tensor<384xf32>
    %v7495 = stablehlo.multiply %v7493, %armeans2b5db : tensor<384xf32>
    %v7496 = stablehlo.add %v7494, %v7495 : tensor<384xf32>
    %v7497 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7498 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7499 = stablehlo.multiply %v7497, %s2b5dbv : tensor<384xf32>
    %v7500 = stablehlo.multiply %armeans2b5db, %armeans2b5db : tensor<384xf32>
    %v7501 = stablehlo.multiply %v7498, %v7500 : tensor<384xf32>
    %v7502 = stablehlo.add %v7499, %v7501 : tensor<384xf32>
    %v7503 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7504 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7505 = stablehlo.multiply %v7503, %s2b5dbm : tensor<384xf32>
    %v7506 = stablehlo.multiply %v7504, %armeans2b5db : tensor<384xf32>
    %v7507 = stablehlo.add %v7505, %v7506 : tensor<384xf32>
    %v7508 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7509 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7510 = stablehlo.multiply %v7508, %s2b5dbv : tensor<384xf32>
    %v7511 = stablehlo.multiply %armeans2b5db, %armeans2b5db : tensor<384xf32>
    %v7512 = stablehlo.multiply %v7509, %v7511 : tensor<384xf32>
    %v7513 = stablehlo.add %v7510, %v7512 : tensor<384xf32>
    %v7514 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7515 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7516 = stablehlo.divide %v7507, %v7514 : tensor<384xf32>
    %v7517 = stablehlo.divide %v7513, %v7515 : tensor<384xf32>
    %v7518 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7519 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7520 = stablehlo.sqrt %v7517 : tensor<384xf32>
    %v7521 = stablehlo.add %v7520, %v7519 : tensor<384xf32>
    %v7522 = stablehlo.divide %v7516, %v7521 : tensor<384xf32>
    %v7523 = stablehlo.multiply %v7518, %v7522 : tensor<384xf32>
    %v7524 = stablehlo.subtract %s2b5db, %v7523 : tensor<384xf32>
    %v7525 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7526 = stablehlo.multiply %v7525, %v7518 : tensor<384xf32>
    %v7527 = stablehlo.multiply %v7526, %s2b5db : tensor<384xf32>
    %v7528 = stablehlo.subtract %v7524, %v7527 : tensor<384xf32>
    %arsums2b5ng = "stablehlo.all_reduce"(%v1964) ({
    ^bb0(%aras2b5ng: tensor<f32>, %arbs2b5ng: tensor<f32>):
      %aradds2b5ng = stablehlo.add %aras2b5ng, %arbs2b5ng : tensor<f32>
      stablehlo.return %aradds2b5ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b5ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b5ng = stablehlo.divide %arsums2b5ng, %arns2b5ng : tensor<f32>
    %v7529 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7530 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7531 = stablehlo.multiply %v7529, %s2b5ngm : tensor<f32>
    %v7532 = stablehlo.multiply %v7530, %armeans2b5ng : tensor<f32>
    %v7533 = stablehlo.add %v7531, %v7532 : tensor<f32>
    %v7534 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7535 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7536 = stablehlo.multiply %v7534, %s2b5ngv : tensor<f32>
    %v7537 = stablehlo.multiply %armeans2b5ng, %armeans2b5ng : tensor<f32>
    %v7538 = stablehlo.multiply %v7535, %v7537 : tensor<f32>
    %v7539 = stablehlo.add %v7536, %v7538 : tensor<f32>
    %v7540 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7541 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7542 = stablehlo.multiply %v7540, %s2b5ngm : tensor<f32>
    %v7543 = stablehlo.multiply %v7541, %armeans2b5ng : tensor<f32>
    %v7544 = stablehlo.add %v7542, %v7543 : tensor<f32>
    %v7545 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7546 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7547 = stablehlo.multiply %v7545, %s2b5ngv : tensor<f32>
    %v7548 = stablehlo.multiply %armeans2b5ng, %armeans2b5ng : tensor<f32>
    %v7549 = stablehlo.multiply %v7546, %v7548 : tensor<f32>
    %v7550 = stablehlo.add %v7547, %v7549 : tensor<f32>
    %v7551 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7552 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7553 = stablehlo.divide %v7544, %v7551 : tensor<f32>
    %v7554 = stablehlo.divide %v7550, %v7552 : tensor<f32>
    %v7555 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7556 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7557 = stablehlo.sqrt %v7554 : tensor<f32>
    %v7558 = stablehlo.add %v7557, %v7556 : tensor<f32>
    %v7559 = stablehlo.divide %v7553, %v7558 : tensor<f32>
    %v7560 = stablehlo.multiply %v7555, %v7559 : tensor<f32>
    %v7561 = stablehlo.subtract %s2b5ng, %v7560 : tensor<f32>
    %v7562 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7563 = stablehlo.multiply %v7562, %v7555 : tensor<f32>
    %v7564 = stablehlo.multiply %v7563, %s2b5ng : tensor<f32>
    %v7565 = stablehlo.subtract %v7561, %v7564 : tensor<f32>
    %arsums2b5nbt = "stablehlo.all_reduce"(%v1966) ({
    ^bb0(%aras2b5nbt: tensor<f32>, %arbs2b5nbt: tensor<f32>):
      %aradds2b5nbt = stablehlo.add %aras2b5nbt, %arbs2b5nbt : tensor<f32>
      stablehlo.return %aradds2b5nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b5nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b5nbt = stablehlo.divide %arsums2b5nbt, %arns2b5nbt : tensor<f32>
    %v7566 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7567 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7568 = stablehlo.multiply %v7566, %s2b5nbtm : tensor<f32>
    %v7569 = stablehlo.multiply %v7567, %armeans2b5nbt : tensor<f32>
    %v7570 = stablehlo.add %v7568, %v7569 : tensor<f32>
    %v7571 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7572 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7573 = stablehlo.multiply %v7571, %s2b5nbtv : tensor<f32>
    %v7574 = stablehlo.multiply %armeans2b5nbt, %armeans2b5nbt : tensor<f32>
    %v7575 = stablehlo.multiply %v7572, %v7574 : tensor<f32>
    %v7576 = stablehlo.add %v7573, %v7575 : tensor<f32>
    %v7577 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7578 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7579 = stablehlo.multiply %v7577, %s2b5nbtm : tensor<f32>
    %v7580 = stablehlo.multiply %v7578, %armeans2b5nbt : tensor<f32>
    %v7581 = stablehlo.add %v7579, %v7580 : tensor<f32>
    %v7582 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7583 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7584 = stablehlo.multiply %v7582, %s2b5nbtv : tensor<f32>
    %v7585 = stablehlo.multiply %armeans2b5nbt, %armeans2b5nbt : tensor<f32>
    %v7586 = stablehlo.multiply %v7583, %v7585 : tensor<f32>
    %v7587 = stablehlo.add %v7584, %v7586 : tensor<f32>
    %v7588 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7589 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7590 = stablehlo.divide %v7581, %v7588 : tensor<f32>
    %v7591 = stablehlo.divide %v7587, %v7589 : tensor<f32>
    %v7592 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7593 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7594 = stablehlo.sqrt %v7591 : tensor<f32>
    %v7595 = stablehlo.add %v7594, %v7593 : tensor<f32>
    %v7596 = stablehlo.divide %v7590, %v7595 : tensor<f32>
    %v7597 = stablehlo.multiply %v7592, %v7596 : tensor<f32>
    %v7598 = stablehlo.subtract %s2b5nbt, %v7597 : tensor<f32>
    %v7599 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7600 = stablehlo.multiply %v7599, %v7592 : tensor<f32>
    %v7601 = stablehlo.multiply %v7600, %s2b5nbt : tensor<f32>
    %v7602 = stablehlo.subtract %v7598, %v7601 : tensor<f32>
    %arsums2b5eW = "stablehlo.all_reduce"(%v1945) ({
    ^bb0(%aras2b5eW: tensor<f32>, %arbs2b5eW: tensor<f32>):
      %aradds2b5eW = stablehlo.add %aras2b5eW, %arbs2b5eW : tensor<f32>
      stablehlo.return %aradds2b5eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536x384x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %arns2b5eW = stablehlo.constant dense<2.0> : tensor<1536x384x1x1xf32>
    %armeans2b5eW = stablehlo.divide %arsums2b5eW, %arns2b5eW : tensor<1536x384x1x1xf32>
    %v7603 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7604 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7605 = stablehlo.multiply %v7603, %s2b5eWm : tensor<1536x384x1x1xf32>
    %v7606 = stablehlo.multiply %v7604, %armeans2b5eW : tensor<1536x384x1x1xf32>
    %v7607 = stablehlo.add %v7605, %v7606 : tensor<1536x384x1x1xf32>
    %v7608 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7609 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7610 = stablehlo.multiply %v7608, %s2b5eWv : tensor<1536x384x1x1xf32>
    %v7611 = stablehlo.multiply %armeans2b5eW, %armeans2b5eW : tensor<1536x384x1x1xf32>
    %v7612 = stablehlo.multiply %v7609, %v7611 : tensor<1536x384x1x1xf32>
    %v7613 = stablehlo.add %v7610, %v7612 : tensor<1536x384x1x1xf32>
    %v7614 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7615 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7616 = stablehlo.multiply %v7614, %s2b5eWm : tensor<1536x384x1x1xf32>
    %v7617 = stablehlo.multiply %v7615, %armeans2b5eW : tensor<1536x384x1x1xf32>
    %v7618 = stablehlo.add %v7616, %v7617 : tensor<1536x384x1x1xf32>
    %v7619 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7620 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7621 = stablehlo.multiply %v7619, %s2b5eWv : tensor<1536x384x1x1xf32>
    %v7622 = stablehlo.multiply %armeans2b5eW, %armeans2b5eW : tensor<1536x384x1x1xf32>
    %v7623 = stablehlo.multiply %v7620, %v7622 : tensor<1536x384x1x1xf32>
    %v7624 = stablehlo.add %v7621, %v7623 : tensor<1536x384x1x1xf32>
    %v7625 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7626 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7627 = stablehlo.divide %v7618, %v7625 : tensor<1536x384x1x1xf32>
    %v7628 = stablehlo.divide %v7624, %v7626 : tensor<1536x384x1x1xf32>
    %v7629 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7630 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7631 = stablehlo.sqrt %v7628 : tensor<1536x384x1x1xf32>
    %v7632 = stablehlo.add %v7631, %v7630 : tensor<1536x384x1x1xf32>
    %v7633 = stablehlo.divide %v7627, %v7632 : tensor<1536x384x1x1xf32>
    %v7634 = stablehlo.multiply %v7629, %v7633 : tensor<1536x384x1x1xf32>
    %v7635 = stablehlo.subtract %s2b5eW, %v7634 : tensor<1536x384x1x1xf32>
    %v7636 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7637 = stablehlo.multiply %v7636, %v7629 : tensor<1536x384x1x1xf32>
    %v7638 = stablehlo.multiply %v7637, %s2b5eW : tensor<1536x384x1x1xf32>
    %v7639 = stablehlo.subtract %v7635, %v7638 : tensor<1536x384x1x1xf32>
    %arsums2b5eb = "stablehlo.all_reduce"(%v1948) ({
    ^bb0(%aras2b5eb: tensor<f32>, %arbs2b5eb: tensor<f32>):
      %aradds2b5eb = stablehlo.add %aras2b5eb, %arbs2b5eb : tensor<f32>
      stablehlo.return %aradds2b5eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536xf32>) -> tensor<1536xf32>
    %arns2b5eb = stablehlo.constant dense<2.0> : tensor<1536xf32>
    %armeans2b5eb = stablehlo.divide %arsums2b5eb, %arns2b5eb : tensor<1536xf32>
    %v7640 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7641 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7642 = stablehlo.multiply %v7640, %s2b5ebm : tensor<1536xf32>
    %v7643 = stablehlo.multiply %v7641, %armeans2b5eb : tensor<1536xf32>
    %v7644 = stablehlo.add %v7642, %v7643 : tensor<1536xf32>
    %v7645 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7646 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7647 = stablehlo.multiply %v7645, %s2b5ebv : tensor<1536xf32>
    %v7648 = stablehlo.multiply %armeans2b5eb, %armeans2b5eb : tensor<1536xf32>
    %v7649 = stablehlo.multiply %v7646, %v7648 : tensor<1536xf32>
    %v7650 = stablehlo.add %v7647, %v7649 : tensor<1536xf32>
    %v7651 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7652 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7653 = stablehlo.multiply %v7651, %s2b5ebm : tensor<1536xf32>
    %v7654 = stablehlo.multiply %v7652, %armeans2b5eb : tensor<1536xf32>
    %v7655 = stablehlo.add %v7653, %v7654 : tensor<1536xf32>
    %v7656 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7657 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7658 = stablehlo.multiply %v7656, %s2b5ebv : tensor<1536xf32>
    %v7659 = stablehlo.multiply %armeans2b5eb, %armeans2b5eb : tensor<1536xf32>
    %v7660 = stablehlo.multiply %v7657, %v7659 : tensor<1536xf32>
    %v7661 = stablehlo.add %v7658, %v7660 : tensor<1536xf32>
    %v7662 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7663 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7664 = stablehlo.divide %v7655, %v7662 : tensor<1536xf32>
    %v7665 = stablehlo.divide %v7661, %v7663 : tensor<1536xf32>
    %v7666 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7667 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7668 = stablehlo.sqrt %v7665 : tensor<1536xf32>
    %v7669 = stablehlo.add %v7668, %v7667 : tensor<1536xf32>
    %v7670 = stablehlo.divide %v7664, %v7669 : tensor<1536xf32>
    %v7671 = stablehlo.multiply %v7666, %v7670 : tensor<1536xf32>
    %v7672 = stablehlo.subtract %s2b5eb, %v7671 : tensor<1536xf32>
    %v7673 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7674 = stablehlo.multiply %v7673, %v7666 : tensor<1536xf32>
    %v7675 = stablehlo.multiply %v7674, %s2b5eb : tensor<1536xf32>
    %v7676 = stablehlo.subtract %v7672, %v7675 : tensor<1536xf32>
    %arsums2b5pW = "stablehlo.all_reduce"(%v1936) ({
    ^bb0(%aras2b5pW: tensor<f32>, %arbs2b5pW: tensor<f32>):
      %aradds2b5pW = stablehlo.add %aras2b5pW, %arbs2b5pW : tensor<f32>
      stablehlo.return %aradds2b5pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1536x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %arns2b5pW = stablehlo.constant dense<2.0> : tensor<384x1536x1x1xf32>
    %armeans2b5pW = stablehlo.divide %arsums2b5pW, %arns2b5pW : tensor<384x1536x1x1xf32>
    %v7677 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7678 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7679 = stablehlo.multiply %v7677, %s2b5pWm : tensor<384x1536x1x1xf32>
    %v7680 = stablehlo.multiply %v7678, %armeans2b5pW : tensor<384x1536x1x1xf32>
    %v7681 = stablehlo.add %v7679, %v7680 : tensor<384x1536x1x1xf32>
    %v7682 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7683 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7684 = stablehlo.multiply %v7682, %s2b5pWv : tensor<384x1536x1x1xf32>
    %v7685 = stablehlo.multiply %armeans2b5pW, %armeans2b5pW : tensor<384x1536x1x1xf32>
    %v7686 = stablehlo.multiply %v7683, %v7685 : tensor<384x1536x1x1xf32>
    %v7687 = stablehlo.add %v7684, %v7686 : tensor<384x1536x1x1xf32>
    %v7688 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7689 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7690 = stablehlo.multiply %v7688, %s2b5pWm : tensor<384x1536x1x1xf32>
    %v7691 = stablehlo.multiply %v7689, %armeans2b5pW : tensor<384x1536x1x1xf32>
    %v7692 = stablehlo.add %v7690, %v7691 : tensor<384x1536x1x1xf32>
    %v7693 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7694 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7695 = stablehlo.multiply %v7693, %s2b5pWv : tensor<384x1536x1x1xf32>
    %v7696 = stablehlo.multiply %armeans2b5pW, %armeans2b5pW : tensor<384x1536x1x1xf32>
    %v7697 = stablehlo.multiply %v7694, %v7696 : tensor<384x1536x1x1xf32>
    %v7698 = stablehlo.add %v7695, %v7697 : tensor<384x1536x1x1xf32>
    %v7699 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7700 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7701 = stablehlo.divide %v7692, %v7699 : tensor<384x1536x1x1xf32>
    %v7702 = stablehlo.divide %v7698, %v7700 : tensor<384x1536x1x1xf32>
    %v7703 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7704 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7705 = stablehlo.sqrt %v7702 : tensor<384x1536x1x1xf32>
    %v7706 = stablehlo.add %v7705, %v7704 : tensor<384x1536x1x1xf32>
    %v7707 = stablehlo.divide %v7701, %v7706 : tensor<384x1536x1x1xf32>
    %v7708 = stablehlo.multiply %v7703, %v7707 : tensor<384x1536x1x1xf32>
    %v7709 = stablehlo.subtract %s2b5pW, %v7708 : tensor<384x1536x1x1xf32>
    %v7710 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7711 = stablehlo.multiply %v7710, %v7703 : tensor<384x1536x1x1xf32>
    %v7712 = stablehlo.multiply %v7711, %s2b5pW : tensor<384x1536x1x1xf32>
    %v7713 = stablehlo.subtract %v7709, %v7712 : tensor<384x1536x1x1xf32>
    %arsums2b5pb = "stablehlo.all_reduce"(%v1939) ({
    ^bb0(%aras2b5pb: tensor<f32>, %arbs2b5pb: tensor<f32>):
      %aradds2b5pb = stablehlo.add %aras2b5pb, %arbs2b5pb : tensor<f32>
      stablehlo.return %aradds2b5pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b5pb = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b5pb = stablehlo.divide %arsums2b5pb, %arns2b5pb : tensor<384xf32>
    %v7714 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7715 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7716 = stablehlo.multiply %v7714, %s2b5pbm : tensor<384xf32>
    %v7717 = stablehlo.multiply %v7715, %armeans2b5pb : tensor<384xf32>
    %v7718 = stablehlo.add %v7716, %v7717 : tensor<384xf32>
    %v7719 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7720 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7721 = stablehlo.multiply %v7719, %s2b5pbv : tensor<384xf32>
    %v7722 = stablehlo.multiply %armeans2b5pb, %armeans2b5pb : tensor<384xf32>
    %v7723 = stablehlo.multiply %v7720, %v7722 : tensor<384xf32>
    %v7724 = stablehlo.add %v7721, %v7723 : tensor<384xf32>
    %v7725 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7726 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7727 = stablehlo.multiply %v7725, %s2b5pbm : tensor<384xf32>
    %v7728 = stablehlo.multiply %v7726, %armeans2b5pb : tensor<384xf32>
    %v7729 = stablehlo.add %v7727, %v7728 : tensor<384xf32>
    %v7730 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7731 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7732 = stablehlo.multiply %v7730, %s2b5pbv : tensor<384xf32>
    %v7733 = stablehlo.multiply %armeans2b5pb, %armeans2b5pb : tensor<384xf32>
    %v7734 = stablehlo.multiply %v7731, %v7733 : tensor<384xf32>
    %v7735 = stablehlo.add %v7732, %v7734 : tensor<384xf32>
    %v7736 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7737 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7738 = stablehlo.divide %v7729, %v7736 : tensor<384xf32>
    %v7739 = stablehlo.divide %v7735, %v7737 : tensor<384xf32>
    %v7740 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7741 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7742 = stablehlo.sqrt %v7739 : tensor<384xf32>
    %v7743 = stablehlo.add %v7742, %v7741 : tensor<384xf32>
    %v7744 = stablehlo.divide %v7738, %v7743 : tensor<384xf32>
    %v7745 = stablehlo.multiply %v7740, %v7744 : tensor<384xf32>
    %v7746 = stablehlo.subtract %s2b5pb, %v7745 : tensor<384xf32>
    %v7747 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7748 = stablehlo.multiply %v7747, %v7740 : tensor<384xf32>
    %v7749 = stablehlo.multiply %v7748, %s2b5pb : tensor<384xf32>
    %v7750 = stablehlo.subtract %v7746, %v7749 : tensor<384xf32>
    %arsums2b5lg = "stablehlo.all_reduce"(%v1930) ({
    ^bb0(%aras2b5lg: tensor<f32>, %arbs2b5lg: tensor<f32>):
      %aradds2b5lg = stablehlo.add %aras2b5lg, %arbs2b5lg : tensor<f32>
      stablehlo.return %aradds2b5lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b5lg = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b5lg = stablehlo.divide %arsums2b5lg, %arns2b5lg : tensor<384xf32>
    %v7751 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7752 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7753 = stablehlo.multiply %v7751, %s2b5lgm : tensor<384xf32>
    %v7754 = stablehlo.multiply %v7752, %armeans2b5lg : tensor<384xf32>
    %v7755 = stablehlo.add %v7753, %v7754 : tensor<384xf32>
    %v7756 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7757 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7758 = stablehlo.multiply %v7756, %s2b5lgv : tensor<384xf32>
    %v7759 = stablehlo.multiply %armeans2b5lg, %armeans2b5lg : tensor<384xf32>
    %v7760 = stablehlo.multiply %v7757, %v7759 : tensor<384xf32>
    %v7761 = stablehlo.add %v7758, %v7760 : tensor<384xf32>
    %v7762 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7763 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7764 = stablehlo.multiply %v7762, %s2b5lgm : tensor<384xf32>
    %v7765 = stablehlo.multiply %v7763, %armeans2b5lg : tensor<384xf32>
    %v7766 = stablehlo.add %v7764, %v7765 : tensor<384xf32>
    %v7767 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7768 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7769 = stablehlo.multiply %v7767, %s2b5lgv : tensor<384xf32>
    %v7770 = stablehlo.multiply %armeans2b5lg, %armeans2b5lg : tensor<384xf32>
    %v7771 = stablehlo.multiply %v7768, %v7770 : tensor<384xf32>
    %v7772 = stablehlo.add %v7769, %v7771 : tensor<384xf32>
    %v7773 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7774 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7775 = stablehlo.divide %v7766, %v7773 : tensor<384xf32>
    %v7776 = stablehlo.divide %v7772, %v7774 : tensor<384xf32>
    %v7777 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7778 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7779 = stablehlo.sqrt %v7776 : tensor<384xf32>
    %v7780 = stablehlo.add %v7779, %v7778 : tensor<384xf32>
    %v7781 = stablehlo.divide %v7775, %v7780 : tensor<384xf32>
    %v7782 = stablehlo.multiply %v7777, %v7781 : tensor<384xf32>
    %v7783 = stablehlo.subtract %s2b5lg, %v7782 : tensor<384xf32>
    %v7784 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7785 = stablehlo.multiply %v7784, %v7777 : tensor<384xf32>
    %v7786 = stablehlo.multiply %v7785, %s2b5lg : tensor<384xf32>
    %v7787 = stablehlo.subtract %v7783, %v7786 : tensor<384xf32>
    %arsums2b6dW = "stablehlo.all_reduce"(%v1853) ({
    ^bb0(%aras2b6dW: tensor<f32>, %arbs2b6dW: tensor<f32>):
      %aradds2b6dW = stablehlo.add %aras2b6dW, %arbs2b6dW : tensor<f32>
      stablehlo.return %aradds2b6dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1x7x7xf32>) -> tensor<384x1x7x7xf32>
    %arns2b6dW = stablehlo.constant dense<2.0> : tensor<384x1x7x7xf32>
    %armeans2b6dW = stablehlo.divide %arsums2b6dW, %arns2b6dW : tensor<384x1x7x7xf32>
    %v7788 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7789 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7790 = stablehlo.multiply %v7788, %s2b6dWm : tensor<384x1x7x7xf32>
    %v7791 = stablehlo.multiply %v7789, %armeans2b6dW : tensor<384x1x7x7xf32>
    %v7792 = stablehlo.add %v7790, %v7791 : tensor<384x1x7x7xf32>
    %v7793 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7794 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7795 = stablehlo.multiply %v7793, %s2b6dWv : tensor<384x1x7x7xf32>
    %v7796 = stablehlo.multiply %armeans2b6dW, %armeans2b6dW : tensor<384x1x7x7xf32>
    %v7797 = stablehlo.multiply %v7794, %v7796 : tensor<384x1x7x7xf32>
    %v7798 = stablehlo.add %v7795, %v7797 : tensor<384x1x7x7xf32>
    %v7799 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7800 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7801 = stablehlo.multiply %v7799, %s2b6dWm : tensor<384x1x7x7xf32>
    %v7802 = stablehlo.multiply %v7800, %armeans2b6dW : tensor<384x1x7x7xf32>
    %v7803 = stablehlo.add %v7801, %v7802 : tensor<384x1x7x7xf32>
    %v7804 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7805 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7806 = stablehlo.multiply %v7804, %s2b6dWv : tensor<384x1x7x7xf32>
    %v7807 = stablehlo.multiply %armeans2b6dW, %armeans2b6dW : tensor<384x1x7x7xf32>
    %v7808 = stablehlo.multiply %v7805, %v7807 : tensor<384x1x7x7xf32>
    %v7809 = stablehlo.add %v7806, %v7808 : tensor<384x1x7x7xf32>
    %v7810 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7811 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7812 = stablehlo.divide %v7803, %v7810 : tensor<384x1x7x7xf32>
    %v7813 = stablehlo.divide %v7809, %v7811 : tensor<384x1x7x7xf32>
    %v7814 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7815 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7816 = stablehlo.sqrt %v7813 : tensor<384x1x7x7xf32>
    %v7817 = stablehlo.add %v7816, %v7815 : tensor<384x1x7x7xf32>
    %v7818 = stablehlo.divide %v7812, %v7817 : tensor<384x1x7x7xf32>
    %v7819 = stablehlo.multiply %v7814, %v7818 : tensor<384x1x7x7xf32>
    %v7820 = stablehlo.subtract %s2b6dW, %v7819 : tensor<384x1x7x7xf32>
    %v7821 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7822 = stablehlo.multiply %v7821, %v7814 : tensor<384x1x7x7xf32>
    %v7823 = stablehlo.multiply %v7822, %s2b6dW : tensor<384x1x7x7xf32>
    %v7824 = stablehlo.subtract %v7820, %v7823 : tensor<384x1x7x7xf32>
    %arsums2b6db = "stablehlo.all_reduce"(%v1856) ({
    ^bb0(%aras2b6db: tensor<f32>, %arbs2b6db: tensor<f32>):
      %aradds2b6db = stablehlo.add %aras2b6db, %arbs2b6db : tensor<f32>
      stablehlo.return %aradds2b6db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b6db = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b6db = stablehlo.divide %arsums2b6db, %arns2b6db : tensor<384xf32>
    %v7825 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7826 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7827 = stablehlo.multiply %v7825, %s2b6dbm : tensor<384xf32>
    %v7828 = stablehlo.multiply %v7826, %armeans2b6db : tensor<384xf32>
    %v7829 = stablehlo.add %v7827, %v7828 : tensor<384xf32>
    %v7830 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7831 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7832 = stablehlo.multiply %v7830, %s2b6dbv : tensor<384xf32>
    %v7833 = stablehlo.multiply %armeans2b6db, %armeans2b6db : tensor<384xf32>
    %v7834 = stablehlo.multiply %v7831, %v7833 : tensor<384xf32>
    %v7835 = stablehlo.add %v7832, %v7834 : tensor<384xf32>
    %v7836 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7837 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7838 = stablehlo.multiply %v7836, %s2b6dbm : tensor<384xf32>
    %v7839 = stablehlo.multiply %v7837, %armeans2b6db : tensor<384xf32>
    %v7840 = stablehlo.add %v7838, %v7839 : tensor<384xf32>
    %v7841 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7842 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7843 = stablehlo.multiply %v7841, %s2b6dbv : tensor<384xf32>
    %v7844 = stablehlo.multiply %armeans2b6db, %armeans2b6db : tensor<384xf32>
    %v7845 = stablehlo.multiply %v7842, %v7844 : tensor<384xf32>
    %v7846 = stablehlo.add %v7843, %v7845 : tensor<384xf32>
    %v7847 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7848 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7849 = stablehlo.divide %v7840, %v7847 : tensor<384xf32>
    %v7850 = stablehlo.divide %v7846, %v7848 : tensor<384xf32>
    %v7851 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7852 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7853 = stablehlo.sqrt %v7850 : tensor<384xf32>
    %v7854 = stablehlo.add %v7853, %v7852 : tensor<384xf32>
    %v7855 = stablehlo.divide %v7849, %v7854 : tensor<384xf32>
    %v7856 = stablehlo.multiply %v7851, %v7855 : tensor<384xf32>
    %v7857 = stablehlo.subtract %s2b6db, %v7856 : tensor<384xf32>
    %v7858 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7859 = stablehlo.multiply %v7858, %v7851 : tensor<384xf32>
    %v7860 = stablehlo.multiply %v7859, %s2b6db : tensor<384xf32>
    %v7861 = stablehlo.subtract %v7857, %v7860 : tensor<384xf32>
    %arsums2b6ng = "stablehlo.all_reduce"(%v1845) ({
    ^bb0(%aras2b6ng: tensor<f32>, %arbs2b6ng: tensor<f32>):
      %aradds2b6ng = stablehlo.add %aras2b6ng, %arbs2b6ng : tensor<f32>
      stablehlo.return %aradds2b6ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b6ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b6ng = stablehlo.divide %arsums2b6ng, %arns2b6ng : tensor<f32>
    %v7862 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7863 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7864 = stablehlo.multiply %v7862, %s2b6ngm : tensor<f32>
    %v7865 = stablehlo.multiply %v7863, %armeans2b6ng : tensor<f32>
    %v7866 = stablehlo.add %v7864, %v7865 : tensor<f32>
    %v7867 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7868 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7869 = stablehlo.multiply %v7867, %s2b6ngv : tensor<f32>
    %v7870 = stablehlo.multiply %armeans2b6ng, %armeans2b6ng : tensor<f32>
    %v7871 = stablehlo.multiply %v7868, %v7870 : tensor<f32>
    %v7872 = stablehlo.add %v7869, %v7871 : tensor<f32>
    %v7873 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7874 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7875 = stablehlo.multiply %v7873, %s2b6ngm : tensor<f32>
    %v7876 = stablehlo.multiply %v7874, %armeans2b6ng : tensor<f32>
    %v7877 = stablehlo.add %v7875, %v7876 : tensor<f32>
    %v7878 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7879 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7880 = stablehlo.multiply %v7878, %s2b6ngv : tensor<f32>
    %v7881 = stablehlo.multiply %armeans2b6ng, %armeans2b6ng : tensor<f32>
    %v7882 = stablehlo.multiply %v7879, %v7881 : tensor<f32>
    %v7883 = stablehlo.add %v7880, %v7882 : tensor<f32>
    %v7884 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7885 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7886 = stablehlo.divide %v7877, %v7884 : tensor<f32>
    %v7887 = stablehlo.divide %v7883, %v7885 : tensor<f32>
    %v7888 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7889 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7890 = stablehlo.sqrt %v7887 : tensor<f32>
    %v7891 = stablehlo.add %v7890, %v7889 : tensor<f32>
    %v7892 = stablehlo.divide %v7886, %v7891 : tensor<f32>
    %v7893 = stablehlo.multiply %v7888, %v7892 : tensor<f32>
    %v7894 = stablehlo.subtract %s2b6ng, %v7893 : tensor<f32>
    %v7895 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7896 = stablehlo.multiply %v7895, %v7888 : tensor<f32>
    %v7897 = stablehlo.multiply %v7896, %s2b6ng : tensor<f32>
    %v7898 = stablehlo.subtract %v7894, %v7897 : tensor<f32>
    %arsums2b6nbt = "stablehlo.all_reduce"(%v1847) ({
    ^bb0(%aras2b6nbt: tensor<f32>, %arbs2b6nbt: tensor<f32>):
      %aradds2b6nbt = stablehlo.add %aras2b6nbt, %arbs2b6nbt : tensor<f32>
      stablehlo.return %aradds2b6nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b6nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b6nbt = stablehlo.divide %arsums2b6nbt, %arns2b6nbt : tensor<f32>
    %v7899 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7900 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7901 = stablehlo.multiply %v7899, %s2b6nbtm : tensor<f32>
    %v7902 = stablehlo.multiply %v7900, %armeans2b6nbt : tensor<f32>
    %v7903 = stablehlo.add %v7901, %v7902 : tensor<f32>
    %v7904 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7905 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7906 = stablehlo.multiply %v7904, %s2b6nbtv : tensor<f32>
    %v7907 = stablehlo.multiply %armeans2b6nbt, %armeans2b6nbt : tensor<f32>
    %v7908 = stablehlo.multiply %v7905, %v7907 : tensor<f32>
    %v7909 = stablehlo.add %v7906, %v7908 : tensor<f32>
    %v7910 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7911 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7912 = stablehlo.multiply %v7910, %s2b6nbtm : tensor<f32>
    %v7913 = stablehlo.multiply %v7911, %armeans2b6nbt : tensor<f32>
    %v7914 = stablehlo.add %v7912, %v7913 : tensor<f32>
    %v7915 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7916 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7917 = stablehlo.multiply %v7915, %s2b6nbtv : tensor<f32>
    %v7918 = stablehlo.multiply %armeans2b6nbt, %armeans2b6nbt : tensor<f32>
    %v7919 = stablehlo.multiply %v7916, %v7918 : tensor<f32>
    %v7920 = stablehlo.add %v7917, %v7919 : tensor<f32>
    %v7921 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7922 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7923 = stablehlo.divide %v7914, %v7921 : tensor<f32>
    %v7924 = stablehlo.divide %v7920, %v7922 : tensor<f32>
    %v7925 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7926 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7927 = stablehlo.sqrt %v7924 : tensor<f32>
    %v7928 = stablehlo.add %v7927, %v7926 : tensor<f32>
    %v7929 = stablehlo.divide %v7923, %v7928 : tensor<f32>
    %v7930 = stablehlo.multiply %v7925, %v7929 : tensor<f32>
    %v7931 = stablehlo.subtract %s2b6nbt, %v7930 : tensor<f32>
    %v7932 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7933 = stablehlo.multiply %v7932, %v7925 : tensor<f32>
    %v7934 = stablehlo.multiply %v7933, %s2b6nbt : tensor<f32>
    %v7935 = stablehlo.subtract %v7931, %v7934 : tensor<f32>
    %arsums2b6eW = "stablehlo.all_reduce"(%v1826) ({
    ^bb0(%aras2b6eW: tensor<f32>, %arbs2b6eW: tensor<f32>):
      %aradds2b6eW = stablehlo.add %aras2b6eW, %arbs2b6eW : tensor<f32>
      stablehlo.return %aradds2b6eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536x384x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %arns2b6eW = stablehlo.constant dense<2.0> : tensor<1536x384x1x1xf32>
    %armeans2b6eW = stablehlo.divide %arsums2b6eW, %arns2b6eW : tensor<1536x384x1x1xf32>
    %v7936 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7937 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7938 = stablehlo.multiply %v7936, %s2b6eWm : tensor<1536x384x1x1xf32>
    %v7939 = stablehlo.multiply %v7937, %armeans2b6eW : tensor<1536x384x1x1xf32>
    %v7940 = stablehlo.add %v7938, %v7939 : tensor<1536x384x1x1xf32>
    %v7941 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7942 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7943 = stablehlo.multiply %v7941, %s2b6eWv : tensor<1536x384x1x1xf32>
    %v7944 = stablehlo.multiply %armeans2b6eW, %armeans2b6eW : tensor<1536x384x1x1xf32>
    %v7945 = stablehlo.multiply %v7942, %v7944 : tensor<1536x384x1x1xf32>
    %v7946 = stablehlo.add %v7943, %v7945 : tensor<1536x384x1x1xf32>
    %v7947 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7948 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7949 = stablehlo.multiply %v7947, %s2b6eWm : tensor<1536x384x1x1xf32>
    %v7950 = stablehlo.multiply %v7948, %armeans2b6eW : tensor<1536x384x1x1xf32>
    %v7951 = stablehlo.add %v7949, %v7950 : tensor<1536x384x1x1xf32>
    %v7952 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7953 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7954 = stablehlo.multiply %v7952, %s2b6eWv : tensor<1536x384x1x1xf32>
    %v7955 = stablehlo.multiply %armeans2b6eW, %armeans2b6eW : tensor<1536x384x1x1xf32>
    %v7956 = stablehlo.multiply %v7953, %v7955 : tensor<1536x384x1x1xf32>
    %v7957 = stablehlo.add %v7954, %v7956 : tensor<1536x384x1x1xf32>
    %v7958 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7959 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7960 = stablehlo.divide %v7951, %v7958 : tensor<1536x384x1x1xf32>
    %v7961 = stablehlo.divide %v7957, %v7959 : tensor<1536x384x1x1xf32>
    %v7962 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7963 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7964 = stablehlo.sqrt %v7961 : tensor<1536x384x1x1xf32>
    %v7965 = stablehlo.add %v7964, %v7963 : tensor<1536x384x1x1xf32>
    %v7966 = stablehlo.divide %v7960, %v7965 : tensor<1536x384x1x1xf32>
    %v7967 = stablehlo.multiply %v7962, %v7966 : tensor<1536x384x1x1xf32>
    %v7968 = stablehlo.subtract %s2b6eW, %v7967 : tensor<1536x384x1x1xf32>
    %v7969 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7970 = stablehlo.multiply %v7969, %v7962 : tensor<1536x384x1x1xf32>
    %v7971 = stablehlo.multiply %v7970, %s2b6eW : tensor<1536x384x1x1xf32>
    %v7972 = stablehlo.subtract %v7968, %v7971 : tensor<1536x384x1x1xf32>
    %arsums2b6eb = "stablehlo.all_reduce"(%v1829) ({
    ^bb0(%aras2b6eb: tensor<f32>, %arbs2b6eb: tensor<f32>):
      %aradds2b6eb = stablehlo.add %aras2b6eb, %arbs2b6eb : tensor<f32>
      stablehlo.return %aradds2b6eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536xf32>) -> tensor<1536xf32>
    %arns2b6eb = stablehlo.constant dense<2.0> : tensor<1536xf32>
    %armeans2b6eb = stablehlo.divide %arsums2b6eb, %arns2b6eb : tensor<1536xf32>
    %v7973 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7974 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7975 = stablehlo.multiply %v7973, %s2b6ebm : tensor<1536xf32>
    %v7976 = stablehlo.multiply %v7974, %armeans2b6eb : tensor<1536xf32>
    %v7977 = stablehlo.add %v7975, %v7976 : tensor<1536xf32>
    %v7978 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7979 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7980 = stablehlo.multiply %v7978, %s2b6ebv : tensor<1536xf32>
    %v7981 = stablehlo.multiply %armeans2b6eb, %armeans2b6eb : tensor<1536xf32>
    %v7982 = stablehlo.multiply %v7979, %v7981 : tensor<1536xf32>
    %v7983 = stablehlo.add %v7980, %v7982 : tensor<1536xf32>
    %v7984 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7985 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7986 = stablehlo.multiply %v7984, %s2b6ebm : tensor<1536xf32>
    %v7987 = stablehlo.multiply %v7985, %armeans2b6eb : tensor<1536xf32>
    %v7988 = stablehlo.add %v7986, %v7987 : tensor<1536xf32>
    %v7989 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7990 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7991 = stablehlo.multiply %v7989, %s2b6ebv : tensor<1536xf32>
    %v7992 = stablehlo.multiply %armeans2b6eb, %armeans2b6eb : tensor<1536xf32>
    %v7993 = stablehlo.multiply %v7990, %v7992 : tensor<1536xf32>
    %v7994 = stablehlo.add %v7991, %v7993 : tensor<1536xf32>
    %v7995 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7996 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7997 = stablehlo.divide %v7988, %v7995 : tensor<1536xf32>
    %v7998 = stablehlo.divide %v7994, %v7996 : tensor<1536xf32>
    %v7999 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8000 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8001 = stablehlo.sqrt %v7998 : tensor<1536xf32>
    %v8002 = stablehlo.add %v8001, %v8000 : tensor<1536xf32>
    %v8003 = stablehlo.divide %v7997, %v8002 : tensor<1536xf32>
    %v8004 = stablehlo.multiply %v7999, %v8003 : tensor<1536xf32>
    %v8005 = stablehlo.subtract %s2b6eb, %v8004 : tensor<1536xf32>
    %v8006 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8007 = stablehlo.multiply %v8006, %v7999 : tensor<1536xf32>
    %v8008 = stablehlo.multiply %v8007, %s2b6eb : tensor<1536xf32>
    %v8009 = stablehlo.subtract %v8005, %v8008 : tensor<1536xf32>
    %arsums2b6pW = "stablehlo.all_reduce"(%v1817) ({
    ^bb0(%aras2b6pW: tensor<f32>, %arbs2b6pW: tensor<f32>):
      %aradds2b6pW = stablehlo.add %aras2b6pW, %arbs2b6pW : tensor<f32>
      stablehlo.return %aradds2b6pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1536x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %arns2b6pW = stablehlo.constant dense<2.0> : tensor<384x1536x1x1xf32>
    %armeans2b6pW = stablehlo.divide %arsums2b6pW, %arns2b6pW : tensor<384x1536x1x1xf32>
    %v8010 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8011 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8012 = stablehlo.multiply %v8010, %s2b6pWm : tensor<384x1536x1x1xf32>
    %v8013 = stablehlo.multiply %v8011, %armeans2b6pW : tensor<384x1536x1x1xf32>
    %v8014 = stablehlo.add %v8012, %v8013 : tensor<384x1536x1x1xf32>
    %v8015 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8016 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8017 = stablehlo.multiply %v8015, %s2b6pWv : tensor<384x1536x1x1xf32>
    %v8018 = stablehlo.multiply %armeans2b6pW, %armeans2b6pW : tensor<384x1536x1x1xf32>
    %v8019 = stablehlo.multiply %v8016, %v8018 : tensor<384x1536x1x1xf32>
    %v8020 = stablehlo.add %v8017, %v8019 : tensor<384x1536x1x1xf32>
    %v8021 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8022 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8023 = stablehlo.multiply %v8021, %s2b6pWm : tensor<384x1536x1x1xf32>
    %v8024 = stablehlo.multiply %v8022, %armeans2b6pW : tensor<384x1536x1x1xf32>
    %v8025 = stablehlo.add %v8023, %v8024 : tensor<384x1536x1x1xf32>
    %v8026 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8027 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8028 = stablehlo.multiply %v8026, %s2b6pWv : tensor<384x1536x1x1xf32>
    %v8029 = stablehlo.multiply %armeans2b6pW, %armeans2b6pW : tensor<384x1536x1x1xf32>
    %v8030 = stablehlo.multiply %v8027, %v8029 : tensor<384x1536x1x1xf32>
    %v8031 = stablehlo.add %v8028, %v8030 : tensor<384x1536x1x1xf32>
    %v8032 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8033 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8034 = stablehlo.divide %v8025, %v8032 : tensor<384x1536x1x1xf32>
    %v8035 = stablehlo.divide %v8031, %v8033 : tensor<384x1536x1x1xf32>
    %v8036 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8037 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8038 = stablehlo.sqrt %v8035 : tensor<384x1536x1x1xf32>
    %v8039 = stablehlo.add %v8038, %v8037 : tensor<384x1536x1x1xf32>
    %v8040 = stablehlo.divide %v8034, %v8039 : tensor<384x1536x1x1xf32>
    %v8041 = stablehlo.multiply %v8036, %v8040 : tensor<384x1536x1x1xf32>
    %v8042 = stablehlo.subtract %s2b6pW, %v8041 : tensor<384x1536x1x1xf32>
    %v8043 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8044 = stablehlo.multiply %v8043, %v8036 : tensor<384x1536x1x1xf32>
    %v8045 = stablehlo.multiply %v8044, %s2b6pW : tensor<384x1536x1x1xf32>
    %v8046 = stablehlo.subtract %v8042, %v8045 : tensor<384x1536x1x1xf32>
    %arsums2b6pb = "stablehlo.all_reduce"(%v1820) ({
    ^bb0(%aras2b6pb: tensor<f32>, %arbs2b6pb: tensor<f32>):
      %aradds2b6pb = stablehlo.add %aras2b6pb, %arbs2b6pb : tensor<f32>
      stablehlo.return %aradds2b6pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b6pb = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b6pb = stablehlo.divide %arsums2b6pb, %arns2b6pb : tensor<384xf32>
    %v8047 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8048 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8049 = stablehlo.multiply %v8047, %s2b6pbm : tensor<384xf32>
    %v8050 = stablehlo.multiply %v8048, %armeans2b6pb : tensor<384xf32>
    %v8051 = stablehlo.add %v8049, %v8050 : tensor<384xf32>
    %v8052 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8053 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8054 = stablehlo.multiply %v8052, %s2b6pbv : tensor<384xf32>
    %v8055 = stablehlo.multiply %armeans2b6pb, %armeans2b6pb : tensor<384xf32>
    %v8056 = stablehlo.multiply %v8053, %v8055 : tensor<384xf32>
    %v8057 = stablehlo.add %v8054, %v8056 : tensor<384xf32>
    %v8058 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8059 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8060 = stablehlo.multiply %v8058, %s2b6pbm : tensor<384xf32>
    %v8061 = stablehlo.multiply %v8059, %armeans2b6pb : tensor<384xf32>
    %v8062 = stablehlo.add %v8060, %v8061 : tensor<384xf32>
    %v8063 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8064 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8065 = stablehlo.multiply %v8063, %s2b6pbv : tensor<384xf32>
    %v8066 = stablehlo.multiply %armeans2b6pb, %armeans2b6pb : tensor<384xf32>
    %v8067 = stablehlo.multiply %v8064, %v8066 : tensor<384xf32>
    %v8068 = stablehlo.add %v8065, %v8067 : tensor<384xf32>
    %v8069 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8070 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8071 = stablehlo.divide %v8062, %v8069 : tensor<384xf32>
    %v8072 = stablehlo.divide %v8068, %v8070 : tensor<384xf32>
    %v8073 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8074 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8075 = stablehlo.sqrt %v8072 : tensor<384xf32>
    %v8076 = stablehlo.add %v8075, %v8074 : tensor<384xf32>
    %v8077 = stablehlo.divide %v8071, %v8076 : tensor<384xf32>
    %v8078 = stablehlo.multiply %v8073, %v8077 : tensor<384xf32>
    %v8079 = stablehlo.subtract %s2b6pb, %v8078 : tensor<384xf32>
    %v8080 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8081 = stablehlo.multiply %v8080, %v8073 : tensor<384xf32>
    %v8082 = stablehlo.multiply %v8081, %s2b6pb : tensor<384xf32>
    %v8083 = stablehlo.subtract %v8079, %v8082 : tensor<384xf32>
    %arsums2b6lg = "stablehlo.all_reduce"(%v1811) ({
    ^bb0(%aras2b6lg: tensor<f32>, %arbs2b6lg: tensor<f32>):
      %aradds2b6lg = stablehlo.add %aras2b6lg, %arbs2b6lg : tensor<f32>
      stablehlo.return %aradds2b6lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b6lg = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b6lg = stablehlo.divide %arsums2b6lg, %arns2b6lg : tensor<384xf32>
    %v8084 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8085 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8086 = stablehlo.multiply %v8084, %s2b6lgm : tensor<384xf32>
    %v8087 = stablehlo.multiply %v8085, %armeans2b6lg : tensor<384xf32>
    %v8088 = stablehlo.add %v8086, %v8087 : tensor<384xf32>
    %v8089 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8090 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8091 = stablehlo.multiply %v8089, %s2b6lgv : tensor<384xf32>
    %v8092 = stablehlo.multiply %armeans2b6lg, %armeans2b6lg : tensor<384xf32>
    %v8093 = stablehlo.multiply %v8090, %v8092 : tensor<384xf32>
    %v8094 = stablehlo.add %v8091, %v8093 : tensor<384xf32>
    %v8095 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8096 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8097 = stablehlo.multiply %v8095, %s2b6lgm : tensor<384xf32>
    %v8098 = stablehlo.multiply %v8096, %armeans2b6lg : tensor<384xf32>
    %v8099 = stablehlo.add %v8097, %v8098 : tensor<384xf32>
    %v8100 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8101 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8102 = stablehlo.multiply %v8100, %s2b6lgv : tensor<384xf32>
    %v8103 = stablehlo.multiply %armeans2b6lg, %armeans2b6lg : tensor<384xf32>
    %v8104 = stablehlo.multiply %v8101, %v8103 : tensor<384xf32>
    %v8105 = stablehlo.add %v8102, %v8104 : tensor<384xf32>
    %v8106 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8107 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8108 = stablehlo.divide %v8099, %v8106 : tensor<384xf32>
    %v8109 = stablehlo.divide %v8105, %v8107 : tensor<384xf32>
    %v8110 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8111 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8112 = stablehlo.sqrt %v8109 : tensor<384xf32>
    %v8113 = stablehlo.add %v8112, %v8111 : tensor<384xf32>
    %v8114 = stablehlo.divide %v8108, %v8113 : tensor<384xf32>
    %v8115 = stablehlo.multiply %v8110, %v8114 : tensor<384xf32>
    %v8116 = stablehlo.subtract %s2b6lg, %v8115 : tensor<384xf32>
    %v8117 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8118 = stablehlo.multiply %v8117, %v8110 : tensor<384xf32>
    %v8119 = stablehlo.multiply %v8118, %s2b6lg : tensor<384xf32>
    %v8120 = stablehlo.subtract %v8116, %v8119 : tensor<384xf32>
    %arsums2b7dW = "stablehlo.all_reduce"(%v1734) ({
    ^bb0(%aras2b7dW: tensor<f32>, %arbs2b7dW: tensor<f32>):
      %aradds2b7dW = stablehlo.add %aras2b7dW, %arbs2b7dW : tensor<f32>
      stablehlo.return %aradds2b7dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1x7x7xf32>) -> tensor<384x1x7x7xf32>
    %arns2b7dW = stablehlo.constant dense<2.0> : tensor<384x1x7x7xf32>
    %armeans2b7dW = stablehlo.divide %arsums2b7dW, %arns2b7dW : tensor<384x1x7x7xf32>
    %v8121 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8122 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8123 = stablehlo.multiply %v8121, %s2b7dWm : tensor<384x1x7x7xf32>
    %v8124 = stablehlo.multiply %v8122, %armeans2b7dW : tensor<384x1x7x7xf32>
    %v8125 = stablehlo.add %v8123, %v8124 : tensor<384x1x7x7xf32>
    %v8126 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8127 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8128 = stablehlo.multiply %v8126, %s2b7dWv : tensor<384x1x7x7xf32>
    %v8129 = stablehlo.multiply %armeans2b7dW, %armeans2b7dW : tensor<384x1x7x7xf32>
    %v8130 = stablehlo.multiply %v8127, %v8129 : tensor<384x1x7x7xf32>
    %v8131 = stablehlo.add %v8128, %v8130 : tensor<384x1x7x7xf32>
    %v8132 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8133 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8134 = stablehlo.multiply %v8132, %s2b7dWm : tensor<384x1x7x7xf32>
    %v8135 = stablehlo.multiply %v8133, %armeans2b7dW : tensor<384x1x7x7xf32>
    %v8136 = stablehlo.add %v8134, %v8135 : tensor<384x1x7x7xf32>
    %v8137 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8138 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8139 = stablehlo.multiply %v8137, %s2b7dWv : tensor<384x1x7x7xf32>
    %v8140 = stablehlo.multiply %armeans2b7dW, %armeans2b7dW : tensor<384x1x7x7xf32>
    %v8141 = stablehlo.multiply %v8138, %v8140 : tensor<384x1x7x7xf32>
    %v8142 = stablehlo.add %v8139, %v8141 : tensor<384x1x7x7xf32>
    %v8143 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8144 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8145 = stablehlo.divide %v8136, %v8143 : tensor<384x1x7x7xf32>
    %v8146 = stablehlo.divide %v8142, %v8144 : tensor<384x1x7x7xf32>
    %v8147 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8148 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8149 = stablehlo.sqrt %v8146 : tensor<384x1x7x7xf32>
    %v8150 = stablehlo.add %v8149, %v8148 : tensor<384x1x7x7xf32>
    %v8151 = stablehlo.divide %v8145, %v8150 : tensor<384x1x7x7xf32>
    %v8152 = stablehlo.multiply %v8147, %v8151 : tensor<384x1x7x7xf32>
    %v8153 = stablehlo.subtract %s2b7dW, %v8152 : tensor<384x1x7x7xf32>
    %v8154 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8155 = stablehlo.multiply %v8154, %v8147 : tensor<384x1x7x7xf32>
    %v8156 = stablehlo.multiply %v8155, %s2b7dW : tensor<384x1x7x7xf32>
    %v8157 = stablehlo.subtract %v8153, %v8156 : tensor<384x1x7x7xf32>
    %arsums2b7db = "stablehlo.all_reduce"(%v1737) ({
    ^bb0(%aras2b7db: tensor<f32>, %arbs2b7db: tensor<f32>):
      %aradds2b7db = stablehlo.add %aras2b7db, %arbs2b7db : tensor<f32>
      stablehlo.return %aradds2b7db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b7db = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b7db = stablehlo.divide %arsums2b7db, %arns2b7db : tensor<384xf32>
    %v8158 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8159 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8160 = stablehlo.multiply %v8158, %s2b7dbm : tensor<384xf32>
    %v8161 = stablehlo.multiply %v8159, %armeans2b7db : tensor<384xf32>
    %v8162 = stablehlo.add %v8160, %v8161 : tensor<384xf32>
    %v8163 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8164 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8165 = stablehlo.multiply %v8163, %s2b7dbv : tensor<384xf32>
    %v8166 = stablehlo.multiply %armeans2b7db, %armeans2b7db : tensor<384xf32>
    %v8167 = stablehlo.multiply %v8164, %v8166 : tensor<384xf32>
    %v8168 = stablehlo.add %v8165, %v8167 : tensor<384xf32>
    %v8169 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8170 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8171 = stablehlo.multiply %v8169, %s2b7dbm : tensor<384xf32>
    %v8172 = stablehlo.multiply %v8170, %armeans2b7db : tensor<384xf32>
    %v8173 = stablehlo.add %v8171, %v8172 : tensor<384xf32>
    %v8174 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8175 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8176 = stablehlo.multiply %v8174, %s2b7dbv : tensor<384xf32>
    %v8177 = stablehlo.multiply %armeans2b7db, %armeans2b7db : tensor<384xf32>
    %v8178 = stablehlo.multiply %v8175, %v8177 : tensor<384xf32>
    %v8179 = stablehlo.add %v8176, %v8178 : tensor<384xf32>
    %v8180 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8181 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8182 = stablehlo.divide %v8173, %v8180 : tensor<384xf32>
    %v8183 = stablehlo.divide %v8179, %v8181 : tensor<384xf32>
    %v8184 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8185 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8186 = stablehlo.sqrt %v8183 : tensor<384xf32>
    %v8187 = stablehlo.add %v8186, %v8185 : tensor<384xf32>
    %v8188 = stablehlo.divide %v8182, %v8187 : tensor<384xf32>
    %v8189 = stablehlo.multiply %v8184, %v8188 : tensor<384xf32>
    %v8190 = stablehlo.subtract %s2b7db, %v8189 : tensor<384xf32>
    %v8191 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8192 = stablehlo.multiply %v8191, %v8184 : tensor<384xf32>
    %v8193 = stablehlo.multiply %v8192, %s2b7db : tensor<384xf32>
    %v8194 = stablehlo.subtract %v8190, %v8193 : tensor<384xf32>
    %arsums2b7ng = "stablehlo.all_reduce"(%v1726) ({
    ^bb0(%aras2b7ng: tensor<f32>, %arbs2b7ng: tensor<f32>):
      %aradds2b7ng = stablehlo.add %aras2b7ng, %arbs2b7ng : tensor<f32>
      stablehlo.return %aradds2b7ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b7ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b7ng = stablehlo.divide %arsums2b7ng, %arns2b7ng : tensor<f32>
    %v8195 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8196 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8197 = stablehlo.multiply %v8195, %s2b7ngm : tensor<f32>
    %v8198 = stablehlo.multiply %v8196, %armeans2b7ng : tensor<f32>
    %v8199 = stablehlo.add %v8197, %v8198 : tensor<f32>
    %v8200 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8201 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8202 = stablehlo.multiply %v8200, %s2b7ngv : tensor<f32>
    %v8203 = stablehlo.multiply %armeans2b7ng, %armeans2b7ng : tensor<f32>
    %v8204 = stablehlo.multiply %v8201, %v8203 : tensor<f32>
    %v8205 = stablehlo.add %v8202, %v8204 : tensor<f32>
    %v8206 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8207 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8208 = stablehlo.multiply %v8206, %s2b7ngm : tensor<f32>
    %v8209 = stablehlo.multiply %v8207, %armeans2b7ng : tensor<f32>
    %v8210 = stablehlo.add %v8208, %v8209 : tensor<f32>
    %v8211 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8212 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8213 = stablehlo.multiply %v8211, %s2b7ngv : tensor<f32>
    %v8214 = stablehlo.multiply %armeans2b7ng, %armeans2b7ng : tensor<f32>
    %v8215 = stablehlo.multiply %v8212, %v8214 : tensor<f32>
    %v8216 = stablehlo.add %v8213, %v8215 : tensor<f32>
    %v8217 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8218 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8219 = stablehlo.divide %v8210, %v8217 : tensor<f32>
    %v8220 = stablehlo.divide %v8216, %v8218 : tensor<f32>
    %v8221 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8222 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8223 = stablehlo.sqrt %v8220 : tensor<f32>
    %v8224 = stablehlo.add %v8223, %v8222 : tensor<f32>
    %v8225 = stablehlo.divide %v8219, %v8224 : tensor<f32>
    %v8226 = stablehlo.multiply %v8221, %v8225 : tensor<f32>
    %v8227 = stablehlo.subtract %s2b7ng, %v8226 : tensor<f32>
    %v8228 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8229 = stablehlo.multiply %v8228, %v8221 : tensor<f32>
    %v8230 = stablehlo.multiply %v8229, %s2b7ng : tensor<f32>
    %v8231 = stablehlo.subtract %v8227, %v8230 : tensor<f32>
    %arsums2b7nbt = "stablehlo.all_reduce"(%v1728) ({
    ^bb0(%aras2b7nbt: tensor<f32>, %arbs2b7nbt: tensor<f32>):
      %aradds2b7nbt = stablehlo.add %aras2b7nbt, %arbs2b7nbt : tensor<f32>
      stablehlo.return %aradds2b7nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b7nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b7nbt = stablehlo.divide %arsums2b7nbt, %arns2b7nbt : tensor<f32>
    %v8232 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8233 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8234 = stablehlo.multiply %v8232, %s2b7nbtm : tensor<f32>
    %v8235 = stablehlo.multiply %v8233, %armeans2b7nbt : tensor<f32>
    %v8236 = stablehlo.add %v8234, %v8235 : tensor<f32>
    %v8237 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8238 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8239 = stablehlo.multiply %v8237, %s2b7nbtv : tensor<f32>
    %v8240 = stablehlo.multiply %armeans2b7nbt, %armeans2b7nbt : tensor<f32>
    %v8241 = stablehlo.multiply %v8238, %v8240 : tensor<f32>
    %v8242 = stablehlo.add %v8239, %v8241 : tensor<f32>
    %v8243 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8244 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8245 = stablehlo.multiply %v8243, %s2b7nbtm : tensor<f32>
    %v8246 = stablehlo.multiply %v8244, %armeans2b7nbt : tensor<f32>
    %v8247 = stablehlo.add %v8245, %v8246 : tensor<f32>
    %v8248 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8249 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8250 = stablehlo.multiply %v8248, %s2b7nbtv : tensor<f32>
    %v8251 = stablehlo.multiply %armeans2b7nbt, %armeans2b7nbt : tensor<f32>
    %v8252 = stablehlo.multiply %v8249, %v8251 : tensor<f32>
    %v8253 = stablehlo.add %v8250, %v8252 : tensor<f32>
    %v8254 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8255 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8256 = stablehlo.divide %v8247, %v8254 : tensor<f32>
    %v8257 = stablehlo.divide %v8253, %v8255 : tensor<f32>
    %v8258 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8259 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8260 = stablehlo.sqrt %v8257 : tensor<f32>
    %v8261 = stablehlo.add %v8260, %v8259 : tensor<f32>
    %v8262 = stablehlo.divide %v8256, %v8261 : tensor<f32>
    %v8263 = stablehlo.multiply %v8258, %v8262 : tensor<f32>
    %v8264 = stablehlo.subtract %s2b7nbt, %v8263 : tensor<f32>
    %v8265 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8266 = stablehlo.multiply %v8265, %v8258 : tensor<f32>
    %v8267 = stablehlo.multiply %v8266, %s2b7nbt : tensor<f32>
    %v8268 = stablehlo.subtract %v8264, %v8267 : tensor<f32>
    %arsums2b7eW = "stablehlo.all_reduce"(%v1707) ({
    ^bb0(%aras2b7eW: tensor<f32>, %arbs2b7eW: tensor<f32>):
      %aradds2b7eW = stablehlo.add %aras2b7eW, %arbs2b7eW : tensor<f32>
      stablehlo.return %aradds2b7eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536x384x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %arns2b7eW = stablehlo.constant dense<2.0> : tensor<1536x384x1x1xf32>
    %armeans2b7eW = stablehlo.divide %arsums2b7eW, %arns2b7eW : tensor<1536x384x1x1xf32>
    %v8269 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8270 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8271 = stablehlo.multiply %v8269, %s2b7eWm : tensor<1536x384x1x1xf32>
    %v8272 = stablehlo.multiply %v8270, %armeans2b7eW : tensor<1536x384x1x1xf32>
    %v8273 = stablehlo.add %v8271, %v8272 : tensor<1536x384x1x1xf32>
    %v8274 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8275 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8276 = stablehlo.multiply %v8274, %s2b7eWv : tensor<1536x384x1x1xf32>
    %v8277 = stablehlo.multiply %armeans2b7eW, %armeans2b7eW : tensor<1536x384x1x1xf32>
    %v8278 = stablehlo.multiply %v8275, %v8277 : tensor<1536x384x1x1xf32>
    %v8279 = stablehlo.add %v8276, %v8278 : tensor<1536x384x1x1xf32>
    %v8280 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8281 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8282 = stablehlo.multiply %v8280, %s2b7eWm : tensor<1536x384x1x1xf32>
    %v8283 = stablehlo.multiply %v8281, %armeans2b7eW : tensor<1536x384x1x1xf32>
    %v8284 = stablehlo.add %v8282, %v8283 : tensor<1536x384x1x1xf32>
    %v8285 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8286 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8287 = stablehlo.multiply %v8285, %s2b7eWv : tensor<1536x384x1x1xf32>
    %v8288 = stablehlo.multiply %armeans2b7eW, %armeans2b7eW : tensor<1536x384x1x1xf32>
    %v8289 = stablehlo.multiply %v8286, %v8288 : tensor<1536x384x1x1xf32>
    %v8290 = stablehlo.add %v8287, %v8289 : tensor<1536x384x1x1xf32>
    %v8291 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8292 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8293 = stablehlo.divide %v8284, %v8291 : tensor<1536x384x1x1xf32>
    %v8294 = stablehlo.divide %v8290, %v8292 : tensor<1536x384x1x1xf32>
    %v8295 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8296 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8297 = stablehlo.sqrt %v8294 : tensor<1536x384x1x1xf32>
    %v8298 = stablehlo.add %v8297, %v8296 : tensor<1536x384x1x1xf32>
    %v8299 = stablehlo.divide %v8293, %v8298 : tensor<1536x384x1x1xf32>
    %v8300 = stablehlo.multiply %v8295, %v8299 : tensor<1536x384x1x1xf32>
    %v8301 = stablehlo.subtract %s2b7eW, %v8300 : tensor<1536x384x1x1xf32>
    %v8302 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8303 = stablehlo.multiply %v8302, %v8295 : tensor<1536x384x1x1xf32>
    %v8304 = stablehlo.multiply %v8303, %s2b7eW : tensor<1536x384x1x1xf32>
    %v8305 = stablehlo.subtract %v8301, %v8304 : tensor<1536x384x1x1xf32>
    %arsums2b7eb = "stablehlo.all_reduce"(%v1710) ({
    ^bb0(%aras2b7eb: tensor<f32>, %arbs2b7eb: tensor<f32>):
      %aradds2b7eb = stablehlo.add %aras2b7eb, %arbs2b7eb : tensor<f32>
      stablehlo.return %aradds2b7eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536xf32>) -> tensor<1536xf32>
    %arns2b7eb = stablehlo.constant dense<2.0> : tensor<1536xf32>
    %armeans2b7eb = stablehlo.divide %arsums2b7eb, %arns2b7eb : tensor<1536xf32>
    %v8306 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8307 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8308 = stablehlo.multiply %v8306, %s2b7ebm : tensor<1536xf32>
    %v8309 = stablehlo.multiply %v8307, %armeans2b7eb : tensor<1536xf32>
    %v8310 = stablehlo.add %v8308, %v8309 : tensor<1536xf32>
    %v8311 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8312 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8313 = stablehlo.multiply %v8311, %s2b7ebv : tensor<1536xf32>
    %v8314 = stablehlo.multiply %armeans2b7eb, %armeans2b7eb : tensor<1536xf32>
    %v8315 = stablehlo.multiply %v8312, %v8314 : tensor<1536xf32>
    %v8316 = stablehlo.add %v8313, %v8315 : tensor<1536xf32>
    %v8317 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8318 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8319 = stablehlo.multiply %v8317, %s2b7ebm : tensor<1536xf32>
    %v8320 = stablehlo.multiply %v8318, %armeans2b7eb : tensor<1536xf32>
    %v8321 = stablehlo.add %v8319, %v8320 : tensor<1536xf32>
    %v8322 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8323 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8324 = stablehlo.multiply %v8322, %s2b7ebv : tensor<1536xf32>
    %v8325 = stablehlo.multiply %armeans2b7eb, %armeans2b7eb : tensor<1536xf32>
    %v8326 = stablehlo.multiply %v8323, %v8325 : tensor<1536xf32>
    %v8327 = stablehlo.add %v8324, %v8326 : tensor<1536xf32>
    %v8328 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8329 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8330 = stablehlo.divide %v8321, %v8328 : tensor<1536xf32>
    %v8331 = stablehlo.divide %v8327, %v8329 : tensor<1536xf32>
    %v8332 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8333 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8334 = stablehlo.sqrt %v8331 : tensor<1536xf32>
    %v8335 = stablehlo.add %v8334, %v8333 : tensor<1536xf32>
    %v8336 = stablehlo.divide %v8330, %v8335 : tensor<1536xf32>
    %v8337 = stablehlo.multiply %v8332, %v8336 : tensor<1536xf32>
    %v8338 = stablehlo.subtract %s2b7eb, %v8337 : tensor<1536xf32>
    %v8339 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8340 = stablehlo.multiply %v8339, %v8332 : tensor<1536xf32>
    %v8341 = stablehlo.multiply %v8340, %s2b7eb : tensor<1536xf32>
    %v8342 = stablehlo.subtract %v8338, %v8341 : tensor<1536xf32>
    %arsums2b7pW = "stablehlo.all_reduce"(%v1698) ({
    ^bb0(%aras2b7pW: tensor<f32>, %arbs2b7pW: tensor<f32>):
      %aradds2b7pW = stablehlo.add %aras2b7pW, %arbs2b7pW : tensor<f32>
      stablehlo.return %aradds2b7pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1536x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %arns2b7pW = stablehlo.constant dense<2.0> : tensor<384x1536x1x1xf32>
    %armeans2b7pW = stablehlo.divide %arsums2b7pW, %arns2b7pW : tensor<384x1536x1x1xf32>
    %v8343 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8344 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8345 = stablehlo.multiply %v8343, %s2b7pWm : tensor<384x1536x1x1xf32>
    %v8346 = stablehlo.multiply %v8344, %armeans2b7pW : tensor<384x1536x1x1xf32>
    %v8347 = stablehlo.add %v8345, %v8346 : tensor<384x1536x1x1xf32>
    %v8348 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8349 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8350 = stablehlo.multiply %v8348, %s2b7pWv : tensor<384x1536x1x1xf32>
    %v8351 = stablehlo.multiply %armeans2b7pW, %armeans2b7pW : tensor<384x1536x1x1xf32>
    %v8352 = stablehlo.multiply %v8349, %v8351 : tensor<384x1536x1x1xf32>
    %v8353 = stablehlo.add %v8350, %v8352 : tensor<384x1536x1x1xf32>
    %v8354 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8355 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8356 = stablehlo.multiply %v8354, %s2b7pWm : tensor<384x1536x1x1xf32>
    %v8357 = stablehlo.multiply %v8355, %armeans2b7pW : tensor<384x1536x1x1xf32>
    %v8358 = stablehlo.add %v8356, %v8357 : tensor<384x1536x1x1xf32>
    %v8359 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8360 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8361 = stablehlo.multiply %v8359, %s2b7pWv : tensor<384x1536x1x1xf32>
    %v8362 = stablehlo.multiply %armeans2b7pW, %armeans2b7pW : tensor<384x1536x1x1xf32>
    %v8363 = stablehlo.multiply %v8360, %v8362 : tensor<384x1536x1x1xf32>
    %v8364 = stablehlo.add %v8361, %v8363 : tensor<384x1536x1x1xf32>
    %v8365 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8366 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8367 = stablehlo.divide %v8358, %v8365 : tensor<384x1536x1x1xf32>
    %v8368 = stablehlo.divide %v8364, %v8366 : tensor<384x1536x1x1xf32>
    %v8369 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8370 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8371 = stablehlo.sqrt %v8368 : tensor<384x1536x1x1xf32>
    %v8372 = stablehlo.add %v8371, %v8370 : tensor<384x1536x1x1xf32>
    %v8373 = stablehlo.divide %v8367, %v8372 : tensor<384x1536x1x1xf32>
    %v8374 = stablehlo.multiply %v8369, %v8373 : tensor<384x1536x1x1xf32>
    %v8375 = stablehlo.subtract %s2b7pW, %v8374 : tensor<384x1536x1x1xf32>
    %v8376 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8377 = stablehlo.multiply %v8376, %v8369 : tensor<384x1536x1x1xf32>
    %v8378 = stablehlo.multiply %v8377, %s2b7pW : tensor<384x1536x1x1xf32>
    %v8379 = stablehlo.subtract %v8375, %v8378 : tensor<384x1536x1x1xf32>
    %arsums2b7pb = "stablehlo.all_reduce"(%v1701) ({
    ^bb0(%aras2b7pb: tensor<f32>, %arbs2b7pb: tensor<f32>):
      %aradds2b7pb = stablehlo.add %aras2b7pb, %arbs2b7pb : tensor<f32>
      stablehlo.return %aradds2b7pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b7pb = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b7pb = stablehlo.divide %arsums2b7pb, %arns2b7pb : tensor<384xf32>
    %v8380 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8381 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8382 = stablehlo.multiply %v8380, %s2b7pbm : tensor<384xf32>
    %v8383 = stablehlo.multiply %v8381, %armeans2b7pb : tensor<384xf32>
    %v8384 = stablehlo.add %v8382, %v8383 : tensor<384xf32>
    %v8385 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8386 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8387 = stablehlo.multiply %v8385, %s2b7pbv : tensor<384xf32>
    %v8388 = stablehlo.multiply %armeans2b7pb, %armeans2b7pb : tensor<384xf32>
    %v8389 = stablehlo.multiply %v8386, %v8388 : tensor<384xf32>
    %v8390 = stablehlo.add %v8387, %v8389 : tensor<384xf32>
    %v8391 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8392 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8393 = stablehlo.multiply %v8391, %s2b7pbm : tensor<384xf32>
    %v8394 = stablehlo.multiply %v8392, %armeans2b7pb : tensor<384xf32>
    %v8395 = stablehlo.add %v8393, %v8394 : tensor<384xf32>
    %v8396 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8397 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8398 = stablehlo.multiply %v8396, %s2b7pbv : tensor<384xf32>
    %v8399 = stablehlo.multiply %armeans2b7pb, %armeans2b7pb : tensor<384xf32>
    %v8400 = stablehlo.multiply %v8397, %v8399 : tensor<384xf32>
    %v8401 = stablehlo.add %v8398, %v8400 : tensor<384xf32>
    %v8402 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8403 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8404 = stablehlo.divide %v8395, %v8402 : tensor<384xf32>
    %v8405 = stablehlo.divide %v8401, %v8403 : tensor<384xf32>
    %v8406 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8407 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8408 = stablehlo.sqrt %v8405 : tensor<384xf32>
    %v8409 = stablehlo.add %v8408, %v8407 : tensor<384xf32>
    %v8410 = stablehlo.divide %v8404, %v8409 : tensor<384xf32>
    %v8411 = stablehlo.multiply %v8406, %v8410 : tensor<384xf32>
    %v8412 = stablehlo.subtract %s2b7pb, %v8411 : tensor<384xf32>
    %v8413 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8414 = stablehlo.multiply %v8413, %v8406 : tensor<384xf32>
    %v8415 = stablehlo.multiply %v8414, %s2b7pb : tensor<384xf32>
    %v8416 = stablehlo.subtract %v8412, %v8415 : tensor<384xf32>
    %arsums2b7lg = "stablehlo.all_reduce"(%v1692) ({
    ^bb0(%aras2b7lg: tensor<f32>, %arbs2b7lg: tensor<f32>):
      %aradds2b7lg = stablehlo.add %aras2b7lg, %arbs2b7lg : tensor<f32>
      stablehlo.return %aradds2b7lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b7lg = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b7lg = stablehlo.divide %arsums2b7lg, %arns2b7lg : tensor<384xf32>
    %v8417 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8418 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8419 = stablehlo.multiply %v8417, %s2b7lgm : tensor<384xf32>
    %v8420 = stablehlo.multiply %v8418, %armeans2b7lg : tensor<384xf32>
    %v8421 = stablehlo.add %v8419, %v8420 : tensor<384xf32>
    %v8422 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8423 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8424 = stablehlo.multiply %v8422, %s2b7lgv : tensor<384xf32>
    %v8425 = stablehlo.multiply %armeans2b7lg, %armeans2b7lg : tensor<384xf32>
    %v8426 = stablehlo.multiply %v8423, %v8425 : tensor<384xf32>
    %v8427 = stablehlo.add %v8424, %v8426 : tensor<384xf32>
    %v8428 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8429 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8430 = stablehlo.multiply %v8428, %s2b7lgm : tensor<384xf32>
    %v8431 = stablehlo.multiply %v8429, %armeans2b7lg : tensor<384xf32>
    %v8432 = stablehlo.add %v8430, %v8431 : tensor<384xf32>
    %v8433 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8434 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8435 = stablehlo.multiply %v8433, %s2b7lgv : tensor<384xf32>
    %v8436 = stablehlo.multiply %armeans2b7lg, %armeans2b7lg : tensor<384xf32>
    %v8437 = stablehlo.multiply %v8434, %v8436 : tensor<384xf32>
    %v8438 = stablehlo.add %v8435, %v8437 : tensor<384xf32>
    %v8439 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8440 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8441 = stablehlo.divide %v8432, %v8439 : tensor<384xf32>
    %v8442 = stablehlo.divide %v8438, %v8440 : tensor<384xf32>
    %v8443 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8444 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8445 = stablehlo.sqrt %v8442 : tensor<384xf32>
    %v8446 = stablehlo.add %v8445, %v8444 : tensor<384xf32>
    %v8447 = stablehlo.divide %v8441, %v8446 : tensor<384xf32>
    %v8448 = stablehlo.multiply %v8443, %v8447 : tensor<384xf32>
    %v8449 = stablehlo.subtract %s2b7lg, %v8448 : tensor<384xf32>
    %v8450 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8451 = stablehlo.multiply %v8450, %v8443 : tensor<384xf32>
    %v8452 = stablehlo.multiply %v8451, %s2b7lg : tensor<384xf32>
    %v8453 = stablehlo.subtract %v8449, %v8452 : tensor<384xf32>
    %arsums2b8dW = "stablehlo.all_reduce"(%v1615) ({
    ^bb0(%aras2b8dW: tensor<f32>, %arbs2b8dW: tensor<f32>):
      %aradds2b8dW = stablehlo.add %aras2b8dW, %arbs2b8dW : tensor<f32>
      stablehlo.return %aradds2b8dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1x7x7xf32>) -> tensor<384x1x7x7xf32>
    %arns2b8dW = stablehlo.constant dense<2.0> : tensor<384x1x7x7xf32>
    %armeans2b8dW = stablehlo.divide %arsums2b8dW, %arns2b8dW : tensor<384x1x7x7xf32>
    %v8454 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8455 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8456 = stablehlo.multiply %v8454, %s2b8dWm : tensor<384x1x7x7xf32>
    %v8457 = stablehlo.multiply %v8455, %armeans2b8dW : tensor<384x1x7x7xf32>
    %v8458 = stablehlo.add %v8456, %v8457 : tensor<384x1x7x7xf32>
    %v8459 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8460 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8461 = stablehlo.multiply %v8459, %s2b8dWv : tensor<384x1x7x7xf32>
    %v8462 = stablehlo.multiply %armeans2b8dW, %armeans2b8dW : tensor<384x1x7x7xf32>
    %v8463 = stablehlo.multiply %v8460, %v8462 : tensor<384x1x7x7xf32>
    %v8464 = stablehlo.add %v8461, %v8463 : tensor<384x1x7x7xf32>
    %v8465 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8466 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8467 = stablehlo.multiply %v8465, %s2b8dWm : tensor<384x1x7x7xf32>
    %v8468 = stablehlo.multiply %v8466, %armeans2b8dW : tensor<384x1x7x7xf32>
    %v8469 = stablehlo.add %v8467, %v8468 : tensor<384x1x7x7xf32>
    %v8470 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8471 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8472 = stablehlo.multiply %v8470, %s2b8dWv : tensor<384x1x7x7xf32>
    %v8473 = stablehlo.multiply %armeans2b8dW, %armeans2b8dW : tensor<384x1x7x7xf32>
    %v8474 = stablehlo.multiply %v8471, %v8473 : tensor<384x1x7x7xf32>
    %v8475 = stablehlo.add %v8472, %v8474 : tensor<384x1x7x7xf32>
    %v8476 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8477 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8478 = stablehlo.divide %v8469, %v8476 : tensor<384x1x7x7xf32>
    %v8479 = stablehlo.divide %v8475, %v8477 : tensor<384x1x7x7xf32>
    %v8480 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8481 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8482 = stablehlo.sqrt %v8479 : tensor<384x1x7x7xf32>
    %v8483 = stablehlo.add %v8482, %v8481 : tensor<384x1x7x7xf32>
    %v8484 = stablehlo.divide %v8478, %v8483 : tensor<384x1x7x7xf32>
    %v8485 = stablehlo.multiply %v8480, %v8484 : tensor<384x1x7x7xf32>
    %v8486 = stablehlo.subtract %s2b8dW, %v8485 : tensor<384x1x7x7xf32>
    %v8487 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8488 = stablehlo.multiply %v8487, %v8480 : tensor<384x1x7x7xf32>
    %v8489 = stablehlo.multiply %v8488, %s2b8dW : tensor<384x1x7x7xf32>
    %v8490 = stablehlo.subtract %v8486, %v8489 : tensor<384x1x7x7xf32>
    %arsums2b8db = "stablehlo.all_reduce"(%v1618) ({
    ^bb0(%aras2b8db: tensor<f32>, %arbs2b8db: tensor<f32>):
      %aradds2b8db = stablehlo.add %aras2b8db, %arbs2b8db : tensor<f32>
      stablehlo.return %aradds2b8db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b8db = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b8db = stablehlo.divide %arsums2b8db, %arns2b8db : tensor<384xf32>
    %v8491 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8492 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8493 = stablehlo.multiply %v8491, %s2b8dbm : tensor<384xf32>
    %v8494 = stablehlo.multiply %v8492, %armeans2b8db : tensor<384xf32>
    %v8495 = stablehlo.add %v8493, %v8494 : tensor<384xf32>
    %v8496 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8497 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8498 = stablehlo.multiply %v8496, %s2b8dbv : tensor<384xf32>
    %v8499 = stablehlo.multiply %armeans2b8db, %armeans2b8db : tensor<384xf32>
    %v8500 = stablehlo.multiply %v8497, %v8499 : tensor<384xf32>
    %v8501 = stablehlo.add %v8498, %v8500 : tensor<384xf32>
    %v8502 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8503 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8504 = stablehlo.multiply %v8502, %s2b8dbm : tensor<384xf32>
    %v8505 = stablehlo.multiply %v8503, %armeans2b8db : tensor<384xf32>
    %v8506 = stablehlo.add %v8504, %v8505 : tensor<384xf32>
    %v8507 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8508 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8509 = stablehlo.multiply %v8507, %s2b8dbv : tensor<384xf32>
    %v8510 = stablehlo.multiply %armeans2b8db, %armeans2b8db : tensor<384xf32>
    %v8511 = stablehlo.multiply %v8508, %v8510 : tensor<384xf32>
    %v8512 = stablehlo.add %v8509, %v8511 : tensor<384xf32>
    %v8513 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8514 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8515 = stablehlo.divide %v8506, %v8513 : tensor<384xf32>
    %v8516 = stablehlo.divide %v8512, %v8514 : tensor<384xf32>
    %v8517 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8518 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8519 = stablehlo.sqrt %v8516 : tensor<384xf32>
    %v8520 = stablehlo.add %v8519, %v8518 : tensor<384xf32>
    %v8521 = stablehlo.divide %v8515, %v8520 : tensor<384xf32>
    %v8522 = stablehlo.multiply %v8517, %v8521 : tensor<384xf32>
    %v8523 = stablehlo.subtract %s2b8db, %v8522 : tensor<384xf32>
    %v8524 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8525 = stablehlo.multiply %v8524, %v8517 : tensor<384xf32>
    %v8526 = stablehlo.multiply %v8525, %s2b8db : tensor<384xf32>
    %v8527 = stablehlo.subtract %v8523, %v8526 : tensor<384xf32>
    %arsums2b8ng = "stablehlo.all_reduce"(%v1607) ({
    ^bb0(%aras2b8ng: tensor<f32>, %arbs2b8ng: tensor<f32>):
      %aradds2b8ng = stablehlo.add %aras2b8ng, %arbs2b8ng : tensor<f32>
      stablehlo.return %aradds2b8ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b8ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b8ng = stablehlo.divide %arsums2b8ng, %arns2b8ng : tensor<f32>
    %v8528 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8529 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8530 = stablehlo.multiply %v8528, %s2b8ngm : tensor<f32>
    %v8531 = stablehlo.multiply %v8529, %armeans2b8ng : tensor<f32>
    %v8532 = stablehlo.add %v8530, %v8531 : tensor<f32>
    %v8533 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8534 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8535 = stablehlo.multiply %v8533, %s2b8ngv : tensor<f32>
    %v8536 = stablehlo.multiply %armeans2b8ng, %armeans2b8ng : tensor<f32>
    %v8537 = stablehlo.multiply %v8534, %v8536 : tensor<f32>
    %v8538 = stablehlo.add %v8535, %v8537 : tensor<f32>
    %v8539 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8540 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8541 = stablehlo.multiply %v8539, %s2b8ngm : tensor<f32>
    %v8542 = stablehlo.multiply %v8540, %armeans2b8ng : tensor<f32>
    %v8543 = stablehlo.add %v8541, %v8542 : tensor<f32>
    %v8544 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8545 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8546 = stablehlo.multiply %v8544, %s2b8ngv : tensor<f32>
    %v8547 = stablehlo.multiply %armeans2b8ng, %armeans2b8ng : tensor<f32>
    %v8548 = stablehlo.multiply %v8545, %v8547 : tensor<f32>
    %v8549 = stablehlo.add %v8546, %v8548 : tensor<f32>
    %v8550 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8551 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8552 = stablehlo.divide %v8543, %v8550 : tensor<f32>
    %v8553 = stablehlo.divide %v8549, %v8551 : tensor<f32>
    %v8554 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8555 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8556 = stablehlo.sqrt %v8553 : tensor<f32>
    %v8557 = stablehlo.add %v8556, %v8555 : tensor<f32>
    %v8558 = stablehlo.divide %v8552, %v8557 : tensor<f32>
    %v8559 = stablehlo.multiply %v8554, %v8558 : tensor<f32>
    %v8560 = stablehlo.subtract %s2b8ng, %v8559 : tensor<f32>
    %v8561 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8562 = stablehlo.multiply %v8561, %v8554 : tensor<f32>
    %v8563 = stablehlo.multiply %v8562, %s2b8ng : tensor<f32>
    %v8564 = stablehlo.subtract %v8560, %v8563 : tensor<f32>
    %arsums2b8nbt = "stablehlo.all_reduce"(%v1609) ({
    ^bb0(%aras2b8nbt: tensor<f32>, %arbs2b8nbt: tensor<f32>):
      %aradds2b8nbt = stablehlo.add %aras2b8nbt, %arbs2b8nbt : tensor<f32>
      stablehlo.return %aradds2b8nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns2b8nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans2b8nbt = stablehlo.divide %arsums2b8nbt, %arns2b8nbt : tensor<f32>
    %v8565 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8566 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8567 = stablehlo.multiply %v8565, %s2b8nbtm : tensor<f32>
    %v8568 = stablehlo.multiply %v8566, %armeans2b8nbt : tensor<f32>
    %v8569 = stablehlo.add %v8567, %v8568 : tensor<f32>
    %v8570 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8571 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8572 = stablehlo.multiply %v8570, %s2b8nbtv : tensor<f32>
    %v8573 = stablehlo.multiply %armeans2b8nbt, %armeans2b8nbt : tensor<f32>
    %v8574 = stablehlo.multiply %v8571, %v8573 : tensor<f32>
    %v8575 = stablehlo.add %v8572, %v8574 : tensor<f32>
    %v8576 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8577 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8578 = stablehlo.multiply %v8576, %s2b8nbtm : tensor<f32>
    %v8579 = stablehlo.multiply %v8577, %armeans2b8nbt : tensor<f32>
    %v8580 = stablehlo.add %v8578, %v8579 : tensor<f32>
    %v8581 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8582 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8583 = stablehlo.multiply %v8581, %s2b8nbtv : tensor<f32>
    %v8584 = stablehlo.multiply %armeans2b8nbt, %armeans2b8nbt : tensor<f32>
    %v8585 = stablehlo.multiply %v8582, %v8584 : tensor<f32>
    %v8586 = stablehlo.add %v8583, %v8585 : tensor<f32>
    %v8587 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8588 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8589 = stablehlo.divide %v8580, %v8587 : tensor<f32>
    %v8590 = stablehlo.divide %v8586, %v8588 : tensor<f32>
    %v8591 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8592 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8593 = stablehlo.sqrt %v8590 : tensor<f32>
    %v8594 = stablehlo.add %v8593, %v8592 : tensor<f32>
    %v8595 = stablehlo.divide %v8589, %v8594 : tensor<f32>
    %v8596 = stablehlo.multiply %v8591, %v8595 : tensor<f32>
    %v8597 = stablehlo.subtract %s2b8nbt, %v8596 : tensor<f32>
    %v8598 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8599 = stablehlo.multiply %v8598, %v8591 : tensor<f32>
    %v8600 = stablehlo.multiply %v8599, %s2b8nbt : tensor<f32>
    %v8601 = stablehlo.subtract %v8597, %v8600 : tensor<f32>
    %arsums2b8eW = "stablehlo.all_reduce"(%v1588) ({
    ^bb0(%aras2b8eW: tensor<f32>, %arbs2b8eW: tensor<f32>):
      %aradds2b8eW = stablehlo.add %aras2b8eW, %arbs2b8eW : tensor<f32>
      stablehlo.return %aradds2b8eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536x384x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %arns2b8eW = stablehlo.constant dense<2.0> : tensor<1536x384x1x1xf32>
    %armeans2b8eW = stablehlo.divide %arsums2b8eW, %arns2b8eW : tensor<1536x384x1x1xf32>
    %v8602 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8603 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8604 = stablehlo.multiply %v8602, %s2b8eWm : tensor<1536x384x1x1xf32>
    %v8605 = stablehlo.multiply %v8603, %armeans2b8eW : tensor<1536x384x1x1xf32>
    %v8606 = stablehlo.add %v8604, %v8605 : tensor<1536x384x1x1xf32>
    %v8607 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8608 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8609 = stablehlo.multiply %v8607, %s2b8eWv : tensor<1536x384x1x1xf32>
    %v8610 = stablehlo.multiply %armeans2b8eW, %armeans2b8eW : tensor<1536x384x1x1xf32>
    %v8611 = stablehlo.multiply %v8608, %v8610 : tensor<1536x384x1x1xf32>
    %v8612 = stablehlo.add %v8609, %v8611 : tensor<1536x384x1x1xf32>
    %v8613 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8614 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8615 = stablehlo.multiply %v8613, %s2b8eWm : tensor<1536x384x1x1xf32>
    %v8616 = stablehlo.multiply %v8614, %armeans2b8eW : tensor<1536x384x1x1xf32>
    %v8617 = stablehlo.add %v8615, %v8616 : tensor<1536x384x1x1xf32>
    %v8618 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8619 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8620 = stablehlo.multiply %v8618, %s2b8eWv : tensor<1536x384x1x1xf32>
    %v8621 = stablehlo.multiply %armeans2b8eW, %armeans2b8eW : tensor<1536x384x1x1xf32>
    %v8622 = stablehlo.multiply %v8619, %v8621 : tensor<1536x384x1x1xf32>
    %v8623 = stablehlo.add %v8620, %v8622 : tensor<1536x384x1x1xf32>
    %v8624 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8625 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8626 = stablehlo.divide %v8617, %v8624 : tensor<1536x384x1x1xf32>
    %v8627 = stablehlo.divide %v8623, %v8625 : tensor<1536x384x1x1xf32>
    %v8628 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8629 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8630 = stablehlo.sqrt %v8627 : tensor<1536x384x1x1xf32>
    %v8631 = stablehlo.add %v8630, %v8629 : tensor<1536x384x1x1xf32>
    %v8632 = stablehlo.divide %v8626, %v8631 : tensor<1536x384x1x1xf32>
    %v8633 = stablehlo.multiply %v8628, %v8632 : tensor<1536x384x1x1xf32>
    %v8634 = stablehlo.subtract %s2b8eW, %v8633 : tensor<1536x384x1x1xf32>
    %v8635 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8636 = stablehlo.multiply %v8635, %v8628 : tensor<1536x384x1x1xf32>
    %v8637 = stablehlo.multiply %v8636, %s2b8eW : tensor<1536x384x1x1xf32>
    %v8638 = stablehlo.subtract %v8634, %v8637 : tensor<1536x384x1x1xf32>
    %arsums2b8eb = "stablehlo.all_reduce"(%v1591) ({
    ^bb0(%aras2b8eb: tensor<f32>, %arbs2b8eb: tensor<f32>):
      %aradds2b8eb = stablehlo.add %aras2b8eb, %arbs2b8eb : tensor<f32>
      stablehlo.return %aradds2b8eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<1536xf32>) -> tensor<1536xf32>
    %arns2b8eb = stablehlo.constant dense<2.0> : tensor<1536xf32>
    %armeans2b8eb = stablehlo.divide %arsums2b8eb, %arns2b8eb : tensor<1536xf32>
    %v8639 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8640 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8641 = stablehlo.multiply %v8639, %s2b8ebm : tensor<1536xf32>
    %v8642 = stablehlo.multiply %v8640, %armeans2b8eb : tensor<1536xf32>
    %v8643 = stablehlo.add %v8641, %v8642 : tensor<1536xf32>
    %v8644 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8645 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8646 = stablehlo.multiply %v8644, %s2b8ebv : tensor<1536xf32>
    %v8647 = stablehlo.multiply %armeans2b8eb, %armeans2b8eb : tensor<1536xf32>
    %v8648 = stablehlo.multiply %v8645, %v8647 : tensor<1536xf32>
    %v8649 = stablehlo.add %v8646, %v8648 : tensor<1536xf32>
    %v8650 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8651 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8652 = stablehlo.multiply %v8650, %s2b8ebm : tensor<1536xf32>
    %v8653 = stablehlo.multiply %v8651, %armeans2b8eb : tensor<1536xf32>
    %v8654 = stablehlo.add %v8652, %v8653 : tensor<1536xf32>
    %v8655 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8656 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8657 = stablehlo.multiply %v8655, %s2b8ebv : tensor<1536xf32>
    %v8658 = stablehlo.multiply %armeans2b8eb, %armeans2b8eb : tensor<1536xf32>
    %v8659 = stablehlo.multiply %v8656, %v8658 : tensor<1536xf32>
    %v8660 = stablehlo.add %v8657, %v8659 : tensor<1536xf32>
    %v8661 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8662 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8663 = stablehlo.divide %v8654, %v8661 : tensor<1536xf32>
    %v8664 = stablehlo.divide %v8660, %v8662 : tensor<1536xf32>
    %v8665 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8666 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8667 = stablehlo.sqrt %v8664 : tensor<1536xf32>
    %v8668 = stablehlo.add %v8667, %v8666 : tensor<1536xf32>
    %v8669 = stablehlo.divide %v8663, %v8668 : tensor<1536xf32>
    %v8670 = stablehlo.multiply %v8665, %v8669 : tensor<1536xf32>
    %v8671 = stablehlo.subtract %s2b8eb, %v8670 : tensor<1536xf32>
    %v8672 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8673 = stablehlo.multiply %v8672, %v8665 : tensor<1536xf32>
    %v8674 = stablehlo.multiply %v8673, %s2b8eb : tensor<1536xf32>
    %v8675 = stablehlo.subtract %v8671, %v8674 : tensor<1536xf32>
    %arsums2b8pW = "stablehlo.all_reduce"(%v1579) ({
    ^bb0(%aras2b8pW: tensor<f32>, %arbs2b8pW: tensor<f32>):
      %aradds2b8pW = stablehlo.add %aras2b8pW, %arbs2b8pW : tensor<f32>
      stablehlo.return %aradds2b8pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384x1536x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %arns2b8pW = stablehlo.constant dense<2.0> : tensor<384x1536x1x1xf32>
    %armeans2b8pW = stablehlo.divide %arsums2b8pW, %arns2b8pW : tensor<384x1536x1x1xf32>
    %v8676 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8677 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8678 = stablehlo.multiply %v8676, %s2b8pWm : tensor<384x1536x1x1xf32>
    %v8679 = stablehlo.multiply %v8677, %armeans2b8pW : tensor<384x1536x1x1xf32>
    %v8680 = stablehlo.add %v8678, %v8679 : tensor<384x1536x1x1xf32>
    %v8681 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8682 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8683 = stablehlo.multiply %v8681, %s2b8pWv : tensor<384x1536x1x1xf32>
    %v8684 = stablehlo.multiply %armeans2b8pW, %armeans2b8pW : tensor<384x1536x1x1xf32>
    %v8685 = stablehlo.multiply %v8682, %v8684 : tensor<384x1536x1x1xf32>
    %v8686 = stablehlo.add %v8683, %v8685 : tensor<384x1536x1x1xf32>
    %v8687 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8688 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8689 = stablehlo.multiply %v8687, %s2b8pWm : tensor<384x1536x1x1xf32>
    %v8690 = stablehlo.multiply %v8688, %armeans2b8pW : tensor<384x1536x1x1xf32>
    %v8691 = stablehlo.add %v8689, %v8690 : tensor<384x1536x1x1xf32>
    %v8692 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8693 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8694 = stablehlo.multiply %v8692, %s2b8pWv : tensor<384x1536x1x1xf32>
    %v8695 = stablehlo.multiply %armeans2b8pW, %armeans2b8pW : tensor<384x1536x1x1xf32>
    %v8696 = stablehlo.multiply %v8693, %v8695 : tensor<384x1536x1x1xf32>
    %v8697 = stablehlo.add %v8694, %v8696 : tensor<384x1536x1x1xf32>
    %v8698 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8699 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8700 = stablehlo.divide %v8691, %v8698 : tensor<384x1536x1x1xf32>
    %v8701 = stablehlo.divide %v8697, %v8699 : tensor<384x1536x1x1xf32>
    %v8702 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8703 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8704 = stablehlo.sqrt %v8701 : tensor<384x1536x1x1xf32>
    %v8705 = stablehlo.add %v8704, %v8703 : tensor<384x1536x1x1xf32>
    %v8706 = stablehlo.divide %v8700, %v8705 : tensor<384x1536x1x1xf32>
    %v8707 = stablehlo.multiply %v8702, %v8706 : tensor<384x1536x1x1xf32>
    %v8708 = stablehlo.subtract %s2b8pW, %v8707 : tensor<384x1536x1x1xf32>
    %v8709 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8710 = stablehlo.multiply %v8709, %v8702 : tensor<384x1536x1x1xf32>
    %v8711 = stablehlo.multiply %v8710, %s2b8pW : tensor<384x1536x1x1xf32>
    %v8712 = stablehlo.subtract %v8708, %v8711 : tensor<384x1536x1x1xf32>
    %arsums2b8pb = "stablehlo.all_reduce"(%v1582) ({
    ^bb0(%aras2b8pb: tensor<f32>, %arbs2b8pb: tensor<f32>):
      %aradds2b8pb = stablehlo.add %aras2b8pb, %arbs2b8pb : tensor<f32>
      stablehlo.return %aradds2b8pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b8pb = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b8pb = stablehlo.divide %arsums2b8pb, %arns2b8pb : tensor<384xf32>
    %v8713 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8714 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8715 = stablehlo.multiply %v8713, %s2b8pbm : tensor<384xf32>
    %v8716 = stablehlo.multiply %v8714, %armeans2b8pb : tensor<384xf32>
    %v8717 = stablehlo.add %v8715, %v8716 : tensor<384xf32>
    %v8718 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8719 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8720 = stablehlo.multiply %v8718, %s2b8pbv : tensor<384xf32>
    %v8721 = stablehlo.multiply %armeans2b8pb, %armeans2b8pb : tensor<384xf32>
    %v8722 = stablehlo.multiply %v8719, %v8721 : tensor<384xf32>
    %v8723 = stablehlo.add %v8720, %v8722 : tensor<384xf32>
    %v8724 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8725 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8726 = stablehlo.multiply %v8724, %s2b8pbm : tensor<384xf32>
    %v8727 = stablehlo.multiply %v8725, %armeans2b8pb : tensor<384xf32>
    %v8728 = stablehlo.add %v8726, %v8727 : tensor<384xf32>
    %v8729 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8730 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8731 = stablehlo.multiply %v8729, %s2b8pbv : tensor<384xf32>
    %v8732 = stablehlo.multiply %armeans2b8pb, %armeans2b8pb : tensor<384xf32>
    %v8733 = stablehlo.multiply %v8730, %v8732 : tensor<384xf32>
    %v8734 = stablehlo.add %v8731, %v8733 : tensor<384xf32>
    %v8735 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8736 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8737 = stablehlo.divide %v8728, %v8735 : tensor<384xf32>
    %v8738 = stablehlo.divide %v8734, %v8736 : tensor<384xf32>
    %v8739 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8740 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8741 = stablehlo.sqrt %v8738 : tensor<384xf32>
    %v8742 = stablehlo.add %v8741, %v8740 : tensor<384xf32>
    %v8743 = stablehlo.divide %v8737, %v8742 : tensor<384xf32>
    %v8744 = stablehlo.multiply %v8739, %v8743 : tensor<384xf32>
    %v8745 = stablehlo.subtract %s2b8pb, %v8744 : tensor<384xf32>
    %v8746 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8747 = stablehlo.multiply %v8746, %v8739 : tensor<384xf32>
    %v8748 = stablehlo.multiply %v8747, %s2b8pb : tensor<384xf32>
    %v8749 = stablehlo.subtract %v8745, %v8748 : tensor<384xf32>
    %arsums2b8lg = "stablehlo.all_reduce"(%v1573) ({
    ^bb0(%aras2b8lg: tensor<f32>, %arbs2b8lg: tensor<f32>):
      %aradds2b8lg = stablehlo.add %aras2b8lg, %arbs2b8lg : tensor<f32>
      stablehlo.return %aradds2b8lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<384xf32>) -> tensor<384xf32>
    %arns2b8lg = stablehlo.constant dense<2.0> : tensor<384xf32>
    %armeans2b8lg = stablehlo.divide %arsums2b8lg, %arns2b8lg : tensor<384xf32>
    %v8750 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8751 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8752 = stablehlo.multiply %v8750, %s2b8lgm : tensor<384xf32>
    %v8753 = stablehlo.multiply %v8751, %armeans2b8lg : tensor<384xf32>
    %v8754 = stablehlo.add %v8752, %v8753 : tensor<384xf32>
    %v8755 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8756 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8757 = stablehlo.multiply %v8755, %s2b8lgv : tensor<384xf32>
    %v8758 = stablehlo.multiply %armeans2b8lg, %armeans2b8lg : tensor<384xf32>
    %v8759 = stablehlo.multiply %v8756, %v8758 : tensor<384xf32>
    %v8760 = stablehlo.add %v8757, %v8759 : tensor<384xf32>
    %v8761 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8762 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8763 = stablehlo.multiply %v8761, %s2b8lgm : tensor<384xf32>
    %v8764 = stablehlo.multiply %v8762, %armeans2b8lg : tensor<384xf32>
    %v8765 = stablehlo.add %v8763, %v8764 : tensor<384xf32>
    %v8766 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8767 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8768 = stablehlo.multiply %v8766, %s2b8lgv : tensor<384xf32>
    %v8769 = stablehlo.multiply %armeans2b8lg, %armeans2b8lg : tensor<384xf32>
    %v8770 = stablehlo.multiply %v8767, %v8769 : tensor<384xf32>
    %v8771 = stablehlo.add %v8768, %v8770 : tensor<384xf32>
    %v8772 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8773 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8774 = stablehlo.divide %v8765, %v8772 : tensor<384xf32>
    %v8775 = stablehlo.divide %v8771, %v8773 : tensor<384xf32>
    %v8776 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8777 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8778 = stablehlo.sqrt %v8775 : tensor<384xf32>
    %v8779 = stablehlo.add %v8778, %v8777 : tensor<384xf32>
    %v8780 = stablehlo.divide %v8774, %v8779 : tensor<384xf32>
    %v8781 = stablehlo.multiply %v8776, %v8780 : tensor<384xf32>
    %v8782 = stablehlo.subtract %s2b8lg, %v8781 : tensor<384xf32>
    %v8783 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8784 = stablehlo.multiply %v8783, %v8776 : tensor<384xf32>
    %v8785 = stablehlo.multiply %v8784, %s2b8lg : tensor<384xf32>
    %v8786 = stablehlo.subtract %v8782, %v8785 : tensor<384xf32>
    %arsumd2ng = "stablehlo.all_reduce"(%v1489) ({
    ^bb0(%arad2ng: tensor<f32>, %arbd2ng: tensor<f32>):
      %araddd2ng = stablehlo.add %arad2ng, %arbd2ng : tensor<f32>
      stablehlo.return %araddd2ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arnd2ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeand2ng = stablehlo.divide %arsumd2ng, %arnd2ng : tensor<f32>
    %v8787 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8788 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8789 = stablehlo.multiply %v8787, %d2ngm : tensor<f32>
    %v8790 = stablehlo.multiply %v8788, %armeand2ng : tensor<f32>
    %v8791 = stablehlo.add %v8789, %v8790 : tensor<f32>
    %v8792 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8793 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8794 = stablehlo.multiply %v8792, %d2ngv : tensor<f32>
    %v8795 = stablehlo.multiply %armeand2ng, %armeand2ng : tensor<f32>
    %v8796 = stablehlo.multiply %v8793, %v8795 : tensor<f32>
    %v8797 = stablehlo.add %v8794, %v8796 : tensor<f32>
    %v8798 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8799 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8800 = stablehlo.multiply %v8798, %d2ngm : tensor<f32>
    %v8801 = stablehlo.multiply %v8799, %armeand2ng : tensor<f32>
    %v8802 = stablehlo.add %v8800, %v8801 : tensor<f32>
    %v8803 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8804 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8805 = stablehlo.multiply %v8803, %d2ngv : tensor<f32>
    %v8806 = stablehlo.multiply %armeand2ng, %armeand2ng : tensor<f32>
    %v8807 = stablehlo.multiply %v8804, %v8806 : tensor<f32>
    %v8808 = stablehlo.add %v8805, %v8807 : tensor<f32>
    %v8809 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8810 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8811 = stablehlo.divide %v8802, %v8809 : tensor<f32>
    %v8812 = stablehlo.divide %v8808, %v8810 : tensor<f32>
    %v8813 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8814 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8815 = stablehlo.sqrt %v8812 : tensor<f32>
    %v8816 = stablehlo.add %v8815, %v8814 : tensor<f32>
    %v8817 = stablehlo.divide %v8811, %v8816 : tensor<f32>
    %v8818 = stablehlo.multiply %v8813, %v8817 : tensor<f32>
    %v8819 = stablehlo.subtract %d2ng, %v8818 : tensor<f32>
    %v8820 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8821 = stablehlo.multiply %v8820, %v8813 : tensor<f32>
    %v8822 = stablehlo.multiply %v8821, %d2ng : tensor<f32>
    %v8823 = stablehlo.subtract %v8819, %v8822 : tensor<f32>
    %arsumd2nbt = "stablehlo.all_reduce"(%v1491) ({
    ^bb0(%arad2nbt: tensor<f32>, %arbd2nbt: tensor<f32>):
      %araddd2nbt = stablehlo.add %arad2nbt, %arbd2nbt : tensor<f32>
      stablehlo.return %araddd2nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arnd2nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeand2nbt = stablehlo.divide %arsumd2nbt, %arnd2nbt : tensor<f32>
    %v8824 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8825 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8826 = stablehlo.multiply %v8824, %d2nbtm : tensor<f32>
    %v8827 = stablehlo.multiply %v8825, %armeand2nbt : tensor<f32>
    %v8828 = stablehlo.add %v8826, %v8827 : tensor<f32>
    %v8829 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8830 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8831 = stablehlo.multiply %v8829, %d2nbtv : tensor<f32>
    %v8832 = stablehlo.multiply %armeand2nbt, %armeand2nbt : tensor<f32>
    %v8833 = stablehlo.multiply %v8830, %v8832 : tensor<f32>
    %v8834 = stablehlo.add %v8831, %v8833 : tensor<f32>
    %v8835 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8836 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8837 = stablehlo.multiply %v8835, %d2nbtm : tensor<f32>
    %v8838 = stablehlo.multiply %v8836, %armeand2nbt : tensor<f32>
    %v8839 = stablehlo.add %v8837, %v8838 : tensor<f32>
    %v8840 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8841 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8842 = stablehlo.multiply %v8840, %d2nbtv : tensor<f32>
    %v8843 = stablehlo.multiply %armeand2nbt, %armeand2nbt : tensor<f32>
    %v8844 = stablehlo.multiply %v8841, %v8843 : tensor<f32>
    %v8845 = stablehlo.add %v8842, %v8844 : tensor<f32>
    %v8846 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8847 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8848 = stablehlo.divide %v8839, %v8846 : tensor<f32>
    %v8849 = stablehlo.divide %v8845, %v8847 : tensor<f32>
    %v8850 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8851 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8852 = stablehlo.sqrt %v8849 : tensor<f32>
    %v8853 = stablehlo.add %v8852, %v8851 : tensor<f32>
    %v8854 = stablehlo.divide %v8848, %v8853 : tensor<f32>
    %v8855 = stablehlo.multiply %v8850, %v8854 : tensor<f32>
    %v8856 = stablehlo.subtract %d2nbt, %v8855 : tensor<f32>
    %v8857 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8858 = stablehlo.multiply %v8857, %v8850 : tensor<f32>
    %v8859 = stablehlo.multiply %v8858, %d2nbt : tensor<f32>
    %v8860 = stablehlo.subtract %v8856, %v8859 : tensor<f32>
    %arsumd2W = "stablehlo.all_reduce"(%v1499) ({
    ^bb0(%arad2W: tensor<f32>, %arbd2W: tensor<f32>):
      %araddd2W = stablehlo.add %arad2W, %arbd2W : tensor<f32>
      stablehlo.return %araddd2W : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768x384x2x2xf32>) -> tensor<768x384x2x2xf32>
    %arnd2W = stablehlo.constant dense<2.0> : tensor<768x384x2x2xf32>
    %armeand2W = stablehlo.divide %arsumd2W, %arnd2W : tensor<768x384x2x2xf32>
    %v8861 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8862 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8863 = stablehlo.multiply %v8861, %d2Wm : tensor<768x384x2x2xf32>
    %v8864 = stablehlo.multiply %v8862, %armeand2W : tensor<768x384x2x2xf32>
    %v8865 = stablehlo.add %v8863, %v8864 : tensor<768x384x2x2xf32>
    %v8866 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8867 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8868 = stablehlo.multiply %v8866, %d2Wv : tensor<768x384x2x2xf32>
    %v8869 = stablehlo.multiply %armeand2W, %armeand2W : tensor<768x384x2x2xf32>
    %v8870 = stablehlo.multiply %v8867, %v8869 : tensor<768x384x2x2xf32>
    %v8871 = stablehlo.add %v8868, %v8870 : tensor<768x384x2x2xf32>
    %v8872 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8873 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8874 = stablehlo.multiply %v8872, %d2Wm : tensor<768x384x2x2xf32>
    %v8875 = stablehlo.multiply %v8873, %armeand2W : tensor<768x384x2x2xf32>
    %v8876 = stablehlo.add %v8874, %v8875 : tensor<768x384x2x2xf32>
    %v8877 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8878 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8879 = stablehlo.multiply %v8877, %d2Wv : tensor<768x384x2x2xf32>
    %v8880 = stablehlo.multiply %armeand2W, %armeand2W : tensor<768x384x2x2xf32>
    %v8881 = stablehlo.multiply %v8878, %v8880 : tensor<768x384x2x2xf32>
    %v8882 = stablehlo.add %v8879, %v8881 : tensor<768x384x2x2xf32>
    %v8883 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8884 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8885 = stablehlo.divide %v8876, %v8883 : tensor<768x384x2x2xf32>
    %v8886 = stablehlo.divide %v8882, %v8884 : tensor<768x384x2x2xf32>
    %v8887 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8888 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8889 = stablehlo.sqrt %v8886 : tensor<768x384x2x2xf32>
    %v8890 = stablehlo.add %v8889, %v8888 : tensor<768x384x2x2xf32>
    %v8891 = stablehlo.divide %v8885, %v8890 : tensor<768x384x2x2xf32>
    %v8892 = stablehlo.multiply %v8887, %v8891 : tensor<768x384x2x2xf32>
    %v8893 = stablehlo.subtract %d2W, %v8892 : tensor<768x384x2x2xf32>
    %v8894 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8895 = stablehlo.multiply %v8894, %v8887 : tensor<768x384x2x2xf32>
    %v8896 = stablehlo.multiply %v8895, %d2W : tensor<768x384x2x2xf32>
    %v8897 = stablehlo.subtract %v8893, %v8896 : tensor<768x384x2x2xf32>
    %arsumd2b = "stablehlo.all_reduce"(%v1473) ({
    ^bb0(%arad2b: tensor<f32>, %arbd2b: tensor<f32>):
      %araddd2b = stablehlo.add %arad2b, %arbd2b : tensor<f32>
      stablehlo.return %araddd2b : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arnd2b = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeand2b = stablehlo.divide %arsumd2b, %arnd2b : tensor<768xf32>
    %v8898 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8899 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8900 = stablehlo.multiply %v8898, %d2bm : tensor<768xf32>
    %v8901 = stablehlo.multiply %v8899, %armeand2b : tensor<768xf32>
    %v8902 = stablehlo.add %v8900, %v8901 : tensor<768xf32>
    %v8903 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8904 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8905 = stablehlo.multiply %v8903, %d2bv : tensor<768xf32>
    %v8906 = stablehlo.multiply %armeand2b, %armeand2b : tensor<768xf32>
    %v8907 = stablehlo.multiply %v8904, %v8906 : tensor<768xf32>
    %v8908 = stablehlo.add %v8905, %v8907 : tensor<768xf32>
    %v8909 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8910 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8911 = stablehlo.multiply %v8909, %d2bm : tensor<768xf32>
    %v8912 = stablehlo.multiply %v8910, %armeand2b : tensor<768xf32>
    %v8913 = stablehlo.add %v8911, %v8912 : tensor<768xf32>
    %v8914 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8915 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8916 = stablehlo.multiply %v8914, %d2bv : tensor<768xf32>
    %v8917 = stablehlo.multiply %armeand2b, %armeand2b : tensor<768xf32>
    %v8918 = stablehlo.multiply %v8915, %v8917 : tensor<768xf32>
    %v8919 = stablehlo.add %v8916, %v8918 : tensor<768xf32>
    %v8920 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8921 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8922 = stablehlo.divide %v8913, %v8920 : tensor<768xf32>
    %v8923 = stablehlo.divide %v8919, %v8921 : tensor<768xf32>
    %v8924 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8925 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8926 = stablehlo.sqrt %v8923 : tensor<768xf32>
    %v8927 = stablehlo.add %v8926, %v8925 : tensor<768xf32>
    %v8928 = stablehlo.divide %v8922, %v8927 : tensor<768xf32>
    %v8929 = stablehlo.multiply %v8924, %v8928 : tensor<768xf32>
    %v8930 = stablehlo.subtract %d2b, %v8929 : tensor<768xf32>
    %v8931 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8932 = stablehlo.multiply %v8931, %v8924 : tensor<768xf32>
    %v8933 = stablehlo.multiply %v8932, %d2b : tensor<768xf32>
    %v8934 = stablehlo.subtract %v8930, %v8933 : tensor<768xf32>
    %arsums3b0dW = "stablehlo.all_reduce"(%v1433) ({
    ^bb0(%aras3b0dW: tensor<f32>, %arbs3b0dW: tensor<f32>):
      %aradds3b0dW = stablehlo.add %aras3b0dW, %arbs3b0dW : tensor<f32>
      stablehlo.return %aradds3b0dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768x1x7x7xf32>) -> tensor<768x1x7x7xf32>
    %arns3b0dW = stablehlo.constant dense<2.0> : tensor<768x1x7x7xf32>
    %armeans3b0dW = stablehlo.divide %arsums3b0dW, %arns3b0dW : tensor<768x1x7x7xf32>
    %v8935 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8936 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8937 = stablehlo.multiply %v8935, %s3b0dWm : tensor<768x1x7x7xf32>
    %v8938 = stablehlo.multiply %v8936, %armeans3b0dW : tensor<768x1x7x7xf32>
    %v8939 = stablehlo.add %v8937, %v8938 : tensor<768x1x7x7xf32>
    %v8940 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8941 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8942 = stablehlo.multiply %v8940, %s3b0dWv : tensor<768x1x7x7xf32>
    %v8943 = stablehlo.multiply %armeans3b0dW, %armeans3b0dW : tensor<768x1x7x7xf32>
    %v8944 = stablehlo.multiply %v8941, %v8943 : tensor<768x1x7x7xf32>
    %v8945 = stablehlo.add %v8942, %v8944 : tensor<768x1x7x7xf32>
    %v8946 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8947 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8948 = stablehlo.multiply %v8946, %s3b0dWm : tensor<768x1x7x7xf32>
    %v8949 = stablehlo.multiply %v8947, %armeans3b0dW : tensor<768x1x7x7xf32>
    %v8950 = stablehlo.add %v8948, %v8949 : tensor<768x1x7x7xf32>
    %v8951 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8952 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8953 = stablehlo.multiply %v8951, %s3b0dWv : tensor<768x1x7x7xf32>
    %v8954 = stablehlo.multiply %armeans3b0dW, %armeans3b0dW : tensor<768x1x7x7xf32>
    %v8955 = stablehlo.multiply %v8952, %v8954 : tensor<768x1x7x7xf32>
    %v8956 = stablehlo.add %v8953, %v8955 : tensor<768x1x7x7xf32>
    %v8957 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8958 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8959 = stablehlo.divide %v8950, %v8957 : tensor<768x1x7x7xf32>
    %v8960 = stablehlo.divide %v8956, %v8958 : tensor<768x1x7x7xf32>
    %v8961 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8962 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8963 = stablehlo.sqrt %v8960 : tensor<768x1x7x7xf32>
    %v8964 = stablehlo.add %v8963, %v8962 : tensor<768x1x7x7xf32>
    %v8965 = stablehlo.divide %v8959, %v8964 : tensor<768x1x7x7xf32>
    %v8966 = stablehlo.multiply %v8961, %v8965 : tensor<768x1x7x7xf32>
    %v8967 = stablehlo.subtract %s3b0dW, %v8966 : tensor<768x1x7x7xf32>
    %v8968 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8969 = stablehlo.multiply %v8968, %v8961 : tensor<768x1x7x7xf32>
    %v8970 = stablehlo.multiply %v8969, %s3b0dW : tensor<768x1x7x7xf32>
    %v8971 = stablehlo.subtract %v8967, %v8970 : tensor<768x1x7x7xf32>
    %arsums3b0db = "stablehlo.all_reduce"(%v1436) ({
    ^bb0(%aras3b0db: tensor<f32>, %arbs3b0db: tensor<f32>):
      %aradds3b0db = stablehlo.add %aras3b0db, %arbs3b0db : tensor<f32>
      stablehlo.return %aradds3b0db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arns3b0db = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeans3b0db = stablehlo.divide %arsums3b0db, %arns3b0db : tensor<768xf32>
    %v8972 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8973 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8974 = stablehlo.multiply %v8972, %s3b0dbm : tensor<768xf32>
    %v8975 = stablehlo.multiply %v8973, %armeans3b0db : tensor<768xf32>
    %v8976 = stablehlo.add %v8974, %v8975 : tensor<768xf32>
    %v8977 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8978 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8979 = stablehlo.multiply %v8977, %s3b0dbv : tensor<768xf32>
    %v8980 = stablehlo.multiply %armeans3b0db, %armeans3b0db : tensor<768xf32>
    %v8981 = stablehlo.multiply %v8978, %v8980 : tensor<768xf32>
    %v8982 = stablehlo.add %v8979, %v8981 : tensor<768xf32>
    %v8983 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8984 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8985 = stablehlo.multiply %v8983, %s3b0dbm : tensor<768xf32>
    %v8986 = stablehlo.multiply %v8984, %armeans3b0db : tensor<768xf32>
    %v8987 = stablehlo.add %v8985, %v8986 : tensor<768xf32>
    %v8988 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8989 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8990 = stablehlo.multiply %v8988, %s3b0dbv : tensor<768xf32>
    %v8991 = stablehlo.multiply %armeans3b0db, %armeans3b0db : tensor<768xf32>
    %v8992 = stablehlo.multiply %v8989, %v8991 : tensor<768xf32>
    %v8993 = stablehlo.add %v8990, %v8992 : tensor<768xf32>
    %v8994 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8995 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8996 = stablehlo.divide %v8987, %v8994 : tensor<768xf32>
    %v8997 = stablehlo.divide %v8993, %v8995 : tensor<768xf32>
    %v8998 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8999 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9000 = stablehlo.sqrt %v8997 : tensor<768xf32>
    %v9001 = stablehlo.add %v9000, %v8999 : tensor<768xf32>
    %v9002 = stablehlo.divide %v8996, %v9001 : tensor<768xf32>
    %v9003 = stablehlo.multiply %v8998, %v9002 : tensor<768xf32>
    %v9004 = stablehlo.subtract %s3b0db, %v9003 : tensor<768xf32>
    %v9005 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9006 = stablehlo.multiply %v9005, %v8998 : tensor<768xf32>
    %v9007 = stablehlo.multiply %v9006, %s3b0db : tensor<768xf32>
    %v9008 = stablehlo.subtract %v9004, %v9007 : tensor<768xf32>
    %arsums3b0ng = "stablehlo.all_reduce"(%v1425) ({
    ^bb0(%aras3b0ng: tensor<f32>, %arbs3b0ng: tensor<f32>):
      %aradds3b0ng = stablehlo.add %aras3b0ng, %arbs3b0ng : tensor<f32>
      stablehlo.return %aradds3b0ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns3b0ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans3b0ng = stablehlo.divide %arsums3b0ng, %arns3b0ng : tensor<f32>
    %v9009 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9010 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9011 = stablehlo.multiply %v9009, %s3b0ngm : tensor<f32>
    %v9012 = stablehlo.multiply %v9010, %armeans3b0ng : tensor<f32>
    %v9013 = stablehlo.add %v9011, %v9012 : tensor<f32>
    %v9014 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9015 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9016 = stablehlo.multiply %v9014, %s3b0ngv : tensor<f32>
    %v9017 = stablehlo.multiply %armeans3b0ng, %armeans3b0ng : tensor<f32>
    %v9018 = stablehlo.multiply %v9015, %v9017 : tensor<f32>
    %v9019 = stablehlo.add %v9016, %v9018 : tensor<f32>
    %v9020 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9021 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9022 = stablehlo.multiply %v9020, %s3b0ngm : tensor<f32>
    %v9023 = stablehlo.multiply %v9021, %armeans3b0ng : tensor<f32>
    %v9024 = stablehlo.add %v9022, %v9023 : tensor<f32>
    %v9025 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9026 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9027 = stablehlo.multiply %v9025, %s3b0ngv : tensor<f32>
    %v9028 = stablehlo.multiply %armeans3b0ng, %armeans3b0ng : tensor<f32>
    %v9029 = stablehlo.multiply %v9026, %v9028 : tensor<f32>
    %v9030 = stablehlo.add %v9027, %v9029 : tensor<f32>
    %v9031 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9032 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9033 = stablehlo.divide %v9024, %v9031 : tensor<f32>
    %v9034 = stablehlo.divide %v9030, %v9032 : tensor<f32>
    %v9035 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9036 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9037 = stablehlo.sqrt %v9034 : tensor<f32>
    %v9038 = stablehlo.add %v9037, %v9036 : tensor<f32>
    %v9039 = stablehlo.divide %v9033, %v9038 : tensor<f32>
    %v9040 = stablehlo.multiply %v9035, %v9039 : tensor<f32>
    %v9041 = stablehlo.subtract %s3b0ng, %v9040 : tensor<f32>
    %v9042 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9043 = stablehlo.multiply %v9042, %v9035 : tensor<f32>
    %v9044 = stablehlo.multiply %v9043, %s3b0ng : tensor<f32>
    %v9045 = stablehlo.subtract %v9041, %v9044 : tensor<f32>
    %arsums3b0nbt = "stablehlo.all_reduce"(%v1427) ({
    ^bb0(%aras3b0nbt: tensor<f32>, %arbs3b0nbt: tensor<f32>):
      %aradds3b0nbt = stablehlo.add %aras3b0nbt, %arbs3b0nbt : tensor<f32>
      stablehlo.return %aradds3b0nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns3b0nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans3b0nbt = stablehlo.divide %arsums3b0nbt, %arns3b0nbt : tensor<f32>
    %v9046 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9047 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9048 = stablehlo.multiply %v9046, %s3b0nbtm : tensor<f32>
    %v9049 = stablehlo.multiply %v9047, %armeans3b0nbt : tensor<f32>
    %v9050 = stablehlo.add %v9048, %v9049 : tensor<f32>
    %v9051 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9052 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9053 = stablehlo.multiply %v9051, %s3b0nbtv : tensor<f32>
    %v9054 = stablehlo.multiply %armeans3b0nbt, %armeans3b0nbt : tensor<f32>
    %v9055 = stablehlo.multiply %v9052, %v9054 : tensor<f32>
    %v9056 = stablehlo.add %v9053, %v9055 : tensor<f32>
    %v9057 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9058 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9059 = stablehlo.multiply %v9057, %s3b0nbtm : tensor<f32>
    %v9060 = stablehlo.multiply %v9058, %armeans3b0nbt : tensor<f32>
    %v9061 = stablehlo.add %v9059, %v9060 : tensor<f32>
    %v9062 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9063 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9064 = stablehlo.multiply %v9062, %s3b0nbtv : tensor<f32>
    %v9065 = stablehlo.multiply %armeans3b0nbt, %armeans3b0nbt : tensor<f32>
    %v9066 = stablehlo.multiply %v9063, %v9065 : tensor<f32>
    %v9067 = stablehlo.add %v9064, %v9066 : tensor<f32>
    %v9068 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9069 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9070 = stablehlo.divide %v9061, %v9068 : tensor<f32>
    %v9071 = stablehlo.divide %v9067, %v9069 : tensor<f32>
    %v9072 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9073 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9074 = stablehlo.sqrt %v9071 : tensor<f32>
    %v9075 = stablehlo.add %v9074, %v9073 : tensor<f32>
    %v9076 = stablehlo.divide %v9070, %v9075 : tensor<f32>
    %v9077 = stablehlo.multiply %v9072, %v9076 : tensor<f32>
    %v9078 = stablehlo.subtract %s3b0nbt, %v9077 : tensor<f32>
    %v9079 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9080 = stablehlo.multiply %v9079, %v9072 : tensor<f32>
    %v9081 = stablehlo.multiply %v9080, %s3b0nbt : tensor<f32>
    %v9082 = stablehlo.subtract %v9078, %v9081 : tensor<f32>
    %arsums3b0eW = "stablehlo.all_reduce"(%v1406) ({
    ^bb0(%aras3b0eW: tensor<f32>, %arbs3b0eW: tensor<f32>):
      %aradds3b0eW = stablehlo.add %aras3b0eW, %arbs3b0eW : tensor<f32>
      stablehlo.return %aradds3b0eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<3072x768x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %arns3b0eW = stablehlo.constant dense<2.0> : tensor<3072x768x1x1xf32>
    %armeans3b0eW = stablehlo.divide %arsums3b0eW, %arns3b0eW : tensor<3072x768x1x1xf32>
    %v9083 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9084 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9085 = stablehlo.multiply %v9083, %s3b0eWm : tensor<3072x768x1x1xf32>
    %v9086 = stablehlo.multiply %v9084, %armeans3b0eW : tensor<3072x768x1x1xf32>
    %v9087 = stablehlo.add %v9085, %v9086 : tensor<3072x768x1x1xf32>
    %v9088 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9089 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9090 = stablehlo.multiply %v9088, %s3b0eWv : tensor<3072x768x1x1xf32>
    %v9091 = stablehlo.multiply %armeans3b0eW, %armeans3b0eW : tensor<3072x768x1x1xf32>
    %v9092 = stablehlo.multiply %v9089, %v9091 : tensor<3072x768x1x1xf32>
    %v9093 = stablehlo.add %v9090, %v9092 : tensor<3072x768x1x1xf32>
    %v9094 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9095 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9096 = stablehlo.multiply %v9094, %s3b0eWm : tensor<3072x768x1x1xf32>
    %v9097 = stablehlo.multiply %v9095, %armeans3b0eW : tensor<3072x768x1x1xf32>
    %v9098 = stablehlo.add %v9096, %v9097 : tensor<3072x768x1x1xf32>
    %v9099 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9100 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9101 = stablehlo.multiply %v9099, %s3b0eWv : tensor<3072x768x1x1xf32>
    %v9102 = stablehlo.multiply %armeans3b0eW, %armeans3b0eW : tensor<3072x768x1x1xf32>
    %v9103 = stablehlo.multiply %v9100, %v9102 : tensor<3072x768x1x1xf32>
    %v9104 = stablehlo.add %v9101, %v9103 : tensor<3072x768x1x1xf32>
    %v9105 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9106 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9107 = stablehlo.divide %v9098, %v9105 : tensor<3072x768x1x1xf32>
    %v9108 = stablehlo.divide %v9104, %v9106 : tensor<3072x768x1x1xf32>
    %v9109 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9110 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9111 = stablehlo.sqrt %v9108 : tensor<3072x768x1x1xf32>
    %v9112 = stablehlo.add %v9111, %v9110 : tensor<3072x768x1x1xf32>
    %v9113 = stablehlo.divide %v9107, %v9112 : tensor<3072x768x1x1xf32>
    %v9114 = stablehlo.multiply %v9109, %v9113 : tensor<3072x768x1x1xf32>
    %v9115 = stablehlo.subtract %s3b0eW, %v9114 : tensor<3072x768x1x1xf32>
    %v9116 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9117 = stablehlo.multiply %v9116, %v9109 : tensor<3072x768x1x1xf32>
    %v9118 = stablehlo.multiply %v9117, %s3b0eW : tensor<3072x768x1x1xf32>
    %v9119 = stablehlo.subtract %v9115, %v9118 : tensor<3072x768x1x1xf32>
    %arsums3b0eb = "stablehlo.all_reduce"(%v1409) ({
    ^bb0(%aras3b0eb: tensor<f32>, %arbs3b0eb: tensor<f32>):
      %aradds3b0eb = stablehlo.add %aras3b0eb, %arbs3b0eb : tensor<f32>
      stablehlo.return %aradds3b0eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<3072xf32>) -> tensor<3072xf32>
    %arns3b0eb = stablehlo.constant dense<2.0> : tensor<3072xf32>
    %armeans3b0eb = stablehlo.divide %arsums3b0eb, %arns3b0eb : tensor<3072xf32>
    %v9120 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9121 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9122 = stablehlo.multiply %v9120, %s3b0ebm : tensor<3072xf32>
    %v9123 = stablehlo.multiply %v9121, %armeans3b0eb : tensor<3072xf32>
    %v9124 = stablehlo.add %v9122, %v9123 : tensor<3072xf32>
    %v9125 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9126 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9127 = stablehlo.multiply %v9125, %s3b0ebv : tensor<3072xf32>
    %v9128 = stablehlo.multiply %armeans3b0eb, %armeans3b0eb : tensor<3072xf32>
    %v9129 = stablehlo.multiply %v9126, %v9128 : tensor<3072xf32>
    %v9130 = stablehlo.add %v9127, %v9129 : tensor<3072xf32>
    %v9131 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9132 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9133 = stablehlo.multiply %v9131, %s3b0ebm : tensor<3072xf32>
    %v9134 = stablehlo.multiply %v9132, %armeans3b0eb : tensor<3072xf32>
    %v9135 = stablehlo.add %v9133, %v9134 : tensor<3072xf32>
    %v9136 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9137 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9138 = stablehlo.multiply %v9136, %s3b0ebv : tensor<3072xf32>
    %v9139 = stablehlo.multiply %armeans3b0eb, %armeans3b0eb : tensor<3072xf32>
    %v9140 = stablehlo.multiply %v9137, %v9139 : tensor<3072xf32>
    %v9141 = stablehlo.add %v9138, %v9140 : tensor<3072xf32>
    %v9142 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9143 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9144 = stablehlo.divide %v9135, %v9142 : tensor<3072xf32>
    %v9145 = stablehlo.divide %v9141, %v9143 : tensor<3072xf32>
    %v9146 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9147 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9148 = stablehlo.sqrt %v9145 : tensor<3072xf32>
    %v9149 = stablehlo.add %v9148, %v9147 : tensor<3072xf32>
    %v9150 = stablehlo.divide %v9144, %v9149 : tensor<3072xf32>
    %v9151 = stablehlo.multiply %v9146, %v9150 : tensor<3072xf32>
    %v9152 = stablehlo.subtract %s3b0eb, %v9151 : tensor<3072xf32>
    %v9153 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9154 = stablehlo.multiply %v9153, %v9146 : tensor<3072xf32>
    %v9155 = stablehlo.multiply %v9154, %s3b0eb : tensor<3072xf32>
    %v9156 = stablehlo.subtract %v9152, %v9155 : tensor<3072xf32>
    %arsums3b0pW = "stablehlo.all_reduce"(%v1397) ({
    ^bb0(%aras3b0pW: tensor<f32>, %arbs3b0pW: tensor<f32>):
      %aradds3b0pW = stablehlo.add %aras3b0pW, %arbs3b0pW : tensor<f32>
      stablehlo.return %aradds3b0pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768x3072x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %arns3b0pW = stablehlo.constant dense<2.0> : tensor<768x3072x1x1xf32>
    %armeans3b0pW = stablehlo.divide %arsums3b0pW, %arns3b0pW : tensor<768x3072x1x1xf32>
    %v9157 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9158 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9159 = stablehlo.multiply %v9157, %s3b0pWm : tensor<768x3072x1x1xf32>
    %v9160 = stablehlo.multiply %v9158, %armeans3b0pW : tensor<768x3072x1x1xf32>
    %v9161 = stablehlo.add %v9159, %v9160 : tensor<768x3072x1x1xf32>
    %v9162 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9163 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9164 = stablehlo.multiply %v9162, %s3b0pWv : tensor<768x3072x1x1xf32>
    %v9165 = stablehlo.multiply %armeans3b0pW, %armeans3b0pW : tensor<768x3072x1x1xf32>
    %v9166 = stablehlo.multiply %v9163, %v9165 : tensor<768x3072x1x1xf32>
    %v9167 = stablehlo.add %v9164, %v9166 : tensor<768x3072x1x1xf32>
    %v9168 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9169 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9170 = stablehlo.multiply %v9168, %s3b0pWm : tensor<768x3072x1x1xf32>
    %v9171 = stablehlo.multiply %v9169, %armeans3b0pW : tensor<768x3072x1x1xf32>
    %v9172 = stablehlo.add %v9170, %v9171 : tensor<768x3072x1x1xf32>
    %v9173 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9174 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9175 = stablehlo.multiply %v9173, %s3b0pWv : tensor<768x3072x1x1xf32>
    %v9176 = stablehlo.multiply %armeans3b0pW, %armeans3b0pW : tensor<768x3072x1x1xf32>
    %v9177 = stablehlo.multiply %v9174, %v9176 : tensor<768x3072x1x1xf32>
    %v9178 = stablehlo.add %v9175, %v9177 : tensor<768x3072x1x1xf32>
    %v9179 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9180 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9181 = stablehlo.divide %v9172, %v9179 : tensor<768x3072x1x1xf32>
    %v9182 = stablehlo.divide %v9178, %v9180 : tensor<768x3072x1x1xf32>
    %v9183 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9184 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9185 = stablehlo.sqrt %v9182 : tensor<768x3072x1x1xf32>
    %v9186 = stablehlo.add %v9185, %v9184 : tensor<768x3072x1x1xf32>
    %v9187 = stablehlo.divide %v9181, %v9186 : tensor<768x3072x1x1xf32>
    %v9188 = stablehlo.multiply %v9183, %v9187 : tensor<768x3072x1x1xf32>
    %v9189 = stablehlo.subtract %s3b0pW, %v9188 : tensor<768x3072x1x1xf32>
    %v9190 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9191 = stablehlo.multiply %v9190, %v9183 : tensor<768x3072x1x1xf32>
    %v9192 = stablehlo.multiply %v9191, %s3b0pW : tensor<768x3072x1x1xf32>
    %v9193 = stablehlo.subtract %v9189, %v9192 : tensor<768x3072x1x1xf32>
    %arsums3b0pb = "stablehlo.all_reduce"(%v1400) ({
    ^bb0(%aras3b0pb: tensor<f32>, %arbs3b0pb: tensor<f32>):
      %aradds3b0pb = stablehlo.add %aras3b0pb, %arbs3b0pb : tensor<f32>
      stablehlo.return %aradds3b0pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arns3b0pb = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeans3b0pb = stablehlo.divide %arsums3b0pb, %arns3b0pb : tensor<768xf32>
    %v9194 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9195 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9196 = stablehlo.multiply %v9194, %s3b0pbm : tensor<768xf32>
    %v9197 = stablehlo.multiply %v9195, %armeans3b0pb : tensor<768xf32>
    %v9198 = stablehlo.add %v9196, %v9197 : tensor<768xf32>
    %v9199 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9200 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9201 = stablehlo.multiply %v9199, %s3b0pbv : tensor<768xf32>
    %v9202 = stablehlo.multiply %armeans3b0pb, %armeans3b0pb : tensor<768xf32>
    %v9203 = stablehlo.multiply %v9200, %v9202 : tensor<768xf32>
    %v9204 = stablehlo.add %v9201, %v9203 : tensor<768xf32>
    %v9205 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9206 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9207 = stablehlo.multiply %v9205, %s3b0pbm : tensor<768xf32>
    %v9208 = stablehlo.multiply %v9206, %armeans3b0pb : tensor<768xf32>
    %v9209 = stablehlo.add %v9207, %v9208 : tensor<768xf32>
    %v9210 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9211 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9212 = stablehlo.multiply %v9210, %s3b0pbv : tensor<768xf32>
    %v9213 = stablehlo.multiply %armeans3b0pb, %armeans3b0pb : tensor<768xf32>
    %v9214 = stablehlo.multiply %v9211, %v9213 : tensor<768xf32>
    %v9215 = stablehlo.add %v9212, %v9214 : tensor<768xf32>
    %v9216 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9217 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9218 = stablehlo.divide %v9209, %v9216 : tensor<768xf32>
    %v9219 = stablehlo.divide %v9215, %v9217 : tensor<768xf32>
    %v9220 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9221 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9222 = stablehlo.sqrt %v9219 : tensor<768xf32>
    %v9223 = stablehlo.add %v9222, %v9221 : tensor<768xf32>
    %v9224 = stablehlo.divide %v9218, %v9223 : tensor<768xf32>
    %v9225 = stablehlo.multiply %v9220, %v9224 : tensor<768xf32>
    %v9226 = stablehlo.subtract %s3b0pb, %v9225 : tensor<768xf32>
    %v9227 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9228 = stablehlo.multiply %v9227, %v9220 : tensor<768xf32>
    %v9229 = stablehlo.multiply %v9228, %s3b0pb : tensor<768xf32>
    %v9230 = stablehlo.subtract %v9226, %v9229 : tensor<768xf32>
    %arsums3b0lg = "stablehlo.all_reduce"(%v1391) ({
    ^bb0(%aras3b0lg: tensor<f32>, %arbs3b0lg: tensor<f32>):
      %aradds3b0lg = stablehlo.add %aras3b0lg, %arbs3b0lg : tensor<f32>
      stablehlo.return %aradds3b0lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arns3b0lg = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeans3b0lg = stablehlo.divide %arsums3b0lg, %arns3b0lg : tensor<768xf32>
    %v9231 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9232 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9233 = stablehlo.multiply %v9231, %s3b0lgm : tensor<768xf32>
    %v9234 = stablehlo.multiply %v9232, %armeans3b0lg : tensor<768xf32>
    %v9235 = stablehlo.add %v9233, %v9234 : tensor<768xf32>
    %v9236 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9237 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9238 = stablehlo.multiply %v9236, %s3b0lgv : tensor<768xf32>
    %v9239 = stablehlo.multiply %armeans3b0lg, %armeans3b0lg : tensor<768xf32>
    %v9240 = stablehlo.multiply %v9237, %v9239 : tensor<768xf32>
    %v9241 = stablehlo.add %v9238, %v9240 : tensor<768xf32>
    %v9242 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9243 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9244 = stablehlo.multiply %v9242, %s3b0lgm : tensor<768xf32>
    %v9245 = stablehlo.multiply %v9243, %armeans3b0lg : tensor<768xf32>
    %v9246 = stablehlo.add %v9244, %v9245 : tensor<768xf32>
    %v9247 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9248 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9249 = stablehlo.multiply %v9247, %s3b0lgv : tensor<768xf32>
    %v9250 = stablehlo.multiply %armeans3b0lg, %armeans3b0lg : tensor<768xf32>
    %v9251 = stablehlo.multiply %v9248, %v9250 : tensor<768xf32>
    %v9252 = stablehlo.add %v9249, %v9251 : tensor<768xf32>
    %v9253 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9254 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9255 = stablehlo.divide %v9246, %v9253 : tensor<768xf32>
    %v9256 = stablehlo.divide %v9252, %v9254 : tensor<768xf32>
    %v9257 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9258 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9259 = stablehlo.sqrt %v9256 : tensor<768xf32>
    %v9260 = stablehlo.add %v9259, %v9258 : tensor<768xf32>
    %v9261 = stablehlo.divide %v9255, %v9260 : tensor<768xf32>
    %v9262 = stablehlo.multiply %v9257, %v9261 : tensor<768xf32>
    %v9263 = stablehlo.subtract %s3b0lg, %v9262 : tensor<768xf32>
    %v9264 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9265 = stablehlo.multiply %v9264, %v9257 : tensor<768xf32>
    %v9266 = stablehlo.multiply %v9265, %s3b0lg : tensor<768xf32>
    %v9267 = stablehlo.subtract %v9263, %v9266 : tensor<768xf32>
    %arsums3b1dW = "stablehlo.all_reduce"(%v1314) ({
    ^bb0(%aras3b1dW: tensor<f32>, %arbs3b1dW: tensor<f32>):
      %aradds3b1dW = stablehlo.add %aras3b1dW, %arbs3b1dW : tensor<f32>
      stablehlo.return %aradds3b1dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768x1x7x7xf32>) -> tensor<768x1x7x7xf32>
    %arns3b1dW = stablehlo.constant dense<2.0> : tensor<768x1x7x7xf32>
    %armeans3b1dW = stablehlo.divide %arsums3b1dW, %arns3b1dW : tensor<768x1x7x7xf32>
    %v9268 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9269 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9270 = stablehlo.multiply %v9268, %s3b1dWm : tensor<768x1x7x7xf32>
    %v9271 = stablehlo.multiply %v9269, %armeans3b1dW : tensor<768x1x7x7xf32>
    %v9272 = stablehlo.add %v9270, %v9271 : tensor<768x1x7x7xf32>
    %v9273 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9274 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9275 = stablehlo.multiply %v9273, %s3b1dWv : tensor<768x1x7x7xf32>
    %v9276 = stablehlo.multiply %armeans3b1dW, %armeans3b1dW : tensor<768x1x7x7xf32>
    %v9277 = stablehlo.multiply %v9274, %v9276 : tensor<768x1x7x7xf32>
    %v9278 = stablehlo.add %v9275, %v9277 : tensor<768x1x7x7xf32>
    %v9279 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9280 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9281 = stablehlo.multiply %v9279, %s3b1dWm : tensor<768x1x7x7xf32>
    %v9282 = stablehlo.multiply %v9280, %armeans3b1dW : tensor<768x1x7x7xf32>
    %v9283 = stablehlo.add %v9281, %v9282 : tensor<768x1x7x7xf32>
    %v9284 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9285 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9286 = stablehlo.multiply %v9284, %s3b1dWv : tensor<768x1x7x7xf32>
    %v9287 = stablehlo.multiply %armeans3b1dW, %armeans3b1dW : tensor<768x1x7x7xf32>
    %v9288 = stablehlo.multiply %v9285, %v9287 : tensor<768x1x7x7xf32>
    %v9289 = stablehlo.add %v9286, %v9288 : tensor<768x1x7x7xf32>
    %v9290 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9291 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9292 = stablehlo.divide %v9283, %v9290 : tensor<768x1x7x7xf32>
    %v9293 = stablehlo.divide %v9289, %v9291 : tensor<768x1x7x7xf32>
    %v9294 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9295 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9296 = stablehlo.sqrt %v9293 : tensor<768x1x7x7xf32>
    %v9297 = stablehlo.add %v9296, %v9295 : tensor<768x1x7x7xf32>
    %v9298 = stablehlo.divide %v9292, %v9297 : tensor<768x1x7x7xf32>
    %v9299 = stablehlo.multiply %v9294, %v9298 : tensor<768x1x7x7xf32>
    %v9300 = stablehlo.subtract %s3b1dW, %v9299 : tensor<768x1x7x7xf32>
    %v9301 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9302 = stablehlo.multiply %v9301, %v9294 : tensor<768x1x7x7xf32>
    %v9303 = stablehlo.multiply %v9302, %s3b1dW : tensor<768x1x7x7xf32>
    %v9304 = stablehlo.subtract %v9300, %v9303 : tensor<768x1x7x7xf32>
    %arsums3b1db = "stablehlo.all_reduce"(%v1317) ({
    ^bb0(%aras3b1db: tensor<f32>, %arbs3b1db: tensor<f32>):
      %aradds3b1db = stablehlo.add %aras3b1db, %arbs3b1db : tensor<f32>
      stablehlo.return %aradds3b1db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arns3b1db = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeans3b1db = stablehlo.divide %arsums3b1db, %arns3b1db : tensor<768xf32>
    %v9305 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9306 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9307 = stablehlo.multiply %v9305, %s3b1dbm : tensor<768xf32>
    %v9308 = stablehlo.multiply %v9306, %armeans3b1db : tensor<768xf32>
    %v9309 = stablehlo.add %v9307, %v9308 : tensor<768xf32>
    %v9310 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9311 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9312 = stablehlo.multiply %v9310, %s3b1dbv : tensor<768xf32>
    %v9313 = stablehlo.multiply %armeans3b1db, %armeans3b1db : tensor<768xf32>
    %v9314 = stablehlo.multiply %v9311, %v9313 : tensor<768xf32>
    %v9315 = stablehlo.add %v9312, %v9314 : tensor<768xf32>
    %v9316 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9317 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9318 = stablehlo.multiply %v9316, %s3b1dbm : tensor<768xf32>
    %v9319 = stablehlo.multiply %v9317, %armeans3b1db : tensor<768xf32>
    %v9320 = stablehlo.add %v9318, %v9319 : tensor<768xf32>
    %v9321 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9322 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9323 = stablehlo.multiply %v9321, %s3b1dbv : tensor<768xf32>
    %v9324 = stablehlo.multiply %armeans3b1db, %armeans3b1db : tensor<768xf32>
    %v9325 = stablehlo.multiply %v9322, %v9324 : tensor<768xf32>
    %v9326 = stablehlo.add %v9323, %v9325 : tensor<768xf32>
    %v9327 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9328 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9329 = stablehlo.divide %v9320, %v9327 : tensor<768xf32>
    %v9330 = stablehlo.divide %v9326, %v9328 : tensor<768xf32>
    %v9331 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9332 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9333 = stablehlo.sqrt %v9330 : tensor<768xf32>
    %v9334 = stablehlo.add %v9333, %v9332 : tensor<768xf32>
    %v9335 = stablehlo.divide %v9329, %v9334 : tensor<768xf32>
    %v9336 = stablehlo.multiply %v9331, %v9335 : tensor<768xf32>
    %v9337 = stablehlo.subtract %s3b1db, %v9336 : tensor<768xf32>
    %v9338 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9339 = stablehlo.multiply %v9338, %v9331 : tensor<768xf32>
    %v9340 = stablehlo.multiply %v9339, %s3b1db : tensor<768xf32>
    %v9341 = stablehlo.subtract %v9337, %v9340 : tensor<768xf32>
    %arsums3b1ng = "stablehlo.all_reduce"(%v1306) ({
    ^bb0(%aras3b1ng: tensor<f32>, %arbs3b1ng: tensor<f32>):
      %aradds3b1ng = stablehlo.add %aras3b1ng, %arbs3b1ng : tensor<f32>
      stablehlo.return %aradds3b1ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns3b1ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans3b1ng = stablehlo.divide %arsums3b1ng, %arns3b1ng : tensor<f32>
    %v9342 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9343 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9344 = stablehlo.multiply %v9342, %s3b1ngm : tensor<f32>
    %v9345 = stablehlo.multiply %v9343, %armeans3b1ng : tensor<f32>
    %v9346 = stablehlo.add %v9344, %v9345 : tensor<f32>
    %v9347 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9348 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9349 = stablehlo.multiply %v9347, %s3b1ngv : tensor<f32>
    %v9350 = stablehlo.multiply %armeans3b1ng, %armeans3b1ng : tensor<f32>
    %v9351 = stablehlo.multiply %v9348, %v9350 : tensor<f32>
    %v9352 = stablehlo.add %v9349, %v9351 : tensor<f32>
    %v9353 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9354 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9355 = stablehlo.multiply %v9353, %s3b1ngm : tensor<f32>
    %v9356 = stablehlo.multiply %v9354, %armeans3b1ng : tensor<f32>
    %v9357 = stablehlo.add %v9355, %v9356 : tensor<f32>
    %v9358 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9359 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9360 = stablehlo.multiply %v9358, %s3b1ngv : tensor<f32>
    %v9361 = stablehlo.multiply %armeans3b1ng, %armeans3b1ng : tensor<f32>
    %v9362 = stablehlo.multiply %v9359, %v9361 : tensor<f32>
    %v9363 = stablehlo.add %v9360, %v9362 : tensor<f32>
    %v9364 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9365 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9366 = stablehlo.divide %v9357, %v9364 : tensor<f32>
    %v9367 = stablehlo.divide %v9363, %v9365 : tensor<f32>
    %v9368 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9369 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9370 = stablehlo.sqrt %v9367 : tensor<f32>
    %v9371 = stablehlo.add %v9370, %v9369 : tensor<f32>
    %v9372 = stablehlo.divide %v9366, %v9371 : tensor<f32>
    %v9373 = stablehlo.multiply %v9368, %v9372 : tensor<f32>
    %v9374 = stablehlo.subtract %s3b1ng, %v9373 : tensor<f32>
    %v9375 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9376 = stablehlo.multiply %v9375, %v9368 : tensor<f32>
    %v9377 = stablehlo.multiply %v9376, %s3b1ng : tensor<f32>
    %v9378 = stablehlo.subtract %v9374, %v9377 : tensor<f32>
    %arsums3b1nbt = "stablehlo.all_reduce"(%v1308) ({
    ^bb0(%aras3b1nbt: tensor<f32>, %arbs3b1nbt: tensor<f32>):
      %aradds3b1nbt = stablehlo.add %aras3b1nbt, %arbs3b1nbt : tensor<f32>
      stablehlo.return %aradds3b1nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns3b1nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans3b1nbt = stablehlo.divide %arsums3b1nbt, %arns3b1nbt : tensor<f32>
    %v9379 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9380 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9381 = stablehlo.multiply %v9379, %s3b1nbtm : tensor<f32>
    %v9382 = stablehlo.multiply %v9380, %armeans3b1nbt : tensor<f32>
    %v9383 = stablehlo.add %v9381, %v9382 : tensor<f32>
    %v9384 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9385 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9386 = stablehlo.multiply %v9384, %s3b1nbtv : tensor<f32>
    %v9387 = stablehlo.multiply %armeans3b1nbt, %armeans3b1nbt : tensor<f32>
    %v9388 = stablehlo.multiply %v9385, %v9387 : tensor<f32>
    %v9389 = stablehlo.add %v9386, %v9388 : tensor<f32>
    %v9390 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9391 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9392 = stablehlo.multiply %v9390, %s3b1nbtm : tensor<f32>
    %v9393 = stablehlo.multiply %v9391, %armeans3b1nbt : tensor<f32>
    %v9394 = stablehlo.add %v9392, %v9393 : tensor<f32>
    %v9395 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9396 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9397 = stablehlo.multiply %v9395, %s3b1nbtv : tensor<f32>
    %v9398 = stablehlo.multiply %armeans3b1nbt, %armeans3b1nbt : tensor<f32>
    %v9399 = stablehlo.multiply %v9396, %v9398 : tensor<f32>
    %v9400 = stablehlo.add %v9397, %v9399 : tensor<f32>
    %v9401 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9402 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9403 = stablehlo.divide %v9394, %v9401 : tensor<f32>
    %v9404 = stablehlo.divide %v9400, %v9402 : tensor<f32>
    %v9405 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9406 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9407 = stablehlo.sqrt %v9404 : tensor<f32>
    %v9408 = stablehlo.add %v9407, %v9406 : tensor<f32>
    %v9409 = stablehlo.divide %v9403, %v9408 : tensor<f32>
    %v9410 = stablehlo.multiply %v9405, %v9409 : tensor<f32>
    %v9411 = stablehlo.subtract %s3b1nbt, %v9410 : tensor<f32>
    %v9412 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9413 = stablehlo.multiply %v9412, %v9405 : tensor<f32>
    %v9414 = stablehlo.multiply %v9413, %s3b1nbt : tensor<f32>
    %v9415 = stablehlo.subtract %v9411, %v9414 : tensor<f32>
    %arsums3b1eW = "stablehlo.all_reduce"(%v1287) ({
    ^bb0(%aras3b1eW: tensor<f32>, %arbs3b1eW: tensor<f32>):
      %aradds3b1eW = stablehlo.add %aras3b1eW, %arbs3b1eW : tensor<f32>
      stablehlo.return %aradds3b1eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<3072x768x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %arns3b1eW = stablehlo.constant dense<2.0> : tensor<3072x768x1x1xf32>
    %armeans3b1eW = stablehlo.divide %arsums3b1eW, %arns3b1eW : tensor<3072x768x1x1xf32>
    %v9416 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9417 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9418 = stablehlo.multiply %v9416, %s3b1eWm : tensor<3072x768x1x1xf32>
    %v9419 = stablehlo.multiply %v9417, %armeans3b1eW : tensor<3072x768x1x1xf32>
    %v9420 = stablehlo.add %v9418, %v9419 : tensor<3072x768x1x1xf32>
    %v9421 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9422 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9423 = stablehlo.multiply %v9421, %s3b1eWv : tensor<3072x768x1x1xf32>
    %v9424 = stablehlo.multiply %armeans3b1eW, %armeans3b1eW : tensor<3072x768x1x1xf32>
    %v9425 = stablehlo.multiply %v9422, %v9424 : tensor<3072x768x1x1xf32>
    %v9426 = stablehlo.add %v9423, %v9425 : tensor<3072x768x1x1xf32>
    %v9427 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9428 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9429 = stablehlo.multiply %v9427, %s3b1eWm : tensor<3072x768x1x1xf32>
    %v9430 = stablehlo.multiply %v9428, %armeans3b1eW : tensor<3072x768x1x1xf32>
    %v9431 = stablehlo.add %v9429, %v9430 : tensor<3072x768x1x1xf32>
    %v9432 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9433 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9434 = stablehlo.multiply %v9432, %s3b1eWv : tensor<3072x768x1x1xf32>
    %v9435 = stablehlo.multiply %armeans3b1eW, %armeans3b1eW : tensor<3072x768x1x1xf32>
    %v9436 = stablehlo.multiply %v9433, %v9435 : tensor<3072x768x1x1xf32>
    %v9437 = stablehlo.add %v9434, %v9436 : tensor<3072x768x1x1xf32>
    %v9438 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9439 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9440 = stablehlo.divide %v9431, %v9438 : tensor<3072x768x1x1xf32>
    %v9441 = stablehlo.divide %v9437, %v9439 : tensor<3072x768x1x1xf32>
    %v9442 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9443 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9444 = stablehlo.sqrt %v9441 : tensor<3072x768x1x1xf32>
    %v9445 = stablehlo.add %v9444, %v9443 : tensor<3072x768x1x1xf32>
    %v9446 = stablehlo.divide %v9440, %v9445 : tensor<3072x768x1x1xf32>
    %v9447 = stablehlo.multiply %v9442, %v9446 : tensor<3072x768x1x1xf32>
    %v9448 = stablehlo.subtract %s3b1eW, %v9447 : tensor<3072x768x1x1xf32>
    %v9449 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9450 = stablehlo.multiply %v9449, %v9442 : tensor<3072x768x1x1xf32>
    %v9451 = stablehlo.multiply %v9450, %s3b1eW : tensor<3072x768x1x1xf32>
    %v9452 = stablehlo.subtract %v9448, %v9451 : tensor<3072x768x1x1xf32>
    %arsums3b1eb = "stablehlo.all_reduce"(%v1290) ({
    ^bb0(%aras3b1eb: tensor<f32>, %arbs3b1eb: tensor<f32>):
      %aradds3b1eb = stablehlo.add %aras3b1eb, %arbs3b1eb : tensor<f32>
      stablehlo.return %aradds3b1eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<3072xf32>) -> tensor<3072xf32>
    %arns3b1eb = stablehlo.constant dense<2.0> : tensor<3072xf32>
    %armeans3b1eb = stablehlo.divide %arsums3b1eb, %arns3b1eb : tensor<3072xf32>
    %v9453 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9454 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9455 = stablehlo.multiply %v9453, %s3b1ebm : tensor<3072xf32>
    %v9456 = stablehlo.multiply %v9454, %armeans3b1eb : tensor<3072xf32>
    %v9457 = stablehlo.add %v9455, %v9456 : tensor<3072xf32>
    %v9458 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9459 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9460 = stablehlo.multiply %v9458, %s3b1ebv : tensor<3072xf32>
    %v9461 = stablehlo.multiply %armeans3b1eb, %armeans3b1eb : tensor<3072xf32>
    %v9462 = stablehlo.multiply %v9459, %v9461 : tensor<3072xf32>
    %v9463 = stablehlo.add %v9460, %v9462 : tensor<3072xf32>
    %v9464 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9465 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9466 = stablehlo.multiply %v9464, %s3b1ebm : tensor<3072xf32>
    %v9467 = stablehlo.multiply %v9465, %armeans3b1eb : tensor<3072xf32>
    %v9468 = stablehlo.add %v9466, %v9467 : tensor<3072xf32>
    %v9469 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9470 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9471 = stablehlo.multiply %v9469, %s3b1ebv : tensor<3072xf32>
    %v9472 = stablehlo.multiply %armeans3b1eb, %armeans3b1eb : tensor<3072xf32>
    %v9473 = stablehlo.multiply %v9470, %v9472 : tensor<3072xf32>
    %v9474 = stablehlo.add %v9471, %v9473 : tensor<3072xf32>
    %v9475 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9476 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9477 = stablehlo.divide %v9468, %v9475 : tensor<3072xf32>
    %v9478 = stablehlo.divide %v9474, %v9476 : tensor<3072xf32>
    %v9479 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9480 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9481 = stablehlo.sqrt %v9478 : tensor<3072xf32>
    %v9482 = stablehlo.add %v9481, %v9480 : tensor<3072xf32>
    %v9483 = stablehlo.divide %v9477, %v9482 : tensor<3072xf32>
    %v9484 = stablehlo.multiply %v9479, %v9483 : tensor<3072xf32>
    %v9485 = stablehlo.subtract %s3b1eb, %v9484 : tensor<3072xf32>
    %v9486 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9487 = stablehlo.multiply %v9486, %v9479 : tensor<3072xf32>
    %v9488 = stablehlo.multiply %v9487, %s3b1eb : tensor<3072xf32>
    %v9489 = stablehlo.subtract %v9485, %v9488 : tensor<3072xf32>
    %arsums3b1pW = "stablehlo.all_reduce"(%v1278) ({
    ^bb0(%aras3b1pW: tensor<f32>, %arbs3b1pW: tensor<f32>):
      %aradds3b1pW = stablehlo.add %aras3b1pW, %arbs3b1pW : tensor<f32>
      stablehlo.return %aradds3b1pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768x3072x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %arns3b1pW = stablehlo.constant dense<2.0> : tensor<768x3072x1x1xf32>
    %armeans3b1pW = stablehlo.divide %arsums3b1pW, %arns3b1pW : tensor<768x3072x1x1xf32>
    %v9490 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9491 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9492 = stablehlo.multiply %v9490, %s3b1pWm : tensor<768x3072x1x1xf32>
    %v9493 = stablehlo.multiply %v9491, %armeans3b1pW : tensor<768x3072x1x1xf32>
    %v9494 = stablehlo.add %v9492, %v9493 : tensor<768x3072x1x1xf32>
    %v9495 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9496 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9497 = stablehlo.multiply %v9495, %s3b1pWv : tensor<768x3072x1x1xf32>
    %v9498 = stablehlo.multiply %armeans3b1pW, %armeans3b1pW : tensor<768x3072x1x1xf32>
    %v9499 = stablehlo.multiply %v9496, %v9498 : tensor<768x3072x1x1xf32>
    %v9500 = stablehlo.add %v9497, %v9499 : tensor<768x3072x1x1xf32>
    %v9501 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9502 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9503 = stablehlo.multiply %v9501, %s3b1pWm : tensor<768x3072x1x1xf32>
    %v9504 = stablehlo.multiply %v9502, %armeans3b1pW : tensor<768x3072x1x1xf32>
    %v9505 = stablehlo.add %v9503, %v9504 : tensor<768x3072x1x1xf32>
    %v9506 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9507 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9508 = stablehlo.multiply %v9506, %s3b1pWv : tensor<768x3072x1x1xf32>
    %v9509 = stablehlo.multiply %armeans3b1pW, %armeans3b1pW : tensor<768x3072x1x1xf32>
    %v9510 = stablehlo.multiply %v9507, %v9509 : tensor<768x3072x1x1xf32>
    %v9511 = stablehlo.add %v9508, %v9510 : tensor<768x3072x1x1xf32>
    %v9512 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9513 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9514 = stablehlo.divide %v9505, %v9512 : tensor<768x3072x1x1xf32>
    %v9515 = stablehlo.divide %v9511, %v9513 : tensor<768x3072x1x1xf32>
    %v9516 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9517 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9518 = stablehlo.sqrt %v9515 : tensor<768x3072x1x1xf32>
    %v9519 = stablehlo.add %v9518, %v9517 : tensor<768x3072x1x1xf32>
    %v9520 = stablehlo.divide %v9514, %v9519 : tensor<768x3072x1x1xf32>
    %v9521 = stablehlo.multiply %v9516, %v9520 : tensor<768x3072x1x1xf32>
    %v9522 = stablehlo.subtract %s3b1pW, %v9521 : tensor<768x3072x1x1xf32>
    %v9523 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9524 = stablehlo.multiply %v9523, %v9516 : tensor<768x3072x1x1xf32>
    %v9525 = stablehlo.multiply %v9524, %s3b1pW : tensor<768x3072x1x1xf32>
    %v9526 = stablehlo.subtract %v9522, %v9525 : tensor<768x3072x1x1xf32>
    %arsums3b1pb = "stablehlo.all_reduce"(%v1281) ({
    ^bb0(%aras3b1pb: tensor<f32>, %arbs3b1pb: tensor<f32>):
      %aradds3b1pb = stablehlo.add %aras3b1pb, %arbs3b1pb : tensor<f32>
      stablehlo.return %aradds3b1pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arns3b1pb = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeans3b1pb = stablehlo.divide %arsums3b1pb, %arns3b1pb : tensor<768xf32>
    %v9527 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9528 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9529 = stablehlo.multiply %v9527, %s3b1pbm : tensor<768xf32>
    %v9530 = stablehlo.multiply %v9528, %armeans3b1pb : tensor<768xf32>
    %v9531 = stablehlo.add %v9529, %v9530 : tensor<768xf32>
    %v9532 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9533 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9534 = stablehlo.multiply %v9532, %s3b1pbv : tensor<768xf32>
    %v9535 = stablehlo.multiply %armeans3b1pb, %armeans3b1pb : tensor<768xf32>
    %v9536 = stablehlo.multiply %v9533, %v9535 : tensor<768xf32>
    %v9537 = stablehlo.add %v9534, %v9536 : tensor<768xf32>
    %v9538 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9539 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9540 = stablehlo.multiply %v9538, %s3b1pbm : tensor<768xf32>
    %v9541 = stablehlo.multiply %v9539, %armeans3b1pb : tensor<768xf32>
    %v9542 = stablehlo.add %v9540, %v9541 : tensor<768xf32>
    %v9543 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9544 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9545 = stablehlo.multiply %v9543, %s3b1pbv : tensor<768xf32>
    %v9546 = stablehlo.multiply %armeans3b1pb, %armeans3b1pb : tensor<768xf32>
    %v9547 = stablehlo.multiply %v9544, %v9546 : tensor<768xf32>
    %v9548 = stablehlo.add %v9545, %v9547 : tensor<768xf32>
    %v9549 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9550 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9551 = stablehlo.divide %v9542, %v9549 : tensor<768xf32>
    %v9552 = stablehlo.divide %v9548, %v9550 : tensor<768xf32>
    %v9553 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9554 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9555 = stablehlo.sqrt %v9552 : tensor<768xf32>
    %v9556 = stablehlo.add %v9555, %v9554 : tensor<768xf32>
    %v9557 = stablehlo.divide %v9551, %v9556 : tensor<768xf32>
    %v9558 = stablehlo.multiply %v9553, %v9557 : tensor<768xf32>
    %v9559 = stablehlo.subtract %s3b1pb, %v9558 : tensor<768xf32>
    %v9560 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9561 = stablehlo.multiply %v9560, %v9553 : tensor<768xf32>
    %v9562 = stablehlo.multiply %v9561, %s3b1pb : tensor<768xf32>
    %v9563 = stablehlo.subtract %v9559, %v9562 : tensor<768xf32>
    %arsums3b1lg = "stablehlo.all_reduce"(%v1272) ({
    ^bb0(%aras3b1lg: tensor<f32>, %arbs3b1lg: tensor<f32>):
      %aradds3b1lg = stablehlo.add %aras3b1lg, %arbs3b1lg : tensor<f32>
      stablehlo.return %aradds3b1lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arns3b1lg = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeans3b1lg = stablehlo.divide %arsums3b1lg, %arns3b1lg : tensor<768xf32>
    %v9564 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9565 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9566 = stablehlo.multiply %v9564, %s3b1lgm : tensor<768xf32>
    %v9567 = stablehlo.multiply %v9565, %armeans3b1lg : tensor<768xf32>
    %v9568 = stablehlo.add %v9566, %v9567 : tensor<768xf32>
    %v9569 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9570 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9571 = stablehlo.multiply %v9569, %s3b1lgv : tensor<768xf32>
    %v9572 = stablehlo.multiply %armeans3b1lg, %armeans3b1lg : tensor<768xf32>
    %v9573 = stablehlo.multiply %v9570, %v9572 : tensor<768xf32>
    %v9574 = stablehlo.add %v9571, %v9573 : tensor<768xf32>
    %v9575 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9576 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9577 = stablehlo.multiply %v9575, %s3b1lgm : tensor<768xf32>
    %v9578 = stablehlo.multiply %v9576, %armeans3b1lg : tensor<768xf32>
    %v9579 = stablehlo.add %v9577, %v9578 : tensor<768xf32>
    %v9580 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9581 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9582 = stablehlo.multiply %v9580, %s3b1lgv : tensor<768xf32>
    %v9583 = stablehlo.multiply %armeans3b1lg, %armeans3b1lg : tensor<768xf32>
    %v9584 = stablehlo.multiply %v9581, %v9583 : tensor<768xf32>
    %v9585 = stablehlo.add %v9582, %v9584 : tensor<768xf32>
    %v9586 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9587 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9588 = stablehlo.divide %v9579, %v9586 : tensor<768xf32>
    %v9589 = stablehlo.divide %v9585, %v9587 : tensor<768xf32>
    %v9590 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9591 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9592 = stablehlo.sqrt %v9589 : tensor<768xf32>
    %v9593 = stablehlo.add %v9592, %v9591 : tensor<768xf32>
    %v9594 = stablehlo.divide %v9588, %v9593 : tensor<768xf32>
    %v9595 = stablehlo.multiply %v9590, %v9594 : tensor<768xf32>
    %v9596 = stablehlo.subtract %s3b1lg, %v9595 : tensor<768xf32>
    %v9597 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9598 = stablehlo.multiply %v9597, %v9590 : tensor<768xf32>
    %v9599 = stablehlo.multiply %v9598, %s3b1lg : tensor<768xf32>
    %v9600 = stablehlo.subtract %v9596, %v9599 : tensor<768xf32>
    %arsums3b2dW = "stablehlo.all_reduce"(%v1195) ({
    ^bb0(%aras3b2dW: tensor<f32>, %arbs3b2dW: tensor<f32>):
      %aradds3b2dW = stablehlo.add %aras3b2dW, %arbs3b2dW : tensor<f32>
      stablehlo.return %aradds3b2dW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768x1x7x7xf32>) -> tensor<768x1x7x7xf32>
    %arns3b2dW = stablehlo.constant dense<2.0> : tensor<768x1x7x7xf32>
    %armeans3b2dW = stablehlo.divide %arsums3b2dW, %arns3b2dW : tensor<768x1x7x7xf32>
    %v9601 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9602 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9603 = stablehlo.multiply %v9601, %s3b2dWm : tensor<768x1x7x7xf32>
    %v9604 = stablehlo.multiply %v9602, %armeans3b2dW : tensor<768x1x7x7xf32>
    %v9605 = stablehlo.add %v9603, %v9604 : tensor<768x1x7x7xf32>
    %v9606 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9607 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9608 = stablehlo.multiply %v9606, %s3b2dWv : tensor<768x1x7x7xf32>
    %v9609 = stablehlo.multiply %armeans3b2dW, %armeans3b2dW : tensor<768x1x7x7xf32>
    %v9610 = stablehlo.multiply %v9607, %v9609 : tensor<768x1x7x7xf32>
    %v9611 = stablehlo.add %v9608, %v9610 : tensor<768x1x7x7xf32>
    %v9612 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9613 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9614 = stablehlo.multiply %v9612, %s3b2dWm : tensor<768x1x7x7xf32>
    %v9615 = stablehlo.multiply %v9613, %armeans3b2dW : tensor<768x1x7x7xf32>
    %v9616 = stablehlo.add %v9614, %v9615 : tensor<768x1x7x7xf32>
    %v9617 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9618 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9619 = stablehlo.multiply %v9617, %s3b2dWv : tensor<768x1x7x7xf32>
    %v9620 = stablehlo.multiply %armeans3b2dW, %armeans3b2dW : tensor<768x1x7x7xf32>
    %v9621 = stablehlo.multiply %v9618, %v9620 : tensor<768x1x7x7xf32>
    %v9622 = stablehlo.add %v9619, %v9621 : tensor<768x1x7x7xf32>
    %v9623 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9624 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9625 = stablehlo.divide %v9616, %v9623 : tensor<768x1x7x7xf32>
    %v9626 = stablehlo.divide %v9622, %v9624 : tensor<768x1x7x7xf32>
    %v9627 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9628 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9629 = stablehlo.sqrt %v9626 : tensor<768x1x7x7xf32>
    %v9630 = stablehlo.add %v9629, %v9628 : tensor<768x1x7x7xf32>
    %v9631 = stablehlo.divide %v9625, %v9630 : tensor<768x1x7x7xf32>
    %v9632 = stablehlo.multiply %v9627, %v9631 : tensor<768x1x7x7xf32>
    %v9633 = stablehlo.subtract %s3b2dW, %v9632 : tensor<768x1x7x7xf32>
    %v9634 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9635 = stablehlo.multiply %v9634, %v9627 : tensor<768x1x7x7xf32>
    %v9636 = stablehlo.multiply %v9635, %s3b2dW : tensor<768x1x7x7xf32>
    %v9637 = stablehlo.subtract %v9633, %v9636 : tensor<768x1x7x7xf32>
    %arsums3b2db = "stablehlo.all_reduce"(%v1198) ({
    ^bb0(%aras3b2db: tensor<f32>, %arbs3b2db: tensor<f32>):
      %aradds3b2db = stablehlo.add %aras3b2db, %arbs3b2db : tensor<f32>
      stablehlo.return %aradds3b2db : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arns3b2db = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeans3b2db = stablehlo.divide %arsums3b2db, %arns3b2db : tensor<768xf32>
    %v9638 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9639 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9640 = stablehlo.multiply %v9638, %s3b2dbm : tensor<768xf32>
    %v9641 = stablehlo.multiply %v9639, %armeans3b2db : tensor<768xf32>
    %v9642 = stablehlo.add %v9640, %v9641 : tensor<768xf32>
    %v9643 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9644 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9645 = stablehlo.multiply %v9643, %s3b2dbv : tensor<768xf32>
    %v9646 = stablehlo.multiply %armeans3b2db, %armeans3b2db : tensor<768xf32>
    %v9647 = stablehlo.multiply %v9644, %v9646 : tensor<768xf32>
    %v9648 = stablehlo.add %v9645, %v9647 : tensor<768xf32>
    %v9649 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9650 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9651 = stablehlo.multiply %v9649, %s3b2dbm : tensor<768xf32>
    %v9652 = stablehlo.multiply %v9650, %armeans3b2db : tensor<768xf32>
    %v9653 = stablehlo.add %v9651, %v9652 : tensor<768xf32>
    %v9654 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9655 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9656 = stablehlo.multiply %v9654, %s3b2dbv : tensor<768xf32>
    %v9657 = stablehlo.multiply %armeans3b2db, %armeans3b2db : tensor<768xf32>
    %v9658 = stablehlo.multiply %v9655, %v9657 : tensor<768xf32>
    %v9659 = stablehlo.add %v9656, %v9658 : tensor<768xf32>
    %v9660 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9661 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9662 = stablehlo.divide %v9653, %v9660 : tensor<768xf32>
    %v9663 = stablehlo.divide %v9659, %v9661 : tensor<768xf32>
    %v9664 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9665 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9666 = stablehlo.sqrt %v9663 : tensor<768xf32>
    %v9667 = stablehlo.add %v9666, %v9665 : tensor<768xf32>
    %v9668 = stablehlo.divide %v9662, %v9667 : tensor<768xf32>
    %v9669 = stablehlo.multiply %v9664, %v9668 : tensor<768xf32>
    %v9670 = stablehlo.subtract %s3b2db, %v9669 : tensor<768xf32>
    %v9671 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9672 = stablehlo.multiply %v9671, %v9664 : tensor<768xf32>
    %v9673 = stablehlo.multiply %v9672, %s3b2db : tensor<768xf32>
    %v9674 = stablehlo.subtract %v9670, %v9673 : tensor<768xf32>
    %arsums3b2ng = "stablehlo.all_reduce"(%v1187) ({
    ^bb0(%aras3b2ng: tensor<f32>, %arbs3b2ng: tensor<f32>):
      %aradds3b2ng = stablehlo.add %aras3b2ng, %arbs3b2ng : tensor<f32>
      stablehlo.return %aradds3b2ng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns3b2ng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans3b2ng = stablehlo.divide %arsums3b2ng, %arns3b2ng : tensor<f32>
    %v9675 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9676 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9677 = stablehlo.multiply %v9675, %s3b2ngm : tensor<f32>
    %v9678 = stablehlo.multiply %v9676, %armeans3b2ng : tensor<f32>
    %v9679 = stablehlo.add %v9677, %v9678 : tensor<f32>
    %v9680 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9681 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9682 = stablehlo.multiply %v9680, %s3b2ngv : tensor<f32>
    %v9683 = stablehlo.multiply %armeans3b2ng, %armeans3b2ng : tensor<f32>
    %v9684 = stablehlo.multiply %v9681, %v9683 : tensor<f32>
    %v9685 = stablehlo.add %v9682, %v9684 : tensor<f32>
    %v9686 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9687 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9688 = stablehlo.multiply %v9686, %s3b2ngm : tensor<f32>
    %v9689 = stablehlo.multiply %v9687, %armeans3b2ng : tensor<f32>
    %v9690 = stablehlo.add %v9688, %v9689 : tensor<f32>
    %v9691 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9692 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9693 = stablehlo.multiply %v9691, %s3b2ngv : tensor<f32>
    %v9694 = stablehlo.multiply %armeans3b2ng, %armeans3b2ng : tensor<f32>
    %v9695 = stablehlo.multiply %v9692, %v9694 : tensor<f32>
    %v9696 = stablehlo.add %v9693, %v9695 : tensor<f32>
    %v9697 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9698 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9699 = stablehlo.divide %v9690, %v9697 : tensor<f32>
    %v9700 = stablehlo.divide %v9696, %v9698 : tensor<f32>
    %v9701 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9702 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9703 = stablehlo.sqrt %v9700 : tensor<f32>
    %v9704 = stablehlo.add %v9703, %v9702 : tensor<f32>
    %v9705 = stablehlo.divide %v9699, %v9704 : tensor<f32>
    %v9706 = stablehlo.multiply %v9701, %v9705 : tensor<f32>
    %v9707 = stablehlo.subtract %s3b2ng, %v9706 : tensor<f32>
    %v9708 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9709 = stablehlo.multiply %v9708, %v9701 : tensor<f32>
    %v9710 = stablehlo.multiply %v9709, %s3b2ng : tensor<f32>
    %v9711 = stablehlo.subtract %v9707, %v9710 : tensor<f32>
    %arsums3b2nbt = "stablehlo.all_reduce"(%v1189) ({
    ^bb0(%aras3b2nbt: tensor<f32>, %arbs3b2nbt: tensor<f32>):
      %aradds3b2nbt = stablehlo.add %aras3b2nbt, %arbs3b2nbt : tensor<f32>
      stablehlo.return %aradds3b2nbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arns3b2nbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeans3b2nbt = stablehlo.divide %arsums3b2nbt, %arns3b2nbt : tensor<f32>
    %v9712 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9713 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9714 = stablehlo.multiply %v9712, %s3b2nbtm : tensor<f32>
    %v9715 = stablehlo.multiply %v9713, %armeans3b2nbt : tensor<f32>
    %v9716 = stablehlo.add %v9714, %v9715 : tensor<f32>
    %v9717 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9718 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9719 = stablehlo.multiply %v9717, %s3b2nbtv : tensor<f32>
    %v9720 = stablehlo.multiply %armeans3b2nbt, %armeans3b2nbt : tensor<f32>
    %v9721 = stablehlo.multiply %v9718, %v9720 : tensor<f32>
    %v9722 = stablehlo.add %v9719, %v9721 : tensor<f32>
    %v9723 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9724 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9725 = stablehlo.multiply %v9723, %s3b2nbtm : tensor<f32>
    %v9726 = stablehlo.multiply %v9724, %armeans3b2nbt : tensor<f32>
    %v9727 = stablehlo.add %v9725, %v9726 : tensor<f32>
    %v9728 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9729 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9730 = stablehlo.multiply %v9728, %s3b2nbtv : tensor<f32>
    %v9731 = stablehlo.multiply %armeans3b2nbt, %armeans3b2nbt : tensor<f32>
    %v9732 = stablehlo.multiply %v9729, %v9731 : tensor<f32>
    %v9733 = stablehlo.add %v9730, %v9732 : tensor<f32>
    %v9734 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9735 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9736 = stablehlo.divide %v9727, %v9734 : tensor<f32>
    %v9737 = stablehlo.divide %v9733, %v9735 : tensor<f32>
    %v9738 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9739 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9740 = stablehlo.sqrt %v9737 : tensor<f32>
    %v9741 = stablehlo.add %v9740, %v9739 : tensor<f32>
    %v9742 = stablehlo.divide %v9736, %v9741 : tensor<f32>
    %v9743 = stablehlo.multiply %v9738, %v9742 : tensor<f32>
    %v9744 = stablehlo.subtract %s3b2nbt, %v9743 : tensor<f32>
    %v9745 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9746 = stablehlo.multiply %v9745, %v9738 : tensor<f32>
    %v9747 = stablehlo.multiply %v9746, %s3b2nbt : tensor<f32>
    %v9748 = stablehlo.subtract %v9744, %v9747 : tensor<f32>
    %arsums3b2eW = "stablehlo.all_reduce"(%v1168) ({
    ^bb0(%aras3b2eW: tensor<f32>, %arbs3b2eW: tensor<f32>):
      %aradds3b2eW = stablehlo.add %aras3b2eW, %arbs3b2eW : tensor<f32>
      stablehlo.return %aradds3b2eW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<3072x768x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %arns3b2eW = stablehlo.constant dense<2.0> : tensor<3072x768x1x1xf32>
    %armeans3b2eW = stablehlo.divide %arsums3b2eW, %arns3b2eW : tensor<3072x768x1x1xf32>
    %v9749 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9750 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9751 = stablehlo.multiply %v9749, %s3b2eWm : tensor<3072x768x1x1xf32>
    %v9752 = stablehlo.multiply %v9750, %armeans3b2eW : tensor<3072x768x1x1xf32>
    %v9753 = stablehlo.add %v9751, %v9752 : tensor<3072x768x1x1xf32>
    %v9754 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9755 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9756 = stablehlo.multiply %v9754, %s3b2eWv : tensor<3072x768x1x1xf32>
    %v9757 = stablehlo.multiply %armeans3b2eW, %armeans3b2eW : tensor<3072x768x1x1xf32>
    %v9758 = stablehlo.multiply %v9755, %v9757 : tensor<3072x768x1x1xf32>
    %v9759 = stablehlo.add %v9756, %v9758 : tensor<3072x768x1x1xf32>
    %v9760 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9761 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9762 = stablehlo.multiply %v9760, %s3b2eWm : tensor<3072x768x1x1xf32>
    %v9763 = stablehlo.multiply %v9761, %armeans3b2eW : tensor<3072x768x1x1xf32>
    %v9764 = stablehlo.add %v9762, %v9763 : tensor<3072x768x1x1xf32>
    %v9765 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9766 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9767 = stablehlo.multiply %v9765, %s3b2eWv : tensor<3072x768x1x1xf32>
    %v9768 = stablehlo.multiply %armeans3b2eW, %armeans3b2eW : tensor<3072x768x1x1xf32>
    %v9769 = stablehlo.multiply %v9766, %v9768 : tensor<3072x768x1x1xf32>
    %v9770 = stablehlo.add %v9767, %v9769 : tensor<3072x768x1x1xf32>
    %v9771 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9772 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9773 = stablehlo.divide %v9764, %v9771 : tensor<3072x768x1x1xf32>
    %v9774 = stablehlo.divide %v9770, %v9772 : tensor<3072x768x1x1xf32>
    %v9775 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9776 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9777 = stablehlo.sqrt %v9774 : tensor<3072x768x1x1xf32>
    %v9778 = stablehlo.add %v9777, %v9776 : tensor<3072x768x1x1xf32>
    %v9779 = stablehlo.divide %v9773, %v9778 : tensor<3072x768x1x1xf32>
    %v9780 = stablehlo.multiply %v9775, %v9779 : tensor<3072x768x1x1xf32>
    %v9781 = stablehlo.subtract %s3b2eW, %v9780 : tensor<3072x768x1x1xf32>
    %v9782 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9783 = stablehlo.multiply %v9782, %v9775 : tensor<3072x768x1x1xf32>
    %v9784 = stablehlo.multiply %v9783, %s3b2eW : tensor<3072x768x1x1xf32>
    %v9785 = stablehlo.subtract %v9781, %v9784 : tensor<3072x768x1x1xf32>
    %arsums3b2eb = "stablehlo.all_reduce"(%v1171) ({
    ^bb0(%aras3b2eb: tensor<f32>, %arbs3b2eb: tensor<f32>):
      %aradds3b2eb = stablehlo.add %aras3b2eb, %arbs3b2eb : tensor<f32>
      stablehlo.return %aradds3b2eb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<3072xf32>) -> tensor<3072xf32>
    %arns3b2eb = stablehlo.constant dense<2.0> : tensor<3072xf32>
    %armeans3b2eb = stablehlo.divide %arsums3b2eb, %arns3b2eb : tensor<3072xf32>
    %v9786 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9787 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9788 = stablehlo.multiply %v9786, %s3b2ebm : tensor<3072xf32>
    %v9789 = stablehlo.multiply %v9787, %armeans3b2eb : tensor<3072xf32>
    %v9790 = stablehlo.add %v9788, %v9789 : tensor<3072xf32>
    %v9791 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9792 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9793 = stablehlo.multiply %v9791, %s3b2ebv : tensor<3072xf32>
    %v9794 = stablehlo.multiply %armeans3b2eb, %armeans3b2eb : tensor<3072xf32>
    %v9795 = stablehlo.multiply %v9792, %v9794 : tensor<3072xf32>
    %v9796 = stablehlo.add %v9793, %v9795 : tensor<3072xf32>
    %v9797 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9798 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9799 = stablehlo.multiply %v9797, %s3b2ebm : tensor<3072xf32>
    %v9800 = stablehlo.multiply %v9798, %armeans3b2eb : tensor<3072xf32>
    %v9801 = stablehlo.add %v9799, %v9800 : tensor<3072xf32>
    %v9802 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9803 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9804 = stablehlo.multiply %v9802, %s3b2ebv : tensor<3072xf32>
    %v9805 = stablehlo.multiply %armeans3b2eb, %armeans3b2eb : tensor<3072xf32>
    %v9806 = stablehlo.multiply %v9803, %v9805 : tensor<3072xf32>
    %v9807 = stablehlo.add %v9804, %v9806 : tensor<3072xf32>
    %v9808 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9809 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9810 = stablehlo.divide %v9801, %v9808 : tensor<3072xf32>
    %v9811 = stablehlo.divide %v9807, %v9809 : tensor<3072xf32>
    %v9812 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9813 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9814 = stablehlo.sqrt %v9811 : tensor<3072xf32>
    %v9815 = stablehlo.add %v9814, %v9813 : tensor<3072xf32>
    %v9816 = stablehlo.divide %v9810, %v9815 : tensor<3072xf32>
    %v9817 = stablehlo.multiply %v9812, %v9816 : tensor<3072xf32>
    %v9818 = stablehlo.subtract %s3b2eb, %v9817 : tensor<3072xf32>
    %v9819 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9820 = stablehlo.multiply %v9819, %v9812 : tensor<3072xf32>
    %v9821 = stablehlo.multiply %v9820, %s3b2eb : tensor<3072xf32>
    %v9822 = stablehlo.subtract %v9818, %v9821 : tensor<3072xf32>
    %arsums3b2pW = "stablehlo.all_reduce"(%v1159) ({
    ^bb0(%aras3b2pW: tensor<f32>, %arbs3b2pW: tensor<f32>):
      %aradds3b2pW = stablehlo.add %aras3b2pW, %arbs3b2pW : tensor<f32>
      stablehlo.return %aradds3b2pW : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768x3072x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %arns3b2pW = stablehlo.constant dense<2.0> : tensor<768x3072x1x1xf32>
    %armeans3b2pW = stablehlo.divide %arsums3b2pW, %arns3b2pW : tensor<768x3072x1x1xf32>
    %v9823 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9824 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9825 = stablehlo.multiply %v9823, %s3b2pWm : tensor<768x3072x1x1xf32>
    %v9826 = stablehlo.multiply %v9824, %armeans3b2pW : tensor<768x3072x1x1xf32>
    %v9827 = stablehlo.add %v9825, %v9826 : tensor<768x3072x1x1xf32>
    %v9828 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9829 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9830 = stablehlo.multiply %v9828, %s3b2pWv : tensor<768x3072x1x1xf32>
    %v9831 = stablehlo.multiply %armeans3b2pW, %armeans3b2pW : tensor<768x3072x1x1xf32>
    %v9832 = stablehlo.multiply %v9829, %v9831 : tensor<768x3072x1x1xf32>
    %v9833 = stablehlo.add %v9830, %v9832 : tensor<768x3072x1x1xf32>
    %v9834 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9835 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9836 = stablehlo.multiply %v9834, %s3b2pWm : tensor<768x3072x1x1xf32>
    %v9837 = stablehlo.multiply %v9835, %armeans3b2pW : tensor<768x3072x1x1xf32>
    %v9838 = stablehlo.add %v9836, %v9837 : tensor<768x3072x1x1xf32>
    %v9839 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9840 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9841 = stablehlo.multiply %v9839, %s3b2pWv : tensor<768x3072x1x1xf32>
    %v9842 = stablehlo.multiply %armeans3b2pW, %armeans3b2pW : tensor<768x3072x1x1xf32>
    %v9843 = stablehlo.multiply %v9840, %v9842 : tensor<768x3072x1x1xf32>
    %v9844 = stablehlo.add %v9841, %v9843 : tensor<768x3072x1x1xf32>
    %v9845 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9846 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9847 = stablehlo.divide %v9838, %v9845 : tensor<768x3072x1x1xf32>
    %v9848 = stablehlo.divide %v9844, %v9846 : tensor<768x3072x1x1xf32>
    %v9849 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9850 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9851 = stablehlo.sqrt %v9848 : tensor<768x3072x1x1xf32>
    %v9852 = stablehlo.add %v9851, %v9850 : tensor<768x3072x1x1xf32>
    %v9853 = stablehlo.divide %v9847, %v9852 : tensor<768x3072x1x1xf32>
    %v9854 = stablehlo.multiply %v9849, %v9853 : tensor<768x3072x1x1xf32>
    %v9855 = stablehlo.subtract %s3b2pW, %v9854 : tensor<768x3072x1x1xf32>
    %v9856 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9857 = stablehlo.multiply %v9856, %v9849 : tensor<768x3072x1x1xf32>
    %v9858 = stablehlo.multiply %v9857, %s3b2pW : tensor<768x3072x1x1xf32>
    %v9859 = stablehlo.subtract %v9855, %v9858 : tensor<768x3072x1x1xf32>
    %arsums3b2pb = "stablehlo.all_reduce"(%v1162) ({
    ^bb0(%aras3b2pb: tensor<f32>, %arbs3b2pb: tensor<f32>):
      %aradds3b2pb = stablehlo.add %aras3b2pb, %arbs3b2pb : tensor<f32>
      stablehlo.return %aradds3b2pb : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arns3b2pb = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeans3b2pb = stablehlo.divide %arsums3b2pb, %arns3b2pb : tensor<768xf32>
    %v9860 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9861 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9862 = stablehlo.multiply %v9860, %s3b2pbm : tensor<768xf32>
    %v9863 = stablehlo.multiply %v9861, %armeans3b2pb : tensor<768xf32>
    %v9864 = stablehlo.add %v9862, %v9863 : tensor<768xf32>
    %v9865 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9866 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9867 = stablehlo.multiply %v9865, %s3b2pbv : tensor<768xf32>
    %v9868 = stablehlo.multiply %armeans3b2pb, %armeans3b2pb : tensor<768xf32>
    %v9869 = stablehlo.multiply %v9866, %v9868 : tensor<768xf32>
    %v9870 = stablehlo.add %v9867, %v9869 : tensor<768xf32>
    %v9871 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9872 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9873 = stablehlo.multiply %v9871, %s3b2pbm : tensor<768xf32>
    %v9874 = stablehlo.multiply %v9872, %armeans3b2pb : tensor<768xf32>
    %v9875 = stablehlo.add %v9873, %v9874 : tensor<768xf32>
    %v9876 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9877 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9878 = stablehlo.multiply %v9876, %s3b2pbv : tensor<768xf32>
    %v9879 = stablehlo.multiply %armeans3b2pb, %armeans3b2pb : tensor<768xf32>
    %v9880 = stablehlo.multiply %v9877, %v9879 : tensor<768xf32>
    %v9881 = stablehlo.add %v9878, %v9880 : tensor<768xf32>
    %v9882 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9883 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9884 = stablehlo.divide %v9875, %v9882 : tensor<768xf32>
    %v9885 = stablehlo.divide %v9881, %v9883 : tensor<768xf32>
    %v9886 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9887 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9888 = stablehlo.sqrt %v9885 : tensor<768xf32>
    %v9889 = stablehlo.add %v9888, %v9887 : tensor<768xf32>
    %v9890 = stablehlo.divide %v9884, %v9889 : tensor<768xf32>
    %v9891 = stablehlo.multiply %v9886, %v9890 : tensor<768xf32>
    %v9892 = stablehlo.subtract %s3b2pb, %v9891 : tensor<768xf32>
    %v9893 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9894 = stablehlo.multiply %v9893, %v9886 : tensor<768xf32>
    %v9895 = stablehlo.multiply %v9894, %s3b2pb : tensor<768xf32>
    %v9896 = stablehlo.subtract %v9892, %v9895 : tensor<768xf32>
    %arsums3b2lg = "stablehlo.all_reduce"(%v1153) ({
    ^bb0(%aras3b2lg: tensor<f32>, %arbs3b2lg: tensor<f32>):
      %aradds3b2lg = stablehlo.add %aras3b2lg, %arbs3b2lg : tensor<f32>
      stablehlo.return %aradds3b2lg : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768xf32>) -> tensor<768xf32>
    %arns3b2lg = stablehlo.constant dense<2.0> : tensor<768xf32>
    %armeans3b2lg = stablehlo.divide %arsums3b2lg, %arns3b2lg : tensor<768xf32>
    %v9897 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9898 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9899 = stablehlo.multiply %v9897, %s3b2lgm : tensor<768xf32>
    %v9900 = stablehlo.multiply %v9898, %armeans3b2lg : tensor<768xf32>
    %v9901 = stablehlo.add %v9899, %v9900 : tensor<768xf32>
    %v9902 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9903 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9904 = stablehlo.multiply %v9902, %s3b2lgv : tensor<768xf32>
    %v9905 = stablehlo.multiply %armeans3b2lg, %armeans3b2lg : tensor<768xf32>
    %v9906 = stablehlo.multiply %v9903, %v9905 : tensor<768xf32>
    %v9907 = stablehlo.add %v9904, %v9906 : tensor<768xf32>
    %v9908 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9909 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9910 = stablehlo.multiply %v9908, %s3b2lgm : tensor<768xf32>
    %v9911 = stablehlo.multiply %v9909, %armeans3b2lg : tensor<768xf32>
    %v9912 = stablehlo.add %v9910, %v9911 : tensor<768xf32>
    %v9913 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9914 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9915 = stablehlo.multiply %v9913, %s3b2lgv : tensor<768xf32>
    %v9916 = stablehlo.multiply %armeans3b2lg, %armeans3b2lg : tensor<768xf32>
    %v9917 = stablehlo.multiply %v9914, %v9916 : tensor<768xf32>
    %v9918 = stablehlo.add %v9915, %v9917 : tensor<768xf32>
    %v9919 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9920 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9921 = stablehlo.divide %v9912, %v9919 : tensor<768xf32>
    %v9922 = stablehlo.divide %v9918, %v9920 : tensor<768xf32>
    %v9923 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9924 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9925 = stablehlo.sqrt %v9922 : tensor<768xf32>
    %v9926 = stablehlo.add %v9925, %v9924 : tensor<768xf32>
    %v9927 = stablehlo.divide %v9921, %v9926 : tensor<768xf32>
    %v9928 = stablehlo.multiply %v9923, %v9927 : tensor<768xf32>
    %v9929 = stablehlo.subtract %s3b2lg, %v9928 : tensor<768xf32>
    %v9930 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9931 = stablehlo.multiply %v9930, %v9923 : tensor<768xf32>
    %v9932 = stablehlo.multiply %v9931, %s3b2lg : tensor<768xf32>
    %v9933 = stablehlo.subtract %v9929, %v9932 : tensor<768xf32>
    %arsumhng = "stablehlo.all_reduce"(%v1077) ({
    ^bb0(%arahng: tensor<f32>, %arbhng: tensor<f32>):
      %araddhng = stablehlo.add %arahng, %arbhng : tensor<f32>
      stablehlo.return %araddhng : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arnhng = stablehlo.constant dense<2.0> : tensor<f32>
    %armeanhng = stablehlo.divide %arsumhng, %arnhng : tensor<f32>
    %v9934 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9935 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9936 = stablehlo.multiply %v9934, %hngm : tensor<f32>
    %v9937 = stablehlo.multiply %v9935, %armeanhng : tensor<f32>
    %v9938 = stablehlo.add %v9936, %v9937 : tensor<f32>
    %v9939 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9940 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9941 = stablehlo.multiply %v9939, %hngv : tensor<f32>
    %v9942 = stablehlo.multiply %armeanhng, %armeanhng : tensor<f32>
    %v9943 = stablehlo.multiply %v9940, %v9942 : tensor<f32>
    %v9944 = stablehlo.add %v9941, %v9943 : tensor<f32>
    %v9945 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9946 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9947 = stablehlo.multiply %v9945, %hngm : tensor<f32>
    %v9948 = stablehlo.multiply %v9946, %armeanhng : tensor<f32>
    %v9949 = stablehlo.add %v9947, %v9948 : tensor<f32>
    %v9950 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9951 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9952 = stablehlo.multiply %v9950, %hngv : tensor<f32>
    %v9953 = stablehlo.multiply %armeanhng, %armeanhng : tensor<f32>
    %v9954 = stablehlo.multiply %v9951, %v9953 : tensor<f32>
    %v9955 = stablehlo.add %v9952, %v9954 : tensor<f32>
    %v9956 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9957 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9958 = stablehlo.divide %v9949, %v9956 : tensor<f32>
    %v9959 = stablehlo.divide %v9955, %v9957 : tensor<f32>
    %v9960 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9961 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9962 = stablehlo.sqrt %v9959 : tensor<f32>
    %v9963 = stablehlo.add %v9962, %v9961 : tensor<f32>
    %v9964 = stablehlo.divide %v9958, %v9963 : tensor<f32>
    %v9965 = stablehlo.multiply %v9960, %v9964 : tensor<f32>
    %v9966 = stablehlo.subtract %hng, %v9965 : tensor<f32>
    %v9967 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9968 = stablehlo.multiply %v9967, %v9960 : tensor<f32>
    %v9969 = stablehlo.multiply %v9968, %hng : tensor<f32>
    %v9970 = stablehlo.subtract %v9966, %v9969 : tensor<f32>
    %arsumhnbt = "stablehlo.all_reduce"(%v1079) ({
    ^bb0(%arahnbt: tensor<f32>, %arbhnbt: tensor<f32>):
      %araddhnbt = stablehlo.add %arahnbt, %arbhnbt : tensor<f32>
      stablehlo.return %araddhnbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
    %arnhnbt = stablehlo.constant dense<2.0> : tensor<f32>
    %armeanhnbt = stablehlo.divide %arsumhnbt, %arnhnbt : tensor<f32>
    %v9971 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9972 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9973 = stablehlo.multiply %v9971, %hnbtm : tensor<f32>
    %v9974 = stablehlo.multiply %v9972, %armeanhnbt : tensor<f32>
    %v9975 = stablehlo.add %v9973, %v9974 : tensor<f32>
    %v9976 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9977 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9978 = stablehlo.multiply %v9976, %hnbtv : tensor<f32>
    %v9979 = stablehlo.multiply %armeanhnbt, %armeanhnbt : tensor<f32>
    %v9980 = stablehlo.multiply %v9977, %v9979 : tensor<f32>
    %v9981 = stablehlo.add %v9978, %v9980 : tensor<f32>
    %v9982 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9983 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9984 = stablehlo.multiply %v9982, %hnbtm : tensor<f32>
    %v9985 = stablehlo.multiply %v9983, %armeanhnbt : tensor<f32>
    %v9986 = stablehlo.add %v9984, %v9985 : tensor<f32>
    %v9987 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9988 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9989 = stablehlo.multiply %v9987, %hnbtv : tensor<f32>
    %v9990 = stablehlo.multiply %armeanhnbt, %armeanhnbt : tensor<f32>
    %v9991 = stablehlo.multiply %v9988, %v9990 : tensor<f32>
    %v9992 = stablehlo.add %v9989, %v9991 : tensor<f32>
    %v9993 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9994 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9995 = stablehlo.divide %v9986, %v9993 : tensor<f32>
    %v9996 = stablehlo.divide %v9992, %v9994 : tensor<f32>
    %v9997 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9998 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9999 = stablehlo.sqrt %v9996 : tensor<f32>
    %v10000 = stablehlo.add %v9999, %v9998 : tensor<f32>
    %v10001 = stablehlo.divide %v9995, %v10000 : tensor<f32>
    %v10002 = stablehlo.multiply %v9997, %v10001 : tensor<f32>
    %v10003 = stablehlo.subtract %hnbt, %v10002 : tensor<f32>
    %v10004 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v10005 = stablehlo.multiply %v10004, %v9997 : tensor<f32>
    %v10006 = stablehlo.multiply %v10005, %hnbt : tensor<f32>
    %v10007 = stablehlo.subtract %v10003, %v10006 : tensor<f32>
    %arsumWd = "stablehlo.all_reduce"(%v1059) ({
    ^bb0(%araWd: tensor<f32>, %arbWd: tensor<f32>):
      %araddWd = stablehlo.add %araWd, %arbWd : tensor<f32>
      stablehlo.return %araddWd : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<768x10xf32>) -> tensor<768x10xf32>
    %arnWd = stablehlo.constant dense<2.0> : tensor<768x10xf32>
    %armeanWd = stablehlo.divide %arsumWd, %arnWd : tensor<768x10xf32>
    %v10008 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10009 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10010 = stablehlo.multiply %v10008, %Wdm : tensor<768x10xf32>
    %v10011 = stablehlo.multiply %v10009, %armeanWd : tensor<768x10xf32>
    %v10012 = stablehlo.add %v10010, %v10011 : tensor<768x10xf32>
    %v10013 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10014 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10015 = stablehlo.multiply %v10013, %Wdv : tensor<768x10xf32>
    %v10016 = stablehlo.multiply %armeanWd, %armeanWd : tensor<768x10xf32>
    %v10017 = stablehlo.multiply %v10014, %v10016 : tensor<768x10xf32>
    %v10018 = stablehlo.add %v10015, %v10017 : tensor<768x10xf32>
    %v10019 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10020 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10021 = stablehlo.multiply %v10019, %Wdm : tensor<768x10xf32>
    %v10022 = stablehlo.multiply %v10020, %armeanWd : tensor<768x10xf32>
    %v10023 = stablehlo.add %v10021, %v10022 : tensor<768x10xf32>
    %v10024 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10025 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10026 = stablehlo.multiply %v10024, %Wdv : tensor<768x10xf32>
    %v10027 = stablehlo.multiply %armeanWd, %armeanWd : tensor<768x10xf32>
    %v10028 = stablehlo.multiply %v10025, %v10027 : tensor<768x10xf32>
    %v10029 = stablehlo.add %v10026, %v10028 : tensor<768x10xf32>
    %v10030 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10031 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10032 = stablehlo.divide %v10023, %v10030 : tensor<768x10xf32>
    %v10033 = stablehlo.divide %v10029, %v10031 : tensor<768x10xf32>
    %v10034 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10035 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10036 = stablehlo.sqrt %v10033 : tensor<768x10xf32>
    %v10037 = stablehlo.add %v10036, %v10035 : tensor<768x10xf32>
    %v10038 = stablehlo.divide %v10032, %v10037 : tensor<768x10xf32>
    %v10039 = stablehlo.multiply %v10034, %v10038 : tensor<768x10xf32>
    %v10040 = stablehlo.subtract %Wd, %v10039 : tensor<768x10xf32>
    %v10041 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10042 = stablehlo.multiply %v10041, %v10034 : tensor<768x10xf32>
    %v10043 = stablehlo.multiply %v10042, %Wd : tensor<768x10xf32>
    %v10044 = stablehlo.subtract %v10040, %v10043 : tensor<768x10xf32>
    %arsumbd = "stablehlo.all_reduce"(%v1061) ({
    ^bb0(%arabd: tensor<f32>, %arbbd: tensor<f32>):
      %araddbd = stablehlo.add %arabd, %arbbd : tensor<f32>
      stablehlo.return %araddbd : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<10xf32>) -> tensor<10xf32>
    %arnbd = stablehlo.constant dense<2.0> : tensor<10xf32>
    %armeanbd = stablehlo.divide %arsumbd, %arnbd : tensor<10xf32>
    %v10045 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10046 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10047 = stablehlo.multiply %v10045, %bdm : tensor<10xf32>
    %v10048 = stablehlo.multiply %v10046, %armeanbd : tensor<10xf32>
    %v10049 = stablehlo.add %v10047, %v10048 : tensor<10xf32>
    %v10050 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10051 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10052 = stablehlo.multiply %v10050, %bdv : tensor<10xf32>
    %v10053 = stablehlo.multiply %armeanbd, %armeanbd : tensor<10xf32>
    %v10054 = stablehlo.multiply %v10051, %v10053 : tensor<10xf32>
    %v10055 = stablehlo.add %v10052, %v10054 : tensor<10xf32>
    %v10056 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10057 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10058 = stablehlo.multiply %v10056, %bdm : tensor<10xf32>
    %v10059 = stablehlo.multiply %v10057, %armeanbd : tensor<10xf32>
    %v10060 = stablehlo.add %v10058, %v10059 : tensor<10xf32>
    %v10061 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10062 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10063 = stablehlo.multiply %v10061, %bdv : tensor<10xf32>
    %v10064 = stablehlo.multiply %armeanbd, %armeanbd : tensor<10xf32>
    %v10065 = stablehlo.multiply %v10062, %v10064 : tensor<10xf32>
    %v10066 = stablehlo.add %v10063, %v10065 : tensor<10xf32>
    %v10067 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10068 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10069 = stablehlo.divide %v10060, %v10067 : tensor<10xf32>
    %v10070 = stablehlo.divide %v10066, %v10068 : tensor<10xf32>
    %v10071 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10072 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10073 = stablehlo.sqrt %v10070 : tensor<10xf32>
    %v10074 = stablehlo.add %v10073, %v10072 : tensor<10xf32>
    %v10075 = stablehlo.divide %v10069, %v10074 : tensor<10xf32>
    %v10076 = stablehlo.multiply %v10071, %v10075 : tensor<10xf32>
    %v10077 = stablehlo.subtract %bd, %v10076 : tensor<10xf32>
    %v10078 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10079 = stablehlo.multiply %v10078, %v10071 : tensor<10xf32>
    %v10080 = stablehlo.multiply %v10079, %bd : tensor<10xf32>
    %v10081 = stablehlo.subtract %v10077, %v10080 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1022 : tensor<32x10xf32>
    %lohll = stablehlo.multiply %onehot, %llog : tensor<32x10xf32>
    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %lomac = stablehlo.constant dense<0.900000> : tensor<32xf32>
    %laKc = stablehlo.constant dense<0.010000> : tensor<32xf32>
    %llt1 = stablehlo.multiply %lomac, %lt1s : tensor<32xf32>
    %llt2 = stablehlo.multiply %laKc, %llsr : tensor<32xf32>
    %llpe = stablehlo.add %llt1, %llt2 : tensor<32xf32>
    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %lbfc = stablehlo.constant dense<32.0> : tensor<f32>
    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>
    %loss = stablehlo.negate %lossm : tensor<f32>
    return %v3458, %v3495, %v3532, %v3569, %v3606, %v3643, %v3680, %v3717, %v3754, %v3791, %v3828, %v3865, %v3902, %v3939, %v3976, %v4013, %v4050, %v4087, %v4124, %v4161, %v4198, %v4235, %v4272, %v4309, %v4346, %v4383, %v4420, %v4457, %v4494, %v4531, %v4568, %v4605, %v4642, %v4679, %v4716, %v4753, %v4790, %v4827, %v4864, %v4901, %v4938, %v4975, %v5012, %v5049, %v5086, %v5123, %v5160, %v5197, %v5234, %v5271, %v5308, %v5345, %v5382, %v5419, %v5456, %v5493, %v5530, %v5567, %v5604, %v5641, %v5678, %v5715, %v5752, %v5789, %v5826, %v5863, %v5900, %v5937, %v5974, %v6011, %v6048, %v6085, %v6122, %v6159, %v6196, %v6233, %v6270, %v6307, %v6344, %v6381, %v6418, %v6455, %v6492, %v6529, %v6566, %v6603, %v6640, %v6677, %v6714, %v6751, %v6788, %v6825, %v6862, %v6899, %v6936, %v6973, %v7010, %v7047, %v7084, %v7121, %v7158, %v7195, %v7232, %v7269, %v7306, %v7343, %v7380, %v7417, %v7454, %v7491, %v7528, %v7565, %v7602, %v7639, %v7676, %v7713, %v7750, %v7787, %v7824, %v7861, %v7898, %v7935, %v7972, %v8009, %v8046, %v8083, %v8120, %v8157, %v8194, %v8231, %v8268, %v8305, %v8342, %v8379, %v8416, %v8453, %v8490, %v8527, %v8564, %v8601, %v8638, %v8675, %v8712, %v8749, %v8786, %v8823, %v8860, %v8897, %v8934, %v8971, %v9008, %v9045, %v9082, %v9119, %v9156, %v9193, %v9230, %v9267, %v9304, %v9341, %v9378, %v9415, %v9452, %v9489, %v9526, %v9563, %v9600, %v9637, %v9674, %v9711, %v9748, %v9785, %v9822, %v9859, %v9896, %v9933, %v9970, %v10007, %v10044, %v10081, %v3426, %v3463, %v3500, %v3537, %v3574, %v3611, %v3648, %v3685, %v3722, %v3759, %v3796, %v3833, %v3870, %v3907, %v3944, %v3981, %v4018, %v4055, %v4092, %v4129, %v4166, %v4203, %v4240, %v4277, %v4314, %v4351, %v4388, %v4425, %v4462, %v4499, %v4536, %v4573, %v4610, %v4647, %v4684, %v4721, %v4758, %v4795, %v4832, %v4869, %v4906, %v4943, %v4980, %v5017, %v5054, %v5091, %v5128, %v5165, %v5202, %v5239, %v5276, %v5313, %v5350, %v5387, %v5424, %v5461, %v5498, %v5535, %v5572, %v5609, %v5646, %v5683, %v5720, %v5757, %v5794, %v5831, %v5868, %v5905, %v5942, %v5979, %v6016, %v6053, %v6090, %v6127, %v6164, %v6201, %v6238, %v6275, %v6312, %v6349, %v6386, %v6423, %v6460, %v6497, %v6534, %v6571, %v6608, %v6645, %v6682, %v6719, %v6756, %v6793, %v6830, %v6867, %v6904, %v6941, %v6978, %v7015, %v7052, %v7089, %v7126, %v7163, %v7200, %v7237, %v7274, %v7311, %v7348, %v7385, %v7422, %v7459, %v7496, %v7533, %v7570, %v7607, %v7644, %v7681, %v7718, %v7755, %v7792, %v7829, %v7866, %v7903, %v7940, %v7977, %v8014, %v8051, %v8088, %v8125, %v8162, %v8199, %v8236, %v8273, %v8310, %v8347, %v8384, %v8421, %v8458, %v8495, %v8532, %v8569, %v8606, %v8643, %v8680, %v8717, %v8754, %v8791, %v8828, %v8865, %v8902, %v8939, %v8976, %v9013, %v9050, %v9087, %v9124, %v9161, %v9198, %v9235, %v9272, %v9309, %v9346, %v9383, %v9420, %v9457, %v9494, %v9531, %v9568, %v9605, %v9642, %v9679, %v9716, %v9753, %v9790, %v9827, %v9864, %v9901, %v9938, %v9975, %v10012, %v10049, %v3432, %v3469, %v3506, %v3543, %v3580, %v3617, %v3654, %v3691, %v3728, %v3765, %v3802, %v3839, %v3876, %v3913, %v3950, %v3987, %v4024, %v4061, %v4098, %v4135, %v4172, %v4209, %v4246, %v4283, %v4320, %v4357, %v4394, %v4431, %v4468, %v4505, %v4542, %v4579, %v4616, %v4653, %v4690, %v4727, %v4764, %v4801, %v4838, %v4875, %v4912, %v4949, %v4986, %v5023, %v5060, %v5097, %v5134, %v5171, %v5208, %v5245, %v5282, %v5319, %v5356, %v5393, %v5430, %v5467, %v5504, %v5541, %v5578, %v5615, %v5652, %v5689, %v5726, %v5763, %v5800, %v5837, %v5874, %v5911, %v5948, %v5985, %v6022, %v6059, %v6096, %v6133, %v6170, %v6207, %v6244, %v6281, %v6318, %v6355, %v6392, %v6429, %v6466, %v6503, %v6540, %v6577, %v6614, %v6651, %v6688, %v6725, %v6762, %v6799, %v6836, %v6873, %v6910, %v6947, %v6984, %v7021, %v7058, %v7095, %v7132, %v7169, %v7206, %v7243, %v7280, %v7317, %v7354, %v7391, %v7428, %v7465, %v7502, %v7539, %v7576, %v7613, %v7650, %v7687, %v7724, %v7761, %v7798, %v7835, %v7872, %v7909, %v7946, %v7983, %v8020, %v8057, %v8094, %v8131, %v8168, %v8205, %v8242, %v8279, %v8316, %v8353, %v8390, %v8427, %v8464, %v8501, %v8538, %v8575, %v8612, %v8649, %v8686, %v8723, %v8760, %v8797, %v8834, %v8871, %v8908, %v8945, %v8982, %v9019, %v9056, %v9093, %v9130, %v9167, %v9204, %v9241, %v9278, %v9315, %v9352, %v9389, %v9426, %v9463, %v9500, %v9537, %v9574, %v9611, %v9648, %v9685, %v9722, %v9759, %v9796, %v9833, %v9870, %v9907, %v9944, %v9981, %v10018, %v10055, %loss, %bc1, %bc2 : tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
