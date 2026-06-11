module @m {
  func.func @convnext_adam_train_step(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<f32>, %s0b0nbt: tensor<f32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<f32>, %s0b1nbt: tensor<f32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<f32>, %s0b2nbt: tensor<f32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<f32>, %d0nbt: tensor<f32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<f32>, %s1b0nbt: tensor<f32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<f32>, %s1b1nbt: tensor<f32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<f32>, %s1b2nbt: tensor<f32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<f32>, %d1nbt: tensor<f32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<f32>, %s2b0nbt: tensor<f32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<f32>, %s2b1nbt: tensor<f32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<f32>, %s2b2nbt: tensor<f32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<f32>, %s2b3nbt: tensor<f32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<f32>, %s2b4nbt: tensor<f32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<f32>, %s2b5nbt: tensor<f32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<f32>, %s2b6nbt: tensor<f32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<f32>, %s2b7nbt: tensor<f32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<f32>, %s2b8nbt: tensor<f32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<f32>, %d2nbt: tensor<f32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<f32>, %s3b0nbt: tensor<f32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<f32>, %s3b1nbt: tensor<f32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<f32>, %s3b2nbt: tensor<f32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<f32>, %hnbt: tensor<f32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>, %psWm: tensor<96x3x4x4xf32>, %psbm: tensor<96xf32>, %s0b0dWm: tensor<96x1x7x7xf32>, %s0b0dbm: tensor<96xf32>, %s0b0ngm: tensor<f32>, %s0b0nbtm: tensor<f32>, %s0b0eWm: tensor<384x96x1x1xf32>, %s0b0ebm: tensor<384xf32>, %s0b0pWm: tensor<96x384x1x1xf32>, %s0b0pbm: tensor<96xf32>, %s0b0lgm: tensor<96xf32>, %s0b1dWm: tensor<96x1x7x7xf32>, %s0b1dbm: tensor<96xf32>, %s0b1ngm: tensor<f32>, %s0b1nbtm: tensor<f32>, %s0b1eWm: tensor<384x96x1x1xf32>, %s0b1ebm: tensor<384xf32>, %s0b1pWm: tensor<96x384x1x1xf32>, %s0b1pbm: tensor<96xf32>, %s0b1lgm: tensor<96xf32>, %s0b2dWm: tensor<96x1x7x7xf32>, %s0b2dbm: tensor<96xf32>, %s0b2ngm: tensor<f32>, %s0b2nbtm: tensor<f32>, %s0b2eWm: tensor<384x96x1x1xf32>, %s0b2ebm: tensor<384xf32>, %s0b2pWm: tensor<96x384x1x1xf32>, %s0b2pbm: tensor<96xf32>, %s0b2lgm: tensor<96xf32>, %d0ngm: tensor<f32>, %d0nbtm: tensor<f32>, %d0Wm: tensor<192x96x2x2xf32>, %d0bm: tensor<192xf32>, %s1b0dWm: tensor<192x1x7x7xf32>, %s1b0dbm: tensor<192xf32>, %s1b0ngm: tensor<f32>, %s1b0nbtm: tensor<f32>, %s1b0eWm: tensor<768x192x1x1xf32>, %s1b0ebm: tensor<768xf32>, %s1b0pWm: tensor<192x768x1x1xf32>, %s1b0pbm: tensor<192xf32>, %s1b0lgm: tensor<192xf32>, %s1b1dWm: tensor<192x1x7x7xf32>, %s1b1dbm: tensor<192xf32>, %s1b1ngm: tensor<f32>, %s1b1nbtm: tensor<f32>, %s1b1eWm: tensor<768x192x1x1xf32>, %s1b1ebm: tensor<768xf32>, %s1b1pWm: tensor<192x768x1x1xf32>, %s1b1pbm: tensor<192xf32>, %s1b1lgm: tensor<192xf32>, %s1b2dWm: tensor<192x1x7x7xf32>, %s1b2dbm: tensor<192xf32>, %s1b2ngm: tensor<f32>, %s1b2nbtm: tensor<f32>, %s1b2eWm: tensor<768x192x1x1xf32>, %s1b2ebm: tensor<768xf32>, %s1b2pWm: tensor<192x768x1x1xf32>, %s1b2pbm: tensor<192xf32>, %s1b2lgm: tensor<192xf32>, %d1ngm: tensor<f32>, %d1nbtm: tensor<f32>, %d1Wm: tensor<384x192x2x2xf32>, %d1bm: tensor<384xf32>, %s2b0dWm: tensor<384x1x7x7xf32>, %s2b0dbm: tensor<384xf32>, %s2b0ngm: tensor<f32>, %s2b0nbtm: tensor<f32>, %s2b0eWm: tensor<1536x384x1x1xf32>, %s2b0ebm: tensor<1536xf32>, %s2b0pWm: tensor<384x1536x1x1xf32>, %s2b0pbm: tensor<384xf32>, %s2b0lgm: tensor<384xf32>, %s2b1dWm: tensor<384x1x7x7xf32>, %s2b1dbm: tensor<384xf32>, %s2b1ngm: tensor<f32>, %s2b1nbtm: tensor<f32>, %s2b1eWm: tensor<1536x384x1x1xf32>, %s2b1ebm: tensor<1536xf32>, %s2b1pWm: tensor<384x1536x1x1xf32>, %s2b1pbm: tensor<384xf32>, %s2b1lgm: tensor<384xf32>, %s2b2dWm: tensor<384x1x7x7xf32>, %s2b2dbm: tensor<384xf32>, %s2b2ngm: tensor<f32>, %s2b2nbtm: tensor<f32>, %s2b2eWm: tensor<1536x384x1x1xf32>, %s2b2ebm: tensor<1536xf32>, %s2b2pWm: tensor<384x1536x1x1xf32>, %s2b2pbm: tensor<384xf32>, %s2b2lgm: tensor<384xf32>, %s2b3dWm: tensor<384x1x7x7xf32>, %s2b3dbm: tensor<384xf32>, %s2b3ngm: tensor<f32>, %s2b3nbtm: tensor<f32>, %s2b3eWm: tensor<1536x384x1x1xf32>, %s2b3ebm: tensor<1536xf32>, %s2b3pWm: tensor<384x1536x1x1xf32>, %s2b3pbm: tensor<384xf32>, %s2b3lgm: tensor<384xf32>, %s2b4dWm: tensor<384x1x7x7xf32>, %s2b4dbm: tensor<384xf32>, %s2b4ngm: tensor<f32>, %s2b4nbtm: tensor<f32>, %s2b4eWm: tensor<1536x384x1x1xf32>, %s2b4ebm: tensor<1536xf32>, %s2b4pWm: tensor<384x1536x1x1xf32>, %s2b4pbm: tensor<384xf32>, %s2b4lgm: tensor<384xf32>, %s2b5dWm: tensor<384x1x7x7xf32>, %s2b5dbm: tensor<384xf32>, %s2b5ngm: tensor<f32>, %s2b5nbtm: tensor<f32>, %s2b5eWm: tensor<1536x384x1x1xf32>, %s2b5ebm: tensor<1536xf32>, %s2b5pWm: tensor<384x1536x1x1xf32>, %s2b5pbm: tensor<384xf32>, %s2b5lgm: tensor<384xf32>, %s2b6dWm: tensor<384x1x7x7xf32>, %s2b6dbm: tensor<384xf32>, %s2b6ngm: tensor<f32>, %s2b6nbtm: tensor<f32>, %s2b6eWm: tensor<1536x384x1x1xf32>, %s2b6ebm: tensor<1536xf32>, %s2b6pWm: tensor<384x1536x1x1xf32>, %s2b6pbm: tensor<384xf32>, %s2b6lgm: tensor<384xf32>, %s2b7dWm: tensor<384x1x7x7xf32>, %s2b7dbm: tensor<384xf32>, %s2b7ngm: tensor<f32>, %s2b7nbtm: tensor<f32>, %s2b7eWm: tensor<1536x384x1x1xf32>, %s2b7ebm: tensor<1536xf32>, %s2b7pWm: tensor<384x1536x1x1xf32>, %s2b7pbm: tensor<384xf32>, %s2b7lgm: tensor<384xf32>, %s2b8dWm: tensor<384x1x7x7xf32>, %s2b8dbm: tensor<384xf32>, %s2b8ngm: tensor<f32>, %s2b8nbtm: tensor<f32>, %s2b8eWm: tensor<1536x384x1x1xf32>, %s2b8ebm: tensor<1536xf32>, %s2b8pWm: tensor<384x1536x1x1xf32>, %s2b8pbm: tensor<384xf32>, %s2b8lgm: tensor<384xf32>, %d2ngm: tensor<f32>, %d2nbtm: tensor<f32>, %d2Wm: tensor<768x384x2x2xf32>, %d2bm: tensor<768xf32>, %s3b0dWm: tensor<768x1x7x7xf32>, %s3b0dbm: tensor<768xf32>, %s3b0ngm: tensor<f32>, %s3b0nbtm: tensor<f32>, %s3b0eWm: tensor<3072x768x1x1xf32>, %s3b0ebm: tensor<3072xf32>, %s3b0pWm: tensor<768x3072x1x1xf32>, %s3b0pbm: tensor<768xf32>, %s3b0lgm: tensor<768xf32>, %s3b1dWm: tensor<768x1x7x7xf32>, %s3b1dbm: tensor<768xf32>, %s3b1ngm: tensor<f32>, %s3b1nbtm: tensor<f32>, %s3b1eWm: tensor<3072x768x1x1xf32>, %s3b1ebm: tensor<3072xf32>, %s3b1pWm: tensor<768x3072x1x1xf32>, %s3b1pbm: tensor<768xf32>, %s3b1lgm: tensor<768xf32>, %s3b2dWm: tensor<768x1x7x7xf32>, %s3b2dbm: tensor<768xf32>, %s3b2ngm: tensor<f32>, %s3b2nbtm: tensor<f32>, %s3b2eWm: tensor<3072x768x1x1xf32>, %s3b2ebm: tensor<3072xf32>, %s3b2pWm: tensor<768x3072x1x1xf32>, %s3b2pbm: tensor<768xf32>, %s3b2lgm: tensor<768xf32>, %hngm: tensor<f32>, %hnbtm: tensor<f32>, %Wdm: tensor<768x10xf32>, %bdm: tensor<10xf32>, %psWv: tensor<96x3x4x4xf32>, %psbv: tensor<96xf32>, %s0b0dWv: tensor<96x1x7x7xf32>, %s0b0dbv: tensor<96xf32>, %s0b0ngv: tensor<f32>, %s0b0nbtv: tensor<f32>, %s0b0eWv: tensor<384x96x1x1xf32>, %s0b0ebv: tensor<384xf32>, %s0b0pWv: tensor<96x384x1x1xf32>, %s0b0pbv: tensor<96xf32>, %s0b0lgv: tensor<96xf32>, %s0b1dWv: tensor<96x1x7x7xf32>, %s0b1dbv: tensor<96xf32>, %s0b1ngv: tensor<f32>, %s0b1nbtv: tensor<f32>, %s0b1eWv: tensor<384x96x1x1xf32>, %s0b1ebv: tensor<384xf32>, %s0b1pWv: tensor<96x384x1x1xf32>, %s0b1pbv: tensor<96xf32>, %s0b1lgv: tensor<96xf32>, %s0b2dWv: tensor<96x1x7x7xf32>, %s0b2dbv: tensor<96xf32>, %s0b2ngv: tensor<f32>, %s0b2nbtv: tensor<f32>, %s0b2eWv: tensor<384x96x1x1xf32>, %s0b2ebv: tensor<384xf32>, %s0b2pWv: tensor<96x384x1x1xf32>, %s0b2pbv: tensor<96xf32>, %s0b2lgv: tensor<96xf32>, %d0ngv: tensor<f32>, %d0nbtv: tensor<f32>, %d0Wv: tensor<192x96x2x2xf32>, %d0bv: tensor<192xf32>, %s1b0dWv: tensor<192x1x7x7xf32>, %s1b0dbv: tensor<192xf32>, %s1b0ngv: tensor<f32>, %s1b0nbtv: tensor<f32>, %s1b0eWv: tensor<768x192x1x1xf32>, %s1b0ebv: tensor<768xf32>, %s1b0pWv: tensor<192x768x1x1xf32>, %s1b0pbv: tensor<192xf32>, %s1b0lgv: tensor<192xf32>, %s1b1dWv: tensor<192x1x7x7xf32>, %s1b1dbv: tensor<192xf32>, %s1b1ngv: tensor<f32>, %s1b1nbtv: tensor<f32>, %s1b1eWv: tensor<768x192x1x1xf32>, %s1b1ebv: tensor<768xf32>, %s1b1pWv: tensor<192x768x1x1xf32>, %s1b1pbv: tensor<192xf32>, %s1b1lgv: tensor<192xf32>, %s1b2dWv: tensor<192x1x7x7xf32>, %s1b2dbv: tensor<192xf32>, %s1b2ngv: tensor<f32>, %s1b2nbtv: tensor<f32>, %s1b2eWv: tensor<768x192x1x1xf32>, %s1b2ebv: tensor<768xf32>, %s1b2pWv: tensor<192x768x1x1xf32>, %s1b2pbv: tensor<192xf32>, %s1b2lgv: tensor<192xf32>, %d1ngv: tensor<f32>, %d1nbtv: tensor<f32>, %d1Wv: tensor<384x192x2x2xf32>, %d1bv: tensor<384xf32>, %s2b0dWv: tensor<384x1x7x7xf32>, %s2b0dbv: tensor<384xf32>, %s2b0ngv: tensor<f32>, %s2b0nbtv: tensor<f32>, %s2b0eWv: tensor<1536x384x1x1xf32>, %s2b0ebv: tensor<1536xf32>, %s2b0pWv: tensor<384x1536x1x1xf32>, %s2b0pbv: tensor<384xf32>, %s2b0lgv: tensor<384xf32>, %s2b1dWv: tensor<384x1x7x7xf32>, %s2b1dbv: tensor<384xf32>, %s2b1ngv: tensor<f32>, %s2b1nbtv: tensor<f32>, %s2b1eWv: tensor<1536x384x1x1xf32>, %s2b1ebv: tensor<1536xf32>, %s2b1pWv: tensor<384x1536x1x1xf32>, %s2b1pbv: tensor<384xf32>, %s2b1lgv: tensor<384xf32>, %s2b2dWv: tensor<384x1x7x7xf32>, %s2b2dbv: tensor<384xf32>, %s2b2ngv: tensor<f32>, %s2b2nbtv: tensor<f32>, %s2b2eWv: tensor<1536x384x1x1xf32>, %s2b2ebv: tensor<1536xf32>, %s2b2pWv: tensor<384x1536x1x1xf32>, %s2b2pbv: tensor<384xf32>, %s2b2lgv: tensor<384xf32>, %s2b3dWv: tensor<384x1x7x7xf32>, %s2b3dbv: tensor<384xf32>, %s2b3ngv: tensor<f32>, %s2b3nbtv: tensor<f32>, %s2b3eWv: tensor<1536x384x1x1xf32>, %s2b3ebv: tensor<1536xf32>, %s2b3pWv: tensor<384x1536x1x1xf32>, %s2b3pbv: tensor<384xf32>, %s2b3lgv: tensor<384xf32>, %s2b4dWv: tensor<384x1x7x7xf32>, %s2b4dbv: tensor<384xf32>, %s2b4ngv: tensor<f32>, %s2b4nbtv: tensor<f32>, %s2b4eWv: tensor<1536x384x1x1xf32>, %s2b4ebv: tensor<1536xf32>, %s2b4pWv: tensor<384x1536x1x1xf32>, %s2b4pbv: tensor<384xf32>, %s2b4lgv: tensor<384xf32>, %s2b5dWv: tensor<384x1x7x7xf32>, %s2b5dbv: tensor<384xf32>, %s2b5ngv: tensor<f32>, %s2b5nbtv: tensor<f32>, %s2b5eWv: tensor<1536x384x1x1xf32>, %s2b5ebv: tensor<1536xf32>, %s2b5pWv: tensor<384x1536x1x1xf32>, %s2b5pbv: tensor<384xf32>, %s2b5lgv: tensor<384xf32>, %s2b6dWv: tensor<384x1x7x7xf32>, %s2b6dbv: tensor<384xf32>, %s2b6ngv: tensor<f32>, %s2b6nbtv: tensor<f32>, %s2b6eWv: tensor<1536x384x1x1xf32>, %s2b6ebv: tensor<1536xf32>, %s2b6pWv: tensor<384x1536x1x1xf32>, %s2b6pbv: tensor<384xf32>, %s2b6lgv: tensor<384xf32>, %s2b7dWv: tensor<384x1x7x7xf32>, %s2b7dbv: tensor<384xf32>, %s2b7ngv: tensor<f32>, %s2b7nbtv: tensor<f32>, %s2b7eWv: tensor<1536x384x1x1xf32>, %s2b7ebv: tensor<1536xf32>, %s2b7pWv: tensor<384x1536x1x1xf32>, %s2b7pbv: tensor<384xf32>, %s2b7lgv: tensor<384xf32>, %s2b8dWv: tensor<384x1x7x7xf32>, %s2b8dbv: tensor<384xf32>, %s2b8ngv: tensor<f32>, %s2b8nbtv: tensor<f32>, %s2b8eWv: tensor<1536x384x1x1xf32>, %s2b8ebv: tensor<1536xf32>, %s2b8pWv: tensor<384x1536x1x1xf32>, %s2b8pbv: tensor<384xf32>, %s2b8lgv: tensor<384xf32>, %d2ngv: tensor<f32>, %d2nbtv: tensor<f32>, %d2Wv: tensor<768x384x2x2xf32>, %d2bv: tensor<768xf32>, %s3b0dWv: tensor<768x1x7x7xf32>, %s3b0dbv: tensor<768xf32>, %s3b0ngv: tensor<f32>, %s3b0nbtv: tensor<f32>, %s3b0eWv: tensor<3072x768x1x1xf32>, %s3b0ebv: tensor<3072xf32>, %s3b0pWv: tensor<768x3072x1x1xf32>, %s3b0pbv: tensor<768xf32>, %s3b0lgv: tensor<768xf32>, %s3b1dWv: tensor<768x1x7x7xf32>, %s3b1dbv: tensor<768xf32>, %s3b1ngv: tensor<f32>, %s3b1nbtv: tensor<f32>, %s3b1eWv: tensor<3072x768x1x1xf32>, %s3b1ebv: tensor<3072xf32>, %s3b1pWv: tensor<768x3072x1x1xf32>, %s3b1pbv: tensor<768xf32>, %s3b1lgv: tensor<768xf32>, %s3b2dWv: tensor<768x1x7x7xf32>, %s3b2dbv: tensor<768xf32>, %s3b2ngv: tensor<f32>, %s3b2nbtv: tensor<f32>, %s3b2eWv: tensor<3072x768x1x1xf32>, %s3b2ebv: tensor<3072xf32>, %s3b2pWv: tensor<768x3072x1x1xf32>, %s3b2pbv: tensor<768xf32>, %s3b2lgv: tensor<768xf32>, %hngv: tensor<f32>, %hnbtv: tensor<f32>, %Wdv: tensor<768x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<32x10xf32>) -> (tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %psc = stablehlo.convolution(%xr, %psW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<96x3x4x4xf32>) -> tensor<32x96x56x56xf32>
    %psbb = stablehlo.broadcast_in_dim %psb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %ps = stablehlo.add %psc, %psbb : tensor<32x96x56x56xf32>
    %s0b0dc = stablehlo.convolution(%ps, %s0b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %s0b0dbb = stablehlo.broadcast_in_dim %s0b0db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b0d = stablehlo.add %s0b0dc, %s0b0dbb : tensor<32x96x56x56xf32>
    %s0b0nri = stablehlo.reshape %s0b0d : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b0nnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %s0b0nep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %s0b0nsmr = stablehlo.reduce(%s0b0nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b0nsm = stablehlo.broadcast_in_dim %s0b0nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b0nmu = stablehlo.divide %s0b0nsm, %s0b0nnf : tensor<32x301056xf32>
    %s0b0nxc = stablehlo.subtract %s0b0nri, %s0b0nmu : tensor<32x301056xf32>
    %s0b0nsq = stablehlo.multiply %s0b0nxc, %s0b0nxc : tensor<32x301056xf32>
    %s0b0nvsr = stablehlo.reduce(%s0b0nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b0nvs = stablehlo.broadcast_in_dim %s0b0nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b0nvr = stablehlo.divide %s0b0nvs, %s0b0nnf : tensor<32x301056xf32>
    %s0b0nve = stablehlo.add %s0b0nvr, %s0b0nep : tensor<32x301056xf32>
    %s0b0nistd = stablehlo.rsqrt %s0b0nve : tensor<32x301056xf32>
    %s0b0nxh = stablehlo.multiply %s0b0nxc, %s0b0nistd : tensor<32x301056xf32>
    %s0b0ngb = stablehlo.broadcast_in_dim %s0b0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b0nbtb = stablehlo.broadcast_in_dim %s0b0nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b0ngx = stablehlo.multiply %s0b0nxh, %s0b0ngb : tensor<32x301056xf32>
    %s0b0nfl = stablehlo.add %s0b0ngx, %s0b0nbtb : tensor<32x301056xf32>
    %s0b0n = stablehlo.reshape %s0b0nfl : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %s0b0ec = stablehlo.convolution(%s0b0n, %s0b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %s0b0ebb = stablehlo.broadcast_in_dim %s0b0eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %s0b0e = stablehlo.add %s0b0ec, %s0b0ebb : tensor<32x384x56x56xf32>
    %s0b0gx2 = stablehlo.multiply %s0b0e, %s0b0e : tensor<32x384x56x56xf32>
    %s0b0gx3 = stablehlo.multiply %s0b0gx2, %s0b0e : tensor<32x384x56x56xf32>
    %s0b0gck = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %s0b0gkx3 = stablehlo.multiply %s0b0gck, %s0b0gx3 : tensor<32x384x56x56xf32>
    %s0b0ginn = stablehlo.add %s0b0e, %s0b0gkx3 : tensor<32x384x56x56xf32>
    %s0b0gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %s0b0gu = stablehlo.multiply %s0b0gcs, %s0b0ginn : tensor<32x384x56x56xf32>
    %s0b0gt = stablehlo.tanh %s0b0gu : tensor<32x384x56x56xf32>
    %s0b0gone = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %s0b0gopt = stablehlo.add %s0b0gone, %s0b0gt : tensor<32x384x56x56xf32>
    %s0b0ghalf = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %s0b0ghx = stablehlo.multiply %s0b0ghalf, %s0b0e : tensor<32x384x56x56xf32>
    %s0b0g = stablehlo.multiply %s0b0ghx, %s0b0gopt : tensor<32x384x56x56xf32>
    %s0b0pc = stablehlo.convolution(%s0b0g, %s0b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %s0b0pbb = stablehlo.broadcast_in_dim %s0b0pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b0p = stablehlo.add %s0b0pc, %s0b0pbb : tensor<32x96x56x56xf32>
    %s0b0lsgb = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b0ls = stablehlo.multiply %s0b0p, %s0b0lsgb : tensor<32x96x56x56xf32>
    %s0b0o = stablehlo.add %s0b0ls, %ps : tensor<32x96x56x56xf32>
    %s0b1dc = stablehlo.convolution(%s0b0o, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %s0b1dbb = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b1d = stablehlo.add %s0b1dc, %s0b1dbb : tensor<32x96x56x56xf32>
    %s0b1nri = stablehlo.reshape %s0b1d : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b1nnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %s0b1nep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %s0b1nsmr = stablehlo.reduce(%s0b1nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b1nsm = stablehlo.broadcast_in_dim %s0b1nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b1nmu = stablehlo.divide %s0b1nsm, %s0b1nnf : tensor<32x301056xf32>
    %s0b1nxc = stablehlo.subtract %s0b1nri, %s0b1nmu : tensor<32x301056xf32>
    %s0b1nsq = stablehlo.multiply %s0b1nxc, %s0b1nxc : tensor<32x301056xf32>
    %s0b1nvsr = stablehlo.reduce(%s0b1nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b1nvs = stablehlo.broadcast_in_dim %s0b1nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b1nvr = stablehlo.divide %s0b1nvs, %s0b1nnf : tensor<32x301056xf32>
    %s0b1nve = stablehlo.add %s0b1nvr, %s0b1nep : tensor<32x301056xf32>
    %s0b1nistd = stablehlo.rsqrt %s0b1nve : tensor<32x301056xf32>
    %s0b1nxh = stablehlo.multiply %s0b1nxc, %s0b1nistd : tensor<32x301056xf32>
    %s0b1ngb = stablehlo.broadcast_in_dim %s0b1ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b1nbtb = stablehlo.broadcast_in_dim %s0b1nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b1ngx = stablehlo.multiply %s0b1nxh, %s0b1ngb : tensor<32x301056xf32>
    %s0b1nfl = stablehlo.add %s0b1ngx, %s0b1nbtb : tensor<32x301056xf32>
    %s0b1n = stablehlo.reshape %s0b1nfl : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %s0b1ec = stablehlo.convolution(%s0b1n, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %s0b1ebb = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %s0b1e = stablehlo.add %s0b1ec, %s0b1ebb : tensor<32x384x56x56xf32>
    %s0b1gx2 = stablehlo.multiply %s0b1e, %s0b1e : tensor<32x384x56x56xf32>
    %s0b1gx3 = stablehlo.multiply %s0b1gx2, %s0b1e : tensor<32x384x56x56xf32>
    %s0b1gck = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %s0b1gkx3 = stablehlo.multiply %s0b1gck, %s0b1gx3 : tensor<32x384x56x56xf32>
    %s0b1ginn = stablehlo.add %s0b1e, %s0b1gkx3 : tensor<32x384x56x56xf32>
    %s0b1gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %s0b1gu = stablehlo.multiply %s0b1gcs, %s0b1ginn : tensor<32x384x56x56xf32>
    %s0b1gt = stablehlo.tanh %s0b1gu : tensor<32x384x56x56xf32>
    %s0b1gone = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %s0b1gopt = stablehlo.add %s0b1gone, %s0b1gt : tensor<32x384x56x56xf32>
    %s0b1ghalf = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %s0b1ghx = stablehlo.multiply %s0b1ghalf, %s0b1e : tensor<32x384x56x56xf32>
    %s0b1g = stablehlo.multiply %s0b1ghx, %s0b1gopt : tensor<32x384x56x56xf32>
    %s0b1pc = stablehlo.convolution(%s0b1g, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %s0b1pbb = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b1p = stablehlo.add %s0b1pc, %s0b1pbb : tensor<32x96x56x56xf32>
    %s0b1lsgb = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b1ls = stablehlo.multiply %s0b1p, %s0b1lsgb : tensor<32x96x56x56xf32>
    %s0b1o = stablehlo.add %s0b1ls, %s0b0o : tensor<32x96x56x56xf32>
    %s0b2dc = stablehlo.convolution(%s0b1o, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %s0b2dbb = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b2d = stablehlo.add %s0b2dc, %s0b2dbb : tensor<32x96x56x56xf32>
    %s0b2nri = stablehlo.reshape %s0b2d : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b2nnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %s0b2nep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %s0b2nsmr = stablehlo.reduce(%s0b2nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b2nsm = stablehlo.broadcast_in_dim %s0b2nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b2nmu = stablehlo.divide %s0b2nsm, %s0b2nnf : tensor<32x301056xf32>
    %s0b2nxc = stablehlo.subtract %s0b2nri, %s0b2nmu : tensor<32x301056xf32>
    %s0b2nsq = stablehlo.multiply %s0b2nxc, %s0b2nxc : tensor<32x301056xf32>
    %s0b2nvsr = stablehlo.reduce(%s0b2nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b2nvs = stablehlo.broadcast_in_dim %s0b2nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b2nvr = stablehlo.divide %s0b2nvs, %s0b2nnf : tensor<32x301056xf32>
    %s0b2nve = stablehlo.add %s0b2nvr, %s0b2nep : tensor<32x301056xf32>
    %s0b2nistd = stablehlo.rsqrt %s0b2nve : tensor<32x301056xf32>
    %s0b2nxh = stablehlo.multiply %s0b2nxc, %s0b2nistd : tensor<32x301056xf32>
    %s0b2ngb = stablehlo.broadcast_in_dim %s0b2ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b2nbtb = stablehlo.broadcast_in_dim %s0b2nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b2ngx = stablehlo.multiply %s0b2nxh, %s0b2ngb : tensor<32x301056xf32>
    %s0b2nfl = stablehlo.add %s0b2ngx, %s0b2nbtb : tensor<32x301056xf32>
    %s0b2n = stablehlo.reshape %s0b2nfl : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %s0b2ec = stablehlo.convolution(%s0b2n, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %s0b2ebb = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %s0b2e = stablehlo.add %s0b2ec, %s0b2ebb : tensor<32x384x56x56xf32>
    %s0b2gx2 = stablehlo.multiply %s0b2e, %s0b2e : tensor<32x384x56x56xf32>
    %s0b2gx3 = stablehlo.multiply %s0b2gx2, %s0b2e : tensor<32x384x56x56xf32>
    %s0b2gck = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %s0b2gkx3 = stablehlo.multiply %s0b2gck, %s0b2gx3 : tensor<32x384x56x56xf32>
    %s0b2ginn = stablehlo.add %s0b2e, %s0b2gkx3 : tensor<32x384x56x56xf32>
    %s0b2gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %s0b2gu = stablehlo.multiply %s0b2gcs, %s0b2ginn : tensor<32x384x56x56xf32>
    %s0b2gt = stablehlo.tanh %s0b2gu : tensor<32x384x56x56xf32>
    %s0b2gone = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %s0b2gopt = stablehlo.add %s0b2gone, %s0b2gt : tensor<32x384x56x56xf32>
    %s0b2ghalf = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %s0b2ghx = stablehlo.multiply %s0b2ghalf, %s0b2e : tensor<32x384x56x56xf32>
    %s0b2g = stablehlo.multiply %s0b2ghx, %s0b2gopt : tensor<32x384x56x56xf32>
    %s0b2pc = stablehlo.convolution(%s0b2g, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %s0b2pbb = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b2p = stablehlo.add %s0b2pc, %s0b2pbb : tensor<32x96x56x56xf32>
    %s0b2lsgb = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b2ls = stablehlo.multiply %s0b2p, %s0b2lsgb : tensor<32x96x56x56xf32>
    %s0b2o = stablehlo.add %s0b2ls, %s0b1o : tensor<32x96x56x56xf32>
    %d0nri = stablehlo.reshape %s0b2o : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %d0nnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %d0nep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %d0nsmr = stablehlo.reduce(%d0nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %d0nsm = stablehlo.broadcast_in_dim %d0nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %d0nmu = stablehlo.divide %d0nsm, %d0nnf : tensor<32x301056xf32>
    %d0nxc = stablehlo.subtract %d0nri, %d0nmu : tensor<32x301056xf32>
    %d0nsq = stablehlo.multiply %d0nxc, %d0nxc : tensor<32x301056xf32>
    %d0nvsr = stablehlo.reduce(%d0nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %d0nvs = stablehlo.broadcast_in_dim %d0nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %d0nvr = stablehlo.divide %d0nvs, %d0nnf : tensor<32x301056xf32>
    %d0nve = stablehlo.add %d0nvr, %d0nep : tensor<32x301056xf32>
    %d0nistd = stablehlo.rsqrt %d0nve : tensor<32x301056xf32>
    %d0nxh = stablehlo.multiply %d0nxc, %d0nistd : tensor<32x301056xf32>
    %d0ngb = stablehlo.broadcast_in_dim %d0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %d0nbtb = stablehlo.broadcast_in_dim %d0nbt, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %d0ngx = stablehlo.multiply %d0nxh, %d0ngb : tensor<32x301056xf32>
    %d0nfl = stablehlo.add %d0ngx, %d0nbtb : tensor<32x301056xf32>
    %d0n = stablehlo.reshape %d0nfl : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %d0cc = stablehlo.convolution(%d0n, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %d0cbb = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %d0c = stablehlo.add %d0cc, %d0cbb : tensor<32x192x28x28xf32>
    %s1b0dc = stablehlo.convolution(%d0c, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %s1b0dbb = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b0d = stablehlo.add %s1b0dc, %s1b0dbb : tensor<32x192x28x28xf32>
    %s1b0nri = stablehlo.reshape %s1b0d : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b0nnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %s1b0nep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %s1b0nsmr = stablehlo.reduce(%s1b0nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b0nsm = stablehlo.broadcast_in_dim %s1b0nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b0nmu = stablehlo.divide %s1b0nsm, %s1b0nnf : tensor<32x150528xf32>
    %s1b0nxc = stablehlo.subtract %s1b0nri, %s1b0nmu : tensor<32x150528xf32>
    %s1b0nsq = stablehlo.multiply %s1b0nxc, %s1b0nxc : tensor<32x150528xf32>
    %s1b0nvsr = stablehlo.reduce(%s1b0nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b0nvs = stablehlo.broadcast_in_dim %s1b0nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b0nvr = stablehlo.divide %s1b0nvs, %s1b0nnf : tensor<32x150528xf32>
    %s1b0nve = stablehlo.add %s1b0nvr, %s1b0nep : tensor<32x150528xf32>
    %s1b0nistd = stablehlo.rsqrt %s1b0nve : tensor<32x150528xf32>
    %s1b0nxh = stablehlo.multiply %s1b0nxc, %s1b0nistd : tensor<32x150528xf32>
    %s1b0ngb = stablehlo.broadcast_in_dim %s1b0ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b0nbtb = stablehlo.broadcast_in_dim %s1b0nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b0ngx = stablehlo.multiply %s1b0nxh, %s1b0ngb : tensor<32x150528xf32>
    %s1b0nfl = stablehlo.add %s1b0ngx, %s1b0nbtb : tensor<32x150528xf32>
    %s1b0n = stablehlo.reshape %s1b0nfl : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %s1b0ec = stablehlo.convolution(%s1b0n, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %s1b0ebb = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %s1b0e = stablehlo.add %s1b0ec, %s1b0ebb : tensor<32x768x28x28xf32>
    %s1b0gx2 = stablehlo.multiply %s1b0e, %s1b0e : tensor<32x768x28x28xf32>
    %s1b0gx3 = stablehlo.multiply %s1b0gx2, %s1b0e : tensor<32x768x28x28xf32>
    %s1b0gck = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %s1b0gkx3 = stablehlo.multiply %s1b0gck, %s1b0gx3 : tensor<32x768x28x28xf32>
    %s1b0ginn = stablehlo.add %s1b0e, %s1b0gkx3 : tensor<32x768x28x28xf32>
    %s1b0gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %s1b0gu = stablehlo.multiply %s1b0gcs, %s1b0ginn : tensor<32x768x28x28xf32>
    %s1b0gt = stablehlo.tanh %s1b0gu : tensor<32x768x28x28xf32>
    %s1b0gone = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %s1b0gopt = stablehlo.add %s1b0gone, %s1b0gt : tensor<32x768x28x28xf32>
    %s1b0ghalf = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %s1b0ghx = stablehlo.multiply %s1b0ghalf, %s1b0e : tensor<32x768x28x28xf32>
    %s1b0g = stablehlo.multiply %s1b0ghx, %s1b0gopt : tensor<32x768x28x28xf32>
    %s1b0pc = stablehlo.convolution(%s1b0g, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %s1b0pbb = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b0p = stablehlo.add %s1b0pc, %s1b0pbb : tensor<32x192x28x28xf32>
    %s1b0lsgb = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b0ls = stablehlo.multiply %s1b0p, %s1b0lsgb : tensor<32x192x28x28xf32>
    %s1b0o = stablehlo.add %s1b0ls, %d0c : tensor<32x192x28x28xf32>
    %s1b1dc = stablehlo.convolution(%s1b0o, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %s1b1dbb = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b1d = stablehlo.add %s1b1dc, %s1b1dbb : tensor<32x192x28x28xf32>
    %s1b1nri = stablehlo.reshape %s1b1d : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b1nnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %s1b1nep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %s1b1nsmr = stablehlo.reduce(%s1b1nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b1nsm = stablehlo.broadcast_in_dim %s1b1nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b1nmu = stablehlo.divide %s1b1nsm, %s1b1nnf : tensor<32x150528xf32>
    %s1b1nxc = stablehlo.subtract %s1b1nri, %s1b1nmu : tensor<32x150528xf32>
    %s1b1nsq = stablehlo.multiply %s1b1nxc, %s1b1nxc : tensor<32x150528xf32>
    %s1b1nvsr = stablehlo.reduce(%s1b1nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b1nvs = stablehlo.broadcast_in_dim %s1b1nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b1nvr = stablehlo.divide %s1b1nvs, %s1b1nnf : tensor<32x150528xf32>
    %s1b1nve = stablehlo.add %s1b1nvr, %s1b1nep : tensor<32x150528xf32>
    %s1b1nistd = stablehlo.rsqrt %s1b1nve : tensor<32x150528xf32>
    %s1b1nxh = stablehlo.multiply %s1b1nxc, %s1b1nistd : tensor<32x150528xf32>
    %s1b1ngb = stablehlo.broadcast_in_dim %s1b1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b1nbtb = stablehlo.broadcast_in_dim %s1b1nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b1ngx = stablehlo.multiply %s1b1nxh, %s1b1ngb : tensor<32x150528xf32>
    %s1b1nfl = stablehlo.add %s1b1ngx, %s1b1nbtb : tensor<32x150528xf32>
    %s1b1n = stablehlo.reshape %s1b1nfl : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %s1b1ec = stablehlo.convolution(%s1b1n, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %s1b1ebb = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %s1b1e = stablehlo.add %s1b1ec, %s1b1ebb : tensor<32x768x28x28xf32>
    %s1b1gx2 = stablehlo.multiply %s1b1e, %s1b1e : tensor<32x768x28x28xf32>
    %s1b1gx3 = stablehlo.multiply %s1b1gx2, %s1b1e : tensor<32x768x28x28xf32>
    %s1b1gck = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %s1b1gkx3 = stablehlo.multiply %s1b1gck, %s1b1gx3 : tensor<32x768x28x28xf32>
    %s1b1ginn = stablehlo.add %s1b1e, %s1b1gkx3 : tensor<32x768x28x28xf32>
    %s1b1gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %s1b1gu = stablehlo.multiply %s1b1gcs, %s1b1ginn : tensor<32x768x28x28xf32>
    %s1b1gt = stablehlo.tanh %s1b1gu : tensor<32x768x28x28xf32>
    %s1b1gone = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %s1b1gopt = stablehlo.add %s1b1gone, %s1b1gt : tensor<32x768x28x28xf32>
    %s1b1ghalf = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %s1b1ghx = stablehlo.multiply %s1b1ghalf, %s1b1e : tensor<32x768x28x28xf32>
    %s1b1g = stablehlo.multiply %s1b1ghx, %s1b1gopt : tensor<32x768x28x28xf32>
    %s1b1pc = stablehlo.convolution(%s1b1g, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %s1b1pbb = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b1p = stablehlo.add %s1b1pc, %s1b1pbb : tensor<32x192x28x28xf32>
    %s1b1lsgb = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b1ls = stablehlo.multiply %s1b1p, %s1b1lsgb : tensor<32x192x28x28xf32>
    %s1b1o = stablehlo.add %s1b1ls, %s1b0o : tensor<32x192x28x28xf32>
    %s1b2dc = stablehlo.convolution(%s1b1o, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %s1b2dbb = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b2d = stablehlo.add %s1b2dc, %s1b2dbb : tensor<32x192x28x28xf32>
    %s1b2nri = stablehlo.reshape %s1b2d : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b2nnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %s1b2nep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %s1b2nsmr = stablehlo.reduce(%s1b2nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b2nsm = stablehlo.broadcast_in_dim %s1b2nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b2nmu = stablehlo.divide %s1b2nsm, %s1b2nnf : tensor<32x150528xf32>
    %s1b2nxc = stablehlo.subtract %s1b2nri, %s1b2nmu : tensor<32x150528xf32>
    %s1b2nsq = stablehlo.multiply %s1b2nxc, %s1b2nxc : tensor<32x150528xf32>
    %s1b2nvsr = stablehlo.reduce(%s1b2nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b2nvs = stablehlo.broadcast_in_dim %s1b2nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b2nvr = stablehlo.divide %s1b2nvs, %s1b2nnf : tensor<32x150528xf32>
    %s1b2nve = stablehlo.add %s1b2nvr, %s1b2nep : tensor<32x150528xf32>
    %s1b2nistd = stablehlo.rsqrt %s1b2nve : tensor<32x150528xf32>
    %s1b2nxh = stablehlo.multiply %s1b2nxc, %s1b2nistd : tensor<32x150528xf32>
    %s1b2ngb = stablehlo.broadcast_in_dim %s1b2ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b2nbtb = stablehlo.broadcast_in_dim %s1b2nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b2ngx = stablehlo.multiply %s1b2nxh, %s1b2ngb : tensor<32x150528xf32>
    %s1b2nfl = stablehlo.add %s1b2ngx, %s1b2nbtb : tensor<32x150528xf32>
    %s1b2n = stablehlo.reshape %s1b2nfl : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %s1b2ec = stablehlo.convolution(%s1b2n, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %s1b2ebb = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %s1b2e = stablehlo.add %s1b2ec, %s1b2ebb : tensor<32x768x28x28xf32>
    %s1b2gx2 = stablehlo.multiply %s1b2e, %s1b2e : tensor<32x768x28x28xf32>
    %s1b2gx3 = stablehlo.multiply %s1b2gx2, %s1b2e : tensor<32x768x28x28xf32>
    %s1b2gck = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %s1b2gkx3 = stablehlo.multiply %s1b2gck, %s1b2gx3 : tensor<32x768x28x28xf32>
    %s1b2ginn = stablehlo.add %s1b2e, %s1b2gkx3 : tensor<32x768x28x28xf32>
    %s1b2gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %s1b2gu = stablehlo.multiply %s1b2gcs, %s1b2ginn : tensor<32x768x28x28xf32>
    %s1b2gt = stablehlo.tanh %s1b2gu : tensor<32x768x28x28xf32>
    %s1b2gone = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %s1b2gopt = stablehlo.add %s1b2gone, %s1b2gt : tensor<32x768x28x28xf32>
    %s1b2ghalf = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %s1b2ghx = stablehlo.multiply %s1b2ghalf, %s1b2e : tensor<32x768x28x28xf32>
    %s1b2g = stablehlo.multiply %s1b2ghx, %s1b2gopt : tensor<32x768x28x28xf32>
    %s1b2pc = stablehlo.convolution(%s1b2g, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %s1b2pbb = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b2p = stablehlo.add %s1b2pc, %s1b2pbb : tensor<32x192x28x28xf32>
    %s1b2lsgb = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b2ls = stablehlo.multiply %s1b2p, %s1b2lsgb : tensor<32x192x28x28xf32>
    %s1b2o = stablehlo.add %s1b2ls, %s1b1o : tensor<32x192x28x28xf32>
    %d1nri = stablehlo.reshape %s1b2o : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %d1nnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %d1nep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %d1nsmr = stablehlo.reduce(%d1nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %d1nsm = stablehlo.broadcast_in_dim %d1nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %d1nmu = stablehlo.divide %d1nsm, %d1nnf : tensor<32x150528xf32>
    %d1nxc = stablehlo.subtract %d1nri, %d1nmu : tensor<32x150528xf32>
    %d1nsq = stablehlo.multiply %d1nxc, %d1nxc : tensor<32x150528xf32>
    %d1nvsr = stablehlo.reduce(%d1nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %d1nvs = stablehlo.broadcast_in_dim %d1nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %d1nvr = stablehlo.divide %d1nvs, %d1nnf : tensor<32x150528xf32>
    %d1nve = stablehlo.add %d1nvr, %d1nep : tensor<32x150528xf32>
    %d1nistd = stablehlo.rsqrt %d1nve : tensor<32x150528xf32>
    %d1nxh = stablehlo.multiply %d1nxc, %d1nistd : tensor<32x150528xf32>
    %d1ngb = stablehlo.broadcast_in_dim %d1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %d1nbtb = stablehlo.broadcast_in_dim %d1nbt, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %d1ngx = stablehlo.multiply %d1nxh, %d1ngb : tensor<32x150528xf32>
    %d1nfl = stablehlo.add %d1ngx, %d1nbtb : tensor<32x150528xf32>
    %d1n = stablehlo.reshape %d1nfl : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %d1cc = stablehlo.convolution(%d1n, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %d1cbb = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %d1c = stablehlo.add %d1cc, %d1cbb : tensor<32x384x14x14xf32>
    %s2b0dc = stablehlo.convolution(%d1c, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b0dbb = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b0d = stablehlo.add %s2b0dc, %s2b0dbb : tensor<32x384x14x14xf32>
    %s2b0nri = stablehlo.reshape %s2b0d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b0nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b0nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b0nsmr = stablehlo.reduce(%s2b0nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b0nsm = stablehlo.broadcast_in_dim %s2b0nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b0nmu = stablehlo.divide %s2b0nsm, %s2b0nnf : tensor<32x75264xf32>
    %s2b0nxc = stablehlo.subtract %s2b0nri, %s2b0nmu : tensor<32x75264xf32>
    %s2b0nsq = stablehlo.multiply %s2b0nxc, %s2b0nxc : tensor<32x75264xf32>
    %s2b0nvsr = stablehlo.reduce(%s2b0nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b0nvs = stablehlo.broadcast_in_dim %s2b0nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b0nvr = stablehlo.divide %s2b0nvs, %s2b0nnf : tensor<32x75264xf32>
    %s2b0nve = stablehlo.add %s2b0nvr, %s2b0nep : tensor<32x75264xf32>
    %s2b0nistd = stablehlo.rsqrt %s2b0nve : tensor<32x75264xf32>
    %s2b0nxh = stablehlo.multiply %s2b0nxc, %s2b0nistd : tensor<32x75264xf32>
    %s2b0ngb = stablehlo.broadcast_in_dim %s2b0ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b0nbtb = stablehlo.broadcast_in_dim %s2b0nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b0ngx = stablehlo.multiply %s2b0nxh, %s2b0ngb : tensor<32x75264xf32>
    %s2b0nfl = stablehlo.add %s2b0ngx, %s2b0nbtb : tensor<32x75264xf32>
    %s2b0n = stablehlo.reshape %s2b0nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b0ec = stablehlo.convolution(%s2b0n, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b0ebb = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b0e = stablehlo.add %s2b0ec, %s2b0ebb : tensor<32x1536x14x14xf32>
    %s2b0gx2 = stablehlo.multiply %s2b0e, %s2b0e : tensor<32x1536x14x14xf32>
    %s2b0gx3 = stablehlo.multiply %s2b0gx2, %s2b0e : tensor<32x1536x14x14xf32>
    %s2b0gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b0gkx3 = stablehlo.multiply %s2b0gck, %s2b0gx3 : tensor<32x1536x14x14xf32>
    %s2b0ginn = stablehlo.add %s2b0e, %s2b0gkx3 : tensor<32x1536x14x14xf32>
    %s2b0gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b0gu = stablehlo.multiply %s2b0gcs, %s2b0ginn : tensor<32x1536x14x14xf32>
    %s2b0gt = stablehlo.tanh %s2b0gu : tensor<32x1536x14x14xf32>
    %s2b0gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b0gopt = stablehlo.add %s2b0gone, %s2b0gt : tensor<32x1536x14x14xf32>
    %s2b0ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b0ghx = stablehlo.multiply %s2b0ghalf, %s2b0e : tensor<32x1536x14x14xf32>
    %s2b0g = stablehlo.multiply %s2b0ghx, %s2b0gopt : tensor<32x1536x14x14xf32>
    %s2b0pc = stablehlo.convolution(%s2b0g, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b0pbb = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b0p = stablehlo.add %s2b0pc, %s2b0pbb : tensor<32x384x14x14xf32>
    %s2b0lsgb = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b0ls = stablehlo.multiply %s2b0p, %s2b0lsgb : tensor<32x384x14x14xf32>
    %s2b0o = stablehlo.add %s2b0ls, %d1c : tensor<32x384x14x14xf32>
    %s2b1dc = stablehlo.convolution(%s2b0o, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b1dbb = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b1d = stablehlo.add %s2b1dc, %s2b1dbb : tensor<32x384x14x14xf32>
    %s2b1nri = stablehlo.reshape %s2b1d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b1nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b1nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b1nsmr = stablehlo.reduce(%s2b1nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b1nsm = stablehlo.broadcast_in_dim %s2b1nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b1nmu = stablehlo.divide %s2b1nsm, %s2b1nnf : tensor<32x75264xf32>
    %s2b1nxc = stablehlo.subtract %s2b1nri, %s2b1nmu : tensor<32x75264xf32>
    %s2b1nsq = stablehlo.multiply %s2b1nxc, %s2b1nxc : tensor<32x75264xf32>
    %s2b1nvsr = stablehlo.reduce(%s2b1nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b1nvs = stablehlo.broadcast_in_dim %s2b1nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b1nvr = stablehlo.divide %s2b1nvs, %s2b1nnf : tensor<32x75264xf32>
    %s2b1nve = stablehlo.add %s2b1nvr, %s2b1nep : tensor<32x75264xf32>
    %s2b1nistd = stablehlo.rsqrt %s2b1nve : tensor<32x75264xf32>
    %s2b1nxh = stablehlo.multiply %s2b1nxc, %s2b1nistd : tensor<32x75264xf32>
    %s2b1ngb = stablehlo.broadcast_in_dim %s2b1ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b1nbtb = stablehlo.broadcast_in_dim %s2b1nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b1ngx = stablehlo.multiply %s2b1nxh, %s2b1ngb : tensor<32x75264xf32>
    %s2b1nfl = stablehlo.add %s2b1ngx, %s2b1nbtb : tensor<32x75264xf32>
    %s2b1n = stablehlo.reshape %s2b1nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b1ec = stablehlo.convolution(%s2b1n, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b1ebb = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b1e = stablehlo.add %s2b1ec, %s2b1ebb : tensor<32x1536x14x14xf32>
    %s2b1gx2 = stablehlo.multiply %s2b1e, %s2b1e : tensor<32x1536x14x14xf32>
    %s2b1gx3 = stablehlo.multiply %s2b1gx2, %s2b1e : tensor<32x1536x14x14xf32>
    %s2b1gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b1gkx3 = stablehlo.multiply %s2b1gck, %s2b1gx3 : tensor<32x1536x14x14xf32>
    %s2b1ginn = stablehlo.add %s2b1e, %s2b1gkx3 : tensor<32x1536x14x14xf32>
    %s2b1gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b1gu = stablehlo.multiply %s2b1gcs, %s2b1ginn : tensor<32x1536x14x14xf32>
    %s2b1gt = stablehlo.tanh %s2b1gu : tensor<32x1536x14x14xf32>
    %s2b1gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b1gopt = stablehlo.add %s2b1gone, %s2b1gt : tensor<32x1536x14x14xf32>
    %s2b1ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b1ghx = stablehlo.multiply %s2b1ghalf, %s2b1e : tensor<32x1536x14x14xf32>
    %s2b1g = stablehlo.multiply %s2b1ghx, %s2b1gopt : tensor<32x1536x14x14xf32>
    %s2b1pc = stablehlo.convolution(%s2b1g, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b1pbb = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b1p = stablehlo.add %s2b1pc, %s2b1pbb : tensor<32x384x14x14xf32>
    %s2b1lsgb = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b1ls = stablehlo.multiply %s2b1p, %s2b1lsgb : tensor<32x384x14x14xf32>
    %s2b1o = stablehlo.add %s2b1ls, %s2b0o : tensor<32x384x14x14xf32>
    %s2b2dc = stablehlo.convolution(%s2b1o, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b2dbb = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b2d = stablehlo.add %s2b2dc, %s2b2dbb : tensor<32x384x14x14xf32>
    %s2b2nri = stablehlo.reshape %s2b2d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b2nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b2nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b2nsmr = stablehlo.reduce(%s2b2nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b2nsm = stablehlo.broadcast_in_dim %s2b2nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b2nmu = stablehlo.divide %s2b2nsm, %s2b2nnf : tensor<32x75264xf32>
    %s2b2nxc = stablehlo.subtract %s2b2nri, %s2b2nmu : tensor<32x75264xf32>
    %s2b2nsq = stablehlo.multiply %s2b2nxc, %s2b2nxc : tensor<32x75264xf32>
    %s2b2nvsr = stablehlo.reduce(%s2b2nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b2nvs = stablehlo.broadcast_in_dim %s2b2nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b2nvr = stablehlo.divide %s2b2nvs, %s2b2nnf : tensor<32x75264xf32>
    %s2b2nve = stablehlo.add %s2b2nvr, %s2b2nep : tensor<32x75264xf32>
    %s2b2nistd = stablehlo.rsqrt %s2b2nve : tensor<32x75264xf32>
    %s2b2nxh = stablehlo.multiply %s2b2nxc, %s2b2nistd : tensor<32x75264xf32>
    %s2b2ngb = stablehlo.broadcast_in_dim %s2b2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b2nbtb = stablehlo.broadcast_in_dim %s2b2nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b2ngx = stablehlo.multiply %s2b2nxh, %s2b2ngb : tensor<32x75264xf32>
    %s2b2nfl = stablehlo.add %s2b2ngx, %s2b2nbtb : tensor<32x75264xf32>
    %s2b2n = stablehlo.reshape %s2b2nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b2ec = stablehlo.convolution(%s2b2n, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b2ebb = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b2e = stablehlo.add %s2b2ec, %s2b2ebb : tensor<32x1536x14x14xf32>
    %s2b2gx2 = stablehlo.multiply %s2b2e, %s2b2e : tensor<32x1536x14x14xf32>
    %s2b2gx3 = stablehlo.multiply %s2b2gx2, %s2b2e : tensor<32x1536x14x14xf32>
    %s2b2gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b2gkx3 = stablehlo.multiply %s2b2gck, %s2b2gx3 : tensor<32x1536x14x14xf32>
    %s2b2ginn = stablehlo.add %s2b2e, %s2b2gkx3 : tensor<32x1536x14x14xf32>
    %s2b2gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b2gu = stablehlo.multiply %s2b2gcs, %s2b2ginn : tensor<32x1536x14x14xf32>
    %s2b2gt = stablehlo.tanh %s2b2gu : tensor<32x1536x14x14xf32>
    %s2b2gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b2gopt = stablehlo.add %s2b2gone, %s2b2gt : tensor<32x1536x14x14xf32>
    %s2b2ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b2ghx = stablehlo.multiply %s2b2ghalf, %s2b2e : tensor<32x1536x14x14xf32>
    %s2b2g = stablehlo.multiply %s2b2ghx, %s2b2gopt : tensor<32x1536x14x14xf32>
    %s2b2pc = stablehlo.convolution(%s2b2g, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b2pbb = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b2p = stablehlo.add %s2b2pc, %s2b2pbb : tensor<32x384x14x14xf32>
    %s2b2lsgb = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b2ls = stablehlo.multiply %s2b2p, %s2b2lsgb : tensor<32x384x14x14xf32>
    %s2b2o = stablehlo.add %s2b2ls, %s2b1o : tensor<32x384x14x14xf32>
    %s2b3dc = stablehlo.convolution(%s2b2o, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b3dbb = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b3d = stablehlo.add %s2b3dc, %s2b3dbb : tensor<32x384x14x14xf32>
    %s2b3nri = stablehlo.reshape %s2b3d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b3nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b3nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b3nsmr = stablehlo.reduce(%s2b3nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b3nsm = stablehlo.broadcast_in_dim %s2b3nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b3nmu = stablehlo.divide %s2b3nsm, %s2b3nnf : tensor<32x75264xf32>
    %s2b3nxc = stablehlo.subtract %s2b3nri, %s2b3nmu : tensor<32x75264xf32>
    %s2b3nsq = stablehlo.multiply %s2b3nxc, %s2b3nxc : tensor<32x75264xf32>
    %s2b3nvsr = stablehlo.reduce(%s2b3nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b3nvs = stablehlo.broadcast_in_dim %s2b3nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b3nvr = stablehlo.divide %s2b3nvs, %s2b3nnf : tensor<32x75264xf32>
    %s2b3nve = stablehlo.add %s2b3nvr, %s2b3nep : tensor<32x75264xf32>
    %s2b3nistd = stablehlo.rsqrt %s2b3nve : tensor<32x75264xf32>
    %s2b3nxh = stablehlo.multiply %s2b3nxc, %s2b3nistd : tensor<32x75264xf32>
    %s2b3ngb = stablehlo.broadcast_in_dim %s2b3ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b3nbtb = stablehlo.broadcast_in_dim %s2b3nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b3ngx = stablehlo.multiply %s2b3nxh, %s2b3ngb : tensor<32x75264xf32>
    %s2b3nfl = stablehlo.add %s2b3ngx, %s2b3nbtb : tensor<32x75264xf32>
    %s2b3n = stablehlo.reshape %s2b3nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b3ec = stablehlo.convolution(%s2b3n, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b3ebb = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b3e = stablehlo.add %s2b3ec, %s2b3ebb : tensor<32x1536x14x14xf32>
    %s2b3gx2 = stablehlo.multiply %s2b3e, %s2b3e : tensor<32x1536x14x14xf32>
    %s2b3gx3 = stablehlo.multiply %s2b3gx2, %s2b3e : tensor<32x1536x14x14xf32>
    %s2b3gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b3gkx3 = stablehlo.multiply %s2b3gck, %s2b3gx3 : tensor<32x1536x14x14xf32>
    %s2b3ginn = stablehlo.add %s2b3e, %s2b3gkx3 : tensor<32x1536x14x14xf32>
    %s2b3gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b3gu = stablehlo.multiply %s2b3gcs, %s2b3ginn : tensor<32x1536x14x14xf32>
    %s2b3gt = stablehlo.tanh %s2b3gu : tensor<32x1536x14x14xf32>
    %s2b3gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b3gopt = stablehlo.add %s2b3gone, %s2b3gt : tensor<32x1536x14x14xf32>
    %s2b3ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b3ghx = stablehlo.multiply %s2b3ghalf, %s2b3e : tensor<32x1536x14x14xf32>
    %s2b3g = stablehlo.multiply %s2b3ghx, %s2b3gopt : tensor<32x1536x14x14xf32>
    %s2b3pc = stablehlo.convolution(%s2b3g, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b3pbb = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b3p = stablehlo.add %s2b3pc, %s2b3pbb : tensor<32x384x14x14xf32>
    %s2b3lsgb = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b3ls = stablehlo.multiply %s2b3p, %s2b3lsgb : tensor<32x384x14x14xf32>
    %s2b3o = stablehlo.add %s2b3ls, %s2b2o : tensor<32x384x14x14xf32>
    %s2b4dc = stablehlo.convolution(%s2b3o, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b4dbb = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b4d = stablehlo.add %s2b4dc, %s2b4dbb : tensor<32x384x14x14xf32>
    %s2b4nri = stablehlo.reshape %s2b4d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b4nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b4nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b4nsmr = stablehlo.reduce(%s2b4nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b4nsm = stablehlo.broadcast_in_dim %s2b4nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b4nmu = stablehlo.divide %s2b4nsm, %s2b4nnf : tensor<32x75264xf32>
    %s2b4nxc = stablehlo.subtract %s2b4nri, %s2b4nmu : tensor<32x75264xf32>
    %s2b4nsq = stablehlo.multiply %s2b4nxc, %s2b4nxc : tensor<32x75264xf32>
    %s2b4nvsr = stablehlo.reduce(%s2b4nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b4nvs = stablehlo.broadcast_in_dim %s2b4nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b4nvr = stablehlo.divide %s2b4nvs, %s2b4nnf : tensor<32x75264xf32>
    %s2b4nve = stablehlo.add %s2b4nvr, %s2b4nep : tensor<32x75264xf32>
    %s2b4nistd = stablehlo.rsqrt %s2b4nve : tensor<32x75264xf32>
    %s2b4nxh = stablehlo.multiply %s2b4nxc, %s2b4nistd : tensor<32x75264xf32>
    %s2b4ngb = stablehlo.broadcast_in_dim %s2b4ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b4nbtb = stablehlo.broadcast_in_dim %s2b4nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b4ngx = stablehlo.multiply %s2b4nxh, %s2b4ngb : tensor<32x75264xf32>
    %s2b4nfl = stablehlo.add %s2b4ngx, %s2b4nbtb : tensor<32x75264xf32>
    %s2b4n = stablehlo.reshape %s2b4nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b4ec = stablehlo.convolution(%s2b4n, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b4ebb = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b4e = stablehlo.add %s2b4ec, %s2b4ebb : tensor<32x1536x14x14xf32>
    %s2b4gx2 = stablehlo.multiply %s2b4e, %s2b4e : tensor<32x1536x14x14xf32>
    %s2b4gx3 = stablehlo.multiply %s2b4gx2, %s2b4e : tensor<32x1536x14x14xf32>
    %s2b4gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b4gkx3 = stablehlo.multiply %s2b4gck, %s2b4gx3 : tensor<32x1536x14x14xf32>
    %s2b4ginn = stablehlo.add %s2b4e, %s2b4gkx3 : tensor<32x1536x14x14xf32>
    %s2b4gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b4gu = stablehlo.multiply %s2b4gcs, %s2b4ginn : tensor<32x1536x14x14xf32>
    %s2b4gt = stablehlo.tanh %s2b4gu : tensor<32x1536x14x14xf32>
    %s2b4gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b4gopt = stablehlo.add %s2b4gone, %s2b4gt : tensor<32x1536x14x14xf32>
    %s2b4ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b4ghx = stablehlo.multiply %s2b4ghalf, %s2b4e : tensor<32x1536x14x14xf32>
    %s2b4g = stablehlo.multiply %s2b4ghx, %s2b4gopt : tensor<32x1536x14x14xf32>
    %s2b4pc = stablehlo.convolution(%s2b4g, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b4pbb = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b4p = stablehlo.add %s2b4pc, %s2b4pbb : tensor<32x384x14x14xf32>
    %s2b4lsgb = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b4ls = stablehlo.multiply %s2b4p, %s2b4lsgb : tensor<32x384x14x14xf32>
    %s2b4o = stablehlo.add %s2b4ls, %s2b3o : tensor<32x384x14x14xf32>
    %s2b5dc = stablehlo.convolution(%s2b4o, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b5dbb = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b5d = stablehlo.add %s2b5dc, %s2b5dbb : tensor<32x384x14x14xf32>
    %s2b5nri = stablehlo.reshape %s2b5d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b5nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b5nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b5nsmr = stablehlo.reduce(%s2b5nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b5nsm = stablehlo.broadcast_in_dim %s2b5nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b5nmu = stablehlo.divide %s2b5nsm, %s2b5nnf : tensor<32x75264xf32>
    %s2b5nxc = stablehlo.subtract %s2b5nri, %s2b5nmu : tensor<32x75264xf32>
    %s2b5nsq = stablehlo.multiply %s2b5nxc, %s2b5nxc : tensor<32x75264xf32>
    %s2b5nvsr = stablehlo.reduce(%s2b5nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b5nvs = stablehlo.broadcast_in_dim %s2b5nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b5nvr = stablehlo.divide %s2b5nvs, %s2b5nnf : tensor<32x75264xf32>
    %s2b5nve = stablehlo.add %s2b5nvr, %s2b5nep : tensor<32x75264xf32>
    %s2b5nistd = stablehlo.rsqrt %s2b5nve : tensor<32x75264xf32>
    %s2b5nxh = stablehlo.multiply %s2b5nxc, %s2b5nistd : tensor<32x75264xf32>
    %s2b5ngb = stablehlo.broadcast_in_dim %s2b5ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b5nbtb = stablehlo.broadcast_in_dim %s2b5nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b5ngx = stablehlo.multiply %s2b5nxh, %s2b5ngb : tensor<32x75264xf32>
    %s2b5nfl = stablehlo.add %s2b5ngx, %s2b5nbtb : tensor<32x75264xf32>
    %s2b5n = stablehlo.reshape %s2b5nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b5ec = stablehlo.convolution(%s2b5n, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b5ebb = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b5e = stablehlo.add %s2b5ec, %s2b5ebb : tensor<32x1536x14x14xf32>
    %s2b5gx2 = stablehlo.multiply %s2b5e, %s2b5e : tensor<32x1536x14x14xf32>
    %s2b5gx3 = stablehlo.multiply %s2b5gx2, %s2b5e : tensor<32x1536x14x14xf32>
    %s2b5gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b5gkx3 = stablehlo.multiply %s2b5gck, %s2b5gx3 : tensor<32x1536x14x14xf32>
    %s2b5ginn = stablehlo.add %s2b5e, %s2b5gkx3 : tensor<32x1536x14x14xf32>
    %s2b5gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b5gu = stablehlo.multiply %s2b5gcs, %s2b5ginn : tensor<32x1536x14x14xf32>
    %s2b5gt = stablehlo.tanh %s2b5gu : tensor<32x1536x14x14xf32>
    %s2b5gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b5gopt = stablehlo.add %s2b5gone, %s2b5gt : tensor<32x1536x14x14xf32>
    %s2b5ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b5ghx = stablehlo.multiply %s2b5ghalf, %s2b5e : tensor<32x1536x14x14xf32>
    %s2b5g = stablehlo.multiply %s2b5ghx, %s2b5gopt : tensor<32x1536x14x14xf32>
    %s2b5pc = stablehlo.convolution(%s2b5g, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b5pbb = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b5p = stablehlo.add %s2b5pc, %s2b5pbb : tensor<32x384x14x14xf32>
    %s2b5lsgb = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b5ls = stablehlo.multiply %s2b5p, %s2b5lsgb : tensor<32x384x14x14xf32>
    %s2b5o = stablehlo.add %s2b5ls, %s2b4o : tensor<32x384x14x14xf32>
    %s2b6dc = stablehlo.convolution(%s2b5o, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b6dbb = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b6d = stablehlo.add %s2b6dc, %s2b6dbb : tensor<32x384x14x14xf32>
    %s2b6nri = stablehlo.reshape %s2b6d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b6nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b6nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b6nsmr = stablehlo.reduce(%s2b6nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b6nsm = stablehlo.broadcast_in_dim %s2b6nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b6nmu = stablehlo.divide %s2b6nsm, %s2b6nnf : tensor<32x75264xf32>
    %s2b6nxc = stablehlo.subtract %s2b6nri, %s2b6nmu : tensor<32x75264xf32>
    %s2b6nsq = stablehlo.multiply %s2b6nxc, %s2b6nxc : tensor<32x75264xf32>
    %s2b6nvsr = stablehlo.reduce(%s2b6nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b6nvs = stablehlo.broadcast_in_dim %s2b6nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b6nvr = stablehlo.divide %s2b6nvs, %s2b6nnf : tensor<32x75264xf32>
    %s2b6nve = stablehlo.add %s2b6nvr, %s2b6nep : tensor<32x75264xf32>
    %s2b6nistd = stablehlo.rsqrt %s2b6nve : tensor<32x75264xf32>
    %s2b6nxh = stablehlo.multiply %s2b6nxc, %s2b6nistd : tensor<32x75264xf32>
    %s2b6ngb = stablehlo.broadcast_in_dim %s2b6ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b6nbtb = stablehlo.broadcast_in_dim %s2b6nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b6ngx = stablehlo.multiply %s2b6nxh, %s2b6ngb : tensor<32x75264xf32>
    %s2b6nfl = stablehlo.add %s2b6ngx, %s2b6nbtb : tensor<32x75264xf32>
    %s2b6n = stablehlo.reshape %s2b6nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b6ec = stablehlo.convolution(%s2b6n, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b6ebb = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b6e = stablehlo.add %s2b6ec, %s2b6ebb : tensor<32x1536x14x14xf32>
    %s2b6gx2 = stablehlo.multiply %s2b6e, %s2b6e : tensor<32x1536x14x14xf32>
    %s2b6gx3 = stablehlo.multiply %s2b6gx2, %s2b6e : tensor<32x1536x14x14xf32>
    %s2b6gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b6gkx3 = stablehlo.multiply %s2b6gck, %s2b6gx3 : tensor<32x1536x14x14xf32>
    %s2b6ginn = stablehlo.add %s2b6e, %s2b6gkx3 : tensor<32x1536x14x14xf32>
    %s2b6gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b6gu = stablehlo.multiply %s2b6gcs, %s2b6ginn : tensor<32x1536x14x14xf32>
    %s2b6gt = stablehlo.tanh %s2b6gu : tensor<32x1536x14x14xf32>
    %s2b6gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b6gopt = stablehlo.add %s2b6gone, %s2b6gt : tensor<32x1536x14x14xf32>
    %s2b6ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b6ghx = stablehlo.multiply %s2b6ghalf, %s2b6e : tensor<32x1536x14x14xf32>
    %s2b6g = stablehlo.multiply %s2b6ghx, %s2b6gopt : tensor<32x1536x14x14xf32>
    %s2b6pc = stablehlo.convolution(%s2b6g, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b6pbb = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b6p = stablehlo.add %s2b6pc, %s2b6pbb : tensor<32x384x14x14xf32>
    %s2b6lsgb = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b6ls = stablehlo.multiply %s2b6p, %s2b6lsgb : tensor<32x384x14x14xf32>
    %s2b6o = stablehlo.add %s2b6ls, %s2b5o : tensor<32x384x14x14xf32>
    %s2b7dc = stablehlo.convolution(%s2b6o, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b7dbb = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b7d = stablehlo.add %s2b7dc, %s2b7dbb : tensor<32x384x14x14xf32>
    %s2b7nri = stablehlo.reshape %s2b7d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b7nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b7nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b7nsmr = stablehlo.reduce(%s2b7nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b7nsm = stablehlo.broadcast_in_dim %s2b7nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b7nmu = stablehlo.divide %s2b7nsm, %s2b7nnf : tensor<32x75264xf32>
    %s2b7nxc = stablehlo.subtract %s2b7nri, %s2b7nmu : tensor<32x75264xf32>
    %s2b7nsq = stablehlo.multiply %s2b7nxc, %s2b7nxc : tensor<32x75264xf32>
    %s2b7nvsr = stablehlo.reduce(%s2b7nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b7nvs = stablehlo.broadcast_in_dim %s2b7nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b7nvr = stablehlo.divide %s2b7nvs, %s2b7nnf : tensor<32x75264xf32>
    %s2b7nve = stablehlo.add %s2b7nvr, %s2b7nep : tensor<32x75264xf32>
    %s2b7nistd = stablehlo.rsqrt %s2b7nve : tensor<32x75264xf32>
    %s2b7nxh = stablehlo.multiply %s2b7nxc, %s2b7nistd : tensor<32x75264xf32>
    %s2b7ngb = stablehlo.broadcast_in_dim %s2b7ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b7nbtb = stablehlo.broadcast_in_dim %s2b7nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b7ngx = stablehlo.multiply %s2b7nxh, %s2b7ngb : tensor<32x75264xf32>
    %s2b7nfl = stablehlo.add %s2b7ngx, %s2b7nbtb : tensor<32x75264xf32>
    %s2b7n = stablehlo.reshape %s2b7nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b7ec = stablehlo.convolution(%s2b7n, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b7ebb = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b7e = stablehlo.add %s2b7ec, %s2b7ebb : tensor<32x1536x14x14xf32>
    %s2b7gx2 = stablehlo.multiply %s2b7e, %s2b7e : tensor<32x1536x14x14xf32>
    %s2b7gx3 = stablehlo.multiply %s2b7gx2, %s2b7e : tensor<32x1536x14x14xf32>
    %s2b7gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b7gkx3 = stablehlo.multiply %s2b7gck, %s2b7gx3 : tensor<32x1536x14x14xf32>
    %s2b7ginn = stablehlo.add %s2b7e, %s2b7gkx3 : tensor<32x1536x14x14xf32>
    %s2b7gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b7gu = stablehlo.multiply %s2b7gcs, %s2b7ginn : tensor<32x1536x14x14xf32>
    %s2b7gt = stablehlo.tanh %s2b7gu : tensor<32x1536x14x14xf32>
    %s2b7gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b7gopt = stablehlo.add %s2b7gone, %s2b7gt : tensor<32x1536x14x14xf32>
    %s2b7ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b7ghx = stablehlo.multiply %s2b7ghalf, %s2b7e : tensor<32x1536x14x14xf32>
    %s2b7g = stablehlo.multiply %s2b7ghx, %s2b7gopt : tensor<32x1536x14x14xf32>
    %s2b7pc = stablehlo.convolution(%s2b7g, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b7pbb = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b7p = stablehlo.add %s2b7pc, %s2b7pbb : tensor<32x384x14x14xf32>
    %s2b7lsgb = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b7ls = stablehlo.multiply %s2b7p, %s2b7lsgb : tensor<32x384x14x14xf32>
    %s2b7o = stablehlo.add %s2b7ls, %s2b6o : tensor<32x384x14x14xf32>
    %s2b8dc = stablehlo.convolution(%s2b7o, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b8dbb = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b8d = stablehlo.add %s2b8dc, %s2b8dbb : tensor<32x384x14x14xf32>
    %s2b8nri = stablehlo.reshape %s2b8d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b8nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b8nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b8nsmr = stablehlo.reduce(%s2b8nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b8nsm = stablehlo.broadcast_in_dim %s2b8nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b8nmu = stablehlo.divide %s2b8nsm, %s2b8nnf : tensor<32x75264xf32>
    %s2b8nxc = stablehlo.subtract %s2b8nri, %s2b8nmu : tensor<32x75264xf32>
    %s2b8nsq = stablehlo.multiply %s2b8nxc, %s2b8nxc : tensor<32x75264xf32>
    %s2b8nvsr = stablehlo.reduce(%s2b8nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b8nvs = stablehlo.broadcast_in_dim %s2b8nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b8nvr = stablehlo.divide %s2b8nvs, %s2b8nnf : tensor<32x75264xf32>
    %s2b8nve = stablehlo.add %s2b8nvr, %s2b8nep : tensor<32x75264xf32>
    %s2b8nistd = stablehlo.rsqrt %s2b8nve : tensor<32x75264xf32>
    %s2b8nxh = stablehlo.multiply %s2b8nxc, %s2b8nistd : tensor<32x75264xf32>
    %s2b8ngb = stablehlo.broadcast_in_dim %s2b8ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b8nbtb = stablehlo.broadcast_in_dim %s2b8nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b8ngx = stablehlo.multiply %s2b8nxh, %s2b8ngb : tensor<32x75264xf32>
    %s2b8nfl = stablehlo.add %s2b8ngx, %s2b8nbtb : tensor<32x75264xf32>
    %s2b8n = stablehlo.reshape %s2b8nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b8ec = stablehlo.convolution(%s2b8n, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b8ebb = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %s2b8e = stablehlo.add %s2b8ec, %s2b8ebb : tensor<32x1536x14x14xf32>
    %s2b8gx2 = stablehlo.multiply %s2b8e, %s2b8e : tensor<32x1536x14x14xf32>
    %s2b8gx3 = stablehlo.multiply %s2b8gx2, %s2b8e : tensor<32x1536x14x14xf32>
    %s2b8gck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b8gkx3 = stablehlo.multiply %s2b8gck, %s2b8gx3 : tensor<32x1536x14x14xf32>
    %s2b8ginn = stablehlo.add %s2b8e, %s2b8gkx3 : tensor<32x1536x14x14xf32>
    %s2b8gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b8gu = stablehlo.multiply %s2b8gcs, %s2b8ginn : tensor<32x1536x14x14xf32>
    %s2b8gt = stablehlo.tanh %s2b8gu : tensor<32x1536x14x14xf32>
    %s2b8gone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b8gopt = stablehlo.add %s2b8gone, %s2b8gt : tensor<32x1536x14x14xf32>
    %s2b8ghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b8ghx = stablehlo.multiply %s2b8ghalf, %s2b8e : tensor<32x1536x14x14xf32>
    %s2b8g = stablehlo.multiply %s2b8ghx, %s2b8gopt : tensor<32x1536x14x14xf32>
    %s2b8pc = stablehlo.convolution(%s2b8g, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b8pbb = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b8p = stablehlo.add %s2b8pc, %s2b8pbb : tensor<32x384x14x14xf32>
    %s2b8lsgb = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b8ls = stablehlo.multiply %s2b8p, %s2b8lsgb : tensor<32x384x14x14xf32>
    %s2b8o = stablehlo.add %s2b8ls, %s2b7o : tensor<32x384x14x14xf32>
    %d2nri = stablehlo.reshape %s2b8o : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %d2nnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %d2nep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %d2nsmr = stablehlo.reduce(%d2nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %d2nsm = stablehlo.broadcast_in_dim %d2nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %d2nmu = stablehlo.divide %d2nsm, %d2nnf : tensor<32x75264xf32>
    %d2nxc = stablehlo.subtract %d2nri, %d2nmu : tensor<32x75264xf32>
    %d2nsq = stablehlo.multiply %d2nxc, %d2nxc : tensor<32x75264xf32>
    %d2nvsr = stablehlo.reduce(%d2nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %d2nvs = stablehlo.broadcast_in_dim %d2nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %d2nvr = stablehlo.divide %d2nvs, %d2nnf : tensor<32x75264xf32>
    %d2nve = stablehlo.add %d2nvr, %d2nep : tensor<32x75264xf32>
    %d2nistd = stablehlo.rsqrt %d2nve : tensor<32x75264xf32>
    %d2nxh = stablehlo.multiply %d2nxc, %d2nistd : tensor<32x75264xf32>
    %d2ngb = stablehlo.broadcast_in_dim %d2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %d2nbtb = stablehlo.broadcast_in_dim %d2nbt, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %d2ngx = stablehlo.multiply %d2nxh, %d2ngb : tensor<32x75264xf32>
    %d2nfl = stablehlo.add %d2ngx, %d2nbtb : tensor<32x75264xf32>
    %d2n = stablehlo.reshape %d2nfl : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %d2cc = stablehlo.convolution(%d2n, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %d2cbb = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %d2c = stablehlo.add %d2cc, %d2cbb : tensor<32x768x7x7xf32>
    %s3b0dc = stablehlo.convolution(%d2c, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %s3b0dbb = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b0d = stablehlo.add %s3b0dc, %s3b0dbb : tensor<32x768x7x7xf32>
    %s3b0nri = stablehlo.reshape %s3b0d : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b0nnf = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %s3b0nep = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %s3b0nsmr = stablehlo.reduce(%s3b0nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b0nsm = stablehlo.broadcast_in_dim %s3b0nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b0nmu = stablehlo.divide %s3b0nsm, %s3b0nnf : tensor<32x37632xf32>
    %s3b0nxc = stablehlo.subtract %s3b0nri, %s3b0nmu : tensor<32x37632xf32>
    %s3b0nsq = stablehlo.multiply %s3b0nxc, %s3b0nxc : tensor<32x37632xf32>
    %s3b0nvsr = stablehlo.reduce(%s3b0nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b0nvs = stablehlo.broadcast_in_dim %s3b0nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b0nvr = stablehlo.divide %s3b0nvs, %s3b0nnf : tensor<32x37632xf32>
    %s3b0nve = stablehlo.add %s3b0nvr, %s3b0nep : tensor<32x37632xf32>
    %s3b0nistd = stablehlo.rsqrt %s3b0nve : tensor<32x37632xf32>
    %s3b0nxh = stablehlo.multiply %s3b0nxc, %s3b0nistd : tensor<32x37632xf32>
    %s3b0ngb = stablehlo.broadcast_in_dim %s3b0ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b0nbtb = stablehlo.broadcast_in_dim %s3b0nbt, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b0ngx = stablehlo.multiply %s3b0nxh, %s3b0ngb : tensor<32x37632xf32>
    %s3b0nfl = stablehlo.add %s3b0ngx, %s3b0nbtb : tensor<32x37632xf32>
    %s3b0n = stablehlo.reshape %s3b0nfl : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %s3b0ec = stablehlo.convolution(%s3b0n, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %s3b0ebb = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %s3b0e = stablehlo.add %s3b0ec, %s3b0ebb : tensor<32x3072x7x7xf32>
    %s3b0gx2 = stablehlo.multiply %s3b0e, %s3b0e : tensor<32x3072x7x7xf32>
    %s3b0gx3 = stablehlo.multiply %s3b0gx2, %s3b0e : tensor<32x3072x7x7xf32>
    %s3b0gck = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %s3b0gkx3 = stablehlo.multiply %s3b0gck, %s3b0gx3 : tensor<32x3072x7x7xf32>
    %s3b0ginn = stablehlo.add %s3b0e, %s3b0gkx3 : tensor<32x3072x7x7xf32>
    %s3b0gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %s3b0gu = stablehlo.multiply %s3b0gcs, %s3b0ginn : tensor<32x3072x7x7xf32>
    %s3b0gt = stablehlo.tanh %s3b0gu : tensor<32x3072x7x7xf32>
    %s3b0gone = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %s3b0gopt = stablehlo.add %s3b0gone, %s3b0gt : tensor<32x3072x7x7xf32>
    %s3b0ghalf = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %s3b0ghx = stablehlo.multiply %s3b0ghalf, %s3b0e : tensor<32x3072x7x7xf32>
    %s3b0g = stablehlo.multiply %s3b0ghx, %s3b0gopt : tensor<32x3072x7x7xf32>
    %s3b0pc = stablehlo.convolution(%s3b0g, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %s3b0pbb = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b0p = stablehlo.add %s3b0pc, %s3b0pbb : tensor<32x768x7x7xf32>
    %s3b0lsgb = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b0ls = stablehlo.multiply %s3b0p, %s3b0lsgb : tensor<32x768x7x7xf32>
    %s3b0o = stablehlo.add %s3b0ls, %d2c : tensor<32x768x7x7xf32>
    %s3b1dc = stablehlo.convolution(%s3b0o, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %s3b1dbb = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b1d = stablehlo.add %s3b1dc, %s3b1dbb : tensor<32x768x7x7xf32>
    %s3b1nri = stablehlo.reshape %s3b1d : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b1nnf = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %s3b1nep = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %s3b1nsmr = stablehlo.reduce(%s3b1nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b1nsm = stablehlo.broadcast_in_dim %s3b1nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b1nmu = stablehlo.divide %s3b1nsm, %s3b1nnf : tensor<32x37632xf32>
    %s3b1nxc = stablehlo.subtract %s3b1nri, %s3b1nmu : tensor<32x37632xf32>
    %s3b1nsq = stablehlo.multiply %s3b1nxc, %s3b1nxc : tensor<32x37632xf32>
    %s3b1nvsr = stablehlo.reduce(%s3b1nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b1nvs = stablehlo.broadcast_in_dim %s3b1nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b1nvr = stablehlo.divide %s3b1nvs, %s3b1nnf : tensor<32x37632xf32>
    %s3b1nve = stablehlo.add %s3b1nvr, %s3b1nep : tensor<32x37632xf32>
    %s3b1nistd = stablehlo.rsqrt %s3b1nve : tensor<32x37632xf32>
    %s3b1nxh = stablehlo.multiply %s3b1nxc, %s3b1nistd : tensor<32x37632xf32>
    %s3b1ngb = stablehlo.broadcast_in_dim %s3b1ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b1nbtb = stablehlo.broadcast_in_dim %s3b1nbt, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b1ngx = stablehlo.multiply %s3b1nxh, %s3b1ngb : tensor<32x37632xf32>
    %s3b1nfl = stablehlo.add %s3b1ngx, %s3b1nbtb : tensor<32x37632xf32>
    %s3b1n = stablehlo.reshape %s3b1nfl : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %s3b1ec = stablehlo.convolution(%s3b1n, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %s3b1ebb = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %s3b1e = stablehlo.add %s3b1ec, %s3b1ebb : tensor<32x3072x7x7xf32>
    %s3b1gx2 = stablehlo.multiply %s3b1e, %s3b1e : tensor<32x3072x7x7xf32>
    %s3b1gx3 = stablehlo.multiply %s3b1gx2, %s3b1e : tensor<32x3072x7x7xf32>
    %s3b1gck = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %s3b1gkx3 = stablehlo.multiply %s3b1gck, %s3b1gx3 : tensor<32x3072x7x7xf32>
    %s3b1ginn = stablehlo.add %s3b1e, %s3b1gkx3 : tensor<32x3072x7x7xf32>
    %s3b1gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %s3b1gu = stablehlo.multiply %s3b1gcs, %s3b1ginn : tensor<32x3072x7x7xf32>
    %s3b1gt = stablehlo.tanh %s3b1gu : tensor<32x3072x7x7xf32>
    %s3b1gone = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %s3b1gopt = stablehlo.add %s3b1gone, %s3b1gt : tensor<32x3072x7x7xf32>
    %s3b1ghalf = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %s3b1ghx = stablehlo.multiply %s3b1ghalf, %s3b1e : tensor<32x3072x7x7xf32>
    %s3b1g = stablehlo.multiply %s3b1ghx, %s3b1gopt : tensor<32x3072x7x7xf32>
    %s3b1pc = stablehlo.convolution(%s3b1g, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %s3b1pbb = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b1p = stablehlo.add %s3b1pc, %s3b1pbb : tensor<32x768x7x7xf32>
    %s3b1lsgb = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b1ls = stablehlo.multiply %s3b1p, %s3b1lsgb : tensor<32x768x7x7xf32>
    %s3b1o = stablehlo.add %s3b1ls, %s3b0o : tensor<32x768x7x7xf32>
    %s3b2dc = stablehlo.convolution(%s3b1o, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %s3b2dbb = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b2d = stablehlo.add %s3b2dc, %s3b2dbb : tensor<32x768x7x7xf32>
    %s3b2nri = stablehlo.reshape %s3b2d : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b2nnf = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %s3b2nep = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %s3b2nsmr = stablehlo.reduce(%s3b2nri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b2nsm = stablehlo.broadcast_in_dim %s3b2nsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b2nmu = stablehlo.divide %s3b2nsm, %s3b2nnf : tensor<32x37632xf32>
    %s3b2nxc = stablehlo.subtract %s3b2nri, %s3b2nmu : tensor<32x37632xf32>
    %s3b2nsq = stablehlo.multiply %s3b2nxc, %s3b2nxc : tensor<32x37632xf32>
    %s3b2nvsr = stablehlo.reduce(%s3b2nsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b2nvs = stablehlo.broadcast_in_dim %s3b2nvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b2nvr = stablehlo.divide %s3b2nvs, %s3b2nnf : tensor<32x37632xf32>
    %s3b2nve = stablehlo.add %s3b2nvr, %s3b2nep : tensor<32x37632xf32>
    %s3b2nistd = stablehlo.rsqrt %s3b2nve : tensor<32x37632xf32>
    %s3b2nxh = stablehlo.multiply %s3b2nxc, %s3b2nistd : tensor<32x37632xf32>
    %s3b2ngb = stablehlo.broadcast_in_dim %s3b2ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b2nbtb = stablehlo.broadcast_in_dim %s3b2nbt, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b2ngx = stablehlo.multiply %s3b2nxh, %s3b2ngb : tensor<32x37632xf32>
    %s3b2nfl = stablehlo.add %s3b2ngx, %s3b2nbtb : tensor<32x37632xf32>
    %s3b2n = stablehlo.reshape %s3b2nfl : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %s3b2ec = stablehlo.convolution(%s3b2n, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %s3b2ebb = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %s3b2e = stablehlo.add %s3b2ec, %s3b2ebb : tensor<32x3072x7x7xf32>
    %s3b2gx2 = stablehlo.multiply %s3b2e, %s3b2e : tensor<32x3072x7x7xf32>
    %s3b2gx3 = stablehlo.multiply %s3b2gx2, %s3b2e : tensor<32x3072x7x7xf32>
    %s3b2gck = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %s3b2gkx3 = stablehlo.multiply %s3b2gck, %s3b2gx3 : tensor<32x3072x7x7xf32>
    %s3b2ginn = stablehlo.add %s3b2e, %s3b2gkx3 : tensor<32x3072x7x7xf32>
    %s3b2gcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %s3b2gu = stablehlo.multiply %s3b2gcs, %s3b2ginn : tensor<32x3072x7x7xf32>
    %s3b2gt = stablehlo.tanh %s3b2gu : tensor<32x3072x7x7xf32>
    %s3b2gone = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %s3b2gopt = stablehlo.add %s3b2gone, %s3b2gt : tensor<32x3072x7x7xf32>
    %s3b2ghalf = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %s3b2ghx = stablehlo.multiply %s3b2ghalf, %s3b2e : tensor<32x3072x7x7xf32>
    %s3b2g = stablehlo.multiply %s3b2ghx, %s3b2gopt : tensor<32x3072x7x7xf32>
    %s3b2pc = stablehlo.convolution(%s3b2g, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %s3b2pbb = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b2p = stablehlo.add %s3b2pc, %s3b2pbb : tensor<32x768x7x7xf32>
    %s3b2lsgb = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b2ls = stablehlo.multiply %s3b2p, %s3b2lsgb : tensor<32x768x7x7xf32>
    %s3b2o = stablehlo.add %s3b2ls, %s3b1o : tensor<32x768x7x7xf32>
    %gaps = stablehlo.reduce(%s3b2o init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %gapnf = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %gap = stablehlo.divide %gaps, %gapnf : tensor<32x768xf32>
    %gapr = stablehlo.reshape %gap : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %hnri = stablehlo.reshape %gapr : (tensor<32x768x1x1xf32>) -> tensor<32x768xf32>
    %hnnf = stablehlo.constant dense<768.0> : tensor<32x768xf32>
    %hnep = stablehlo.constant dense<1.0e-6> : tensor<32x768xf32>
    %hnsmr = stablehlo.reduce(%hnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %hnsm = stablehlo.broadcast_in_dim %hnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %hnmu = stablehlo.divide %hnsm, %hnnf : tensor<32x768xf32>
    %hnxc = stablehlo.subtract %hnri, %hnmu : tensor<32x768xf32>
    %hnsq = stablehlo.multiply %hnxc, %hnxc : tensor<32x768xf32>
    %hnvsr = stablehlo.reduce(%hnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %hnvs = stablehlo.broadcast_in_dim %hnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %hnvr = stablehlo.divide %hnvs, %hnnf : tensor<32x768xf32>
    %hnve = stablehlo.add %hnvr, %hnep : tensor<32x768xf32>
    %hnistd = stablehlo.rsqrt %hnve : tensor<32x768xf32>
    %hnxh = stablehlo.multiply %hnxc, %hnistd : tensor<32x768xf32>
    %hngb = stablehlo.broadcast_in_dim %hng, dims = [] : (tensor<f32>) -> tensor<32x768xf32>
    %hnbtb = stablehlo.broadcast_in_dim %hnbt, dims = [] : (tensor<f32>) -> tensor<32x768xf32>
    %hngx = stablehlo.multiply %hnxh, %hngb : tensor<32x768xf32>
    %hnfl = stablehlo.add %hngx, %hnbtb : tensor<32x768xf32>
    %hn = stablehlo.reshape %hnfl : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %hnf = stablehlo.reshape %hn : (tensor<32x768x1x1xf32>) -> tensor<32x768xf32>
    %ld = stablehlo.dot_general %hnf, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x10xf32>) -> tensor<32x10xf32>
    %ldb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %logits = stablehlo.add %ld, %ldb : tensor<32x10xf32>
    %le = stablehlo.exponential %logits : tensor<32x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<32x10xf32>
    %dyr0 = stablehlo.subtract %lsm, %onehot : tensor<32x10xf32>
    %lsa = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %lsaoh = stablehlo.multiply %lsa, %onehot : tensor<32x10xf32>
    %dyr1 = stablehlo.add %dyr0, %lsaoh : tensor<32x10xf32>
    %lsaik = stablehlo.constant dense<0.010000> : tensor<32x10xf32>
    %dyr = stablehlo.subtract %dyr1, %lsaik : tensor<32x10xf32>
    %bnc = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %dy = stablehlo.divide %dyr, %bnc : tensor<32x10xf32>
    %llog = stablehlo.log %lsm : tensor<32x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<32x10xf32>
    %t1s = stablehlo.reduce(%ohll init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %lls = stablehlo.reduce(%llog init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %omac = stablehlo.constant dense<0.900000> : tensor<32xf32>
    %aKc = stablehlo.constant dense<0.010000> : tensor<32xf32>
    %lt1 = stablehlo.multiply %omac, %t1s : tensor<32xf32>
    %lt2 = stablehlo.multiply %aKc, %lls : tensor<32xf32>
    %lpe = stablehlo.add %lt1, %lt2 : tensor<32xf32>
    %lsum2 = stablehlo.reduce(%lpe init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %lbfc = stablehlo.constant dense<32.0> : tensor<f32>
    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>
    %loss = stablehlo.negate %lossm : tensor<f32>
    %dhnf = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<768x10xf32>) -> tensor<32x768xf32>
    %dWd = stablehlo.dot_general %hnf, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<32x10xf32>) -> tensor<768x10xf32>
    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dhnr = stablehlo.reshape %dhnf : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %hdri = stablehlo.reshape %gapr : (tensor<32x768x1x1xf32>) -> tensor<32x768xf32>
    %hdrdy = stablehlo.reshape %dhnr : (tensor<32x768x1x1xf32>) -> tensor<32x768xf32>
    %hdnf = stablehlo.constant dense<768.0> : tensor<32x768xf32>
    %hdep = stablehlo.constant dense<1.0e-6> : tensor<32x768xf32>
    %hdsmr = stablehlo.reduce(%hdri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %hdsm = stablehlo.broadcast_in_dim %hdsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %hdmu = stablehlo.divide %hdsm, %hdnf : tensor<32x768xf32>
    %hdxc = stablehlo.subtract %hdri, %hdmu : tensor<32x768xf32>
    %hdsq = stablehlo.multiply %hdxc, %hdxc : tensor<32x768xf32>
    %hdvsr = stablehlo.reduce(%hdsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %hdvs = stablehlo.broadcast_in_dim %hdvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %hdvr = stablehlo.divide %hdvs, %hdnf : tensor<32x768xf32>
    %hdve = stablehlo.add %hdvr, %hdep : tensor<32x768xf32>
    %hdistd = stablehlo.rsqrt %hdve : tensor<32x768xf32>
    %hdxh = stablehlo.multiply %hdxc, %hdistd : tensor<32x768xf32>
    %hdgb = stablehlo.broadcast_in_dim %hng, dims = [] : (tensor<f32>) -> tensor<32x768xf32>
    %hddxh = stablehlo.multiply %hdgb, %hdrdy : tensor<32x768xf32>
    %hdsdxr = stablehlo.reduce(%hddxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %hdsdx = stablehlo.broadcast_in_dim %hdsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %hdxd = stablehlo.multiply %hdxh, %hddxh : tensor<32x768xf32>
    %hdsxdr = stablehlo.reduce(%hdxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<32xf32>
    %hdsxd = stablehlo.broadcast_in_dim %hdsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x768xf32>
    %hdt1 = stablehlo.multiply %hddxh, %hdnf : tensor<32x768xf32>
    %hdi1 = stablehlo.subtract %hdt1, %hdsdx : tensor<32x768xf32>
    %hdxs = stablehlo.multiply %hdxh, %hdsxd : tensor<32x768xf32>
    %hdi2 = stablehlo.subtract %hdi1, %hdxs : tensor<32x768xf32>
    %hdsN = stablehlo.divide %hdistd, %hdnf : tensor<32x768xf32>
    %hdgin = stablehlo.multiply %hdsN, %hdi2 : tensor<32x768xf32>
    %hd = stablehlo.reshape %hdgin : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %hddgp = stablehlo.multiply %hdrdy, %hdxh : tensor<32x768xf32>
    %hddg = stablehlo.reduce(%hddgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<f32>
    %hddb = stablehlo.reduce(%hdrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x768xf32>, tensor<f32>) -> tensor<f32>
    %hdf = stablehlo.reshape %hd : (tensor<32x768x1x1xf32>) -> tensor<32x768xf32>
    %dgd = stablehlo.divide %hdf, %gapnf : tensor<32x768xf32>
    %dgap = stablehlo.broadcast_in_dim %dgd, dims = [0, 1] : (tensor<32x768xf32>) -> tensor<32x768x7x7xf32>
    %s3b2dlsgb = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b2dls = stablehlo.multiply %s3b2dlsgb, %dgap : tensor<32x768x7x7xf32>
    %s3b2dlsxdy = stablehlo.multiply %s3b2p, %dgap : tensor<32x768x7x7xf32>
    %s3b2dlsdg = stablehlo.reduce(%s3b2dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b2dpt = stablehlo.transpose %s3b2pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %s3b2dp = stablehlo.convolution(%s3b2dls, %s3b2dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %s3b2dpWxt = stablehlo.transpose %s3b2g, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %s3b2dpWdt = stablehlo.transpose %s3b2dls, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b2dpWraw = stablehlo.convolution(%s3b2dpWxt, %s3b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %s3b2dpW = stablehlo.transpose %s3b2dpWraw, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %s3b2dpb = stablehlo.reduce(%s3b2dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b2dgx2 = stablehlo.multiply %s3b2e, %s3b2e : tensor<32x3072x7x7xf32>
    %s3b2dgx3 = stablehlo.multiply %s3b2dgx2, %s3b2e : tensor<32x3072x7x7xf32>
    %s3b2dgck = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %s3b2dgkx3 = stablehlo.multiply %s3b2dgck, %s3b2dgx3 : tensor<32x3072x7x7xf32>
    %s3b2dginn = stablehlo.add %s3b2e, %s3b2dgkx3 : tensor<32x3072x7x7xf32>
    %s3b2dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %s3b2dgu = stablehlo.multiply %s3b2dgcs, %s3b2dginn : tensor<32x3072x7x7xf32>
    %s3b2dgt = stablehlo.tanh %s3b2dgu : tensor<32x3072x7x7xf32>
    %s3b2dgone = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %s3b2dgopt = stablehlo.add %s3b2dgone, %s3b2dgt : tensor<32x3072x7x7xf32>
    %s3b2dghalf = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %s3b2dgterm1 = stablehlo.multiply %s3b2dghalf, %s3b2dgopt : tensor<32x3072x7x7xf32>
    %s3b2dgt2 = stablehlo.multiply %s3b2dgt, %s3b2dgt : tensor<32x3072x7x7xf32>
    %s3b2dgomt2 = stablehlo.subtract %s3b2dgone, %s3b2dgt2 : tensor<32x3072x7x7xf32>
    %s3b2dghx = stablehlo.multiply %s3b2dghalf, %s3b2e : tensor<32x3072x7x7xf32>
    %s3b2dghxo = stablehlo.multiply %s3b2dghx, %s3b2dgomt2 : tensor<32x3072x7x7xf32>
    %s3b2dgc3b = stablehlo.constant dense<0.134145> : tensor<32x3072x7x7xf32>
    %s3b2dga3x2 = stablehlo.multiply %s3b2dgc3b, %s3b2dgx2 : tensor<32x3072x7x7xf32>
    %s3b2dgin2 = stablehlo.add %s3b2dgone, %s3b2dga3x2 : tensor<32x3072x7x7xf32>
    %s3b2dgup = stablehlo.multiply %s3b2dgcs, %s3b2dgin2 : tensor<32x3072x7x7xf32>
    %s3b2dgterm2 = stablehlo.multiply %s3b2dghxo, %s3b2dgup : tensor<32x3072x7x7xf32>
    %s3b2dggp = stablehlo.add %s3b2dgterm1, %s3b2dgterm2 : tensor<32x3072x7x7xf32>
    %s3b2dg = stablehlo.multiply %s3b2dp, %s3b2dggp : tensor<32x3072x7x7xf32>
    %s3b2det = stablehlo.transpose %s3b2eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %s3b2de = stablehlo.convolution(%s3b2dg, %s3b2det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %s3b2deWxt = stablehlo.transpose %s3b2n, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b2deWdt = stablehlo.transpose %s3b2dg, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %s3b2deWraw = stablehlo.convolution(%s3b2deWxt, %s3b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %s3b2deW = stablehlo.transpose %s3b2deWraw, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %s3b2deb = stablehlo.reduce(%s3b2dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %s3b2dnri = stablehlo.reshape %s3b2d : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b2dnrdy = stablehlo.reshape %s3b2de : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b2dnnf = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %s3b2dnep = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %s3b2dnsmr = stablehlo.reduce(%s3b2dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b2dnsm = stablehlo.broadcast_in_dim %s3b2dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b2dnmu = stablehlo.divide %s3b2dnsm, %s3b2dnnf : tensor<32x37632xf32>
    %s3b2dnxc = stablehlo.subtract %s3b2dnri, %s3b2dnmu : tensor<32x37632xf32>
    %s3b2dnsq = stablehlo.multiply %s3b2dnxc, %s3b2dnxc : tensor<32x37632xf32>
    %s3b2dnvsr = stablehlo.reduce(%s3b2dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b2dnvs = stablehlo.broadcast_in_dim %s3b2dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b2dnvr = stablehlo.divide %s3b2dnvs, %s3b2dnnf : tensor<32x37632xf32>
    %s3b2dnve = stablehlo.add %s3b2dnvr, %s3b2dnep : tensor<32x37632xf32>
    %s3b2dnistd = stablehlo.rsqrt %s3b2dnve : tensor<32x37632xf32>
    %s3b2dnxh = stablehlo.multiply %s3b2dnxc, %s3b2dnistd : tensor<32x37632xf32>
    %s3b2dngb = stablehlo.broadcast_in_dim %s3b2ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b2dndxh = stablehlo.multiply %s3b2dngb, %s3b2dnrdy : tensor<32x37632xf32>
    %s3b2dnsdxr = stablehlo.reduce(%s3b2dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b2dnsdx = stablehlo.broadcast_in_dim %s3b2dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b2dnxd = stablehlo.multiply %s3b2dnxh, %s3b2dndxh : tensor<32x37632xf32>
    %s3b2dnsxdr = stablehlo.reduce(%s3b2dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b2dnsxd = stablehlo.broadcast_in_dim %s3b2dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b2dnt1 = stablehlo.multiply %s3b2dndxh, %s3b2dnnf : tensor<32x37632xf32>
    %s3b2dni1 = stablehlo.subtract %s3b2dnt1, %s3b2dnsdx : tensor<32x37632xf32>
    %s3b2dnxs = stablehlo.multiply %s3b2dnxh, %s3b2dnsxd : tensor<32x37632xf32>
    %s3b2dni2 = stablehlo.subtract %s3b2dni1, %s3b2dnxs : tensor<32x37632xf32>
    %s3b2dnsN = stablehlo.divide %s3b2dnistd, %s3b2dnnf : tensor<32x37632xf32>
    %s3b2dngin = stablehlo.multiply %s3b2dnsN, %s3b2dni2 : tensor<32x37632xf32>
    %s3b2dn = stablehlo.reshape %s3b2dngin : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %s3b2dndgp = stablehlo.multiply %s3b2dnrdy, %s3b2dnxh : tensor<32x37632xf32>
    %s3b2dndg = stablehlo.reduce(%s3b2dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %s3b2dndb = stablehlo.reduce(%s3b2dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %s3b2ddrev = stablehlo.reverse %s3b2dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %s3b2dd = stablehlo.convolution(%s3b2dn, %s3b2ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %s3b2ddWxt = stablehlo.transpose %s3b1o, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b2ddWdt = stablehlo.transpose %s3b2dn, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b2ddWraw = stablehlo.convolution(%s3b2ddWxt, %s3b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %s3b2ddW = stablehlo.reshape %s3b2ddWraw : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %s3b2ddb = stablehlo.reduce(%s3b2dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b2dx = stablehlo.add %s3b2dd, %dgap : tensor<32x768x7x7xf32>
    %s3b1dlsgb = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b1dls = stablehlo.multiply %s3b1dlsgb, %s3b2dx : tensor<32x768x7x7xf32>
    %s3b1dlsxdy = stablehlo.multiply %s3b1p, %s3b2dx : tensor<32x768x7x7xf32>
    %s3b1dlsdg = stablehlo.reduce(%s3b1dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b1dpt = stablehlo.transpose %s3b1pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %s3b1dp = stablehlo.convolution(%s3b1dls, %s3b1dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %s3b1dpWxt = stablehlo.transpose %s3b1g, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %s3b1dpWdt = stablehlo.transpose %s3b1dls, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b1dpWraw = stablehlo.convolution(%s3b1dpWxt, %s3b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %s3b1dpW = stablehlo.transpose %s3b1dpWraw, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %s3b1dpb = stablehlo.reduce(%s3b1dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b1dgx2 = stablehlo.multiply %s3b1e, %s3b1e : tensor<32x3072x7x7xf32>
    %s3b1dgx3 = stablehlo.multiply %s3b1dgx2, %s3b1e : tensor<32x3072x7x7xf32>
    %s3b1dgck = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %s3b1dgkx3 = stablehlo.multiply %s3b1dgck, %s3b1dgx3 : tensor<32x3072x7x7xf32>
    %s3b1dginn = stablehlo.add %s3b1e, %s3b1dgkx3 : tensor<32x3072x7x7xf32>
    %s3b1dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %s3b1dgu = stablehlo.multiply %s3b1dgcs, %s3b1dginn : tensor<32x3072x7x7xf32>
    %s3b1dgt = stablehlo.tanh %s3b1dgu : tensor<32x3072x7x7xf32>
    %s3b1dgone = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %s3b1dgopt = stablehlo.add %s3b1dgone, %s3b1dgt : tensor<32x3072x7x7xf32>
    %s3b1dghalf = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %s3b1dgterm1 = stablehlo.multiply %s3b1dghalf, %s3b1dgopt : tensor<32x3072x7x7xf32>
    %s3b1dgt2 = stablehlo.multiply %s3b1dgt, %s3b1dgt : tensor<32x3072x7x7xf32>
    %s3b1dgomt2 = stablehlo.subtract %s3b1dgone, %s3b1dgt2 : tensor<32x3072x7x7xf32>
    %s3b1dghx = stablehlo.multiply %s3b1dghalf, %s3b1e : tensor<32x3072x7x7xf32>
    %s3b1dghxo = stablehlo.multiply %s3b1dghx, %s3b1dgomt2 : tensor<32x3072x7x7xf32>
    %s3b1dgc3b = stablehlo.constant dense<0.134145> : tensor<32x3072x7x7xf32>
    %s3b1dga3x2 = stablehlo.multiply %s3b1dgc3b, %s3b1dgx2 : tensor<32x3072x7x7xf32>
    %s3b1dgin2 = stablehlo.add %s3b1dgone, %s3b1dga3x2 : tensor<32x3072x7x7xf32>
    %s3b1dgup = stablehlo.multiply %s3b1dgcs, %s3b1dgin2 : tensor<32x3072x7x7xf32>
    %s3b1dgterm2 = stablehlo.multiply %s3b1dghxo, %s3b1dgup : tensor<32x3072x7x7xf32>
    %s3b1dggp = stablehlo.add %s3b1dgterm1, %s3b1dgterm2 : tensor<32x3072x7x7xf32>
    %s3b1dg = stablehlo.multiply %s3b1dp, %s3b1dggp : tensor<32x3072x7x7xf32>
    %s3b1det = stablehlo.transpose %s3b1eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %s3b1de = stablehlo.convolution(%s3b1dg, %s3b1det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %s3b1deWxt = stablehlo.transpose %s3b1n, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b1deWdt = stablehlo.transpose %s3b1dg, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %s3b1deWraw = stablehlo.convolution(%s3b1deWxt, %s3b1deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %s3b1deW = stablehlo.transpose %s3b1deWraw, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %s3b1deb = stablehlo.reduce(%s3b1dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %s3b1dnri = stablehlo.reshape %s3b1d : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b1dnrdy = stablehlo.reshape %s3b1de : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b1dnnf = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %s3b1dnep = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %s3b1dnsmr = stablehlo.reduce(%s3b1dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b1dnsm = stablehlo.broadcast_in_dim %s3b1dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b1dnmu = stablehlo.divide %s3b1dnsm, %s3b1dnnf : tensor<32x37632xf32>
    %s3b1dnxc = stablehlo.subtract %s3b1dnri, %s3b1dnmu : tensor<32x37632xf32>
    %s3b1dnsq = stablehlo.multiply %s3b1dnxc, %s3b1dnxc : tensor<32x37632xf32>
    %s3b1dnvsr = stablehlo.reduce(%s3b1dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b1dnvs = stablehlo.broadcast_in_dim %s3b1dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b1dnvr = stablehlo.divide %s3b1dnvs, %s3b1dnnf : tensor<32x37632xf32>
    %s3b1dnve = stablehlo.add %s3b1dnvr, %s3b1dnep : tensor<32x37632xf32>
    %s3b1dnistd = stablehlo.rsqrt %s3b1dnve : tensor<32x37632xf32>
    %s3b1dnxh = stablehlo.multiply %s3b1dnxc, %s3b1dnistd : tensor<32x37632xf32>
    %s3b1dngb = stablehlo.broadcast_in_dim %s3b1ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b1dndxh = stablehlo.multiply %s3b1dngb, %s3b1dnrdy : tensor<32x37632xf32>
    %s3b1dnsdxr = stablehlo.reduce(%s3b1dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b1dnsdx = stablehlo.broadcast_in_dim %s3b1dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b1dnxd = stablehlo.multiply %s3b1dnxh, %s3b1dndxh : tensor<32x37632xf32>
    %s3b1dnsxdr = stablehlo.reduce(%s3b1dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b1dnsxd = stablehlo.broadcast_in_dim %s3b1dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b1dnt1 = stablehlo.multiply %s3b1dndxh, %s3b1dnnf : tensor<32x37632xf32>
    %s3b1dni1 = stablehlo.subtract %s3b1dnt1, %s3b1dnsdx : tensor<32x37632xf32>
    %s3b1dnxs = stablehlo.multiply %s3b1dnxh, %s3b1dnsxd : tensor<32x37632xf32>
    %s3b1dni2 = stablehlo.subtract %s3b1dni1, %s3b1dnxs : tensor<32x37632xf32>
    %s3b1dnsN = stablehlo.divide %s3b1dnistd, %s3b1dnnf : tensor<32x37632xf32>
    %s3b1dngin = stablehlo.multiply %s3b1dnsN, %s3b1dni2 : tensor<32x37632xf32>
    %s3b1dn = stablehlo.reshape %s3b1dngin : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %s3b1dndgp = stablehlo.multiply %s3b1dnrdy, %s3b1dnxh : tensor<32x37632xf32>
    %s3b1dndg = stablehlo.reduce(%s3b1dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %s3b1dndb = stablehlo.reduce(%s3b1dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %s3b1ddrev = stablehlo.reverse %s3b1dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %s3b1dd = stablehlo.convolution(%s3b1dn, %s3b1ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %s3b1ddWxt = stablehlo.transpose %s3b0o, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b1ddWdt = stablehlo.transpose %s3b1dn, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b1ddWraw = stablehlo.convolution(%s3b1ddWxt, %s3b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %s3b1ddW = stablehlo.reshape %s3b1ddWraw : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %s3b1ddb = stablehlo.reduce(%s3b1dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b1dx = stablehlo.add %s3b1dd, %s3b2dx : tensor<32x768x7x7xf32>
    %s3b0dlsgb = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %s3b0dls = stablehlo.multiply %s3b0dlsgb, %s3b1dx : tensor<32x768x7x7xf32>
    %s3b0dlsxdy = stablehlo.multiply %s3b0p, %s3b1dx : tensor<32x768x7x7xf32>
    %s3b0dlsdg = stablehlo.reduce(%s3b0dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b0dpt = stablehlo.transpose %s3b0pW, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %s3b0dp = stablehlo.convolution(%s3b0dls, %s3b0dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %s3b0dpWxt = stablehlo.transpose %s3b0g, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %s3b0dpWdt = stablehlo.transpose %s3b0dls, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b0dpWraw = stablehlo.convolution(%s3b0dpWxt, %s3b0dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %s3b0dpW = stablehlo.transpose %s3b0dpWraw, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %s3b0dpb = stablehlo.reduce(%s3b0dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b0dgx2 = stablehlo.multiply %s3b0e, %s3b0e : tensor<32x3072x7x7xf32>
    %s3b0dgx3 = stablehlo.multiply %s3b0dgx2, %s3b0e : tensor<32x3072x7x7xf32>
    %s3b0dgck = stablehlo.constant dense<0.044715> : tensor<32x3072x7x7xf32>
    %s3b0dgkx3 = stablehlo.multiply %s3b0dgck, %s3b0dgx3 : tensor<32x3072x7x7xf32>
    %s3b0dginn = stablehlo.add %s3b0e, %s3b0dgkx3 : tensor<32x3072x7x7xf32>
    %s3b0dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x3072x7x7xf32>
    %s3b0dgu = stablehlo.multiply %s3b0dgcs, %s3b0dginn : tensor<32x3072x7x7xf32>
    %s3b0dgt = stablehlo.tanh %s3b0dgu : tensor<32x3072x7x7xf32>
    %s3b0dgone = stablehlo.constant dense<1.0> : tensor<32x3072x7x7xf32>
    %s3b0dgopt = stablehlo.add %s3b0dgone, %s3b0dgt : tensor<32x3072x7x7xf32>
    %s3b0dghalf = stablehlo.constant dense<0.5> : tensor<32x3072x7x7xf32>
    %s3b0dgterm1 = stablehlo.multiply %s3b0dghalf, %s3b0dgopt : tensor<32x3072x7x7xf32>
    %s3b0dgt2 = stablehlo.multiply %s3b0dgt, %s3b0dgt : tensor<32x3072x7x7xf32>
    %s3b0dgomt2 = stablehlo.subtract %s3b0dgone, %s3b0dgt2 : tensor<32x3072x7x7xf32>
    %s3b0dghx = stablehlo.multiply %s3b0dghalf, %s3b0e : tensor<32x3072x7x7xf32>
    %s3b0dghxo = stablehlo.multiply %s3b0dghx, %s3b0dgomt2 : tensor<32x3072x7x7xf32>
    %s3b0dgc3b = stablehlo.constant dense<0.134145> : tensor<32x3072x7x7xf32>
    %s3b0dga3x2 = stablehlo.multiply %s3b0dgc3b, %s3b0dgx2 : tensor<32x3072x7x7xf32>
    %s3b0dgin2 = stablehlo.add %s3b0dgone, %s3b0dga3x2 : tensor<32x3072x7x7xf32>
    %s3b0dgup = stablehlo.multiply %s3b0dgcs, %s3b0dgin2 : tensor<32x3072x7x7xf32>
    %s3b0dgterm2 = stablehlo.multiply %s3b0dghxo, %s3b0dgup : tensor<32x3072x7x7xf32>
    %s3b0dggp = stablehlo.add %s3b0dgterm1, %s3b0dgterm2 : tensor<32x3072x7x7xf32>
    %s3b0dg = stablehlo.multiply %s3b0dp, %s3b0dggp : tensor<32x3072x7x7xf32>
    %s3b0det = stablehlo.transpose %s3b0eW, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %s3b0de = stablehlo.convolution(%s3b0dg, %s3b0det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %s3b0deWxt = stablehlo.transpose %s3b0n, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b0deWdt = stablehlo.transpose %s3b0dg, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %s3b0deWraw = stablehlo.convolution(%s3b0deWxt, %s3b0deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %s3b0deW = stablehlo.transpose %s3b0deWraw, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %s3b0deb = stablehlo.reduce(%s3b0dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %s3b0dnri = stablehlo.reshape %s3b0d : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b0dnrdy = stablehlo.reshape %s3b0de : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %s3b0dnnf = stablehlo.constant dense<37632.0> : tensor<32x37632xf32>
    %s3b0dnep = stablehlo.constant dense<1.0e-6> : tensor<32x37632xf32>
    %s3b0dnsmr = stablehlo.reduce(%s3b0dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b0dnsm = stablehlo.broadcast_in_dim %s3b0dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b0dnmu = stablehlo.divide %s3b0dnsm, %s3b0dnnf : tensor<32x37632xf32>
    %s3b0dnxc = stablehlo.subtract %s3b0dnri, %s3b0dnmu : tensor<32x37632xf32>
    %s3b0dnsq = stablehlo.multiply %s3b0dnxc, %s3b0dnxc : tensor<32x37632xf32>
    %s3b0dnvsr = stablehlo.reduce(%s3b0dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b0dnvs = stablehlo.broadcast_in_dim %s3b0dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b0dnvr = stablehlo.divide %s3b0dnvs, %s3b0dnnf : tensor<32x37632xf32>
    %s3b0dnve = stablehlo.add %s3b0dnvr, %s3b0dnep : tensor<32x37632xf32>
    %s3b0dnistd = stablehlo.rsqrt %s3b0dnve : tensor<32x37632xf32>
    %s3b0dnxh = stablehlo.multiply %s3b0dnxc, %s3b0dnistd : tensor<32x37632xf32>
    %s3b0dngb = stablehlo.broadcast_in_dim %s3b0ng, dims = [] : (tensor<f32>) -> tensor<32x37632xf32>
    %s3b0dndxh = stablehlo.multiply %s3b0dngb, %s3b0dnrdy : tensor<32x37632xf32>
    %s3b0dnsdxr = stablehlo.reduce(%s3b0dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b0dnsdx = stablehlo.broadcast_in_dim %s3b0dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b0dnxd = stablehlo.multiply %s3b0dnxh, %s3b0dndxh : tensor<32x37632xf32>
    %s3b0dnsxdr = stablehlo.reduce(%s3b0dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<32xf32>
    %s3b0dnsxd = stablehlo.broadcast_in_dim %s3b0dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %s3b0dnt1 = stablehlo.multiply %s3b0dndxh, %s3b0dnnf : tensor<32x37632xf32>
    %s3b0dni1 = stablehlo.subtract %s3b0dnt1, %s3b0dnsdx : tensor<32x37632xf32>
    %s3b0dnxs = stablehlo.multiply %s3b0dnxh, %s3b0dnsxd : tensor<32x37632xf32>
    %s3b0dni2 = stablehlo.subtract %s3b0dni1, %s3b0dnxs : tensor<32x37632xf32>
    %s3b0dnsN = stablehlo.divide %s3b0dnistd, %s3b0dnnf : tensor<32x37632xf32>
    %s3b0dngin = stablehlo.multiply %s3b0dnsN, %s3b0dni2 : tensor<32x37632xf32>
    %s3b0dn = stablehlo.reshape %s3b0dngin : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %s3b0dndgp = stablehlo.multiply %s3b0dnrdy, %s3b0dnxh : tensor<32x37632xf32>
    %s3b0dndg = stablehlo.reduce(%s3b0dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %s3b0dndb = stablehlo.reduce(%s3b0dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x37632xf32>, tensor<f32>) -> tensor<f32>
    %s3b0ddrev = stablehlo.reverse %s3b0dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %s3b0dd = stablehlo.convolution(%s3b0dn, %s3b0ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %s3b0ddWxt = stablehlo.transpose %d2c, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b0ddWdt = stablehlo.transpose %s3b0dn, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %s3b0ddWraw = stablehlo.convolution(%s3b0ddWxt, %s3b0ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %s3b0ddW = stablehlo.reshape %s3b0ddWraw : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %s3b0ddb = stablehlo.reduce(%s3b0dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %s3b0dx = stablehlo.add %s3b0dd, %s3b1dx : tensor<32x768x7x7xf32>
    %d2dcu = stablehlo.pad %s3b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x14x14xf32>
    %d2dct = stablehlo.transpose %d2W, dims = [1, 0, 2, 3] : (tensor<768x384x2x2xf32>) -> tensor<384x768x2x2xf32>
    %d2dcr = stablehlo.reverse %d2dct, dims = [2, 3] : tensor<384x768x2x2xf32>
    %d2dc = stablehlo.convolution(%d2dcu, %d2dcr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x14x14xf32>, tensor<384x768x2x2xf32>) -> tensor<32x384x14x14xf32>
    %d2dWu = stablehlo.pad %s3b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x13x13xf32>
    %d2dWxt = stablehlo.transpose %d2n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %d2dWdt = stablehlo.transpose %d2dWu, dims = [1, 0, 2, 3] : (tensor<32x768x13x13xf32>) -> tensor<768x32x13x13xf32>
    %d2dWraw = stablehlo.convolution(%d2dWxt, %d2dWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<768x32x13x13xf32>) -> tensor<384x768x2x2xf32>
    %d2dW = stablehlo.transpose %d2dWraw, dims = [1, 0, 2, 3] : (tensor<384x768x2x2xf32>) -> tensor<768x384x2x2xf32>
    %d2db = stablehlo.reduce(%s3b0dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %d2dnri = stablehlo.reshape %s2b8o : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %d2dnrdy = stablehlo.reshape %d2dc : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %d2dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %d2dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %d2dnsmr = stablehlo.reduce(%d2dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %d2dnsm = stablehlo.broadcast_in_dim %d2dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %d2dnmu = stablehlo.divide %d2dnsm, %d2dnnf : tensor<32x75264xf32>
    %d2dnxc = stablehlo.subtract %d2dnri, %d2dnmu : tensor<32x75264xf32>
    %d2dnsq = stablehlo.multiply %d2dnxc, %d2dnxc : tensor<32x75264xf32>
    %d2dnvsr = stablehlo.reduce(%d2dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %d2dnvs = stablehlo.broadcast_in_dim %d2dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %d2dnvr = stablehlo.divide %d2dnvs, %d2dnnf : tensor<32x75264xf32>
    %d2dnve = stablehlo.add %d2dnvr, %d2dnep : tensor<32x75264xf32>
    %d2dnistd = stablehlo.rsqrt %d2dnve : tensor<32x75264xf32>
    %d2dnxh = stablehlo.multiply %d2dnxc, %d2dnistd : tensor<32x75264xf32>
    %d2dngb = stablehlo.broadcast_in_dim %d2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %d2dndxh = stablehlo.multiply %d2dngb, %d2dnrdy : tensor<32x75264xf32>
    %d2dnsdxr = stablehlo.reduce(%d2dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %d2dnsdx = stablehlo.broadcast_in_dim %d2dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %d2dnxd = stablehlo.multiply %d2dnxh, %d2dndxh : tensor<32x75264xf32>
    %d2dnsxdr = stablehlo.reduce(%d2dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %d2dnsxd = stablehlo.broadcast_in_dim %d2dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %d2dnt1 = stablehlo.multiply %d2dndxh, %d2dnnf : tensor<32x75264xf32>
    %d2dni1 = stablehlo.subtract %d2dnt1, %d2dnsdx : tensor<32x75264xf32>
    %d2dnxs = stablehlo.multiply %d2dnxh, %d2dnsxd : tensor<32x75264xf32>
    %d2dni2 = stablehlo.subtract %d2dni1, %d2dnxs : tensor<32x75264xf32>
    %d2dnsN = stablehlo.divide %d2dnistd, %d2dnnf : tensor<32x75264xf32>
    %d2dngin = stablehlo.multiply %d2dnsN, %d2dni2 : tensor<32x75264xf32>
    %d2dn = stablehlo.reshape %d2dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %d2dndgp = stablehlo.multiply %d2dnrdy, %d2dnxh : tensor<32x75264xf32>
    %d2dndg = stablehlo.reduce(%d2dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %d2dndb = stablehlo.reduce(%d2dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b8dlsgb = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b8dls = stablehlo.multiply %s2b8dlsgb, %d2dn : tensor<32x384x14x14xf32>
    %s2b8dlsxdy = stablehlo.multiply %s2b8p, %d2dn : tensor<32x384x14x14xf32>
    %s2b8dlsdg = stablehlo.reduce(%s2b8dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b8dpt = stablehlo.transpose %s2b8pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b8dp = stablehlo.convolution(%s2b8dls, %s2b8dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b8dpWxt = stablehlo.transpose %s2b8g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b8dpWdt = stablehlo.transpose %s2b8dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b8dpWraw = stablehlo.convolution(%s2b8dpWxt, %s2b8dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b8dpW = stablehlo.transpose %s2b8dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b8dpb = stablehlo.reduce(%s2b8dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b8dgx2 = stablehlo.multiply %s2b8e, %s2b8e : tensor<32x1536x14x14xf32>
    %s2b8dgx3 = stablehlo.multiply %s2b8dgx2, %s2b8e : tensor<32x1536x14x14xf32>
    %s2b8dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b8dgkx3 = stablehlo.multiply %s2b8dgck, %s2b8dgx3 : tensor<32x1536x14x14xf32>
    %s2b8dginn = stablehlo.add %s2b8e, %s2b8dgkx3 : tensor<32x1536x14x14xf32>
    %s2b8dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b8dgu = stablehlo.multiply %s2b8dgcs, %s2b8dginn : tensor<32x1536x14x14xf32>
    %s2b8dgt = stablehlo.tanh %s2b8dgu : tensor<32x1536x14x14xf32>
    %s2b8dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b8dgopt = stablehlo.add %s2b8dgone, %s2b8dgt : tensor<32x1536x14x14xf32>
    %s2b8dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b8dgterm1 = stablehlo.multiply %s2b8dghalf, %s2b8dgopt : tensor<32x1536x14x14xf32>
    %s2b8dgt2 = stablehlo.multiply %s2b8dgt, %s2b8dgt : tensor<32x1536x14x14xf32>
    %s2b8dgomt2 = stablehlo.subtract %s2b8dgone, %s2b8dgt2 : tensor<32x1536x14x14xf32>
    %s2b8dghx = stablehlo.multiply %s2b8dghalf, %s2b8e : tensor<32x1536x14x14xf32>
    %s2b8dghxo = stablehlo.multiply %s2b8dghx, %s2b8dgomt2 : tensor<32x1536x14x14xf32>
    %s2b8dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b8dga3x2 = stablehlo.multiply %s2b8dgc3b, %s2b8dgx2 : tensor<32x1536x14x14xf32>
    %s2b8dgin2 = stablehlo.add %s2b8dgone, %s2b8dga3x2 : tensor<32x1536x14x14xf32>
    %s2b8dgup = stablehlo.multiply %s2b8dgcs, %s2b8dgin2 : tensor<32x1536x14x14xf32>
    %s2b8dgterm2 = stablehlo.multiply %s2b8dghxo, %s2b8dgup : tensor<32x1536x14x14xf32>
    %s2b8dggp = stablehlo.add %s2b8dgterm1, %s2b8dgterm2 : tensor<32x1536x14x14xf32>
    %s2b8dg = stablehlo.multiply %s2b8dp, %s2b8dggp : tensor<32x1536x14x14xf32>
    %s2b8det = stablehlo.transpose %s2b8eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b8de = stablehlo.convolution(%s2b8dg, %s2b8det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b8deWxt = stablehlo.transpose %s2b8n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b8deWdt = stablehlo.transpose %s2b8dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b8deWraw = stablehlo.convolution(%s2b8deWxt, %s2b8deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b8deW = stablehlo.transpose %s2b8deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b8deb = stablehlo.reduce(%s2b8dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b8dnri = stablehlo.reshape %s2b8d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b8dnrdy = stablehlo.reshape %s2b8de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b8dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b8dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b8dnsmr = stablehlo.reduce(%s2b8dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b8dnsm = stablehlo.broadcast_in_dim %s2b8dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b8dnmu = stablehlo.divide %s2b8dnsm, %s2b8dnnf : tensor<32x75264xf32>
    %s2b8dnxc = stablehlo.subtract %s2b8dnri, %s2b8dnmu : tensor<32x75264xf32>
    %s2b8dnsq = stablehlo.multiply %s2b8dnxc, %s2b8dnxc : tensor<32x75264xf32>
    %s2b8dnvsr = stablehlo.reduce(%s2b8dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b8dnvs = stablehlo.broadcast_in_dim %s2b8dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b8dnvr = stablehlo.divide %s2b8dnvs, %s2b8dnnf : tensor<32x75264xf32>
    %s2b8dnve = stablehlo.add %s2b8dnvr, %s2b8dnep : tensor<32x75264xf32>
    %s2b8dnistd = stablehlo.rsqrt %s2b8dnve : tensor<32x75264xf32>
    %s2b8dnxh = stablehlo.multiply %s2b8dnxc, %s2b8dnistd : tensor<32x75264xf32>
    %s2b8dngb = stablehlo.broadcast_in_dim %s2b8ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b8dndxh = stablehlo.multiply %s2b8dngb, %s2b8dnrdy : tensor<32x75264xf32>
    %s2b8dnsdxr = stablehlo.reduce(%s2b8dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b8dnsdx = stablehlo.broadcast_in_dim %s2b8dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b8dnxd = stablehlo.multiply %s2b8dnxh, %s2b8dndxh : tensor<32x75264xf32>
    %s2b8dnsxdr = stablehlo.reduce(%s2b8dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b8dnsxd = stablehlo.broadcast_in_dim %s2b8dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b8dnt1 = stablehlo.multiply %s2b8dndxh, %s2b8dnnf : tensor<32x75264xf32>
    %s2b8dni1 = stablehlo.subtract %s2b8dnt1, %s2b8dnsdx : tensor<32x75264xf32>
    %s2b8dnxs = stablehlo.multiply %s2b8dnxh, %s2b8dnsxd : tensor<32x75264xf32>
    %s2b8dni2 = stablehlo.subtract %s2b8dni1, %s2b8dnxs : tensor<32x75264xf32>
    %s2b8dnsN = stablehlo.divide %s2b8dnistd, %s2b8dnnf : tensor<32x75264xf32>
    %s2b8dngin = stablehlo.multiply %s2b8dnsN, %s2b8dni2 : tensor<32x75264xf32>
    %s2b8dn = stablehlo.reshape %s2b8dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b8dndgp = stablehlo.multiply %s2b8dnrdy, %s2b8dnxh : tensor<32x75264xf32>
    %s2b8dndg = stablehlo.reduce(%s2b8dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b8dndb = stablehlo.reduce(%s2b8dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b8ddrev = stablehlo.reverse %s2b8dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b8dd = stablehlo.convolution(%s2b8dn, %s2b8ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b8ddWxt = stablehlo.transpose %s2b7o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b8ddWdt = stablehlo.transpose %s2b8dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b8ddWraw = stablehlo.convolution(%s2b8ddWxt, %s2b8ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b8ddW = stablehlo.reshape %s2b8ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b8ddb = stablehlo.reduce(%s2b8dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b8dx = stablehlo.add %s2b8dd, %d2dn : tensor<32x384x14x14xf32>
    %s2b7dlsgb = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b7dls = stablehlo.multiply %s2b7dlsgb, %s2b8dx : tensor<32x384x14x14xf32>
    %s2b7dlsxdy = stablehlo.multiply %s2b7p, %s2b8dx : tensor<32x384x14x14xf32>
    %s2b7dlsdg = stablehlo.reduce(%s2b7dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b7dpt = stablehlo.transpose %s2b7pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b7dp = stablehlo.convolution(%s2b7dls, %s2b7dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b7dpWxt = stablehlo.transpose %s2b7g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b7dpWdt = stablehlo.transpose %s2b7dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b7dpWraw = stablehlo.convolution(%s2b7dpWxt, %s2b7dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b7dpW = stablehlo.transpose %s2b7dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b7dpb = stablehlo.reduce(%s2b7dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b7dgx2 = stablehlo.multiply %s2b7e, %s2b7e : tensor<32x1536x14x14xf32>
    %s2b7dgx3 = stablehlo.multiply %s2b7dgx2, %s2b7e : tensor<32x1536x14x14xf32>
    %s2b7dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b7dgkx3 = stablehlo.multiply %s2b7dgck, %s2b7dgx3 : tensor<32x1536x14x14xf32>
    %s2b7dginn = stablehlo.add %s2b7e, %s2b7dgkx3 : tensor<32x1536x14x14xf32>
    %s2b7dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b7dgu = stablehlo.multiply %s2b7dgcs, %s2b7dginn : tensor<32x1536x14x14xf32>
    %s2b7dgt = stablehlo.tanh %s2b7dgu : tensor<32x1536x14x14xf32>
    %s2b7dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b7dgopt = stablehlo.add %s2b7dgone, %s2b7dgt : tensor<32x1536x14x14xf32>
    %s2b7dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b7dgterm1 = stablehlo.multiply %s2b7dghalf, %s2b7dgopt : tensor<32x1536x14x14xf32>
    %s2b7dgt2 = stablehlo.multiply %s2b7dgt, %s2b7dgt : tensor<32x1536x14x14xf32>
    %s2b7dgomt2 = stablehlo.subtract %s2b7dgone, %s2b7dgt2 : tensor<32x1536x14x14xf32>
    %s2b7dghx = stablehlo.multiply %s2b7dghalf, %s2b7e : tensor<32x1536x14x14xf32>
    %s2b7dghxo = stablehlo.multiply %s2b7dghx, %s2b7dgomt2 : tensor<32x1536x14x14xf32>
    %s2b7dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b7dga3x2 = stablehlo.multiply %s2b7dgc3b, %s2b7dgx2 : tensor<32x1536x14x14xf32>
    %s2b7dgin2 = stablehlo.add %s2b7dgone, %s2b7dga3x2 : tensor<32x1536x14x14xf32>
    %s2b7dgup = stablehlo.multiply %s2b7dgcs, %s2b7dgin2 : tensor<32x1536x14x14xf32>
    %s2b7dgterm2 = stablehlo.multiply %s2b7dghxo, %s2b7dgup : tensor<32x1536x14x14xf32>
    %s2b7dggp = stablehlo.add %s2b7dgterm1, %s2b7dgterm2 : tensor<32x1536x14x14xf32>
    %s2b7dg = stablehlo.multiply %s2b7dp, %s2b7dggp : tensor<32x1536x14x14xf32>
    %s2b7det = stablehlo.transpose %s2b7eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b7de = stablehlo.convolution(%s2b7dg, %s2b7det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b7deWxt = stablehlo.transpose %s2b7n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b7deWdt = stablehlo.transpose %s2b7dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b7deWraw = stablehlo.convolution(%s2b7deWxt, %s2b7deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b7deW = stablehlo.transpose %s2b7deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b7deb = stablehlo.reduce(%s2b7dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b7dnri = stablehlo.reshape %s2b7d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b7dnrdy = stablehlo.reshape %s2b7de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b7dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b7dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b7dnsmr = stablehlo.reduce(%s2b7dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b7dnsm = stablehlo.broadcast_in_dim %s2b7dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b7dnmu = stablehlo.divide %s2b7dnsm, %s2b7dnnf : tensor<32x75264xf32>
    %s2b7dnxc = stablehlo.subtract %s2b7dnri, %s2b7dnmu : tensor<32x75264xf32>
    %s2b7dnsq = stablehlo.multiply %s2b7dnxc, %s2b7dnxc : tensor<32x75264xf32>
    %s2b7dnvsr = stablehlo.reduce(%s2b7dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b7dnvs = stablehlo.broadcast_in_dim %s2b7dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b7dnvr = stablehlo.divide %s2b7dnvs, %s2b7dnnf : tensor<32x75264xf32>
    %s2b7dnve = stablehlo.add %s2b7dnvr, %s2b7dnep : tensor<32x75264xf32>
    %s2b7dnistd = stablehlo.rsqrt %s2b7dnve : tensor<32x75264xf32>
    %s2b7dnxh = stablehlo.multiply %s2b7dnxc, %s2b7dnistd : tensor<32x75264xf32>
    %s2b7dngb = stablehlo.broadcast_in_dim %s2b7ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b7dndxh = stablehlo.multiply %s2b7dngb, %s2b7dnrdy : tensor<32x75264xf32>
    %s2b7dnsdxr = stablehlo.reduce(%s2b7dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b7dnsdx = stablehlo.broadcast_in_dim %s2b7dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b7dnxd = stablehlo.multiply %s2b7dnxh, %s2b7dndxh : tensor<32x75264xf32>
    %s2b7dnsxdr = stablehlo.reduce(%s2b7dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b7dnsxd = stablehlo.broadcast_in_dim %s2b7dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b7dnt1 = stablehlo.multiply %s2b7dndxh, %s2b7dnnf : tensor<32x75264xf32>
    %s2b7dni1 = stablehlo.subtract %s2b7dnt1, %s2b7dnsdx : tensor<32x75264xf32>
    %s2b7dnxs = stablehlo.multiply %s2b7dnxh, %s2b7dnsxd : tensor<32x75264xf32>
    %s2b7dni2 = stablehlo.subtract %s2b7dni1, %s2b7dnxs : tensor<32x75264xf32>
    %s2b7dnsN = stablehlo.divide %s2b7dnistd, %s2b7dnnf : tensor<32x75264xf32>
    %s2b7dngin = stablehlo.multiply %s2b7dnsN, %s2b7dni2 : tensor<32x75264xf32>
    %s2b7dn = stablehlo.reshape %s2b7dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b7dndgp = stablehlo.multiply %s2b7dnrdy, %s2b7dnxh : tensor<32x75264xf32>
    %s2b7dndg = stablehlo.reduce(%s2b7dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b7dndb = stablehlo.reduce(%s2b7dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b7ddrev = stablehlo.reverse %s2b7dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b7dd = stablehlo.convolution(%s2b7dn, %s2b7ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b7ddWxt = stablehlo.transpose %s2b6o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b7ddWdt = stablehlo.transpose %s2b7dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b7ddWraw = stablehlo.convolution(%s2b7ddWxt, %s2b7ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b7ddW = stablehlo.reshape %s2b7ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b7ddb = stablehlo.reduce(%s2b7dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b7dx = stablehlo.add %s2b7dd, %s2b8dx : tensor<32x384x14x14xf32>
    %s2b6dlsgb = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b6dls = stablehlo.multiply %s2b6dlsgb, %s2b7dx : tensor<32x384x14x14xf32>
    %s2b6dlsxdy = stablehlo.multiply %s2b6p, %s2b7dx : tensor<32x384x14x14xf32>
    %s2b6dlsdg = stablehlo.reduce(%s2b6dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b6dpt = stablehlo.transpose %s2b6pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b6dp = stablehlo.convolution(%s2b6dls, %s2b6dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b6dpWxt = stablehlo.transpose %s2b6g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b6dpWdt = stablehlo.transpose %s2b6dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b6dpWraw = stablehlo.convolution(%s2b6dpWxt, %s2b6dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b6dpW = stablehlo.transpose %s2b6dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b6dpb = stablehlo.reduce(%s2b6dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b6dgx2 = stablehlo.multiply %s2b6e, %s2b6e : tensor<32x1536x14x14xf32>
    %s2b6dgx3 = stablehlo.multiply %s2b6dgx2, %s2b6e : tensor<32x1536x14x14xf32>
    %s2b6dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b6dgkx3 = stablehlo.multiply %s2b6dgck, %s2b6dgx3 : tensor<32x1536x14x14xf32>
    %s2b6dginn = stablehlo.add %s2b6e, %s2b6dgkx3 : tensor<32x1536x14x14xf32>
    %s2b6dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b6dgu = stablehlo.multiply %s2b6dgcs, %s2b6dginn : tensor<32x1536x14x14xf32>
    %s2b6dgt = stablehlo.tanh %s2b6dgu : tensor<32x1536x14x14xf32>
    %s2b6dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b6dgopt = stablehlo.add %s2b6dgone, %s2b6dgt : tensor<32x1536x14x14xf32>
    %s2b6dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b6dgterm1 = stablehlo.multiply %s2b6dghalf, %s2b6dgopt : tensor<32x1536x14x14xf32>
    %s2b6dgt2 = stablehlo.multiply %s2b6dgt, %s2b6dgt : tensor<32x1536x14x14xf32>
    %s2b6dgomt2 = stablehlo.subtract %s2b6dgone, %s2b6dgt2 : tensor<32x1536x14x14xf32>
    %s2b6dghx = stablehlo.multiply %s2b6dghalf, %s2b6e : tensor<32x1536x14x14xf32>
    %s2b6dghxo = stablehlo.multiply %s2b6dghx, %s2b6dgomt2 : tensor<32x1536x14x14xf32>
    %s2b6dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b6dga3x2 = stablehlo.multiply %s2b6dgc3b, %s2b6dgx2 : tensor<32x1536x14x14xf32>
    %s2b6dgin2 = stablehlo.add %s2b6dgone, %s2b6dga3x2 : tensor<32x1536x14x14xf32>
    %s2b6dgup = stablehlo.multiply %s2b6dgcs, %s2b6dgin2 : tensor<32x1536x14x14xf32>
    %s2b6dgterm2 = stablehlo.multiply %s2b6dghxo, %s2b6dgup : tensor<32x1536x14x14xf32>
    %s2b6dggp = stablehlo.add %s2b6dgterm1, %s2b6dgterm2 : tensor<32x1536x14x14xf32>
    %s2b6dg = stablehlo.multiply %s2b6dp, %s2b6dggp : tensor<32x1536x14x14xf32>
    %s2b6det = stablehlo.transpose %s2b6eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b6de = stablehlo.convolution(%s2b6dg, %s2b6det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b6deWxt = stablehlo.transpose %s2b6n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b6deWdt = stablehlo.transpose %s2b6dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b6deWraw = stablehlo.convolution(%s2b6deWxt, %s2b6deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b6deW = stablehlo.transpose %s2b6deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b6deb = stablehlo.reduce(%s2b6dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b6dnri = stablehlo.reshape %s2b6d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b6dnrdy = stablehlo.reshape %s2b6de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b6dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b6dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b6dnsmr = stablehlo.reduce(%s2b6dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b6dnsm = stablehlo.broadcast_in_dim %s2b6dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b6dnmu = stablehlo.divide %s2b6dnsm, %s2b6dnnf : tensor<32x75264xf32>
    %s2b6dnxc = stablehlo.subtract %s2b6dnri, %s2b6dnmu : tensor<32x75264xf32>
    %s2b6dnsq = stablehlo.multiply %s2b6dnxc, %s2b6dnxc : tensor<32x75264xf32>
    %s2b6dnvsr = stablehlo.reduce(%s2b6dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b6dnvs = stablehlo.broadcast_in_dim %s2b6dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b6dnvr = stablehlo.divide %s2b6dnvs, %s2b6dnnf : tensor<32x75264xf32>
    %s2b6dnve = stablehlo.add %s2b6dnvr, %s2b6dnep : tensor<32x75264xf32>
    %s2b6dnistd = stablehlo.rsqrt %s2b6dnve : tensor<32x75264xf32>
    %s2b6dnxh = stablehlo.multiply %s2b6dnxc, %s2b6dnistd : tensor<32x75264xf32>
    %s2b6dngb = stablehlo.broadcast_in_dim %s2b6ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b6dndxh = stablehlo.multiply %s2b6dngb, %s2b6dnrdy : tensor<32x75264xf32>
    %s2b6dnsdxr = stablehlo.reduce(%s2b6dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b6dnsdx = stablehlo.broadcast_in_dim %s2b6dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b6dnxd = stablehlo.multiply %s2b6dnxh, %s2b6dndxh : tensor<32x75264xf32>
    %s2b6dnsxdr = stablehlo.reduce(%s2b6dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b6dnsxd = stablehlo.broadcast_in_dim %s2b6dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b6dnt1 = stablehlo.multiply %s2b6dndxh, %s2b6dnnf : tensor<32x75264xf32>
    %s2b6dni1 = stablehlo.subtract %s2b6dnt1, %s2b6dnsdx : tensor<32x75264xf32>
    %s2b6dnxs = stablehlo.multiply %s2b6dnxh, %s2b6dnsxd : tensor<32x75264xf32>
    %s2b6dni2 = stablehlo.subtract %s2b6dni1, %s2b6dnxs : tensor<32x75264xf32>
    %s2b6dnsN = stablehlo.divide %s2b6dnistd, %s2b6dnnf : tensor<32x75264xf32>
    %s2b6dngin = stablehlo.multiply %s2b6dnsN, %s2b6dni2 : tensor<32x75264xf32>
    %s2b6dn = stablehlo.reshape %s2b6dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b6dndgp = stablehlo.multiply %s2b6dnrdy, %s2b6dnxh : tensor<32x75264xf32>
    %s2b6dndg = stablehlo.reduce(%s2b6dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b6dndb = stablehlo.reduce(%s2b6dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b6ddrev = stablehlo.reverse %s2b6dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b6dd = stablehlo.convolution(%s2b6dn, %s2b6ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b6ddWxt = stablehlo.transpose %s2b5o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b6ddWdt = stablehlo.transpose %s2b6dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b6ddWraw = stablehlo.convolution(%s2b6ddWxt, %s2b6ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b6ddW = stablehlo.reshape %s2b6ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b6ddb = stablehlo.reduce(%s2b6dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b6dx = stablehlo.add %s2b6dd, %s2b7dx : tensor<32x384x14x14xf32>
    %s2b5dlsgb = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b5dls = stablehlo.multiply %s2b5dlsgb, %s2b6dx : tensor<32x384x14x14xf32>
    %s2b5dlsxdy = stablehlo.multiply %s2b5p, %s2b6dx : tensor<32x384x14x14xf32>
    %s2b5dlsdg = stablehlo.reduce(%s2b5dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b5dpt = stablehlo.transpose %s2b5pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b5dp = stablehlo.convolution(%s2b5dls, %s2b5dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b5dpWxt = stablehlo.transpose %s2b5g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b5dpWdt = stablehlo.transpose %s2b5dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b5dpWraw = stablehlo.convolution(%s2b5dpWxt, %s2b5dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b5dpW = stablehlo.transpose %s2b5dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b5dpb = stablehlo.reduce(%s2b5dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b5dgx2 = stablehlo.multiply %s2b5e, %s2b5e : tensor<32x1536x14x14xf32>
    %s2b5dgx3 = stablehlo.multiply %s2b5dgx2, %s2b5e : tensor<32x1536x14x14xf32>
    %s2b5dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b5dgkx3 = stablehlo.multiply %s2b5dgck, %s2b5dgx3 : tensor<32x1536x14x14xf32>
    %s2b5dginn = stablehlo.add %s2b5e, %s2b5dgkx3 : tensor<32x1536x14x14xf32>
    %s2b5dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b5dgu = stablehlo.multiply %s2b5dgcs, %s2b5dginn : tensor<32x1536x14x14xf32>
    %s2b5dgt = stablehlo.tanh %s2b5dgu : tensor<32x1536x14x14xf32>
    %s2b5dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b5dgopt = stablehlo.add %s2b5dgone, %s2b5dgt : tensor<32x1536x14x14xf32>
    %s2b5dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b5dgterm1 = stablehlo.multiply %s2b5dghalf, %s2b5dgopt : tensor<32x1536x14x14xf32>
    %s2b5dgt2 = stablehlo.multiply %s2b5dgt, %s2b5dgt : tensor<32x1536x14x14xf32>
    %s2b5dgomt2 = stablehlo.subtract %s2b5dgone, %s2b5dgt2 : tensor<32x1536x14x14xf32>
    %s2b5dghx = stablehlo.multiply %s2b5dghalf, %s2b5e : tensor<32x1536x14x14xf32>
    %s2b5dghxo = stablehlo.multiply %s2b5dghx, %s2b5dgomt2 : tensor<32x1536x14x14xf32>
    %s2b5dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b5dga3x2 = stablehlo.multiply %s2b5dgc3b, %s2b5dgx2 : tensor<32x1536x14x14xf32>
    %s2b5dgin2 = stablehlo.add %s2b5dgone, %s2b5dga3x2 : tensor<32x1536x14x14xf32>
    %s2b5dgup = stablehlo.multiply %s2b5dgcs, %s2b5dgin2 : tensor<32x1536x14x14xf32>
    %s2b5dgterm2 = stablehlo.multiply %s2b5dghxo, %s2b5dgup : tensor<32x1536x14x14xf32>
    %s2b5dggp = stablehlo.add %s2b5dgterm1, %s2b5dgterm2 : tensor<32x1536x14x14xf32>
    %s2b5dg = stablehlo.multiply %s2b5dp, %s2b5dggp : tensor<32x1536x14x14xf32>
    %s2b5det = stablehlo.transpose %s2b5eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b5de = stablehlo.convolution(%s2b5dg, %s2b5det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b5deWxt = stablehlo.transpose %s2b5n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b5deWdt = stablehlo.transpose %s2b5dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b5deWraw = stablehlo.convolution(%s2b5deWxt, %s2b5deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b5deW = stablehlo.transpose %s2b5deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b5deb = stablehlo.reduce(%s2b5dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b5dnri = stablehlo.reshape %s2b5d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b5dnrdy = stablehlo.reshape %s2b5de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b5dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b5dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b5dnsmr = stablehlo.reduce(%s2b5dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b5dnsm = stablehlo.broadcast_in_dim %s2b5dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b5dnmu = stablehlo.divide %s2b5dnsm, %s2b5dnnf : tensor<32x75264xf32>
    %s2b5dnxc = stablehlo.subtract %s2b5dnri, %s2b5dnmu : tensor<32x75264xf32>
    %s2b5dnsq = stablehlo.multiply %s2b5dnxc, %s2b5dnxc : tensor<32x75264xf32>
    %s2b5dnvsr = stablehlo.reduce(%s2b5dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b5dnvs = stablehlo.broadcast_in_dim %s2b5dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b5dnvr = stablehlo.divide %s2b5dnvs, %s2b5dnnf : tensor<32x75264xf32>
    %s2b5dnve = stablehlo.add %s2b5dnvr, %s2b5dnep : tensor<32x75264xf32>
    %s2b5dnistd = stablehlo.rsqrt %s2b5dnve : tensor<32x75264xf32>
    %s2b5dnxh = stablehlo.multiply %s2b5dnxc, %s2b5dnistd : tensor<32x75264xf32>
    %s2b5dngb = stablehlo.broadcast_in_dim %s2b5ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b5dndxh = stablehlo.multiply %s2b5dngb, %s2b5dnrdy : tensor<32x75264xf32>
    %s2b5dnsdxr = stablehlo.reduce(%s2b5dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b5dnsdx = stablehlo.broadcast_in_dim %s2b5dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b5dnxd = stablehlo.multiply %s2b5dnxh, %s2b5dndxh : tensor<32x75264xf32>
    %s2b5dnsxdr = stablehlo.reduce(%s2b5dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b5dnsxd = stablehlo.broadcast_in_dim %s2b5dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b5dnt1 = stablehlo.multiply %s2b5dndxh, %s2b5dnnf : tensor<32x75264xf32>
    %s2b5dni1 = stablehlo.subtract %s2b5dnt1, %s2b5dnsdx : tensor<32x75264xf32>
    %s2b5dnxs = stablehlo.multiply %s2b5dnxh, %s2b5dnsxd : tensor<32x75264xf32>
    %s2b5dni2 = stablehlo.subtract %s2b5dni1, %s2b5dnxs : tensor<32x75264xf32>
    %s2b5dnsN = stablehlo.divide %s2b5dnistd, %s2b5dnnf : tensor<32x75264xf32>
    %s2b5dngin = stablehlo.multiply %s2b5dnsN, %s2b5dni2 : tensor<32x75264xf32>
    %s2b5dn = stablehlo.reshape %s2b5dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b5dndgp = stablehlo.multiply %s2b5dnrdy, %s2b5dnxh : tensor<32x75264xf32>
    %s2b5dndg = stablehlo.reduce(%s2b5dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b5dndb = stablehlo.reduce(%s2b5dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b5ddrev = stablehlo.reverse %s2b5dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b5dd = stablehlo.convolution(%s2b5dn, %s2b5ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b5ddWxt = stablehlo.transpose %s2b4o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b5ddWdt = stablehlo.transpose %s2b5dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b5ddWraw = stablehlo.convolution(%s2b5ddWxt, %s2b5ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b5ddW = stablehlo.reshape %s2b5ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b5ddb = stablehlo.reduce(%s2b5dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b5dx = stablehlo.add %s2b5dd, %s2b6dx : tensor<32x384x14x14xf32>
    %s2b4dlsgb = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b4dls = stablehlo.multiply %s2b4dlsgb, %s2b5dx : tensor<32x384x14x14xf32>
    %s2b4dlsxdy = stablehlo.multiply %s2b4p, %s2b5dx : tensor<32x384x14x14xf32>
    %s2b4dlsdg = stablehlo.reduce(%s2b4dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b4dpt = stablehlo.transpose %s2b4pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b4dp = stablehlo.convolution(%s2b4dls, %s2b4dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b4dpWxt = stablehlo.transpose %s2b4g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b4dpWdt = stablehlo.transpose %s2b4dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b4dpWraw = stablehlo.convolution(%s2b4dpWxt, %s2b4dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b4dpW = stablehlo.transpose %s2b4dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b4dpb = stablehlo.reduce(%s2b4dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b4dgx2 = stablehlo.multiply %s2b4e, %s2b4e : tensor<32x1536x14x14xf32>
    %s2b4dgx3 = stablehlo.multiply %s2b4dgx2, %s2b4e : tensor<32x1536x14x14xf32>
    %s2b4dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b4dgkx3 = stablehlo.multiply %s2b4dgck, %s2b4dgx3 : tensor<32x1536x14x14xf32>
    %s2b4dginn = stablehlo.add %s2b4e, %s2b4dgkx3 : tensor<32x1536x14x14xf32>
    %s2b4dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b4dgu = stablehlo.multiply %s2b4dgcs, %s2b4dginn : tensor<32x1536x14x14xf32>
    %s2b4dgt = stablehlo.tanh %s2b4dgu : tensor<32x1536x14x14xf32>
    %s2b4dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b4dgopt = stablehlo.add %s2b4dgone, %s2b4dgt : tensor<32x1536x14x14xf32>
    %s2b4dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b4dgterm1 = stablehlo.multiply %s2b4dghalf, %s2b4dgopt : tensor<32x1536x14x14xf32>
    %s2b4dgt2 = stablehlo.multiply %s2b4dgt, %s2b4dgt : tensor<32x1536x14x14xf32>
    %s2b4dgomt2 = stablehlo.subtract %s2b4dgone, %s2b4dgt2 : tensor<32x1536x14x14xf32>
    %s2b4dghx = stablehlo.multiply %s2b4dghalf, %s2b4e : tensor<32x1536x14x14xf32>
    %s2b4dghxo = stablehlo.multiply %s2b4dghx, %s2b4dgomt2 : tensor<32x1536x14x14xf32>
    %s2b4dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b4dga3x2 = stablehlo.multiply %s2b4dgc3b, %s2b4dgx2 : tensor<32x1536x14x14xf32>
    %s2b4dgin2 = stablehlo.add %s2b4dgone, %s2b4dga3x2 : tensor<32x1536x14x14xf32>
    %s2b4dgup = stablehlo.multiply %s2b4dgcs, %s2b4dgin2 : tensor<32x1536x14x14xf32>
    %s2b4dgterm2 = stablehlo.multiply %s2b4dghxo, %s2b4dgup : tensor<32x1536x14x14xf32>
    %s2b4dggp = stablehlo.add %s2b4dgterm1, %s2b4dgterm2 : tensor<32x1536x14x14xf32>
    %s2b4dg = stablehlo.multiply %s2b4dp, %s2b4dggp : tensor<32x1536x14x14xf32>
    %s2b4det = stablehlo.transpose %s2b4eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b4de = stablehlo.convolution(%s2b4dg, %s2b4det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b4deWxt = stablehlo.transpose %s2b4n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b4deWdt = stablehlo.transpose %s2b4dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b4deWraw = stablehlo.convolution(%s2b4deWxt, %s2b4deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b4deW = stablehlo.transpose %s2b4deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b4deb = stablehlo.reduce(%s2b4dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b4dnri = stablehlo.reshape %s2b4d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b4dnrdy = stablehlo.reshape %s2b4de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b4dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b4dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b4dnsmr = stablehlo.reduce(%s2b4dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b4dnsm = stablehlo.broadcast_in_dim %s2b4dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b4dnmu = stablehlo.divide %s2b4dnsm, %s2b4dnnf : tensor<32x75264xf32>
    %s2b4dnxc = stablehlo.subtract %s2b4dnri, %s2b4dnmu : tensor<32x75264xf32>
    %s2b4dnsq = stablehlo.multiply %s2b4dnxc, %s2b4dnxc : tensor<32x75264xf32>
    %s2b4dnvsr = stablehlo.reduce(%s2b4dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b4dnvs = stablehlo.broadcast_in_dim %s2b4dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b4dnvr = stablehlo.divide %s2b4dnvs, %s2b4dnnf : tensor<32x75264xf32>
    %s2b4dnve = stablehlo.add %s2b4dnvr, %s2b4dnep : tensor<32x75264xf32>
    %s2b4dnistd = stablehlo.rsqrt %s2b4dnve : tensor<32x75264xf32>
    %s2b4dnxh = stablehlo.multiply %s2b4dnxc, %s2b4dnistd : tensor<32x75264xf32>
    %s2b4dngb = stablehlo.broadcast_in_dim %s2b4ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b4dndxh = stablehlo.multiply %s2b4dngb, %s2b4dnrdy : tensor<32x75264xf32>
    %s2b4dnsdxr = stablehlo.reduce(%s2b4dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b4dnsdx = stablehlo.broadcast_in_dim %s2b4dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b4dnxd = stablehlo.multiply %s2b4dnxh, %s2b4dndxh : tensor<32x75264xf32>
    %s2b4dnsxdr = stablehlo.reduce(%s2b4dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b4dnsxd = stablehlo.broadcast_in_dim %s2b4dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b4dnt1 = stablehlo.multiply %s2b4dndxh, %s2b4dnnf : tensor<32x75264xf32>
    %s2b4dni1 = stablehlo.subtract %s2b4dnt1, %s2b4dnsdx : tensor<32x75264xf32>
    %s2b4dnxs = stablehlo.multiply %s2b4dnxh, %s2b4dnsxd : tensor<32x75264xf32>
    %s2b4dni2 = stablehlo.subtract %s2b4dni1, %s2b4dnxs : tensor<32x75264xf32>
    %s2b4dnsN = stablehlo.divide %s2b4dnistd, %s2b4dnnf : tensor<32x75264xf32>
    %s2b4dngin = stablehlo.multiply %s2b4dnsN, %s2b4dni2 : tensor<32x75264xf32>
    %s2b4dn = stablehlo.reshape %s2b4dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b4dndgp = stablehlo.multiply %s2b4dnrdy, %s2b4dnxh : tensor<32x75264xf32>
    %s2b4dndg = stablehlo.reduce(%s2b4dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b4dndb = stablehlo.reduce(%s2b4dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b4ddrev = stablehlo.reverse %s2b4dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b4dd = stablehlo.convolution(%s2b4dn, %s2b4ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b4ddWxt = stablehlo.transpose %s2b3o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b4ddWdt = stablehlo.transpose %s2b4dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b4ddWraw = stablehlo.convolution(%s2b4ddWxt, %s2b4ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b4ddW = stablehlo.reshape %s2b4ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b4ddb = stablehlo.reduce(%s2b4dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b4dx = stablehlo.add %s2b4dd, %s2b5dx : tensor<32x384x14x14xf32>
    %s2b3dlsgb = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b3dls = stablehlo.multiply %s2b3dlsgb, %s2b4dx : tensor<32x384x14x14xf32>
    %s2b3dlsxdy = stablehlo.multiply %s2b3p, %s2b4dx : tensor<32x384x14x14xf32>
    %s2b3dlsdg = stablehlo.reduce(%s2b3dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b3dpt = stablehlo.transpose %s2b3pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b3dp = stablehlo.convolution(%s2b3dls, %s2b3dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b3dpWxt = stablehlo.transpose %s2b3g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b3dpWdt = stablehlo.transpose %s2b3dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b3dpWraw = stablehlo.convolution(%s2b3dpWxt, %s2b3dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b3dpW = stablehlo.transpose %s2b3dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b3dpb = stablehlo.reduce(%s2b3dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b3dgx2 = stablehlo.multiply %s2b3e, %s2b3e : tensor<32x1536x14x14xf32>
    %s2b3dgx3 = stablehlo.multiply %s2b3dgx2, %s2b3e : tensor<32x1536x14x14xf32>
    %s2b3dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b3dgkx3 = stablehlo.multiply %s2b3dgck, %s2b3dgx3 : tensor<32x1536x14x14xf32>
    %s2b3dginn = stablehlo.add %s2b3e, %s2b3dgkx3 : tensor<32x1536x14x14xf32>
    %s2b3dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b3dgu = stablehlo.multiply %s2b3dgcs, %s2b3dginn : tensor<32x1536x14x14xf32>
    %s2b3dgt = stablehlo.tanh %s2b3dgu : tensor<32x1536x14x14xf32>
    %s2b3dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b3dgopt = stablehlo.add %s2b3dgone, %s2b3dgt : tensor<32x1536x14x14xf32>
    %s2b3dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b3dgterm1 = stablehlo.multiply %s2b3dghalf, %s2b3dgopt : tensor<32x1536x14x14xf32>
    %s2b3dgt2 = stablehlo.multiply %s2b3dgt, %s2b3dgt : tensor<32x1536x14x14xf32>
    %s2b3dgomt2 = stablehlo.subtract %s2b3dgone, %s2b3dgt2 : tensor<32x1536x14x14xf32>
    %s2b3dghx = stablehlo.multiply %s2b3dghalf, %s2b3e : tensor<32x1536x14x14xf32>
    %s2b3dghxo = stablehlo.multiply %s2b3dghx, %s2b3dgomt2 : tensor<32x1536x14x14xf32>
    %s2b3dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b3dga3x2 = stablehlo.multiply %s2b3dgc3b, %s2b3dgx2 : tensor<32x1536x14x14xf32>
    %s2b3dgin2 = stablehlo.add %s2b3dgone, %s2b3dga3x2 : tensor<32x1536x14x14xf32>
    %s2b3dgup = stablehlo.multiply %s2b3dgcs, %s2b3dgin2 : tensor<32x1536x14x14xf32>
    %s2b3dgterm2 = stablehlo.multiply %s2b3dghxo, %s2b3dgup : tensor<32x1536x14x14xf32>
    %s2b3dggp = stablehlo.add %s2b3dgterm1, %s2b3dgterm2 : tensor<32x1536x14x14xf32>
    %s2b3dg = stablehlo.multiply %s2b3dp, %s2b3dggp : tensor<32x1536x14x14xf32>
    %s2b3det = stablehlo.transpose %s2b3eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b3de = stablehlo.convolution(%s2b3dg, %s2b3det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b3deWxt = stablehlo.transpose %s2b3n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b3deWdt = stablehlo.transpose %s2b3dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b3deWraw = stablehlo.convolution(%s2b3deWxt, %s2b3deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b3deW = stablehlo.transpose %s2b3deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b3deb = stablehlo.reduce(%s2b3dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b3dnri = stablehlo.reshape %s2b3d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b3dnrdy = stablehlo.reshape %s2b3de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b3dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b3dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b3dnsmr = stablehlo.reduce(%s2b3dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b3dnsm = stablehlo.broadcast_in_dim %s2b3dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b3dnmu = stablehlo.divide %s2b3dnsm, %s2b3dnnf : tensor<32x75264xf32>
    %s2b3dnxc = stablehlo.subtract %s2b3dnri, %s2b3dnmu : tensor<32x75264xf32>
    %s2b3dnsq = stablehlo.multiply %s2b3dnxc, %s2b3dnxc : tensor<32x75264xf32>
    %s2b3dnvsr = stablehlo.reduce(%s2b3dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b3dnvs = stablehlo.broadcast_in_dim %s2b3dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b3dnvr = stablehlo.divide %s2b3dnvs, %s2b3dnnf : tensor<32x75264xf32>
    %s2b3dnve = stablehlo.add %s2b3dnvr, %s2b3dnep : tensor<32x75264xf32>
    %s2b3dnistd = stablehlo.rsqrt %s2b3dnve : tensor<32x75264xf32>
    %s2b3dnxh = stablehlo.multiply %s2b3dnxc, %s2b3dnistd : tensor<32x75264xf32>
    %s2b3dngb = stablehlo.broadcast_in_dim %s2b3ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b3dndxh = stablehlo.multiply %s2b3dngb, %s2b3dnrdy : tensor<32x75264xf32>
    %s2b3dnsdxr = stablehlo.reduce(%s2b3dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b3dnsdx = stablehlo.broadcast_in_dim %s2b3dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b3dnxd = stablehlo.multiply %s2b3dnxh, %s2b3dndxh : tensor<32x75264xf32>
    %s2b3dnsxdr = stablehlo.reduce(%s2b3dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b3dnsxd = stablehlo.broadcast_in_dim %s2b3dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b3dnt1 = stablehlo.multiply %s2b3dndxh, %s2b3dnnf : tensor<32x75264xf32>
    %s2b3dni1 = stablehlo.subtract %s2b3dnt1, %s2b3dnsdx : tensor<32x75264xf32>
    %s2b3dnxs = stablehlo.multiply %s2b3dnxh, %s2b3dnsxd : tensor<32x75264xf32>
    %s2b3dni2 = stablehlo.subtract %s2b3dni1, %s2b3dnxs : tensor<32x75264xf32>
    %s2b3dnsN = stablehlo.divide %s2b3dnistd, %s2b3dnnf : tensor<32x75264xf32>
    %s2b3dngin = stablehlo.multiply %s2b3dnsN, %s2b3dni2 : tensor<32x75264xf32>
    %s2b3dn = stablehlo.reshape %s2b3dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b3dndgp = stablehlo.multiply %s2b3dnrdy, %s2b3dnxh : tensor<32x75264xf32>
    %s2b3dndg = stablehlo.reduce(%s2b3dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b3dndb = stablehlo.reduce(%s2b3dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b3ddrev = stablehlo.reverse %s2b3dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b3dd = stablehlo.convolution(%s2b3dn, %s2b3ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b3ddWxt = stablehlo.transpose %s2b2o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b3ddWdt = stablehlo.transpose %s2b3dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b3ddWraw = stablehlo.convolution(%s2b3ddWxt, %s2b3ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b3ddW = stablehlo.reshape %s2b3ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b3ddb = stablehlo.reduce(%s2b3dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b3dx = stablehlo.add %s2b3dd, %s2b4dx : tensor<32x384x14x14xf32>
    %s2b2dlsgb = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b2dls = stablehlo.multiply %s2b2dlsgb, %s2b3dx : tensor<32x384x14x14xf32>
    %s2b2dlsxdy = stablehlo.multiply %s2b2p, %s2b3dx : tensor<32x384x14x14xf32>
    %s2b2dlsdg = stablehlo.reduce(%s2b2dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b2dpt = stablehlo.transpose %s2b2pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b2dp = stablehlo.convolution(%s2b2dls, %s2b2dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b2dpWxt = stablehlo.transpose %s2b2g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b2dpWdt = stablehlo.transpose %s2b2dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b2dpWraw = stablehlo.convolution(%s2b2dpWxt, %s2b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b2dpW = stablehlo.transpose %s2b2dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b2dpb = stablehlo.reduce(%s2b2dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b2dgx2 = stablehlo.multiply %s2b2e, %s2b2e : tensor<32x1536x14x14xf32>
    %s2b2dgx3 = stablehlo.multiply %s2b2dgx2, %s2b2e : tensor<32x1536x14x14xf32>
    %s2b2dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b2dgkx3 = stablehlo.multiply %s2b2dgck, %s2b2dgx3 : tensor<32x1536x14x14xf32>
    %s2b2dginn = stablehlo.add %s2b2e, %s2b2dgkx3 : tensor<32x1536x14x14xf32>
    %s2b2dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b2dgu = stablehlo.multiply %s2b2dgcs, %s2b2dginn : tensor<32x1536x14x14xf32>
    %s2b2dgt = stablehlo.tanh %s2b2dgu : tensor<32x1536x14x14xf32>
    %s2b2dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b2dgopt = stablehlo.add %s2b2dgone, %s2b2dgt : tensor<32x1536x14x14xf32>
    %s2b2dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b2dgterm1 = stablehlo.multiply %s2b2dghalf, %s2b2dgopt : tensor<32x1536x14x14xf32>
    %s2b2dgt2 = stablehlo.multiply %s2b2dgt, %s2b2dgt : tensor<32x1536x14x14xf32>
    %s2b2dgomt2 = stablehlo.subtract %s2b2dgone, %s2b2dgt2 : tensor<32x1536x14x14xf32>
    %s2b2dghx = stablehlo.multiply %s2b2dghalf, %s2b2e : tensor<32x1536x14x14xf32>
    %s2b2dghxo = stablehlo.multiply %s2b2dghx, %s2b2dgomt2 : tensor<32x1536x14x14xf32>
    %s2b2dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b2dga3x2 = stablehlo.multiply %s2b2dgc3b, %s2b2dgx2 : tensor<32x1536x14x14xf32>
    %s2b2dgin2 = stablehlo.add %s2b2dgone, %s2b2dga3x2 : tensor<32x1536x14x14xf32>
    %s2b2dgup = stablehlo.multiply %s2b2dgcs, %s2b2dgin2 : tensor<32x1536x14x14xf32>
    %s2b2dgterm2 = stablehlo.multiply %s2b2dghxo, %s2b2dgup : tensor<32x1536x14x14xf32>
    %s2b2dggp = stablehlo.add %s2b2dgterm1, %s2b2dgterm2 : tensor<32x1536x14x14xf32>
    %s2b2dg = stablehlo.multiply %s2b2dp, %s2b2dggp : tensor<32x1536x14x14xf32>
    %s2b2det = stablehlo.transpose %s2b2eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b2de = stablehlo.convolution(%s2b2dg, %s2b2det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b2deWxt = stablehlo.transpose %s2b2n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b2deWdt = stablehlo.transpose %s2b2dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b2deWraw = stablehlo.convolution(%s2b2deWxt, %s2b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b2deW = stablehlo.transpose %s2b2deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b2deb = stablehlo.reduce(%s2b2dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b2dnri = stablehlo.reshape %s2b2d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b2dnrdy = stablehlo.reshape %s2b2de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b2dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b2dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b2dnsmr = stablehlo.reduce(%s2b2dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b2dnsm = stablehlo.broadcast_in_dim %s2b2dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b2dnmu = stablehlo.divide %s2b2dnsm, %s2b2dnnf : tensor<32x75264xf32>
    %s2b2dnxc = stablehlo.subtract %s2b2dnri, %s2b2dnmu : tensor<32x75264xf32>
    %s2b2dnsq = stablehlo.multiply %s2b2dnxc, %s2b2dnxc : tensor<32x75264xf32>
    %s2b2dnvsr = stablehlo.reduce(%s2b2dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b2dnvs = stablehlo.broadcast_in_dim %s2b2dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b2dnvr = stablehlo.divide %s2b2dnvs, %s2b2dnnf : tensor<32x75264xf32>
    %s2b2dnve = stablehlo.add %s2b2dnvr, %s2b2dnep : tensor<32x75264xf32>
    %s2b2dnistd = stablehlo.rsqrt %s2b2dnve : tensor<32x75264xf32>
    %s2b2dnxh = stablehlo.multiply %s2b2dnxc, %s2b2dnistd : tensor<32x75264xf32>
    %s2b2dngb = stablehlo.broadcast_in_dim %s2b2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b2dndxh = stablehlo.multiply %s2b2dngb, %s2b2dnrdy : tensor<32x75264xf32>
    %s2b2dnsdxr = stablehlo.reduce(%s2b2dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b2dnsdx = stablehlo.broadcast_in_dim %s2b2dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b2dnxd = stablehlo.multiply %s2b2dnxh, %s2b2dndxh : tensor<32x75264xf32>
    %s2b2dnsxdr = stablehlo.reduce(%s2b2dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b2dnsxd = stablehlo.broadcast_in_dim %s2b2dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b2dnt1 = stablehlo.multiply %s2b2dndxh, %s2b2dnnf : tensor<32x75264xf32>
    %s2b2dni1 = stablehlo.subtract %s2b2dnt1, %s2b2dnsdx : tensor<32x75264xf32>
    %s2b2dnxs = stablehlo.multiply %s2b2dnxh, %s2b2dnsxd : tensor<32x75264xf32>
    %s2b2dni2 = stablehlo.subtract %s2b2dni1, %s2b2dnxs : tensor<32x75264xf32>
    %s2b2dnsN = stablehlo.divide %s2b2dnistd, %s2b2dnnf : tensor<32x75264xf32>
    %s2b2dngin = stablehlo.multiply %s2b2dnsN, %s2b2dni2 : tensor<32x75264xf32>
    %s2b2dn = stablehlo.reshape %s2b2dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b2dndgp = stablehlo.multiply %s2b2dnrdy, %s2b2dnxh : tensor<32x75264xf32>
    %s2b2dndg = stablehlo.reduce(%s2b2dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b2dndb = stablehlo.reduce(%s2b2dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b2ddrev = stablehlo.reverse %s2b2dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b2dd = stablehlo.convolution(%s2b2dn, %s2b2ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b2ddWxt = stablehlo.transpose %s2b1o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b2ddWdt = stablehlo.transpose %s2b2dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b2ddWraw = stablehlo.convolution(%s2b2ddWxt, %s2b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b2ddW = stablehlo.reshape %s2b2ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b2ddb = stablehlo.reduce(%s2b2dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b2dx = stablehlo.add %s2b2dd, %s2b3dx : tensor<32x384x14x14xf32>
    %s2b1dlsgb = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b1dls = stablehlo.multiply %s2b1dlsgb, %s2b2dx : tensor<32x384x14x14xf32>
    %s2b1dlsxdy = stablehlo.multiply %s2b1p, %s2b2dx : tensor<32x384x14x14xf32>
    %s2b1dlsdg = stablehlo.reduce(%s2b1dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b1dpt = stablehlo.transpose %s2b1pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b1dp = stablehlo.convolution(%s2b1dls, %s2b1dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b1dpWxt = stablehlo.transpose %s2b1g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b1dpWdt = stablehlo.transpose %s2b1dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b1dpWraw = stablehlo.convolution(%s2b1dpWxt, %s2b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b1dpW = stablehlo.transpose %s2b1dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b1dpb = stablehlo.reduce(%s2b1dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b1dgx2 = stablehlo.multiply %s2b1e, %s2b1e : tensor<32x1536x14x14xf32>
    %s2b1dgx3 = stablehlo.multiply %s2b1dgx2, %s2b1e : tensor<32x1536x14x14xf32>
    %s2b1dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b1dgkx3 = stablehlo.multiply %s2b1dgck, %s2b1dgx3 : tensor<32x1536x14x14xf32>
    %s2b1dginn = stablehlo.add %s2b1e, %s2b1dgkx3 : tensor<32x1536x14x14xf32>
    %s2b1dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b1dgu = stablehlo.multiply %s2b1dgcs, %s2b1dginn : tensor<32x1536x14x14xf32>
    %s2b1dgt = stablehlo.tanh %s2b1dgu : tensor<32x1536x14x14xf32>
    %s2b1dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b1dgopt = stablehlo.add %s2b1dgone, %s2b1dgt : tensor<32x1536x14x14xf32>
    %s2b1dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b1dgterm1 = stablehlo.multiply %s2b1dghalf, %s2b1dgopt : tensor<32x1536x14x14xf32>
    %s2b1dgt2 = stablehlo.multiply %s2b1dgt, %s2b1dgt : tensor<32x1536x14x14xf32>
    %s2b1dgomt2 = stablehlo.subtract %s2b1dgone, %s2b1dgt2 : tensor<32x1536x14x14xf32>
    %s2b1dghx = stablehlo.multiply %s2b1dghalf, %s2b1e : tensor<32x1536x14x14xf32>
    %s2b1dghxo = stablehlo.multiply %s2b1dghx, %s2b1dgomt2 : tensor<32x1536x14x14xf32>
    %s2b1dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b1dga3x2 = stablehlo.multiply %s2b1dgc3b, %s2b1dgx2 : tensor<32x1536x14x14xf32>
    %s2b1dgin2 = stablehlo.add %s2b1dgone, %s2b1dga3x2 : tensor<32x1536x14x14xf32>
    %s2b1dgup = stablehlo.multiply %s2b1dgcs, %s2b1dgin2 : tensor<32x1536x14x14xf32>
    %s2b1dgterm2 = stablehlo.multiply %s2b1dghxo, %s2b1dgup : tensor<32x1536x14x14xf32>
    %s2b1dggp = stablehlo.add %s2b1dgterm1, %s2b1dgterm2 : tensor<32x1536x14x14xf32>
    %s2b1dg = stablehlo.multiply %s2b1dp, %s2b1dggp : tensor<32x1536x14x14xf32>
    %s2b1det = stablehlo.transpose %s2b1eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b1de = stablehlo.convolution(%s2b1dg, %s2b1det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b1deWxt = stablehlo.transpose %s2b1n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b1deWdt = stablehlo.transpose %s2b1dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b1deWraw = stablehlo.convolution(%s2b1deWxt, %s2b1deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b1deW = stablehlo.transpose %s2b1deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b1deb = stablehlo.reduce(%s2b1dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b1dnri = stablehlo.reshape %s2b1d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b1dnrdy = stablehlo.reshape %s2b1de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b1dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b1dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b1dnsmr = stablehlo.reduce(%s2b1dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b1dnsm = stablehlo.broadcast_in_dim %s2b1dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b1dnmu = stablehlo.divide %s2b1dnsm, %s2b1dnnf : tensor<32x75264xf32>
    %s2b1dnxc = stablehlo.subtract %s2b1dnri, %s2b1dnmu : tensor<32x75264xf32>
    %s2b1dnsq = stablehlo.multiply %s2b1dnxc, %s2b1dnxc : tensor<32x75264xf32>
    %s2b1dnvsr = stablehlo.reduce(%s2b1dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b1dnvs = stablehlo.broadcast_in_dim %s2b1dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b1dnvr = stablehlo.divide %s2b1dnvs, %s2b1dnnf : tensor<32x75264xf32>
    %s2b1dnve = stablehlo.add %s2b1dnvr, %s2b1dnep : tensor<32x75264xf32>
    %s2b1dnistd = stablehlo.rsqrt %s2b1dnve : tensor<32x75264xf32>
    %s2b1dnxh = stablehlo.multiply %s2b1dnxc, %s2b1dnistd : tensor<32x75264xf32>
    %s2b1dngb = stablehlo.broadcast_in_dim %s2b1ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b1dndxh = stablehlo.multiply %s2b1dngb, %s2b1dnrdy : tensor<32x75264xf32>
    %s2b1dnsdxr = stablehlo.reduce(%s2b1dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b1dnsdx = stablehlo.broadcast_in_dim %s2b1dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b1dnxd = stablehlo.multiply %s2b1dnxh, %s2b1dndxh : tensor<32x75264xf32>
    %s2b1dnsxdr = stablehlo.reduce(%s2b1dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b1dnsxd = stablehlo.broadcast_in_dim %s2b1dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b1dnt1 = stablehlo.multiply %s2b1dndxh, %s2b1dnnf : tensor<32x75264xf32>
    %s2b1dni1 = stablehlo.subtract %s2b1dnt1, %s2b1dnsdx : tensor<32x75264xf32>
    %s2b1dnxs = stablehlo.multiply %s2b1dnxh, %s2b1dnsxd : tensor<32x75264xf32>
    %s2b1dni2 = stablehlo.subtract %s2b1dni1, %s2b1dnxs : tensor<32x75264xf32>
    %s2b1dnsN = stablehlo.divide %s2b1dnistd, %s2b1dnnf : tensor<32x75264xf32>
    %s2b1dngin = stablehlo.multiply %s2b1dnsN, %s2b1dni2 : tensor<32x75264xf32>
    %s2b1dn = stablehlo.reshape %s2b1dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b1dndgp = stablehlo.multiply %s2b1dnrdy, %s2b1dnxh : tensor<32x75264xf32>
    %s2b1dndg = stablehlo.reduce(%s2b1dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b1dndb = stablehlo.reduce(%s2b1dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b1ddrev = stablehlo.reverse %s2b1dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b1dd = stablehlo.convolution(%s2b1dn, %s2b1ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b1ddWxt = stablehlo.transpose %s2b0o, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b1ddWdt = stablehlo.transpose %s2b1dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b1ddWraw = stablehlo.convolution(%s2b1ddWxt, %s2b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b1ddW = stablehlo.reshape %s2b1ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b1ddb = stablehlo.reduce(%s2b1dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b1dx = stablehlo.add %s2b1dd, %s2b2dx : tensor<32x384x14x14xf32>
    %s2b0dlsgb = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %s2b0dls = stablehlo.multiply %s2b0dlsgb, %s2b1dx : tensor<32x384x14x14xf32>
    %s2b0dlsxdy = stablehlo.multiply %s2b0p, %s2b1dx : tensor<32x384x14x14xf32>
    %s2b0dlsdg = stablehlo.reduce(%s2b0dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b0dpt = stablehlo.transpose %s2b0pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b0dp = stablehlo.convolution(%s2b0dls, %s2b0dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %s2b0dpWxt = stablehlo.transpose %s2b0g, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b0dpWdt = stablehlo.transpose %s2b0dls, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b0dpWraw = stablehlo.convolution(%s2b0dpWxt, %s2b0dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %s2b0dpW = stablehlo.transpose %s2b0dpWraw, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b0dpb = stablehlo.reduce(%s2b0dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b0dgx2 = stablehlo.multiply %s2b0e, %s2b0e : tensor<32x1536x14x14xf32>
    %s2b0dgx3 = stablehlo.multiply %s2b0dgx2, %s2b0e : tensor<32x1536x14x14xf32>
    %s2b0dgck = stablehlo.constant dense<0.044715> : tensor<32x1536x14x14xf32>
    %s2b0dgkx3 = stablehlo.multiply %s2b0dgck, %s2b0dgx3 : tensor<32x1536x14x14xf32>
    %s2b0dginn = stablehlo.add %s2b0e, %s2b0dgkx3 : tensor<32x1536x14x14xf32>
    %s2b0dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x1536x14x14xf32>
    %s2b0dgu = stablehlo.multiply %s2b0dgcs, %s2b0dginn : tensor<32x1536x14x14xf32>
    %s2b0dgt = stablehlo.tanh %s2b0dgu : tensor<32x1536x14x14xf32>
    %s2b0dgone = stablehlo.constant dense<1.0> : tensor<32x1536x14x14xf32>
    %s2b0dgopt = stablehlo.add %s2b0dgone, %s2b0dgt : tensor<32x1536x14x14xf32>
    %s2b0dghalf = stablehlo.constant dense<0.5> : tensor<32x1536x14x14xf32>
    %s2b0dgterm1 = stablehlo.multiply %s2b0dghalf, %s2b0dgopt : tensor<32x1536x14x14xf32>
    %s2b0dgt2 = stablehlo.multiply %s2b0dgt, %s2b0dgt : tensor<32x1536x14x14xf32>
    %s2b0dgomt2 = stablehlo.subtract %s2b0dgone, %s2b0dgt2 : tensor<32x1536x14x14xf32>
    %s2b0dghx = stablehlo.multiply %s2b0dghalf, %s2b0e : tensor<32x1536x14x14xf32>
    %s2b0dghxo = stablehlo.multiply %s2b0dghx, %s2b0dgomt2 : tensor<32x1536x14x14xf32>
    %s2b0dgc3b = stablehlo.constant dense<0.134145> : tensor<32x1536x14x14xf32>
    %s2b0dga3x2 = stablehlo.multiply %s2b0dgc3b, %s2b0dgx2 : tensor<32x1536x14x14xf32>
    %s2b0dgin2 = stablehlo.add %s2b0dgone, %s2b0dga3x2 : tensor<32x1536x14x14xf32>
    %s2b0dgup = stablehlo.multiply %s2b0dgcs, %s2b0dgin2 : tensor<32x1536x14x14xf32>
    %s2b0dgterm2 = stablehlo.multiply %s2b0dghxo, %s2b0dgup : tensor<32x1536x14x14xf32>
    %s2b0dggp = stablehlo.add %s2b0dgterm1, %s2b0dgterm2 : tensor<32x1536x14x14xf32>
    %s2b0dg = stablehlo.multiply %s2b0dp, %s2b0dggp : tensor<32x1536x14x14xf32>
    %s2b0det = stablehlo.transpose %s2b0eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %s2b0de = stablehlo.convolution(%s2b0dg, %s2b0det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %s2b0deWxt = stablehlo.transpose %s2b0n, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b0deWdt = stablehlo.transpose %s2b0dg, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %s2b0deWraw = stablehlo.convolution(%s2b0deWxt, %s2b0deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %s2b0deW = stablehlo.transpose %s2b0deWraw, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %s2b0deb = stablehlo.reduce(%s2b0dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %s2b0dnri = stablehlo.reshape %s2b0d : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b0dnrdy = stablehlo.reshape %s2b0de : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %s2b0dnnf = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %s2b0dnep = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %s2b0dnsmr = stablehlo.reduce(%s2b0dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b0dnsm = stablehlo.broadcast_in_dim %s2b0dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b0dnmu = stablehlo.divide %s2b0dnsm, %s2b0dnnf : tensor<32x75264xf32>
    %s2b0dnxc = stablehlo.subtract %s2b0dnri, %s2b0dnmu : tensor<32x75264xf32>
    %s2b0dnsq = stablehlo.multiply %s2b0dnxc, %s2b0dnxc : tensor<32x75264xf32>
    %s2b0dnvsr = stablehlo.reduce(%s2b0dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b0dnvs = stablehlo.broadcast_in_dim %s2b0dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b0dnvr = stablehlo.divide %s2b0dnvs, %s2b0dnnf : tensor<32x75264xf32>
    %s2b0dnve = stablehlo.add %s2b0dnvr, %s2b0dnep : tensor<32x75264xf32>
    %s2b0dnistd = stablehlo.rsqrt %s2b0dnve : tensor<32x75264xf32>
    %s2b0dnxh = stablehlo.multiply %s2b0dnxc, %s2b0dnistd : tensor<32x75264xf32>
    %s2b0dngb = stablehlo.broadcast_in_dim %s2b0ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %s2b0dndxh = stablehlo.multiply %s2b0dngb, %s2b0dnrdy : tensor<32x75264xf32>
    %s2b0dnsdxr = stablehlo.reduce(%s2b0dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b0dnsdx = stablehlo.broadcast_in_dim %s2b0dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b0dnxd = stablehlo.multiply %s2b0dnxh, %s2b0dndxh : tensor<32x75264xf32>
    %s2b0dnsxdr = stablehlo.reduce(%s2b0dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %s2b0dnsxd = stablehlo.broadcast_in_dim %s2b0dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %s2b0dnt1 = stablehlo.multiply %s2b0dndxh, %s2b0dnnf : tensor<32x75264xf32>
    %s2b0dni1 = stablehlo.subtract %s2b0dnt1, %s2b0dnsdx : tensor<32x75264xf32>
    %s2b0dnxs = stablehlo.multiply %s2b0dnxh, %s2b0dnsxd : tensor<32x75264xf32>
    %s2b0dni2 = stablehlo.subtract %s2b0dni1, %s2b0dnxs : tensor<32x75264xf32>
    %s2b0dnsN = stablehlo.divide %s2b0dnistd, %s2b0dnnf : tensor<32x75264xf32>
    %s2b0dngin = stablehlo.multiply %s2b0dnsN, %s2b0dni2 : tensor<32x75264xf32>
    %s2b0dn = stablehlo.reshape %s2b0dngin : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %s2b0dndgp = stablehlo.multiply %s2b0dnrdy, %s2b0dnxh : tensor<32x75264xf32>
    %s2b0dndg = stablehlo.reduce(%s2b0dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b0dndb = stablehlo.reduce(%s2b0dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %s2b0ddrev = stablehlo.reverse %s2b0dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %s2b0dd = stablehlo.convolution(%s2b0dn, %s2b0ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %s2b0ddWxt = stablehlo.transpose %d1c, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b0ddWdt = stablehlo.transpose %s2b0dn, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %s2b0ddWraw = stablehlo.convolution(%s2b0ddWxt, %s2b0ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %s2b0ddW = stablehlo.reshape %s2b0ddWraw : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %s2b0ddb = stablehlo.reduce(%s2b0dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %s2b0dx = stablehlo.add %s2b0dd, %s2b1dx : tensor<32x384x14x14xf32>
    %d1dcu = stablehlo.pad %s2b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x28x28xf32>
    %d1dct = stablehlo.transpose %d1W, dims = [1, 0, 2, 3] : (tensor<384x192x2x2xf32>) -> tensor<192x384x2x2xf32>
    %d1dcr = stablehlo.reverse %d1dct, dims = [2, 3] : tensor<192x384x2x2xf32>
    %d1dc = stablehlo.convolution(%d1dcu, %d1dcr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x28x28xf32>, tensor<192x384x2x2xf32>) -> tensor<32x192x28x28xf32>
    %d1dWu = stablehlo.pad %s2b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x27x27xf32>
    %d1dWxt = stablehlo.transpose %d1n, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %d1dWdt = stablehlo.transpose %d1dWu, dims = [1, 0, 2, 3] : (tensor<32x384x27x27xf32>) -> tensor<384x32x27x27xf32>
    %d1dWraw = stablehlo.convolution(%d1dWxt, %d1dWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<384x32x27x27xf32>) -> tensor<192x384x2x2xf32>
    %d1dW = stablehlo.transpose %d1dWraw, dims = [1, 0, 2, 3] : (tensor<192x384x2x2xf32>) -> tensor<384x192x2x2xf32>
    %d1db = stablehlo.reduce(%s2b0dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %d1dnri = stablehlo.reshape %s1b2o : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %d1dnrdy = stablehlo.reshape %d1dc : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %d1dnnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %d1dnep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %d1dnsmr = stablehlo.reduce(%d1dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %d1dnsm = stablehlo.broadcast_in_dim %d1dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %d1dnmu = stablehlo.divide %d1dnsm, %d1dnnf : tensor<32x150528xf32>
    %d1dnxc = stablehlo.subtract %d1dnri, %d1dnmu : tensor<32x150528xf32>
    %d1dnsq = stablehlo.multiply %d1dnxc, %d1dnxc : tensor<32x150528xf32>
    %d1dnvsr = stablehlo.reduce(%d1dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %d1dnvs = stablehlo.broadcast_in_dim %d1dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %d1dnvr = stablehlo.divide %d1dnvs, %d1dnnf : tensor<32x150528xf32>
    %d1dnve = stablehlo.add %d1dnvr, %d1dnep : tensor<32x150528xf32>
    %d1dnistd = stablehlo.rsqrt %d1dnve : tensor<32x150528xf32>
    %d1dnxh = stablehlo.multiply %d1dnxc, %d1dnistd : tensor<32x150528xf32>
    %d1dngb = stablehlo.broadcast_in_dim %d1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %d1dndxh = stablehlo.multiply %d1dngb, %d1dnrdy : tensor<32x150528xf32>
    %d1dnsdxr = stablehlo.reduce(%d1dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %d1dnsdx = stablehlo.broadcast_in_dim %d1dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %d1dnxd = stablehlo.multiply %d1dnxh, %d1dndxh : tensor<32x150528xf32>
    %d1dnsxdr = stablehlo.reduce(%d1dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %d1dnsxd = stablehlo.broadcast_in_dim %d1dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %d1dnt1 = stablehlo.multiply %d1dndxh, %d1dnnf : tensor<32x150528xf32>
    %d1dni1 = stablehlo.subtract %d1dnt1, %d1dnsdx : tensor<32x150528xf32>
    %d1dnxs = stablehlo.multiply %d1dnxh, %d1dnsxd : tensor<32x150528xf32>
    %d1dni2 = stablehlo.subtract %d1dni1, %d1dnxs : tensor<32x150528xf32>
    %d1dnsN = stablehlo.divide %d1dnistd, %d1dnnf : tensor<32x150528xf32>
    %d1dngin = stablehlo.multiply %d1dnsN, %d1dni2 : tensor<32x150528xf32>
    %d1dn = stablehlo.reshape %d1dngin : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %d1dndgp = stablehlo.multiply %d1dnrdy, %d1dnxh : tensor<32x150528xf32>
    %d1dndg = stablehlo.reduce(%d1dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %d1dndb = stablehlo.reduce(%d1dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b2dlsgb = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b2dls = stablehlo.multiply %s1b2dlsgb, %d1dn : tensor<32x192x28x28xf32>
    %s1b2dlsxdy = stablehlo.multiply %s1b2p, %d1dn : tensor<32x192x28x28xf32>
    %s1b2dlsdg = stablehlo.reduce(%s1b2dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b2dpt = stablehlo.transpose %s1b2pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %s1b2dp = stablehlo.convolution(%s1b2dls, %s1b2dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %s1b2dpWxt = stablehlo.transpose %s1b2g, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %s1b2dpWdt = stablehlo.transpose %s1b2dls, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b2dpWraw = stablehlo.convolution(%s1b2dpWxt, %s1b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %s1b2dpW = stablehlo.transpose %s1b2dpWraw, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %s1b2dpb = stablehlo.reduce(%s1b2dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b2dgx2 = stablehlo.multiply %s1b2e, %s1b2e : tensor<32x768x28x28xf32>
    %s1b2dgx3 = stablehlo.multiply %s1b2dgx2, %s1b2e : tensor<32x768x28x28xf32>
    %s1b2dgck = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %s1b2dgkx3 = stablehlo.multiply %s1b2dgck, %s1b2dgx3 : tensor<32x768x28x28xf32>
    %s1b2dginn = stablehlo.add %s1b2e, %s1b2dgkx3 : tensor<32x768x28x28xf32>
    %s1b2dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %s1b2dgu = stablehlo.multiply %s1b2dgcs, %s1b2dginn : tensor<32x768x28x28xf32>
    %s1b2dgt = stablehlo.tanh %s1b2dgu : tensor<32x768x28x28xf32>
    %s1b2dgone = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %s1b2dgopt = stablehlo.add %s1b2dgone, %s1b2dgt : tensor<32x768x28x28xf32>
    %s1b2dghalf = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %s1b2dgterm1 = stablehlo.multiply %s1b2dghalf, %s1b2dgopt : tensor<32x768x28x28xf32>
    %s1b2dgt2 = stablehlo.multiply %s1b2dgt, %s1b2dgt : tensor<32x768x28x28xf32>
    %s1b2dgomt2 = stablehlo.subtract %s1b2dgone, %s1b2dgt2 : tensor<32x768x28x28xf32>
    %s1b2dghx = stablehlo.multiply %s1b2dghalf, %s1b2e : tensor<32x768x28x28xf32>
    %s1b2dghxo = stablehlo.multiply %s1b2dghx, %s1b2dgomt2 : tensor<32x768x28x28xf32>
    %s1b2dgc3b = stablehlo.constant dense<0.134145> : tensor<32x768x28x28xf32>
    %s1b2dga3x2 = stablehlo.multiply %s1b2dgc3b, %s1b2dgx2 : tensor<32x768x28x28xf32>
    %s1b2dgin2 = stablehlo.add %s1b2dgone, %s1b2dga3x2 : tensor<32x768x28x28xf32>
    %s1b2dgup = stablehlo.multiply %s1b2dgcs, %s1b2dgin2 : tensor<32x768x28x28xf32>
    %s1b2dgterm2 = stablehlo.multiply %s1b2dghxo, %s1b2dgup : tensor<32x768x28x28xf32>
    %s1b2dggp = stablehlo.add %s1b2dgterm1, %s1b2dgterm2 : tensor<32x768x28x28xf32>
    %s1b2dg = stablehlo.multiply %s1b2dp, %s1b2dggp : tensor<32x768x28x28xf32>
    %s1b2det = stablehlo.transpose %s1b2eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %s1b2de = stablehlo.convolution(%s1b2dg, %s1b2det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %s1b2deWxt = stablehlo.transpose %s1b2n, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b2deWdt = stablehlo.transpose %s1b2dg, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %s1b2deWraw = stablehlo.convolution(%s1b2deWxt, %s1b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %s1b2deW = stablehlo.transpose %s1b2deWraw, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %s1b2deb = stablehlo.reduce(%s1b2dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %s1b2dnri = stablehlo.reshape %s1b2d : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b2dnrdy = stablehlo.reshape %s1b2de : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b2dnnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %s1b2dnep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %s1b2dnsmr = stablehlo.reduce(%s1b2dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b2dnsm = stablehlo.broadcast_in_dim %s1b2dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b2dnmu = stablehlo.divide %s1b2dnsm, %s1b2dnnf : tensor<32x150528xf32>
    %s1b2dnxc = stablehlo.subtract %s1b2dnri, %s1b2dnmu : tensor<32x150528xf32>
    %s1b2dnsq = stablehlo.multiply %s1b2dnxc, %s1b2dnxc : tensor<32x150528xf32>
    %s1b2dnvsr = stablehlo.reduce(%s1b2dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b2dnvs = stablehlo.broadcast_in_dim %s1b2dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b2dnvr = stablehlo.divide %s1b2dnvs, %s1b2dnnf : tensor<32x150528xf32>
    %s1b2dnve = stablehlo.add %s1b2dnvr, %s1b2dnep : tensor<32x150528xf32>
    %s1b2dnistd = stablehlo.rsqrt %s1b2dnve : tensor<32x150528xf32>
    %s1b2dnxh = stablehlo.multiply %s1b2dnxc, %s1b2dnistd : tensor<32x150528xf32>
    %s1b2dngb = stablehlo.broadcast_in_dim %s1b2ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b2dndxh = stablehlo.multiply %s1b2dngb, %s1b2dnrdy : tensor<32x150528xf32>
    %s1b2dnsdxr = stablehlo.reduce(%s1b2dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b2dnsdx = stablehlo.broadcast_in_dim %s1b2dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b2dnxd = stablehlo.multiply %s1b2dnxh, %s1b2dndxh : tensor<32x150528xf32>
    %s1b2dnsxdr = stablehlo.reduce(%s1b2dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b2dnsxd = stablehlo.broadcast_in_dim %s1b2dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b2dnt1 = stablehlo.multiply %s1b2dndxh, %s1b2dnnf : tensor<32x150528xf32>
    %s1b2dni1 = stablehlo.subtract %s1b2dnt1, %s1b2dnsdx : tensor<32x150528xf32>
    %s1b2dnxs = stablehlo.multiply %s1b2dnxh, %s1b2dnsxd : tensor<32x150528xf32>
    %s1b2dni2 = stablehlo.subtract %s1b2dni1, %s1b2dnxs : tensor<32x150528xf32>
    %s1b2dnsN = stablehlo.divide %s1b2dnistd, %s1b2dnnf : tensor<32x150528xf32>
    %s1b2dngin = stablehlo.multiply %s1b2dnsN, %s1b2dni2 : tensor<32x150528xf32>
    %s1b2dn = stablehlo.reshape %s1b2dngin : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %s1b2dndgp = stablehlo.multiply %s1b2dnrdy, %s1b2dnxh : tensor<32x150528xf32>
    %s1b2dndg = stablehlo.reduce(%s1b2dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b2dndb = stablehlo.reduce(%s1b2dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b2ddrev = stablehlo.reverse %s1b2dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %s1b2dd = stablehlo.convolution(%s1b2dn, %s1b2ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %s1b2ddWxt = stablehlo.transpose %s1b1o, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b2ddWdt = stablehlo.transpose %s1b2dn, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b2ddWraw = stablehlo.convolution(%s1b2ddWxt, %s1b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %s1b2ddW = stablehlo.reshape %s1b2ddWraw : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %s1b2ddb = stablehlo.reduce(%s1b2dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b2dx = stablehlo.add %s1b2dd, %d1dn : tensor<32x192x28x28xf32>
    %s1b1dlsgb = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b1dls = stablehlo.multiply %s1b1dlsgb, %s1b2dx : tensor<32x192x28x28xf32>
    %s1b1dlsxdy = stablehlo.multiply %s1b1p, %s1b2dx : tensor<32x192x28x28xf32>
    %s1b1dlsdg = stablehlo.reduce(%s1b1dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b1dpt = stablehlo.transpose %s1b1pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %s1b1dp = stablehlo.convolution(%s1b1dls, %s1b1dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %s1b1dpWxt = stablehlo.transpose %s1b1g, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %s1b1dpWdt = stablehlo.transpose %s1b1dls, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b1dpWraw = stablehlo.convolution(%s1b1dpWxt, %s1b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %s1b1dpW = stablehlo.transpose %s1b1dpWraw, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %s1b1dpb = stablehlo.reduce(%s1b1dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b1dgx2 = stablehlo.multiply %s1b1e, %s1b1e : tensor<32x768x28x28xf32>
    %s1b1dgx3 = stablehlo.multiply %s1b1dgx2, %s1b1e : tensor<32x768x28x28xf32>
    %s1b1dgck = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %s1b1dgkx3 = stablehlo.multiply %s1b1dgck, %s1b1dgx3 : tensor<32x768x28x28xf32>
    %s1b1dginn = stablehlo.add %s1b1e, %s1b1dgkx3 : tensor<32x768x28x28xf32>
    %s1b1dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %s1b1dgu = stablehlo.multiply %s1b1dgcs, %s1b1dginn : tensor<32x768x28x28xf32>
    %s1b1dgt = stablehlo.tanh %s1b1dgu : tensor<32x768x28x28xf32>
    %s1b1dgone = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %s1b1dgopt = stablehlo.add %s1b1dgone, %s1b1dgt : tensor<32x768x28x28xf32>
    %s1b1dghalf = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %s1b1dgterm1 = stablehlo.multiply %s1b1dghalf, %s1b1dgopt : tensor<32x768x28x28xf32>
    %s1b1dgt2 = stablehlo.multiply %s1b1dgt, %s1b1dgt : tensor<32x768x28x28xf32>
    %s1b1dgomt2 = stablehlo.subtract %s1b1dgone, %s1b1dgt2 : tensor<32x768x28x28xf32>
    %s1b1dghx = stablehlo.multiply %s1b1dghalf, %s1b1e : tensor<32x768x28x28xf32>
    %s1b1dghxo = stablehlo.multiply %s1b1dghx, %s1b1dgomt2 : tensor<32x768x28x28xf32>
    %s1b1dgc3b = stablehlo.constant dense<0.134145> : tensor<32x768x28x28xf32>
    %s1b1dga3x2 = stablehlo.multiply %s1b1dgc3b, %s1b1dgx2 : tensor<32x768x28x28xf32>
    %s1b1dgin2 = stablehlo.add %s1b1dgone, %s1b1dga3x2 : tensor<32x768x28x28xf32>
    %s1b1dgup = stablehlo.multiply %s1b1dgcs, %s1b1dgin2 : tensor<32x768x28x28xf32>
    %s1b1dgterm2 = stablehlo.multiply %s1b1dghxo, %s1b1dgup : tensor<32x768x28x28xf32>
    %s1b1dggp = stablehlo.add %s1b1dgterm1, %s1b1dgterm2 : tensor<32x768x28x28xf32>
    %s1b1dg = stablehlo.multiply %s1b1dp, %s1b1dggp : tensor<32x768x28x28xf32>
    %s1b1det = stablehlo.transpose %s1b1eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %s1b1de = stablehlo.convolution(%s1b1dg, %s1b1det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %s1b1deWxt = stablehlo.transpose %s1b1n, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b1deWdt = stablehlo.transpose %s1b1dg, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %s1b1deWraw = stablehlo.convolution(%s1b1deWxt, %s1b1deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %s1b1deW = stablehlo.transpose %s1b1deWraw, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %s1b1deb = stablehlo.reduce(%s1b1dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %s1b1dnri = stablehlo.reshape %s1b1d : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b1dnrdy = stablehlo.reshape %s1b1de : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b1dnnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %s1b1dnep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %s1b1dnsmr = stablehlo.reduce(%s1b1dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b1dnsm = stablehlo.broadcast_in_dim %s1b1dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b1dnmu = stablehlo.divide %s1b1dnsm, %s1b1dnnf : tensor<32x150528xf32>
    %s1b1dnxc = stablehlo.subtract %s1b1dnri, %s1b1dnmu : tensor<32x150528xf32>
    %s1b1dnsq = stablehlo.multiply %s1b1dnxc, %s1b1dnxc : tensor<32x150528xf32>
    %s1b1dnvsr = stablehlo.reduce(%s1b1dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b1dnvs = stablehlo.broadcast_in_dim %s1b1dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b1dnvr = stablehlo.divide %s1b1dnvs, %s1b1dnnf : tensor<32x150528xf32>
    %s1b1dnve = stablehlo.add %s1b1dnvr, %s1b1dnep : tensor<32x150528xf32>
    %s1b1dnistd = stablehlo.rsqrt %s1b1dnve : tensor<32x150528xf32>
    %s1b1dnxh = stablehlo.multiply %s1b1dnxc, %s1b1dnistd : tensor<32x150528xf32>
    %s1b1dngb = stablehlo.broadcast_in_dim %s1b1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b1dndxh = stablehlo.multiply %s1b1dngb, %s1b1dnrdy : tensor<32x150528xf32>
    %s1b1dnsdxr = stablehlo.reduce(%s1b1dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b1dnsdx = stablehlo.broadcast_in_dim %s1b1dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b1dnxd = stablehlo.multiply %s1b1dnxh, %s1b1dndxh : tensor<32x150528xf32>
    %s1b1dnsxdr = stablehlo.reduce(%s1b1dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b1dnsxd = stablehlo.broadcast_in_dim %s1b1dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b1dnt1 = stablehlo.multiply %s1b1dndxh, %s1b1dnnf : tensor<32x150528xf32>
    %s1b1dni1 = stablehlo.subtract %s1b1dnt1, %s1b1dnsdx : tensor<32x150528xf32>
    %s1b1dnxs = stablehlo.multiply %s1b1dnxh, %s1b1dnsxd : tensor<32x150528xf32>
    %s1b1dni2 = stablehlo.subtract %s1b1dni1, %s1b1dnxs : tensor<32x150528xf32>
    %s1b1dnsN = stablehlo.divide %s1b1dnistd, %s1b1dnnf : tensor<32x150528xf32>
    %s1b1dngin = stablehlo.multiply %s1b1dnsN, %s1b1dni2 : tensor<32x150528xf32>
    %s1b1dn = stablehlo.reshape %s1b1dngin : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %s1b1dndgp = stablehlo.multiply %s1b1dnrdy, %s1b1dnxh : tensor<32x150528xf32>
    %s1b1dndg = stablehlo.reduce(%s1b1dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b1dndb = stablehlo.reduce(%s1b1dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b1ddrev = stablehlo.reverse %s1b1dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %s1b1dd = stablehlo.convolution(%s1b1dn, %s1b1ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %s1b1ddWxt = stablehlo.transpose %s1b0o, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b1ddWdt = stablehlo.transpose %s1b1dn, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b1ddWraw = stablehlo.convolution(%s1b1ddWxt, %s1b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %s1b1ddW = stablehlo.reshape %s1b1ddWraw : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %s1b1ddb = stablehlo.reduce(%s1b1dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b1dx = stablehlo.add %s1b1dd, %s1b2dx : tensor<32x192x28x28xf32>
    %s1b0dlsgb = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %s1b0dls = stablehlo.multiply %s1b0dlsgb, %s1b1dx : tensor<32x192x28x28xf32>
    %s1b0dlsxdy = stablehlo.multiply %s1b0p, %s1b1dx : tensor<32x192x28x28xf32>
    %s1b0dlsdg = stablehlo.reduce(%s1b0dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b0dpt = stablehlo.transpose %s1b0pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %s1b0dp = stablehlo.convolution(%s1b0dls, %s1b0dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %s1b0dpWxt = stablehlo.transpose %s1b0g, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %s1b0dpWdt = stablehlo.transpose %s1b0dls, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b0dpWraw = stablehlo.convolution(%s1b0dpWxt, %s1b0dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %s1b0dpW = stablehlo.transpose %s1b0dpWraw, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %s1b0dpb = stablehlo.reduce(%s1b0dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b0dgx2 = stablehlo.multiply %s1b0e, %s1b0e : tensor<32x768x28x28xf32>
    %s1b0dgx3 = stablehlo.multiply %s1b0dgx2, %s1b0e : tensor<32x768x28x28xf32>
    %s1b0dgck = stablehlo.constant dense<0.044715> : tensor<32x768x28x28xf32>
    %s1b0dgkx3 = stablehlo.multiply %s1b0dgck, %s1b0dgx3 : tensor<32x768x28x28xf32>
    %s1b0dginn = stablehlo.add %s1b0e, %s1b0dgkx3 : tensor<32x768x28x28xf32>
    %s1b0dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x768x28x28xf32>
    %s1b0dgu = stablehlo.multiply %s1b0dgcs, %s1b0dginn : tensor<32x768x28x28xf32>
    %s1b0dgt = stablehlo.tanh %s1b0dgu : tensor<32x768x28x28xf32>
    %s1b0dgone = stablehlo.constant dense<1.0> : tensor<32x768x28x28xf32>
    %s1b0dgopt = stablehlo.add %s1b0dgone, %s1b0dgt : tensor<32x768x28x28xf32>
    %s1b0dghalf = stablehlo.constant dense<0.5> : tensor<32x768x28x28xf32>
    %s1b0dgterm1 = stablehlo.multiply %s1b0dghalf, %s1b0dgopt : tensor<32x768x28x28xf32>
    %s1b0dgt2 = stablehlo.multiply %s1b0dgt, %s1b0dgt : tensor<32x768x28x28xf32>
    %s1b0dgomt2 = stablehlo.subtract %s1b0dgone, %s1b0dgt2 : tensor<32x768x28x28xf32>
    %s1b0dghx = stablehlo.multiply %s1b0dghalf, %s1b0e : tensor<32x768x28x28xf32>
    %s1b0dghxo = stablehlo.multiply %s1b0dghx, %s1b0dgomt2 : tensor<32x768x28x28xf32>
    %s1b0dgc3b = stablehlo.constant dense<0.134145> : tensor<32x768x28x28xf32>
    %s1b0dga3x2 = stablehlo.multiply %s1b0dgc3b, %s1b0dgx2 : tensor<32x768x28x28xf32>
    %s1b0dgin2 = stablehlo.add %s1b0dgone, %s1b0dga3x2 : tensor<32x768x28x28xf32>
    %s1b0dgup = stablehlo.multiply %s1b0dgcs, %s1b0dgin2 : tensor<32x768x28x28xf32>
    %s1b0dgterm2 = stablehlo.multiply %s1b0dghxo, %s1b0dgup : tensor<32x768x28x28xf32>
    %s1b0dggp = stablehlo.add %s1b0dgterm1, %s1b0dgterm2 : tensor<32x768x28x28xf32>
    %s1b0dg = stablehlo.multiply %s1b0dp, %s1b0dggp : tensor<32x768x28x28xf32>
    %s1b0det = stablehlo.transpose %s1b0eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %s1b0de = stablehlo.convolution(%s1b0dg, %s1b0det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %s1b0deWxt = stablehlo.transpose %s1b0n, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b0deWdt = stablehlo.transpose %s1b0dg, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %s1b0deWraw = stablehlo.convolution(%s1b0deWxt, %s1b0deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %s1b0deW = stablehlo.transpose %s1b0deWraw, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %s1b0deb = stablehlo.reduce(%s1b0dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %s1b0dnri = stablehlo.reshape %s1b0d : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b0dnrdy = stablehlo.reshape %s1b0de : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %s1b0dnnf = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %s1b0dnep = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %s1b0dnsmr = stablehlo.reduce(%s1b0dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b0dnsm = stablehlo.broadcast_in_dim %s1b0dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b0dnmu = stablehlo.divide %s1b0dnsm, %s1b0dnnf : tensor<32x150528xf32>
    %s1b0dnxc = stablehlo.subtract %s1b0dnri, %s1b0dnmu : tensor<32x150528xf32>
    %s1b0dnsq = stablehlo.multiply %s1b0dnxc, %s1b0dnxc : tensor<32x150528xf32>
    %s1b0dnvsr = stablehlo.reduce(%s1b0dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b0dnvs = stablehlo.broadcast_in_dim %s1b0dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b0dnvr = stablehlo.divide %s1b0dnvs, %s1b0dnnf : tensor<32x150528xf32>
    %s1b0dnve = stablehlo.add %s1b0dnvr, %s1b0dnep : tensor<32x150528xf32>
    %s1b0dnistd = stablehlo.rsqrt %s1b0dnve : tensor<32x150528xf32>
    %s1b0dnxh = stablehlo.multiply %s1b0dnxc, %s1b0dnistd : tensor<32x150528xf32>
    %s1b0dngb = stablehlo.broadcast_in_dim %s1b0ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %s1b0dndxh = stablehlo.multiply %s1b0dngb, %s1b0dnrdy : tensor<32x150528xf32>
    %s1b0dnsdxr = stablehlo.reduce(%s1b0dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b0dnsdx = stablehlo.broadcast_in_dim %s1b0dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b0dnxd = stablehlo.multiply %s1b0dnxh, %s1b0dndxh : tensor<32x150528xf32>
    %s1b0dnsxdr = stablehlo.reduce(%s1b0dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %s1b0dnsxd = stablehlo.broadcast_in_dim %s1b0dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %s1b0dnt1 = stablehlo.multiply %s1b0dndxh, %s1b0dnnf : tensor<32x150528xf32>
    %s1b0dni1 = stablehlo.subtract %s1b0dnt1, %s1b0dnsdx : tensor<32x150528xf32>
    %s1b0dnxs = stablehlo.multiply %s1b0dnxh, %s1b0dnsxd : tensor<32x150528xf32>
    %s1b0dni2 = stablehlo.subtract %s1b0dni1, %s1b0dnxs : tensor<32x150528xf32>
    %s1b0dnsN = stablehlo.divide %s1b0dnistd, %s1b0dnnf : tensor<32x150528xf32>
    %s1b0dngin = stablehlo.multiply %s1b0dnsN, %s1b0dni2 : tensor<32x150528xf32>
    %s1b0dn = stablehlo.reshape %s1b0dngin : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %s1b0dndgp = stablehlo.multiply %s1b0dnrdy, %s1b0dnxh : tensor<32x150528xf32>
    %s1b0dndg = stablehlo.reduce(%s1b0dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b0dndb = stablehlo.reduce(%s1b0dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %s1b0ddrev = stablehlo.reverse %s1b0dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %s1b0dd = stablehlo.convolution(%s1b0dn, %s1b0ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %s1b0ddWxt = stablehlo.transpose %d0c, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b0ddWdt = stablehlo.transpose %s1b0dn, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %s1b0ddWraw = stablehlo.convolution(%s1b0ddWxt, %s1b0ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %s1b0ddW = stablehlo.reshape %s1b0ddWraw : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %s1b0ddb = stablehlo.reduce(%s1b0dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %s1b0dx = stablehlo.add %s1b0dd, %s1b1dx : tensor<32x192x28x28xf32>
    %d0dcu = stablehlo.pad %s1b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x56x56xf32>
    %d0dct = stablehlo.transpose %d0W, dims = [1, 0, 2, 3] : (tensor<192x96x2x2xf32>) -> tensor<96x192x2x2xf32>
    %d0dcr = stablehlo.reverse %d0dct, dims = [2, 3] : tensor<96x192x2x2xf32>
    %d0dc = stablehlo.convolution(%d0dcu, %d0dcr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x56x56xf32>, tensor<96x192x2x2xf32>) -> tensor<32x96x56x56xf32>
    %d0dWu = stablehlo.pad %s1b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x55x55xf32>
    %d0dWxt = stablehlo.transpose %d0n, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %d0dWdt = stablehlo.transpose %d0dWu, dims = [1, 0, 2, 3] : (tensor<32x192x55x55xf32>) -> tensor<192x32x55x55xf32>
    %d0dWraw = stablehlo.convolution(%d0dWxt, %d0dWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<192x32x55x55xf32>) -> tensor<96x192x2x2xf32>
    %d0dW = stablehlo.transpose %d0dWraw, dims = [1, 0, 2, 3] : (tensor<96x192x2x2xf32>) -> tensor<192x96x2x2xf32>
    %d0db = stablehlo.reduce(%s1b0dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %d0dnri = stablehlo.reshape %s0b2o : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %d0dnrdy = stablehlo.reshape %d0dc : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %d0dnnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %d0dnep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %d0dnsmr = stablehlo.reduce(%d0dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %d0dnsm = stablehlo.broadcast_in_dim %d0dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %d0dnmu = stablehlo.divide %d0dnsm, %d0dnnf : tensor<32x301056xf32>
    %d0dnxc = stablehlo.subtract %d0dnri, %d0dnmu : tensor<32x301056xf32>
    %d0dnsq = stablehlo.multiply %d0dnxc, %d0dnxc : tensor<32x301056xf32>
    %d0dnvsr = stablehlo.reduce(%d0dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %d0dnvs = stablehlo.broadcast_in_dim %d0dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %d0dnvr = stablehlo.divide %d0dnvs, %d0dnnf : tensor<32x301056xf32>
    %d0dnve = stablehlo.add %d0dnvr, %d0dnep : tensor<32x301056xf32>
    %d0dnistd = stablehlo.rsqrt %d0dnve : tensor<32x301056xf32>
    %d0dnxh = stablehlo.multiply %d0dnxc, %d0dnistd : tensor<32x301056xf32>
    %d0dngb = stablehlo.broadcast_in_dim %d0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %d0dndxh = stablehlo.multiply %d0dngb, %d0dnrdy : tensor<32x301056xf32>
    %d0dnsdxr = stablehlo.reduce(%d0dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %d0dnsdx = stablehlo.broadcast_in_dim %d0dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %d0dnxd = stablehlo.multiply %d0dnxh, %d0dndxh : tensor<32x301056xf32>
    %d0dnsxdr = stablehlo.reduce(%d0dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %d0dnsxd = stablehlo.broadcast_in_dim %d0dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %d0dnt1 = stablehlo.multiply %d0dndxh, %d0dnnf : tensor<32x301056xf32>
    %d0dni1 = stablehlo.subtract %d0dnt1, %d0dnsdx : tensor<32x301056xf32>
    %d0dnxs = stablehlo.multiply %d0dnxh, %d0dnsxd : tensor<32x301056xf32>
    %d0dni2 = stablehlo.subtract %d0dni1, %d0dnxs : tensor<32x301056xf32>
    %d0dnsN = stablehlo.divide %d0dnistd, %d0dnnf : tensor<32x301056xf32>
    %d0dngin = stablehlo.multiply %d0dnsN, %d0dni2 : tensor<32x301056xf32>
    %d0dn = stablehlo.reshape %d0dngin : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %d0dndgp = stablehlo.multiply %d0dnrdy, %d0dnxh : tensor<32x301056xf32>
    %d0dndg = stablehlo.reduce(%d0dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %d0dndb = stablehlo.reduce(%d0dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b2dlsgb = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b2dls = stablehlo.multiply %s0b2dlsgb, %d0dn : tensor<32x96x56x56xf32>
    %s0b2dlsxdy = stablehlo.multiply %s0b2p, %d0dn : tensor<32x96x56x56xf32>
    %s0b2dlsdg = stablehlo.reduce(%s0b2dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b2dpt = stablehlo.transpose %s0b2pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %s0b2dp = stablehlo.convolution(%s0b2dls, %s0b2dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %s0b2dpWxt = stablehlo.transpose %s0b2g, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %s0b2dpWdt = stablehlo.transpose %s0b2dls, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b2dpWraw = stablehlo.convolution(%s0b2dpWxt, %s0b2dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %s0b2dpW = stablehlo.transpose %s0b2dpWraw, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %s0b2dpb = stablehlo.reduce(%s0b2dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b2dgx2 = stablehlo.multiply %s0b2e, %s0b2e : tensor<32x384x56x56xf32>
    %s0b2dgx3 = stablehlo.multiply %s0b2dgx2, %s0b2e : tensor<32x384x56x56xf32>
    %s0b2dgck = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %s0b2dgkx3 = stablehlo.multiply %s0b2dgck, %s0b2dgx3 : tensor<32x384x56x56xf32>
    %s0b2dginn = stablehlo.add %s0b2e, %s0b2dgkx3 : tensor<32x384x56x56xf32>
    %s0b2dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %s0b2dgu = stablehlo.multiply %s0b2dgcs, %s0b2dginn : tensor<32x384x56x56xf32>
    %s0b2dgt = stablehlo.tanh %s0b2dgu : tensor<32x384x56x56xf32>
    %s0b2dgone = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %s0b2dgopt = stablehlo.add %s0b2dgone, %s0b2dgt : tensor<32x384x56x56xf32>
    %s0b2dghalf = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %s0b2dgterm1 = stablehlo.multiply %s0b2dghalf, %s0b2dgopt : tensor<32x384x56x56xf32>
    %s0b2dgt2 = stablehlo.multiply %s0b2dgt, %s0b2dgt : tensor<32x384x56x56xf32>
    %s0b2dgomt2 = stablehlo.subtract %s0b2dgone, %s0b2dgt2 : tensor<32x384x56x56xf32>
    %s0b2dghx = stablehlo.multiply %s0b2dghalf, %s0b2e : tensor<32x384x56x56xf32>
    %s0b2dghxo = stablehlo.multiply %s0b2dghx, %s0b2dgomt2 : tensor<32x384x56x56xf32>
    %s0b2dgc3b = stablehlo.constant dense<0.134145> : tensor<32x384x56x56xf32>
    %s0b2dga3x2 = stablehlo.multiply %s0b2dgc3b, %s0b2dgx2 : tensor<32x384x56x56xf32>
    %s0b2dgin2 = stablehlo.add %s0b2dgone, %s0b2dga3x2 : tensor<32x384x56x56xf32>
    %s0b2dgup = stablehlo.multiply %s0b2dgcs, %s0b2dgin2 : tensor<32x384x56x56xf32>
    %s0b2dgterm2 = stablehlo.multiply %s0b2dghxo, %s0b2dgup : tensor<32x384x56x56xf32>
    %s0b2dggp = stablehlo.add %s0b2dgterm1, %s0b2dgterm2 : tensor<32x384x56x56xf32>
    %s0b2dg = stablehlo.multiply %s0b2dp, %s0b2dggp : tensor<32x384x56x56xf32>
    %s0b2det = stablehlo.transpose %s0b2eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %s0b2de = stablehlo.convolution(%s0b2dg, %s0b2det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %s0b2deWxt = stablehlo.transpose %s0b2n, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b2deWdt = stablehlo.transpose %s0b2dg, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %s0b2deWraw = stablehlo.convolution(%s0b2deWxt, %s0b2deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %s0b2deW = stablehlo.transpose %s0b2deWraw, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %s0b2deb = stablehlo.reduce(%s0b2dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %s0b2dnri = stablehlo.reshape %s0b2d : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b2dnrdy = stablehlo.reshape %s0b2de : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b2dnnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %s0b2dnep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %s0b2dnsmr = stablehlo.reduce(%s0b2dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b2dnsm = stablehlo.broadcast_in_dim %s0b2dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b2dnmu = stablehlo.divide %s0b2dnsm, %s0b2dnnf : tensor<32x301056xf32>
    %s0b2dnxc = stablehlo.subtract %s0b2dnri, %s0b2dnmu : tensor<32x301056xf32>
    %s0b2dnsq = stablehlo.multiply %s0b2dnxc, %s0b2dnxc : tensor<32x301056xf32>
    %s0b2dnvsr = stablehlo.reduce(%s0b2dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b2dnvs = stablehlo.broadcast_in_dim %s0b2dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b2dnvr = stablehlo.divide %s0b2dnvs, %s0b2dnnf : tensor<32x301056xf32>
    %s0b2dnve = stablehlo.add %s0b2dnvr, %s0b2dnep : tensor<32x301056xf32>
    %s0b2dnistd = stablehlo.rsqrt %s0b2dnve : tensor<32x301056xf32>
    %s0b2dnxh = stablehlo.multiply %s0b2dnxc, %s0b2dnistd : tensor<32x301056xf32>
    %s0b2dngb = stablehlo.broadcast_in_dim %s0b2ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b2dndxh = stablehlo.multiply %s0b2dngb, %s0b2dnrdy : tensor<32x301056xf32>
    %s0b2dnsdxr = stablehlo.reduce(%s0b2dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b2dnsdx = stablehlo.broadcast_in_dim %s0b2dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b2dnxd = stablehlo.multiply %s0b2dnxh, %s0b2dndxh : tensor<32x301056xf32>
    %s0b2dnsxdr = stablehlo.reduce(%s0b2dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b2dnsxd = stablehlo.broadcast_in_dim %s0b2dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b2dnt1 = stablehlo.multiply %s0b2dndxh, %s0b2dnnf : tensor<32x301056xf32>
    %s0b2dni1 = stablehlo.subtract %s0b2dnt1, %s0b2dnsdx : tensor<32x301056xf32>
    %s0b2dnxs = stablehlo.multiply %s0b2dnxh, %s0b2dnsxd : tensor<32x301056xf32>
    %s0b2dni2 = stablehlo.subtract %s0b2dni1, %s0b2dnxs : tensor<32x301056xf32>
    %s0b2dnsN = stablehlo.divide %s0b2dnistd, %s0b2dnnf : tensor<32x301056xf32>
    %s0b2dngin = stablehlo.multiply %s0b2dnsN, %s0b2dni2 : tensor<32x301056xf32>
    %s0b2dn = stablehlo.reshape %s0b2dngin : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %s0b2dndgp = stablehlo.multiply %s0b2dnrdy, %s0b2dnxh : tensor<32x301056xf32>
    %s0b2dndg = stablehlo.reduce(%s0b2dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b2dndb = stablehlo.reduce(%s0b2dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b2ddrev = stablehlo.reverse %s0b2dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %s0b2dd = stablehlo.convolution(%s0b2dn, %s0b2ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %s0b2ddWxt = stablehlo.transpose %s0b1o, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b2ddWdt = stablehlo.transpose %s0b2dn, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b2ddWraw = stablehlo.convolution(%s0b2ddWxt, %s0b2ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %s0b2ddW = stablehlo.reshape %s0b2ddWraw : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %s0b2ddb = stablehlo.reduce(%s0b2dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b2dx = stablehlo.add %s0b2dd, %d0dn : tensor<32x96x56x56xf32>
    %s0b1dlsgb = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b1dls = stablehlo.multiply %s0b1dlsgb, %s0b2dx : tensor<32x96x56x56xf32>
    %s0b1dlsxdy = stablehlo.multiply %s0b1p, %s0b2dx : tensor<32x96x56x56xf32>
    %s0b1dlsdg = stablehlo.reduce(%s0b1dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b1dpt = stablehlo.transpose %s0b1pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %s0b1dp = stablehlo.convolution(%s0b1dls, %s0b1dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %s0b1dpWxt = stablehlo.transpose %s0b1g, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %s0b1dpWdt = stablehlo.transpose %s0b1dls, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b1dpWraw = stablehlo.convolution(%s0b1dpWxt, %s0b1dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %s0b1dpW = stablehlo.transpose %s0b1dpWraw, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %s0b1dpb = stablehlo.reduce(%s0b1dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b1dgx2 = stablehlo.multiply %s0b1e, %s0b1e : tensor<32x384x56x56xf32>
    %s0b1dgx3 = stablehlo.multiply %s0b1dgx2, %s0b1e : tensor<32x384x56x56xf32>
    %s0b1dgck = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %s0b1dgkx3 = stablehlo.multiply %s0b1dgck, %s0b1dgx3 : tensor<32x384x56x56xf32>
    %s0b1dginn = stablehlo.add %s0b1e, %s0b1dgkx3 : tensor<32x384x56x56xf32>
    %s0b1dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %s0b1dgu = stablehlo.multiply %s0b1dgcs, %s0b1dginn : tensor<32x384x56x56xf32>
    %s0b1dgt = stablehlo.tanh %s0b1dgu : tensor<32x384x56x56xf32>
    %s0b1dgone = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %s0b1dgopt = stablehlo.add %s0b1dgone, %s0b1dgt : tensor<32x384x56x56xf32>
    %s0b1dghalf = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %s0b1dgterm1 = stablehlo.multiply %s0b1dghalf, %s0b1dgopt : tensor<32x384x56x56xf32>
    %s0b1dgt2 = stablehlo.multiply %s0b1dgt, %s0b1dgt : tensor<32x384x56x56xf32>
    %s0b1dgomt2 = stablehlo.subtract %s0b1dgone, %s0b1dgt2 : tensor<32x384x56x56xf32>
    %s0b1dghx = stablehlo.multiply %s0b1dghalf, %s0b1e : tensor<32x384x56x56xf32>
    %s0b1dghxo = stablehlo.multiply %s0b1dghx, %s0b1dgomt2 : tensor<32x384x56x56xf32>
    %s0b1dgc3b = stablehlo.constant dense<0.134145> : tensor<32x384x56x56xf32>
    %s0b1dga3x2 = stablehlo.multiply %s0b1dgc3b, %s0b1dgx2 : tensor<32x384x56x56xf32>
    %s0b1dgin2 = stablehlo.add %s0b1dgone, %s0b1dga3x2 : tensor<32x384x56x56xf32>
    %s0b1dgup = stablehlo.multiply %s0b1dgcs, %s0b1dgin2 : tensor<32x384x56x56xf32>
    %s0b1dgterm2 = stablehlo.multiply %s0b1dghxo, %s0b1dgup : tensor<32x384x56x56xf32>
    %s0b1dggp = stablehlo.add %s0b1dgterm1, %s0b1dgterm2 : tensor<32x384x56x56xf32>
    %s0b1dg = stablehlo.multiply %s0b1dp, %s0b1dggp : tensor<32x384x56x56xf32>
    %s0b1det = stablehlo.transpose %s0b1eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %s0b1de = stablehlo.convolution(%s0b1dg, %s0b1det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %s0b1deWxt = stablehlo.transpose %s0b1n, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b1deWdt = stablehlo.transpose %s0b1dg, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %s0b1deWraw = stablehlo.convolution(%s0b1deWxt, %s0b1deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %s0b1deW = stablehlo.transpose %s0b1deWraw, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %s0b1deb = stablehlo.reduce(%s0b1dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %s0b1dnri = stablehlo.reshape %s0b1d : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b1dnrdy = stablehlo.reshape %s0b1de : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b1dnnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %s0b1dnep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %s0b1dnsmr = stablehlo.reduce(%s0b1dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b1dnsm = stablehlo.broadcast_in_dim %s0b1dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b1dnmu = stablehlo.divide %s0b1dnsm, %s0b1dnnf : tensor<32x301056xf32>
    %s0b1dnxc = stablehlo.subtract %s0b1dnri, %s0b1dnmu : tensor<32x301056xf32>
    %s0b1dnsq = stablehlo.multiply %s0b1dnxc, %s0b1dnxc : tensor<32x301056xf32>
    %s0b1dnvsr = stablehlo.reduce(%s0b1dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b1dnvs = stablehlo.broadcast_in_dim %s0b1dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b1dnvr = stablehlo.divide %s0b1dnvs, %s0b1dnnf : tensor<32x301056xf32>
    %s0b1dnve = stablehlo.add %s0b1dnvr, %s0b1dnep : tensor<32x301056xf32>
    %s0b1dnistd = stablehlo.rsqrt %s0b1dnve : tensor<32x301056xf32>
    %s0b1dnxh = stablehlo.multiply %s0b1dnxc, %s0b1dnistd : tensor<32x301056xf32>
    %s0b1dngb = stablehlo.broadcast_in_dim %s0b1ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b1dndxh = stablehlo.multiply %s0b1dngb, %s0b1dnrdy : tensor<32x301056xf32>
    %s0b1dnsdxr = stablehlo.reduce(%s0b1dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b1dnsdx = stablehlo.broadcast_in_dim %s0b1dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b1dnxd = stablehlo.multiply %s0b1dnxh, %s0b1dndxh : tensor<32x301056xf32>
    %s0b1dnsxdr = stablehlo.reduce(%s0b1dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b1dnsxd = stablehlo.broadcast_in_dim %s0b1dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b1dnt1 = stablehlo.multiply %s0b1dndxh, %s0b1dnnf : tensor<32x301056xf32>
    %s0b1dni1 = stablehlo.subtract %s0b1dnt1, %s0b1dnsdx : tensor<32x301056xf32>
    %s0b1dnxs = stablehlo.multiply %s0b1dnxh, %s0b1dnsxd : tensor<32x301056xf32>
    %s0b1dni2 = stablehlo.subtract %s0b1dni1, %s0b1dnxs : tensor<32x301056xf32>
    %s0b1dnsN = stablehlo.divide %s0b1dnistd, %s0b1dnnf : tensor<32x301056xf32>
    %s0b1dngin = stablehlo.multiply %s0b1dnsN, %s0b1dni2 : tensor<32x301056xf32>
    %s0b1dn = stablehlo.reshape %s0b1dngin : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %s0b1dndgp = stablehlo.multiply %s0b1dnrdy, %s0b1dnxh : tensor<32x301056xf32>
    %s0b1dndg = stablehlo.reduce(%s0b1dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b1dndb = stablehlo.reduce(%s0b1dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b1ddrev = stablehlo.reverse %s0b1dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %s0b1dd = stablehlo.convolution(%s0b1dn, %s0b1ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %s0b1ddWxt = stablehlo.transpose %s0b0o, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b1ddWdt = stablehlo.transpose %s0b1dn, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b1ddWraw = stablehlo.convolution(%s0b1ddWxt, %s0b1ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %s0b1ddW = stablehlo.reshape %s0b1ddWraw : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %s0b1ddb = stablehlo.reduce(%s0b1dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b1dx = stablehlo.add %s0b1dd, %s0b2dx : tensor<32x96x56x56xf32>
    %s0b0dlsgb = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %s0b0dls = stablehlo.multiply %s0b0dlsgb, %s0b1dx : tensor<32x96x56x56xf32>
    %s0b0dlsxdy = stablehlo.multiply %s0b0p, %s0b1dx : tensor<32x96x56x56xf32>
    %s0b0dlsdg = stablehlo.reduce(%s0b0dlsxdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b0dpt = stablehlo.transpose %s0b0pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %s0b0dp = stablehlo.convolution(%s0b0dls, %s0b0dpt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %s0b0dpWxt = stablehlo.transpose %s0b0g, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %s0b0dpWdt = stablehlo.transpose %s0b0dls, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b0dpWraw = stablehlo.convolution(%s0b0dpWxt, %s0b0dpWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %s0b0dpW = stablehlo.transpose %s0b0dpWraw, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %s0b0dpb = stablehlo.reduce(%s0b0dls init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b0dgx2 = stablehlo.multiply %s0b0e, %s0b0e : tensor<32x384x56x56xf32>
    %s0b0dgx3 = stablehlo.multiply %s0b0dgx2, %s0b0e : tensor<32x384x56x56xf32>
    %s0b0dgck = stablehlo.constant dense<0.044715> : tensor<32x384x56x56xf32>
    %s0b0dgkx3 = stablehlo.multiply %s0b0dgck, %s0b0dgx3 : tensor<32x384x56x56xf32>
    %s0b0dginn = stablehlo.add %s0b0e, %s0b0dgkx3 : tensor<32x384x56x56xf32>
    %s0b0dgcs = stablehlo.constant dense<0.7978845608028654> : tensor<32x384x56x56xf32>
    %s0b0dgu = stablehlo.multiply %s0b0dgcs, %s0b0dginn : tensor<32x384x56x56xf32>
    %s0b0dgt = stablehlo.tanh %s0b0dgu : tensor<32x384x56x56xf32>
    %s0b0dgone = stablehlo.constant dense<1.0> : tensor<32x384x56x56xf32>
    %s0b0dgopt = stablehlo.add %s0b0dgone, %s0b0dgt : tensor<32x384x56x56xf32>
    %s0b0dghalf = stablehlo.constant dense<0.5> : tensor<32x384x56x56xf32>
    %s0b0dgterm1 = stablehlo.multiply %s0b0dghalf, %s0b0dgopt : tensor<32x384x56x56xf32>
    %s0b0dgt2 = stablehlo.multiply %s0b0dgt, %s0b0dgt : tensor<32x384x56x56xf32>
    %s0b0dgomt2 = stablehlo.subtract %s0b0dgone, %s0b0dgt2 : tensor<32x384x56x56xf32>
    %s0b0dghx = stablehlo.multiply %s0b0dghalf, %s0b0e : tensor<32x384x56x56xf32>
    %s0b0dghxo = stablehlo.multiply %s0b0dghx, %s0b0dgomt2 : tensor<32x384x56x56xf32>
    %s0b0dgc3b = stablehlo.constant dense<0.134145> : tensor<32x384x56x56xf32>
    %s0b0dga3x2 = stablehlo.multiply %s0b0dgc3b, %s0b0dgx2 : tensor<32x384x56x56xf32>
    %s0b0dgin2 = stablehlo.add %s0b0dgone, %s0b0dga3x2 : tensor<32x384x56x56xf32>
    %s0b0dgup = stablehlo.multiply %s0b0dgcs, %s0b0dgin2 : tensor<32x384x56x56xf32>
    %s0b0dgterm2 = stablehlo.multiply %s0b0dghxo, %s0b0dgup : tensor<32x384x56x56xf32>
    %s0b0dggp = stablehlo.add %s0b0dgterm1, %s0b0dgterm2 : tensor<32x384x56x56xf32>
    %s0b0dg = stablehlo.multiply %s0b0dp, %s0b0dggp : tensor<32x384x56x56xf32>
    %s0b0det = stablehlo.transpose %s0b0eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %s0b0de = stablehlo.convolution(%s0b0dg, %s0b0det)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %s0b0deWxt = stablehlo.transpose %s0b0n, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b0deWdt = stablehlo.transpose %s0b0dg, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %s0b0deWraw = stablehlo.convolution(%s0b0deWxt, %s0b0deWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %s0b0deW = stablehlo.transpose %s0b0deWraw, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %s0b0deb = stablehlo.reduce(%s0b0dg init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %s0b0dnri = stablehlo.reshape %s0b0d : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b0dnrdy = stablehlo.reshape %s0b0de : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %s0b0dnnf = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %s0b0dnep = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %s0b0dnsmr = stablehlo.reduce(%s0b0dnri init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b0dnsm = stablehlo.broadcast_in_dim %s0b0dnsmr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b0dnmu = stablehlo.divide %s0b0dnsm, %s0b0dnnf : tensor<32x301056xf32>
    %s0b0dnxc = stablehlo.subtract %s0b0dnri, %s0b0dnmu : tensor<32x301056xf32>
    %s0b0dnsq = stablehlo.multiply %s0b0dnxc, %s0b0dnxc : tensor<32x301056xf32>
    %s0b0dnvsr = stablehlo.reduce(%s0b0dnsq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b0dnvs = stablehlo.broadcast_in_dim %s0b0dnvsr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b0dnvr = stablehlo.divide %s0b0dnvs, %s0b0dnnf : tensor<32x301056xf32>
    %s0b0dnve = stablehlo.add %s0b0dnvr, %s0b0dnep : tensor<32x301056xf32>
    %s0b0dnistd = stablehlo.rsqrt %s0b0dnve : tensor<32x301056xf32>
    %s0b0dnxh = stablehlo.multiply %s0b0dnxc, %s0b0dnistd : tensor<32x301056xf32>
    %s0b0dngb = stablehlo.broadcast_in_dim %s0b0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %s0b0dndxh = stablehlo.multiply %s0b0dngb, %s0b0dnrdy : tensor<32x301056xf32>
    %s0b0dnsdxr = stablehlo.reduce(%s0b0dndxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b0dnsdx = stablehlo.broadcast_in_dim %s0b0dnsdxr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b0dnxd = stablehlo.multiply %s0b0dnxh, %s0b0dndxh : tensor<32x301056xf32>
    %s0b0dnsxdr = stablehlo.reduce(%s0b0dnxd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %s0b0dnsxd = stablehlo.broadcast_in_dim %s0b0dnsxdr, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %s0b0dnt1 = stablehlo.multiply %s0b0dndxh, %s0b0dnnf : tensor<32x301056xf32>
    %s0b0dni1 = stablehlo.subtract %s0b0dnt1, %s0b0dnsdx : tensor<32x301056xf32>
    %s0b0dnxs = stablehlo.multiply %s0b0dnxh, %s0b0dnsxd : tensor<32x301056xf32>
    %s0b0dni2 = stablehlo.subtract %s0b0dni1, %s0b0dnxs : tensor<32x301056xf32>
    %s0b0dnsN = stablehlo.divide %s0b0dnistd, %s0b0dnnf : tensor<32x301056xf32>
    %s0b0dngin = stablehlo.multiply %s0b0dnsN, %s0b0dni2 : tensor<32x301056xf32>
    %s0b0dn = stablehlo.reshape %s0b0dngin : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %s0b0dndgp = stablehlo.multiply %s0b0dnrdy, %s0b0dnxh : tensor<32x301056xf32>
    %s0b0dndg = stablehlo.reduce(%s0b0dndgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b0dndb = stablehlo.reduce(%s0b0dnrdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %s0b0ddrev = stablehlo.reverse %s0b0dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %s0b0dd = stablehlo.convolution(%s0b0dn, %s0b0ddrev)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %s0b0ddWxt = stablehlo.transpose %ps, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b0ddWdt = stablehlo.transpose %s0b0dn, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %s0b0ddWraw = stablehlo.convolution(%s0b0ddWxt, %s0b0ddWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %s0b0ddW = stablehlo.reshape %s0b0ddWraw : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %s0b0ddb = stablehlo.reduce(%s0b0dn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %s0b0dx = stablehlo.add %s0b0dd, %s0b1dx : tensor<32x96x56x56xf32>
    %psdWu = stablehlo.pad %s0b0dx, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x221x221xf32>
    %psdWxt = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %psdWdt = stablehlo.transpose %psdWu, dims = [1, 0, 2, 3] : (tensor<32x96x221x221xf32>) -> tensor<96x32x221x221xf32>
    %psdWraw = stablehlo.convolution(%psdWxt, %psdWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<96x32x221x221xf32>) -> tensor<3x96x4x4xf32>
    %psdW = stablehlo.transpose %psdWraw, dims = [1, 0, 2, 3] : (tensor<3x96x4x4xf32>) -> tensor<96x3x4x4xf32>
    %psdb = stablehlo.reduce(%s0b0dx init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %adb1psW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %adob1psW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %admspsW = stablehlo.multiply %adb1psW, %psWm : tensor<96x3x4x4xf32>
    %admgpsW = stablehlo.multiply %adob1psW, %psdW : tensor<96x3x4x4xf32>
    %admnpsW = stablehlo.add %admspsW, %admgpsW : tensor<96x3x4x4xf32>
    %adb2psW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %adob2psW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %advspsW = stablehlo.multiply %adb2psW, %psWv : tensor<96x3x4x4xf32>
    %adg2psW = stablehlo.multiply %psdW, %psdW : tensor<96x3x4x4xf32>
    %advgpsW = stablehlo.multiply %adob2psW, %adg2psW : tensor<96x3x4x4xf32>
    %advnpsW = stablehlo.add %advspsW, %advgpsW : tensor<96x3x4x4xf32>
    %adbc1psW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %adbc2psW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %admhpsW = stablehlo.divide %admnpsW, %adbc1psW : tensor<96x3x4x4xf32>
    %advhpsW = stablehlo.divide %advnpsW, %adbc2psW : tensor<96x3x4x4xf32>
    %adlrpsW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %adepspsW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %adsqpsW = stablehlo.sqrt %advhpsW : tensor<96x3x4x4xf32>
    %addenpsW = stablehlo.add %adsqpsW, %adepspsW : tensor<96x3x4x4xf32>
    %adratpsW = stablehlo.divide %admhpsW, %addenpsW : tensor<96x3x4x4xf32>
    %adstpsW = stablehlo.multiply %adlrpsW, %adratpsW : tensor<96x3x4x4xf32>
    %adsubpsW = stablehlo.subtract %psW, %adstpsW : tensor<96x3x4x4xf32>
    %adwdpsW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %adwdlrpsW = stablehlo.multiply %adwdpsW, %adlrpsW : tensor<96x3x4x4xf32>
    %adwdppsW = stablehlo.multiply %adwdlrpsW, %psW : tensor<96x3x4x4xf32>
    %adnewpsW = stablehlo.subtract %adsubpsW, %adwdppsW : tensor<96x3x4x4xf32>
    %adb1psb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1psb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admspsb = stablehlo.multiply %adb1psb, %psbm : tensor<96xf32>
    %admgpsb = stablehlo.multiply %adob1psb, %psdb : tensor<96xf32>
    %admnpsb = stablehlo.add %admspsb, %admgpsb : tensor<96xf32>
    %adb2psb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2psb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advspsb = stablehlo.multiply %adb2psb, %psbv : tensor<96xf32>
    %adg2psb = stablehlo.multiply %psdb, %psdb : tensor<96xf32>
    %advgpsb = stablehlo.multiply %adob2psb, %adg2psb : tensor<96xf32>
    %advnpsb = stablehlo.add %advspsb, %advgpsb : tensor<96xf32>
    %adbc1psb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2psb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhpsb = stablehlo.divide %admnpsb, %adbc1psb : tensor<96xf32>
    %advhpsb = stablehlo.divide %advnpsb, %adbc2psb : tensor<96xf32>
    %adlrpsb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepspsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqpsb = stablehlo.sqrt %advhpsb : tensor<96xf32>
    %addenpsb = stablehlo.add %adsqpsb, %adepspsb : tensor<96xf32>
    %adratpsb = stablehlo.divide %admhpsb, %addenpsb : tensor<96xf32>
    %adstpsb = stablehlo.multiply %adlrpsb, %adratpsb : tensor<96xf32>
    %adsubpsb = stablehlo.subtract %psb, %adstpsb : tensor<96xf32>
    %adwdpsb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrpsb = stablehlo.multiply %adwdpsb, %adlrpsb : tensor<96xf32>
    %adwdppsb = stablehlo.multiply %adwdlrpsb, %psb : tensor<96xf32>
    %adnewpsb = stablehlo.subtract %adsubpsb, %adwdppsb : tensor<96xf32>
    %adb1s0b0dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adob1s0b0dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %admss0b0dW = stablehlo.multiply %adb1s0b0dW, %s0b0dWm : tensor<96x1x7x7xf32>
    %admgs0b0dW = stablehlo.multiply %adob1s0b0dW, %s0b0ddW : tensor<96x1x7x7xf32>
    %admns0b0dW = stablehlo.add %admss0b0dW, %admgs0b0dW : tensor<96x1x7x7xf32>
    %adb2s0b0dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adob2s0b0dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %advss0b0dW = stablehlo.multiply %adb2s0b0dW, %s0b0dWv : tensor<96x1x7x7xf32>
    %adg2s0b0dW = stablehlo.multiply %s0b0ddW, %s0b0ddW : tensor<96x1x7x7xf32>
    %advgs0b0dW = stablehlo.multiply %adob2s0b0dW, %adg2s0b0dW : tensor<96x1x7x7xf32>
    %advns0b0dW = stablehlo.add %advss0b0dW, %advgs0b0dW : tensor<96x1x7x7xf32>
    %adbc1s0b0dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adbc2s0b0dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %admhs0b0dW = stablehlo.divide %admns0b0dW, %adbc1s0b0dW : tensor<96x1x7x7xf32>
    %advhs0b0dW = stablehlo.divide %advns0b0dW, %adbc2s0b0dW : tensor<96x1x7x7xf32>
    %adlrs0b0dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adepss0b0dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adsqs0b0dW = stablehlo.sqrt %advhs0b0dW : tensor<96x1x7x7xf32>
    %addens0b0dW = stablehlo.add %adsqs0b0dW, %adepss0b0dW : tensor<96x1x7x7xf32>
    %adrats0b0dW = stablehlo.divide %admhs0b0dW, %addens0b0dW : tensor<96x1x7x7xf32>
    %adsts0b0dW = stablehlo.multiply %adlrs0b0dW, %adrats0b0dW : tensor<96x1x7x7xf32>
    %adsubs0b0dW = stablehlo.subtract %s0b0dW, %adsts0b0dW : tensor<96x1x7x7xf32>
    %adwds0b0dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adwdlrs0b0dW = stablehlo.multiply %adwds0b0dW, %adlrs0b0dW : tensor<96x1x7x7xf32>
    %adwdps0b0dW = stablehlo.multiply %adwdlrs0b0dW, %s0b0dW : tensor<96x1x7x7xf32>
    %adnews0b0dW = stablehlo.subtract %adsubs0b0dW, %adwdps0b0dW : tensor<96x1x7x7xf32>
    %adb1s0b0db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1s0b0db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admss0b0db = stablehlo.multiply %adb1s0b0db, %s0b0dbm : tensor<96xf32>
    %admgs0b0db = stablehlo.multiply %adob1s0b0db, %s0b0ddb : tensor<96xf32>
    %admns0b0db = stablehlo.add %admss0b0db, %admgs0b0db : tensor<96xf32>
    %adb2s0b0db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2s0b0db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advss0b0db = stablehlo.multiply %adb2s0b0db, %s0b0dbv : tensor<96xf32>
    %adg2s0b0db = stablehlo.multiply %s0b0ddb, %s0b0ddb : tensor<96xf32>
    %advgs0b0db = stablehlo.multiply %adob2s0b0db, %adg2s0b0db : tensor<96xf32>
    %advns0b0db = stablehlo.add %advss0b0db, %advgs0b0db : tensor<96xf32>
    %adbc1s0b0db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2s0b0db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhs0b0db = stablehlo.divide %admns0b0db, %adbc1s0b0db : tensor<96xf32>
    %advhs0b0db = stablehlo.divide %advns0b0db, %adbc2s0b0db : tensor<96xf32>
    %adlrs0b0db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepss0b0db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqs0b0db = stablehlo.sqrt %advhs0b0db : tensor<96xf32>
    %addens0b0db = stablehlo.add %adsqs0b0db, %adepss0b0db : tensor<96xf32>
    %adrats0b0db = stablehlo.divide %admhs0b0db, %addens0b0db : tensor<96xf32>
    %adsts0b0db = stablehlo.multiply %adlrs0b0db, %adrats0b0db : tensor<96xf32>
    %adsubs0b0db = stablehlo.subtract %s0b0db, %adsts0b0db : tensor<96xf32>
    %adwds0b0db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrs0b0db = stablehlo.multiply %adwds0b0db, %adlrs0b0db : tensor<96xf32>
    %adwdps0b0db = stablehlo.multiply %adwdlrs0b0db, %s0b0db : tensor<96xf32>
    %adnews0b0db = stablehlo.subtract %adsubs0b0db, %adwdps0b0db : tensor<96xf32>
    %adb1s0b0ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s0b0ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss0b0ng = stablehlo.multiply %adb1s0b0ng, %s0b0ngm : tensor<f32>
    %admgs0b0ng = stablehlo.multiply %adob1s0b0ng, %s0b0dndg : tensor<f32>
    %admns0b0ng = stablehlo.add %admss0b0ng, %admgs0b0ng : tensor<f32>
    %adb2s0b0ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s0b0ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss0b0ng = stablehlo.multiply %adb2s0b0ng, %s0b0ngv : tensor<f32>
    %adg2s0b0ng = stablehlo.multiply %s0b0dndg, %s0b0dndg : tensor<f32>
    %advgs0b0ng = stablehlo.multiply %adob2s0b0ng, %adg2s0b0ng : tensor<f32>
    %advns0b0ng = stablehlo.add %advss0b0ng, %advgs0b0ng : tensor<f32>
    %adbc1s0b0ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s0b0ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs0b0ng = stablehlo.divide %admns0b0ng, %adbc1s0b0ng : tensor<f32>
    %advhs0b0ng = stablehlo.divide %advns0b0ng, %adbc2s0b0ng : tensor<f32>
    %adlrs0b0ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss0b0ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs0b0ng = stablehlo.sqrt %advhs0b0ng : tensor<f32>
    %addens0b0ng = stablehlo.add %adsqs0b0ng, %adepss0b0ng : tensor<f32>
    %adrats0b0ng = stablehlo.divide %admhs0b0ng, %addens0b0ng : tensor<f32>
    %adsts0b0ng = stablehlo.multiply %adlrs0b0ng, %adrats0b0ng : tensor<f32>
    %adsubs0b0ng = stablehlo.subtract %s0b0ng, %adsts0b0ng : tensor<f32>
    %adwds0b0ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs0b0ng = stablehlo.multiply %adwds0b0ng, %adlrs0b0ng : tensor<f32>
    %adwdps0b0ng = stablehlo.multiply %adwdlrs0b0ng, %s0b0ng : tensor<f32>
    %adnews0b0ng = stablehlo.subtract %adsubs0b0ng, %adwdps0b0ng : tensor<f32>
    %adb1s0b0nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s0b0nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss0b0nbt = stablehlo.multiply %adb1s0b0nbt, %s0b0nbtm : tensor<f32>
    %admgs0b0nbt = stablehlo.multiply %adob1s0b0nbt, %s0b0dndb : tensor<f32>
    %admns0b0nbt = stablehlo.add %admss0b0nbt, %admgs0b0nbt : tensor<f32>
    %adb2s0b0nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s0b0nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss0b0nbt = stablehlo.multiply %adb2s0b0nbt, %s0b0nbtv : tensor<f32>
    %adg2s0b0nbt = stablehlo.multiply %s0b0dndb, %s0b0dndb : tensor<f32>
    %advgs0b0nbt = stablehlo.multiply %adob2s0b0nbt, %adg2s0b0nbt : tensor<f32>
    %advns0b0nbt = stablehlo.add %advss0b0nbt, %advgs0b0nbt : tensor<f32>
    %adbc1s0b0nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s0b0nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs0b0nbt = stablehlo.divide %admns0b0nbt, %adbc1s0b0nbt : tensor<f32>
    %advhs0b0nbt = stablehlo.divide %advns0b0nbt, %adbc2s0b0nbt : tensor<f32>
    %adlrs0b0nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss0b0nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs0b0nbt = stablehlo.sqrt %advhs0b0nbt : tensor<f32>
    %addens0b0nbt = stablehlo.add %adsqs0b0nbt, %adepss0b0nbt : tensor<f32>
    %adrats0b0nbt = stablehlo.divide %admhs0b0nbt, %addens0b0nbt : tensor<f32>
    %adsts0b0nbt = stablehlo.multiply %adlrs0b0nbt, %adrats0b0nbt : tensor<f32>
    %adsubs0b0nbt = stablehlo.subtract %s0b0nbt, %adsts0b0nbt : tensor<f32>
    %adwds0b0nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs0b0nbt = stablehlo.multiply %adwds0b0nbt, %adlrs0b0nbt : tensor<f32>
    %adwdps0b0nbt = stablehlo.multiply %adwdlrs0b0nbt, %s0b0nbt : tensor<f32>
    %adnews0b0nbt = stablehlo.subtract %adsubs0b0nbt, %adwdps0b0nbt : tensor<f32>
    %adb1s0b0eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adob1s0b0eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %admss0b0eW = stablehlo.multiply %adb1s0b0eW, %s0b0eWm : tensor<384x96x1x1xf32>
    %admgs0b0eW = stablehlo.multiply %adob1s0b0eW, %s0b0deW : tensor<384x96x1x1xf32>
    %admns0b0eW = stablehlo.add %admss0b0eW, %admgs0b0eW : tensor<384x96x1x1xf32>
    %adb2s0b0eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adob2s0b0eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %advss0b0eW = stablehlo.multiply %adb2s0b0eW, %s0b0eWv : tensor<384x96x1x1xf32>
    %adg2s0b0eW = stablehlo.multiply %s0b0deW, %s0b0deW : tensor<384x96x1x1xf32>
    %advgs0b0eW = stablehlo.multiply %adob2s0b0eW, %adg2s0b0eW : tensor<384x96x1x1xf32>
    %advns0b0eW = stablehlo.add %advss0b0eW, %advgs0b0eW : tensor<384x96x1x1xf32>
    %adbc1s0b0eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adbc2s0b0eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %admhs0b0eW = stablehlo.divide %admns0b0eW, %adbc1s0b0eW : tensor<384x96x1x1xf32>
    %advhs0b0eW = stablehlo.divide %advns0b0eW, %adbc2s0b0eW : tensor<384x96x1x1xf32>
    %adlrs0b0eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adepss0b0eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adsqs0b0eW = stablehlo.sqrt %advhs0b0eW : tensor<384x96x1x1xf32>
    %addens0b0eW = stablehlo.add %adsqs0b0eW, %adepss0b0eW : tensor<384x96x1x1xf32>
    %adrats0b0eW = stablehlo.divide %admhs0b0eW, %addens0b0eW : tensor<384x96x1x1xf32>
    %adsts0b0eW = stablehlo.multiply %adlrs0b0eW, %adrats0b0eW : tensor<384x96x1x1xf32>
    %adsubs0b0eW = stablehlo.subtract %s0b0eW, %adsts0b0eW : tensor<384x96x1x1xf32>
    %adwds0b0eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adwdlrs0b0eW = stablehlo.multiply %adwds0b0eW, %adlrs0b0eW : tensor<384x96x1x1xf32>
    %adwdps0b0eW = stablehlo.multiply %adwdlrs0b0eW, %s0b0eW : tensor<384x96x1x1xf32>
    %adnews0b0eW = stablehlo.subtract %adsubs0b0eW, %adwdps0b0eW : tensor<384x96x1x1xf32>
    %adb1s0b0eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s0b0eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss0b0eb = stablehlo.multiply %adb1s0b0eb, %s0b0ebm : tensor<384xf32>
    %admgs0b0eb = stablehlo.multiply %adob1s0b0eb, %s0b0deb : tensor<384xf32>
    %admns0b0eb = stablehlo.add %admss0b0eb, %admgs0b0eb : tensor<384xf32>
    %adb2s0b0eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s0b0eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss0b0eb = stablehlo.multiply %adb2s0b0eb, %s0b0ebv : tensor<384xf32>
    %adg2s0b0eb = stablehlo.multiply %s0b0deb, %s0b0deb : tensor<384xf32>
    %advgs0b0eb = stablehlo.multiply %adob2s0b0eb, %adg2s0b0eb : tensor<384xf32>
    %advns0b0eb = stablehlo.add %advss0b0eb, %advgs0b0eb : tensor<384xf32>
    %adbc1s0b0eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s0b0eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs0b0eb = stablehlo.divide %admns0b0eb, %adbc1s0b0eb : tensor<384xf32>
    %advhs0b0eb = stablehlo.divide %advns0b0eb, %adbc2s0b0eb : tensor<384xf32>
    %adlrs0b0eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss0b0eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs0b0eb = stablehlo.sqrt %advhs0b0eb : tensor<384xf32>
    %addens0b0eb = stablehlo.add %adsqs0b0eb, %adepss0b0eb : tensor<384xf32>
    %adrats0b0eb = stablehlo.divide %admhs0b0eb, %addens0b0eb : tensor<384xf32>
    %adsts0b0eb = stablehlo.multiply %adlrs0b0eb, %adrats0b0eb : tensor<384xf32>
    %adsubs0b0eb = stablehlo.subtract %s0b0eb, %adsts0b0eb : tensor<384xf32>
    %adwds0b0eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs0b0eb = stablehlo.multiply %adwds0b0eb, %adlrs0b0eb : tensor<384xf32>
    %adwdps0b0eb = stablehlo.multiply %adwdlrs0b0eb, %s0b0eb : tensor<384xf32>
    %adnews0b0eb = stablehlo.subtract %adsubs0b0eb, %adwdps0b0eb : tensor<384xf32>
    %adb1s0b0pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adob1s0b0pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %admss0b0pW = stablehlo.multiply %adb1s0b0pW, %s0b0pWm : tensor<96x384x1x1xf32>
    %admgs0b0pW = stablehlo.multiply %adob1s0b0pW, %s0b0dpW : tensor<96x384x1x1xf32>
    %admns0b0pW = stablehlo.add %admss0b0pW, %admgs0b0pW : tensor<96x384x1x1xf32>
    %adb2s0b0pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adob2s0b0pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %advss0b0pW = stablehlo.multiply %adb2s0b0pW, %s0b0pWv : tensor<96x384x1x1xf32>
    %adg2s0b0pW = stablehlo.multiply %s0b0dpW, %s0b0dpW : tensor<96x384x1x1xf32>
    %advgs0b0pW = stablehlo.multiply %adob2s0b0pW, %adg2s0b0pW : tensor<96x384x1x1xf32>
    %advns0b0pW = stablehlo.add %advss0b0pW, %advgs0b0pW : tensor<96x384x1x1xf32>
    %adbc1s0b0pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adbc2s0b0pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %admhs0b0pW = stablehlo.divide %admns0b0pW, %adbc1s0b0pW : tensor<96x384x1x1xf32>
    %advhs0b0pW = stablehlo.divide %advns0b0pW, %adbc2s0b0pW : tensor<96x384x1x1xf32>
    %adlrs0b0pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adepss0b0pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adsqs0b0pW = stablehlo.sqrt %advhs0b0pW : tensor<96x384x1x1xf32>
    %addens0b0pW = stablehlo.add %adsqs0b0pW, %adepss0b0pW : tensor<96x384x1x1xf32>
    %adrats0b0pW = stablehlo.divide %admhs0b0pW, %addens0b0pW : tensor<96x384x1x1xf32>
    %adsts0b0pW = stablehlo.multiply %adlrs0b0pW, %adrats0b0pW : tensor<96x384x1x1xf32>
    %adsubs0b0pW = stablehlo.subtract %s0b0pW, %adsts0b0pW : tensor<96x384x1x1xf32>
    %adwds0b0pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adwdlrs0b0pW = stablehlo.multiply %adwds0b0pW, %adlrs0b0pW : tensor<96x384x1x1xf32>
    %adwdps0b0pW = stablehlo.multiply %adwdlrs0b0pW, %s0b0pW : tensor<96x384x1x1xf32>
    %adnews0b0pW = stablehlo.subtract %adsubs0b0pW, %adwdps0b0pW : tensor<96x384x1x1xf32>
    %adb1s0b0pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1s0b0pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admss0b0pb = stablehlo.multiply %adb1s0b0pb, %s0b0pbm : tensor<96xf32>
    %admgs0b0pb = stablehlo.multiply %adob1s0b0pb, %s0b0dpb : tensor<96xf32>
    %admns0b0pb = stablehlo.add %admss0b0pb, %admgs0b0pb : tensor<96xf32>
    %adb2s0b0pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2s0b0pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advss0b0pb = stablehlo.multiply %adb2s0b0pb, %s0b0pbv : tensor<96xf32>
    %adg2s0b0pb = stablehlo.multiply %s0b0dpb, %s0b0dpb : tensor<96xf32>
    %advgs0b0pb = stablehlo.multiply %adob2s0b0pb, %adg2s0b0pb : tensor<96xf32>
    %advns0b0pb = stablehlo.add %advss0b0pb, %advgs0b0pb : tensor<96xf32>
    %adbc1s0b0pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2s0b0pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhs0b0pb = stablehlo.divide %admns0b0pb, %adbc1s0b0pb : tensor<96xf32>
    %advhs0b0pb = stablehlo.divide %advns0b0pb, %adbc2s0b0pb : tensor<96xf32>
    %adlrs0b0pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepss0b0pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqs0b0pb = stablehlo.sqrt %advhs0b0pb : tensor<96xf32>
    %addens0b0pb = stablehlo.add %adsqs0b0pb, %adepss0b0pb : tensor<96xf32>
    %adrats0b0pb = stablehlo.divide %admhs0b0pb, %addens0b0pb : tensor<96xf32>
    %adsts0b0pb = stablehlo.multiply %adlrs0b0pb, %adrats0b0pb : tensor<96xf32>
    %adsubs0b0pb = stablehlo.subtract %s0b0pb, %adsts0b0pb : tensor<96xf32>
    %adwds0b0pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrs0b0pb = stablehlo.multiply %adwds0b0pb, %adlrs0b0pb : tensor<96xf32>
    %adwdps0b0pb = stablehlo.multiply %adwdlrs0b0pb, %s0b0pb : tensor<96xf32>
    %adnews0b0pb = stablehlo.subtract %adsubs0b0pb, %adwdps0b0pb : tensor<96xf32>
    %adb1s0b0lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1s0b0lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admss0b0lg = stablehlo.multiply %adb1s0b0lg, %s0b0lgm : tensor<96xf32>
    %admgs0b0lg = stablehlo.multiply %adob1s0b0lg, %s0b0dlsdg : tensor<96xf32>
    %admns0b0lg = stablehlo.add %admss0b0lg, %admgs0b0lg : tensor<96xf32>
    %adb2s0b0lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2s0b0lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advss0b0lg = stablehlo.multiply %adb2s0b0lg, %s0b0lgv : tensor<96xf32>
    %adg2s0b0lg = stablehlo.multiply %s0b0dlsdg, %s0b0dlsdg : tensor<96xf32>
    %advgs0b0lg = stablehlo.multiply %adob2s0b0lg, %adg2s0b0lg : tensor<96xf32>
    %advns0b0lg = stablehlo.add %advss0b0lg, %advgs0b0lg : tensor<96xf32>
    %adbc1s0b0lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2s0b0lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhs0b0lg = stablehlo.divide %admns0b0lg, %adbc1s0b0lg : tensor<96xf32>
    %advhs0b0lg = stablehlo.divide %advns0b0lg, %adbc2s0b0lg : tensor<96xf32>
    %adlrs0b0lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepss0b0lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqs0b0lg = stablehlo.sqrt %advhs0b0lg : tensor<96xf32>
    %addens0b0lg = stablehlo.add %adsqs0b0lg, %adepss0b0lg : tensor<96xf32>
    %adrats0b0lg = stablehlo.divide %admhs0b0lg, %addens0b0lg : tensor<96xf32>
    %adsts0b0lg = stablehlo.multiply %adlrs0b0lg, %adrats0b0lg : tensor<96xf32>
    %adsubs0b0lg = stablehlo.subtract %s0b0lg, %adsts0b0lg : tensor<96xf32>
    %adwds0b0lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrs0b0lg = stablehlo.multiply %adwds0b0lg, %adlrs0b0lg : tensor<96xf32>
    %adwdps0b0lg = stablehlo.multiply %adwdlrs0b0lg, %s0b0lg : tensor<96xf32>
    %adnews0b0lg = stablehlo.subtract %adsubs0b0lg, %adwdps0b0lg : tensor<96xf32>
    %adb1s0b1dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adob1s0b1dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %admss0b1dW = stablehlo.multiply %adb1s0b1dW, %s0b1dWm : tensor<96x1x7x7xf32>
    %admgs0b1dW = stablehlo.multiply %adob1s0b1dW, %s0b1ddW : tensor<96x1x7x7xf32>
    %admns0b1dW = stablehlo.add %admss0b1dW, %admgs0b1dW : tensor<96x1x7x7xf32>
    %adb2s0b1dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adob2s0b1dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %advss0b1dW = stablehlo.multiply %adb2s0b1dW, %s0b1dWv : tensor<96x1x7x7xf32>
    %adg2s0b1dW = stablehlo.multiply %s0b1ddW, %s0b1ddW : tensor<96x1x7x7xf32>
    %advgs0b1dW = stablehlo.multiply %adob2s0b1dW, %adg2s0b1dW : tensor<96x1x7x7xf32>
    %advns0b1dW = stablehlo.add %advss0b1dW, %advgs0b1dW : tensor<96x1x7x7xf32>
    %adbc1s0b1dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adbc2s0b1dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %admhs0b1dW = stablehlo.divide %admns0b1dW, %adbc1s0b1dW : tensor<96x1x7x7xf32>
    %advhs0b1dW = stablehlo.divide %advns0b1dW, %adbc2s0b1dW : tensor<96x1x7x7xf32>
    %adlrs0b1dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adepss0b1dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adsqs0b1dW = stablehlo.sqrt %advhs0b1dW : tensor<96x1x7x7xf32>
    %addens0b1dW = stablehlo.add %adsqs0b1dW, %adepss0b1dW : tensor<96x1x7x7xf32>
    %adrats0b1dW = stablehlo.divide %admhs0b1dW, %addens0b1dW : tensor<96x1x7x7xf32>
    %adsts0b1dW = stablehlo.multiply %adlrs0b1dW, %adrats0b1dW : tensor<96x1x7x7xf32>
    %adsubs0b1dW = stablehlo.subtract %s0b1dW, %adsts0b1dW : tensor<96x1x7x7xf32>
    %adwds0b1dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adwdlrs0b1dW = stablehlo.multiply %adwds0b1dW, %adlrs0b1dW : tensor<96x1x7x7xf32>
    %adwdps0b1dW = stablehlo.multiply %adwdlrs0b1dW, %s0b1dW : tensor<96x1x7x7xf32>
    %adnews0b1dW = stablehlo.subtract %adsubs0b1dW, %adwdps0b1dW : tensor<96x1x7x7xf32>
    %adb1s0b1db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1s0b1db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admss0b1db = stablehlo.multiply %adb1s0b1db, %s0b1dbm : tensor<96xf32>
    %admgs0b1db = stablehlo.multiply %adob1s0b1db, %s0b1ddb : tensor<96xf32>
    %admns0b1db = stablehlo.add %admss0b1db, %admgs0b1db : tensor<96xf32>
    %adb2s0b1db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2s0b1db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advss0b1db = stablehlo.multiply %adb2s0b1db, %s0b1dbv : tensor<96xf32>
    %adg2s0b1db = stablehlo.multiply %s0b1ddb, %s0b1ddb : tensor<96xf32>
    %advgs0b1db = stablehlo.multiply %adob2s0b1db, %adg2s0b1db : tensor<96xf32>
    %advns0b1db = stablehlo.add %advss0b1db, %advgs0b1db : tensor<96xf32>
    %adbc1s0b1db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2s0b1db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhs0b1db = stablehlo.divide %admns0b1db, %adbc1s0b1db : tensor<96xf32>
    %advhs0b1db = stablehlo.divide %advns0b1db, %adbc2s0b1db : tensor<96xf32>
    %adlrs0b1db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepss0b1db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqs0b1db = stablehlo.sqrt %advhs0b1db : tensor<96xf32>
    %addens0b1db = stablehlo.add %adsqs0b1db, %adepss0b1db : tensor<96xf32>
    %adrats0b1db = stablehlo.divide %admhs0b1db, %addens0b1db : tensor<96xf32>
    %adsts0b1db = stablehlo.multiply %adlrs0b1db, %adrats0b1db : tensor<96xf32>
    %adsubs0b1db = stablehlo.subtract %s0b1db, %adsts0b1db : tensor<96xf32>
    %adwds0b1db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrs0b1db = stablehlo.multiply %adwds0b1db, %adlrs0b1db : tensor<96xf32>
    %adwdps0b1db = stablehlo.multiply %adwdlrs0b1db, %s0b1db : tensor<96xf32>
    %adnews0b1db = stablehlo.subtract %adsubs0b1db, %adwdps0b1db : tensor<96xf32>
    %adb1s0b1ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s0b1ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss0b1ng = stablehlo.multiply %adb1s0b1ng, %s0b1ngm : tensor<f32>
    %admgs0b1ng = stablehlo.multiply %adob1s0b1ng, %s0b1dndg : tensor<f32>
    %admns0b1ng = stablehlo.add %admss0b1ng, %admgs0b1ng : tensor<f32>
    %adb2s0b1ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s0b1ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss0b1ng = stablehlo.multiply %adb2s0b1ng, %s0b1ngv : tensor<f32>
    %adg2s0b1ng = stablehlo.multiply %s0b1dndg, %s0b1dndg : tensor<f32>
    %advgs0b1ng = stablehlo.multiply %adob2s0b1ng, %adg2s0b1ng : tensor<f32>
    %advns0b1ng = stablehlo.add %advss0b1ng, %advgs0b1ng : tensor<f32>
    %adbc1s0b1ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s0b1ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs0b1ng = stablehlo.divide %admns0b1ng, %adbc1s0b1ng : tensor<f32>
    %advhs0b1ng = stablehlo.divide %advns0b1ng, %adbc2s0b1ng : tensor<f32>
    %adlrs0b1ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss0b1ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs0b1ng = stablehlo.sqrt %advhs0b1ng : tensor<f32>
    %addens0b1ng = stablehlo.add %adsqs0b1ng, %adepss0b1ng : tensor<f32>
    %adrats0b1ng = stablehlo.divide %admhs0b1ng, %addens0b1ng : tensor<f32>
    %adsts0b1ng = stablehlo.multiply %adlrs0b1ng, %adrats0b1ng : tensor<f32>
    %adsubs0b1ng = stablehlo.subtract %s0b1ng, %adsts0b1ng : tensor<f32>
    %adwds0b1ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs0b1ng = stablehlo.multiply %adwds0b1ng, %adlrs0b1ng : tensor<f32>
    %adwdps0b1ng = stablehlo.multiply %adwdlrs0b1ng, %s0b1ng : tensor<f32>
    %adnews0b1ng = stablehlo.subtract %adsubs0b1ng, %adwdps0b1ng : tensor<f32>
    %adb1s0b1nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s0b1nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss0b1nbt = stablehlo.multiply %adb1s0b1nbt, %s0b1nbtm : tensor<f32>
    %admgs0b1nbt = stablehlo.multiply %adob1s0b1nbt, %s0b1dndb : tensor<f32>
    %admns0b1nbt = stablehlo.add %admss0b1nbt, %admgs0b1nbt : tensor<f32>
    %adb2s0b1nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s0b1nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss0b1nbt = stablehlo.multiply %adb2s0b1nbt, %s0b1nbtv : tensor<f32>
    %adg2s0b1nbt = stablehlo.multiply %s0b1dndb, %s0b1dndb : tensor<f32>
    %advgs0b1nbt = stablehlo.multiply %adob2s0b1nbt, %adg2s0b1nbt : tensor<f32>
    %advns0b1nbt = stablehlo.add %advss0b1nbt, %advgs0b1nbt : tensor<f32>
    %adbc1s0b1nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s0b1nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs0b1nbt = stablehlo.divide %admns0b1nbt, %adbc1s0b1nbt : tensor<f32>
    %advhs0b1nbt = stablehlo.divide %advns0b1nbt, %adbc2s0b1nbt : tensor<f32>
    %adlrs0b1nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss0b1nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs0b1nbt = stablehlo.sqrt %advhs0b1nbt : tensor<f32>
    %addens0b1nbt = stablehlo.add %adsqs0b1nbt, %adepss0b1nbt : tensor<f32>
    %adrats0b1nbt = stablehlo.divide %admhs0b1nbt, %addens0b1nbt : tensor<f32>
    %adsts0b1nbt = stablehlo.multiply %adlrs0b1nbt, %adrats0b1nbt : tensor<f32>
    %adsubs0b1nbt = stablehlo.subtract %s0b1nbt, %adsts0b1nbt : tensor<f32>
    %adwds0b1nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs0b1nbt = stablehlo.multiply %adwds0b1nbt, %adlrs0b1nbt : tensor<f32>
    %adwdps0b1nbt = stablehlo.multiply %adwdlrs0b1nbt, %s0b1nbt : tensor<f32>
    %adnews0b1nbt = stablehlo.subtract %adsubs0b1nbt, %adwdps0b1nbt : tensor<f32>
    %adb1s0b1eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adob1s0b1eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %admss0b1eW = stablehlo.multiply %adb1s0b1eW, %s0b1eWm : tensor<384x96x1x1xf32>
    %admgs0b1eW = stablehlo.multiply %adob1s0b1eW, %s0b1deW : tensor<384x96x1x1xf32>
    %admns0b1eW = stablehlo.add %admss0b1eW, %admgs0b1eW : tensor<384x96x1x1xf32>
    %adb2s0b1eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adob2s0b1eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %advss0b1eW = stablehlo.multiply %adb2s0b1eW, %s0b1eWv : tensor<384x96x1x1xf32>
    %adg2s0b1eW = stablehlo.multiply %s0b1deW, %s0b1deW : tensor<384x96x1x1xf32>
    %advgs0b1eW = stablehlo.multiply %adob2s0b1eW, %adg2s0b1eW : tensor<384x96x1x1xf32>
    %advns0b1eW = stablehlo.add %advss0b1eW, %advgs0b1eW : tensor<384x96x1x1xf32>
    %adbc1s0b1eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adbc2s0b1eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %admhs0b1eW = stablehlo.divide %admns0b1eW, %adbc1s0b1eW : tensor<384x96x1x1xf32>
    %advhs0b1eW = stablehlo.divide %advns0b1eW, %adbc2s0b1eW : tensor<384x96x1x1xf32>
    %adlrs0b1eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adepss0b1eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adsqs0b1eW = stablehlo.sqrt %advhs0b1eW : tensor<384x96x1x1xf32>
    %addens0b1eW = stablehlo.add %adsqs0b1eW, %adepss0b1eW : tensor<384x96x1x1xf32>
    %adrats0b1eW = stablehlo.divide %admhs0b1eW, %addens0b1eW : tensor<384x96x1x1xf32>
    %adsts0b1eW = stablehlo.multiply %adlrs0b1eW, %adrats0b1eW : tensor<384x96x1x1xf32>
    %adsubs0b1eW = stablehlo.subtract %s0b1eW, %adsts0b1eW : tensor<384x96x1x1xf32>
    %adwds0b1eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adwdlrs0b1eW = stablehlo.multiply %adwds0b1eW, %adlrs0b1eW : tensor<384x96x1x1xf32>
    %adwdps0b1eW = stablehlo.multiply %adwdlrs0b1eW, %s0b1eW : tensor<384x96x1x1xf32>
    %adnews0b1eW = stablehlo.subtract %adsubs0b1eW, %adwdps0b1eW : tensor<384x96x1x1xf32>
    %adb1s0b1eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s0b1eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss0b1eb = stablehlo.multiply %adb1s0b1eb, %s0b1ebm : tensor<384xf32>
    %admgs0b1eb = stablehlo.multiply %adob1s0b1eb, %s0b1deb : tensor<384xf32>
    %admns0b1eb = stablehlo.add %admss0b1eb, %admgs0b1eb : tensor<384xf32>
    %adb2s0b1eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s0b1eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss0b1eb = stablehlo.multiply %adb2s0b1eb, %s0b1ebv : tensor<384xf32>
    %adg2s0b1eb = stablehlo.multiply %s0b1deb, %s0b1deb : tensor<384xf32>
    %advgs0b1eb = stablehlo.multiply %adob2s0b1eb, %adg2s0b1eb : tensor<384xf32>
    %advns0b1eb = stablehlo.add %advss0b1eb, %advgs0b1eb : tensor<384xf32>
    %adbc1s0b1eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s0b1eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs0b1eb = stablehlo.divide %admns0b1eb, %adbc1s0b1eb : tensor<384xf32>
    %advhs0b1eb = stablehlo.divide %advns0b1eb, %adbc2s0b1eb : tensor<384xf32>
    %adlrs0b1eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss0b1eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs0b1eb = stablehlo.sqrt %advhs0b1eb : tensor<384xf32>
    %addens0b1eb = stablehlo.add %adsqs0b1eb, %adepss0b1eb : tensor<384xf32>
    %adrats0b1eb = stablehlo.divide %admhs0b1eb, %addens0b1eb : tensor<384xf32>
    %adsts0b1eb = stablehlo.multiply %adlrs0b1eb, %adrats0b1eb : tensor<384xf32>
    %adsubs0b1eb = stablehlo.subtract %s0b1eb, %adsts0b1eb : tensor<384xf32>
    %adwds0b1eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs0b1eb = stablehlo.multiply %adwds0b1eb, %adlrs0b1eb : tensor<384xf32>
    %adwdps0b1eb = stablehlo.multiply %adwdlrs0b1eb, %s0b1eb : tensor<384xf32>
    %adnews0b1eb = stablehlo.subtract %adsubs0b1eb, %adwdps0b1eb : tensor<384xf32>
    %adb1s0b1pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adob1s0b1pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %admss0b1pW = stablehlo.multiply %adb1s0b1pW, %s0b1pWm : tensor<96x384x1x1xf32>
    %admgs0b1pW = stablehlo.multiply %adob1s0b1pW, %s0b1dpW : tensor<96x384x1x1xf32>
    %admns0b1pW = stablehlo.add %admss0b1pW, %admgs0b1pW : tensor<96x384x1x1xf32>
    %adb2s0b1pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adob2s0b1pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %advss0b1pW = stablehlo.multiply %adb2s0b1pW, %s0b1pWv : tensor<96x384x1x1xf32>
    %adg2s0b1pW = stablehlo.multiply %s0b1dpW, %s0b1dpW : tensor<96x384x1x1xf32>
    %advgs0b1pW = stablehlo.multiply %adob2s0b1pW, %adg2s0b1pW : tensor<96x384x1x1xf32>
    %advns0b1pW = stablehlo.add %advss0b1pW, %advgs0b1pW : tensor<96x384x1x1xf32>
    %adbc1s0b1pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adbc2s0b1pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %admhs0b1pW = stablehlo.divide %admns0b1pW, %adbc1s0b1pW : tensor<96x384x1x1xf32>
    %advhs0b1pW = stablehlo.divide %advns0b1pW, %adbc2s0b1pW : tensor<96x384x1x1xf32>
    %adlrs0b1pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adepss0b1pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adsqs0b1pW = stablehlo.sqrt %advhs0b1pW : tensor<96x384x1x1xf32>
    %addens0b1pW = stablehlo.add %adsqs0b1pW, %adepss0b1pW : tensor<96x384x1x1xf32>
    %adrats0b1pW = stablehlo.divide %admhs0b1pW, %addens0b1pW : tensor<96x384x1x1xf32>
    %adsts0b1pW = stablehlo.multiply %adlrs0b1pW, %adrats0b1pW : tensor<96x384x1x1xf32>
    %adsubs0b1pW = stablehlo.subtract %s0b1pW, %adsts0b1pW : tensor<96x384x1x1xf32>
    %adwds0b1pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adwdlrs0b1pW = stablehlo.multiply %adwds0b1pW, %adlrs0b1pW : tensor<96x384x1x1xf32>
    %adwdps0b1pW = stablehlo.multiply %adwdlrs0b1pW, %s0b1pW : tensor<96x384x1x1xf32>
    %adnews0b1pW = stablehlo.subtract %adsubs0b1pW, %adwdps0b1pW : tensor<96x384x1x1xf32>
    %adb1s0b1pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1s0b1pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admss0b1pb = stablehlo.multiply %adb1s0b1pb, %s0b1pbm : tensor<96xf32>
    %admgs0b1pb = stablehlo.multiply %adob1s0b1pb, %s0b1dpb : tensor<96xf32>
    %admns0b1pb = stablehlo.add %admss0b1pb, %admgs0b1pb : tensor<96xf32>
    %adb2s0b1pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2s0b1pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advss0b1pb = stablehlo.multiply %adb2s0b1pb, %s0b1pbv : tensor<96xf32>
    %adg2s0b1pb = stablehlo.multiply %s0b1dpb, %s0b1dpb : tensor<96xf32>
    %advgs0b1pb = stablehlo.multiply %adob2s0b1pb, %adg2s0b1pb : tensor<96xf32>
    %advns0b1pb = stablehlo.add %advss0b1pb, %advgs0b1pb : tensor<96xf32>
    %adbc1s0b1pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2s0b1pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhs0b1pb = stablehlo.divide %admns0b1pb, %adbc1s0b1pb : tensor<96xf32>
    %advhs0b1pb = stablehlo.divide %advns0b1pb, %adbc2s0b1pb : tensor<96xf32>
    %adlrs0b1pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepss0b1pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqs0b1pb = stablehlo.sqrt %advhs0b1pb : tensor<96xf32>
    %addens0b1pb = stablehlo.add %adsqs0b1pb, %adepss0b1pb : tensor<96xf32>
    %adrats0b1pb = stablehlo.divide %admhs0b1pb, %addens0b1pb : tensor<96xf32>
    %adsts0b1pb = stablehlo.multiply %adlrs0b1pb, %adrats0b1pb : tensor<96xf32>
    %adsubs0b1pb = stablehlo.subtract %s0b1pb, %adsts0b1pb : tensor<96xf32>
    %adwds0b1pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrs0b1pb = stablehlo.multiply %adwds0b1pb, %adlrs0b1pb : tensor<96xf32>
    %adwdps0b1pb = stablehlo.multiply %adwdlrs0b1pb, %s0b1pb : tensor<96xf32>
    %adnews0b1pb = stablehlo.subtract %adsubs0b1pb, %adwdps0b1pb : tensor<96xf32>
    %adb1s0b1lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1s0b1lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admss0b1lg = stablehlo.multiply %adb1s0b1lg, %s0b1lgm : tensor<96xf32>
    %admgs0b1lg = stablehlo.multiply %adob1s0b1lg, %s0b1dlsdg : tensor<96xf32>
    %admns0b1lg = stablehlo.add %admss0b1lg, %admgs0b1lg : tensor<96xf32>
    %adb2s0b1lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2s0b1lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advss0b1lg = stablehlo.multiply %adb2s0b1lg, %s0b1lgv : tensor<96xf32>
    %adg2s0b1lg = stablehlo.multiply %s0b1dlsdg, %s0b1dlsdg : tensor<96xf32>
    %advgs0b1lg = stablehlo.multiply %adob2s0b1lg, %adg2s0b1lg : tensor<96xf32>
    %advns0b1lg = stablehlo.add %advss0b1lg, %advgs0b1lg : tensor<96xf32>
    %adbc1s0b1lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2s0b1lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhs0b1lg = stablehlo.divide %admns0b1lg, %adbc1s0b1lg : tensor<96xf32>
    %advhs0b1lg = stablehlo.divide %advns0b1lg, %adbc2s0b1lg : tensor<96xf32>
    %adlrs0b1lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepss0b1lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqs0b1lg = stablehlo.sqrt %advhs0b1lg : tensor<96xf32>
    %addens0b1lg = stablehlo.add %adsqs0b1lg, %adepss0b1lg : tensor<96xf32>
    %adrats0b1lg = stablehlo.divide %admhs0b1lg, %addens0b1lg : tensor<96xf32>
    %adsts0b1lg = stablehlo.multiply %adlrs0b1lg, %adrats0b1lg : tensor<96xf32>
    %adsubs0b1lg = stablehlo.subtract %s0b1lg, %adsts0b1lg : tensor<96xf32>
    %adwds0b1lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrs0b1lg = stablehlo.multiply %adwds0b1lg, %adlrs0b1lg : tensor<96xf32>
    %adwdps0b1lg = stablehlo.multiply %adwdlrs0b1lg, %s0b1lg : tensor<96xf32>
    %adnews0b1lg = stablehlo.subtract %adsubs0b1lg, %adwdps0b1lg : tensor<96xf32>
    %adb1s0b2dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adob1s0b2dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %admss0b2dW = stablehlo.multiply %adb1s0b2dW, %s0b2dWm : tensor<96x1x7x7xf32>
    %admgs0b2dW = stablehlo.multiply %adob1s0b2dW, %s0b2ddW : tensor<96x1x7x7xf32>
    %admns0b2dW = stablehlo.add %admss0b2dW, %admgs0b2dW : tensor<96x1x7x7xf32>
    %adb2s0b2dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adob2s0b2dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %advss0b2dW = stablehlo.multiply %adb2s0b2dW, %s0b2dWv : tensor<96x1x7x7xf32>
    %adg2s0b2dW = stablehlo.multiply %s0b2ddW, %s0b2ddW : tensor<96x1x7x7xf32>
    %advgs0b2dW = stablehlo.multiply %adob2s0b2dW, %adg2s0b2dW : tensor<96x1x7x7xf32>
    %advns0b2dW = stablehlo.add %advss0b2dW, %advgs0b2dW : tensor<96x1x7x7xf32>
    %adbc1s0b2dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adbc2s0b2dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %admhs0b2dW = stablehlo.divide %admns0b2dW, %adbc1s0b2dW : tensor<96x1x7x7xf32>
    %advhs0b2dW = stablehlo.divide %advns0b2dW, %adbc2s0b2dW : tensor<96x1x7x7xf32>
    %adlrs0b2dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adepss0b2dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adsqs0b2dW = stablehlo.sqrt %advhs0b2dW : tensor<96x1x7x7xf32>
    %addens0b2dW = stablehlo.add %adsqs0b2dW, %adepss0b2dW : tensor<96x1x7x7xf32>
    %adrats0b2dW = stablehlo.divide %admhs0b2dW, %addens0b2dW : tensor<96x1x7x7xf32>
    %adsts0b2dW = stablehlo.multiply %adlrs0b2dW, %adrats0b2dW : tensor<96x1x7x7xf32>
    %adsubs0b2dW = stablehlo.subtract %s0b2dW, %adsts0b2dW : tensor<96x1x7x7xf32>
    %adwds0b2dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %adwdlrs0b2dW = stablehlo.multiply %adwds0b2dW, %adlrs0b2dW : tensor<96x1x7x7xf32>
    %adwdps0b2dW = stablehlo.multiply %adwdlrs0b2dW, %s0b2dW : tensor<96x1x7x7xf32>
    %adnews0b2dW = stablehlo.subtract %adsubs0b2dW, %adwdps0b2dW : tensor<96x1x7x7xf32>
    %adb1s0b2db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1s0b2db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admss0b2db = stablehlo.multiply %adb1s0b2db, %s0b2dbm : tensor<96xf32>
    %admgs0b2db = stablehlo.multiply %adob1s0b2db, %s0b2ddb : tensor<96xf32>
    %admns0b2db = stablehlo.add %admss0b2db, %admgs0b2db : tensor<96xf32>
    %adb2s0b2db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2s0b2db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advss0b2db = stablehlo.multiply %adb2s0b2db, %s0b2dbv : tensor<96xf32>
    %adg2s0b2db = stablehlo.multiply %s0b2ddb, %s0b2ddb : tensor<96xf32>
    %advgs0b2db = stablehlo.multiply %adob2s0b2db, %adg2s0b2db : tensor<96xf32>
    %advns0b2db = stablehlo.add %advss0b2db, %advgs0b2db : tensor<96xf32>
    %adbc1s0b2db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2s0b2db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhs0b2db = stablehlo.divide %admns0b2db, %adbc1s0b2db : tensor<96xf32>
    %advhs0b2db = stablehlo.divide %advns0b2db, %adbc2s0b2db : tensor<96xf32>
    %adlrs0b2db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepss0b2db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqs0b2db = stablehlo.sqrt %advhs0b2db : tensor<96xf32>
    %addens0b2db = stablehlo.add %adsqs0b2db, %adepss0b2db : tensor<96xf32>
    %adrats0b2db = stablehlo.divide %admhs0b2db, %addens0b2db : tensor<96xf32>
    %adsts0b2db = stablehlo.multiply %adlrs0b2db, %adrats0b2db : tensor<96xf32>
    %adsubs0b2db = stablehlo.subtract %s0b2db, %adsts0b2db : tensor<96xf32>
    %adwds0b2db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrs0b2db = stablehlo.multiply %adwds0b2db, %adlrs0b2db : tensor<96xf32>
    %adwdps0b2db = stablehlo.multiply %adwdlrs0b2db, %s0b2db : tensor<96xf32>
    %adnews0b2db = stablehlo.subtract %adsubs0b2db, %adwdps0b2db : tensor<96xf32>
    %adb1s0b2ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s0b2ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss0b2ng = stablehlo.multiply %adb1s0b2ng, %s0b2ngm : tensor<f32>
    %admgs0b2ng = stablehlo.multiply %adob1s0b2ng, %s0b2dndg : tensor<f32>
    %admns0b2ng = stablehlo.add %admss0b2ng, %admgs0b2ng : tensor<f32>
    %adb2s0b2ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s0b2ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss0b2ng = stablehlo.multiply %adb2s0b2ng, %s0b2ngv : tensor<f32>
    %adg2s0b2ng = stablehlo.multiply %s0b2dndg, %s0b2dndg : tensor<f32>
    %advgs0b2ng = stablehlo.multiply %adob2s0b2ng, %adg2s0b2ng : tensor<f32>
    %advns0b2ng = stablehlo.add %advss0b2ng, %advgs0b2ng : tensor<f32>
    %adbc1s0b2ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s0b2ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs0b2ng = stablehlo.divide %admns0b2ng, %adbc1s0b2ng : tensor<f32>
    %advhs0b2ng = stablehlo.divide %advns0b2ng, %adbc2s0b2ng : tensor<f32>
    %adlrs0b2ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss0b2ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs0b2ng = stablehlo.sqrt %advhs0b2ng : tensor<f32>
    %addens0b2ng = stablehlo.add %adsqs0b2ng, %adepss0b2ng : tensor<f32>
    %adrats0b2ng = stablehlo.divide %admhs0b2ng, %addens0b2ng : tensor<f32>
    %adsts0b2ng = stablehlo.multiply %adlrs0b2ng, %adrats0b2ng : tensor<f32>
    %adsubs0b2ng = stablehlo.subtract %s0b2ng, %adsts0b2ng : tensor<f32>
    %adwds0b2ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs0b2ng = stablehlo.multiply %adwds0b2ng, %adlrs0b2ng : tensor<f32>
    %adwdps0b2ng = stablehlo.multiply %adwdlrs0b2ng, %s0b2ng : tensor<f32>
    %adnews0b2ng = stablehlo.subtract %adsubs0b2ng, %adwdps0b2ng : tensor<f32>
    %adb1s0b2nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s0b2nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss0b2nbt = stablehlo.multiply %adb1s0b2nbt, %s0b2nbtm : tensor<f32>
    %admgs0b2nbt = stablehlo.multiply %adob1s0b2nbt, %s0b2dndb : tensor<f32>
    %admns0b2nbt = stablehlo.add %admss0b2nbt, %admgs0b2nbt : tensor<f32>
    %adb2s0b2nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s0b2nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss0b2nbt = stablehlo.multiply %adb2s0b2nbt, %s0b2nbtv : tensor<f32>
    %adg2s0b2nbt = stablehlo.multiply %s0b2dndb, %s0b2dndb : tensor<f32>
    %advgs0b2nbt = stablehlo.multiply %adob2s0b2nbt, %adg2s0b2nbt : tensor<f32>
    %advns0b2nbt = stablehlo.add %advss0b2nbt, %advgs0b2nbt : tensor<f32>
    %adbc1s0b2nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s0b2nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs0b2nbt = stablehlo.divide %admns0b2nbt, %adbc1s0b2nbt : tensor<f32>
    %advhs0b2nbt = stablehlo.divide %advns0b2nbt, %adbc2s0b2nbt : tensor<f32>
    %adlrs0b2nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss0b2nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs0b2nbt = stablehlo.sqrt %advhs0b2nbt : tensor<f32>
    %addens0b2nbt = stablehlo.add %adsqs0b2nbt, %adepss0b2nbt : tensor<f32>
    %adrats0b2nbt = stablehlo.divide %admhs0b2nbt, %addens0b2nbt : tensor<f32>
    %adsts0b2nbt = stablehlo.multiply %adlrs0b2nbt, %adrats0b2nbt : tensor<f32>
    %adsubs0b2nbt = stablehlo.subtract %s0b2nbt, %adsts0b2nbt : tensor<f32>
    %adwds0b2nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs0b2nbt = stablehlo.multiply %adwds0b2nbt, %adlrs0b2nbt : tensor<f32>
    %adwdps0b2nbt = stablehlo.multiply %adwdlrs0b2nbt, %s0b2nbt : tensor<f32>
    %adnews0b2nbt = stablehlo.subtract %adsubs0b2nbt, %adwdps0b2nbt : tensor<f32>
    %adb1s0b2eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adob1s0b2eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %admss0b2eW = stablehlo.multiply %adb1s0b2eW, %s0b2eWm : tensor<384x96x1x1xf32>
    %admgs0b2eW = stablehlo.multiply %adob1s0b2eW, %s0b2deW : tensor<384x96x1x1xf32>
    %admns0b2eW = stablehlo.add %admss0b2eW, %admgs0b2eW : tensor<384x96x1x1xf32>
    %adb2s0b2eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adob2s0b2eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %advss0b2eW = stablehlo.multiply %adb2s0b2eW, %s0b2eWv : tensor<384x96x1x1xf32>
    %adg2s0b2eW = stablehlo.multiply %s0b2deW, %s0b2deW : tensor<384x96x1x1xf32>
    %advgs0b2eW = stablehlo.multiply %adob2s0b2eW, %adg2s0b2eW : tensor<384x96x1x1xf32>
    %advns0b2eW = stablehlo.add %advss0b2eW, %advgs0b2eW : tensor<384x96x1x1xf32>
    %adbc1s0b2eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adbc2s0b2eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %admhs0b2eW = stablehlo.divide %admns0b2eW, %adbc1s0b2eW : tensor<384x96x1x1xf32>
    %advhs0b2eW = stablehlo.divide %advns0b2eW, %adbc2s0b2eW : tensor<384x96x1x1xf32>
    %adlrs0b2eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adepss0b2eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adsqs0b2eW = stablehlo.sqrt %advhs0b2eW : tensor<384x96x1x1xf32>
    %addens0b2eW = stablehlo.add %adsqs0b2eW, %adepss0b2eW : tensor<384x96x1x1xf32>
    %adrats0b2eW = stablehlo.divide %admhs0b2eW, %addens0b2eW : tensor<384x96x1x1xf32>
    %adsts0b2eW = stablehlo.multiply %adlrs0b2eW, %adrats0b2eW : tensor<384x96x1x1xf32>
    %adsubs0b2eW = stablehlo.subtract %s0b2eW, %adsts0b2eW : tensor<384x96x1x1xf32>
    %adwds0b2eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %adwdlrs0b2eW = stablehlo.multiply %adwds0b2eW, %adlrs0b2eW : tensor<384x96x1x1xf32>
    %adwdps0b2eW = stablehlo.multiply %adwdlrs0b2eW, %s0b2eW : tensor<384x96x1x1xf32>
    %adnews0b2eW = stablehlo.subtract %adsubs0b2eW, %adwdps0b2eW : tensor<384x96x1x1xf32>
    %adb1s0b2eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s0b2eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss0b2eb = stablehlo.multiply %adb1s0b2eb, %s0b2ebm : tensor<384xf32>
    %admgs0b2eb = stablehlo.multiply %adob1s0b2eb, %s0b2deb : tensor<384xf32>
    %admns0b2eb = stablehlo.add %admss0b2eb, %admgs0b2eb : tensor<384xf32>
    %adb2s0b2eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s0b2eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss0b2eb = stablehlo.multiply %adb2s0b2eb, %s0b2ebv : tensor<384xf32>
    %adg2s0b2eb = stablehlo.multiply %s0b2deb, %s0b2deb : tensor<384xf32>
    %advgs0b2eb = stablehlo.multiply %adob2s0b2eb, %adg2s0b2eb : tensor<384xf32>
    %advns0b2eb = stablehlo.add %advss0b2eb, %advgs0b2eb : tensor<384xf32>
    %adbc1s0b2eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s0b2eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs0b2eb = stablehlo.divide %admns0b2eb, %adbc1s0b2eb : tensor<384xf32>
    %advhs0b2eb = stablehlo.divide %advns0b2eb, %adbc2s0b2eb : tensor<384xf32>
    %adlrs0b2eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss0b2eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs0b2eb = stablehlo.sqrt %advhs0b2eb : tensor<384xf32>
    %addens0b2eb = stablehlo.add %adsqs0b2eb, %adepss0b2eb : tensor<384xf32>
    %adrats0b2eb = stablehlo.divide %admhs0b2eb, %addens0b2eb : tensor<384xf32>
    %adsts0b2eb = stablehlo.multiply %adlrs0b2eb, %adrats0b2eb : tensor<384xf32>
    %adsubs0b2eb = stablehlo.subtract %s0b2eb, %adsts0b2eb : tensor<384xf32>
    %adwds0b2eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs0b2eb = stablehlo.multiply %adwds0b2eb, %adlrs0b2eb : tensor<384xf32>
    %adwdps0b2eb = stablehlo.multiply %adwdlrs0b2eb, %s0b2eb : tensor<384xf32>
    %adnews0b2eb = stablehlo.subtract %adsubs0b2eb, %adwdps0b2eb : tensor<384xf32>
    %adb1s0b2pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adob1s0b2pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %admss0b2pW = stablehlo.multiply %adb1s0b2pW, %s0b2pWm : tensor<96x384x1x1xf32>
    %admgs0b2pW = stablehlo.multiply %adob1s0b2pW, %s0b2dpW : tensor<96x384x1x1xf32>
    %admns0b2pW = stablehlo.add %admss0b2pW, %admgs0b2pW : tensor<96x384x1x1xf32>
    %adb2s0b2pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adob2s0b2pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %advss0b2pW = stablehlo.multiply %adb2s0b2pW, %s0b2pWv : tensor<96x384x1x1xf32>
    %adg2s0b2pW = stablehlo.multiply %s0b2dpW, %s0b2dpW : tensor<96x384x1x1xf32>
    %advgs0b2pW = stablehlo.multiply %adob2s0b2pW, %adg2s0b2pW : tensor<96x384x1x1xf32>
    %advns0b2pW = stablehlo.add %advss0b2pW, %advgs0b2pW : tensor<96x384x1x1xf32>
    %adbc1s0b2pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adbc2s0b2pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %admhs0b2pW = stablehlo.divide %admns0b2pW, %adbc1s0b2pW : tensor<96x384x1x1xf32>
    %advhs0b2pW = stablehlo.divide %advns0b2pW, %adbc2s0b2pW : tensor<96x384x1x1xf32>
    %adlrs0b2pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adepss0b2pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adsqs0b2pW = stablehlo.sqrt %advhs0b2pW : tensor<96x384x1x1xf32>
    %addens0b2pW = stablehlo.add %adsqs0b2pW, %adepss0b2pW : tensor<96x384x1x1xf32>
    %adrats0b2pW = stablehlo.divide %admhs0b2pW, %addens0b2pW : tensor<96x384x1x1xf32>
    %adsts0b2pW = stablehlo.multiply %adlrs0b2pW, %adrats0b2pW : tensor<96x384x1x1xf32>
    %adsubs0b2pW = stablehlo.subtract %s0b2pW, %adsts0b2pW : tensor<96x384x1x1xf32>
    %adwds0b2pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %adwdlrs0b2pW = stablehlo.multiply %adwds0b2pW, %adlrs0b2pW : tensor<96x384x1x1xf32>
    %adwdps0b2pW = stablehlo.multiply %adwdlrs0b2pW, %s0b2pW : tensor<96x384x1x1xf32>
    %adnews0b2pW = stablehlo.subtract %adsubs0b2pW, %adwdps0b2pW : tensor<96x384x1x1xf32>
    %adb1s0b2pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1s0b2pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admss0b2pb = stablehlo.multiply %adb1s0b2pb, %s0b2pbm : tensor<96xf32>
    %admgs0b2pb = stablehlo.multiply %adob1s0b2pb, %s0b2dpb : tensor<96xf32>
    %admns0b2pb = stablehlo.add %admss0b2pb, %admgs0b2pb : tensor<96xf32>
    %adb2s0b2pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2s0b2pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advss0b2pb = stablehlo.multiply %adb2s0b2pb, %s0b2pbv : tensor<96xf32>
    %adg2s0b2pb = stablehlo.multiply %s0b2dpb, %s0b2dpb : tensor<96xf32>
    %advgs0b2pb = stablehlo.multiply %adob2s0b2pb, %adg2s0b2pb : tensor<96xf32>
    %advns0b2pb = stablehlo.add %advss0b2pb, %advgs0b2pb : tensor<96xf32>
    %adbc1s0b2pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2s0b2pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhs0b2pb = stablehlo.divide %admns0b2pb, %adbc1s0b2pb : tensor<96xf32>
    %advhs0b2pb = stablehlo.divide %advns0b2pb, %adbc2s0b2pb : tensor<96xf32>
    %adlrs0b2pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepss0b2pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqs0b2pb = stablehlo.sqrt %advhs0b2pb : tensor<96xf32>
    %addens0b2pb = stablehlo.add %adsqs0b2pb, %adepss0b2pb : tensor<96xf32>
    %adrats0b2pb = stablehlo.divide %admhs0b2pb, %addens0b2pb : tensor<96xf32>
    %adsts0b2pb = stablehlo.multiply %adlrs0b2pb, %adrats0b2pb : tensor<96xf32>
    %adsubs0b2pb = stablehlo.subtract %s0b2pb, %adsts0b2pb : tensor<96xf32>
    %adwds0b2pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrs0b2pb = stablehlo.multiply %adwds0b2pb, %adlrs0b2pb : tensor<96xf32>
    %adwdps0b2pb = stablehlo.multiply %adwdlrs0b2pb, %s0b2pb : tensor<96xf32>
    %adnews0b2pb = stablehlo.subtract %adsubs0b2pb, %adwdps0b2pb : tensor<96xf32>
    %adb1s0b2lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob1s0b2lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admss0b2lg = stablehlo.multiply %adb1s0b2lg, %s0b2lgm : tensor<96xf32>
    %admgs0b2lg = stablehlo.multiply %adob1s0b2lg, %s0b2dlsdg : tensor<96xf32>
    %admns0b2lg = stablehlo.add %admss0b2lg, %admgs0b2lg : tensor<96xf32>
    %adb2s0b2lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adob2s0b2lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %advss0b2lg = stablehlo.multiply %adb2s0b2lg, %s0b2lgv : tensor<96xf32>
    %adg2s0b2lg = stablehlo.multiply %s0b2dlsdg, %s0b2dlsdg : tensor<96xf32>
    %advgs0b2lg = stablehlo.multiply %adob2s0b2lg, %adg2s0b2lg : tensor<96xf32>
    %advns0b2lg = stablehlo.add %advss0b2lg, %advgs0b2lg : tensor<96xf32>
    %adbc1s0b2lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adbc2s0b2lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %admhs0b2lg = stablehlo.divide %admns0b2lg, %adbc1s0b2lg : tensor<96xf32>
    %advhs0b2lg = stablehlo.divide %advns0b2lg, %adbc2s0b2lg : tensor<96xf32>
    %adlrs0b2lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adepss0b2lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adsqs0b2lg = stablehlo.sqrt %advhs0b2lg : tensor<96xf32>
    %addens0b2lg = stablehlo.add %adsqs0b2lg, %adepss0b2lg : tensor<96xf32>
    %adrats0b2lg = stablehlo.divide %admhs0b2lg, %addens0b2lg : tensor<96xf32>
    %adsts0b2lg = stablehlo.multiply %adlrs0b2lg, %adrats0b2lg : tensor<96xf32>
    %adsubs0b2lg = stablehlo.subtract %s0b2lg, %adsts0b2lg : tensor<96xf32>
    %adwds0b2lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %adwdlrs0b2lg = stablehlo.multiply %adwds0b2lg, %adlrs0b2lg : tensor<96xf32>
    %adwdps0b2lg = stablehlo.multiply %adwdlrs0b2lg, %s0b2lg : tensor<96xf32>
    %adnews0b2lg = stablehlo.subtract %adsubs0b2lg, %adwdps0b2lg : tensor<96xf32>
    %adb1d0ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1d0ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admsd0ng = stablehlo.multiply %adb1d0ng, %d0ngm : tensor<f32>
    %admgd0ng = stablehlo.multiply %adob1d0ng, %d0dndg : tensor<f32>
    %admnd0ng = stablehlo.add %admsd0ng, %admgd0ng : tensor<f32>
    %adb2d0ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2d0ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advsd0ng = stablehlo.multiply %adb2d0ng, %d0ngv : tensor<f32>
    %adg2d0ng = stablehlo.multiply %d0dndg, %d0dndg : tensor<f32>
    %advgd0ng = stablehlo.multiply %adob2d0ng, %adg2d0ng : tensor<f32>
    %advnd0ng = stablehlo.add %advsd0ng, %advgd0ng : tensor<f32>
    %adbc1d0ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2d0ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhd0ng = stablehlo.divide %admnd0ng, %adbc1d0ng : tensor<f32>
    %advhd0ng = stablehlo.divide %advnd0ng, %adbc2d0ng : tensor<f32>
    %adlrd0ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepsd0ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqd0ng = stablehlo.sqrt %advhd0ng : tensor<f32>
    %addend0ng = stablehlo.add %adsqd0ng, %adepsd0ng : tensor<f32>
    %adratd0ng = stablehlo.divide %admhd0ng, %addend0ng : tensor<f32>
    %adstd0ng = stablehlo.multiply %adlrd0ng, %adratd0ng : tensor<f32>
    %adsubd0ng = stablehlo.subtract %d0ng, %adstd0ng : tensor<f32>
    %adwdd0ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrd0ng = stablehlo.multiply %adwdd0ng, %adlrd0ng : tensor<f32>
    %adwdpd0ng = stablehlo.multiply %adwdlrd0ng, %d0ng : tensor<f32>
    %adnewd0ng = stablehlo.subtract %adsubd0ng, %adwdpd0ng : tensor<f32>
    %adb1d0nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1d0nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admsd0nbt = stablehlo.multiply %adb1d0nbt, %d0nbtm : tensor<f32>
    %admgd0nbt = stablehlo.multiply %adob1d0nbt, %d0dndb : tensor<f32>
    %admnd0nbt = stablehlo.add %admsd0nbt, %admgd0nbt : tensor<f32>
    %adb2d0nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2d0nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advsd0nbt = stablehlo.multiply %adb2d0nbt, %d0nbtv : tensor<f32>
    %adg2d0nbt = stablehlo.multiply %d0dndb, %d0dndb : tensor<f32>
    %advgd0nbt = stablehlo.multiply %adob2d0nbt, %adg2d0nbt : tensor<f32>
    %advnd0nbt = stablehlo.add %advsd0nbt, %advgd0nbt : tensor<f32>
    %adbc1d0nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2d0nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhd0nbt = stablehlo.divide %admnd0nbt, %adbc1d0nbt : tensor<f32>
    %advhd0nbt = stablehlo.divide %advnd0nbt, %adbc2d0nbt : tensor<f32>
    %adlrd0nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepsd0nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqd0nbt = stablehlo.sqrt %advhd0nbt : tensor<f32>
    %addend0nbt = stablehlo.add %adsqd0nbt, %adepsd0nbt : tensor<f32>
    %adratd0nbt = stablehlo.divide %admhd0nbt, %addend0nbt : tensor<f32>
    %adstd0nbt = stablehlo.multiply %adlrd0nbt, %adratd0nbt : tensor<f32>
    %adsubd0nbt = stablehlo.subtract %d0nbt, %adstd0nbt : tensor<f32>
    %adwdd0nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrd0nbt = stablehlo.multiply %adwdd0nbt, %adlrd0nbt : tensor<f32>
    %adwdpd0nbt = stablehlo.multiply %adwdlrd0nbt, %d0nbt : tensor<f32>
    %adnewd0nbt = stablehlo.subtract %adsubd0nbt, %adwdpd0nbt : tensor<f32>
    %adb1d0W = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %adob1d0W = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %admsd0W = stablehlo.multiply %adb1d0W, %d0Wm : tensor<192x96x2x2xf32>
    %admgd0W = stablehlo.multiply %adob1d0W, %d0dW : tensor<192x96x2x2xf32>
    %admnd0W = stablehlo.add %admsd0W, %admgd0W : tensor<192x96x2x2xf32>
    %adb2d0W = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %adob2d0W = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %advsd0W = stablehlo.multiply %adb2d0W, %d0Wv : tensor<192x96x2x2xf32>
    %adg2d0W = stablehlo.multiply %d0dW, %d0dW : tensor<192x96x2x2xf32>
    %advgd0W = stablehlo.multiply %adob2d0W, %adg2d0W : tensor<192x96x2x2xf32>
    %advnd0W = stablehlo.add %advsd0W, %advgd0W : tensor<192x96x2x2xf32>
    %adbc1d0W = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %adbc2d0W = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %admhd0W = stablehlo.divide %admnd0W, %adbc1d0W : tensor<192x96x2x2xf32>
    %advhd0W = stablehlo.divide %advnd0W, %adbc2d0W : tensor<192x96x2x2xf32>
    %adlrd0W = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %adepsd0W = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %adsqd0W = stablehlo.sqrt %advhd0W : tensor<192x96x2x2xf32>
    %addend0W = stablehlo.add %adsqd0W, %adepsd0W : tensor<192x96x2x2xf32>
    %adratd0W = stablehlo.divide %admhd0W, %addend0W : tensor<192x96x2x2xf32>
    %adstd0W = stablehlo.multiply %adlrd0W, %adratd0W : tensor<192x96x2x2xf32>
    %adsubd0W = stablehlo.subtract %d0W, %adstd0W : tensor<192x96x2x2xf32>
    %adwdd0W = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %adwdlrd0W = stablehlo.multiply %adwdd0W, %adlrd0W : tensor<192x96x2x2xf32>
    %adwdpd0W = stablehlo.multiply %adwdlrd0W, %d0W : tensor<192x96x2x2xf32>
    %adnewd0W = stablehlo.subtract %adsubd0W, %adwdpd0W : tensor<192x96x2x2xf32>
    %adb1d0b = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1d0b = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admsd0b = stablehlo.multiply %adb1d0b, %d0bm : tensor<192xf32>
    %admgd0b = stablehlo.multiply %adob1d0b, %d0db : tensor<192xf32>
    %admnd0b = stablehlo.add %admsd0b, %admgd0b : tensor<192xf32>
    %adb2d0b = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2d0b = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advsd0b = stablehlo.multiply %adb2d0b, %d0bv : tensor<192xf32>
    %adg2d0b = stablehlo.multiply %d0db, %d0db : tensor<192xf32>
    %advgd0b = stablehlo.multiply %adob2d0b, %adg2d0b : tensor<192xf32>
    %advnd0b = stablehlo.add %advsd0b, %advgd0b : tensor<192xf32>
    %adbc1d0b = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2d0b = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhd0b = stablehlo.divide %admnd0b, %adbc1d0b : tensor<192xf32>
    %advhd0b = stablehlo.divide %advnd0b, %adbc2d0b : tensor<192xf32>
    %adlrd0b = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepsd0b = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqd0b = stablehlo.sqrt %advhd0b : tensor<192xf32>
    %addend0b = stablehlo.add %adsqd0b, %adepsd0b : tensor<192xf32>
    %adratd0b = stablehlo.divide %admhd0b, %addend0b : tensor<192xf32>
    %adstd0b = stablehlo.multiply %adlrd0b, %adratd0b : tensor<192xf32>
    %adsubd0b = stablehlo.subtract %d0b, %adstd0b : tensor<192xf32>
    %adwdd0b = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrd0b = stablehlo.multiply %adwdd0b, %adlrd0b : tensor<192xf32>
    %adwdpd0b = stablehlo.multiply %adwdlrd0b, %d0b : tensor<192xf32>
    %adnewd0b = stablehlo.subtract %adsubd0b, %adwdpd0b : tensor<192xf32>
    %adb1s1b0dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adob1s1b0dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %admss1b0dW = stablehlo.multiply %adb1s1b0dW, %s1b0dWm : tensor<192x1x7x7xf32>
    %admgs1b0dW = stablehlo.multiply %adob1s1b0dW, %s1b0ddW : tensor<192x1x7x7xf32>
    %admns1b0dW = stablehlo.add %admss1b0dW, %admgs1b0dW : tensor<192x1x7x7xf32>
    %adb2s1b0dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adob2s1b0dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %advss1b0dW = stablehlo.multiply %adb2s1b0dW, %s1b0dWv : tensor<192x1x7x7xf32>
    %adg2s1b0dW = stablehlo.multiply %s1b0ddW, %s1b0ddW : tensor<192x1x7x7xf32>
    %advgs1b0dW = stablehlo.multiply %adob2s1b0dW, %adg2s1b0dW : tensor<192x1x7x7xf32>
    %advns1b0dW = stablehlo.add %advss1b0dW, %advgs1b0dW : tensor<192x1x7x7xf32>
    %adbc1s1b0dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adbc2s1b0dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %admhs1b0dW = stablehlo.divide %admns1b0dW, %adbc1s1b0dW : tensor<192x1x7x7xf32>
    %advhs1b0dW = stablehlo.divide %advns1b0dW, %adbc2s1b0dW : tensor<192x1x7x7xf32>
    %adlrs1b0dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adepss1b0dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adsqs1b0dW = stablehlo.sqrt %advhs1b0dW : tensor<192x1x7x7xf32>
    %addens1b0dW = stablehlo.add %adsqs1b0dW, %adepss1b0dW : tensor<192x1x7x7xf32>
    %adrats1b0dW = stablehlo.divide %admhs1b0dW, %addens1b0dW : tensor<192x1x7x7xf32>
    %adsts1b0dW = stablehlo.multiply %adlrs1b0dW, %adrats1b0dW : tensor<192x1x7x7xf32>
    %adsubs1b0dW = stablehlo.subtract %s1b0dW, %adsts1b0dW : tensor<192x1x7x7xf32>
    %adwds1b0dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adwdlrs1b0dW = stablehlo.multiply %adwds1b0dW, %adlrs1b0dW : tensor<192x1x7x7xf32>
    %adwdps1b0dW = stablehlo.multiply %adwdlrs1b0dW, %s1b0dW : tensor<192x1x7x7xf32>
    %adnews1b0dW = stablehlo.subtract %adsubs1b0dW, %adwdps1b0dW : tensor<192x1x7x7xf32>
    %adb1s1b0db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1s1b0db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admss1b0db = stablehlo.multiply %adb1s1b0db, %s1b0dbm : tensor<192xf32>
    %admgs1b0db = stablehlo.multiply %adob1s1b0db, %s1b0ddb : tensor<192xf32>
    %admns1b0db = stablehlo.add %admss1b0db, %admgs1b0db : tensor<192xf32>
    %adb2s1b0db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2s1b0db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advss1b0db = stablehlo.multiply %adb2s1b0db, %s1b0dbv : tensor<192xf32>
    %adg2s1b0db = stablehlo.multiply %s1b0ddb, %s1b0ddb : tensor<192xf32>
    %advgs1b0db = stablehlo.multiply %adob2s1b0db, %adg2s1b0db : tensor<192xf32>
    %advns1b0db = stablehlo.add %advss1b0db, %advgs1b0db : tensor<192xf32>
    %adbc1s1b0db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2s1b0db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhs1b0db = stablehlo.divide %admns1b0db, %adbc1s1b0db : tensor<192xf32>
    %advhs1b0db = stablehlo.divide %advns1b0db, %adbc2s1b0db : tensor<192xf32>
    %adlrs1b0db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepss1b0db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqs1b0db = stablehlo.sqrt %advhs1b0db : tensor<192xf32>
    %addens1b0db = stablehlo.add %adsqs1b0db, %adepss1b0db : tensor<192xf32>
    %adrats1b0db = stablehlo.divide %admhs1b0db, %addens1b0db : tensor<192xf32>
    %adsts1b0db = stablehlo.multiply %adlrs1b0db, %adrats1b0db : tensor<192xf32>
    %adsubs1b0db = stablehlo.subtract %s1b0db, %adsts1b0db : tensor<192xf32>
    %adwds1b0db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrs1b0db = stablehlo.multiply %adwds1b0db, %adlrs1b0db : tensor<192xf32>
    %adwdps1b0db = stablehlo.multiply %adwdlrs1b0db, %s1b0db : tensor<192xf32>
    %adnews1b0db = stablehlo.subtract %adsubs1b0db, %adwdps1b0db : tensor<192xf32>
    %adb1s1b0ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s1b0ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss1b0ng = stablehlo.multiply %adb1s1b0ng, %s1b0ngm : tensor<f32>
    %admgs1b0ng = stablehlo.multiply %adob1s1b0ng, %s1b0dndg : tensor<f32>
    %admns1b0ng = stablehlo.add %admss1b0ng, %admgs1b0ng : tensor<f32>
    %adb2s1b0ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s1b0ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss1b0ng = stablehlo.multiply %adb2s1b0ng, %s1b0ngv : tensor<f32>
    %adg2s1b0ng = stablehlo.multiply %s1b0dndg, %s1b0dndg : tensor<f32>
    %advgs1b0ng = stablehlo.multiply %adob2s1b0ng, %adg2s1b0ng : tensor<f32>
    %advns1b0ng = stablehlo.add %advss1b0ng, %advgs1b0ng : tensor<f32>
    %adbc1s1b0ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s1b0ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs1b0ng = stablehlo.divide %admns1b0ng, %adbc1s1b0ng : tensor<f32>
    %advhs1b0ng = stablehlo.divide %advns1b0ng, %adbc2s1b0ng : tensor<f32>
    %adlrs1b0ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss1b0ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs1b0ng = stablehlo.sqrt %advhs1b0ng : tensor<f32>
    %addens1b0ng = stablehlo.add %adsqs1b0ng, %adepss1b0ng : tensor<f32>
    %adrats1b0ng = stablehlo.divide %admhs1b0ng, %addens1b0ng : tensor<f32>
    %adsts1b0ng = stablehlo.multiply %adlrs1b0ng, %adrats1b0ng : tensor<f32>
    %adsubs1b0ng = stablehlo.subtract %s1b0ng, %adsts1b0ng : tensor<f32>
    %adwds1b0ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs1b0ng = stablehlo.multiply %adwds1b0ng, %adlrs1b0ng : tensor<f32>
    %adwdps1b0ng = stablehlo.multiply %adwdlrs1b0ng, %s1b0ng : tensor<f32>
    %adnews1b0ng = stablehlo.subtract %adsubs1b0ng, %adwdps1b0ng : tensor<f32>
    %adb1s1b0nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s1b0nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss1b0nbt = stablehlo.multiply %adb1s1b0nbt, %s1b0nbtm : tensor<f32>
    %admgs1b0nbt = stablehlo.multiply %adob1s1b0nbt, %s1b0dndb : tensor<f32>
    %admns1b0nbt = stablehlo.add %admss1b0nbt, %admgs1b0nbt : tensor<f32>
    %adb2s1b0nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s1b0nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss1b0nbt = stablehlo.multiply %adb2s1b0nbt, %s1b0nbtv : tensor<f32>
    %adg2s1b0nbt = stablehlo.multiply %s1b0dndb, %s1b0dndb : tensor<f32>
    %advgs1b0nbt = stablehlo.multiply %adob2s1b0nbt, %adg2s1b0nbt : tensor<f32>
    %advns1b0nbt = stablehlo.add %advss1b0nbt, %advgs1b0nbt : tensor<f32>
    %adbc1s1b0nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s1b0nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs1b0nbt = stablehlo.divide %admns1b0nbt, %adbc1s1b0nbt : tensor<f32>
    %advhs1b0nbt = stablehlo.divide %advns1b0nbt, %adbc2s1b0nbt : tensor<f32>
    %adlrs1b0nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss1b0nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs1b0nbt = stablehlo.sqrt %advhs1b0nbt : tensor<f32>
    %addens1b0nbt = stablehlo.add %adsqs1b0nbt, %adepss1b0nbt : tensor<f32>
    %adrats1b0nbt = stablehlo.divide %admhs1b0nbt, %addens1b0nbt : tensor<f32>
    %adsts1b0nbt = stablehlo.multiply %adlrs1b0nbt, %adrats1b0nbt : tensor<f32>
    %adsubs1b0nbt = stablehlo.subtract %s1b0nbt, %adsts1b0nbt : tensor<f32>
    %adwds1b0nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs1b0nbt = stablehlo.multiply %adwds1b0nbt, %adlrs1b0nbt : tensor<f32>
    %adwdps1b0nbt = stablehlo.multiply %adwdlrs1b0nbt, %s1b0nbt : tensor<f32>
    %adnews1b0nbt = stablehlo.subtract %adsubs1b0nbt, %adwdps1b0nbt : tensor<f32>
    %adb1s1b0eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adob1s1b0eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %admss1b0eW = stablehlo.multiply %adb1s1b0eW, %s1b0eWm : tensor<768x192x1x1xf32>
    %admgs1b0eW = stablehlo.multiply %adob1s1b0eW, %s1b0deW : tensor<768x192x1x1xf32>
    %admns1b0eW = stablehlo.add %admss1b0eW, %admgs1b0eW : tensor<768x192x1x1xf32>
    %adb2s1b0eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adob2s1b0eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %advss1b0eW = stablehlo.multiply %adb2s1b0eW, %s1b0eWv : tensor<768x192x1x1xf32>
    %adg2s1b0eW = stablehlo.multiply %s1b0deW, %s1b0deW : tensor<768x192x1x1xf32>
    %advgs1b0eW = stablehlo.multiply %adob2s1b0eW, %adg2s1b0eW : tensor<768x192x1x1xf32>
    %advns1b0eW = stablehlo.add %advss1b0eW, %advgs1b0eW : tensor<768x192x1x1xf32>
    %adbc1s1b0eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adbc2s1b0eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %admhs1b0eW = stablehlo.divide %admns1b0eW, %adbc1s1b0eW : tensor<768x192x1x1xf32>
    %advhs1b0eW = stablehlo.divide %advns1b0eW, %adbc2s1b0eW : tensor<768x192x1x1xf32>
    %adlrs1b0eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adepss1b0eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adsqs1b0eW = stablehlo.sqrt %advhs1b0eW : tensor<768x192x1x1xf32>
    %addens1b0eW = stablehlo.add %adsqs1b0eW, %adepss1b0eW : tensor<768x192x1x1xf32>
    %adrats1b0eW = stablehlo.divide %admhs1b0eW, %addens1b0eW : tensor<768x192x1x1xf32>
    %adsts1b0eW = stablehlo.multiply %adlrs1b0eW, %adrats1b0eW : tensor<768x192x1x1xf32>
    %adsubs1b0eW = stablehlo.subtract %s1b0eW, %adsts1b0eW : tensor<768x192x1x1xf32>
    %adwds1b0eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adwdlrs1b0eW = stablehlo.multiply %adwds1b0eW, %adlrs1b0eW : tensor<768x192x1x1xf32>
    %adwdps1b0eW = stablehlo.multiply %adwdlrs1b0eW, %s1b0eW : tensor<768x192x1x1xf32>
    %adnews1b0eW = stablehlo.subtract %adsubs1b0eW, %adwdps1b0eW : tensor<768x192x1x1xf32>
    %adb1s1b0eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1s1b0eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admss1b0eb = stablehlo.multiply %adb1s1b0eb, %s1b0ebm : tensor<768xf32>
    %admgs1b0eb = stablehlo.multiply %adob1s1b0eb, %s1b0deb : tensor<768xf32>
    %admns1b0eb = stablehlo.add %admss1b0eb, %admgs1b0eb : tensor<768xf32>
    %adb2s1b0eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2s1b0eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advss1b0eb = stablehlo.multiply %adb2s1b0eb, %s1b0ebv : tensor<768xf32>
    %adg2s1b0eb = stablehlo.multiply %s1b0deb, %s1b0deb : tensor<768xf32>
    %advgs1b0eb = stablehlo.multiply %adob2s1b0eb, %adg2s1b0eb : tensor<768xf32>
    %advns1b0eb = stablehlo.add %advss1b0eb, %advgs1b0eb : tensor<768xf32>
    %adbc1s1b0eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2s1b0eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhs1b0eb = stablehlo.divide %admns1b0eb, %adbc1s1b0eb : tensor<768xf32>
    %advhs1b0eb = stablehlo.divide %advns1b0eb, %adbc2s1b0eb : tensor<768xf32>
    %adlrs1b0eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepss1b0eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqs1b0eb = stablehlo.sqrt %advhs1b0eb : tensor<768xf32>
    %addens1b0eb = stablehlo.add %adsqs1b0eb, %adepss1b0eb : tensor<768xf32>
    %adrats1b0eb = stablehlo.divide %admhs1b0eb, %addens1b0eb : tensor<768xf32>
    %adsts1b0eb = stablehlo.multiply %adlrs1b0eb, %adrats1b0eb : tensor<768xf32>
    %adsubs1b0eb = stablehlo.subtract %s1b0eb, %adsts1b0eb : tensor<768xf32>
    %adwds1b0eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrs1b0eb = stablehlo.multiply %adwds1b0eb, %adlrs1b0eb : tensor<768xf32>
    %adwdps1b0eb = stablehlo.multiply %adwdlrs1b0eb, %s1b0eb : tensor<768xf32>
    %adnews1b0eb = stablehlo.subtract %adsubs1b0eb, %adwdps1b0eb : tensor<768xf32>
    %adb1s1b0pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adob1s1b0pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %admss1b0pW = stablehlo.multiply %adb1s1b0pW, %s1b0pWm : tensor<192x768x1x1xf32>
    %admgs1b0pW = stablehlo.multiply %adob1s1b0pW, %s1b0dpW : tensor<192x768x1x1xf32>
    %admns1b0pW = stablehlo.add %admss1b0pW, %admgs1b0pW : tensor<192x768x1x1xf32>
    %adb2s1b0pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adob2s1b0pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %advss1b0pW = stablehlo.multiply %adb2s1b0pW, %s1b0pWv : tensor<192x768x1x1xf32>
    %adg2s1b0pW = stablehlo.multiply %s1b0dpW, %s1b0dpW : tensor<192x768x1x1xf32>
    %advgs1b0pW = stablehlo.multiply %adob2s1b0pW, %adg2s1b0pW : tensor<192x768x1x1xf32>
    %advns1b0pW = stablehlo.add %advss1b0pW, %advgs1b0pW : tensor<192x768x1x1xf32>
    %adbc1s1b0pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adbc2s1b0pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %admhs1b0pW = stablehlo.divide %admns1b0pW, %adbc1s1b0pW : tensor<192x768x1x1xf32>
    %advhs1b0pW = stablehlo.divide %advns1b0pW, %adbc2s1b0pW : tensor<192x768x1x1xf32>
    %adlrs1b0pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adepss1b0pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adsqs1b0pW = stablehlo.sqrt %advhs1b0pW : tensor<192x768x1x1xf32>
    %addens1b0pW = stablehlo.add %adsqs1b0pW, %adepss1b0pW : tensor<192x768x1x1xf32>
    %adrats1b0pW = stablehlo.divide %admhs1b0pW, %addens1b0pW : tensor<192x768x1x1xf32>
    %adsts1b0pW = stablehlo.multiply %adlrs1b0pW, %adrats1b0pW : tensor<192x768x1x1xf32>
    %adsubs1b0pW = stablehlo.subtract %s1b0pW, %adsts1b0pW : tensor<192x768x1x1xf32>
    %adwds1b0pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adwdlrs1b0pW = stablehlo.multiply %adwds1b0pW, %adlrs1b0pW : tensor<192x768x1x1xf32>
    %adwdps1b0pW = stablehlo.multiply %adwdlrs1b0pW, %s1b0pW : tensor<192x768x1x1xf32>
    %adnews1b0pW = stablehlo.subtract %adsubs1b0pW, %adwdps1b0pW : tensor<192x768x1x1xf32>
    %adb1s1b0pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1s1b0pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admss1b0pb = stablehlo.multiply %adb1s1b0pb, %s1b0pbm : tensor<192xf32>
    %admgs1b0pb = stablehlo.multiply %adob1s1b0pb, %s1b0dpb : tensor<192xf32>
    %admns1b0pb = stablehlo.add %admss1b0pb, %admgs1b0pb : tensor<192xf32>
    %adb2s1b0pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2s1b0pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advss1b0pb = stablehlo.multiply %adb2s1b0pb, %s1b0pbv : tensor<192xf32>
    %adg2s1b0pb = stablehlo.multiply %s1b0dpb, %s1b0dpb : tensor<192xf32>
    %advgs1b0pb = stablehlo.multiply %adob2s1b0pb, %adg2s1b0pb : tensor<192xf32>
    %advns1b0pb = stablehlo.add %advss1b0pb, %advgs1b0pb : tensor<192xf32>
    %adbc1s1b0pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2s1b0pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhs1b0pb = stablehlo.divide %admns1b0pb, %adbc1s1b0pb : tensor<192xf32>
    %advhs1b0pb = stablehlo.divide %advns1b0pb, %adbc2s1b0pb : tensor<192xf32>
    %adlrs1b0pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepss1b0pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqs1b0pb = stablehlo.sqrt %advhs1b0pb : tensor<192xf32>
    %addens1b0pb = stablehlo.add %adsqs1b0pb, %adepss1b0pb : tensor<192xf32>
    %adrats1b0pb = stablehlo.divide %admhs1b0pb, %addens1b0pb : tensor<192xf32>
    %adsts1b0pb = stablehlo.multiply %adlrs1b0pb, %adrats1b0pb : tensor<192xf32>
    %adsubs1b0pb = stablehlo.subtract %s1b0pb, %adsts1b0pb : tensor<192xf32>
    %adwds1b0pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrs1b0pb = stablehlo.multiply %adwds1b0pb, %adlrs1b0pb : tensor<192xf32>
    %adwdps1b0pb = stablehlo.multiply %adwdlrs1b0pb, %s1b0pb : tensor<192xf32>
    %adnews1b0pb = stablehlo.subtract %adsubs1b0pb, %adwdps1b0pb : tensor<192xf32>
    %adb1s1b0lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1s1b0lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admss1b0lg = stablehlo.multiply %adb1s1b0lg, %s1b0lgm : tensor<192xf32>
    %admgs1b0lg = stablehlo.multiply %adob1s1b0lg, %s1b0dlsdg : tensor<192xf32>
    %admns1b0lg = stablehlo.add %admss1b0lg, %admgs1b0lg : tensor<192xf32>
    %adb2s1b0lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2s1b0lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advss1b0lg = stablehlo.multiply %adb2s1b0lg, %s1b0lgv : tensor<192xf32>
    %adg2s1b0lg = stablehlo.multiply %s1b0dlsdg, %s1b0dlsdg : tensor<192xf32>
    %advgs1b0lg = stablehlo.multiply %adob2s1b0lg, %adg2s1b0lg : tensor<192xf32>
    %advns1b0lg = stablehlo.add %advss1b0lg, %advgs1b0lg : tensor<192xf32>
    %adbc1s1b0lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2s1b0lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhs1b0lg = stablehlo.divide %admns1b0lg, %adbc1s1b0lg : tensor<192xf32>
    %advhs1b0lg = stablehlo.divide %advns1b0lg, %adbc2s1b0lg : tensor<192xf32>
    %adlrs1b0lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepss1b0lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqs1b0lg = stablehlo.sqrt %advhs1b0lg : tensor<192xf32>
    %addens1b0lg = stablehlo.add %adsqs1b0lg, %adepss1b0lg : tensor<192xf32>
    %adrats1b0lg = stablehlo.divide %admhs1b0lg, %addens1b0lg : tensor<192xf32>
    %adsts1b0lg = stablehlo.multiply %adlrs1b0lg, %adrats1b0lg : tensor<192xf32>
    %adsubs1b0lg = stablehlo.subtract %s1b0lg, %adsts1b0lg : tensor<192xf32>
    %adwds1b0lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrs1b0lg = stablehlo.multiply %adwds1b0lg, %adlrs1b0lg : tensor<192xf32>
    %adwdps1b0lg = stablehlo.multiply %adwdlrs1b0lg, %s1b0lg : tensor<192xf32>
    %adnews1b0lg = stablehlo.subtract %adsubs1b0lg, %adwdps1b0lg : tensor<192xf32>
    %adb1s1b1dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adob1s1b1dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %admss1b1dW = stablehlo.multiply %adb1s1b1dW, %s1b1dWm : tensor<192x1x7x7xf32>
    %admgs1b1dW = stablehlo.multiply %adob1s1b1dW, %s1b1ddW : tensor<192x1x7x7xf32>
    %admns1b1dW = stablehlo.add %admss1b1dW, %admgs1b1dW : tensor<192x1x7x7xf32>
    %adb2s1b1dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adob2s1b1dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %advss1b1dW = stablehlo.multiply %adb2s1b1dW, %s1b1dWv : tensor<192x1x7x7xf32>
    %adg2s1b1dW = stablehlo.multiply %s1b1ddW, %s1b1ddW : tensor<192x1x7x7xf32>
    %advgs1b1dW = stablehlo.multiply %adob2s1b1dW, %adg2s1b1dW : tensor<192x1x7x7xf32>
    %advns1b1dW = stablehlo.add %advss1b1dW, %advgs1b1dW : tensor<192x1x7x7xf32>
    %adbc1s1b1dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adbc2s1b1dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %admhs1b1dW = stablehlo.divide %admns1b1dW, %adbc1s1b1dW : tensor<192x1x7x7xf32>
    %advhs1b1dW = stablehlo.divide %advns1b1dW, %adbc2s1b1dW : tensor<192x1x7x7xf32>
    %adlrs1b1dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adepss1b1dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adsqs1b1dW = stablehlo.sqrt %advhs1b1dW : tensor<192x1x7x7xf32>
    %addens1b1dW = stablehlo.add %adsqs1b1dW, %adepss1b1dW : tensor<192x1x7x7xf32>
    %adrats1b1dW = stablehlo.divide %admhs1b1dW, %addens1b1dW : tensor<192x1x7x7xf32>
    %adsts1b1dW = stablehlo.multiply %adlrs1b1dW, %adrats1b1dW : tensor<192x1x7x7xf32>
    %adsubs1b1dW = stablehlo.subtract %s1b1dW, %adsts1b1dW : tensor<192x1x7x7xf32>
    %adwds1b1dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adwdlrs1b1dW = stablehlo.multiply %adwds1b1dW, %adlrs1b1dW : tensor<192x1x7x7xf32>
    %adwdps1b1dW = stablehlo.multiply %adwdlrs1b1dW, %s1b1dW : tensor<192x1x7x7xf32>
    %adnews1b1dW = stablehlo.subtract %adsubs1b1dW, %adwdps1b1dW : tensor<192x1x7x7xf32>
    %adb1s1b1db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1s1b1db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admss1b1db = stablehlo.multiply %adb1s1b1db, %s1b1dbm : tensor<192xf32>
    %admgs1b1db = stablehlo.multiply %adob1s1b1db, %s1b1ddb : tensor<192xf32>
    %admns1b1db = stablehlo.add %admss1b1db, %admgs1b1db : tensor<192xf32>
    %adb2s1b1db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2s1b1db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advss1b1db = stablehlo.multiply %adb2s1b1db, %s1b1dbv : tensor<192xf32>
    %adg2s1b1db = stablehlo.multiply %s1b1ddb, %s1b1ddb : tensor<192xf32>
    %advgs1b1db = stablehlo.multiply %adob2s1b1db, %adg2s1b1db : tensor<192xf32>
    %advns1b1db = stablehlo.add %advss1b1db, %advgs1b1db : tensor<192xf32>
    %adbc1s1b1db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2s1b1db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhs1b1db = stablehlo.divide %admns1b1db, %adbc1s1b1db : tensor<192xf32>
    %advhs1b1db = stablehlo.divide %advns1b1db, %adbc2s1b1db : tensor<192xf32>
    %adlrs1b1db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepss1b1db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqs1b1db = stablehlo.sqrt %advhs1b1db : tensor<192xf32>
    %addens1b1db = stablehlo.add %adsqs1b1db, %adepss1b1db : tensor<192xf32>
    %adrats1b1db = stablehlo.divide %admhs1b1db, %addens1b1db : tensor<192xf32>
    %adsts1b1db = stablehlo.multiply %adlrs1b1db, %adrats1b1db : tensor<192xf32>
    %adsubs1b1db = stablehlo.subtract %s1b1db, %adsts1b1db : tensor<192xf32>
    %adwds1b1db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrs1b1db = stablehlo.multiply %adwds1b1db, %adlrs1b1db : tensor<192xf32>
    %adwdps1b1db = stablehlo.multiply %adwdlrs1b1db, %s1b1db : tensor<192xf32>
    %adnews1b1db = stablehlo.subtract %adsubs1b1db, %adwdps1b1db : tensor<192xf32>
    %adb1s1b1ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s1b1ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss1b1ng = stablehlo.multiply %adb1s1b1ng, %s1b1ngm : tensor<f32>
    %admgs1b1ng = stablehlo.multiply %adob1s1b1ng, %s1b1dndg : tensor<f32>
    %admns1b1ng = stablehlo.add %admss1b1ng, %admgs1b1ng : tensor<f32>
    %adb2s1b1ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s1b1ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss1b1ng = stablehlo.multiply %adb2s1b1ng, %s1b1ngv : tensor<f32>
    %adg2s1b1ng = stablehlo.multiply %s1b1dndg, %s1b1dndg : tensor<f32>
    %advgs1b1ng = stablehlo.multiply %adob2s1b1ng, %adg2s1b1ng : tensor<f32>
    %advns1b1ng = stablehlo.add %advss1b1ng, %advgs1b1ng : tensor<f32>
    %adbc1s1b1ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s1b1ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs1b1ng = stablehlo.divide %admns1b1ng, %adbc1s1b1ng : tensor<f32>
    %advhs1b1ng = stablehlo.divide %advns1b1ng, %adbc2s1b1ng : tensor<f32>
    %adlrs1b1ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss1b1ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs1b1ng = stablehlo.sqrt %advhs1b1ng : tensor<f32>
    %addens1b1ng = stablehlo.add %adsqs1b1ng, %adepss1b1ng : tensor<f32>
    %adrats1b1ng = stablehlo.divide %admhs1b1ng, %addens1b1ng : tensor<f32>
    %adsts1b1ng = stablehlo.multiply %adlrs1b1ng, %adrats1b1ng : tensor<f32>
    %adsubs1b1ng = stablehlo.subtract %s1b1ng, %adsts1b1ng : tensor<f32>
    %adwds1b1ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs1b1ng = stablehlo.multiply %adwds1b1ng, %adlrs1b1ng : tensor<f32>
    %adwdps1b1ng = stablehlo.multiply %adwdlrs1b1ng, %s1b1ng : tensor<f32>
    %adnews1b1ng = stablehlo.subtract %adsubs1b1ng, %adwdps1b1ng : tensor<f32>
    %adb1s1b1nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s1b1nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss1b1nbt = stablehlo.multiply %adb1s1b1nbt, %s1b1nbtm : tensor<f32>
    %admgs1b1nbt = stablehlo.multiply %adob1s1b1nbt, %s1b1dndb : tensor<f32>
    %admns1b1nbt = stablehlo.add %admss1b1nbt, %admgs1b1nbt : tensor<f32>
    %adb2s1b1nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s1b1nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss1b1nbt = stablehlo.multiply %adb2s1b1nbt, %s1b1nbtv : tensor<f32>
    %adg2s1b1nbt = stablehlo.multiply %s1b1dndb, %s1b1dndb : tensor<f32>
    %advgs1b1nbt = stablehlo.multiply %adob2s1b1nbt, %adg2s1b1nbt : tensor<f32>
    %advns1b1nbt = stablehlo.add %advss1b1nbt, %advgs1b1nbt : tensor<f32>
    %adbc1s1b1nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s1b1nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs1b1nbt = stablehlo.divide %admns1b1nbt, %adbc1s1b1nbt : tensor<f32>
    %advhs1b1nbt = stablehlo.divide %advns1b1nbt, %adbc2s1b1nbt : tensor<f32>
    %adlrs1b1nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss1b1nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs1b1nbt = stablehlo.sqrt %advhs1b1nbt : tensor<f32>
    %addens1b1nbt = stablehlo.add %adsqs1b1nbt, %adepss1b1nbt : tensor<f32>
    %adrats1b1nbt = stablehlo.divide %admhs1b1nbt, %addens1b1nbt : tensor<f32>
    %adsts1b1nbt = stablehlo.multiply %adlrs1b1nbt, %adrats1b1nbt : tensor<f32>
    %adsubs1b1nbt = stablehlo.subtract %s1b1nbt, %adsts1b1nbt : tensor<f32>
    %adwds1b1nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs1b1nbt = stablehlo.multiply %adwds1b1nbt, %adlrs1b1nbt : tensor<f32>
    %adwdps1b1nbt = stablehlo.multiply %adwdlrs1b1nbt, %s1b1nbt : tensor<f32>
    %adnews1b1nbt = stablehlo.subtract %adsubs1b1nbt, %adwdps1b1nbt : tensor<f32>
    %adb1s1b1eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adob1s1b1eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %admss1b1eW = stablehlo.multiply %adb1s1b1eW, %s1b1eWm : tensor<768x192x1x1xf32>
    %admgs1b1eW = stablehlo.multiply %adob1s1b1eW, %s1b1deW : tensor<768x192x1x1xf32>
    %admns1b1eW = stablehlo.add %admss1b1eW, %admgs1b1eW : tensor<768x192x1x1xf32>
    %adb2s1b1eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adob2s1b1eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %advss1b1eW = stablehlo.multiply %adb2s1b1eW, %s1b1eWv : tensor<768x192x1x1xf32>
    %adg2s1b1eW = stablehlo.multiply %s1b1deW, %s1b1deW : tensor<768x192x1x1xf32>
    %advgs1b1eW = stablehlo.multiply %adob2s1b1eW, %adg2s1b1eW : tensor<768x192x1x1xf32>
    %advns1b1eW = stablehlo.add %advss1b1eW, %advgs1b1eW : tensor<768x192x1x1xf32>
    %adbc1s1b1eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adbc2s1b1eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %admhs1b1eW = stablehlo.divide %admns1b1eW, %adbc1s1b1eW : tensor<768x192x1x1xf32>
    %advhs1b1eW = stablehlo.divide %advns1b1eW, %adbc2s1b1eW : tensor<768x192x1x1xf32>
    %adlrs1b1eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adepss1b1eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adsqs1b1eW = stablehlo.sqrt %advhs1b1eW : tensor<768x192x1x1xf32>
    %addens1b1eW = stablehlo.add %adsqs1b1eW, %adepss1b1eW : tensor<768x192x1x1xf32>
    %adrats1b1eW = stablehlo.divide %admhs1b1eW, %addens1b1eW : tensor<768x192x1x1xf32>
    %adsts1b1eW = stablehlo.multiply %adlrs1b1eW, %adrats1b1eW : tensor<768x192x1x1xf32>
    %adsubs1b1eW = stablehlo.subtract %s1b1eW, %adsts1b1eW : tensor<768x192x1x1xf32>
    %adwds1b1eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adwdlrs1b1eW = stablehlo.multiply %adwds1b1eW, %adlrs1b1eW : tensor<768x192x1x1xf32>
    %adwdps1b1eW = stablehlo.multiply %adwdlrs1b1eW, %s1b1eW : tensor<768x192x1x1xf32>
    %adnews1b1eW = stablehlo.subtract %adsubs1b1eW, %adwdps1b1eW : tensor<768x192x1x1xf32>
    %adb1s1b1eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1s1b1eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admss1b1eb = stablehlo.multiply %adb1s1b1eb, %s1b1ebm : tensor<768xf32>
    %admgs1b1eb = stablehlo.multiply %adob1s1b1eb, %s1b1deb : tensor<768xf32>
    %admns1b1eb = stablehlo.add %admss1b1eb, %admgs1b1eb : tensor<768xf32>
    %adb2s1b1eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2s1b1eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advss1b1eb = stablehlo.multiply %adb2s1b1eb, %s1b1ebv : tensor<768xf32>
    %adg2s1b1eb = stablehlo.multiply %s1b1deb, %s1b1deb : tensor<768xf32>
    %advgs1b1eb = stablehlo.multiply %adob2s1b1eb, %adg2s1b1eb : tensor<768xf32>
    %advns1b1eb = stablehlo.add %advss1b1eb, %advgs1b1eb : tensor<768xf32>
    %adbc1s1b1eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2s1b1eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhs1b1eb = stablehlo.divide %admns1b1eb, %adbc1s1b1eb : tensor<768xf32>
    %advhs1b1eb = stablehlo.divide %advns1b1eb, %adbc2s1b1eb : tensor<768xf32>
    %adlrs1b1eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepss1b1eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqs1b1eb = stablehlo.sqrt %advhs1b1eb : tensor<768xf32>
    %addens1b1eb = stablehlo.add %adsqs1b1eb, %adepss1b1eb : tensor<768xf32>
    %adrats1b1eb = stablehlo.divide %admhs1b1eb, %addens1b1eb : tensor<768xf32>
    %adsts1b1eb = stablehlo.multiply %adlrs1b1eb, %adrats1b1eb : tensor<768xf32>
    %adsubs1b1eb = stablehlo.subtract %s1b1eb, %adsts1b1eb : tensor<768xf32>
    %adwds1b1eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrs1b1eb = stablehlo.multiply %adwds1b1eb, %adlrs1b1eb : tensor<768xf32>
    %adwdps1b1eb = stablehlo.multiply %adwdlrs1b1eb, %s1b1eb : tensor<768xf32>
    %adnews1b1eb = stablehlo.subtract %adsubs1b1eb, %adwdps1b1eb : tensor<768xf32>
    %adb1s1b1pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adob1s1b1pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %admss1b1pW = stablehlo.multiply %adb1s1b1pW, %s1b1pWm : tensor<192x768x1x1xf32>
    %admgs1b1pW = stablehlo.multiply %adob1s1b1pW, %s1b1dpW : tensor<192x768x1x1xf32>
    %admns1b1pW = stablehlo.add %admss1b1pW, %admgs1b1pW : tensor<192x768x1x1xf32>
    %adb2s1b1pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adob2s1b1pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %advss1b1pW = stablehlo.multiply %adb2s1b1pW, %s1b1pWv : tensor<192x768x1x1xf32>
    %adg2s1b1pW = stablehlo.multiply %s1b1dpW, %s1b1dpW : tensor<192x768x1x1xf32>
    %advgs1b1pW = stablehlo.multiply %adob2s1b1pW, %adg2s1b1pW : tensor<192x768x1x1xf32>
    %advns1b1pW = stablehlo.add %advss1b1pW, %advgs1b1pW : tensor<192x768x1x1xf32>
    %adbc1s1b1pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adbc2s1b1pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %admhs1b1pW = stablehlo.divide %admns1b1pW, %adbc1s1b1pW : tensor<192x768x1x1xf32>
    %advhs1b1pW = stablehlo.divide %advns1b1pW, %adbc2s1b1pW : tensor<192x768x1x1xf32>
    %adlrs1b1pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adepss1b1pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adsqs1b1pW = stablehlo.sqrt %advhs1b1pW : tensor<192x768x1x1xf32>
    %addens1b1pW = stablehlo.add %adsqs1b1pW, %adepss1b1pW : tensor<192x768x1x1xf32>
    %adrats1b1pW = stablehlo.divide %admhs1b1pW, %addens1b1pW : tensor<192x768x1x1xf32>
    %adsts1b1pW = stablehlo.multiply %adlrs1b1pW, %adrats1b1pW : tensor<192x768x1x1xf32>
    %adsubs1b1pW = stablehlo.subtract %s1b1pW, %adsts1b1pW : tensor<192x768x1x1xf32>
    %adwds1b1pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adwdlrs1b1pW = stablehlo.multiply %adwds1b1pW, %adlrs1b1pW : tensor<192x768x1x1xf32>
    %adwdps1b1pW = stablehlo.multiply %adwdlrs1b1pW, %s1b1pW : tensor<192x768x1x1xf32>
    %adnews1b1pW = stablehlo.subtract %adsubs1b1pW, %adwdps1b1pW : tensor<192x768x1x1xf32>
    %adb1s1b1pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1s1b1pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admss1b1pb = stablehlo.multiply %adb1s1b1pb, %s1b1pbm : tensor<192xf32>
    %admgs1b1pb = stablehlo.multiply %adob1s1b1pb, %s1b1dpb : tensor<192xf32>
    %admns1b1pb = stablehlo.add %admss1b1pb, %admgs1b1pb : tensor<192xf32>
    %adb2s1b1pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2s1b1pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advss1b1pb = stablehlo.multiply %adb2s1b1pb, %s1b1pbv : tensor<192xf32>
    %adg2s1b1pb = stablehlo.multiply %s1b1dpb, %s1b1dpb : tensor<192xf32>
    %advgs1b1pb = stablehlo.multiply %adob2s1b1pb, %adg2s1b1pb : tensor<192xf32>
    %advns1b1pb = stablehlo.add %advss1b1pb, %advgs1b1pb : tensor<192xf32>
    %adbc1s1b1pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2s1b1pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhs1b1pb = stablehlo.divide %admns1b1pb, %adbc1s1b1pb : tensor<192xf32>
    %advhs1b1pb = stablehlo.divide %advns1b1pb, %adbc2s1b1pb : tensor<192xf32>
    %adlrs1b1pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepss1b1pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqs1b1pb = stablehlo.sqrt %advhs1b1pb : tensor<192xf32>
    %addens1b1pb = stablehlo.add %adsqs1b1pb, %adepss1b1pb : tensor<192xf32>
    %adrats1b1pb = stablehlo.divide %admhs1b1pb, %addens1b1pb : tensor<192xf32>
    %adsts1b1pb = stablehlo.multiply %adlrs1b1pb, %adrats1b1pb : tensor<192xf32>
    %adsubs1b1pb = stablehlo.subtract %s1b1pb, %adsts1b1pb : tensor<192xf32>
    %adwds1b1pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrs1b1pb = stablehlo.multiply %adwds1b1pb, %adlrs1b1pb : tensor<192xf32>
    %adwdps1b1pb = stablehlo.multiply %adwdlrs1b1pb, %s1b1pb : tensor<192xf32>
    %adnews1b1pb = stablehlo.subtract %adsubs1b1pb, %adwdps1b1pb : tensor<192xf32>
    %adb1s1b1lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1s1b1lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admss1b1lg = stablehlo.multiply %adb1s1b1lg, %s1b1lgm : tensor<192xf32>
    %admgs1b1lg = stablehlo.multiply %adob1s1b1lg, %s1b1dlsdg : tensor<192xf32>
    %admns1b1lg = stablehlo.add %admss1b1lg, %admgs1b1lg : tensor<192xf32>
    %adb2s1b1lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2s1b1lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advss1b1lg = stablehlo.multiply %adb2s1b1lg, %s1b1lgv : tensor<192xf32>
    %adg2s1b1lg = stablehlo.multiply %s1b1dlsdg, %s1b1dlsdg : tensor<192xf32>
    %advgs1b1lg = stablehlo.multiply %adob2s1b1lg, %adg2s1b1lg : tensor<192xf32>
    %advns1b1lg = stablehlo.add %advss1b1lg, %advgs1b1lg : tensor<192xf32>
    %adbc1s1b1lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2s1b1lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhs1b1lg = stablehlo.divide %admns1b1lg, %adbc1s1b1lg : tensor<192xf32>
    %advhs1b1lg = stablehlo.divide %advns1b1lg, %adbc2s1b1lg : tensor<192xf32>
    %adlrs1b1lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepss1b1lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqs1b1lg = stablehlo.sqrt %advhs1b1lg : tensor<192xf32>
    %addens1b1lg = stablehlo.add %adsqs1b1lg, %adepss1b1lg : tensor<192xf32>
    %adrats1b1lg = stablehlo.divide %admhs1b1lg, %addens1b1lg : tensor<192xf32>
    %adsts1b1lg = stablehlo.multiply %adlrs1b1lg, %adrats1b1lg : tensor<192xf32>
    %adsubs1b1lg = stablehlo.subtract %s1b1lg, %adsts1b1lg : tensor<192xf32>
    %adwds1b1lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrs1b1lg = stablehlo.multiply %adwds1b1lg, %adlrs1b1lg : tensor<192xf32>
    %adwdps1b1lg = stablehlo.multiply %adwdlrs1b1lg, %s1b1lg : tensor<192xf32>
    %adnews1b1lg = stablehlo.subtract %adsubs1b1lg, %adwdps1b1lg : tensor<192xf32>
    %adb1s1b2dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adob1s1b2dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %admss1b2dW = stablehlo.multiply %adb1s1b2dW, %s1b2dWm : tensor<192x1x7x7xf32>
    %admgs1b2dW = stablehlo.multiply %adob1s1b2dW, %s1b2ddW : tensor<192x1x7x7xf32>
    %admns1b2dW = stablehlo.add %admss1b2dW, %admgs1b2dW : tensor<192x1x7x7xf32>
    %adb2s1b2dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adob2s1b2dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %advss1b2dW = stablehlo.multiply %adb2s1b2dW, %s1b2dWv : tensor<192x1x7x7xf32>
    %adg2s1b2dW = stablehlo.multiply %s1b2ddW, %s1b2ddW : tensor<192x1x7x7xf32>
    %advgs1b2dW = stablehlo.multiply %adob2s1b2dW, %adg2s1b2dW : tensor<192x1x7x7xf32>
    %advns1b2dW = stablehlo.add %advss1b2dW, %advgs1b2dW : tensor<192x1x7x7xf32>
    %adbc1s1b2dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adbc2s1b2dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %admhs1b2dW = stablehlo.divide %admns1b2dW, %adbc1s1b2dW : tensor<192x1x7x7xf32>
    %advhs1b2dW = stablehlo.divide %advns1b2dW, %adbc2s1b2dW : tensor<192x1x7x7xf32>
    %adlrs1b2dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adepss1b2dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adsqs1b2dW = stablehlo.sqrt %advhs1b2dW : tensor<192x1x7x7xf32>
    %addens1b2dW = stablehlo.add %adsqs1b2dW, %adepss1b2dW : tensor<192x1x7x7xf32>
    %adrats1b2dW = stablehlo.divide %admhs1b2dW, %addens1b2dW : tensor<192x1x7x7xf32>
    %adsts1b2dW = stablehlo.multiply %adlrs1b2dW, %adrats1b2dW : tensor<192x1x7x7xf32>
    %adsubs1b2dW = stablehlo.subtract %s1b2dW, %adsts1b2dW : tensor<192x1x7x7xf32>
    %adwds1b2dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %adwdlrs1b2dW = stablehlo.multiply %adwds1b2dW, %adlrs1b2dW : tensor<192x1x7x7xf32>
    %adwdps1b2dW = stablehlo.multiply %adwdlrs1b2dW, %s1b2dW : tensor<192x1x7x7xf32>
    %adnews1b2dW = stablehlo.subtract %adsubs1b2dW, %adwdps1b2dW : tensor<192x1x7x7xf32>
    %adb1s1b2db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1s1b2db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admss1b2db = stablehlo.multiply %adb1s1b2db, %s1b2dbm : tensor<192xf32>
    %admgs1b2db = stablehlo.multiply %adob1s1b2db, %s1b2ddb : tensor<192xf32>
    %admns1b2db = stablehlo.add %admss1b2db, %admgs1b2db : tensor<192xf32>
    %adb2s1b2db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2s1b2db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advss1b2db = stablehlo.multiply %adb2s1b2db, %s1b2dbv : tensor<192xf32>
    %adg2s1b2db = stablehlo.multiply %s1b2ddb, %s1b2ddb : tensor<192xf32>
    %advgs1b2db = stablehlo.multiply %adob2s1b2db, %adg2s1b2db : tensor<192xf32>
    %advns1b2db = stablehlo.add %advss1b2db, %advgs1b2db : tensor<192xf32>
    %adbc1s1b2db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2s1b2db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhs1b2db = stablehlo.divide %admns1b2db, %adbc1s1b2db : tensor<192xf32>
    %advhs1b2db = stablehlo.divide %advns1b2db, %adbc2s1b2db : tensor<192xf32>
    %adlrs1b2db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepss1b2db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqs1b2db = stablehlo.sqrt %advhs1b2db : tensor<192xf32>
    %addens1b2db = stablehlo.add %adsqs1b2db, %adepss1b2db : tensor<192xf32>
    %adrats1b2db = stablehlo.divide %admhs1b2db, %addens1b2db : tensor<192xf32>
    %adsts1b2db = stablehlo.multiply %adlrs1b2db, %adrats1b2db : tensor<192xf32>
    %adsubs1b2db = stablehlo.subtract %s1b2db, %adsts1b2db : tensor<192xf32>
    %adwds1b2db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrs1b2db = stablehlo.multiply %adwds1b2db, %adlrs1b2db : tensor<192xf32>
    %adwdps1b2db = stablehlo.multiply %adwdlrs1b2db, %s1b2db : tensor<192xf32>
    %adnews1b2db = stablehlo.subtract %adsubs1b2db, %adwdps1b2db : tensor<192xf32>
    %adb1s1b2ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s1b2ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss1b2ng = stablehlo.multiply %adb1s1b2ng, %s1b2ngm : tensor<f32>
    %admgs1b2ng = stablehlo.multiply %adob1s1b2ng, %s1b2dndg : tensor<f32>
    %admns1b2ng = stablehlo.add %admss1b2ng, %admgs1b2ng : tensor<f32>
    %adb2s1b2ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s1b2ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss1b2ng = stablehlo.multiply %adb2s1b2ng, %s1b2ngv : tensor<f32>
    %adg2s1b2ng = stablehlo.multiply %s1b2dndg, %s1b2dndg : tensor<f32>
    %advgs1b2ng = stablehlo.multiply %adob2s1b2ng, %adg2s1b2ng : tensor<f32>
    %advns1b2ng = stablehlo.add %advss1b2ng, %advgs1b2ng : tensor<f32>
    %adbc1s1b2ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s1b2ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs1b2ng = stablehlo.divide %admns1b2ng, %adbc1s1b2ng : tensor<f32>
    %advhs1b2ng = stablehlo.divide %advns1b2ng, %adbc2s1b2ng : tensor<f32>
    %adlrs1b2ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss1b2ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs1b2ng = stablehlo.sqrt %advhs1b2ng : tensor<f32>
    %addens1b2ng = stablehlo.add %adsqs1b2ng, %adepss1b2ng : tensor<f32>
    %adrats1b2ng = stablehlo.divide %admhs1b2ng, %addens1b2ng : tensor<f32>
    %adsts1b2ng = stablehlo.multiply %adlrs1b2ng, %adrats1b2ng : tensor<f32>
    %adsubs1b2ng = stablehlo.subtract %s1b2ng, %adsts1b2ng : tensor<f32>
    %adwds1b2ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs1b2ng = stablehlo.multiply %adwds1b2ng, %adlrs1b2ng : tensor<f32>
    %adwdps1b2ng = stablehlo.multiply %adwdlrs1b2ng, %s1b2ng : tensor<f32>
    %adnews1b2ng = stablehlo.subtract %adsubs1b2ng, %adwdps1b2ng : tensor<f32>
    %adb1s1b2nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s1b2nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss1b2nbt = stablehlo.multiply %adb1s1b2nbt, %s1b2nbtm : tensor<f32>
    %admgs1b2nbt = stablehlo.multiply %adob1s1b2nbt, %s1b2dndb : tensor<f32>
    %admns1b2nbt = stablehlo.add %admss1b2nbt, %admgs1b2nbt : tensor<f32>
    %adb2s1b2nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s1b2nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss1b2nbt = stablehlo.multiply %adb2s1b2nbt, %s1b2nbtv : tensor<f32>
    %adg2s1b2nbt = stablehlo.multiply %s1b2dndb, %s1b2dndb : tensor<f32>
    %advgs1b2nbt = stablehlo.multiply %adob2s1b2nbt, %adg2s1b2nbt : tensor<f32>
    %advns1b2nbt = stablehlo.add %advss1b2nbt, %advgs1b2nbt : tensor<f32>
    %adbc1s1b2nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s1b2nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs1b2nbt = stablehlo.divide %admns1b2nbt, %adbc1s1b2nbt : tensor<f32>
    %advhs1b2nbt = stablehlo.divide %advns1b2nbt, %adbc2s1b2nbt : tensor<f32>
    %adlrs1b2nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss1b2nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs1b2nbt = stablehlo.sqrt %advhs1b2nbt : tensor<f32>
    %addens1b2nbt = stablehlo.add %adsqs1b2nbt, %adepss1b2nbt : tensor<f32>
    %adrats1b2nbt = stablehlo.divide %admhs1b2nbt, %addens1b2nbt : tensor<f32>
    %adsts1b2nbt = stablehlo.multiply %adlrs1b2nbt, %adrats1b2nbt : tensor<f32>
    %adsubs1b2nbt = stablehlo.subtract %s1b2nbt, %adsts1b2nbt : tensor<f32>
    %adwds1b2nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs1b2nbt = stablehlo.multiply %adwds1b2nbt, %adlrs1b2nbt : tensor<f32>
    %adwdps1b2nbt = stablehlo.multiply %adwdlrs1b2nbt, %s1b2nbt : tensor<f32>
    %adnews1b2nbt = stablehlo.subtract %adsubs1b2nbt, %adwdps1b2nbt : tensor<f32>
    %adb1s1b2eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adob1s1b2eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %admss1b2eW = stablehlo.multiply %adb1s1b2eW, %s1b2eWm : tensor<768x192x1x1xf32>
    %admgs1b2eW = stablehlo.multiply %adob1s1b2eW, %s1b2deW : tensor<768x192x1x1xf32>
    %admns1b2eW = stablehlo.add %admss1b2eW, %admgs1b2eW : tensor<768x192x1x1xf32>
    %adb2s1b2eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adob2s1b2eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %advss1b2eW = stablehlo.multiply %adb2s1b2eW, %s1b2eWv : tensor<768x192x1x1xf32>
    %adg2s1b2eW = stablehlo.multiply %s1b2deW, %s1b2deW : tensor<768x192x1x1xf32>
    %advgs1b2eW = stablehlo.multiply %adob2s1b2eW, %adg2s1b2eW : tensor<768x192x1x1xf32>
    %advns1b2eW = stablehlo.add %advss1b2eW, %advgs1b2eW : tensor<768x192x1x1xf32>
    %adbc1s1b2eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adbc2s1b2eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %admhs1b2eW = stablehlo.divide %admns1b2eW, %adbc1s1b2eW : tensor<768x192x1x1xf32>
    %advhs1b2eW = stablehlo.divide %advns1b2eW, %adbc2s1b2eW : tensor<768x192x1x1xf32>
    %adlrs1b2eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adepss1b2eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adsqs1b2eW = stablehlo.sqrt %advhs1b2eW : tensor<768x192x1x1xf32>
    %addens1b2eW = stablehlo.add %adsqs1b2eW, %adepss1b2eW : tensor<768x192x1x1xf32>
    %adrats1b2eW = stablehlo.divide %admhs1b2eW, %addens1b2eW : tensor<768x192x1x1xf32>
    %adsts1b2eW = stablehlo.multiply %adlrs1b2eW, %adrats1b2eW : tensor<768x192x1x1xf32>
    %adsubs1b2eW = stablehlo.subtract %s1b2eW, %adsts1b2eW : tensor<768x192x1x1xf32>
    %adwds1b2eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %adwdlrs1b2eW = stablehlo.multiply %adwds1b2eW, %adlrs1b2eW : tensor<768x192x1x1xf32>
    %adwdps1b2eW = stablehlo.multiply %adwdlrs1b2eW, %s1b2eW : tensor<768x192x1x1xf32>
    %adnews1b2eW = stablehlo.subtract %adsubs1b2eW, %adwdps1b2eW : tensor<768x192x1x1xf32>
    %adb1s1b2eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1s1b2eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admss1b2eb = stablehlo.multiply %adb1s1b2eb, %s1b2ebm : tensor<768xf32>
    %admgs1b2eb = stablehlo.multiply %adob1s1b2eb, %s1b2deb : tensor<768xf32>
    %admns1b2eb = stablehlo.add %admss1b2eb, %admgs1b2eb : tensor<768xf32>
    %adb2s1b2eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2s1b2eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advss1b2eb = stablehlo.multiply %adb2s1b2eb, %s1b2ebv : tensor<768xf32>
    %adg2s1b2eb = stablehlo.multiply %s1b2deb, %s1b2deb : tensor<768xf32>
    %advgs1b2eb = stablehlo.multiply %adob2s1b2eb, %adg2s1b2eb : tensor<768xf32>
    %advns1b2eb = stablehlo.add %advss1b2eb, %advgs1b2eb : tensor<768xf32>
    %adbc1s1b2eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2s1b2eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhs1b2eb = stablehlo.divide %admns1b2eb, %adbc1s1b2eb : tensor<768xf32>
    %advhs1b2eb = stablehlo.divide %advns1b2eb, %adbc2s1b2eb : tensor<768xf32>
    %adlrs1b2eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepss1b2eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqs1b2eb = stablehlo.sqrt %advhs1b2eb : tensor<768xf32>
    %addens1b2eb = stablehlo.add %adsqs1b2eb, %adepss1b2eb : tensor<768xf32>
    %adrats1b2eb = stablehlo.divide %admhs1b2eb, %addens1b2eb : tensor<768xf32>
    %adsts1b2eb = stablehlo.multiply %adlrs1b2eb, %adrats1b2eb : tensor<768xf32>
    %adsubs1b2eb = stablehlo.subtract %s1b2eb, %adsts1b2eb : tensor<768xf32>
    %adwds1b2eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrs1b2eb = stablehlo.multiply %adwds1b2eb, %adlrs1b2eb : tensor<768xf32>
    %adwdps1b2eb = stablehlo.multiply %adwdlrs1b2eb, %s1b2eb : tensor<768xf32>
    %adnews1b2eb = stablehlo.subtract %adsubs1b2eb, %adwdps1b2eb : tensor<768xf32>
    %adb1s1b2pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adob1s1b2pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %admss1b2pW = stablehlo.multiply %adb1s1b2pW, %s1b2pWm : tensor<192x768x1x1xf32>
    %admgs1b2pW = stablehlo.multiply %adob1s1b2pW, %s1b2dpW : tensor<192x768x1x1xf32>
    %admns1b2pW = stablehlo.add %admss1b2pW, %admgs1b2pW : tensor<192x768x1x1xf32>
    %adb2s1b2pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adob2s1b2pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %advss1b2pW = stablehlo.multiply %adb2s1b2pW, %s1b2pWv : tensor<192x768x1x1xf32>
    %adg2s1b2pW = stablehlo.multiply %s1b2dpW, %s1b2dpW : tensor<192x768x1x1xf32>
    %advgs1b2pW = stablehlo.multiply %adob2s1b2pW, %adg2s1b2pW : tensor<192x768x1x1xf32>
    %advns1b2pW = stablehlo.add %advss1b2pW, %advgs1b2pW : tensor<192x768x1x1xf32>
    %adbc1s1b2pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adbc2s1b2pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %admhs1b2pW = stablehlo.divide %admns1b2pW, %adbc1s1b2pW : tensor<192x768x1x1xf32>
    %advhs1b2pW = stablehlo.divide %advns1b2pW, %adbc2s1b2pW : tensor<192x768x1x1xf32>
    %adlrs1b2pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adepss1b2pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adsqs1b2pW = stablehlo.sqrt %advhs1b2pW : tensor<192x768x1x1xf32>
    %addens1b2pW = stablehlo.add %adsqs1b2pW, %adepss1b2pW : tensor<192x768x1x1xf32>
    %adrats1b2pW = stablehlo.divide %admhs1b2pW, %addens1b2pW : tensor<192x768x1x1xf32>
    %adsts1b2pW = stablehlo.multiply %adlrs1b2pW, %adrats1b2pW : tensor<192x768x1x1xf32>
    %adsubs1b2pW = stablehlo.subtract %s1b2pW, %adsts1b2pW : tensor<192x768x1x1xf32>
    %adwds1b2pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %adwdlrs1b2pW = stablehlo.multiply %adwds1b2pW, %adlrs1b2pW : tensor<192x768x1x1xf32>
    %adwdps1b2pW = stablehlo.multiply %adwdlrs1b2pW, %s1b2pW : tensor<192x768x1x1xf32>
    %adnews1b2pW = stablehlo.subtract %adsubs1b2pW, %adwdps1b2pW : tensor<192x768x1x1xf32>
    %adb1s1b2pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1s1b2pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admss1b2pb = stablehlo.multiply %adb1s1b2pb, %s1b2pbm : tensor<192xf32>
    %admgs1b2pb = stablehlo.multiply %adob1s1b2pb, %s1b2dpb : tensor<192xf32>
    %admns1b2pb = stablehlo.add %admss1b2pb, %admgs1b2pb : tensor<192xf32>
    %adb2s1b2pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2s1b2pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advss1b2pb = stablehlo.multiply %adb2s1b2pb, %s1b2pbv : tensor<192xf32>
    %adg2s1b2pb = stablehlo.multiply %s1b2dpb, %s1b2dpb : tensor<192xf32>
    %advgs1b2pb = stablehlo.multiply %adob2s1b2pb, %adg2s1b2pb : tensor<192xf32>
    %advns1b2pb = stablehlo.add %advss1b2pb, %advgs1b2pb : tensor<192xf32>
    %adbc1s1b2pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2s1b2pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhs1b2pb = stablehlo.divide %admns1b2pb, %adbc1s1b2pb : tensor<192xf32>
    %advhs1b2pb = stablehlo.divide %advns1b2pb, %adbc2s1b2pb : tensor<192xf32>
    %adlrs1b2pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepss1b2pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqs1b2pb = stablehlo.sqrt %advhs1b2pb : tensor<192xf32>
    %addens1b2pb = stablehlo.add %adsqs1b2pb, %adepss1b2pb : tensor<192xf32>
    %adrats1b2pb = stablehlo.divide %admhs1b2pb, %addens1b2pb : tensor<192xf32>
    %adsts1b2pb = stablehlo.multiply %adlrs1b2pb, %adrats1b2pb : tensor<192xf32>
    %adsubs1b2pb = stablehlo.subtract %s1b2pb, %adsts1b2pb : tensor<192xf32>
    %adwds1b2pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrs1b2pb = stablehlo.multiply %adwds1b2pb, %adlrs1b2pb : tensor<192xf32>
    %adwdps1b2pb = stablehlo.multiply %adwdlrs1b2pb, %s1b2pb : tensor<192xf32>
    %adnews1b2pb = stablehlo.subtract %adsubs1b2pb, %adwdps1b2pb : tensor<192xf32>
    %adb1s1b2lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob1s1b2lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admss1b2lg = stablehlo.multiply %adb1s1b2lg, %s1b2lgm : tensor<192xf32>
    %admgs1b2lg = stablehlo.multiply %adob1s1b2lg, %s1b2dlsdg : tensor<192xf32>
    %admns1b2lg = stablehlo.add %admss1b2lg, %admgs1b2lg : tensor<192xf32>
    %adb2s1b2lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adob2s1b2lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %advss1b2lg = stablehlo.multiply %adb2s1b2lg, %s1b2lgv : tensor<192xf32>
    %adg2s1b2lg = stablehlo.multiply %s1b2dlsdg, %s1b2dlsdg : tensor<192xf32>
    %advgs1b2lg = stablehlo.multiply %adob2s1b2lg, %adg2s1b2lg : tensor<192xf32>
    %advns1b2lg = stablehlo.add %advss1b2lg, %advgs1b2lg : tensor<192xf32>
    %adbc1s1b2lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adbc2s1b2lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %admhs1b2lg = stablehlo.divide %admns1b2lg, %adbc1s1b2lg : tensor<192xf32>
    %advhs1b2lg = stablehlo.divide %advns1b2lg, %adbc2s1b2lg : tensor<192xf32>
    %adlrs1b2lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adepss1b2lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adsqs1b2lg = stablehlo.sqrt %advhs1b2lg : tensor<192xf32>
    %addens1b2lg = stablehlo.add %adsqs1b2lg, %adepss1b2lg : tensor<192xf32>
    %adrats1b2lg = stablehlo.divide %admhs1b2lg, %addens1b2lg : tensor<192xf32>
    %adsts1b2lg = stablehlo.multiply %adlrs1b2lg, %adrats1b2lg : tensor<192xf32>
    %adsubs1b2lg = stablehlo.subtract %s1b2lg, %adsts1b2lg : tensor<192xf32>
    %adwds1b2lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %adwdlrs1b2lg = stablehlo.multiply %adwds1b2lg, %adlrs1b2lg : tensor<192xf32>
    %adwdps1b2lg = stablehlo.multiply %adwdlrs1b2lg, %s1b2lg : tensor<192xf32>
    %adnews1b2lg = stablehlo.subtract %adsubs1b2lg, %adwdps1b2lg : tensor<192xf32>
    %adb1d1ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1d1ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admsd1ng = stablehlo.multiply %adb1d1ng, %d1ngm : tensor<f32>
    %admgd1ng = stablehlo.multiply %adob1d1ng, %d1dndg : tensor<f32>
    %admnd1ng = stablehlo.add %admsd1ng, %admgd1ng : tensor<f32>
    %adb2d1ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2d1ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advsd1ng = stablehlo.multiply %adb2d1ng, %d1ngv : tensor<f32>
    %adg2d1ng = stablehlo.multiply %d1dndg, %d1dndg : tensor<f32>
    %advgd1ng = stablehlo.multiply %adob2d1ng, %adg2d1ng : tensor<f32>
    %advnd1ng = stablehlo.add %advsd1ng, %advgd1ng : tensor<f32>
    %adbc1d1ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2d1ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhd1ng = stablehlo.divide %admnd1ng, %adbc1d1ng : tensor<f32>
    %advhd1ng = stablehlo.divide %advnd1ng, %adbc2d1ng : tensor<f32>
    %adlrd1ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepsd1ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqd1ng = stablehlo.sqrt %advhd1ng : tensor<f32>
    %addend1ng = stablehlo.add %adsqd1ng, %adepsd1ng : tensor<f32>
    %adratd1ng = stablehlo.divide %admhd1ng, %addend1ng : tensor<f32>
    %adstd1ng = stablehlo.multiply %adlrd1ng, %adratd1ng : tensor<f32>
    %adsubd1ng = stablehlo.subtract %d1ng, %adstd1ng : tensor<f32>
    %adwdd1ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrd1ng = stablehlo.multiply %adwdd1ng, %adlrd1ng : tensor<f32>
    %adwdpd1ng = stablehlo.multiply %adwdlrd1ng, %d1ng : tensor<f32>
    %adnewd1ng = stablehlo.subtract %adsubd1ng, %adwdpd1ng : tensor<f32>
    %adb1d1nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1d1nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admsd1nbt = stablehlo.multiply %adb1d1nbt, %d1nbtm : tensor<f32>
    %admgd1nbt = stablehlo.multiply %adob1d1nbt, %d1dndb : tensor<f32>
    %admnd1nbt = stablehlo.add %admsd1nbt, %admgd1nbt : tensor<f32>
    %adb2d1nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2d1nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advsd1nbt = stablehlo.multiply %adb2d1nbt, %d1nbtv : tensor<f32>
    %adg2d1nbt = stablehlo.multiply %d1dndb, %d1dndb : tensor<f32>
    %advgd1nbt = stablehlo.multiply %adob2d1nbt, %adg2d1nbt : tensor<f32>
    %advnd1nbt = stablehlo.add %advsd1nbt, %advgd1nbt : tensor<f32>
    %adbc1d1nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2d1nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhd1nbt = stablehlo.divide %admnd1nbt, %adbc1d1nbt : tensor<f32>
    %advhd1nbt = stablehlo.divide %advnd1nbt, %adbc2d1nbt : tensor<f32>
    %adlrd1nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepsd1nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqd1nbt = stablehlo.sqrt %advhd1nbt : tensor<f32>
    %addend1nbt = stablehlo.add %adsqd1nbt, %adepsd1nbt : tensor<f32>
    %adratd1nbt = stablehlo.divide %admhd1nbt, %addend1nbt : tensor<f32>
    %adstd1nbt = stablehlo.multiply %adlrd1nbt, %adratd1nbt : tensor<f32>
    %adsubd1nbt = stablehlo.subtract %d1nbt, %adstd1nbt : tensor<f32>
    %adwdd1nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrd1nbt = stablehlo.multiply %adwdd1nbt, %adlrd1nbt : tensor<f32>
    %adwdpd1nbt = stablehlo.multiply %adwdlrd1nbt, %d1nbt : tensor<f32>
    %adnewd1nbt = stablehlo.subtract %adsubd1nbt, %adwdpd1nbt : tensor<f32>
    %adb1d1W = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %adob1d1W = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %admsd1W = stablehlo.multiply %adb1d1W, %d1Wm : tensor<384x192x2x2xf32>
    %admgd1W = stablehlo.multiply %adob1d1W, %d1dW : tensor<384x192x2x2xf32>
    %admnd1W = stablehlo.add %admsd1W, %admgd1W : tensor<384x192x2x2xf32>
    %adb2d1W = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %adob2d1W = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %advsd1W = stablehlo.multiply %adb2d1W, %d1Wv : tensor<384x192x2x2xf32>
    %adg2d1W = stablehlo.multiply %d1dW, %d1dW : tensor<384x192x2x2xf32>
    %advgd1W = stablehlo.multiply %adob2d1W, %adg2d1W : tensor<384x192x2x2xf32>
    %advnd1W = stablehlo.add %advsd1W, %advgd1W : tensor<384x192x2x2xf32>
    %adbc1d1W = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %adbc2d1W = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %admhd1W = stablehlo.divide %admnd1W, %adbc1d1W : tensor<384x192x2x2xf32>
    %advhd1W = stablehlo.divide %advnd1W, %adbc2d1W : tensor<384x192x2x2xf32>
    %adlrd1W = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %adepsd1W = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %adsqd1W = stablehlo.sqrt %advhd1W : tensor<384x192x2x2xf32>
    %addend1W = stablehlo.add %adsqd1W, %adepsd1W : tensor<384x192x2x2xf32>
    %adratd1W = stablehlo.divide %admhd1W, %addend1W : tensor<384x192x2x2xf32>
    %adstd1W = stablehlo.multiply %adlrd1W, %adratd1W : tensor<384x192x2x2xf32>
    %adsubd1W = stablehlo.subtract %d1W, %adstd1W : tensor<384x192x2x2xf32>
    %adwdd1W = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %adwdlrd1W = stablehlo.multiply %adwdd1W, %adlrd1W : tensor<384x192x2x2xf32>
    %adwdpd1W = stablehlo.multiply %adwdlrd1W, %d1W : tensor<384x192x2x2xf32>
    %adnewd1W = stablehlo.subtract %adsubd1W, %adwdpd1W : tensor<384x192x2x2xf32>
    %adb1d1b = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1d1b = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admsd1b = stablehlo.multiply %adb1d1b, %d1bm : tensor<384xf32>
    %admgd1b = stablehlo.multiply %adob1d1b, %d1db : tensor<384xf32>
    %admnd1b = stablehlo.add %admsd1b, %admgd1b : tensor<384xf32>
    %adb2d1b = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2d1b = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advsd1b = stablehlo.multiply %adb2d1b, %d1bv : tensor<384xf32>
    %adg2d1b = stablehlo.multiply %d1db, %d1db : tensor<384xf32>
    %advgd1b = stablehlo.multiply %adob2d1b, %adg2d1b : tensor<384xf32>
    %advnd1b = stablehlo.add %advsd1b, %advgd1b : tensor<384xf32>
    %adbc1d1b = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2d1b = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhd1b = stablehlo.divide %admnd1b, %adbc1d1b : tensor<384xf32>
    %advhd1b = stablehlo.divide %advnd1b, %adbc2d1b : tensor<384xf32>
    %adlrd1b = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepsd1b = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqd1b = stablehlo.sqrt %advhd1b : tensor<384xf32>
    %addend1b = stablehlo.add %adsqd1b, %adepsd1b : tensor<384xf32>
    %adratd1b = stablehlo.divide %admhd1b, %addend1b : tensor<384xf32>
    %adstd1b = stablehlo.multiply %adlrd1b, %adratd1b : tensor<384xf32>
    %adsubd1b = stablehlo.subtract %d1b, %adstd1b : tensor<384xf32>
    %adwdd1b = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrd1b = stablehlo.multiply %adwdd1b, %adlrd1b : tensor<384xf32>
    %adwdpd1b = stablehlo.multiply %adwdlrd1b, %d1b : tensor<384xf32>
    %adnewd1b = stablehlo.subtract %adsubd1b, %adwdpd1b : tensor<384xf32>
    %adb1s2b0dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob1s2b0dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admss2b0dW = stablehlo.multiply %adb1s2b0dW, %s2b0dWm : tensor<384x1x7x7xf32>
    %admgs2b0dW = stablehlo.multiply %adob1s2b0dW, %s2b0ddW : tensor<384x1x7x7xf32>
    %admns2b0dW = stablehlo.add %admss2b0dW, %admgs2b0dW : tensor<384x1x7x7xf32>
    %adb2s2b0dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob2s2b0dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %advss2b0dW = stablehlo.multiply %adb2s2b0dW, %s2b0dWv : tensor<384x1x7x7xf32>
    %adg2s2b0dW = stablehlo.multiply %s2b0ddW, %s2b0ddW : tensor<384x1x7x7xf32>
    %advgs2b0dW = stablehlo.multiply %adob2s2b0dW, %adg2s2b0dW : tensor<384x1x7x7xf32>
    %advns2b0dW = stablehlo.add %advss2b0dW, %advgs2b0dW : tensor<384x1x7x7xf32>
    %adbc1s2b0dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adbc2s2b0dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admhs2b0dW = stablehlo.divide %admns2b0dW, %adbc1s2b0dW : tensor<384x1x7x7xf32>
    %advhs2b0dW = stablehlo.divide %advns2b0dW, %adbc2s2b0dW : tensor<384x1x7x7xf32>
    %adlrs2b0dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adepss2b0dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adsqs2b0dW = stablehlo.sqrt %advhs2b0dW : tensor<384x1x7x7xf32>
    %addens2b0dW = stablehlo.add %adsqs2b0dW, %adepss2b0dW : tensor<384x1x7x7xf32>
    %adrats2b0dW = stablehlo.divide %admhs2b0dW, %addens2b0dW : tensor<384x1x7x7xf32>
    %adsts2b0dW = stablehlo.multiply %adlrs2b0dW, %adrats2b0dW : tensor<384x1x7x7xf32>
    %adsubs2b0dW = stablehlo.subtract %s2b0dW, %adsts2b0dW : tensor<384x1x7x7xf32>
    %adwds2b0dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adwdlrs2b0dW = stablehlo.multiply %adwds2b0dW, %adlrs2b0dW : tensor<384x1x7x7xf32>
    %adwdps2b0dW = stablehlo.multiply %adwdlrs2b0dW, %s2b0dW : tensor<384x1x7x7xf32>
    %adnews2b0dW = stablehlo.subtract %adsubs2b0dW, %adwdps2b0dW : tensor<384x1x7x7xf32>
    %adb1s2b0db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b0db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b0db = stablehlo.multiply %adb1s2b0db, %s2b0dbm : tensor<384xf32>
    %admgs2b0db = stablehlo.multiply %adob1s2b0db, %s2b0ddb : tensor<384xf32>
    %admns2b0db = stablehlo.add %admss2b0db, %admgs2b0db : tensor<384xf32>
    %adb2s2b0db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b0db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b0db = stablehlo.multiply %adb2s2b0db, %s2b0dbv : tensor<384xf32>
    %adg2s2b0db = stablehlo.multiply %s2b0ddb, %s2b0ddb : tensor<384xf32>
    %advgs2b0db = stablehlo.multiply %adob2s2b0db, %adg2s2b0db : tensor<384xf32>
    %advns2b0db = stablehlo.add %advss2b0db, %advgs2b0db : tensor<384xf32>
    %adbc1s2b0db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b0db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b0db = stablehlo.divide %admns2b0db, %adbc1s2b0db : tensor<384xf32>
    %advhs2b0db = stablehlo.divide %advns2b0db, %adbc2s2b0db : tensor<384xf32>
    %adlrs2b0db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b0db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b0db = stablehlo.sqrt %advhs2b0db : tensor<384xf32>
    %addens2b0db = stablehlo.add %adsqs2b0db, %adepss2b0db : tensor<384xf32>
    %adrats2b0db = stablehlo.divide %admhs2b0db, %addens2b0db : tensor<384xf32>
    %adsts2b0db = stablehlo.multiply %adlrs2b0db, %adrats2b0db : tensor<384xf32>
    %adsubs2b0db = stablehlo.subtract %s2b0db, %adsts2b0db : tensor<384xf32>
    %adwds2b0db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b0db = stablehlo.multiply %adwds2b0db, %adlrs2b0db : tensor<384xf32>
    %adwdps2b0db = stablehlo.multiply %adwdlrs2b0db, %s2b0db : tensor<384xf32>
    %adnews2b0db = stablehlo.subtract %adsubs2b0db, %adwdps2b0db : tensor<384xf32>
    %adb1s2b0ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b0ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b0ng = stablehlo.multiply %adb1s2b0ng, %s2b0ngm : tensor<f32>
    %admgs2b0ng = stablehlo.multiply %adob1s2b0ng, %s2b0dndg : tensor<f32>
    %admns2b0ng = stablehlo.add %admss2b0ng, %admgs2b0ng : tensor<f32>
    %adb2s2b0ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b0ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b0ng = stablehlo.multiply %adb2s2b0ng, %s2b0ngv : tensor<f32>
    %adg2s2b0ng = stablehlo.multiply %s2b0dndg, %s2b0dndg : tensor<f32>
    %advgs2b0ng = stablehlo.multiply %adob2s2b0ng, %adg2s2b0ng : tensor<f32>
    %advns2b0ng = stablehlo.add %advss2b0ng, %advgs2b0ng : tensor<f32>
    %adbc1s2b0ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b0ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b0ng = stablehlo.divide %admns2b0ng, %adbc1s2b0ng : tensor<f32>
    %advhs2b0ng = stablehlo.divide %advns2b0ng, %adbc2s2b0ng : tensor<f32>
    %adlrs2b0ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b0ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b0ng = stablehlo.sqrt %advhs2b0ng : tensor<f32>
    %addens2b0ng = stablehlo.add %adsqs2b0ng, %adepss2b0ng : tensor<f32>
    %adrats2b0ng = stablehlo.divide %admhs2b0ng, %addens2b0ng : tensor<f32>
    %adsts2b0ng = stablehlo.multiply %adlrs2b0ng, %adrats2b0ng : tensor<f32>
    %adsubs2b0ng = stablehlo.subtract %s2b0ng, %adsts2b0ng : tensor<f32>
    %adwds2b0ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b0ng = stablehlo.multiply %adwds2b0ng, %adlrs2b0ng : tensor<f32>
    %adwdps2b0ng = stablehlo.multiply %adwdlrs2b0ng, %s2b0ng : tensor<f32>
    %adnews2b0ng = stablehlo.subtract %adsubs2b0ng, %adwdps2b0ng : tensor<f32>
    %adb1s2b0nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b0nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b0nbt = stablehlo.multiply %adb1s2b0nbt, %s2b0nbtm : tensor<f32>
    %admgs2b0nbt = stablehlo.multiply %adob1s2b0nbt, %s2b0dndb : tensor<f32>
    %admns2b0nbt = stablehlo.add %admss2b0nbt, %admgs2b0nbt : tensor<f32>
    %adb2s2b0nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b0nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b0nbt = stablehlo.multiply %adb2s2b0nbt, %s2b0nbtv : tensor<f32>
    %adg2s2b0nbt = stablehlo.multiply %s2b0dndb, %s2b0dndb : tensor<f32>
    %advgs2b0nbt = stablehlo.multiply %adob2s2b0nbt, %adg2s2b0nbt : tensor<f32>
    %advns2b0nbt = stablehlo.add %advss2b0nbt, %advgs2b0nbt : tensor<f32>
    %adbc1s2b0nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b0nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b0nbt = stablehlo.divide %admns2b0nbt, %adbc1s2b0nbt : tensor<f32>
    %advhs2b0nbt = stablehlo.divide %advns2b0nbt, %adbc2s2b0nbt : tensor<f32>
    %adlrs2b0nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b0nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b0nbt = stablehlo.sqrt %advhs2b0nbt : tensor<f32>
    %addens2b0nbt = stablehlo.add %adsqs2b0nbt, %adepss2b0nbt : tensor<f32>
    %adrats2b0nbt = stablehlo.divide %admhs2b0nbt, %addens2b0nbt : tensor<f32>
    %adsts2b0nbt = stablehlo.multiply %adlrs2b0nbt, %adrats2b0nbt : tensor<f32>
    %adsubs2b0nbt = stablehlo.subtract %s2b0nbt, %adsts2b0nbt : tensor<f32>
    %adwds2b0nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b0nbt = stablehlo.multiply %adwds2b0nbt, %adlrs2b0nbt : tensor<f32>
    %adwdps2b0nbt = stablehlo.multiply %adwdlrs2b0nbt, %s2b0nbt : tensor<f32>
    %adnews2b0nbt = stablehlo.subtract %adsubs2b0nbt, %adwdps2b0nbt : tensor<f32>
    %adb1s2b0eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob1s2b0eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admss2b0eW = stablehlo.multiply %adb1s2b0eW, %s2b0eWm : tensor<1536x384x1x1xf32>
    %admgs2b0eW = stablehlo.multiply %adob1s2b0eW, %s2b0deW : tensor<1536x384x1x1xf32>
    %admns2b0eW = stablehlo.add %admss2b0eW, %admgs2b0eW : tensor<1536x384x1x1xf32>
    %adb2s2b0eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob2s2b0eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %advss2b0eW = stablehlo.multiply %adb2s2b0eW, %s2b0eWv : tensor<1536x384x1x1xf32>
    %adg2s2b0eW = stablehlo.multiply %s2b0deW, %s2b0deW : tensor<1536x384x1x1xf32>
    %advgs2b0eW = stablehlo.multiply %adob2s2b0eW, %adg2s2b0eW : tensor<1536x384x1x1xf32>
    %advns2b0eW = stablehlo.add %advss2b0eW, %advgs2b0eW : tensor<1536x384x1x1xf32>
    %adbc1s2b0eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adbc2s2b0eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admhs2b0eW = stablehlo.divide %admns2b0eW, %adbc1s2b0eW : tensor<1536x384x1x1xf32>
    %advhs2b0eW = stablehlo.divide %advns2b0eW, %adbc2s2b0eW : tensor<1536x384x1x1xf32>
    %adlrs2b0eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adepss2b0eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adsqs2b0eW = stablehlo.sqrt %advhs2b0eW : tensor<1536x384x1x1xf32>
    %addens2b0eW = stablehlo.add %adsqs2b0eW, %adepss2b0eW : tensor<1536x384x1x1xf32>
    %adrats2b0eW = stablehlo.divide %admhs2b0eW, %addens2b0eW : tensor<1536x384x1x1xf32>
    %adsts2b0eW = stablehlo.multiply %adlrs2b0eW, %adrats2b0eW : tensor<1536x384x1x1xf32>
    %adsubs2b0eW = stablehlo.subtract %s2b0eW, %adsts2b0eW : tensor<1536x384x1x1xf32>
    %adwds2b0eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adwdlrs2b0eW = stablehlo.multiply %adwds2b0eW, %adlrs2b0eW : tensor<1536x384x1x1xf32>
    %adwdps2b0eW = stablehlo.multiply %adwdlrs2b0eW, %s2b0eW : tensor<1536x384x1x1xf32>
    %adnews2b0eW = stablehlo.subtract %adsubs2b0eW, %adwdps2b0eW : tensor<1536x384x1x1xf32>
    %adb1s2b0eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob1s2b0eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admss2b0eb = stablehlo.multiply %adb1s2b0eb, %s2b0ebm : tensor<1536xf32>
    %admgs2b0eb = stablehlo.multiply %adob1s2b0eb, %s2b0deb : tensor<1536xf32>
    %admns2b0eb = stablehlo.add %admss2b0eb, %admgs2b0eb : tensor<1536xf32>
    %adb2s2b0eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob2s2b0eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %advss2b0eb = stablehlo.multiply %adb2s2b0eb, %s2b0ebv : tensor<1536xf32>
    %adg2s2b0eb = stablehlo.multiply %s2b0deb, %s2b0deb : tensor<1536xf32>
    %advgs2b0eb = stablehlo.multiply %adob2s2b0eb, %adg2s2b0eb : tensor<1536xf32>
    %advns2b0eb = stablehlo.add %advss2b0eb, %advgs2b0eb : tensor<1536xf32>
    %adbc1s2b0eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adbc2s2b0eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admhs2b0eb = stablehlo.divide %admns2b0eb, %adbc1s2b0eb : tensor<1536xf32>
    %advhs2b0eb = stablehlo.divide %advns2b0eb, %adbc2s2b0eb : tensor<1536xf32>
    %adlrs2b0eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adepss2b0eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adsqs2b0eb = stablehlo.sqrt %advhs2b0eb : tensor<1536xf32>
    %addens2b0eb = stablehlo.add %adsqs2b0eb, %adepss2b0eb : tensor<1536xf32>
    %adrats2b0eb = stablehlo.divide %admhs2b0eb, %addens2b0eb : tensor<1536xf32>
    %adsts2b0eb = stablehlo.multiply %adlrs2b0eb, %adrats2b0eb : tensor<1536xf32>
    %adsubs2b0eb = stablehlo.subtract %s2b0eb, %adsts2b0eb : tensor<1536xf32>
    %adwds2b0eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adwdlrs2b0eb = stablehlo.multiply %adwds2b0eb, %adlrs2b0eb : tensor<1536xf32>
    %adwdps2b0eb = stablehlo.multiply %adwdlrs2b0eb, %s2b0eb : tensor<1536xf32>
    %adnews2b0eb = stablehlo.subtract %adsubs2b0eb, %adwdps2b0eb : tensor<1536xf32>
    %adb1s2b0pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob1s2b0pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admss2b0pW = stablehlo.multiply %adb1s2b0pW, %s2b0pWm : tensor<384x1536x1x1xf32>
    %admgs2b0pW = stablehlo.multiply %adob1s2b0pW, %s2b0dpW : tensor<384x1536x1x1xf32>
    %admns2b0pW = stablehlo.add %admss2b0pW, %admgs2b0pW : tensor<384x1536x1x1xf32>
    %adb2s2b0pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob2s2b0pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %advss2b0pW = stablehlo.multiply %adb2s2b0pW, %s2b0pWv : tensor<384x1536x1x1xf32>
    %adg2s2b0pW = stablehlo.multiply %s2b0dpW, %s2b0dpW : tensor<384x1536x1x1xf32>
    %advgs2b0pW = stablehlo.multiply %adob2s2b0pW, %adg2s2b0pW : tensor<384x1536x1x1xf32>
    %advns2b0pW = stablehlo.add %advss2b0pW, %advgs2b0pW : tensor<384x1536x1x1xf32>
    %adbc1s2b0pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adbc2s2b0pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admhs2b0pW = stablehlo.divide %admns2b0pW, %adbc1s2b0pW : tensor<384x1536x1x1xf32>
    %advhs2b0pW = stablehlo.divide %advns2b0pW, %adbc2s2b0pW : tensor<384x1536x1x1xf32>
    %adlrs2b0pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adepss2b0pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adsqs2b0pW = stablehlo.sqrt %advhs2b0pW : tensor<384x1536x1x1xf32>
    %addens2b0pW = stablehlo.add %adsqs2b0pW, %adepss2b0pW : tensor<384x1536x1x1xf32>
    %adrats2b0pW = stablehlo.divide %admhs2b0pW, %addens2b0pW : tensor<384x1536x1x1xf32>
    %adsts2b0pW = stablehlo.multiply %adlrs2b0pW, %adrats2b0pW : tensor<384x1536x1x1xf32>
    %adsubs2b0pW = stablehlo.subtract %s2b0pW, %adsts2b0pW : tensor<384x1536x1x1xf32>
    %adwds2b0pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adwdlrs2b0pW = stablehlo.multiply %adwds2b0pW, %adlrs2b0pW : tensor<384x1536x1x1xf32>
    %adwdps2b0pW = stablehlo.multiply %adwdlrs2b0pW, %s2b0pW : tensor<384x1536x1x1xf32>
    %adnews2b0pW = stablehlo.subtract %adsubs2b0pW, %adwdps2b0pW : tensor<384x1536x1x1xf32>
    %adb1s2b0pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b0pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b0pb = stablehlo.multiply %adb1s2b0pb, %s2b0pbm : tensor<384xf32>
    %admgs2b0pb = stablehlo.multiply %adob1s2b0pb, %s2b0dpb : tensor<384xf32>
    %admns2b0pb = stablehlo.add %admss2b0pb, %admgs2b0pb : tensor<384xf32>
    %adb2s2b0pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b0pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b0pb = stablehlo.multiply %adb2s2b0pb, %s2b0pbv : tensor<384xf32>
    %adg2s2b0pb = stablehlo.multiply %s2b0dpb, %s2b0dpb : tensor<384xf32>
    %advgs2b0pb = stablehlo.multiply %adob2s2b0pb, %adg2s2b0pb : tensor<384xf32>
    %advns2b0pb = stablehlo.add %advss2b0pb, %advgs2b0pb : tensor<384xf32>
    %adbc1s2b0pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b0pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b0pb = stablehlo.divide %admns2b0pb, %adbc1s2b0pb : tensor<384xf32>
    %advhs2b0pb = stablehlo.divide %advns2b0pb, %adbc2s2b0pb : tensor<384xf32>
    %adlrs2b0pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b0pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b0pb = stablehlo.sqrt %advhs2b0pb : tensor<384xf32>
    %addens2b0pb = stablehlo.add %adsqs2b0pb, %adepss2b0pb : tensor<384xf32>
    %adrats2b0pb = stablehlo.divide %admhs2b0pb, %addens2b0pb : tensor<384xf32>
    %adsts2b0pb = stablehlo.multiply %adlrs2b0pb, %adrats2b0pb : tensor<384xf32>
    %adsubs2b0pb = stablehlo.subtract %s2b0pb, %adsts2b0pb : tensor<384xf32>
    %adwds2b0pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b0pb = stablehlo.multiply %adwds2b0pb, %adlrs2b0pb : tensor<384xf32>
    %adwdps2b0pb = stablehlo.multiply %adwdlrs2b0pb, %s2b0pb : tensor<384xf32>
    %adnews2b0pb = stablehlo.subtract %adsubs2b0pb, %adwdps2b0pb : tensor<384xf32>
    %adb1s2b0lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b0lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b0lg = stablehlo.multiply %adb1s2b0lg, %s2b0lgm : tensor<384xf32>
    %admgs2b0lg = stablehlo.multiply %adob1s2b0lg, %s2b0dlsdg : tensor<384xf32>
    %admns2b0lg = stablehlo.add %admss2b0lg, %admgs2b0lg : tensor<384xf32>
    %adb2s2b0lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b0lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b0lg = stablehlo.multiply %adb2s2b0lg, %s2b0lgv : tensor<384xf32>
    %adg2s2b0lg = stablehlo.multiply %s2b0dlsdg, %s2b0dlsdg : tensor<384xf32>
    %advgs2b0lg = stablehlo.multiply %adob2s2b0lg, %adg2s2b0lg : tensor<384xf32>
    %advns2b0lg = stablehlo.add %advss2b0lg, %advgs2b0lg : tensor<384xf32>
    %adbc1s2b0lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b0lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b0lg = stablehlo.divide %admns2b0lg, %adbc1s2b0lg : tensor<384xf32>
    %advhs2b0lg = stablehlo.divide %advns2b0lg, %adbc2s2b0lg : tensor<384xf32>
    %adlrs2b0lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b0lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b0lg = stablehlo.sqrt %advhs2b0lg : tensor<384xf32>
    %addens2b0lg = stablehlo.add %adsqs2b0lg, %adepss2b0lg : tensor<384xf32>
    %adrats2b0lg = stablehlo.divide %admhs2b0lg, %addens2b0lg : tensor<384xf32>
    %adsts2b0lg = stablehlo.multiply %adlrs2b0lg, %adrats2b0lg : tensor<384xf32>
    %adsubs2b0lg = stablehlo.subtract %s2b0lg, %adsts2b0lg : tensor<384xf32>
    %adwds2b0lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b0lg = stablehlo.multiply %adwds2b0lg, %adlrs2b0lg : tensor<384xf32>
    %adwdps2b0lg = stablehlo.multiply %adwdlrs2b0lg, %s2b0lg : tensor<384xf32>
    %adnews2b0lg = stablehlo.subtract %adsubs2b0lg, %adwdps2b0lg : tensor<384xf32>
    %adb1s2b1dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob1s2b1dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admss2b1dW = stablehlo.multiply %adb1s2b1dW, %s2b1dWm : tensor<384x1x7x7xf32>
    %admgs2b1dW = stablehlo.multiply %adob1s2b1dW, %s2b1ddW : tensor<384x1x7x7xf32>
    %admns2b1dW = stablehlo.add %admss2b1dW, %admgs2b1dW : tensor<384x1x7x7xf32>
    %adb2s2b1dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob2s2b1dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %advss2b1dW = stablehlo.multiply %adb2s2b1dW, %s2b1dWv : tensor<384x1x7x7xf32>
    %adg2s2b1dW = stablehlo.multiply %s2b1ddW, %s2b1ddW : tensor<384x1x7x7xf32>
    %advgs2b1dW = stablehlo.multiply %adob2s2b1dW, %adg2s2b1dW : tensor<384x1x7x7xf32>
    %advns2b1dW = stablehlo.add %advss2b1dW, %advgs2b1dW : tensor<384x1x7x7xf32>
    %adbc1s2b1dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adbc2s2b1dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admhs2b1dW = stablehlo.divide %admns2b1dW, %adbc1s2b1dW : tensor<384x1x7x7xf32>
    %advhs2b1dW = stablehlo.divide %advns2b1dW, %adbc2s2b1dW : tensor<384x1x7x7xf32>
    %adlrs2b1dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adepss2b1dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adsqs2b1dW = stablehlo.sqrt %advhs2b1dW : tensor<384x1x7x7xf32>
    %addens2b1dW = stablehlo.add %adsqs2b1dW, %adepss2b1dW : tensor<384x1x7x7xf32>
    %adrats2b1dW = stablehlo.divide %admhs2b1dW, %addens2b1dW : tensor<384x1x7x7xf32>
    %adsts2b1dW = stablehlo.multiply %adlrs2b1dW, %adrats2b1dW : tensor<384x1x7x7xf32>
    %adsubs2b1dW = stablehlo.subtract %s2b1dW, %adsts2b1dW : tensor<384x1x7x7xf32>
    %adwds2b1dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adwdlrs2b1dW = stablehlo.multiply %adwds2b1dW, %adlrs2b1dW : tensor<384x1x7x7xf32>
    %adwdps2b1dW = stablehlo.multiply %adwdlrs2b1dW, %s2b1dW : tensor<384x1x7x7xf32>
    %adnews2b1dW = stablehlo.subtract %adsubs2b1dW, %adwdps2b1dW : tensor<384x1x7x7xf32>
    %adb1s2b1db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b1db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b1db = stablehlo.multiply %adb1s2b1db, %s2b1dbm : tensor<384xf32>
    %admgs2b1db = stablehlo.multiply %adob1s2b1db, %s2b1ddb : tensor<384xf32>
    %admns2b1db = stablehlo.add %admss2b1db, %admgs2b1db : tensor<384xf32>
    %adb2s2b1db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b1db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b1db = stablehlo.multiply %adb2s2b1db, %s2b1dbv : tensor<384xf32>
    %adg2s2b1db = stablehlo.multiply %s2b1ddb, %s2b1ddb : tensor<384xf32>
    %advgs2b1db = stablehlo.multiply %adob2s2b1db, %adg2s2b1db : tensor<384xf32>
    %advns2b1db = stablehlo.add %advss2b1db, %advgs2b1db : tensor<384xf32>
    %adbc1s2b1db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b1db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b1db = stablehlo.divide %admns2b1db, %adbc1s2b1db : tensor<384xf32>
    %advhs2b1db = stablehlo.divide %advns2b1db, %adbc2s2b1db : tensor<384xf32>
    %adlrs2b1db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b1db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b1db = stablehlo.sqrt %advhs2b1db : tensor<384xf32>
    %addens2b1db = stablehlo.add %adsqs2b1db, %adepss2b1db : tensor<384xf32>
    %adrats2b1db = stablehlo.divide %admhs2b1db, %addens2b1db : tensor<384xf32>
    %adsts2b1db = stablehlo.multiply %adlrs2b1db, %adrats2b1db : tensor<384xf32>
    %adsubs2b1db = stablehlo.subtract %s2b1db, %adsts2b1db : tensor<384xf32>
    %adwds2b1db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b1db = stablehlo.multiply %adwds2b1db, %adlrs2b1db : tensor<384xf32>
    %adwdps2b1db = stablehlo.multiply %adwdlrs2b1db, %s2b1db : tensor<384xf32>
    %adnews2b1db = stablehlo.subtract %adsubs2b1db, %adwdps2b1db : tensor<384xf32>
    %adb1s2b1ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b1ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b1ng = stablehlo.multiply %adb1s2b1ng, %s2b1ngm : tensor<f32>
    %admgs2b1ng = stablehlo.multiply %adob1s2b1ng, %s2b1dndg : tensor<f32>
    %admns2b1ng = stablehlo.add %admss2b1ng, %admgs2b1ng : tensor<f32>
    %adb2s2b1ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b1ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b1ng = stablehlo.multiply %adb2s2b1ng, %s2b1ngv : tensor<f32>
    %adg2s2b1ng = stablehlo.multiply %s2b1dndg, %s2b1dndg : tensor<f32>
    %advgs2b1ng = stablehlo.multiply %adob2s2b1ng, %adg2s2b1ng : tensor<f32>
    %advns2b1ng = stablehlo.add %advss2b1ng, %advgs2b1ng : tensor<f32>
    %adbc1s2b1ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b1ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b1ng = stablehlo.divide %admns2b1ng, %adbc1s2b1ng : tensor<f32>
    %advhs2b1ng = stablehlo.divide %advns2b1ng, %adbc2s2b1ng : tensor<f32>
    %adlrs2b1ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b1ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b1ng = stablehlo.sqrt %advhs2b1ng : tensor<f32>
    %addens2b1ng = stablehlo.add %adsqs2b1ng, %adepss2b1ng : tensor<f32>
    %adrats2b1ng = stablehlo.divide %admhs2b1ng, %addens2b1ng : tensor<f32>
    %adsts2b1ng = stablehlo.multiply %adlrs2b1ng, %adrats2b1ng : tensor<f32>
    %adsubs2b1ng = stablehlo.subtract %s2b1ng, %adsts2b1ng : tensor<f32>
    %adwds2b1ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b1ng = stablehlo.multiply %adwds2b1ng, %adlrs2b1ng : tensor<f32>
    %adwdps2b1ng = stablehlo.multiply %adwdlrs2b1ng, %s2b1ng : tensor<f32>
    %adnews2b1ng = stablehlo.subtract %adsubs2b1ng, %adwdps2b1ng : tensor<f32>
    %adb1s2b1nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b1nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b1nbt = stablehlo.multiply %adb1s2b1nbt, %s2b1nbtm : tensor<f32>
    %admgs2b1nbt = stablehlo.multiply %adob1s2b1nbt, %s2b1dndb : tensor<f32>
    %admns2b1nbt = stablehlo.add %admss2b1nbt, %admgs2b1nbt : tensor<f32>
    %adb2s2b1nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b1nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b1nbt = stablehlo.multiply %adb2s2b1nbt, %s2b1nbtv : tensor<f32>
    %adg2s2b1nbt = stablehlo.multiply %s2b1dndb, %s2b1dndb : tensor<f32>
    %advgs2b1nbt = stablehlo.multiply %adob2s2b1nbt, %adg2s2b1nbt : tensor<f32>
    %advns2b1nbt = stablehlo.add %advss2b1nbt, %advgs2b1nbt : tensor<f32>
    %adbc1s2b1nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b1nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b1nbt = stablehlo.divide %admns2b1nbt, %adbc1s2b1nbt : tensor<f32>
    %advhs2b1nbt = stablehlo.divide %advns2b1nbt, %adbc2s2b1nbt : tensor<f32>
    %adlrs2b1nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b1nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b1nbt = stablehlo.sqrt %advhs2b1nbt : tensor<f32>
    %addens2b1nbt = stablehlo.add %adsqs2b1nbt, %adepss2b1nbt : tensor<f32>
    %adrats2b1nbt = stablehlo.divide %admhs2b1nbt, %addens2b1nbt : tensor<f32>
    %adsts2b1nbt = stablehlo.multiply %adlrs2b1nbt, %adrats2b1nbt : tensor<f32>
    %adsubs2b1nbt = stablehlo.subtract %s2b1nbt, %adsts2b1nbt : tensor<f32>
    %adwds2b1nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b1nbt = stablehlo.multiply %adwds2b1nbt, %adlrs2b1nbt : tensor<f32>
    %adwdps2b1nbt = stablehlo.multiply %adwdlrs2b1nbt, %s2b1nbt : tensor<f32>
    %adnews2b1nbt = stablehlo.subtract %adsubs2b1nbt, %adwdps2b1nbt : tensor<f32>
    %adb1s2b1eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob1s2b1eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admss2b1eW = stablehlo.multiply %adb1s2b1eW, %s2b1eWm : tensor<1536x384x1x1xf32>
    %admgs2b1eW = stablehlo.multiply %adob1s2b1eW, %s2b1deW : tensor<1536x384x1x1xf32>
    %admns2b1eW = stablehlo.add %admss2b1eW, %admgs2b1eW : tensor<1536x384x1x1xf32>
    %adb2s2b1eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob2s2b1eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %advss2b1eW = stablehlo.multiply %adb2s2b1eW, %s2b1eWv : tensor<1536x384x1x1xf32>
    %adg2s2b1eW = stablehlo.multiply %s2b1deW, %s2b1deW : tensor<1536x384x1x1xf32>
    %advgs2b1eW = stablehlo.multiply %adob2s2b1eW, %adg2s2b1eW : tensor<1536x384x1x1xf32>
    %advns2b1eW = stablehlo.add %advss2b1eW, %advgs2b1eW : tensor<1536x384x1x1xf32>
    %adbc1s2b1eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adbc2s2b1eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admhs2b1eW = stablehlo.divide %admns2b1eW, %adbc1s2b1eW : tensor<1536x384x1x1xf32>
    %advhs2b1eW = stablehlo.divide %advns2b1eW, %adbc2s2b1eW : tensor<1536x384x1x1xf32>
    %adlrs2b1eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adepss2b1eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adsqs2b1eW = stablehlo.sqrt %advhs2b1eW : tensor<1536x384x1x1xf32>
    %addens2b1eW = stablehlo.add %adsqs2b1eW, %adepss2b1eW : tensor<1536x384x1x1xf32>
    %adrats2b1eW = stablehlo.divide %admhs2b1eW, %addens2b1eW : tensor<1536x384x1x1xf32>
    %adsts2b1eW = stablehlo.multiply %adlrs2b1eW, %adrats2b1eW : tensor<1536x384x1x1xf32>
    %adsubs2b1eW = stablehlo.subtract %s2b1eW, %adsts2b1eW : tensor<1536x384x1x1xf32>
    %adwds2b1eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adwdlrs2b1eW = stablehlo.multiply %adwds2b1eW, %adlrs2b1eW : tensor<1536x384x1x1xf32>
    %adwdps2b1eW = stablehlo.multiply %adwdlrs2b1eW, %s2b1eW : tensor<1536x384x1x1xf32>
    %adnews2b1eW = stablehlo.subtract %adsubs2b1eW, %adwdps2b1eW : tensor<1536x384x1x1xf32>
    %adb1s2b1eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob1s2b1eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admss2b1eb = stablehlo.multiply %adb1s2b1eb, %s2b1ebm : tensor<1536xf32>
    %admgs2b1eb = stablehlo.multiply %adob1s2b1eb, %s2b1deb : tensor<1536xf32>
    %admns2b1eb = stablehlo.add %admss2b1eb, %admgs2b1eb : tensor<1536xf32>
    %adb2s2b1eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob2s2b1eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %advss2b1eb = stablehlo.multiply %adb2s2b1eb, %s2b1ebv : tensor<1536xf32>
    %adg2s2b1eb = stablehlo.multiply %s2b1deb, %s2b1deb : tensor<1536xf32>
    %advgs2b1eb = stablehlo.multiply %adob2s2b1eb, %adg2s2b1eb : tensor<1536xf32>
    %advns2b1eb = stablehlo.add %advss2b1eb, %advgs2b1eb : tensor<1536xf32>
    %adbc1s2b1eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adbc2s2b1eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admhs2b1eb = stablehlo.divide %admns2b1eb, %adbc1s2b1eb : tensor<1536xf32>
    %advhs2b1eb = stablehlo.divide %advns2b1eb, %adbc2s2b1eb : tensor<1536xf32>
    %adlrs2b1eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adepss2b1eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adsqs2b1eb = stablehlo.sqrt %advhs2b1eb : tensor<1536xf32>
    %addens2b1eb = stablehlo.add %adsqs2b1eb, %adepss2b1eb : tensor<1536xf32>
    %adrats2b1eb = stablehlo.divide %admhs2b1eb, %addens2b1eb : tensor<1536xf32>
    %adsts2b1eb = stablehlo.multiply %adlrs2b1eb, %adrats2b1eb : tensor<1536xf32>
    %adsubs2b1eb = stablehlo.subtract %s2b1eb, %adsts2b1eb : tensor<1536xf32>
    %adwds2b1eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adwdlrs2b1eb = stablehlo.multiply %adwds2b1eb, %adlrs2b1eb : tensor<1536xf32>
    %adwdps2b1eb = stablehlo.multiply %adwdlrs2b1eb, %s2b1eb : tensor<1536xf32>
    %adnews2b1eb = stablehlo.subtract %adsubs2b1eb, %adwdps2b1eb : tensor<1536xf32>
    %adb1s2b1pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob1s2b1pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admss2b1pW = stablehlo.multiply %adb1s2b1pW, %s2b1pWm : tensor<384x1536x1x1xf32>
    %admgs2b1pW = stablehlo.multiply %adob1s2b1pW, %s2b1dpW : tensor<384x1536x1x1xf32>
    %admns2b1pW = stablehlo.add %admss2b1pW, %admgs2b1pW : tensor<384x1536x1x1xf32>
    %adb2s2b1pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob2s2b1pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %advss2b1pW = stablehlo.multiply %adb2s2b1pW, %s2b1pWv : tensor<384x1536x1x1xf32>
    %adg2s2b1pW = stablehlo.multiply %s2b1dpW, %s2b1dpW : tensor<384x1536x1x1xf32>
    %advgs2b1pW = stablehlo.multiply %adob2s2b1pW, %adg2s2b1pW : tensor<384x1536x1x1xf32>
    %advns2b1pW = stablehlo.add %advss2b1pW, %advgs2b1pW : tensor<384x1536x1x1xf32>
    %adbc1s2b1pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adbc2s2b1pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admhs2b1pW = stablehlo.divide %admns2b1pW, %adbc1s2b1pW : tensor<384x1536x1x1xf32>
    %advhs2b1pW = stablehlo.divide %advns2b1pW, %adbc2s2b1pW : tensor<384x1536x1x1xf32>
    %adlrs2b1pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adepss2b1pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adsqs2b1pW = stablehlo.sqrt %advhs2b1pW : tensor<384x1536x1x1xf32>
    %addens2b1pW = stablehlo.add %adsqs2b1pW, %adepss2b1pW : tensor<384x1536x1x1xf32>
    %adrats2b1pW = stablehlo.divide %admhs2b1pW, %addens2b1pW : tensor<384x1536x1x1xf32>
    %adsts2b1pW = stablehlo.multiply %adlrs2b1pW, %adrats2b1pW : tensor<384x1536x1x1xf32>
    %adsubs2b1pW = stablehlo.subtract %s2b1pW, %adsts2b1pW : tensor<384x1536x1x1xf32>
    %adwds2b1pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adwdlrs2b1pW = stablehlo.multiply %adwds2b1pW, %adlrs2b1pW : tensor<384x1536x1x1xf32>
    %adwdps2b1pW = stablehlo.multiply %adwdlrs2b1pW, %s2b1pW : tensor<384x1536x1x1xf32>
    %adnews2b1pW = stablehlo.subtract %adsubs2b1pW, %adwdps2b1pW : tensor<384x1536x1x1xf32>
    %adb1s2b1pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b1pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b1pb = stablehlo.multiply %adb1s2b1pb, %s2b1pbm : tensor<384xf32>
    %admgs2b1pb = stablehlo.multiply %adob1s2b1pb, %s2b1dpb : tensor<384xf32>
    %admns2b1pb = stablehlo.add %admss2b1pb, %admgs2b1pb : tensor<384xf32>
    %adb2s2b1pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b1pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b1pb = stablehlo.multiply %adb2s2b1pb, %s2b1pbv : tensor<384xf32>
    %adg2s2b1pb = stablehlo.multiply %s2b1dpb, %s2b1dpb : tensor<384xf32>
    %advgs2b1pb = stablehlo.multiply %adob2s2b1pb, %adg2s2b1pb : tensor<384xf32>
    %advns2b1pb = stablehlo.add %advss2b1pb, %advgs2b1pb : tensor<384xf32>
    %adbc1s2b1pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b1pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b1pb = stablehlo.divide %admns2b1pb, %adbc1s2b1pb : tensor<384xf32>
    %advhs2b1pb = stablehlo.divide %advns2b1pb, %adbc2s2b1pb : tensor<384xf32>
    %adlrs2b1pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b1pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b1pb = stablehlo.sqrt %advhs2b1pb : tensor<384xf32>
    %addens2b1pb = stablehlo.add %adsqs2b1pb, %adepss2b1pb : tensor<384xf32>
    %adrats2b1pb = stablehlo.divide %admhs2b1pb, %addens2b1pb : tensor<384xf32>
    %adsts2b1pb = stablehlo.multiply %adlrs2b1pb, %adrats2b1pb : tensor<384xf32>
    %adsubs2b1pb = stablehlo.subtract %s2b1pb, %adsts2b1pb : tensor<384xf32>
    %adwds2b1pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b1pb = stablehlo.multiply %adwds2b1pb, %adlrs2b1pb : tensor<384xf32>
    %adwdps2b1pb = stablehlo.multiply %adwdlrs2b1pb, %s2b1pb : tensor<384xf32>
    %adnews2b1pb = stablehlo.subtract %adsubs2b1pb, %adwdps2b1pb : tensor<384xf32>
    %adb1s2b1lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b1lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b1lg = stablehlo.multiply %adb1s2b1lg, %s2b1lgm : tensor<384xf32>
    %admgs2b1lg = stablehlo.multiply %adob1s2b1lg, %s2b1dlsdg : tensor<384xf32>
    %admns2b1lg = stablehlo.add %admss2b1lg, %admgs2b1lg : tensor<384xf32>
    %adb2s2b1lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b1lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b1lg = stablehlo.multiply %adb2s2b1lg, %s2b1lgv : tensor<384xf32>
    %adg2s2b1lg = stablehlo.multiply %s2b1dlsdg, %s2b1dlsdg : tensor<384xf32>
    %advgs2b1lg = stablehlo.multiply %adob2s2b1lg, %adg2s2b1lg : tensor<384xf32>
    %advns2b1lg = stablehlo.add %advss2b1lg, %advgs2b1lg : tensor<384xf32>
    %adbc1s2b1lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b1lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b1lg = stablehlo.divide %admns2b1lg, %adbc1s2b1lg : tensor<384xf32>
    %advhs2b1lg = stablehlo.divide %advns2b1lg, %adbc2s2b1lg : tensor<384xf32>
    %adlrs2b1lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b1lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b1lg = stablehlo.sqrt %advhs2b1lg : tensor<384xf32>
    %addens2b1lg = stablehlo.add %adsqs2b1lg, %adepss2b1lg : tensor<384xf32>
    %adrats2b1lg = stablehlo.divide %admhs2b1lg, %addens2b1lg : tensor<384xf32>
    %adsts2b1lg = stablehlo.multiply %adlrs2b1lg, %adrats2b1lg : tensor<384xf32>
    %adsubs2b1lg = stablehlo.subtract %s2b1lg, %adsts2b1lg : tensor<384xf32>
    %adwds2b1lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b1lg = stablehlo.multiply %adwds2b1lg, %adlrs2b1lg : tensor<384xf32>
    %adwdps2b1lg = stablehlo.multiply %adwdlrs2b1lg, %s2b1lg : tensor<384xf32>
    %adnews2b1lg = stablehlo.subtract %adsubs2b1lg, %adwdps2b1lg : tensor<384xf32>
    %adb1s2b2dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob1s2b2dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admss2b2dW = stablehlo.multiply %adb1s2b2dW, %s2b2dWm : tensor<384x1x7x7xf32>
    %admgs2b2dW = stablehlo.multiply %adob1s2b2dW, %s2b2ddW : tensor<384x1x7x7xf32>
    %admns2b2dW = stablehlo.add %admss2b2dW, %admgs2b2dW : tensor<384x1x7x7xf32>
    %adb2s2b2dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob2s2b2dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %advss2b2dW = stablehlo.multiply %adb2s2b2dW, %s2b2dWv : tensor<384x1x7x7xf32>
    %adg2s2b2dW = stablehlo.multiply %s2b2ddW, %s2b2ddW : tensor<384x1x7x7xf32>
    %advgs2b2dW = stablehlo.multiply %adob2s2b2dW, %adg2s2b2dW : tensor<384x1x7x7xf32>
    %advns2b2dW = stablehlo.add %advss2b2dW, %advgs2b2dW : tensor<384x1x7x7xf32>
    %adbc1s2b2dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adbc2s2b2dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admhs2b2dW = stablehlo.divide %admns2b2dW, %adbc1s2b2dW : tensor<384x1x7x7xf32>
    %advhs2b2dW = stablehlo.divide %advns2b2dW, %adbc2s2b2dW : tensor<384x1x7x7xf32>
    %adlrs2b2dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adepss2b2dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adsqs2b2dW = stablehlo.sqrt %advhs2b2dW : tensor<384x1x7x7xf32>
    %addens2b2dW = stablehlo.add %adsqs2b2dW, %adepss2b2dW : tensor<384x1x7x7xf32>
    %adrats2b2dW = stablehlo.divide %admhs2b2dW, %addens2b2dW : tensor<384x1x7x7xf32>
    %adsts2b2dW = stablehlo.multiply %adlrs2b2dW, %adrats2b2dW : tensor<384x1x7x7xf32>
    %adsubs2b2dW = stablehlo.subtract %s2b2dW, %adsts2b2dW : tensor<384x1x7x7xf32>
    %adwds2b2dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adwdlrs2b2dW = stablehlo.multiply %adwds2b2dW, %adlrs2b2dW : tensor<384x1x7x7xf32>
    %adwdps2b2dW = stablehlo.multiply %adwdlrs2b2dW, %s2b2dW : tensor<384x1x7x7xf32>
    %adnews2b2dW = stablehlo.subtract %adsubs2b2dW, %adwdps2b2dW : tensor<384x1x7x7xf32>
    %adb1s2b2db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b2db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b2db = stablehlo.multiply %adb1s2b2db, %s2b2dbm : tensor<384xf32>
    %admgs2b2db = stablehlo.multiply %adob1s2b2db, %s2b2ddb : tensor<384xf32>
    %admns2b2db = stablehlo.add %admss2b2db, %admgs2b2db : tensor<384xf32>
    %adb2s2b2db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b2db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b2db = stablehlo.multiply %adb2s2b2db, %s2b2dbv : tensor<384xf32>
    %adg2s2b2db = stablehlo.multiply %s2b2ddb, %s2b2ddb : tensor<384xf32>
    %advgs2b2db = stablehlo.multiply %adob2s2b2db, %adg2s2b2db : tensor<384xf32>
    %advns2b2db = stablehlo.add %advss2b2db, %advgs2b2db : tensor<384xf32>
    %adbc1s2b2db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b2db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b2db = stablehlo.divide %admns2b2db, %adbc1s2b2db : tensor<384xf32>
    %advhs2b2db = stablehlo.divide %advns2b2db, %adbc2s2b2db : tensor<384xf32>
    %adlrs2b2db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b2db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b2db = stablehlo.sqrt %advhs2b2db : tensor<384xf32>
    %addens2b2db = stablehlo.add %adsqs2b2db, %adepss2b2db : tensor<384xf32>
    %adrats2b2db = stablehlo.divide %admhs2b2db, %addens2b2db : tensor<384xf32>
    %adsts2b2db = stablehlo.multiply %adlrs2b2db, %adrats2b2db : tensor<384xf32>
    %adsubs2b2db = stablehlo.subtract %s2b2db, %adsts2b2db : tensor<384xf32>
    %adwds2b2db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b2db = stablehlo.multiply %adwds2b2db, %adlrs2b2db : tensor<384xf32>
    %adwdps2b2db = stablehlo.multiply %adwdlrs2b2db, %s2b2db : tensor<384xf32>
    %adnews2b2db = stablehlo.subtract %adsubs2b2db, %adwdps2b2db : tensor<384xf32>
    %adb1s2b2ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b2ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b2ng = stablehlo.multiply %adb1s2b2ng, %s2b2ngm : tensor<f32>
    %admgs2b2ng = stablehlo.multiply %adob1s2b2ng, %s2b2dndg : tensor<f32>
    %admns2b2ng = stablehlo.add %admss2b2ng, %admgs2b2ng : tensor<f32>
    %adb2s2b2ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b2ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b2ng = stablehlo.multiply %adb2s2b2ng, %s2b2ngv : tensor<f32>
    %adg2s2b2ng = stablehlo.multiply %s2b2dndg, %s2b2dndg : tensor<f32>
    %advgs2b2ng = stablehlo.multiply %adob2s2b2ng, %adg2s2b2ng : tensor<f32>
    %advns2b2ng = stablehlo.add %advss2b2ng, %advgs2b2ng : tensor<f32>
    %adbc1s2b2ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b2ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b2ng = stablehlo.divide %admns2b2ng, %adbc1s2b2ng : tensor<f32>
    %advhs2b2ng = stablehlo.divide %advns2b2ng, %adbc2s2b2ng : tensor<f32>
    %adlrs2b2ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b2ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b2ng = stablehlo.sqrt %advhs2b2ng : tensor<f32>
    %addens2b2ng = stablehlo.add %adsqs2b2ng, %adepss2b2ng : tensor<f32>
    %adrats2b2ng = stablehlo.divide %admhs2b2ng, %addens2b2ng : tensor<f32>
    %adsts2b2ng = stablehlo.multiply %adlrs2b2ng, %adrats2b2ng : tensor<f32>
    %adsubs2b2ng = stablehlo.subtract %s2b2ng, %adsts2b2ng : tensor<f32>
    %adwds2b2ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b2ng = stablehlo.multiply %adwds2b2ng, %adlrs2b2ng : tensor<f32>
    %adwdps2b2ng = stablehlo.multiply %adwdlrs2b2ng, %s2b2ng : tensor<f32>
    %adnews2b2ng = stablehlo.subtract %adsubs2b2ng, %adwdps2b2ng : tensor<f32>
    %adb1s2b2nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b2nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b2nbt = stablehlo.multiply %adb1s2b2nbt, %s2b2nbtm : tensor<f32>
    %admgs2b2nbt = stablehlo.multiply %adob1s2b2nbt, %s2b2dndb : tensor<f32>
    %admns2b2nbt = stablehlo.add %admss2b2nbt, %admgs2b2nbt : tensor<f32>
    %adb2s2b2nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b2nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b2nbt = stablehlo.multiply %adb2s2b2nbt, %s2b2nbtv : tensor<f32>
    %adg2s2b2nbt = stablehlo.multiply %s2b2dndb, %s2b2dndb : tensor<f32>
    %advgs2b2nbt = stablehlo.multiply %adob2s2b2nbt, %adg2s2b2nbt : tensor<f32>
    %advns2b2nbt = stablehlo.add %advss2b2nbt, %advgs2b2nbt : tensor<f32>
    %adbc1s2b2nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b2nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b2nbt = stablehlo.divide %admns2b2nbt, %adbc1s2b2nbt : tensor<f32>
    %advhs2b2nbt = stablehlo.divide %advns2b2nbt, %adbc2s2b2nbt : tensor<f32>
    %adlrs2b2nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b2nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b2nbt = stablehlo.sqrt %advhs2b2nbt : tensor<f32>
    %addens2b2nbt = stablehlo.add %adsqs2b2nbt, %adepss2b2nbt : tensor<f32>
    %adrats2b2nbt = stablehlo.divide %admhs2b2nbt, %addens2b2nbt : tensor<f32>
    %adsts2b2nbt = stablehlo.multiply %adlrs2b2nbt, %adrats2b2nbt : tensor<f32>
    %adsubs2b2nbt = stablehlo.subtract %s2b2nbt, %adsts2b2nbt : tensor<f32>
    %adwds2b2nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b2nbt = stablehlo.multiply %adwds2b2nbt, %adlrs2b2nbt : tensor<f32>
    %adwdps2b2nbt = stablehlo.multiply %adwdlrs2b2nbt, %s2b2nbt : tensor<f32>
    %adnews2b2nbt = stablehlo.subtract %adsubs2b2nbt, %adwdps2b2nbt : tensor<f32>
    %adb1s2b2eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob1s2b2eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admss2b2eW = stablehlo.multiply %adb1s2b2eW, %s2b2eWm : tensor<1536x384x1x1xf32>
    %admgs2b2eW = stablehlo.multiply %adob1s2b2eW, %s2b2deW : tensor<1536x384x1x1xf32>
    %admns2b2eW = stablehlo.add %admss2b2eW, %admgs2b2eW : tensor<1536x384x1x1xf32>
    %adb2s2b2eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob2s2b2eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %advss2b2eW = stablehlo.multiply %adb2s2b2eW, %s2b2eWv : tensor<1536x384x1x1xf32>
    %adg2s2b2eW = stablehlo.multiply %s2b2deW, %s2b2deW : tensor<1536x384x1x1xf32>
    %advgs2b2eW = stablehlo.multiply %adob2s2b2eW, %adg2s2b2eW : tensor<1536x384x1x1xf32>
    %advns2b2eW = stablehlo.add %advss2b2eW, %advgs2b2eW : tensor<1536x384x1x1xf32>
    %adbc1s2b2eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adbc2s2b2eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admhs2b2eW = stablehlo.divide %admns2b2eW, %adbc1s2b2eW : tensor<1536x384x1x1xf32>
    %advhs2b2eW = stablehlo.divide %advns2b2eW, %adbc2s2b2eW : tensor<1536x384x1x1xf32>
    %adlrs2b2eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adepss2b2eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adsqs2b2eW = stablehlo.sqrt %advhs2b2eW : tensor<1536x384x1x1xf32>
    %addens2b2eW = stablehlo.add %adsqs2b2eW, %adepss2b2eW : tensor<1536x384x1x1xf32>
    %adrats2b2eW = stablehlo.divide %admhs2b2eW, %addens2b2eW : tensor<1536x384x1x1xf32>
    %adsts2b2eW = stablehlo.multiply %adlrs2b2eW, %adrats2b2eW : tensor<1536x384x1x1xf32>
    %adsubs2b2eW = stablehlo.subtract %s2b2eW, %adsts2b2eW : tensor<1536x384x1x1xf32>
    %adwds2b2eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adwdlrs2b2eW = stablehlo.multiply %adwds2b2eW, %adlrs2b2eW : tensor<1536x384x1x1xf32>
    %adwdps2b2eW = stablehlo.multiply %adwdlrs2b2eW, %s2b2eW : tensor<1536x384x1x1xf32>
    %adnews2b2eW = stablehlo.subtract %adsubs2b2eW, %adwdps2b2eW : tensor<1536x384x1x1xf32>
    %adb1s2b2eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob1s2b2eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admss2b2eb = stablehlo.multiply %adb1s2b2eb, %s2b2ebm : tensor<1536xf32>
    %admgs2b2eb = stablehlo.multiply %adob1s2b2eb, %s2b2deb : tensor<1536xf32>
    %admns2b2eb = stablehlo.add %admss2b2eb, %admgs2b2eb : tensor<1536xf32>
    %adb2s2b2eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob2s2b2eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %advss2b2eb = stablehlo.multiply %adb2s2b2eb, %s2b2ebv : tensor<1536xf32>
    %adg2s2b2eb = stablehlo.multiply %s2b2deb, %s2b2deb : tensor<1536xf32>
    %advgs2b2eb = stablehlo.multiply %adob2s2b2eb, %adg2s2b2eb : tensor<1536xf32>
    %advns2b2eb = stablehlo.add %advss2b2eb, %advgs2b2eb : tensor<1536xf32>
    %adbc1s2b2eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adbc2s2b2eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admhs2b2eb = stablehlo.divide %admns2b2eb, %adbc1s2b2eb : tensor<1536xf32>
    %advhs2b2eb = stablehlo.divide %advns2b2eb, %adbc2s2b2eb : tensor<1536xf32>
    %adlrs2b2eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adepss2b2eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adsqs2b2eb = stablehlo.sqrt %advhs2b2eb : tensor<1536xf32>
    %addens2b2eb = stablehlo.add %adsqs2b2eb, %adepss2b2eb : tensor<1536xf32>
    %adrats2b2eb = stablehlo.divide %admhs2b2eb, %addens2b2eb : tensor<1536xf32>
    %adsts2b2eb = stablehlo.multiply %adlrs2b2eb, %adrats2b2eb : tensor<1536xf32>
    %adsubs2b2eb = stablehlo.subtract %s2b2eb, %adsts2b2eb : tensor<1536xf32>
    %adwds2b2eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adwdlrs2b2eb = stablehlo.multiply %adwds2b2eb, %adlrs2b2eb : tensor<1536xf32>
    %adwdps2b2eb = stablehlo.multiply %adwdlrs2b2eb, %s2b2eb : tensor<1536xf32>
    %adnews2b2eb = stablehlo.subtract %adsubs2b2eb, %adwdps2b2eb : tensor<1536xf32>
    %adb1s2b2pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob1s2b2pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admss2b2pW = stablehlo.multiply %adb1s2b2pW, %s2b2pWm : tensor<384x1536x1x1xf32>
    %admgs2b2pW = stablehlo.multiply %adob1s2b2pW, %s2b2dpW : tensor<384x1536x1x1xf32>
    %admns2b2pW = stablehlo.add %admss2b2pW, %admgs2b2pW : tensor<384x1536x1x1xf32>
    %adb2s2b2pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob2s2b2pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %advss2b2pW = stablehlo.multiply %adb2s2b2pW, %s2b2pWv : tensor<384x1536x1x1xf32>
    %adg2s2b2pW = stablehlo.multiply %s2b2dpW, %s2b2dpW : tensor<384x1536x1x1xf32>
    %advgs2b2pW = stablehlo.multiply %adob2s2b2pW, %adg2s2b2pW : tensor<384x1536x1x1xf32>
    %advns2b2pW = stablehlo.add %advss2b2pW, %advgs2b2pW : tensor<384x1536x1x1xf32>
    %adbc1s2b2pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adbc2s2b2pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admhs2b2pW = stablehlo.divide %admns2b2pW, %adbc1s2b2pW : tensor<384x1536x1x1xf32>
    %advhs2b2pW = stablehlo.divide %advns2b2pW, %adbc2s2b2pW : tensor<384x1536x1x1xf32>
    %adlrs2b2pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adepss2b2pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adsqs2b2pW = stablehlo.sqrt %advhs2b2pW : tensor<384x1536x1x1xf32>
    %addens2b2pW = stablehlo.add %adsqs2b2pW, %adepss2b2pW : tensor<384x1536x1x1xf32>
    %adrats2b2pW = stablehlo.divide %admhs2b2pW, %addens2b2pW : tensor<384x1536x1x1xf32>
    %adsts2b2pW = stablehlo.multiply %adlrs2b2pW, %adrats2b2pW : tensor<384x1536x1x1xf32>
    %adsubs2b2pW = stablehlo.subtract %s2b2pW, %adsts2b2pW : tensor<384x1536x1x1xf32>
    %adwds2b2pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adwdlrs2b2pW = stablehlo.multiply %adwds2b2pW, %adlrs2b2pW : tensor<384x1536x1x1xf32>
    %adwdps2b2pW = stablehlo.multiply %adwdlrs2b2pW, %s2b2pW : tensor<384x1536x1x1xf32>
    %adnews2b2pW = stablehlo.subtract %adsubs2b2pW, %adwdps2b2pW : tensor<384x1536x1x1xf32>
    %adb1s2b2pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b2pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b2pb = stablehlo.multiply %adb1s2b2pb, %s2b2pbm : tensor<384xf32>
    %admgs2b2pb = stablehlo.multiply %adob1s2b2pb, %s2b2dpb : tensor<384xf32>
    %admns2b2pb = stablehlo.add %admss2b2pb, %admgs2b2pb : tensor<384xf32>
    %adb2s2b2pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b2pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b2pb = stablehlo.multiply %adb2s2b2pb, %s2b2pbv : tensor<384xf32>
    %adg2s2b2pb = stablehlo.multiply %s2b2dpb, %s2b2dpb : tensor<384xf32>
    %advgs2b2pb = stablehlo.multiply %adob2s2b2pb, %adg2s2b2pb : tensor<384xf32>
    %advns2b2pb = stablehlo.add %advss2b2pb, %advgs2b2pb : tensor<384xf32>
    %adbc1s2b2pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b2pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b2pb = stablehlo.divide %admns2b2pb, %adbc1s2b2pb : tensor<384xf32>
    %advhs2b2pb = stablehlo.divide %advns2b2pb, %adbc2s2b2pb : tensor<384xf32>
    %adlrs2b2pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b2pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b2pb = stablehlo.sqrt %advhs2b2pb : tensor<384xf32>
    %addens2b2pb = stablehlo.add %adsqs2b2pb, %adepss2b2pb : tensor<384xf32>
    %adrats2b2pb = stablehlo.divide %admhs2b2pb, %addens2b2pb : tensor<384xf32>
    %adsts2b2pb = stablehlo.multiply %adlrs2b2pb, %adrats2b2pb : tensor<384xf32>
    %adsubs2b2pb = stablehlo.subtract %s2b2pb, %adsts2b2pb : tensor<384xf32>
    %adwds2b2pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b2pb = stablehlo.multiply %adwds2b2pb, %adlrs2b2pb : tensor<384xf32>
    %adwdps2b2pb = stablehlo.multiply %adwdlrs2b2pb, %s2b2pb : tensor<384xf32>
    %adnews2b2pb = stablehlo.subtract %adsubs2b2pb, %adwdps2b2pb : tensor<384xf32>
    %adb1s2b2lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b2lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b2lg = stablehlo.multiply %adb1s2b2lg, %s2b2lgm : tensor<384xf32>
    %admgs2b2lg = stablehlo.multiply %adob1s2b2lg, %s2b2dlsdg : tensor<384xf32>
    %admns2b2lg = stablehlo.add %admss2b2lg, %admgs2b2lg : tensor<384xf32>
    %adb2s2b2lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b2lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b2lg = stablehlo.multiply %adb2s2b2lg, %s2b2lgv : tensor<384xf32>
    %adg2s2b2lg = stablehlo.multiply %s2b2dlsdg, %s2b2dlsdg : tensor<384xf32>
    %advgs2b2lg = stablehlo.multiply %adob2s2b2lg, %adg2s2b2lg : tensor<384xf32>
    %advns2b2lg = stablehlo.add %advss2b2lg, %advgs2b2lg : tensor<384xf32>
    %adbc1s2b2lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b2lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b2lg = stablehlo.divide %admns2b2lg, %adbc1s2b2lg : tensor<384xf32>
    %advhs2b2lg = stablehlo.divide %advns2b2lg, %adbc2s2b2lg : tensor<384xf32>
    %adlrs2b2lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b2lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b2lg = stablehlo.sqrt %advhs2b2lg : tensor<384xf32>
    %addens2b2lg = stablehlo.add %adsqs2b2lg, %adepss2b2lg : tensor<384xf32>
    %adrats2b2lg = stablehlo.divide %admhs2b2lg, %addens2b2lg : tensor<384xf32>
    %adsts2b2lg = stablehlo.multiply %adlrs2b2lg, %adrats2b2lg : tensor<384xf32>
    %adsubs2b2lg = stablehlo.subtract %s2b2lg, %adsts2b2lg : tensor<384xf32>
    %adwds2b2lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b2lg = stablehlo.multiply %adwds2b2lg, %adlrs2b2lg : tensor<384xf32>
    %adwdps2b2lg = stablehlo.multiply %adwdlrs2b2lg, %s2b2lg : tensor<384xf32>
    %adnews2b2lg = stablehlo.subtract %adsubs2b2lg, %adwdps2b2lg : tensor<384xf32>
    %adb1s2b3dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob1s2b3dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admss2b3dW = stablehlo.multiply %adb1s2b3dW, %s2b3dWm : tensor<384x1x7x7xf32>
    %admgs2b3dW = stablehlo.multiply %adob1s2b3dW, %s2b3ddW : tensor<384x1x7x7xf32>
    %admns2b3dW = stablehlo.add %admss2b3dW, %admgs2b3dW : tensor<384x1x7x7xf32>
    %adb2s2b3dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob2s2b3dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %advss2b3dW = stablehlo.multiply %adb2s2b3dW, %s2b3dWv : tensor<384x1x7x7xf32>
    %adg2s2b3dW = stablehlo.multiply %s2b3ddW, %s2b3ddW : tensor<384x1x7x7xf32>
    %advgs2b3dW = stablehlo.multiply %adob2s2b3dW, %adg2s2b3dW : tensor<384x1x7x7xf32>
    %advns2b3dW = stablehlo.add %advss2b3dW, %advgs2b3dW : tensor<384x1x7x7xf32>
    %adbc1s2b3dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adbc2s2b3dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admhs2b3dW = stablehlo.divide %admns2b3dW, %adbc1s2b3dW : tensor<384x1x7x7xf32>
    %advhs2b3dW = stablehlo.divide %advns2b3dW, %adbc2s2b3dW : tensor<384x1x7x7xf32>
    %adlrs2b3dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adepss2b3dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adsqs2b3dW = stablehlo.sqrt %advhs2b3dW : tensor<384x1x7x7xf32>
    %addens2b3dW = stablehlo.add %adsqs2b3dW, %adepss2b3dW : tensor<384x1x7x7xf32>
    %adrats2b3dW = stablehlo.divide %admhs2b3dW, %addens2b3dW : tensor<384x1x7x7xf32>
    %adsts2b3dW = stablehlo.multiply %adlrs2b3dW, %adrats2b3dW : tensor<384x1x7x7xf32>
    %adsubs2b3dW = stablehlo.subtract %s2b3dW, %adsts2b3dW : tensor<384x1x7x7xf32>
    %adwds2b3dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adwdlrs2b3dW = stablehlo.multiply %adwds2b3dW, %adlrs2b3dW : tensor<384x1x7x7xf32>
    %adwdps2b3dW = stablehlo.multiply %adwdlrs2b3dW, %s2b3dW : tensor<384x1x7x7xf32>
    %adnews2b3dW = stablehlo.subtract %adsubs2b3dW, %adwdps2b3dW : tensor<384x1x7x7xf32>
    %adb1s2b3db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b3db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b3db = stablehlo.multiply %adb1s2b3db, %s2b3dbm : tensor<384xf32>
    %admgs2b3db = stablehlo.multiply %adob1s2b3db, %s2b3ddb : tensor<384xf32>
    %admns2b3db = stablehlo.add %admss2b3db, %admgs2b3db : tensor<384xf32>
    %adb2s2b3db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b3db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b3db = stablehlo.multiply %adb2s2b3db, %s2b3dbv : tensor<384xf32>
    %adg2s2b3db = stablehlo.multiply %s2b3ddb, %s2b3ddb : tensor<384xf32>
    %advgs2b3db = stablehlo.multiply %adob2s2b3db, %adg2s2b3db : tensor<384xf32>
    %advns2b3db = stablehlo.add %advss2b3db, %advgs2b3db : tensor<384xf32>
    %adbc1s2b3db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b3db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b3db = stablehlo.divide %admns2b3db, %adbc1s2b3db : tensor<384xf32>
    %advhs2b3db = stablehlo.divide %advns2b3db, %adbc2s2b3db : tensor<384xf32>
    %adlrs2b3db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b3db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b3db = stablehlo.sqrt %advhs2b3db : tensor<384xf32>
    %addens2b3db = stablehlo.add %adsqs2b3db, %adepss2b3db : tensor<384xf32>
    %adrats2b3db = stablehlo.divide %admhs2b3db, %addens2b3db : tensor<384xf32>
    %adsts2b3db = stablehlo.multiply %adlrs2b3db, %adrats2b3db : tensor<384xf32>
    %adsubs2b3db = stablehlo.subtract %s2b3db, %adsts2b3db : tensor<384xf32>
    %adwds2b3db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b3db = stablehlo.multiply %adwds2b3db, %adlrs2b3db : tensor<384xf32>
    %adwdps2b3db = stablehlo.multiply %adwdlrs2b3db, %s2b3db : tensor<384xf32>
    %adnews2b3db = stablehlo.subtract %adsubs2b3db, %adwdps2b3db : tensor<384xf32>
    %adb1s2b3ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b3ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b3ng = stablehlo.multiply %adb1s2b3ng, %s2b3ngm : tensor<f32>
    %admgs2b3ng = stablehlo.multiply %adob1s2b3ng, %s2b3dndg : tensor<f32>
    %admns2b3ng = stablehlo.add %admss2b3ng, %admgs2b3ng : tensor<f32>
    %adb2s2b3ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b3ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b3ng = stablehlo.multiply %adb2s2b3ng, %s2b3ngv : tensor<f32>
    %adg2s2b3ng = stablehlo.multiply %s2b3dndg, %s2b3dndg : tensor<f32>
    %advgs2b3ng = stablehlo.multiply %adob2s2b3ng, %adg2s2b3ng : tensor<f32>
    %advns2b3ng = stablehlo.add %advss2b3ng, %advgs2b3ng : tensor<f32>
    %adbc1s2b3ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b3ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b3ng = stablehlo.divide %admns2b3ng, %adbc1s2b3ng : tensor<f32>
    %advhs2b3ng = stablehlo.divide %advns2b3ng, %adbc2s2b3ng : tensor<f32>
    %adlrs2b3ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b3ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b3ng = stablehlo.sqrt %advhs2b3ng : tensor<f32>
    %addens2b3ng = stablehlo.add %adsqs2b3ng, %adepss2b3ng : tensor<f32>
    %adrats2b3ng = stablehlo.divide %admhs2b3ng, %addens2b3ng : tensor<f32>
    %adsts2b3ng = stablehlo.multiply %adlrs2b3ng, %adrats2b3ng : tensor<f32>
    %adsubs2b3ng = stablehlo.subtract %s2b3ng, %adsts2b3ng : tensor<f32>
    %adwds2b3ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b3ng = stablehlo.multiply %adwds2b3ng, %adlrs2b3ng : tensor<f32>
    %adwdps2b3ng = stablehlo.multiply %adwdlrs2b3ng, %s2b3ng : tensor<f32>
    %adnews2b3ng = stablehlo.subtract %adsubs2b3ng, %adwdps2b3ng : tensor<f32>
    %adb1s2b3nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b3nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b3nbt = stablehlo.multiply %adb1s2b3nbt, %s2b3nbtm : tensor<f32>
    %admgs2b3nbt = stablehlo.multiply %adob1s2b3nbt, %s2b3dndb : tensor<f32>
    %admns2b3nbt = stablehlo.add %admss2b3nbt, %admgs2b3nbt : tensor<f32>
    %adb2s2b3nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b3nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b3nbt = stablehlo.multiply %adb2s2b3nbt, %s2b3nbtv : tensor<f32>
    %adg2s2b3nbt = stablehlo.multiply %s2b3dndb, %s2b3dndb : tensor<f32>
    %advgs2b3nbt = stablehlo.multiply %adob2s2b3nbt, %adg2s2b3nbt : tensor<f32>
    %advns2b3nbt = stablehlo.add %advss2b3nbt, %advgs2b3nbt : tensor<f32>
    %adbc1s2b3nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b3nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b3nbt = stablehlo.divide %admns2b3nbt, %adbc1s2b3nbt : tensor<f32>
    %advhs2b3nbt = stablehlo.divide %advns2b3nbt, %adbc2s2b3nbt : tensor<f32>
    %adlrs2b3nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b3nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b3nbt = stablehlo.sqrt %advhs2b3nbt : tensor<f32>
    %addens2b3nbt = stablehlo.add %adsqs2b3nbt, %adepss2b3nbt : tensor<f32>
    %adrats2b3nbt = stablehlo.divide %admhs2b3nbt, %addens2b3nbt : tensor<f32>
    %adsts2b3nbt = stablehlo.multiply %adlrs2b3nbt, %adrats2b3nbt : tensor<f32>
    %adsubs2b3nbt = stablehlo.subtract %s2b3nbt, %adsts2b3nbt : tensor<f32>
    %adwds2b3nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b3nbt = stablehlo.multiply %adwds2b3nbt, %adlrs2b3nbt : tensor<f32>
    %adwdps2b3nbt = stablehlo.multiply %adwdlrs2b3nbt, %s2b3nbt : tensor<f32>
    %adnews2b3nbt = stablehlo.subtract %adsubs2b3nbt, %adwdps2b3nbt : tensor<f32>
    %adb1s2b3eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob1s2b3eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admss2b3eW = stablehlo.multiply %adb1s2b3eW, %s2b3eWm : tensor<1536x384x1x1xf32>
    %admgs2b3eW = stablehlo.multiply %adob1s2b3eW, %s2b3deW : tensor<1536x384x1x1xf32>
    %admns2b3eW = stablehlo.add %admss2b3eW, %admgs2b3eW : tensor<1536x384x1x1xf32>
    %adb2s2b3eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob2s2b3eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %advss2b3eW = stablehlo.multiply %adb2s2b3eW, %s2b3eWv : tensor<1536x384x1x1xf32>
    %adg2s2b3eW = stablehlo.multiply %s2b3deW, %s2b3deW : tensor<1536x384x1x1xf32>
    %advgs2b3eW = stablehlo.multiply %adob2s2b3eW, %adg2s2b3eW : tensor<1536x384x1x1xf32>
    %advns2b3eW = stablehlo.add %advss2b3eW, %advgs2b3eW : tensor<1536x384x1x1xf32>
    %adbc1s2b3eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adbc2s2b3eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admhs2b3eW = stablehlo.divide %admns2b3eW, %adbc1s2b3eW : tensor<1536x384x1x1xf32>
    %advhs2b3eW = stablehlo.divide %advns2b3eW, %adbc2s2b3eW : tensor<1536x384x1x1xf32>
    %adlrs2b3eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adepss2b3eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adsqs2b3eW = stablehlo.sqrt %advhs2b3eW : tensor<1536x384x1x1xf32>
    %addens2b3eW = stablehlo.add %adsqs2b3eW, %adepss2b3eW : tensor<1536x384x1x1xf32>
    %adrats2b3eW = stablehlo.divide %admhs2b3eW, %addens2b3eW : tensor<1536x384x1x1xf32>
    %adsts2b3eW = stablehlo.multiply %adlrs2b3eW, %adrats2b3eW : tensor<1536x384x1x1xf32>
    %adsubs2b3eW = stablehlo.subtract %s2b3eW, %adsts2b3eW : tensor<1536x384x1x1xf32>
    %adwds2b3eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adwdlrs2b3eW = stablehlo.multiply %adwds2b3eW, %adlrs2b3eW : tensor<1536x384x1x1xf32>
    %adwdps2b3eW = stablehlo.multiply %adwdlrs2b3eW, %s2b3eW : tensor<1536x384x1x1xf32>
    %adnews2b3eW = stablehlo.subtract %adsubs2b3eW, %adwdps2b3eW : tensor<1536x384x1x1xf32>
    %adb1s2b3eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob1s2b3eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admss2b3eb = stablehlo.multiply %adb1s2b3eb, %s2b3ebm : tensor<1536xf32>
    %admgs2b3eb = stablehlo.multiply %adob1s2b3eb, %s2b3deb : tensor<1536xf32>
    %admns2b3eb = stablehlo.add %admss2b3eb, %admgs2b3eb : tensor<1536xf32>
    %adb2s2b3eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob2s2b3eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %advss2b3eb = stablehlo.multiply %adb2s2b3eb, %s2b3ebv : tensor<1536xf32>
    %adg2s2b3eb = stablehlo.multiply %s2b3deb, %s2b3deb : tensor<1536xf32>
    %advgs2b3eb = stablehlo.multiply %adob2s2b3eb, %adg2s2b3eb : tensor<1536xf32>
    %advns2b3eb = stablehlo.add %advss2b3eb, %advgs2b3eb : tensor<1536xf32>
    %adbc1s2b3eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adbc2s2b3eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admhs2b3eb = stablehlo.divide %admns2b3eb, %adbc1s2b3eb : tensor<1536xf32>
    %advhs2b3eb = stablehlo.divide %advns2b3eb, %adbc2s2b3eb : tensor<1536xf32>
    %adlrs2b3eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adepss2b3eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adsqs2b3eb = stablehlo.sqrt %advhs2b3eb : tensor<1536xf32>
    %addens2b3eb = stablehlo.add %adsqs2b3eb, %adepss2b3eb : tensor<1536xf32>
    %adrats2b3eb = stablehlo.divide %admhs2b3eb, %addens2b3eb : tensor<1536xf32>
    %adsts2b3eb = stablehlo.multiply %adlrs2b3eb, %adrats2b3eb : tensor<1536xf32>
    %adsubs2b3eb = stablehlo.subtract %s2b3eb, %adsts2b3eb : tensor<1536xf32>
    %adwds2b3eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adwdlrs2b3eb = stablehlo.multiply %adwds2b3eb, %adlrs2b3eb : tensor<1536xf32>
    %adwdps2b3eb = stablehlo.multiply %adwdlrs2b3eb, %s2b3eb : tensor<1536xf32>
    %adnews2b3eb = stablehlo.subtract %adsubs2b3eb, %adwdps2b3eb : tensor<1536xf32>
    %adb1s2b3pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob1s2b3pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admss2b3pW = stablehlo.multiply %adb1s2b3pW, %s2b3pWm : tensor<384x1536x1x1xf32>
    %admgs2b3pW = stablehlo.multiply %adob1s2b3pW, %s2b3dpW : tensor<384x1536x1x1xf32>
    %admns2b3pW = stablehlo.add %admss2b3pW, %admgs2b3pW : tensor<384x1536x1x1xf32>
    %adb2s2b3pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob2s2b3pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %advss2b3pW = stablehlo.multiply %adb2s2b3pW, %s2b3pWv : tensor<384x1536x1x1xf32>
    %adg2s2b3pW = stablehlo.multiply %s2b3dpW, %s2b3dpW : tensor<384x1536x1x1xf32>
    %advgs2b3pW = stablehlo.multiply %adob2s2b3pW, %adg2s2b3pW : tensor<384x1536x1x1xf32>
    %advns2b3pW = stablehlo.add %advss2b3pW, %advgs2b3pW : tensor<384x1536x1x1xf32>
    %adbc1s2b3pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adbc2s2b3pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admhs2b3pW = stablehlo.divide %admns2b3pW, %adbc1s2b3pW : tensor<384x1536x1x1xf32>
    %advhs2b3pW = stablehlo.divide %advns2b3pW, %adbc2s2b3pW : tensor<384x1536x1x1xf32>
    %adlrs2b3pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adepss2b3pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adsqs2b3pW = stablehlo.sqrt %advhs2b3pW : tensor<384x1536x1x1xf32>
    %addens2b3pW = stablehlo.add %adsqs2b3pW, %adepss2b3pW : tensor<384x1536x1x1xf32>
    %adrats2b3pW = stablehlo.divide %admhs2b3pW, %addens2b3pW : tensor<384x1536x1x1xf32>
    %adsts2b3pW = stablehlo.multiply %adlrs2b3pW, %adrats2b3pW : tensor<384x1536x1x1xf32>
    %adsubs2b3pW = stablehlo.subtract %s2b3pW, %adsts2b3pW : tensor<384x1536x1x1xf32>
    %adwds2b3pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adwdlrs2b3pW = stablehlo.multiply %adwds2b3pW, %adlrs2b3pW : tensor<384x1536x1x1xf32>
    %adwdps2b3pW = stablehlo.multiply %adwdlrs2b3pW, %s2b3pW : tensor<384x1536x1x1xf32>
    %adnews2b3pW = stablehlo.subtract %adsubs2b3pW, %adwdps2b3pW : tensor<384x1536x1x1xf32>
    %adb1s2b3pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b3pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b3pb = stablehlo.multiply %adb1s2b3pb, %s2b3pbm : tensor<384xf32>
    %admgs2b3pb = stablehlo.multiply %adob1s2b3pb, %s2b3dpb : tensor<384xf32>
    %admns2b3pb = stablehlo.add %admss2b3pb, %admgs2b3pb : tensor<384xf32>
    %adb2s2b3pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b3pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b3pb = stablehlo.multiply %adb2s2b3pb, %s2b3pbv : tensor<384xf32>
    %adg2s2b3pb = stablehlo.multiply %s2b3dpb, %s2b3dpb : tensor<384xf32>
    %advgs2b3pb = stablehlo.multiply %adob2s2b3pb, %adg2s2b3pb : tensor<384xf32>
    %advns2b3pb = stablehlo.add %advss2b3pb, %advgs2b3pb : tensor<384xf32>
    %adbc1s2b3pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b3pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b3pb = stablehlo.divide %admns2b3pb, %adbc1s2b3pb : tensor<384xf32>
    %advhs2b3pb = stablehlo.divide %advns2b3pb, %adbc2s2b3pb : tensor<384xf32>
    %adlrs2b3pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b3pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b3pb = stablehlo.sqrt %advhs2b3pb : tensor<384xf32>
    %addens2b3pb = stablehlo.add %adsqs2b3pb, %adepss2b3pb : tensor<384xf32>
    %adrats2b3pb = stablehlo.divide %admhs2b3pb, %addens2b3pb : tensor<384xf32>
    %adsts2b3pb = stablehlo.multiply %adlrs2b3pb, %adrats2b3pb : tensor<384xf32>
    %adsubs2b3pb = stablehlo.subtract %s2b3pb, %adsts2b3pb : tensor<384xf32>
    %adwds2b3pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b3pb = stablehlo.multiply %adwds2b3pb, %adlrs2b3pb : tensor<384xf32>
    %adwdps2b3pb = stablehlo.multiply %adwdlrs2b3pb, %s2b3pb : tensor<384xf32>
    %adnews2b3pb = stablehlo.subtract %adsubs2b3pb, %adwdps2b3pb : tensor<384xf32>
    %adb1s2b3lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b3lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b3lg = stablehlo.multiply %adb1s2b3lg, %s2b3lgm : tensor<384xf32>
    %admgs2b3lg = stablehlo.multiply %adob1s2b3lg, %s2b3dlsdg : tensor<384xf32>
    %admns2b3lg = stablehlo.add %admss2b3lg, %admgs2b3lg : tensor<384xf32>
    %adb2s2b3lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b3lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b3lg = stablehlo.multiply %adb2s2b3lg, %s2b3lgv : tensor<384xf32>
    %adg2s2b3lg = stablehlo.multiply %s2b3dlsdg, %s2b3dlsdg : tensor<384xf32>
    %advgs2b3lg = stablehlo.multiply %adob2s2b3lg, %adg2s2b3lg : tensor<384xf32>
    %advns2b3lg = stablehlo.add %advss2b3lg, %advgs2b3lg : tensor<384xf32>
    %adbc1s2b3lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b3lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b3lg = stablehlo.divide %admns2b3lg, %adbc1s2b3lg : tensor<384xf32>
    %advhs2b3lg = stablehlo.divide %advns2b3lg, %adbc2s2b3lg : tensor<384xf32>
    %adlrs2b3lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b3lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b3lg = stablehlo.sqrt %advhs2b3lg : tensor<384xf32>
    %addens2b3lg = stablehlo.add %adsqs2b3lg, %adepss2b3lg : tensor<384xf32>
    %adrats2b3lg = stablehlo.divide %admhs2b3lg, %addens2b3lg : tensor<384xf32>
    %adsts2b3lg = stablehlo.multiply %adlrs2b3lg, %adrats2b3lg : tensor<384xf32>
    %adsubs2b3lg = stablehlo.subtract %s2b3lg, %adsts2b3lg : tensor<384xf32>
    %adwds2b3lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b3lg = stablehlo.multiply %adwds2b3lg, %adlrs2b3lg : tensor<384xf32>
    %adwdps2b3lg = stablehlo.multiply %adwdlrs2b3lg, %s2b3lg : tensor<384xf32>
    %adnews2b3lg = stablehlo.subtract %adsubs2b3lg, %adwdps2b3lg : tensor<384xf32>
    %adb1s2b4dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob1s2b4dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admss2b4dW = stablehlo.multiply %adb1s2b4dW, %s2b4dWm : tensor<384x1x7x7xf32>
    %admgs2b4dW = stablehlo.multiply %adob1s2b4dW, %s2b4ddW : tensor<384x1x7x7xf32>
    %admns2b4dW = stablehlo.add %admss2b4dW, %admgs2b4dW : tensor<384x1x7x7xf32>
    %adb2s2b4dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob2s2b4dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %advss2b4dW = stablehlo.multiply %adb2s2b4dW, %s2b4dWv : tensor<384x1x7x7xf32>
    %adg2s2b4dW = stablehlo.multiply %s2b4ddW, %s2b4ddW : tensor<384x1x7x7xf32>
    %advgs2b4dW = stablehlo.multiply %adob2s2b4dW, %adg2s2b4dW : tensor<384x1x7x7xf32>
    %advns2b4dW = stablehlo.add %advss2b4dW, %advgs2b4dW : tensor<384x1x7x7xf32>
    %adbc1s2b4dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adbc2s2b4dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admhs2b4dW = stablehlo.divide %admns2b4dW, %adbc1s2b4dW : tensor<384x1x7x7xf32>
    %advhs2b4dW = stablehlo.divide %advns2b4dW, %adbc2s2b4dW : tensor<384x1x7x7xf32>
    %adlrs2b4dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adepss2b4dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adsqs2b4dW = stablehlo.sqrt %advhs2b4dW : tensor<384x1x7x7xf32>
    %addens2b4dW = stablehlo.add %adsqs2b4dW, %adepss2b4dW : tensor<384x1x7x7xf32>
    %adrats2b4dW = stablehlo.divide %admhs2b4dW, %addens2b4dW : tensor<384x1x7x7xf32>
    %adsts2b4dW = stablehlo.multiply %adlrs2b4dW, %adrats2b4dW : tensor<384x1x7x7xf32>
    %adsubs2b4dW = stablehlo.subtract %s2b4dW, %adsts2b4dW : tensor<384x1x7x7xf32>
    %adwds2b4dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adwdlrs2b4dW = stablehlo.multiply %adwds2b4dW, %adlrs2b4dW : tensor<384x1x7x7xf32>
    %adwdps2b4dW = stablehlo.multiply %adwdlrs2b4dW, %s2b4dW : tensor<384x1x7x7xf32>
    %adnews2b4dW = stablehlo.subtract %adsubs2b4dW, %adwdps2b4dW : tensor<384x1x7x7xf32>
    %adb1s2b4db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b4db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b4db = stablehlo.multiply %adb1s2b4db, %s2b4dbm : tensor<384xf32>
    %admgs2b4db = stablehlo.multiply %adob1s2b4db, %s2b4ddb : tensor<384xf32>
    %admns2b4db = stablehlo.add %admss2b4db, %admgs2b4db : tensor<384xf32>
    %adb2s2b4db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b4db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b4db = stablehlo.multiply %adb2s2b4db, %s2b4dbv : tensor<384xf32>
    %adg2s2b4db = stablehlo.multiply %s2b4ddb, %s2b4ddb : tensor<384xf32>
    %advgs2b4db = stablehlo.multiply %adob2s2b4db, %adg2s2b4db : tensor<384xf32>
    %advns2b4db = stablehlo.add %advss2b4db, %advgs2b4db : tensor<384xf32>
    %adbc1s2b4db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b4db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b4db = stablehlo.divide %admns2b4db, %adbc1s2b4db : tensor<384xf32>
    %advhs2b4db = stablehlo.divide %advns2b4db, %adbc2s2b4db : tensor<384xf32>
    %adlrs2b4db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b4db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b4db = stablehlo.sqrt %advhs2b4db : tensor<384xf32>
    %addens2b4db = stablehlo.add %adsqs2b4db, %adepss2b4db : tensor<384xf32>
    %adrats2b4db = stablehlo.divide %admhs2b4db, %addens2b4db : tensor<384xf32>
    %adsts2b4db = stablehlo.multiply %adlrs2b4db, %adrats2b4db : tensor<384xf32>
    %adsubs2b4db = stablehlo.subtract %s2b4db, %adsts2b4db : tensor<384xf32>
    %adwds2b4db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b4db = stablehlo.multiply %adwds2b4db, %adlrs2b4db : tensor<384xf32>
    %adwdps2b4db = stablehlo.multiply %adwdlrs2b4db, %s2b4db : tensor<384xf32>
    %adnews2b4db = stablehlo.subtract %adsubs2b4db, %adwdps2b4db : tensor<384xf32>
    %adb1s2b4ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b4ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b4ng = stablehlo.multiply %adb1s2b4ng, %s2b4ngm : tensor<f32>
    %admgs2b4ng = stablehlo.multiply %adob1s2b4ng, %s2b4dndg : tensor<f32>
    %admns2b4ng = stablehlo.add %admss2b4ng, %admgs2b4ng : tensor<f32>
    %adb2s2b4ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b4ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b4ng = stablehlo.multiply %adb2s2b4ng, %s2b4ngv : tensor<f32>
    %adg2s2b4ng = stablehlo.multiply %s2b4dndg, %s2b4dndg : tensor<f32>
    %advgs2b4ng = stablehlo.multiply %adob2s2b4ng, %adg2s2b4ng : tensor<f32>
    %advns2b4ng = stablehlo.add %advss2b4ng, %advgs2b4ng : tensor<f32>
    %adbc1s2b4ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b4ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b4ng = stablehlo.divide %admns2b4ng, %adbc1s2b4ng : tensor<f32>
    %advhs2b4ng = stablehlo.divide %advns2b4ng, %adbc2s2b4ng : tensor<f32>
    %adlrs2b4ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b4ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b4ng = stablehlo.sqrt %advhs2b4ng : tensor<f32>
    %addens2b4ng = stablehlo.add %adsqs2b4ng, %adepss2b4ng : tensor<f32>
    %adrats2b4ng = stablehlo.divide %admhs2b4ng, %addens2b4ng : tensor<f32>
    %adsts2b4ng = stablehlo.multiply %adlrs2b4ng, %adrats2b4ng : tensor<f32>
    %adsubs2b4ng = stablehlo.subtract %s2b4ng, %adsts2b4ng : tensor<f32>
    %adwds2b4ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b4ng = stablehlo.multiply %adwds2b4ng, %adlrs2b4ng : tensor<f32>
    %adwdps2b4ng = stablehlo.multiply %adwdlrs2b4ng, %s2b4ng : tensor<f32>
    %adnews2b4ng = stablehlo.subtract %adsubs2b4ng, %adwdps2b4ng : tensor<f32>
    %adb1s2b4nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b4nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b4nbt = stablehlo.multiply %adb1s2b4nbt, %s2b4nbtm : tensor<f32>
    %admgs2b4nbt = stablehlo.multiply %adob1s2b4nbt, %s2b4dndb : tensor<f32>
    %admns2b4nbt = stablehlo.add %admss2b4nbt, %admgs2b4nbt : tensor<f32>
    %adb2s2b4nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b4nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b4nbt = stablehlo.multiply %adb2s2b4nbt, %s2b4nbtv : tensor<f32>
    %adg2s2b4nbt = stablehlo.multiply %s2b4dndb, %s2b4dndb : tensor<f32>
    %advgs2b4nbt = stablehlo.multiply %adob2s2b4nbt, %adg2s2b4nbt : tensor<f32>
    %advns2b4nbt = stablehlo.add %advss2b4nbt, %advgs2b4nbt : tensor<f32>
    %adbc1s2b4nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b4nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b4nbt = stablehlo.divide %admns2b4nbt, %adbc1s2b4nbt : tensor<f32>
    %advhs2b4nbt = stablehlo.divide %advns2b4nbt, %adbc2s2b4nbt : tensor<f32>
    %adlrs2b4nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b4nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b4nbt = stablehlo.sqrt %advhs2b4nbt : tensor<f32>
    %addens2b4nbt = stablehlo.add %adsqs2b4nbt, %adepss2b4nbt : tensor<f32>
    %adrats2b4nbt = stablehlo.divide %admhs2b4nbt, %addens2b4nbt : tensor<f32>
    %adsts2b4nbt = stablehlo.multiply %adlrs2b4nbt, %adrats2b4nbt : tensor<f32>
    %adsubs2b4nbt = stablehlo.subtract %s2b4nbt, %adsts2b4nbt : tensor<f32>
    %adwds2b4nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b4nbt = stablehlo.multiply %adwds2b4nbt, %adlrs2b4nbt : tensor<f32>
    %adwdps2b4nbt = stablehlo.multiply %adwdlrs2b4nbt, %s2b4nbt : tensor<f32>
    %adnews2b4nbt = stablehlo.subtract %adsubs2b4nbt, %adwdps2b4nbt : tensor<f32>
    %adb1s2b4eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob1s2b4eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admss2b4eW = stablehlo.multiply %adb1s2b4eW, %s2b4eWm : tensor<1536x384x1x1xf32>
    %admgs2b4eW = stablehlo.multiply %adob1s2b4eW, %s2b4deW : tensor<1536x384x1x1xf32>
    %admns2b4eW = stablehlo.add %admss2b4eW, %admgs2b4eW : tensor<1536x384x1x1xf32>
    %adb2s2b4eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob2s2b4eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %advss2b4eW = stablehlo.multiply %adb2s2b4eW, %s2b4eWv : tensor<1536x384x1x1xf32>
    %adg2s2b4eW = stablehlo.multiply %s2b4deW, %s2b4deW : tensor<1536x384x1x1xf32>
    %advgs2b4eW = stablehlo.multiply %adob2s2b4eW, %adg2s2b4eW : tensor<1536x384x1x1xf32>
    %advns2b4eW = stablehlo.add %advss2b4eW, %advgs2b4eW : tensor<1536x384x1x1xf32>
    %adbc1s2b4eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adbc2s2b4eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admhs2b4eW = stablehlo.divide %admns2b4eW, %adbc1s2b4eW : tensor<1536x384x1x1xf32>
    %advhs2b4eW = stablehlo.divide %advns2b4eW, %adbc2s2b4eW : tensor<1536x384x1x1xf32>
    %adlrs2b4eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adepss2b4eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adsqs2b4eW = stablehlo.sqrt %advhs2b4eW : tensor<1536x384x1x1xf32>
    %addens2b4eW = stablehlo.add %adsqs2b4eW, %adepss2b4eW : tensor<1536x384x1x1xf32>
    %adrats2b4eW = stablehlo.divide %admhs2b4eW, %addens2b4eW : tensor<1536x384x1x1xf32>
    %adsts2b4eW = stablehlo.multiply %adlrs2b4eW, %adrats2b4eW : tensor<1536x384x1x1xf32>
    %adsubs2b4eW = stablehlo.subtract %s2b4eW, %adsts2b4eW : tensor<1536x384x1x1xf32>
    %adwds2b4eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adwdlrs2b4eW = stablehlo.multiply %adwds2b4eW, %adlrs2b4eW : tensor<1536x384x1x1xf32>
    %adwdps2b4eW = stablehlo.multiply %adwdlrs2b4eW, %s2b4eW : tensor<1536x384x1x1xf32>
    %adnews2b4eW = stablehlo.subtract %adsubs2b4eW, %adwdps2b4eW : tensor<1536x384x1x1xf32>
    %adb1s2b4eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob1s2b4eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admss2b4eb = stablehlo.multiply %adb1s2b4eb, %s2b4ebm : tensor<1536xf32>
    %admgs2b4eb = stablehlo.multiply %adob1s2b4eb, %s2b4deb : tensor<1536xf32>
    %admns2b4eb = stablehlo.add %admss2b4eb, %admgs2b4eb : tensor<1536xf32>
    %adb2s2b4eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob2s2b4eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %advss2b4eb = stablehlo.multiply %adb2s2b4eb, %s2b4ebv : tensor<1536xf32>
    %adg2s2b4eb = stablehlo.multiply %s2b4deb, %s2b4deb : tensor<1536xf32>
    %advgs2b4eb = stablehlo.multiply %adob2s2b4eb, %adg2s2b4eb : tensor<1536xf32>
    %advns2b4eb = stablehlo.add %advss2b4eb, %advgs2b4eb : tensor<1536xf32>
    %adbc1s2b4eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adbc2s2b4eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admhs2b4eb = stablehlo.divide %admns2b4eb, %adbc1s2b4eb : tensor<1536xf32>
    %advhs2b4eb = stablehlo.divide %advns2b4eb, %adbc2s2b4eb : tensor<1536xf32>
    %adlrs2b4eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adepss2b4eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adsqs2b4eb = stablehlo.sqrt %advhs2b4eb : tensor<1536xf32>
    %addens2b4eb = stablehlo.add %adsqs2b4eb, %adepss2b4eb : tensor<1536xf32>
    %adrats2b4eb = stablehlo.divide %admhs2b4eb, %addens2b4eb : tensor<1536xf32>
    %adsts2b4eb = stablehlo.multiply %adlrs2b4eb, %adrats2b4eb : tensor<1536xf32>
    %adsubs2b4eb = stablehlo.subtract %s2b4eb, %adsts2b4eb : tensor<1536xf32>
    %adwds2b4eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adwdlrs2b4eb = stablehlo.multiply %adwds2b4eb, %adlrs2b4eb : tensor<1536xf32>
    %adwdps2b4eb = stablehlo.multiply %adwdlrs2b4eb, %s2b4eb : tensor<1536xf32>
    %adnews2b4eb = stablehlo.subtract %adsubs2b4eb, %adwdps2b4eb : tensor<1536xf32>
    %adb1s2b4pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob1s2b4pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admss2b4pW = stablehlo.multiply %adb1s2b4pW, %s2b4pWm : tensor<384x1536x1x1xf32>
    %admgs2b4pW = stablehlo.multiply %adob1s2b4pW, %s2b4dpW : tensor<384x1536x1x1xf32>
    %admns2b4pW = stablehlo.add %admss2b4pW, %admgs2b4pW : tensor<384x1536x1x1xf32>
    %adb2s2b4pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob2s2b4pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %advss2b4pW = stablehlo.multiply %adb2s2b4pW, %s2b4pWv : tensor<384x1536x1x1xf32>
    %adg2s2b4pW = stablehlo.multiply %s2b4dpW, %s2b4dpW : tensor<384x1536x1x1xf32>
    %advgs2b4pW = stablehlo.multiply %adob2s2b4pW, %adg2s2b4pW : tensor<384x1536x1x1xf32>
    %advns2b4pW = stablehlo.add %advss2b4pW, %advgs2b4pW : tensor<384x1536x1x1xf32>
    %adbc1s2b4pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adbc2s2b4pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admhs2b4pW = stablehlo.divide %admns2b4pW, %adbc1s2b4pW : tensor<384x1536x1x1xf32>
    %advhs2b4pW = stablehlo.divide %advns2b4pW, %adbc2s2b4pW : tensor<384x1536x1x1xf32>
    %adlrs2b4pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adepss2b4pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adsqs2b4pW = stablehlo.sqrt %advhs2b4pW : tensor<384x1536x1x1xf32>
    %addens2b4pW = stablehlo.add %adsqs2b4pW, %adepss2b4pW : tensor<384x1536x1x1xf32>
    %adrats2b4pW = stablehlo.divide %admhs2b4pW, %addens2b4pW : tensor<384x1536x1x1xf32>
    %adsts2b4pW = stablehlo.multiply %adlrs2b4pW, %adrats2b4pW : tensor<384x1536x1x1xf32>
    %adsubs2b4pW = stablehlo.subtract %s2b4pW, %adsts2b4pW : tensor<384x1536x1x1xf32>
    %adwds2b4pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adwdlrs2b4pW = stablehlo.multiply %adwds2b4pW, %adlrs2b4pW : tensor<384x1536x1x1xf32>
    %adwdps2b4pW = stablehlo.multiply %adwdlrs2b4pW, %s2b4pW : tensor<384x1536x1x1xf32>
    %adnews2b4pW = stablehlo.subtract %adsubs2b4pW, %adwdps2b4pW : tensor<384x1536x1x1xf32>
    %adb1s2b4pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b4pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b4pb = stablehlo.multiply %adb1s2b4pb, %s2b4pbm : tensor<384xf32>
    %admgs2b4pb = stablehlo.multiply %adob1s2b4pb, %s2b4dpb : tensor<384xf32>
    %admns2b4pb = stablehlo.add %admss2b4pb, %admgs2b4pb : tensor<384xf32>
    %adb2s2b4pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b4pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b4pb = stablehlo.multiply %adb2s2b4pb, %s2b4pbv : tensor<384xf32>
    %adg2s2b4pb = stablehlo.multiply %s2b4dpb, %s2b4dpb : tensor<384xf32>
    %advgs2b4pb = stablehlo.multiply %adob2s2b4pb, %adg2s2b4pb : tensor<384xf32>
    %advns2b4pb = stablehlo.add %advss2b4pb, %advgs2b4pb : tensor<384xf32>
    %adbc1s2b4pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b4pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b4pb = stablehlo.divide %admns2b4pb, %adbc1s2b4pb : tensor<384xf32>
    %advhs2b4pb = stablehlo.divide %advns2b4pb, %adbc2s2b4pb : tensor<384xf32>
    %adlrs2b4pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b4pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b4pb = stablehlo.sqrt %advhs2b4pb : tensor<384xf32>
    %addens2b4pb = stablehlo.add %adsqs2b4pb, %adepss2b4pb : tensor<384xf32>
    %adrats2b4pb = stablehlo.divide %admhs2b4pb, %addens2b4pb : tensor<384xf32>
    %adsts2b4pb = stablehlo.multiply %adlrs2b4pb, %adrats2b4pb : tensor<384xf32>
    %adsubs2b4pb = stablehlo.subtract %s2b4pb, %adsts2b4pb : tensor<384xf32>
    %adwds2b4pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b4pb = stablehlo.multiply %adwds2b4pb, %adlrs2b4pb : tensor<384xf32>
    %adwdps2b4pb = stablehlo.multiply %adwdlrs2b4pb, %s2b4pb : tensor<384xf32>
    %adnews2b4pb = stablehlo.subtract %adsubs2b4pb, %adwdps2b4pb : tensor<384xf32>
    %adb1s2b4lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b4lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b4lg = stablehlo.multiply %adb1s2b4lg, %s2b4lgm : tensor<384xf32>
    %admgs2b4lg = stablehlo.multiply %adob1s2b4lg, %s2b4dlsdg : tensor<384xf32>
    %admns2b4lg = stablehlo.add %admss2b4lg, %admgs2b4lg : tensor<384xf32>
    %adb2s2b4lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b4lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b4lg = stablehlo.multiply %adb2s2b4lg, %s2b4lgv : tensor<384xf32>
    %adg2s2b4lg = stablehlo.multiply %s2b4dlsdg, %s2b4dlsdg : tensor<384xf32>
    %advgs2b4lg = stablehlo.multiply %adob2s2b4lg, %adg2s2b4lg : tensor<384xf32>
    %advns2b4lg = stablehlo.add %advss2b4lg, %advgs2b4lg : tensor<384xf32>
    %adbc1s2b4lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b4lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b4lg = stablehlo.divide %admns2b4lg, %adbc1s2b4lg : tensor<384xf32>
    %advhs2b4lg = stablehlo.divide %advns2b4lg, %adbc2s2b4lg : tensor<384xf32>
    %adlrs2b4lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b4lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b4lg = stablehlo.sqrt %advhs2b4lg : tensor<384xf32>
    %addens2b4lg = stablehlo.add %adsqs2b4lg, %adepss2b4lg : tensor<384xf32>
    %adrats2b4lg = stablehlo.divide %admhs2b4lg, %addens2b4lg : tensor<384xf32>
    %adsts2b4lg = stablehlo.multiply %adlrs2b4lg, %adrats2b4lg : tensor<384xf32>
    %adsubs2b4lg = stablehlo.subtract %s2b4lg, %adsts2b4lg : tensor<384xf32>
    %adwds2b4lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b4lg = stablehlo.multiply %adwds2b4lg, %adlrs2b4lg : tensor<384xf32>
    %adwdps2b4lg = stablehlo.multiply %adwdlrs2b4lg, %s2b4lg : tensor<384xf32>
    %adnews2b4lg = stablehlo.subtract %adsubs2b4lg, %adwdps2b4lg : tensor<384xf32>
    %adb1s2b5dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob1s2b5dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admss2b5dW = stablehlo.multiply %adb1s2b5dW, %s2b5dWm : tensor<384x1x7x7xf32>
    %admgs2b5dW = stablehlo.multiply %adob1s2b5dW, %s2b5ddW : tensor<384x1x7x7xf32>
    %admns2b5dW = stablehlo.add %admss2b5dW, %admgs2b5dW : tensor<384x1x7x7xf32>
    %adb2s2b5dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob2s2b5dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %advss2b5dW = stablehlo.multiply %adb2s2b5dW, %s2b5dWv : tensor<384x1x7x7xf32>
    %adg2s2b5dW = stablehlo.multiply %s2b5ddW, %s2b5ddW : tensor<384x1x7x7xf32>
    %advgs2b5dW = stablehlo.multiply %adob2s2b5dW, %adg2s2b5dW : tensor<384x1x7x7xf32>
    %advns2b5dW = stablehlo.add %advss2b5dW, %advgs2b5dW : tensor<384x1x7x7xf32>
    %adbc1s2b5dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adbc2s2b5dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admhs2b5dW = stablehlo.divide %admns2b5dW, %adbc1s2b5dW : tensor<384x1x7x7xf32>
    %advhs2b5dW = stablehlo.divide %advns2b5dW, %adbc2s2b5dW : tensor<384x1x7x7xf32>
    %adlrs2b5dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adepss2b5dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adsqs2b5dW = stablehlo.sqrt %advhs2b5dW : tensor<384x1x7x7xf32>
    %addens2b5dW = stablehlo.add %adsqs2b5dW, %adepss2b5dW : tensor<384x1x7x7xf32>
    %adrats2b5dW = stablehlo.divide %admhs2b5dW, %addens2b5dW : tensor<384x1x7x7xf32>
    %adsts2b5dW = stablehlo.multiply %adlrs2b5dW, %adrats2b5dW : tensor<384x1x7x7xf32>
    %adsubs2b5dW = stablehlo.subtract %s2b5dW, %adsts2b5dW : tensor<384x1x7x7xf32>
    %adwds2b5dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adwdlrs2b5dW = stablehlo.multiply %adwds2b5dW, %adlrs2b5dW : tensor<384x1x7x7xf32>
    %adwdps2b5dW = stablehlo.multiply %adwdlrs2b5dW, %s2b5dW : tensor<384x1x7x7xf32>
    %adnews2b5dW = stablehlo.subtract %adsubs2b5dW, %adwdps2b5dW : tensor<384x1x7x7xf32>
    %adb1s2b5db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b5db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b5db = stablehlo.multiply %adb1s2b5db, %s2b5dbm : tensor<384xf32>
    %admgs2b5db = stablehlo.multiply %adob1s2b5db, %s2b5ddb : tensor<384xf32>
    %admns2b5db = stablehlo.add %admss2b5db, %admgs2b5db : tensor<384xf32>
    %adb2s2b5db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b5db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b5db = stablehlo.multiply %adb2s2b5db, %s2b5dbv : tensor<384xf32>
    %adg2s2b5db = stablehlo.multiply %s2b5ddb, %s2b5ddb : tensor<384xf32>
    %advgs2b5db = stablehlo.multiply %adob2s2b5db, %adg2s2b5db : tensor<384xf32>
    %advns2b5db = stablehlo.add %advss2b5db, %advgs2b5db : tensor<384xf32>
    %adbc1s2b5db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b5db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b5db = stablehlo.divide %admns2b5db, %adbc1s2b5db : tensor<384xf32>
    %advhs2b5db = stablehlo.divide %advns2b5db, %adbc2s2b5db : tensor<384xf32>
    %adlrs2b5db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b5db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b5db = stablehlo.sqrt %advhs2b5db : tensor<384xf32>
    %addens2b5db = stablehlo.add %adsqs2b5db, %adepss2b5db : tensor<384xf32>
    %adrats2b5db = stablehlo.divide %admhs2b5db, %addens2b5db : tensor<384xf32>
    %adsts2b5db = stablehlo.multiply %adlrs2b5db, %adrats2b5db : tensor<384xf32>
    %adsubs2b5db = stablehlo.subtract %s2b5db, %adsts2b5db : tensor<384xf32>
    %adwds2b5db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b5db = stablehlo.multiply %adwds2b5db, %adlrs2b5db : tensor<384xf32>
    %adwdps2b5db = stablehlo.multiply %adwdlrs2b5db, %s2b5db : tensor<384xf32>
    %adnews2b5db = stablehlo.subtract %adsubs2b5db, %adwdps2b5db : tensor<384xf32>
    %adb1s2b5ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b5ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b5ng = stablehlo.multiply %adb1s2b5ng, %s2b5ngm : tensor<f32>
    %admgs2b5ng = stablehlo.multiply %adob1s2b5ng, %s2b5dndg : tensor<f32>
    %admns2b5ng = stablehlo.add %admss2b5ng, %admgs2b5ng : tensor<f32>
    %adb2s2b5ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b5ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b5ng = stablehlo.multiply %adb2s2b5ng, %s2b5ngv : tensor<f32>
    %adg2s2b5ng = stablehlo.multiply %s2b5dndg, %s2b5dndg : tensor<f32>
    %advgs2b5ng = stablehlo.multiply %adob2s2b5ng, %adg2s2b5ng : tensor<f32>
    %advns2b5ng = stablehlo.add %advss2b5ng, %advgs2b5ng : tensor<f32>
    %adbc1s2b5ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b5ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b5ng = stablehlo.divide %admns2b5ng, %adbc1s2b5ng : tensor<f32>
    %advhs2b5ng = stablehlo.divide %advns2b5ng, %adbc2s2b5ng : tensor<f32>
    %adlrs2b5ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b5ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b5ng = stablehlo.sqrt %advhs2b5ng : tensor<f32>
    %addens2b5ng = stablehlo.add %adsqs2b5ng, %adepss2b5ng : tensor<f32>
    %adrats2b5ng = stablehlo.divide %admhs2b5ng, %addens2b5ng : tensor<f32>
    %adsts2b5ng = stablehlo.multiply %adlrs2b5ng, %adrats2b5ng : tensor<f32>
    %adsubs2b5ng = stablehlo.subtract %s2b5ng, %adsts2b5ng : tensor<f32>
    %adwds2b5ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b5ng = stablehlo.multiply %adwds2b5ng, %adlrs2b5ng : tensor<f32>
    %adwdps2b5ng = stablehlo.multiply %adwdlrs2b5ng, %s2b5ng : tensor<f32>
    %adnews2b5ng = stablehlo.subtract %adsubs2b5ng, %adwdps2b5ng : tensor<f32>
    %adb1s2b5nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b5nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b5nbt = stablehlo.multiply %adb1s2b5nbt, %s2b5nbtm : tensor<f32>
    %admgs2b5nbt = stablehlo.multiply %adob1s2b5nbt, %s2b5dndb : tensor<f32>
    %admns2b5nbt = stablehlo.add %admss2b5nbt, %admgs2b5nbt : tensor<f32>
    %adb2s2b5nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b5nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b5nbt = stablehlo.multiply %adb2s2b5nbt, %s2b5nbtv : tensor<f32>
    %adg2s2b5nbt = stablehlo.multiply %s2b5dndb, %s2b5dndb : tensor<f32>
    %advgs2b5nbt = stablehlo.multiply %adob2s2b5nbt, %adg2s2b5nbt : tensor<f32>
    %advns2b5nbt = stablehlo.add %advss2b5nbt, %advgs2b5nbt : tensor<f32>
    %adbc1s2b5nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b5nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b5nbt = stablehlo.divide %admns2b5nbt, %adbc1s2b5nbt : tensor<f32>
    %advhs2b5nbt = stablehlo.divide %advns2b5nbt, %adbc2s2b5nbt : tensor<f32>
    %adlrs2b5nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b5nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b5nbt = stablehlo.sqrt %advhs2b5nbt : tensor<f32>
    %addens2b5nbt = stablehlo.add %adsqs2b5nbt, %adepss2b5nbt : tensor<f32>
    %adrats2b5nbt = stablehlo.divide %admhs2b5nbt, %addens2b5nbt : tensor<f32>
    %adsts2b5nbt = stablehlo.multiply %adlrs2b5nbt, %adrats2b5nbt : tensor<f32>
    %adsubs2b5nbt = stablehlo.subtract %s2b5nbt, %adsts2b5nbt : tensor<f32>
    %adwds2b5nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b5nbt = stablehlo.multiply %adwds2b5nbt, %adlrs2b5nbt : tensor<f32>
    %adwdps2b5nbt = stablehlo.multiply %adwdlrs2b5nbt, %s2b5nbt : tensor<f32>
    %adnews2b5nbt = stablehlo.subtract %adsubs2b5nbt, %adwdps2b5nbt : tensor<f32>
    %adb1s2b5eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob1s2b5eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admss2b5eW = stablehlo.multiply %adb1s2b5eW, %s2b5eWm : tensor<1536x384x1x1xf32>
    %admgs2b5eW = stablehlo.multiply %adob1s2b5eW, %s2b5deW : tensor<1536x384x1x1xf32>
    %admns2b5eW = stablehlo.add %admss2b5eW, %admgs2b5eW : tensor<1536x384x1x1xf32>
    %adb2s2b5eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob2s2b5eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %advss2b5eW = stablehlo.multiply %adb2s2b5eW, %s2b5eWv : tensor<1536x384x1x1xf32>
    %adg2s2b5eW = stablehlo.multiply %s2b5deW, %s2b5deW : tensor<1536x384x1x1xf32>
    %advgs2b5eW = stablehlo.multiply %adob2s2b5eW, %adg2s2b5eW : tensor<1536x384x1x1xf32>
    %advns2b5eW = stablehlo.add %advss2b5eW, %advgs2b5eW : tensor<1536x384x1x1xf32>
    %adbc1s2b5eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adbc2s2b5eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admhs2b5eW = stablehlo.divide %admns2b5eW, %adbc1s2b5eW : tensor<1536x384x1x1xf32>
    %advhs2b5eW = stablehlo.divide %advns2b5eW, %adbc2s2b5eW : tensor<1536x384x1x1xf32>
    %adlrs2b5eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adepss2b5eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adsqs2b5eW = stablehlo.sqrt %advhs2b5eW : tensor<1536x384x1x1xf32>
    %addens2b5eW = stablehlo.add %adsqs2b5eW, %adepss2b5eW : tensor<1536x384x1x1xf32>
    %adrats2b5eW = stablehlo.divide %admhs2b5eW, %addens2b5eW : tensor<1536x384x1x1xf32>
    %adsts2b5eW = stablehlo.multiply %adlrs2b5eW, %adrats2b5eW : tensor<1536x384x1x1xf32>
    %adsubs2b5eW = stablehlo.subtract %s2b5eW, %adsts2b5eW : tensor<1536x384x1x1xf32>
    %adwds2b5eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adwdlrs2b5eW = stablehlo.multiply %adwds2b5eW, %adlrs2b5eW : tensor<1536x384x1x1xf32>
    %adwdps2b5eW = stablehlo.multiply %adwdlrs2b5eW, %s2b5eW : tensor<1536x384x1x1xf32>
    %adnews2b5eW = stablehlo.subtract %adsubs2b5eW, %adwdps2b5eW : tensor<1536x384x1x1xf32>
    %adb1s2b5eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob1s2b5eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admss2b5eb = stablehlo.multiply %adb1s2b5eb, %s2b5ebm : tensor<1536xf32>
    %admgs2b5eb = stablehlo.multiply %adob1s2b5eb, %s2b5deb : tensor<1536xf32>
    %admns2b5eb = stablehlo.add %admss2b5eb, %admgs2b5eb : tensor<1536xf32>
    %adb2s2b5eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob2s2b5eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %advss2b5eb = stablehlo.multiply %adb2s2b5eb, %s2b5ebv : tensor<1536xf32>
    %adg2s2b5eb = stablehlo.multiply %s2b5deb, %s2b5deb : tensor<1536xf32>
    %advgs2b5eb = stablehlo.multiply %adob2s2b5eb, %adg2s2b5eb : tensor<1536xf32>
    %advns2b5eb = stablehlo.add %advss2b5eb, %advgs2b5eb : tensor<1536xf32>
    %adbc1s2b5eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adbc2s2b5eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admhs2b5eb = stablehlo.divide %admns2b5eb, %adbc1s2b5eb : tensor<1536xf32>
    %advhs2b5eb = stablehlo.divide %advns2b5eb, %adbc2s2b5eb : tensor<1536xf32>
    %adlrs2b5eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adepss2b5eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adsqs2b5eb = stablehlo.sqrt %advhs2b5eb : tensor<1536xf32>
    %addens2b5eb = stablehlo.add %adsqs2b5eb, %adepss2b5eb : tensor<1536xf32>
    %adrats2b5eb = stablehlo.divide %admhs2b5eb, %addens2b5eb : tensor<1536xf32>
    %adsts2b5eb = stablehlo.multiply %adlrs2b5eb, %adrats2b5eb : tensor<1536xf32>
    %adsubs2b5eb = stablehlo.subtract %s2b5eb, %adsts2b5eb : tensor<1536xf32>
    %adwds2b5eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adwdlrs2b5eb = stablehlo.multiply %adwds2b5eb, %adlrs2b5eb : tensor<1536xf32>
    %adwdps2b5eb = stablehlo.multiply %adwdlrs2b5eb, %s2b5eb : tensor<1536xf32>
    %adnews2b5eb = stablehlo.subtract %adsubs2b5eb, %adwdps2b5eb : tensor<1536xf32>
    %adb1s2b5pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob1s2b5pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admss2b5pW = stablehlo.multiply %adb1s2b5pW, %s2b5pWm : tensor<384x1536x1x1xf32>
    %admgs2b5pW = stablehlo.multiply %adob1s2b5pW, %s2b5dpW : tensor<384x1536x1x1xf32>
    %admns2b5pW = stablehlo.add %admss2b5pW, %admgs2b5pW : tensor<384x1536x1x1xf32>
    %adb2s2b5pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob2s2b5pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %advss2b5pW = stablehlo.multiply %adb2s2b5pW, %s2b5pWv : tensor<384x1536x1x1xf32>
    %adg2s2b5pW = stablehlo.multiply %s2b5dpW, %s2b5dpW : tensor<384x1536x1x1xf32>
    %advgs2b5pW = stablehlo.multiply %adob2s2b5pW, %adg2s2b5pW : tensor<384x1536x1x1xf32>
    %advns2b5pW = stablehlo.add %advss2b5pW, %advgs2b5pW : tensor<384x1536x1x1xf32>
    %adbc1s2b5pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adbc2s2b5pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admhs2b5pW = stablehlo.divide %admns2b5pW, %adbc1s2b5pW : tensor<384x1536x1x1xf32>
    %advhs2b5pW = stablehlo.divide %advns2b5pW, %adbc2s2b5pW : tensor<384x1536x1x1xf32>
    %adlrs2b5pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adepss2b5pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adsqs2b5pW = stablehlo.sqrt %advhs2b5pW : tensor<384x1536x1x1xf32>
    %addens2b5pW = stablehlo.add %adsqs2b5pW, %adepss2b5pW : tensor<384x1536x1x1xf32>
    %adrats2b5pW = stablehlo.divide %admhs2b5pW, %addens2b5pW : tensor<384x1536x1x1xf32>
    %adsts2b5pW = stablehlo.multiply %adlrs2b5pW, %adrats2b5pW : tensor<384x1536x1x1xf32>
    %adsubs2b5pW = stablehlo.subtract %s2b5pW, %adsts2b5pW : tensor<384x1536x1x1xf32>
    %adwds2b5pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adwdlrs2b5pW = stablehlo.multiply %adwds2b5pW, %adlrs2b5pW : tensor<384x1536x1x1xf32>
    %adwdps2b5pW = stablehlo.multiply %adwdlrs2b5pW, %s2b5pW : tensor<384x1536x1x1xf32>
    %adnews2b5pW = stablehlo.subtract %adsubs2b5pW, %adwdps2b5pW : tensor<384x1536x1x1xf32>
    %adb1s2b5pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b5pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b5pb = stablehlo.multiply %adb1s2b5pb, %s2b5pbm : tensor<384xf32>
    %admgs2b5pb = stablehlo.multiply %adob1s2b5pb, %s2b5dpb : tensor<384xf32>
    %admns2b5pb = stablehlo.add %admss2b5pb, %admgs2b5pb : tensor<384xf32>
    %adb2s2b5pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b5pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b5pb = stablehlo.multiply %adb2s2b5pb, %s2b5pbv : tensor<384xf32>
    %adg2s2b5pb = stablehlo.multiply %s2b5dpb, %s2b5dpb : tensor<384xf32>
    %advgs2b5pb = stablehlo.multiply %adob2s2b5pb, %adg2s2b5pb : tensor<384xf32>
    %advns2b5pb = stablehlo.add %advss2b5pb, %advgs2b5pb : tensor<384xf32>
    %adbc1s2b5pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b5pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b5pb = stablehlo.divide %admns2b5pb, %adbc1s2b5pb : tensor<384xf32>
    %advhs2b5pb = stablehlo.divide %advns2b5pb, %adbc2s2b5pb : tensor<384xf32>
    %adlrs2b5pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b5pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b5pb = stablehlo.sqrt %advhs2b5pb : tensor<384xf32>
    %addens2b5pb = stablehlo.add %adsqs2b5pb, %adepss2b5pb : tensor<384xf32>
    %adrats2b5pb = stablehlo.divide %admhs2b5pb, %addens2b5pb : tensor<384xf32>
    %adsts2b5pb = stablehlo.multiply %adlrs2b5pb, %adrats2b5pb : tensor<384xf32>
    %adsubs2b5pb = stablehlo.subtract %s2b5pb, %adsts2b5pb : tensor<384xf32>
    %adwds2b5pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b5pb = stablehlo.multiply %adwds2b5pb, %adlrs2b5pb : tensor<384xf32>
    %adwdps2b5pb = stablehlo.multiply %adwdlrs2b5pb, %s2b5pb : tensor<384xf32>
    %adnews2b5pb = stablehlo.subtract %adsubs2b5pb, %adwdps2b5pb : tensor<384xf32>
    %adb1s2b5lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b5lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b5lg = stablehlo.multiply %adb1s2b5lg, %s2b5lgm : tensor<384xf32>
    %admgs2b5lg = stablehlo.multiply %adob1s2b5lg, %s2b5dlsdg : tensor<384xf32>
    %admns2b5lg = stablehlo.add %admss2b5lg, %admgs2b5lg : tensor<384xf32>
    %adb2s2b5lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b5lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b5lg = stablehlo.multiply %adb2s2b5lg, %s2b5lgv : tensor<384xf32>
    %adg2s2b5lg = stablehlo.multiply %s2b5dlsdg, %s2b5dlsdg : tensor<384xf32>
    %advgs2b5lg = stablehlo.multiply %adob2s2b5lg, %adg2s2b5lg : tensor<384xf32>
    %advns2b5lg = stablehlo.add %advss2b5lg, %advgs2b5lg : tensor<384xf32>
    %adbc1s2b5lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b5lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b5lg = stablehlo.divide %admns2b5lg, %adbc1s2b5lg : tensor<384xf32>
    %advhs2b5lg = stablehlo.divide %advns2b5lg, %adbc2s2b5lg : tensor<384xf32>
    %adlrs2b5lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b5lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b5lg = stablehlo.sqrt %advhs2b5lg : tensor<384xf32>
    %addens2b5lg = stablehlo.add %adsqs2b5lg, %adepss2b5lg : tensor<384xf32>
    %adrats2b5lg = stablehlo.divide %admhs2b5lg, %addens2b5lg : tensor<384xf32>
    %adsts2b5lg = stablehlo.multiply %adlrs2b5lg, %adrats2b5lg : tensor<384xf32>
    %adsubs2b5lg = stablehlo.subtract %s2b5lg, %adsts2b5lg : tensor<384xf32>
    %adwds2b5lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b5lg = stablehlo.multiply %adwds2b5lg, %adlrs2b5lg : tensor<384xf32>
    %adwdps2b5lg = stablehlo.multiply %adwdlrs2b5lg, %s2b5lg : tensor<384xf32>
    %adnews2b5lg = stablehlo.subtract %adsubs2b5lg, %adwdps2b5lg : tensor<384xf32>
    %adb1s2b6dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob1s2b6dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admss2b6dW = stablehlo.multiply %adb1s2b6dW, %s2b6dWm : tensor<384x1x7x7xf32>
    %admgs2b6dW = stablehlo.multiply %adob1s2b6dW, %s2b6ddW : tensor<384x1x7x7xf32>
    %admns2b6dW = stablehlo.add %admss2b6dW, %admgs2b6dW : tensor<384x1x7x7xf32>
    %adb2s2b6dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob2s2b6dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %advss2b6dW = stablehlo.multiply %adb2s2b6dW, %s2b6dWv : tensor<384x1x7x7xf32>
    %adg2s2b6dW = stablehlo.multiply %s2b6ddW, %s2b6ddW : tensor<384x1x7x7xf32>
    %advgs2b6dW = stablehlo.multiply %adob2s2b6dW, %adg2s2b6dW : tensor<384x1x7x7xf32>
    %advns2b6dW = stablehlo.add %advss2b6dW, %advgs2b6dW : tensor<384x1x7x7xf32>
    %adbc1s2b6dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adbc2s2b6dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admhs2b6dW = stablehlo.divide %admns2b6dW, %adbc1s2b6dW : tensor<384x1x7x7xf32>
    %advhs2b6dW = stablehlo.divide %advns2b6dW, %adbc2s2b6dW : tensor<384x1x7x7xf32>
    %adlrs2b6dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adepss2b6dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adsqs2b6dW = stablehlo.sqrt %advhs2b6dW : tensor<384x1x7x7xf32>
    %addens2b6dW = stablehlo.add %adsqs2b6dW, %adepss2b6dW : tensor<384x1x7x7xf32>
    %adrats2b6dW = stablehlo.divide %admhs2b6dW, %addens2b6dW : tensor<384x1x7x7xf32>
    %adsts2b6dW = stablehlo.multiply %adlrs2b6dW, %adrats2b6dW : tensor<384x1x7x7xf32>
    %adsubs2b6dW = stablehlo.subtract %s2b6dW, %adsts2b6dW : tensor<384x1x7x7xf32>
    %adwds2b6dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adwdlrs2b6dW = stablehlo.multiply %adwds2b6dW, %adlrs2b6dW : tensor<384x1x7x7xf32>
    %adwdps2b6dW = stablehlo.multiply %adwdlrs2b6dW, %s2b6dW : tensor<384x1x7x7xf32>
    %adnews2b6dW = stablehlo.subtract %adsubs2b6dW, %adwdps2b6dW : tensor<384x1x7x7xf32>
    %adb1s2b6db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b6db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b6db = stablehlo.multiply %adb1s2b6db, %s2b6dbm : tensor<384xf32>
    %admgs2b6db = stablehlo.multiply %adob1s2b6db, %s2b6ddb : tensor<384xf32>
    %admns2b6db = stablehlo.add %admss2b6db, %admgs2b6db : tensor<384xf32>
    %adb2s2b6db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b6db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b6db = stablehlo.multiply %adb2s2b6db, %s2b6dbv : tensor<384xf32>
    %adg2s2b6db = stablehlo.multiply %s2b6ddb, %s2b6ddb : tensor<384xf32>
    %advgs2b6db = stablehlo.multiply %adob2s2b6db, %adg2s2b6db : tensor<384xf32>
    %advns2b6db = stablehlo.add %advss2b6db, %advgs2b6db : tensor<384xf32>
    %adbc1s2b6db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b6db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b6db = stablehlo.divide %admns2b6db, %adbc1s2b6db : tensor<384xf32>
    %advhs2b6db = stablehlo.divide %advns2b6db, %adbc2s2b6db : tensor<384xf32>
    %adlrs2b6db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b6db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b6db = stablehlo.sqrt %advhs2b6db : tensor<384xf32>
    %addens2b6db = stablehlo.add %adsqs2b6db, %adepss2b6db : tensor<384xf32>
    %adrats2b6db = stablehlo.divide %admhs2b6db, %addens2b6db : tensor<384xf32>
    %adsts2b6db = stablehlo.multiply %adlrs2b6db, %adrats2b6db : tensor<384xf32>
    %adsubs2b6db = stablehlo.subtract %s2b6db, %adsts2b6db : tensor<384xf32>
    %adwds2b6db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b6db = stablehlo.multiply %adwds2b6db, %adlrs2b6db : tensor<384xf32>
    %adwdps2b6db = stablehlo.multiply %adwdlrs2b6db, %s2b6db : tensor<384xf32>
    %adnews2b6db = stablehlo.subtract %adsubs2b6db, %adwdps2b6db : tensor<384xf32>
    %adb1s2b6ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b6ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b6ng = stablehlo.multiply %adb1s2b6ng, %s2b6ngm : tensor<f32>
    %admgs2b6ng = stablehlo.multiply %adob1s2b6ng, %s2b6dndg : tensor<f32>
    %admns2b6ng = stablehlo.add %admss2b6ng, %admgs2b6ng : tensor<f32>
    %adb2s2b6ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b6ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b6ng = stablehlo.multiply %adb2s2b6ng, %s2b6ngv : tensor<f32>
    %adg2s2b6ng = stablehlo.multiply %s2b6dndg, %s2b6dndg : tensor<f32>
    %advgs2b6ng = stablehlo.multiply %adob2s2b6ng, %adg2s2b6ng : tensor<f32>
    %advns2b6ng = stablehlo.add %advss2b6ng, %advgs2b6ng : tensor<f32>
    %adbc1s2b6ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b6ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b6ng = stablehlo.divide %admns2b6ng, %adbc1s2b6ng : tensor<f32>
    %advhs2b6ng = stablehlo.divide %advns2b6ng, %adbc2s2b6ng : tensor<f32>
    %adlrs2b6ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b6ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b6ng = stablehlo.sqrt %advhs2b6ng : tensor<f32>
    %addens2b6ng = stablehlo.add %adsqs2b6ng, %adepss2b6ng : tensor<f32>
    %adrats2b6ng = stablehlo.divide %admhs2b6ng, %addens2b6ng : tensor<f32>
    %adsts2b6ng = stablehlo.multiply %adlrs2b6ng, %adrats2b6ng : tensor<f32>
    %adsubs2b6ng = stablehlo.subtract %s2b6ng, %adsts2b6ng : tensor<f32>
    %adwds2b6ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b6ng = stablehlo.multiply %adwds2b6ng, %adlrs2b6ng : tensor<f32>
    %adwdps2b6ng = stablehlo.multiply %adwdlrs2b6ng, %s2b6ng : tensor<f32>
    %adnews2b6ng = stablehlo.subtract %adsubs2b6ng, %adwdps2b6ng : tensor<f32>
    %adb1s2b6nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b6nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b6nbt = stablehlo.multiply %adb1s2b6nbt, %s2b6nbtm : tensor<f32>
    %admgs2b6nbt = stablehlo.multiply %adob1s2b6nbt, %s2b6dndb : tensor<f32>
    %admns2b6nbt = stablehlo.add %admss2b6nbt, %admgs2b6nbt : tensor<f32>
    %adb2s2b6nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b6nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b6nbt = stablehlo.multiply %adb2s2b6nbt, %s2b6nbtv : tensor<f32>
    %adg2s2b6nbt = stablehlo.multiply %s2b6dndb, %s2b6dndb : tensor<f32>
    %advgs2b6nbt = stablehlo.multiply %adob2s2b6nbt, %adg2s2b6nbt : tensor<f32>
    %advns2b6nbt = stablehlo.add %advss2b6nbt, %advgs2b6nbt : tensor<f32>
    %adbc1s2b6nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b6nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b6nbt = stablehlo.divide %admns2b6nbt, %adbc1s2b6nbt : tensor<f32>
    %advhs2b6nbt = stablehlo.divide %advns2b6nbt, %adbc2s2b6nbt : tensor<f32>
    %adlrs2b6nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b6nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b6nbt = stablehlo.sqrt %advhs2b6nbt : tensor<f32>
    %addens2b6nbt = stablehlo.add %adsqs2b6nbt, %adepss2b6nbt : tensor<f32>
    %adrats2b6nbt = stablehlo.divide %admhs2b6nbt, %addens2b6nbt : tensor<f32>
    %adsts2b6nbt = stablehlo.multiply %adlrs2b6nbt, %adrats2b6nbt : tensor<f32>
    %adsubs2b6nbt = stablehlo.subtract %s2b6nbt, %adsts2b6nbt : tensor<f32>
    %adwds2b6nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b6nbt = stablehlo.multiply %adwds2b6nbt, %adlrs2b6nbt : tensor<f32>
    %adwdps2b6nbt = stablehlo.multiply %adwdlrs2b6nbt, %s2b6nbt : tensor<f32>
    %adnews2b6nbt = stablehlo.subtract %adsubs2b6nbt, %adwdps2b6nbt : tensor<f32>
    %adb1s2b6eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob1s2b6eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admss2b6eW = stablehlo.multiply %adb1s2b6eW, %s2b6eWm : tensor<1536x384x1x1xf32>
    %admgs2b6eW = stablehlo.multiply %adob1s2b6eW, %s2b6deW : tensor<1536x384x1x1xf32>
    %admns2b6eW = stablehlo.add %admss2b6eW, %admgs2b6eW : tensor<1536x384x1x1xf32>
    %adb2s2b6eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob2s2b6eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %advss2b6eW = stablehlo.multiply %adb2s2b6eW, %s2b6eWv : tensor<1536x384x1x1xf32>
    %adg2s2b6eW = stablehlo.multiply %s2b6deW, %s2b6deW : tensor<1536x384x1x1xf32>
    %advgs2b6eW = stablehlo.multiply %adob2s2b6eW, %adg2s2b6eW : tensor<1536x384x1x1xf32>
    %advns2b6eW = stablehlo.add %advss2b6eW, %advgs2b6eW : tensor<1536x384x1x1xf32>
    %adbc1s2b6eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adbc2s2b6eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admhs2b6eW = stablehlo.divide %admns2b6eW, %adbc1s2b6eW : tensor<1536x384x1x1xf32>
    %advhs2b6eW = stablehlo.divide %advns2b6eW, %adbc2s2b6eW : tensor<1536x384x1x1xf32>
    %adlrs2b6eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adepss2b6eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adsqs2b6eW = stablehlo.sqrt %advhs2b6eW : tensor<1536x384x1x1xf32>
    %addens2b6eW = stablehlo.add %adsqs2b6eW, %adepss2b6eW : tensor<1536x384x1x1xf32>
    %adrats2b6eW = stablehlo.divide %admhs2b6eW, %addens2b6eW : tensor<1536x384x1x1xf32>
    %adsts2b6eW = stablehlo.multiply %adlrs2b6eW, %adrats2b6eW : tensor<1536x384x1x1xf32>
    %adsubs2b6eW = stablehlo.subtract %s2b6eW, %adsts2b6eW : tensor<1536x384x1x1xf32>
    %adwds2b6eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adwdlrs2b6eW = stablehlo.multiply %adwds2b6eW, %adlrs2b6eW : tensor<1536x384x1x1xf32>
    %adwdps2b6eW = stablehlo.multiply %adwdlrs2b6eW, %s2b6eW : tensor<1536x384x1x1xf32>
    %adnews2b6eW = stablehlo.subtract %adsubs2b6eW, %adwdps2b6eW : tensor<1536x384x1x1xf32>
    %adb1s2b6eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob1s2b6eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admss2b6eb = stablehlo.multiply %adb1s2b6eb, %s2b6ebm : tensor<1536xf32>
    %admgs2b6eb = stablehlo.multiply %adob1s2b6eb, %s2b6deb : tensor<1536xf32>
    %admns2b6eb = stablehlo.add %admss2b6eb, %admgs2b6eb : tensor<1536xf32>
    %adb2s2b6eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob2s2b6eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %advss2b6eb = stablehlo.multiply %adb2s2b6eb, %s2b6ebv : tensor<1536xf32>
    %adg2s2b6eb = stablehlo.multiply %s2b6deb, %s2b6deb : tensor<1536xf32>
    %advgs2b6eb = stablehlo.multiply %adob2s2b6eb, %adg2s2b6eb : tensor<1536xf32>
    %advns2b6eb = stablehlo.add %advss2b6eb, %advgs2b6eb : tensor<1536xf32>
    %adbc1s2b6eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adbc2s2b6eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admhs2b6eb = stablehlo.divide %admns2b6eb, %adbc1s2b6eb : tensor<1536xf32>
    %advhs2b6eb = stablehlo.divide %advns2b6eb, %adbc2s2b6eb : tensor<1536xf32>
    %adlrs2b6eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adepss2b6eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adsqs2b6eb = stablehlo.sqrt %advhs2b6eb : tensor<1536xf32>
    %addens2b6eb = stablehlo.add %adsqs2b6eb, %adepss2b6eb : tensor<1536xf32>
    %adrats2b6eb = stablehlo.divide %admhs2b6eb, %addens2b6eb : tensor<1536xf32>
    %adsts2b6eb = stablehlo.multiply %adlrs2b6eb, %adrats2b6eb : tensor<1536xf32>
    %adsubs2b6eb = stablehlo.subtract %s2b6eb, %adsts2b6eb : tensor<1536xf32>
    %adwds2b6eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adwdlrs2b6eb = stablehlo.multiply %adwds2b6eb, %adlrs2b6eb : tensor<1536xf32>
    %adwdps2b6eb = stablehlo.multiply %adwdlrs2b6eb, %s2b6eb : tensor<1536xf32>
    %adnews2b6eb = stablehlo.subtract %adsubs2b6eb, %adwdps2b6eb : tensor<1536xf32>
    %adb1s2b6pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob1s2b6pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admss2b6pW = stablehlo.multiply %adb1s2b6pW, %s2b6pWm : tensor<384x1536x1x1xf32>
    %admgs2b6pW = stablehlo.multiply %adob1s2b6pW, %s2b6dpW : tensor<384x1536x1x1xf32>
    %admns2b6pW = stablehlo.add %admss2b6pW, %admgs2b6pW : tensor<384x1536x1x1xf32>
    %adb2s2b6pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob2s2b6pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %advss2b6pW = stablehlo.multiply %adb2s2b6pW, %s2b6pWv : tensor<384x1536x1x1xf32>
    %adg2s2b6pW = stablehlo.multiply %s2b6dpW, %s2b6dpW : tensor<384x1536x1x1xf32>
    %advgs2b6pW = stablehlo.multiply %adob2s2b6pW, %adg2s2b6pW : tensor<384x1536x1x1xf32>
    %advns2b6pW = stablehlo.add %advss2b6pW, %advgs2b6pW : tensor<384x1536x1x1xf32>
    %adbc1s2b6pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adbc2s2b6pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admhs2b6pW = stablehlo.divide %admns2b6pW, %adbc1s2b6pW : tensor<384x1536x1x1xf32>
    %advhs2b6pW = stablehlo.divide %advns2b6pW, %adbc2s2b6pW : tensor<384x1536x1x1xf32>
    %adlrs2b6pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adepss2b6pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adsqs2b6pW = stablehlo.sqrt %advhs2b6pW : tensor<384x1536x1x1xf32>
    %addens2b6pW = stablehlo.add %adsqs2b6pW, %adepss2b6pW : tensor<384x1536x1x1xf32>
    %adrats2b6pW = stablehlo.divide %admhs2b6pW, %addens2b6pW : tensor<384x1536x1x1xf32>
    %adsts2b6pW = stablehlo.multiply %adlrs2b6pW, %adrats2b6pW : tensor<384x1536x1x1xf32>
    %adsubs2b6pW = stablehlo.subtract %s2b6pW, %adsts2b6pW : tensor<384x1536x1x1xf32>
    %adwds2b6pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adwdlrs2b6pW = stablehlo.multiply %adwds2b6pW, %adlrs2b6pW : tensor<384x1536x1x1xf32>
    %adwdps2b6pW = stablehlo.multiply %adwdlrs2b6pW, %s2b6pW : tensor<384x1536x1x1xf32>
    %adnews2b6pW = stablehlo.subtract %adsubs2b6pW, %adwdps2b6pW : tensor<384x1536x1x1xf32>
    %adb1s2b6pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b6pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b6pb = stablehlo.multiply %adb1s2b6pb, %s2b6pbm : tensor<384xf32>
    %admgs2b6pb = stablehlo.multiply %adob1s2b6pb, %s2b6dpb : tensor<384xf32>
    %admns2b6pb = stablehlo.add %admss2b6pb, %admgs2b6pb : tensor<384xf32>
    %adb2s2b6pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b6pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b6pb = stablehlo.multiply %adb2s2b6pb, %s2b6pbv : tensor<384xf32>
    %adg2s2b6pb = stablehlo.multiply %s2b6dpb, %s2b6dpb : tensor<384xf32>
    %advgs2b6pb = stablehlo.multiply %adob2s2b6pb, %adg2s2b6pb : tensor<384xf32>
    %advns2b6pb = stablehlo.add %advss2b6pb, %advgs2b6pb : tensor<384xf32>
    %adbc1s2b6pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b6pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b6pb = stablehlo.divide %admns2b6pb, %adbc1s2b6pb : tensor<384xf32>
    %advhs2b6pb = stablehlo.divide %advns2b6pb, %adbc2s2b6pb : tensor<384xf32>
    %adlrs2b6pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b6pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b6pb = stablehlo.sqrt %advhs2b6pb : tensor<384xf32>
    %addens2b6pb = stablehlo.add %adsqs2b6pb, %adepss2b6pb : tensor<384xf32>
    %adrats2b6pb = stablehlo.divide %admhs2b6pb, %addens2b6pb : tensor<384xf32>
    %adsts2b6pb = stablehlo.multiply %adlrs2b6pb, %adrats2b6pb : tensor<384xf32>
    %adsubs2b6pb = stablehlo.subtract %s2b6pb, %adsts2b6pb : tensor<384xf32>
    %adwds2b6pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b6pb = stablehlo.multiply %adwds2b6pb, %adlrs2b6pb : tensor<384xf32>
    %adwdps2b6pb = stablehlo.multiply %adwdlrs2b6pb, %s2b6pb : tensor<384xf32>
    %adnews2b6pb = stablehlo.subtract %adsubs2b6pb, %adwdps2b6pb : tensor<384xf32>
    %adb1s2b6lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b6lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b6lg = stablehlo.multiply %adb1s2b6lg, %s2b6lgm : tensor<384xf32>
    %admgs2b6lg = stablehlo.multiply %adob1s2b6lg, %s2b6dlsdg : tensor<384xf32>
    %admns2b6lg = stablehlo.add %admss2b6lg, %admgs2b6lg : tensor<384xf32>
    %adb2s2b6lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b6lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b6lg = stablehlo.multiply %adb2s2b6lg, %s2b6lgv : tensor<384xf32>
    %adg2s2b6lg = stablehlo.multiply %s2b6dlsdg, %s2b6dlsdg : tensor<384xf32>
    %advgs2b6lg = stablehlo.multiply %adob2s2b6lg, %adg2s2b6lg : tensor<384xf32>
    %advns2b6lg = stablehlo.add %advss2b6lg, %advgs2b6lg : tensor<384xf32>
    %adbc1s2b6lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b6lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b6lg = stablehlo.divide %admns2b6lg, %adbc1s2b6lg : tensor<384xf32>
    %advhs2b6lg = stablehlo.divide %advns2b6lg, %adbc2s2b6lg : tensor<384xf32>
    %adlrs2b6lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b6lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b6lg = stablehlo.sqrt %advhs2b6lg : tensor<384xf32>
    %addens2b6lg = stablehlo.add %adsqs2b6lg, %adepss2b6lg : tensor<384xf32>
    %adrats2b6lg = stablehlo.divide %admhs2b6lg, %addens2b6lg : tensor<384xf32>
    %adsts2b6lg = stablehlo.multiply %adlrs2b6lg, %adrats2b6lg : tensor<384xf32>
    %adsubs2b6lg = stablehlo.subtract %s2b6lg, %adsts2b6lg : tensor<384xf32>
    %adwds2b6lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b6lg = stablehlo.multiply %adwds2b6lg, %adlrs2b6lg : tensor<384xf32>
    %adwdps2b6lg = stablehlo.multiply %adwdlrs2b6lg, %s2b6lg : tensor<384xf32>
    %adnews2b6lg = stablehlo.subtract %adsubs2b6lg, %adwdps2b6lg : tensor<384xf32>
    %adb1s2b7dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob1s2b7dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admss2b7dW = stablehlo.multiply %adb1s2b7dW, %s2b7dWm : tensor<384x1x7x7xf32>
    %admgs2b7dW = stablehlo.multiply %adob1s2b7dW, %s2b7ddW : tensor<384x1x7x7xf32>
    %admns2b7dW = stablehlo.add %admss2b7dW, %admgs2b7dW : tensor<384x1x7x7xf32>
    %adb2s2b7dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob2s2b7dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %advss2b7dW = stablehlo.multiply %adb2s2b7dW, %s2b7dWv : tensor<384x1x7x7xf32>
    %adg2s2b7dW = stablehlo.multiply %s2b7ddW, %s2b7ddW : tensor<384x1x7x7xf32>
    %advgs2b7dW = stablehlo.multiply %adob2s2b7dW, %adg2s2b7dW : tensor<384x1x7x7xf32>
    %advns2b7dW = stablehlo.add %advss2b7dW, %advgs2b7dW : tensor<384x1x7x7xf32>
    %adbc1s2b7dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adbc2s2b7dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admhs2b7dW = stablehlo.divide %admns2b7dW, %adbc1s2b7dW : tensor<384x1x7x7xf32>
    %advhs2b7dW = stablehlo.divide %advns2b7dW, %adbc2s2b7dW : tensor<384x1x7x7xf32>
    %adlrs2b7dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adepss2b7dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adsqs2b7dW = stablehlo.sqrt %advhs2b7dW : tensor<384x1x7x7xf32>
    %addens2b7dW = stablehlo.add %adsqs2b7dW, %adepss2b7dW : tensor<384x1x7x7xf32>
    %adrats2b7dW = stablehlo.divide %admhs2b7dW, %addens2b7dW : tensor<384x1x7x7xf32>
    %adsts2b7dW = stablehlo.multiply %adlrs2b7dW, %adrats2b7dW : tensor<384x1x7x7xf32>
    %adsubs2b7dW = stablehlo.subtract %s2b7dW, %adsts2b7dW : tensor<384x1x7x7xf32>
    %adwds2b7dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adwdlrs2b7dW = stablehlo.multiply %adwds2b7dW, %adlrs2b7dW : tensor<384x1x7x7xf32>
    %adwdps2b7dW = stablehlo.multiply %adwdlrs2b7dW, %s2b7dW : tensor<384x1x7x7xf32>
    %adnews2b7dW = stablehlo.subtract %adsubs2b7dW, %adwdps2b7dW : tensor<384x1x7x7xf32>
    %adb1s2b7db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b7db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b7db = stablehlo.multiply %adb1s2b7db, %s2b7dbm : tensor<384xf32>
    %admgs2b7db = stablehlo.multiply %adob1s2b7db, %s2b7ddb : tensor<384xf32>
    %admns2b7db = stablehlo.add %admss2b7db, %admgs2b7db : tensor<384xf32>
    %adb2s2b7db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b7db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b7db = stablehlo.multiply %adb2s2b7db, %s2b7dbv : tensor<384xf32>
    %adg2s2b7db = stablehlo.multiply %s2b7ddb, %s2b7ddb : tensor<384xf32>
    %advgs2b7db = stablehlo.multiply %adob2s2b7db, %adg2s2b7db : tensor<384xf32>
    %advns2b7db = stablehlo.add %advss2b7db, %advgs2b7db : tensor<384xf32>
    %adbc1s2b7db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b7db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b7db = stablehlo.divide %admns2b7db, %adbc1s2b7db : tensor<384xf32>
    %advhs2b7db = stablehlo.divide %advns2b7db, %adbc2s2b7db : tensor<384xf32>
    %adlrs2b7db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b7db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b7db = stablehlo.sqrt %advhs2b7db : tensor<384xf32>
    %addens2b7db = stablehlo.add %adsqs2b7db, %adepss2b7db : tensor<384xf32>
    %adrats2b7db = stablehlo.divide %admhs2b7db, %addens2b7db : tensor<384xf32>
    %adsts2b7db = stablehlo.multiply %adlrs2b7db, %adrats2b7db : tensor<384xf32>
    %adsubs2b7db = stablehlo.subtract %s2b7db, %adsts2b7db : tensor<384xf32>
    %adwds2b7db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b7db = stablehlo.multiply %adwds2b7db, %adlrs2b7db : tensor<384xf32>
    %adwdps2b7db = stablehlo.multiply %adwdlrs2b7db, %s2b7db : tensor<384xf32>
    %adnews2b7db = stablehlo.subtract %adsubs2b7db, %adwdps2b7db : tensor<384xf32>
    %adb1s2b7ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b7ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b7ng = stablehlo.multiply %adb1s2b7ng, %s2b7ngm : tensor<f32>
    %admgs2b7ng = stablehlo.multiply %adob1s2b7ng, %s2b7dndg : tensor<f32>
    %admns2b7ng = stablehlo.add %admss2b7ng, %admgs2b7ng : tensor<f32>
    %adb2s2b7ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b7ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b7ng = stablehlo.multiply %adb2s2b7ng, %s2b7ngv : tensor<f32>
    %adg2s2b7ng = stablehlo.multiply %s2b7dndg, %s2b7dndg : tensor<f32>
    %advgs2b7ng = stablehlo.multiply %adob2s2b7ng, %adg2s2b7ng : tensor<f32>
    %advns2b7ng = stablehlo.add %advss2b7ng, %advgs2b7ng : tensor<f32>
    %adbc1s2b7ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b7ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b7ng = stablehlo.divide %admns2b7ng, %adbc1s2b7ng : tensor<f32>
    %advhs2b7ng = stablehlo.divide %advns2b7ng, %adbc2s2b7ng : tensor<f32>
    %adlrs2b7ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b7ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b7ng = stablehlo.sqrt %advhs2b7ng : tensor<f32>
    %addens2b7ng = stablehlo.add %adsqs2b7ng, %adepss2b7ng : tensor<f32>
    %adrats2b7ng = stablehlo.divide %admhs2b7ng, %addens2b7ng : tensor<f32>
    %adsts2b7ng = stablehlo.multiply %adlrs2b7ng, %adrats2b7ng : tensor<f32>
    %adsubs2b7ng = stablehlo.subtract %s2b7ng, %adsts2b7ng : tensor<f32>
    %adwds2b7ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b7ng = stablehlo.multiply %adwds2b7ng, %adlrs2b7ng : tensor<f32>
    %adwdps2b7ng = stablehlo.multiply %adwdlrs2b7ng, %s2b7ng : tensor<f32>
    %adnews2b7ng = stablehlo.subtract %adsubs2b7ng, %adwdps2b7ng : tensor<f32>
    %adb1s2b7nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b7nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b7nbt = stablehlo.multiply %adb1s2b7nbt, %s2b7nbtm : tensor<f32>
    %admgs2b7nbt = stablehlo.multiply %adob1s2b7nbt, %s2b7dndb : tensor<f32>
    %admns2b7nbt = stablehlo.add %admss2b7nbt, %admgs2b7nbt : tensor<f32>
    %adb2s2b7nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b7nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b7nbt = stablehlo.multiply %adb2s2b7nbt, %s2b7nbtv : tensor<f32>
    %adg2s2b7nbt = stablehlo.multiply %s2b7dndb, %s2b7dndb : tensor<f32>
    %advgs2b7nbt = stablehlo.multiply %adob2s2b7nbt, %adg2s2b7nbt : tensor<f32>
    %advns2b7nbt = stablehlo.add %advss2b7nbt, %advgs2b7nbt : tensor<f32>
    %adbc1s2b7nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b7nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b7nbt = stablehlo.divide %admns2b7nbt, %adbc1s2b7nbt : tensor<f32>
    %advhs2b7nbt = stablehlo.divide %advns2b7nbt, %adbc2s2b7nbt : tensor<f32>
    %adlrs2b7nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b7nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b7nbt = stablehlo.sqrt %advhs2b7nbt : tensor<f32>
    %addens2b7nbt = stablehlo.add %adsqs2b7nbt, %adepss2b7nbt : tensor<f32>
    %adrats2b7nbt = stablehlo.divide %admhs2b7nbt, %addens2b7nbt : tensor<f32>
    %adsts2b7nbt = stablehlo.multiply %adlrs2b7nbt, %adrats2b7nbt : tensor<f32>
    %adsubs2b7nbt = stablehlo.subtract %s2b7nbt, %adsts2b7nbt : tensor<f32>
    %adwds2b7nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b7nbt = stablehlo.multiply %adwds2b7nbt, %adlrs2b7nbt : tensor<f32>
    %adwdps2b7nbt = stablehlo.multiply %adwdlrs2b7nbt, %s2b7nbt : tensor<f32>
    %adnews2b7nbt = stablehlo.subtract %adsubs2b7nbt, %adwdps2b7nbt : tensor<f32>
    %adb1s2b7eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob1s2b7eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admss2b7eW = stablehlo.multiply %adb1s2b7eW, %s2b7eWm : tensor<1536x384x1x1xf32>
    %admgs2b7eW = stablehlo.multiply %adob1s2b7eW, %s2b7deW : tensor<1536x384x1x1xf32>
    %admns2b7eW = stablehlo.add %admss2b7eW, %admgs2b7eW : tensor<1536x384x1x1xf32>
    %adb2s2b7eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob2s2b7eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %advss2b7eW = stablehlo.multiply %adb2s2b7eW, %s2b7eWv : tensor<1536x384x1x1xf32>
    %adg2s2b7eW = stablehlo.multiply %s2b7deW, %s2b7deW : tensor<1536x384x1x1xf32>
    %advgs2b7eW = stablehlo.multiply %adob2s2b7eW, %adg2s2b7eW : tensor<1536x384x1x1xf32>
    %advns2b7eW = stablehlo.add %advss2b7eW, %advgs2b7eW : tensor<1536x384x1x1xf32>
    %adbc1s2b7eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adbc2s2b7eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admhs2b7eW = stablehlo.divide %admns2b7eW, %adbc1s2b7eW : tensor<1536x384x1x1xf32>
    %advhs2b7eW = stablehlo.divide %advns2b7eW, %adbc2s2b7eW : tensor<1536x384x1x1xf32>
    %adlrs2b7eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adepss2b7eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adsqs2b7eW = stablehlo.sqrt %advhs2b7eW : tensor<1536x384x1x1xf32>
    %addens2b7eW = stablehlo.add %adsqs2b7eW, %adepss2b7eW : tensor<1536x384x1x1xf32>
    %adrats2b7eW = stablehlo.divide %admhs2b7eW, %addens2b7eW : tensor<1536x384x1x1xf32>
    %adsts2b7eW = stablehlo.multiply %adlrs2b7eW, %adrats2b7eW : tensor<1536x384x1x1xf32>
    %adsubs2b7eW = stablehlo.subtract %s2b7eW, %adsts2b7eW : tensor<1536x384x1x1xf32>
    %adwds2b7eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adwdlrs2b7eW = stablehlo.multiply %adwds2b7eW, %adlrs2b7eW : tensor<1536x384x1x1xf32>
    %adwdps2b7eW = stablehlo.multiply %adwdlrs2b7eW, %s2b7eW : tensor<1536x384x1x1xf32>
    %adnews2b7eW = stablehlo.subtract %adsubs2b7eW, %adwdps2b7eW : tensor<1536x384x1x1xf32>
    %adb1s2b7eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob1s2b7eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admss2b7eb = stablehlo.multiply %adb1s2b7eb, %s2b7ebm : tensor<1536xf32>
    %admgs2b7eb = stablehlo.multiply %adob1s2b7eb, %s2b7deb : tensor<1536xf32>
    %admns2b7eb = stablehlo.add %admss2b7eb, %admgs2b7eb : tensor<1536xf32>
    %adb2s2b7eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob2s2b7eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %advss2b7eb = stablehlo.multiply %adb2s2b7eb, %s2b7ebv : tensor<1536xf32>
    %adg2s2b7eb = stablehlo.multiply %s2b7deb, %s2b7deb : tensor<1536xf32>
    %advgs2b7eb = stablehlo.multiply %adob2s2b7eb, %adg2s2b7eb : tensor<1536xf32>
    %advns2b7eb = stablehlo.add %advss2b7eb, %advgs2b7eb : tensor<1536xf32>
    %adbc1s2b7eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adbc2s2b7eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admhs2b7eb = stablehlo.divide %admns2b7eb, %adbc1s2b7eb : tensor<1536xf32>
    %advhs2b7eb = stablehlo.divide %advns2b7eb, %adbc2s2b7eb : tensor<1536xf32>
    %adlrs2b7eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adepss2b7eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adsqs2b7eb = stablehlo.sqrt %advhs2b7eb : tensor<1536xf32>
    %addens2b7eb = stablehlo.add %adsqs2b7eb, %adepss2b7eb : tensor<1536xf32>
    %adrats2b7eb = stablehlo.divide %admhs2b7eb, %addens2b7eb : tensor<1536xf32>
    %adsts2b7eb = stablehlo.multiply %adlrs2b7eb, %adrats2b7eb : tensor<1536xf32>
    %adsubs2b7eb = stablehlo.subtract %s2b7eb, %adsts2b7eb : tensor<1536xf32>
    %adwds2b7eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adwdlrs2b7eb = stablehlo.multiply %adwds2b7eb, %adlrs2b7eb : tensor<1536xf32>
    %adwdps2b7eb = stablehlo.multiply %adwdlrs2b7eb, %s2b7eb : tensor<1536xf32>
    %adnews2b7eb = stablehlo.subtract %adsubs2b7eb, %adwdps2b7eb : tensor<1536xf32>
    %adb1s2b7pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob1s2b7pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admss2b7pW = stablehlo.multiply %adb1s2b7pW, %s2b7pWm : tensor<384x1536x1x1xf32>
    %admgs2b7pW = stablehlo.multiply %adob1s2b7pW, %s2b7dpW : tensor<384x1536x1x1xf32>
    %admns2b7pW = stablehlo.add %admss2b7pW, %admgs2b7pW : tensor<384x1536x1x1xf32>
    %adb2s2b7pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob2s2b7pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %advss2b7pW = stablehlo.multiply %adb2s2b7pW, %s2b7pWv : tensor<384x1536x1x1xf32>
    %adg2s2b7pW = stablehlo.multiply %s2b7dpW, %s2b7dpW : tensor<384x1536x1x1xf32>
    %advgs2b7pW = stablehlo.multiply %adob2s2b7pW, %adg2s2b7pW : tensor<384x1536x1x1xf32>
    %advns2b7pW = stablehlo.add %advss2b7pW, %advgs2b7pW : tensor<384x1536x1x1xf32>
    %adbc1s2b7pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adbc2s2b7pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admhs2b7pW = stablehlo.divide %admns2b7pW, %adbc1s2b7pW : tensor<384x1536x1x1xf32>
    %advhs2b7pW = stablehlo.divide %advns2b7pW, %adbc2s2b7pW : tensor<384x1536x1x1xf32>
    %adlrs2b7pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adepss2b7pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adsqs2b7pW = stablehlo.sqrt %advhs2b7pW : tensor<384x1536x1x1xf32>
    %addens2b7pW = stablehlo.add %adsqs2b7pW, %adepss2b7pW : tensor<384x1536x1x1xf32>
    %adrats2b7pW = stablehlo.divide %admhs2b7pW, %addens2b7pW : tensor<384x1536x1x1xf32>
    %adsts2b7pW = stablehlo.multiply %adlrs2b7pW, %adrats2b7pW : tensor<384x1536x1x1xf32>
    %adsubs2b7pW = stablehlo.subtract %s2b7pW, %adsts2b7pW : tensor<384x1536x1x1xf32>
    %adwds2b7pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adwdlrs2b7pW = stablehlo.multiply %adwds2b7pW, %adlrs2b7pW : tensor<384x1536x1x1xf32>
    %adwdps2b7pW = stablehlo.multiply %adwdlrs2b7pW, %s2b7pW : tensor<384x1536x1x1xf32>
    %adnews2b7pW = stablehlo.subtract %adsubs2b7pW, %adwdps2b7pW : tensor<384x1536x1x1xf32>
    %adb1s2b7pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b7pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b7pb = stablehlo.multiply %adb1s2b7pb, %s2b7pbm : tensor<384xf32>
    %admgs2b7pb = stablehlo.multiply %adob1s2b7pb, %s2b7dpb : tensor<384xf32>
    %admns2b7pb = stablehlo.add %admss2b7pb, %admgs2b7pb : tensor<384xf32>
    %adb2s2b7pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b7pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b7pb = stablehlo.multiply %adb2s2b7pb, %s2b7pbv : tensor<384xf32>
    %adg2s2b7pb = stablehlo.multiply %s2b7dpb, %s2b7dpb : tensor<384xf32>
    %advgs2b7pb = stablehlo.multiply %adob2s2b7pb, %adg2s2b7pb : tensor<384xf32>
    %advns2b7pb = stablehlo.add %advss2b7pb, %advgs2b7pb : tensor<384xf32>
    %adbc1s2b7pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b7pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b7pb = stablehlo.divide %admns2b7pb, %adbc1s2b7pb : tensor<384xf32>
    %advhs2b7pb = stablehlo.divide %advns2b7pb, %adbc2s2b7pb : tensor<384xf32>
    %adlrs2b7pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b7pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b7pb = stablehlo.sqrt %advhs2b7pb : tensor<384xf32>
    %addens2b7pb = stablehlo.add %adsqs2b7pb, %adepss2b7pb : tensor<384xf32>
    %adrats2b7pb = stablehlo.divide %admhs2b7pb, %addens2b7pb : tensor<384xf32>
    %adsts2b7pb = stablehlo.multiply %adlrs2b7pb, %adrats2b7pb : tensor<384xf32>
    %adsubs2b7pb = stablehlo.subtract %s2b7pb, %adsts2b7pb : tensor<384xf32>
    %adwds2b7pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b7pb = stablehlo.multiply %adwds2b7pb, %adlrs2b7pb : tensor<384xf32>
    %adwdps2b7pb = stablehlo.multiply %adwdlrs2b7pb, %s2b7pb : tensor<384xf32>
    %adnews2b7pb = stablehlo.subtract %adsubs2b7pb, %adwdps2b7pb : tensor<384xf32>
    %adb1s2b7lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b7lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b7lg = stablehlo.multiply %adb1s2b7lg, %s2b7lgm : tensor<384xf32>
    %admgs2b7lg = stablehlo.multiply %adob1s2b7lg, %s2b7dlsdg : tensor<384xf32>
    %admns2b7lg = stablehlo.add %admss2b7lg, %admgs2b7lg : tensor<384xf32>
    %adb2s2b7lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b7lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b7lg = stablehlo.multiply %adb2s2b7lg, %s2b7lgv : tensor<384xf32>
    %adg2s2b7lg = stablehlo.multiply %s2b7dlsdg, %s2b7dlsdg : tensor<384xf32>
    %advgs2b7lg = stablehlo.multiply %adob2s2b7lg, %adg2s2b7lg : tensor<384xf32>
    %advns2b7lg = stablehlo.add %advss2b7lg, %advgs2b7lg : tensor<384xf32>
    %adbc1s2b7lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b7lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b7lg = stablehlo.divide %admns2b7lg, %adbc1s2b7lg : tensor<384xf32>
    %advhs2b7lg = stablehlo.divide %advns2b7lg, %adbc2s2b7lg : tensor<384xf32>
    %adlrs2b7lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b7lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b7lg = stablehlo.sqrt %advhs2b7lg : tensor<384xf32>
    %addens2b7lg = stablehlo.add %adsqs2b7lg, %adepss2b7lg : tensor<384xf32>
    %adrats2b7lg = stablehlo.divide %admhs2b7lg, %addens2b7lg : tensor<384xf32>
    %adsts2b7lg = stablehlo.multiply %adlrs2b7lg, %adrats2b7lg : tensor<384xf32>
    %adsubs2b7lg = stablehlo.subtract %s2b7lg, %adsts2b7lg : tensor<384xf32>
    %adwds2b7lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b7lg = stablehlo.multiply %adwds2b7lg, %adlrs2b7lg : tensor<384xf32>
    %adwdps2b7lg = stablehlo.multiply %adwdlrs2b7lg, %s2b7lg : tensor<384xf32>
    %adnews2b7lg = stablehlo.subtract %adsubs2b7lg, %adwdps2b7lg : tensor<384xf32>
    %adb1s2b8dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob1s2b8dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admss2b8dW = stablehlo.multiply %adb1s2b8dW, %s2b8dWm : tensor<384x1x7x7xf32>
    %admgs2b8dW = stablehlo.multiply %adob1s2b8dW, %s2b8ddW : tensor<384x1x7x7xf32>
    %admns2b8dW = stablehlo.add %admss2b8dW, %admgs2b8dW : tensor<384x1x7x7xf32>
    %adb2s2b8dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adob2s2b8dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %advss2b8dW = stablehlo.multiply %adb2s2b8dW, %s2b8dWv : tensor<384x1x7x7xf32>
    %adg2s2b8dW = stablehlo.multiply %s2b8ddW, %s2b8ddW : tensor<384x1x7x7xf32>
    %advgs2b8dW = stablehlo.multiply %adob2s2b8dW, %adg2s2b8dW : tensor<384x1x7x7xf32>
    %advns2b8dW = stablehlo.add %advss2b8dW, %advgs2b8dW : tensor<384x1x7x7xf32>
    %adbc1s2b8dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adbc2s2b8dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %admhs2b8dW = stablehlo.divide %admns2b8dW, %adbc1s2b8dW : tensor<384x1x7x7xf32>
    %advhs2b8dW = stablehlo.divide %advns2b8dW, %adbc2s2b8dW : tensor<384x1x7x7xf32>
    %adlrs2b8dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adepss2b8dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adsqs2b8dW = stablehlo.sqrt %advhs2b8dW : tensor<384x1x7x7xf32>
    %addens2b8dW = stablehlo.add %adsqs2b8dW, %adepss2b8dW : tensor<384x1x7x7xf32>
    %adrats2b8dW = stablehlo.divide %admhs2b8dW, %addens2b8dW : tensor<384x1x7x7xf32>
    %adsts2b8dW = stablehlo.multiply %adlrs2b8dW, %adrats2b8dW : tensor<384x1x7x7xf32>
    %adsubs2b8dW = stablehlo.subtract %s2b8dW, %adsts2b8dW : tensor<384x1x7x7xf32>
    %adwds2b8dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %adwdlrs2b8dW = stablehlo.multiply %adwds2b8dW, %adlrs2b8dW : tensor<384x1x7x7xf32>
    %adwdps2b8dW = stablehlo.multiply %adwdlrs2b8dW, %s2b8dW : tensor<384x1x7x7xf32>
    %adnews2b8dW = stablehlo.subtract %adsubs2b8dW, %adwdps2b8dW : tensor<384x1x7x7xf32>
    %adb1s2b8db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b8db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b8db = stablehlo.multiply %adb1s2b8db, %s2b8dbm : tensor<384xf32>
    %admgs2b8db = stablehlo.multiply %adob1s2b8db, %s2b8ddb : tensor<384xf32>
    %admns2b8db = stablehlo.add %admss2b8db, %admgs2b8db : tensor<384xf32>
    %adb2s2b8db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b8db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b8db = stablehlo.multiply %adb2s2b8db, %s2b8dbv : tensor<384xf32>
    %adg2s2b8db = stablehlo.multiply %s2b8ddb, %s2b8ddb : tensor<384xf32>
    %advgs2b8db = stablehlo.multiply %adob2s2b8db, %adg2s2b8db : tensor<384xf32>
    %advns2b8db = stablehlo.add %advss2b8db, %advgs2b8db : tensor<384xf32>
    %adbc1s2b8db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b8db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b8db = stablehlo.divide %admns2b8db, %adbc1s2b8db : tensor<384xf32>
    %advhs2b8db = stablehlo.divide %advns2b8db, %adbc2s2b8db : tensor<384xf32>
    %adlrs2b8db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b8db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b8db = stablehlo.sqrt %advhs2b8db : tensor<384xf32>
    %addens2b8db = stablehlo.add %adsqs2b8db, %adepss2b8db : tensor<384xf32>
    %adrats2b8db = stablehlo.divide %admhs2b8db, %addens2b8db : tensor<384xf32>
    %adsts2b8db = stablehlo.multiply %adlrs2b8db, %adrats2b8db : tensor<384xf32>
    %adsubs2b8db = stablehlo.subtract %s2b8db, %adsts2b8db : tensor<384xf32>
    %adwds2b8db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b8db = stablehlo.multiply %adwds2b8db, %adlrs2b8db : tensor<384xf32>
    %adwdps2b8db = stablehlo.multiply %adwdlrs2b8db, %s2b8db : tensor<384xf32>
    %adnews2b8db = stablehlo.subtract %adsubs2b8db, %adwdps2b8db : tensor<384xf32>
    %adb1s2b8ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b8ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b8ng = stablehlo.multiply %adb1s2b8ng, %s2b8ngm : tensor<f32>
    %admgs2b8ng = stablehlo.multiply %adob1s2b8ng, %s2b8dndg : tensor<f32>
    %admns2b8ng = stablehlo.add %admss2b8ng, %admgs2b8ng : tensor<f32>
    %adb2s2b8ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b8ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b8ng = stablehlo.multiply %adb2s2b8ng, %s2b8ngv : tensor<f32>
    %adg2s2b8ng = stablehlo.multiply %s2b8dndg, %s2b8dndg : tensor<f32>
    %advgs2b8ng = stablehlo.multiply %adob2s2b8ng, %adg2s2b8ng : tensor<f32>
    %advns2b8ng = stablehlo.add %advss2b8ng, %advgs2b8ng : tensor<f32>
    %adbc1s2b8ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b8ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b8ng = stablehlo.divide %admns2b8ng, %adbc1s2b8ng : tensor<f32>
    %advhs2b8ng = stablehlo.divide %advns2b8ng, %adbc2s2b8ng : tensor<f32>
    %adlrs2b8ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b8ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b8ng = stablehlo.sqrt %advhs2b8ng : tensor<f32>
    %addens2b8ng = stablehlo.add %adsqs2b8ng, %adepss2b8ng : tensor<f32>
    %adrats2b8ng = stablehlo.divide %admhs2b8ng, %addens2b8ng : tensor<f32>
    %adsts2b8ng = stablehlo.multiply %adlrs2b8ng, %adrats2b8ng : tensor<f32>
    %adsubs2b8ng = stablehlo.subtract %s2b8ng, %adsts2b8ng : tensor<f32>
    %adwds2b8ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b8ng = stablehlo.multiply %adwds2b8ng, %adlrs2b8ng : tensor<f32>
    %adwdps2b8ng = stablehlo.multiply %adwdlrs2b8ng, %s2b8ng : tensor<f32>
    %adnews2b8ng = stablehlo.subtract %adsubs2b8ng, %adwdps2b8ng : tensor<f32>
    %adb1s2b8nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s2b8nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss2b8nbt = stablehlo.multiply %adb1s2b8nbt, %s2b8nbtm : tensor<f32>
    %admgs2b8nbt = stablehlo.multiply %adob1s2b8nbt, %s2b8dndb : tensor<f32>
    %admns2b8nbt = stablehlo.add %admss2b8nbt, %admgs2b8nbt : tensor<f32>
    %adb2s2b8nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s2b8nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss2b8nbt = stablehlo.multiply %adb2s2b8nbt, %s2b8nbtv : tensor<f32>
    %adg2s2b8nbt = stablehlo.multiply %s2b8dndb, %s2b8dndb : tensor<f32>
    %advgs2b8nbt = stablehlo.multiply %adob2s2b8nbt, %adg2s2b8nbt : tensor<f32>
    %advns2b8nbt = stablehlo.add %advss2b8nbt, %advgs2b8nbt : tensor<f32>
    %adbc1s2b8nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s2b8nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs2b8nbt = stablehlo.divide %admns2b8nbt, %adbc1s2b8nbt : tensor<f32>
    %advhs2b8nbt = stablehlo.divide %advns2b8nbt, %adbc2s2b8nbt : tensor<f32>
    %adlrs2b8nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss2b8nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs2b8nbt = stablehlo.sqrt %advhs2b8nbt : tensor<f32>
    %addens2b8nbt = stablehlo.add %adsqs2b8nbt, %adepss2b8nbt : tensor<f32>
    %adrats2b8nbt = stablehlo.divide %admhs2b8nbt, %addens2b8nbt : tensor<f32>
    %adsts2b8nbt = stablehlo.multiply %adlrs2b8nbt, %adrats2b8nbt : tensor<f32>
    %adsubs2b8nbt = stablehlo.subtract %s2b8nbt, %adsts2b8nbt : tensor<f32>
    %adwds2b8nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs2b8nbt = stablehlo.multiply %adwds2b8nbt, %adlrs2b8nbt : tensor<f32>
    %adwdps2b8nbt = stablehlo.multiply %adwdlrs2b8nbt, %s2b8nbt : tensor<f32>
    %adnews2b8nbt = stablehlo.subtract %adsubs2b8nbt, %adwdps2b8nbt : tensor<f32>
    %adb1s2b8eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob1s2b8eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admss2b8eW = stablehlo.multiply %adb1s2b8eW, %s2b8eWm : tensor<1536x384x1x1xf32>
    %admgs2b8eW = stablehlo.multiply %adob1s2b8eW, %s2b8deW : tensor<1536x384x1x1xf32>
    %admns2b8eW = stablehlo.add %admss2b8eW, %admgs2b8eW : tensor<1536x384x1x1xf32>
    %adb2s2b8eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adob2s2b8eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %advss2b8eW = stablehlo.multiply %adb2s2b8eW, %s2b8eWv : tensor<1536x384x1x1xf32>
    %adg2s2b8eW = stablehlo.multiply %s2b8deW, %s2b8deW : tensor<1536x384x1x1xf32>
    %advgs2b8eW = stablehlo.multiply %adob2s2b8eW, %adg2s2b8eW : tensor<1536x384x1x1xf32>
    %advns2b8eW = stablehlo.add %advss2b8eW, %advgs2b8eW : tensor<1536x384x1x1xf32>
    %adbc1s2b8eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adbc2s2b8eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %admhs2b8eW = stablehlo.divide %admns2b8eW, %adbc1s2b8eW : tensor<1536x384x1x1xf32>
    %advhs2b8eW = stablehlo.divide %advns2b8eW, %adbc2s2b8eW : tensor<1536x384x1x1xf32>
    %adlrs2b8eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adepss2b8eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adsqs2b8eW = stablehlo.sqrt %advhs2b8eW : tensor<1536x384x1x1xf32>
    %addens2b8eW = stablehlo.add %adsqs2b8eW, %adepss2b8eW : tensor<1536x384x1x1xf32>
    %adrats2b8eW = stablehlo.divide %admhs2b8eW, %addens2b8eW : tensor<1536x384x1x1xf32>
    %adsts2b8eW = stablehlo.multiply %adlrs2b8eW, %adrats2b8eW : tensor<1536x384x1x1xf32>
    %adsubs2b8eW = stablehlo.subtract %s2b8eW, %adsts2b8eW : tensor<1536x384x1x1xf32>
    %adwds2b8eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %adwdlrs2b8eW = stablehlo.multiply %adwds2b8eW, %adlrs2b8eW : tensor<1536x384x1x1xf32>
    %adwdps2b8eW = stablehlo.multiply %adwdlrs2b8eW, %s2b8eW : tensor<1536x384x1x1xf32>
    %adnews2b8eW = stablehlo.subtract %adsubs2b8eW, %adwdps2b8eW : tensor<1536x384x1x1xf32>
    %adb1s2b8eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob1s2b8eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admss2b8eb = stablehlo.multiply %adb1s2b8eb, %s2b8ebm : tensor<1536xf32>
    %admgs2b8eb = stablehlo.multiply %adob1s2b8eb, %s2b8deb : tensor<1536xf32>
    %admns2b8eb = stablehlo.add %admss2b8eb, %admgs2b8eb : tensor<1536xf32>
    %adb2s2b8eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adob2s2b8eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %advss2b8eb = stablehlo.multiply %adb2s2b8eb, %s2b8ebv : tensor<1536xf32>
    %adg2s2b8eb = stablehlo.multiply %s2b8deb, %s2b8deb : tensor<1536xf32>
    %advgs2b8eb = stablehlo.multiply %adob2s2b8eb, %adg2s2b8eb : tensor<1536xf32>
    %advns2b8eb = stablehlo.add %advss2b8eb, %advgs2b8eb : tensor<1536xf32>
    %adbc1s2b8eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adbc2s2b8eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %admhs2b8eb = stablehlo.divide %admns2b8eb, %adbc1s2b8eb : tensor<1536xf32>
    %advhs2b8eb = stablehlo.divide %advns2b8eb, %adbc2s2b8eb : tensor<1536xf32>
    %adlrs2b8eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adepss2b8eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adsqs2b8eb = stablehlo.sqrt %advhs2b8eb : tensor<1536xf32>
    %addens2b8eb = stablehlo.add %adsqs2b8eb, %adepss2b8eb : tensor<1536xf32>
    %adrats2b8eb = stablehlo.divide %admhs2b8eb, %addens2b8eb : tensor<1536xf32>
    %adsts2b8eb = stablehlo.multiply %adlrs2b8eb, %adrats2b8eb : tensor<1536xf32>
    %adsubs2b8eb = stablehlo.subtract %s2b8eb, %adsts2b8eb : tensor<1536xf32>
    %adwds2b8eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %adwdlrs2b8eb = stablehlo.multiply %adwds2b8eb, %adlrs2b8eb : tensor<1536xf32>
    %adwdps2b8eb = stablehlo.multiply %adwdlrs2b8eb, %s2b8eb : tensor<1536xf32>
    %adnews2b8eb = stablehlo.subtract %adsubs2b8eb, %adwdps2b8eb : tensor<1536xf32>
    %adb1s2b8pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob1s2b8pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admss2b8pW = stablehlo.multiply %adb1s2b8pW, %s2b8pWm : tensor<384x1536x1x1xf32>
    %admgs2b8pW = stablehlo.multiply %adob1s2b8pW, %s2b8dpW : tensor<384x1536x1x1xf32>
    %admns2b8pW = stablehlo.add %admss2b8pW, %admgs2b8pW : tensor<384x1536x1x1xf32>
    %adb2s2b8pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adob2s2b8pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %advss2b8pW = stablehlo.multiply %adb2s2b8pW, %s2b8pWv : tensor<384x1536x1x1xf32>
    %adg2s2b8pW = stablehlo.multiply %s2b8dpW, %s2b8dpW : tensor<384x1536x1x1xf32>
    %advgs2b8pW = stablehlo.multiply %adob2s2b8pW, %adg2s2b8pW : tensor<384x1536x1x1xf32>
    %advns2b8pW = stablehlo.add %advss2b8pW, %advgs2b8pW : tensor<384x1536x1x1xf32>
    %adbc1s2b8pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adbc2s2b8pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %admhs2b8pW = stablehlo.divide %admns2b8pW, %adbc1s2b8pW : tensor<384x1536x1x1xf32>
    %advhs2b8pW = stablehlo.divide %advns2b8pW, %adbc2s2b8pW : tensor<384x1536x1x1xf32>
    %adlrs2b8pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adepss2b8pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adsqs2b8pW = stablehlo.sqrt %advhs2b8pW : tensor<384x1536x1x1xf32>
    %addens2b8pW = stablehlo.add %adsqs2b8pW, %adepss2b8pW : tensor<384x1536x1x1xf32>
    %adrats2b8pW = stablehlo.divide %admhs2b8pW, %addens2b8pW : tensor<384x1536x1x1xf32>
    %adsts2b8pW = stablehlo.multiply %adlrs2b8pW, %adrats2b8pW : tensor<384x1536x1x1xf32>
    %adsubs2b8pW = stablehlo.subtract %s2b8pW, %adsts2b8pW : tensor<384x1536x1x1xf32>
    %adwds2b8pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %adwdlrs2b8pW = stablehlo.multiply %adwds2b8pW, %adlrs2b8pW : tensor<384x1536x1x1xf32>
    %adwdps2b8pW = stablehlo.multiply %adwdlrs2b8pW, %s2b8pW : tensor<384x1536x1x1xf32>
    %adnews2b8pW = stablehlo.subtract %adsubs2b8pW, %adwdps2b8pW : tensor<384x1536x1x1xf32>
    %adb1s2b8pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b8pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b8pb = stablehlo.multiply %adb1s2b8pb, %s2b8pbm : tensor<384xf32>
    %admgs2b8pb = stablehlo.multiply %adob1s2b8pb, %s2b8dpb : tensor<384xf32>
    %admns2b8pb = stablehlo.add %admss2b8pb, %admgs2b8pb : tensor<384xf32>
    %adb2s2b8pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b8pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b8pb = stablehlo.multiply %adb2s2b8pb, %s2b8pbv : tensor<384xf32>
    %adg2s2b8pb = stablehlo.multiply %s2b8dpb, %s2b8dpb : tensor<384xf32>
    %advgs2b8pb = stablehlo.multiply %adob2s2b8pb, %adg2s2b8pb : tensor<384xf32>
    %advns2b8pb = stablehlo.add %advss2b8pb, %advgs2b8pb : tensor<384xf32>
    %adbc1s2b8pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b8pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b8pb = stablehlo.divide %admns2b8pb, %adbc1s2b8pb : tensor<384xf32>
    %advhs2b8pb = stablehlo.divide %advns2b8pb, %adbc2s2b8pb : tensor<384xf32>
    %adlrs2b8pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b8pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b8pb = stablehlo.sqrt %advhs2b8pb : tensor<384xf32>
    %addens2b8pb = stablehlo.add %adsqs2b8pb, %adepss2b8pb : tensor<384xf32>
    %adrats2b8pb = stablehlo.divide %admhs2b8pb, %addens2b8pb : tensor<384xf32>
    %adsts2b8pb = stablehlo.multiply %adlrs2b8pb, %adrats2b8pb : tensor<384xf32>
    %adsubs2b8pb = stablehlo.subtract %s2b8pb, %adsts2b8pb : tensor<384xf32>
    %adwds2b8pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b8pb = stablehlo.multiply %adwds2b8pb, %adlrs2b8pb : tensor<384xf32>
    %adwdps2b8pb = stablehlo.multiply %adwdlrs2b8pb, %s2b8pb : tensor<384xf32>
    %adnews2b8pb = stablehlo.subtract %adsubs2b8pb, %adwdps2b8pb : tensor<384xf32>
    %adb1s2b8lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob1s2b8lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admss2b8lg = stablehlo.multiply %adb1s2b8lg, %s2b8lgm : tensor<384xf32>
    %admgs2b8lg = stablehlo.multiply %adob1s2b8lg, %s2b8dlsdg : tensor<384xf32>
    %admns2b8lg = stablehlo.add %admss2b8lg, %admgs2b8lg : tensor<384xf32>
    %adb2s2b8lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adob2s2b8lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %advss2b8lg = stablehlo.multiply %adb2s2b8lg, %s2b8lgv : tensor<384xf32>
    %adg2s2b8lg = stablehlo.multiply %s2b8dlsdg, %s2b8dlsdg : tensor<384xf32>
    %advgs2b8lg = stablehlo.multiply %adob2s2b8lg, %adg2s2b8lg : tensor<384xf32>
    %advns2b8lg = stablehlo.add %advss2b8lg, %advgs2b8lg : tensor<384xf32>
    %adbc1s2b8lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adbc2s2b8lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %admhs2b8lg = stablehlo.divide %admns2b8lg, %adbc1s2b8lg : tensor<384xf32>
    %advhs2b8lg = stablehlo.divide %advns2b8lg, %adbc2s2b8lg : tensor<384xf32>
    %adlrs2b8lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adepss2b8lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adsqs2b8lg = stablehlo.sqrt %advhs2b8lg : tensor<384xf32>
    %addens2b8lg = stablehlo.add %adsqs2b8lg, %adepss2b8lg : tensor<384xf32>
    %adrats2b8lg = stablehlo.divide %admhs2b8lg, %addens2b8lg : tensor<384xf32>
    %adsts2b8lg = stablehlo.multiply %adlrs2b8lg, %adrats2b8lg : tensor<384xf32>
    %adsubs2b8lg = stablehlo.subtract %s2b8lg, %adsts2b8lg : tensor<384xf32>
    %adwds2b8lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %adwdlrs2b8lg = stablehlo.multiply %adwds2b8lg, %adlrs2b8lg : tensor<384xf32>
    %adwdps2b8lg = stablehlo.multiply %adwdlrs2b8lg, %s2b8lg : tensor<384xf32>
    %adnews2b8lg = stablehlo.subtract %adsubs2b8lg, %adwdps2b8lg : tensor<384xf32>
    %adb1d2ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1d2ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admsd2ng = stablehlo.multiply %adb1d2ng, %d2ngm : tensor<f32>
    %admgd2ng = stablehlo.multiply %adob1d2ng, %d2dndg : tensor<f32>
    %admnd2ng = stablehlo.add %admsd2ng, %admgd2ng : tensor<f32>
    %adb2d2ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2d2ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advsd2ng = stablehlo.multiply %adb2d2ng, %d2ngv : tensor<f32>
    %adg2d2ng = stablehlo.multiply %d2dndg, %d2dndg : tensor<f32>
    %advgd2ng = stablehlo.multiply %adob2d2ng, %adg2d2ng : tensor<f32>
    %advnd2ng = stablehlo.add %advsd2ng, %advgd2ng : tensor<f32>
    %adbc1d2ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2d2ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhd2ng = stablehlo.divide %admnd2ng, %adbc1d2ng : tensor<f32>
    %advhd2ng = stablehlo.divide %advnd2ng, %adbc2d2ng : tensor<f32>
    %adlrd2ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepsd2ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqd2ng = stablehlo.sqrt %advhd2ng : tensor<f32>
    %addend2ng = stablehlo.add %adsqd2ng, %adepsd2ng : tensor<f32>
    %adratd2ng = stablehlo.divide %admhd2ng, %addend2ng : tensor<f32>
    %adstd2ng = stablehlo.multiply %adlrd2ng, %adratd2ng : tensor<f32>
    %adsubd2ng = stablehlo.subtract %d2ng, %adstd2ng : tensor<f32>
    %adwdd2ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrd2ng = stablehlo.multiply %adwdd2ng, %adlrd2ng : tensor<f32>
    %adwdpd2ng = stablehlo.multiply %adwdlrd2ng, %d2ng : tensor<f32>
    %adnewd2ng = stablehlo.subtract %adsubd2ng, %adwdpd2ng : tensor<f32>
    %adb1d2nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1d2nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admsd2nbt = stablehlo.multiply %adb1d2nbt, %d2nbtm : tensor<f32>
    %admgd2nbt = stablehlo.multiply %adob1d2nbt, %d2dndb : tensor<f32>
    %admnd2nbt = stablehlo.add %admsd2nbt, %admgd2nbt : tensor<f32>
    %adb2d2nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2d2nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advsd2nbt = stablehlo.multiply %adb2d2nbt, %d2nbtv : tensor<f32>
    %adg2d2nbt = stablehlo.multiply %d2dndb, %d2dndb : tensor<f32>
    %advgd2nbt = stablehlo.multiply %adob2d2nbt, %adg2d2nbt : tensor<f32>
    %advnd2nbt = stablehlo.add %advsd2nbt, %advgd2nbt : tensor<f32>
    %adbc1d2nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2d2nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhd2nbt = stablehlo.divide %admnd2nbt, %adbc1d2nbt : tensor<f32>
    %advhd2nbt = stablehlo.divide %advnd2nbt, %adbc2d2nbt : tensor<f32>
    %adlrd2nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepsd2nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqd2nbt = stablehlo.sqrt %advhd2nbt : tensor<f32>
    %addend2nbt = stablehlo.add %adsqd2nbt, %adepsd2nbt : tensor<f32>
    %adratd2nbt = stablehlo.divide %admhd2nbt, %addend2nbt : tensor<f32>
    %adstd2nbt = stablehlo.multiply %adlrd2nbt, %adratd2nbt : tensor<f32>
    %adsubd2nbt = stablehlo.subtract %d2nbt, %adstd2nbt : tensor<f32>
    %adwdd2nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrd2nbt = stablehlo.multiply %adwdd2nbt, %adlrd2nbt : tensor<f32>
    %adwdpd2nbt = stablehlo.multiply %adwdlrd2nbt, %d2nbt : tensor<f32>
    %adnewd2nbt = stablehlo.subtract %adsubd2nbt, %adwdpd2nbt : tensor<f32>
    %adb1d2W = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %adob1d2W = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %admsd2W = stablehlo.multiply %adb1d2W, %d2Wm : tensor<768x384x2x2xf32>
    %admgd2W = stablehlo.multiply %adob1d2W, %d2dW : tensor<768x384x2x2xf32>
    %admnd2W = stablehlo.add %admsd2W, %admgd2W : tensor<768x384x2x2xf32>
    %adb2d2W = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %adob2d2W = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %advsd2W = stablehlo.multiply %adb2d2W, %d2Wv : tensor<768x384x2x2xf32>
    %adg2d2W = stablehlo.multiply %d2dW, %d2dW : tensor<768x384x2x2xf32>
    %advgd2W = stablehlo.multiply %adob2d2W, %adg2d2W : tensor<768x384x2x2xf32>
    %advnd2W = stablehlo.add %advsd2W, %advgd2W : tensor<768x384x2x2xf32>
    %adbc1d2W = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %adbc2d2W = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %admhd2W = stablehlo.divide %admnd2W, %adbc1d2W : tensor<768x384x2x2xf32>
    %advhd2W = stablehlo.divide %advnd2W, %adbc2d2W : tensor<768x384x2x2xf32>
    %adlrd2W = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %adepsd2W = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %adsqd2W = stablehlo.sqrt %advhd2W : tensor<768x384x2x2xf32>
    %addend2W = stablehlo.add %adsqd2W, %adepsd2W : tensor<768x384x2x2xf32>
    %adratd2W = stablehlo.divide %admhd2W, %addend2W : tensor<768x384x2x2xf32>
    %adstd2W = stablehlo.multiply %adlrd2W, %adratd2W : tensor<768x384x2x2xf32>
    %adsubd2W = stablehlo.subtract %d2W, %adstd2W : tensor<768x384x2x2xf32>
    %adwdd2W = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %adwdlrd2W = stablehlo.multiply %adwdd2W, %adlrd2W : tensor<768x384x2x2xf32>
    %adwdpd2W = stablehlo.multiply %adwdlrd2W, %d2W : tensor<768x384x2x2xf32>
    %adnewd2W = stablehlo.subtract %adsubd2W, %adwdpd2W : tensor<768x384x2x2xf32>
    %adb1d2b = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1d2b = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admsd2b = stablehlo.multiply %adb1d2b, %d2bm : tensor<768xf32>
    %admgd2b = stablehlo.multiply %adob1d2b, %d2db : tensor<768xf32>
    %admnd2b = stablehlo.add %admsd2b, %admgd2b : tensor<768xf32>
    %adb2d2b = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2d2b = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advsd2b = stablehlo.multiply %adb2d2b, %d2bv : tensor<768xf32>
    %adg2d2b = stablehlo.multiply %d2db, %d2db : tensor<768xf32>
    %advgd2b = stablehlo.multiply %adob2d2b, %adg2d2b : tensor<768xf32>
    %advnd2b = stablehlo.add %advsd2b, %advgd2b : tensor<768xf32>
    %adbc1d2b = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2d2b = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhd2b = stablehlo.divide %admnd2b, %adbc1d2b : tensor<768xf32>
    %advhd2b = stablehlo.divide %advnd2b, %adbc2d2b : tensor<768xf32>
    %adlrd2b = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepsd2b = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqd2b = stablehlo.sqrt %advhd2b : tensor<768xf32>
    %addend2b = stablehlo.add %adsqd2b, %adepsd2b : tensor<768xf32>
    %adratd2b = stablehlo.divide %admhd2b, %addend2b : tensor<768xf32>
    %adstd2b = stablehlo.multiply %adlrd2b, %adratd2b : tensor<768xf32>
    %adsubd2b = stablehlo.subtract %d2b, %adstd2b : tensor<768xf32>
    %adwdd2b = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrd2b = stablehlo.multiply %adwdd2b, %adlrd2b : tensor<768xf32>
    %adwdpd2b = stablehlo.multiply %adwdlrd2b, %d2b : tensor<768xf32>
    %adnewd2b = stablehlo.subtract %adsubd2b, %adwdpd2b : tensor<768xf32>
    %adb1s3b0dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adob1s3b0dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %admss3b0dW = stablehlo.multiply %adb1s3b0dW, %s3b0dWm : tensor<768x1x7x7xf32>
    %admgs3b0dW = stablehlo.multiply %adob1s3b0dW, %s3b0ddW : tensor<768x1x7x7xf32>
    %admns3b0dW = stablehlo.add %admss3b0dW, %admgs3b0dW : tensor<768x1x7x7xf32>
    %adb2s3b0dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adob2s3b0dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %advss3b0dW = stablehlo.multiply %adb2s3b0dW, %s3b0dWv : tensor<768x1x7x7xf32>
    %adg2s3b0dW = stablehlo.multiply %s3b0ddW, %s3b0ddW : tensor<768x1x7x7xf32>
    %advgs3b0dW = stablehlo.multiply %adob2s3b0dW, %adg2s3b0dW : tensor<768x1x7x7xf32>
    %advns3b0dW = stablehlo.add %advss3b0dW, %advgs3b0dW : tensor<768x1x7x7xf32>
    %adbc1s3b0dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adbc2s3b0dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %admhs3b0dW = stablehlo.divide %admns3b0dW, %adbc1s3b0dW : tensor<768x1x7x7xf32>
    %advhs3b0dW = stablehlo.divide %advns3b0dW, %adbc2s3b0dW : tensor<768x1x7x7xf32>
    %adlrs3b0dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adepss3b0dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adsqs3b0dW = stablehlo.sqrt %advhs3b0dW : tensor<768x1x7x7xf32>
    %addens3b0dW = stablehlo.add %adsqs3b0dW, %adepss3b0dW : tensor<768x1x7x7xf32>
    %adrats3b0dW = stablehlo.divide %admhs3b0dW, %addens3b0dW : tensor<768x1x7x7xf32>
    %adsts3b0dW = stablehlo.multiply %adlrs3b0dW, %adrats3b0dW : tensor<768x1x7x7xf32>
    %adsubs3b0dW = stablehlo.subtract %s3b0dW, %adsts3b0dW : tensor<768x1x7x7xf32>
    %adwds3b0dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adwdlrs3b0dW = stablehlo.multiply %adwds3b0dW, %adlrs3b0dW : tensor<768x1x7x7xf32>
    %adwdps3b0dW = stablehlo.multiply %adwdlrs3b0dW, %s3b0dW : tensor<768x1x7x7xf32>
    %adnews3b0dW = stablehlo.subtract %adsubs3b0dW, %adwdps3b0dW : tensor<768x1x7x7xf32>
    %adb1s3b0db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1s3b0db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admss3b0db = stablehlo.multiply %adb1s3b0db, %s3b0dbm : tensor<768xf32>
    %admgs3b0db = stablehlo.multiply %adob1s3b0db, %s3b0ddb : tensor<768xf32>
    %admns3b0db = stablehlo.add %admss3b0db, %admgs3b0db : tensor<768xf32>
    %adb2s3b0db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2s3b0db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advss3b0db = stablehlo.multiply %adb2s3b0db, %s3b0dbv : tensor<768xf32>
    %adg2s3b0db = stablehlo.multiply %s3b0ddb, %s3b0ddb : tensor<768xf32>
    %advgs3b0db = stablehlo.multiply %adob2s3b0db, %adg2s3b0db : tensor<768xf32>
    %advns3b0db = stablehlo.add %advss3b0db, %advgs3b0db : tensor<768xf32>
    %adbc1s3b0db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2s3b0db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhs3b0db = stablehlo.divide %admns3b0db, %adbc1s3b0db : tensor<768xf32>
    %advhs3b0db = stablehlo.divide %advns3b0db, %adbc2s3b0db : tensor<768xf32>
    %adlrs3b0db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepss3b0db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqs3b0db = stablehlo.sqrt %advhs3b0db : tensor<768xf32>
    %addens3b0db = stablehlo.add %adsqs3b0db, %adepss3b0db : tensor<768xf32>
    %adrats3b0db = stablehlo.divide %admhs3b0db, %addens3b0db : tensor<768xf32>
    %adsts3b0db = stablehlo.multiply %adlrs3b0db, %adrats3b0db : tensor<768xf32>
    %adsubs3b0db = stablehlo.subtract %s3b0db, %adsts3b0db : tensor<768xf32>
    %adwds3b0db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrs3b0db = stablehlo.multiply %adwds3b0db, %adlrs3b0db : tensor<768xf32>
    %adwdps3b0db = stablehlo.multiply %adwdlrs3b0db, %s3b0db : tensor<768xf32>
    %adnews3b0db = stablehlo.subtract %adsubs3b0db, %adwdps3b0db : tensor<768xf32>
    %adb1s3b0ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s3b0ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss3b0ng = stablehlo.multiply %adb1s3b0ng, %s3b0ngm : tensor<f32>
    %admgs3b0ng = stablehlo.multiply %adob1s3b0ng, %s3b0dndg : tensor<f32>
    %admns3b0ng = stablehlo.add %admss3b0ng, %admgs3b0ng : tensor<f32>
    %adb2s3b0ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s3b0ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss3b0ng = stablehlo.multiply %adb2s3b0ng, %s3b0ngv : tensor<f32>
    %adg2s3b0ng = stablehlo.multiply %s3b0dndg, %s3b0dndg : tensor<f32>
    %advgs3b0ng = stablehlo.multiply %adob2s3b0ng, %adg2s3b0ng : tensor<f32>
    %advns3b0ng = stablehlo.add %advss3b0ng, %advgs3b0ng : tensor<f32>
    %adbc1s3b0ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s3b0ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs3b0ng = stablehlo.divide %admns3b0ng, %adbc1s3b0ng : tensor<f32>
    %advhs3b0ng = stablehlo.divide %advns3b0ng, %adbc2s3b0ng : tensor<f32>
    %adlrs3b0ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss3b0ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs3b0ng = stablehlo.sqrt %advhs3b0ng : tensor<f32>
    %addens3b0ng = stablehlo.add %adsqs3b0ng, %adepss3b0ng : tensor<f32>
    %adrats3b0ng = stablehlo.divide %admhs3b0ng, %addens3b0ng : tensor<f32>
    %adsts3b0ng = stablehlo.multiply %adlrs3b0ng, %adrats3b0ng : tensor<f32>
    %adsubs3b0ng = stablehlo.subtract %s3b0ng, %adsts3b0ng : tensor<f32>
    %adwds3b0ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs3b0ng = stablehlo.multiply %adwds3b0ng, %adlrs3b0ng : tensor<f32>
    %adwdps3b0ng = stablehlo.multiply %adwdlrs3b0ng, %s3b0ng : tensor<f32>
    %adnews3b0ng = stablehlo.subtract %adsubs3b0ng, %adwdps3b0ng : tensor<f32>
    %adb1s3b0nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s3b0nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss3b0nbt = stablehlo.multiply %adb1s3b0nbt, %s3b0nbtm : tensor<f32>
    %admgs3b0nbt = stablehlo.multiply %adob1s3b0nbt, %s3b0dndb : tensor<f32>
    %admns3b0nbt = stablehlo.add %admss3b0nbt, %admgs3b0nbt : tensor<f32>
    %adb2s3b0nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s3b0nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss3b0nbt = stablehlo.multiply %adb2s3b0nbt, %s3b0nbtv : tensor<f32>
    %adg2s3b0nbt = stablehlo.multiply %s3b0dndb, %s3b0dndb : tensor<f32>
    %advgs3b0nbt = stablehlo.multiply %adob2s3b0nbt, %adg2s3b0nbt : tensor<f32>
    %advns3b0nbt = stablehlo.add %advss3b0nbt, %advgs3b0nbt : tensor<f32>
    %adbc1s3b0nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s3b0nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs3b0nbt = stablehlo.divide %admns3b0nbt, %adbc1s3b0nbt : tensor<f32>
    %advhs3b0nbt = stablehlo.divide %advns3b0nbt, %adbc2s3b0nbt : tensor<f32>
    %adlrs3b0nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss3b0nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs3b0nbt = stablehlo.sqrt %advhs3b0nbt : tensor<f32>
    %addens3b0nbt = stablehlo.add %adsqs3b0nbt, %adepss3b0nbt : tensor<f32>
    %adrats3b0nbt = stablehlo.divide %admhs3b0nbt, %addens3b0nbt : tensor<f32>
    %adsts3b0nbt = stablehlo.multiply %adlrs3b0nbt, %adrats3b0nbt : tensor<f32>
    %adsubs3b0nbt = stablehlo.subtract %s3b0nbt, %adsts3b0nbt : tensor<f32>
    %adwds3b0nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs3b0nbt = stablehlo.multiply %adwds3b0nbt, %adlrs3b0nbt : tensor<f32>
    %adwdps3b0nbt = stablehlo.multiply %adwdlrs3b0nbt, %s3b0nbt : tensor<f32>
    %adnews3b0nbt = stablehlo.subtract %adsubs3b0nbt, %adwdps3b0nbt : tensor<f32>
    %adb1s3b0eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adob1s3b0eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %admss3b0eW = stablehlo.multiply %adb1s3b0eW, %s3b0eWm : tensor<3072x768x1x1xf32>
    %admgs3b0eW = stablehlo.multiply %adob1s3b0eW, %s3b0deW : tensor<3072x768x1x1xf32>
    %admns3b0eW = stablehlo.add %admss3b0eW, %admgs3b0eW : tensor<3072x768x1x1xf32>
    %adb2s3b0eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adob2s3b0eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %advss3b0eW = stablehlo.multiply %adb2s3b0eW, %s3b0eWv : tensor<3072x768x1x1xf32>
    %adg2s3b0eW = stablehlo.multiply %s3b0deW, %s3b0deW : tensor<3072x768x1x1xf32>
    %advgs3b0eW = stablehlo.multiply %adob2s3b0eW, %adg2s3b0eW : tensor<3072x768x1x1xf32>
    %advns3b0eW = stablehlo.add %advss3b0eW, %advgs3b0eW : tensor<3072x768x1x1xf32>
    %adbc1s3b0eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adbc2s3b0eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %admhs3b0eW = stablehlo.divide %admns3b0eW, %adbc1s3b0eW : tensor<3072x768x1x1xf32>
    %advhs3b0eW = stablehlo.divide %advns3b0eW, %adbc2s3b0eW : tensor<3072x768x1x1xf32>
    %adlrs3b0eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adepss3b0eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adsqs3b0eW = stablehlo.sqrt %advhs3b0eW : tensor<3072x768x1x1xf32>
    %addens3b0eW = stablehlo.add %adsqs3b0eW, %adepss3b0eW : tensor<3072x768x1x1xf32>
    %adrats3b0eW = stablehlo.divide %admhs3b0eW, %addens3b0eW : tensor<3072x768x1x1xf32>
    %adsts3b0eW = stablehlo.multiply %adlrs3b0eW, %adrats3b0eW : tensor<3072x768x1x1xf32>
    %adsubs3b0eW = stablehlo.subtract %s3b0eW, %adsts3b0eW : tensor<3072x768x1x1xf32>
    %adwds3b0eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adwdlrs3b0eW = stablehlo.multiply %adwds3b0eW, %adlrs3b0eW : tensor<3072x768x1x1xf32>
    %adwdps3b0eW = stablehlo.multiply %adwdlrs3b0eW, %s3b0eW : tensor<3072x768x1x1xf32>
    %adnews3b0eW = stablehlo.subtract %adsubs3b0eW, %adwdps3b0eW : tensor<3072x768x1x1xf32>
    %adb1s3b0eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adob1s3b0eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %admss3b0eb = stablehlo.multiply %adb1s3b0eb, %s3b0ebm : tensor<3072xf32>
    %admgs3b0eb = stablehlo.multiply %adob1s3b0eb, %s3b0deb : tensor<3072xf32>
    %admns3b0eb = stablehlo.add %admss3b0eb, %admgs3b0eb : tensor<3072xf32>
    %adb2s3b0eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adob2s3b0eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %advss3b0eb = stablehlo.multiply %adb2s3b0eb, %s3b0ebv : tensor<3072xf32>
    %adg2s3b0eb = stablehlo.multiply %s3b0deb, %s3b0deb : tensor<3072xf32>
    %advgs3b0eb = stablehlo.multiply %adob2s3b0eb, %adg2s3b0eb : tensor<3072xf32>
    %advns3b0eb = stablehlo.add %advss3b0eb, %advgs3b0eb : tensor<3072xf32>
    %adbc1s3b0eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adbc2s3b0eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %admhs3b0eb = stablehlo.divide %admns3b0eb, %adbc1s3b0eb : tensor<3072xf32>
    %advhs3b0eb = stablehlo.divide %advns3b0eb, %adbc2s3b0eb : tensor<3072xf32>
    %adlrs3b0eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adepss3b0eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adsqs3b0eb = stablehlo.sqrt %advhs3b0eb : tensor<3072xf32>
    %addens3b0eb = stablehlo.add %adsqs3b0eb, %adepss3b0eb : tensor<3072xf32>
    %adrats3b0eb = stablehlo.divide %admhs3b0eb, %addens3b0eb : tensor<3072xf32>
    %adsts3b0eb = stablehlo.multiply %adlrs3b0eb, %adrats3b0eb : tensor<3072xf32>
    %adsubs3b0eb = stablehlo.subtract %s3b0eb, %adsts3b0eb : tensor<3072xf32>
    %adwds3b0eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adwdlrs3b0eb = stablehlo.multiply %adwds3b0eb, %adlrs3b0eb : tensor<3072xf32>
    %adwdps3b0eb = stablehlo.multiply %adwdlrs3b0eb, %s3b0eb : tensor<3072xf32>
    %adnews3b0eb = stablehlo.subtract %adsubs3b0eb, %adwdps3b0eb : tensor<3072xf32>
    %adb1s3b0pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adob1s3b0pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %admss3b0pW = stablehlo.multiply %adb1s3b0pW, %s3b0pWm : tensor<768x3072x1x1xf32>
    %admgs3b0pW = stablehlo.multiply %adob1s3b0pW, %s3b0dpW : tensor<768x3072x1x1xf32>
    %admns3b0pW = stablehlo.add %admss3b0pW, %admgs3b0pW : tensor<768x3072x1x1xf32>
    %adb2s3b0pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adob2s3b0pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %advss3b0pW = stablehlo.multiply %adb2s3b0pW, %s3b0pWv : tensor<768x3072x1x1xf32>
    %adg2s3b0pW = stablehlo.multiply %s3b0dpW, %s3b0dpW : tensor<768x3072x1x1xf32>
    %advgs3b0pW = stablehlo.multiply %adob2s3b0pW, %adg2s3b0pW : tensor<768x3072x1x1xf32>
    %advns3b0pW = stablehlo.add %advss3b0pW, %advgs3b0pW : tensor<768x3072x1x1xf32>
    %adbc1s3b0pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adbc2s3b0pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %admhs3b0pW = stablehlo.divide %admns3b0pW, %adbc1s3b0pW : tensor<768x3072x1x1xf32>
    %advhs3b0pW = stablehlo.divide %advns3b0pW, %adbc2s3b0pW : tensor<768x3072x1x1xf32>
    %adlrs3b0pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adepss3b0pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adsqs3b0pW = stablehlo.sqrt %advhs3b0pW : tensor<768x3072x1x1xf32>
    %addens3b0pW = stablehlo.add %adsqs3b0pW, %adepss3b0pW : tensor<768x3072x1x1xf32>
    %adrats3b0pW = stablehlo.divide %admhs3b0pW, %addens3b0pW : tensor<768x3072x1x1xf32>
    %adsts3b0pW = stablehlo.multiply %adlrs3b0pW, %adrats3b0pW : tensor<768x3072x1x1xf32>
    %adsubs3b0pW = stablehlo.subtract %s3b0pW, %adsts3b0pW : tensor<768x3072x1x1xf32>
    %adwds3b0pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adwdlrs3b0pW = stablehlo.multiply %adwds3b0pW, %adlrs3b0pW : tensor<768x3072x1x1xf32>
    %adwdps3b0pW = stablehlo.multiply %adwdlrs3b0pW, %s3b0pW : tensor<768x3072x1x1xf32>
    %adnews3b0pW = stablehlo.subtract %adsubs3b0pW, %adwdps3b0pW : tensor<768x3072x1x1xf32>
    %adb1s3b0pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1s3b0pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admss3b0pb = stablehlo.multiply %adb1s3b0pb, %s3b0pbm : tensor<768xf32>
    %admgs3b0pb = stablehlo.multiply %adob1s3b0pb, %s3b0dpb : tensor<768xf32>
    %admns3b0pb = stablehlo.add %admss3b0pb, %admgs3b0pb : tensor<768xf32>
    %adb2s3b0pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2s3b0pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advss3b0pb = stablehlo.multiply %adb2s3b0pb, %s3b0pbv : tensor<768xf32>
    %adg2s3b0pb = stablehlo.multiply %s3b0dpb, %s3b0dpb : tensor<768xf32>
    %advgs3b0pb = stablehlo.multiply %adob2s3b0pb, %adg2s3b0pb : tensor<768xf32>
    %advns3b0pb = stablehlo.add %advss3b0pb, %advgs3b0pb : tensor<768xf32>
    %adbc1s3b0pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2s3b0pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhs3b0pb = stablehlo.divide %admns3b0pb, %adbc1s3b0pb : tensor<768xf32>
    %advhs3b0pb = stablehlo.divide %advns3b0pb, %adbc2s3b0pb : tensor<768xf32>
    %adlrs3b0pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepss3b0pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqs3b0pb = stablehlo.sqrt %advhs3b0pb : tensor<768xf32>
    %addens3b0pb = stablehlo.add %adsqs3b0pb, %adepss3b0pb : tensor<768xf32>
    %adrats3b0pb = stablehlo.divide %admhs3b0pb, %addens3b0pb : tensor<768xf32>
    %adsts3b0pb = stablehlo.multiply %adlrs3b0pb, %adrats3b0pb : tensor<768xf32>
    %adsubs3b0pb = stablehlo.subtract %s3b0pb, %adsts3b0pb : tensor<768xf32>
    %adwds3b0pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrs3b0pb = stablehlo.multiply %adwds3b0pb, %adlrs3b0pb : tensor<768xf32>
    %adwdps3b0pb = stablehlo.multiply %adwdlrs3b0pb, %s3b0pb : tensor<768xf32>
    %adnews3b0pb = stablehlo.subtract %adsubs3b0pb, %adwdps3b0pb : tensor<768xf32>
    %adb1s3b0lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1s3b0lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admss3b0lg = stablehlo.multiply %adb1s3b0lg, %s3b0lgm : tensor<768xf32>
    %admgs3b0lg = stablehlo.multiply %adob1s3b0lg, %s3b0dlsdg : tensor<768xf32>
    %admns3b0lg = stablehlo.add %admss3b0lg, %admgs3b0lg : tensor<768xf32>
    %adb2s3b0lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2s3b0lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advss3b0lg = stablehlo.multiply %adb2s3b0lg, %s3b0lgv : tensor<768xf32>
    %adg2s3b0lg = stablehlo.multiply %s3b0dlsdg, %s3b0dlsdg : tensor<768xf32>
    %advgs3b0lg = stablehlo.multiply %adob2s3b0lg, %adg2s3b0lg : tensor<768xf32>
    %advns3b0lg = stablehlo.add %advss3b0lg, %advgs3b0lg : tensor<768xf32>
    %adbc1s3b0lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2s3b0lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhs3b0lg = stablehlo.divide %admns3b0lg, %adbc1s3b0lg : tensor<768xf32>
    %advhs3b0lg = stablehlo.divide %advns3b0lg, %adbc2s3b0lg : tensor<768xf32>
    %adlrs3b0lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepss3b0lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqs3b0lg = stablehlo.sqrt %advhs3b0lg : tensor<768xf32>
    %addens3b0lg = stablehlo.add %adsqs3b0lg, %adepss3b0lg : tensor<768xf32>
    %adrats3b0lg = stablehlo.divide %admhs3b0lg, %addens3b0lg : tensor<768xf32>
    %adsts3b0lg = stablehlo.multiply %adlrs3b0lg, %adrats3b0lg : tensor<768xf32>
    %adsubs3b0lg = stablehlo.subtract %s3b0lg, %adsts3b0lg : tensor<768xf32>
    %adwds3b0lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrs3b0lg = stablehlo.multiply %adwds3b0lg, %adlrs3b0lg : tensor<768xf32>
    %adwdps3b0lg = stablehlo.multiply %adwdlrs3b0lg, %s3b0lg : tensor<768xf32>
    %adnews3b0lg = stablehlo.subtract %adsubs3b0lg, %adwdps3b0lg : tensor<768xf32>
    %adb1s3b1dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adob1s3b1dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %admss3b1dW = stablehlo.multiply %adb1s3b1dW, %s3b1dWm : tensor<768x1x7x7xf32>
    %admgs3b1dW = stablehlo.multiply %adob1s3b1dW, %s3b1ddW : tensor<768x1x7x7xf32>
    %admns3b1dW = stablehlo.add %admss3b1dW, %admgs3b1dW : tensor<768x1x7x7xf32>
    %adb2s3b1dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adob2s3b1dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %advss3b1dW = stablehlo.multiply %adb2s3b1dW, %s3b1dWv : tensor<768x1x7x7xf32>
    %adg2s3b1dW = stablehlo.multiply %s3b1ddW, %s3b1ddW : tensor<768x1x7x7xf32>
    %advgs3b1dW = stablehlo.multiply %adob2s3b1dW, %adg2s3b1dW : tensor<768x1x7x7xf32>
    %advns3b1dW = stablehlo.add %advss3b1dW, %advgs3b1dW : tensor<768x1x7x7xf32>
    %adbc1s3b1dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adbc2s3b1dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %admhs3b1dW = stablehlo.divide %admns3b1dW, %adbc1s3b1dW : tensor<768x1x7x7xf32>
    %advhs3b1dW = stablehlo.divide %advns3b1dW, %adbc2s3b1dW : tensor<768x1x7x7xf32>
    %adlrs3b1dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adepss3b1dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adsqs3b1dW = stablehlo.sqrt %advhs3b1dW : tensor<768x1x7x7xf32>
    %addens3b1dW = stablehlo.add %adsqs3b1dW, %adepss3b1dW : tensor<768x1x7x7xf32>
    %adrats3b1dW = stablehlo.divide %admhs3b1dW, %addens3b1dW : tensor<768x1x7x7xf32>
    %adsts3b1dW = stablehlo.multiply %adlrs3b1dW, %adrats3b1dW : tensor<768x1x7x7xf32>
    %adsubs3b1dW = stablehlo.subtract %s3b1dW, %adsts3b1dW : tensor<768x1x7x7xf32>
    %adwds3b1dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adwdlrs3b1dW = stablehlo.multiply %adwds3b1dW, %adlrs3b1dW : tensor<768x1x7x7xf32>
    %adwdps3b1dW = stablehlo.multiply %adwdlrs3b1dW, %s3b1dW : tensor<768x1x7x7xf32>
    %adnews3b1dW = stablehlo.subtract %adsubs3b1dW, %adwdps3b1dW : tensor<768x1x7x7xf32>
    %adb1s3b1db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1s3b1db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admss3b1db = stablehlo.multiply %adb1s3b1db, %s3b1dbm : tensor<768xf32>
    %admgs3b1db = stablehlo.multiply %adob1s3b1db, %s3b1ddb : tensor<768xf32>
    %admns3b1db = stablehlo.add %admss3b1db, %admgs3b1db : tensor<768xf32>
    %adb2s3b1db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2s3b1db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advss3b1db = stablehlo.multiply %adb2s3b1db, %s3b1dbv : tensor<768xf32>
    %adg2s3b1db = stablehlo.multiply %s3b1ddb, %s3b1ddb : tensor<768xf32>
    %advgs3b1db = stablehlo.multiply %adob2s3b1db, %adg2s3b1db : tensor<768xf32>
    %advns3b1db = stablehlo.add %advss3b1db, %advgs3b1db : tensor<768xf32>
    %adbc1s3b1db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2s3b1db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhs3b1db = stablehlo.divide %admns3b1db, %adbc1s3b1db : tensor<768xf32>
    %advhs3b1db = stablehlo.divide %advns3b1db, %adbc2s3b1db : tensor<768xf32>
    %adlrs3b1db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepss3b1db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqs3b1db = stablehlo.sqrt %advhs3b1db : tensor<768xf32>
    %addens3b1db = stablehlo.add %adsqs3b1db, %adepss3b1db : tensor<768xf32>
    %adrats3b1db = stablehlo.divide %admhs3b1db, %addens3b1db : tensor<768xf32>
    %adsts3b1db = stablehlo.multiply %adlrs3b1db, %adrats3b1db : tensor<768xf32>
    %adsubs3b1db = stablehlo.subtract %s3b1db, %adsts3b1db : tensor<768xf32>
    %adwds3b1db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrs3b1db = stablehlo.multiply %adwds3b1db, %adlrs3b1db : tensor<768xf32>
    %adwdps3b1db = stablehlo.multiply %adwdlrs3b1db, %s3b1db : tensor<768xf32>
    %adnews3b1db = stablehlo.subtract %adsubs3b1db, %adwdps3b1db : tensor<768xf32>
    %adb1s3b1ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s3b1ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss3b1ng = stablehlo.multiply %adb1s3b1ng, %s3b1ngm : tensor<f32>
    %admgs3b1ng = stablehlo.multiply %adob1s3b1ng, %s3b1dndg : tensor<f32>
    %admns3b1ng = stablehlo.add %admss3b1ng, %admgs3b1ng : tensor<f32>
    %adb2s3b1ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s3b1ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss3b1ng = stablehlo.multiply %adb2s3b1ng, %s3b1ngv : tensor<f32>
    %adg2s3b1ng = stablehlo.multiply %s3b1dndg, %s3b1dndg : tensor<f32>
    %advgs3b1ng = stablehlo.multiply %adob2s3b1ng, %adg2s3b1ng : tensor<f32>
    %advns3b1ng = stablehlo.add %advss3b1ng, %advgs3b1ng : tensor<f32>
    %adbc1s3b1ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s3b1ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs3b1ng = stablehlo.divide %admns3b1ng, %adbc1s3b1ng : tensor<f32>
    %advhs3b1ng = stablehlo.divide %advns3b1ng, %adbc2s3b1ng : tensor<f32>
    %adlrs3b1ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss3b1ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs3b1ng = stablehlo.sqrt %advhs3b1ng : tensor<f32>
    %addens3b1ng = stablehlo.add %adsqs3b1ng, %adepss3b1ng : tensor<f32>
    %adrats3b1ng = stablehlo.divide %admhs3b1ng, %addens3b1ng : tensor<f32>
    %adsts3b1ng = stablehlo.multiply %adlrs3b1ng, %adrats3b1ng : tensor<f32>
    %adsubs3b1ng = stablehlo.subtract %s3b1ng, %adsts3b1ng : tensor<f32>
    %adwds3b1ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs3b1ng = stablehlo.multiply %adwds3b1ng, %adlrs3b1ng : tensor<f32>
    %adwdps3b1ng = stablehlo.multiply %adwdlrs3b1ng, %s3b1ng : tensor<f32>
    %adnews3b1ng = stablehlo.subtract %adsubs3b1ng, %adwdps3b1ng : tensor<f32>
    %adb1s3b1nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s3b1nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss3b1nbt = stablehlo.multiply %adb1s3b1nbt, %s3b1nbtm : tensor<f32>
    %admgs3b1nbt = stablehlo.multiply %adob1s3b1nbt, %s3b1dndb : tensor<f32>
    %admns3b1nbt = stablehlo.add %admss3b1nbt, %admgs3b1nbt : tensor<f32>
    %adb2s3b1nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s3b1nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss3b1nbt = stablehlo.multiply %adb2s3b1nbt, %s3b1nbtv : tensor<f32>
    %adg2s3b1nbt = stablehlo.multiply %s3b1dndb, %s3b1dndb : tensor<f32>
    %advgs3b1nbt = stablehlo.multiply %adob2s3b1nbt, %adg2s3b1nbt : tensor<f32>
    %advns3b1nbt = stablehlo.add %advss3b1nbt, %advgs3b1nbt : tensor<f32>
    %adbc1s3b1nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s3b1nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs3b1nbt = stablehlo.divide %admns3b1nbt, %adbc1s3b1nbt : tensor<f32>
    %advhs3b1nbt = stablehlo.divide %advns3b1nbt, %adbc2s3b1nbt : tensor<f32>
    %adlrs3b1nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss3b1nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs3b1nbt = stablehlo.sqrt %advhs3b1nbt : tensor<f32>
    %addens3b1nbt = stablehlo.add %adsqs3b1nbt, %adepss3b1nbt : tensor<f32>
    %adrats3b1nbt = stablehlo.divide %admhs3b1nbt, %addens3b1nbt : tensor<f32>
    %adsts3b1nbt = stablehlo.multiply %adlrs3b1nbt, %adrats3b1nbt : tensor<f32>
    %adsubs3b1nbt = stablehlo.subtract %s3b1nbt, %adsts3b1nbt : tensor<f32>
    %adwds3b1nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs3b1nbt = stablehlo.multiply %adwds3b1nbt, %adlrs3b1nbt : tensor<f32>
    %adwdps3b1nbt = stablehlo.multiply %adwdlrs3b1nbt, %s3b1nbt : tensor<f32>
    %adnews3b1nbt = stablehlo.subtract %adsubs3b1nbt, %adwdps3b1nbt : tensor<f32>
    %adb1s3b1eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adob1s3b1eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %admss3b1eW = stablehlo.multiply %adb1s3b1eW, %s3b1eWm : tensor<3072x768x1x1xf32>
    %admgs3b1eW = stablehlo.multiply %adob1s3b1eW, %s3b1deW : tensor<3072x768x1x1xf32>
    %admns3b1eW = stablehlo.add %admss3b1eW, %admgs3b1eW : tensor<3072x768x1x1xf32>
    %adb2s3b1eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adob2s3b1eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %advss3b1eW = stablehlo.multiply %adb2s3b1eW, %s3b1eWv : tensor<3072x768x1x1xf32>
    %adg2s3b1eW = stablehlo.multiply %s3b1deW, %s3b1deW : tensor<3072x768x1x1xf32>
    %advgs3b1eW = stablehlo.multiply %adob2s3b1eW, %adg2s3b1eW : tensor<3072x768x1x1xf32>
    %advns3b1eW = stablehlo.add %advss3b1eW, %advgs3b1eW : tensor<3072x768x1x1xf32>
    %adbc1s3b1eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adbc2s3b1eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %admhs3b1eW = stablehlo.divide %admns3b1eW, %adbc1s3b1eW : tensor<3072x768x1x1xf32>
    %advhs3b1eW = stablehlo.divide %advns3b1eW, %adbc2s3b1eW : tensor<3072x768x1x1xf32>
    %adlrs3b1eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adepss3b1eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adsqs3b1eW = stablehlo.sqrt %advhs3b1eW : tensor<3072x768x1x1xf32>
    %addens3b1eW = stablehlo.add %adsqs3b1eW, %adepss3b1eW : tensor<3072x768x1x1xf32>
    %adrats3b1eW = stablehlo.divide %admhs3b1eW, %addens3b1eW : tensor<3072x768x1x1xf32>
    %adsts3b1eW = stablehlo.multiply %adlrs3b1eW, %adrats3b1eW : tensor<3072x768x1x1xf32>
    %adsubs3b1eW = stablehlo.subtract %s3b1eW, %adsts3b1eW : tensor<3072x768x1x1xf32>
    %adwds3b1eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adwdlrs3b1eW = stablehlo.multiply %adwds3b1eW, %adlrs3b1eW : tensor<3072x768x1x1xf32>
    %adwdps3b1eW = stablehlo.multiply %adwdlrs3b1eW, %s3b1eW : tensor<3072x768x1x1xf32>
    %adnews3b1eW = stablehlo.subtract %adsubs3b1eW, %adwdps3b1eW : tensor<3072x768x1x1xf32>
    %adb1s3b1eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adob1s3b1eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %admss3b1eb = stablehlo.multiply %adb1s3b1eb, %s3b1ebm : tensor<3072xf32>
    %admgs3b1eb = stablehlo.multiply %adob1s3b1eb, %s3b1deb : tensor<3072xf32>
    %admns3b1eb = stablehlo.add %admss3b1eb, %admgs3b1eb : tensor<3072xf32>
    %adb2s3b1eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adob2s3b1eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %advss3b1eb = stablehlo.multiply %adb2s3b1eb, %s3b1ebv : tensor<3072xf32>
    %adg2s3b1eb = stablehlo.multiply %s3b1deb, %s3b1deb : tensor<3072xf32>
    %advgs3b1eb = stablehlo.multiply %adob2s3b1eb, %adg2s3b1eb : tensor<3072xf32>
    %advns3b1eb = stablehlo.add %advss3b1eb, %advgs3b1eb : tensor<3072xf32>
    %adbc1s3b1eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adbc2s3b1eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %admhs3b1eb = stablehlo.divide %admns3b1eb, %adbc1s3b1eb : tensor<3072xf32>
    %advhs3b1eb = stablehlo.divide %advns3b1eb, %adbc2s3b1eb : tensor<3072xf32>
    %adlrs3b1eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adepss3b1eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adsqs3b1eb = stablehlo.sqrt %advhs3b1eb : tensor<3072xf32>
    %addens3b1eb = stablehlo.add %adsqs3b1eb, %adepss3b1eb : tensor<3072xf32>
    %adrats3b1eb = stablehlo.divide %admhs3b1eb, %addens3b1eb : tensor<3072xf32>
    %adsts3b1eb = stablehlo.multiply %adlrs3b1eb, %adrats3b1eb : tensor<3072xf32>
    %adsubs3b1eb = stablehlo.subtract %s3b1eb, %adsts3b1eb : tensor<3072xf32>
    %adwds3b1eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adwdlrs3b1eb = stablehlo.multiply %adwds3b1eb, %adlrs3b1eb : tensor<3072xf32>
    %adwdps3b1eb = stablehlo.multiply %adwdlrs3b1eb, %s3b1eb : tensor<3072xf32>
    %adnews3b1eb = stablehlo.subtract %adsubs3b1eb, %adwdps3b1eb : tensor<3072xf32>
    %adb1s3b1pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adob1s3b1pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %admss3b1pW = stablehlo.multiply %adb1s3b1pW, %s3b1pWm : tensor<768x3072x1x1xf32>
    %admgs3b1pW = stablehlo.multiply %adob1s3b1pW, %s3b1dpW : tensor<768x3072x1x1xf32>
    %admns3b1pW = stablehlo.add %admss3b1pW, %admgs3b1pW : tensor<768x3072x1x1xf32>
    %adb2s3b1pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adob2s3b1pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %advss3b1pW = stablehlo.multiply %adb2s3b1pW, %s3b1pWv : tensor<768x3072x1x1xf32>
    %adg2s3b1pW = stablehlo.multiply %s3b1dpW, %s3b1dpW : tensor<768x3072x1x1xf32>
    %advgs3b1pW = stablehlo.multiply %adob2s3b1pW, %adg2s3b1pW : tensor<768x3072x1x1xf32>
    %advns3b1pW = stablehlo.add %advss3b1pW, %advgs3b1pW : tensor<768x3072x1x1xf32>
    %adbc1s3b1pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adbc2s3b1pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %admhs3b1pW = stablehlo.divide %admns3b1pW, %adbc1s3b1pW : tensor<768x3072x1x1xf32>
    %advhs3b1pW = stablehlo.divide %advns3b1pW, %adbc2s3b1pW : tensor<768x3072x1x1xf32>
    %adlrs3b1pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adepss3b1pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adsqs3b1pW = stablehlo.sqrt %advhs3b1pW : tensor<768x3072x1x1xf32>
    %addens3b1pW = stablehlo.add %adsqs3b1pW, %adepss3b1pW : tensor<768x3072x1x1xf32>
    %adrats3b1pW = stablehlo.divide %admhs3b1pW, %addens3b1pW : tensor<768x3072x1x1xf32>
    %adsts3b1pW = stablehlo.multiply %adlrs3b1pW, %adrats3b1pW : tensor<768x3072x1x1xf32>
    %adsubs3b1pW = stablehlo.subtract %s3b1pW, %adsts3b1pW : tensor<768x3072x1x1xf32>
    %adwds3b1pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adwdlrs3b1pW = stablehlo.multiply %adwds3b1pW, %adlrs3b1pW : tensor<768x3072x1x1xf32>
    %adwdps3b1pW = stablehlo.multiply %adwdlrs3b1pW, %s3b1pW : tensor<768x3072x1x1xf32>
    %adnews3b1pW = stablehlo.subtract %adsubs3b1pW, %adwdps3b1pW : tensor<768x3072x1x1xf32>
    %adb1s3b1pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1s3b1pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admss3b1pb = stablehlo.multiply %adb1s3b1pb, %s3b1pbm : tensor<768xf32>
    %admgs3b1pb = stablehlo.multiply %adob1s3b1pb, %s3b1dpb : tensor<768xf32>
    %admns3b1pb = stablehlo.add %admss3b1pb, %admgs3b1pb : tensor<768xf32>
    %adb2s3b1pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2s3b1pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advss3b1pb = stablehlo.multiply %adb2s3b1pb, %s3b1pbv : tensor<768xf32>
    %adg2s3b1pb = stablehlo.multiply %s3b1dpb, %s3b1dpb : tensor<768xf32>
    %advgs3b1pb = stablehlo.multiply %adob2s3b1pb, %adg2s3b1pb : tensor<768xf32>
    %advns3b1pb = stablehlo.add %advss3b1pb, %advgs3b1pb : tensor<768xf32>
    %adbc1s3b1pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2s3b1pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhs3b1pb = stablehlo.divide %admns3b1pb, %adbc1s3b1pb : tensor<768xf32>
    %advhs3b1pb = stablehlo.divide %advns3b1pb, %adbc2s3b1pb : tensor<768xf32>
    %adlrs3b1pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepss3b1pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqs3b1pb = stablehlo.sqrt %advhs3b1pb : tensor<768xf32>
    %addens3b1pb = stablehlo.add %adsqs3b1pb, %adepss3b1pb : tensor<768xf32>
    %adrats3b1pb = stablehlo.divide %admhs3b1pb, %addens3b1pb : tensor<768xf32>
    %adsts3b1pb = stablehlo.multiply %adlrs3b1pb, %adrats3b1pb : tensor<768xf32>
    %adsubs3b1pb = stablehlo.subtract %s3b1pb, %adsts3b1pb : tensor<768xf32>
    %adwds3b1pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrs3b1pb = stablehlo.multiply %adwds3b1pb, %adlrs3b1pb : tensor<768xf32>
    %adwdps3b1pb = stablehlo.multiply %adwdlrs3b1pb, %s3b1pb : tensor<768xf32>
    %adnews3b1pb = stablehlo.subtract %adsubs3b1pb, %adwdps3b1pb : tensor<768xf32>
    %adb1s3b1lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1s3b1lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admss3b1lg = stablehlo.multiply %adb1s3b1lg, %s3b1lgm : tensor<768xf32>
    %admgs3b1lg = stablehlo.multiply %adob1s3b1lg, %s3b1dlsdg : tensor<768xf32>
    %admns3b1lg = stablehlo.add %admss3b1lg, %admgs3b1lg : tensor<768xf32>
    %adb2s3b1lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2s3b1lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advss3b1lg = stablehlo.multiply %adb2s3b1lg, %s3b1lgv : tensor<768xf32>
    %adg2s3b1lg = stablehlo.multiply %s3b1dlsdg, %s3b1dlsdg : tensor<768xf32>
    %advgs3b1lg = stablehlo.multiply %adob2s3b1lg, %adg2s3b1lg : tensor<768xf32>
    %advns3b1lg = stablehlo.add %advss3b1lg, %advgs3b1lg : tensor<768xf32>
    %adbc1s3b1lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2s3b1lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhs3b1lg = stablehlo.divide %admns3b1lg, %adbc1s3b1lg : tensor<768xf32>
    %advhs3b1lg = stablehlo.divide %advns3b1lg, %adbc2s3b1lg : tensor<768xf32>
    %adlrs3b1lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepss3b1lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqs3b1lg = stablehlo.sqrt %advhs3b1lg : tensor<768xf32>
    %addens3b1lg = stablehlo.add %adsqs3b1lg, %adepss3b1lg : tensor<768xf32>
    %adrats3b1lg = stablehlo.divide %admhs3b1lg, %addens3b1lg : tensor<768xf32>
    %adsts3b1lg = stablehlo.multiply %adlrs3b1lg, %adrats3b1lg : tensor<768xf32>
    %adsubs3b1lg = stablehlo.subtract %s3b1lg, %adsts3b1lg : tensor<768xf32>
    %adwds3b1lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrs3b1lg = stablehlo.multiply %adwds3b1lg, %adlrs3b1lg : tensor<768xf32>
    %adwdps3b1lg = stablehlo.multiply %adwdlrs3b1lg, %s3b1lg : tensor<768xf32>
    %adnews3b1lg = stablehlo.subtract %adsubs3b1lg, %adwdps3b1lg : tensor<768xf32>
    %adb1s3b2dW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adob1s3b2dW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %admss3b2dW = stablehlo.multiply %adb1s3b2dW, %s3b2dWm : tensor<768x1x7x7xf32>
    %admgs3b2dW = stablehlo.multiply %adob1s3b2dW, %s3b2ddW : tensor<768x1x7x7xf32>
    %admns3b2dW = stablehlo.add %admss3b2dW, %admgs3b2dW : tensor<768x1x7x7xf32>
    %adb2s3b2dW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adob2s3b2dW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %advss3b2dW = stablehlo.multiply %adb2s3b2dW, %s3b2dWv : tensor<768x1x7x7xf32>
    %adg2s3b2dW = stablehlo.multiply %s3b2ddW, %s3b2ddW : tensor<768x1x7x7xf32>
    %advgs3b2dW = stablehlo.multiply %adob2s3b2dW, %adg2s3b2dW : tensor<768x1x7x7xf32>
    %advns3b2dW = stablehlo.add %advss3b2dW, %advgs3b2dW : tensor<768x1x7x7xf32>
    %adbc1s3b2dW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adbc2s3b2dW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %admhs3b2dW = stablehlo.divide %admns3b2dW, %adbc1s3b2dW : tensor<768x1x7x7xf32>
    %advhs3b2dW = stablehlo.divide %advns3b2dW, %adbc2s3b2dW : tensor<768x1x7x7xf32>
    %adlrs3b2dW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adepss3b2dW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adsqs3b2dW = stablehlo.sqrt %advhs3b2dW : tensor<768x1x7x7xf32>
    %addens3b2dW = stablehlo.add %adsqs3b2dW, %adepss3b2dW : tensor<768x1x7x7xf32>
    %adrats3b2dW = stablehlo.divide %admhs3b2dW, %addens3b2dW : tensor<768x1x7x7xf32>
    %adsts3b2dW = stablehlo.multiply %adlrs3b2dW, %adrats3b2dW : tensor<768x1x7x7xf32>
    %adsubs3b2dW = stablehlo.subtract %s3b2dW, %adsts3b2dW : tensor<768x1x7x7xf32>
    %adwds3b2dW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %adwdlrs3b2dW = stablehlo.multiply %adwds3b2dW, %adlrs3b2dW : tensor<768x1x7x7xf32>
    %adwdps3b2dW = stablehlo.multiply %adwdlrs3b2dW, %s3b2dW : tensor<768x1x7x7xf32>
    %adnews3b2dW = stablehlo.subtract %adsubs3b2dW, %adwdps3b2dW : tensor<768x1x7x7xf32>
    %adb1s3b2db = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1s3b2db = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admss3b2db = stablehlo.multiply %adb1s3b2db, %s3b2dbm : tensor<768xf32>
    %admgs3b2db = stablehlo.multiply %adob1s3b2db, %s3b2ddb : tensor<768xf32>
    %admns3b2db = stablehlo.add %admss3b2db, %admgs3b2db : tensor<768xf32>
    %adb2s3b2db = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2s3b2db = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advss3b2db = stablehlo.multiply %adb2s3b2db, %s3b2dbv : tensor<768xf32>
    %adg2s3b2db = stablehlo.multiply %s3b2ddb, %s3b2ddb : tensor<768xf32>
    %advgs3b2db = stablehlo.multiply %adob2s3b2db, %adg2s3b2db : tensor<768xf32>
    %advns3b2db = stablehlo.add %advss3b2db, %advgs3b2db : tensor<768xf32>
    %adbc1s3b2db = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2s3b2db = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhs3b2db = stablehlo.divide %admns3b2db, %adbc1s3b2db : tensor<768xf32>
    %advhs3b2db = stablehlo.divide %advns3b2db, %adbc2s3b2db : tensor<768xf32>
    %adlrs3b2db = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepss3b2db = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqs3b2db = stablehlo.sqrt %advhs3b2db : tensor<768xf32>
    %addens3b2db = stablehlo.add %adsqs3b2db, %adepss3b2db : tensor<768xf32>
    %adrats3b2db = stablehlo.divide %admhs3b2db, %addens3b2db : tensor<768xf32>
    %adsts3b2db = stablehlo.multiply %adlrs3b2db, %adrats3b2db : tensor<768xf32>
    %adsubs3b2db = stablehlo.subtract %s3b2db, %adsts3b2db : tensor<768xf32>
    %adwds3b2db = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrs3b2db = stablehlo.multiply %adwds3b2db, %adlrs3b2db : tensor<768xf32>
    %adwdps3b2db = stablehlo.multiply %adwdlrs3b2db, %s3b2db : tensor<768xf32>
    %adnews3b2db = stablehlo.subtract %adsubs3b2db, %adwdps3b2db : tensor<768xf32>
    %adb1s3b2ng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s3b2ng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss3b2ng = stablehlo.multiply %adb1s3b2ng, %s3b2ngm : tensor<f32>
    %admgs3b2ng = stablehlo.multiply %adob1s3b2ng, %s3b2dndg : tensor<f32>
    %admns3b2ng = stablehlo.add %admss3b2ng, %admgs3b2ng : tensor<f32>
    %adb2s3b2ng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s3b2ng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss3b2ng = stablehlo.multiply %adb2s3b2ng, %s3b2ngv : tensor<f32>
    %adg2s3b2ng = stablehlo.multiply %s3b2dndg, %s3b2dndg : tensor<f32>
    %advgs3b2ng = stablehlo.multiply %adob2s3b2ng, %adg2s3b2ng : tensor<f32>
    %advns3b2ng = stablehlo.add %advss3b2ng, %advgs3b2ng : tensor<f32>
    %adbc1s3b2ng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s3b2ng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs3b2ng = stablehlo.divide %admns3b2ng, %adbc1s3b2ng : tensor<f32>
    %advhs3b2ng = stablehlo.divide %advns3b2ng, %adbc2s3b2ng : tensor<f32>
    %adlrs3b2ng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss3b2ng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs3b2ng = stablehlo.sqrt %advhs3b2ng : tensor<f32>
    %addens3b2ng = stablehlo.add %adsqs3b2ng, %adepss3b2ng : tensor<f32>
    %adrats3b2ng = stablehlo.divide %admhs3b2ng, %addens3b2ng : tensor<f32>
    %adsts3b2ng = stablehlo.multiply %adlrs3b2ng, %adrats3b2ng : tensor<f32>
    %adsubs3b2ng = stablehlo.subtract %s3b2ng, %adsts3b2ng : tensor<f32>
    %adwds3b2ng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs3b2ng = stablehlo.multiply %adwds3b2ng, %adlrs3b2ng : tensor<f32>
    %adwdps3b2ng = stablehlo.multiply %adwdlrs3b2ng, %s3b2ng : tensor<f32>
    %adnews3b2ng = stablehlo.subtract %adsubs3b2ng, %adwdps3b2ng : tensor<f32>
    %adb1s3b2nbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1s3b2nbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admss3b2nbt = stablehlo.multiply %adb1s3b2nbt, %s3b2nbtm : tensor<f32>
    %admgs3b2nbt = stablehlo.multiply %adob1s3b2nbt, %s3b2dndb : tensor<f32>
    %admns3b2nbt = stablehlo.add %admss3b2nbt, %admgs3b2nbt : tensor<f32>
    %adb2s3b2nbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2s3b2nbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advss3b2nbt = stablehlo.multiply %adb2s3b2nbt, %s3b2nbtv : tensor<f32>
    %adg2s3b2nbt = stablehlo.multiply %s3b2dndb, %s3b2dndb : tensor<f32>
    %advgs3b2nbt = stablehlo.multiply %adob2s3b2nbt, %adg2s3b2nbt : tensor<f32>
    %advns3b2nbt = stablehlo.add %advss3b2nbt, %advgs3b2nbt : tensor<f32>
    %adbc1s3b2nbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2s3b2nbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhs3b2nbt = stablehlo.divide %admns3b2nbt, %adbc1s3b2nbt : tensor<f32>
    %advhs3b2nbt = stablehlo.divide %advns3b2nbt, %adbc2s3b2nbt : tensor<f32>
    %adlrs3b2nbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepss3b2nbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqs3b2nbt = stablehlo.sqrt %advhs3b2nbt : tensor<f32>
    %addens3b2nbt = stablehlo.add %adsqs3b2nbt, %adepss3b2nbt : tensor<f32>
    %adrats3b2nbt = stablehlo.divide %admhs3b2nbt, %addens3b2nbt : tensor<f32>
    %adsts3b2nbt = stablehlo.multiply %adlrs3b2nbt, %adrats3b2nbt : tensor<f32>
    %adsubs3b2nbt = stablehlo.subtract %s3b2nbt, %adsts3b2nbt : tensor<f32>
    %adwds3b2nbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrs3b2nbt = stablehlo.multiply %adwds3b2nbt, %adlrs3b2nbt : tensor<f32>
    %adwdps3b2nbt = stablehlo.multiply %adwdlrs3b2nbt, %s3b2nbt : tensor<f32>
    %adnews3b2nbt = stablehlo.subtract %adsubs3b2nbt, %adwdps3b2nbt : tensor<f32>
    %adb1s3b2eW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adob1s3b2eW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %admss3b2eW = stablehlo.multiply %adb1s3b2eW, %s3b2eWm : tensor<3072x768x1x1xf32>
    %admgs3b2eW = stablehlo.multiply %adob1s3b2eW, %s3b2deW : tensor<3072x768x1x1xf32>
    %admns3b2eW = stablehlo.add %admss3b2eW, %admgs3b2eW : tensor<3072x768x1x1xf32>
    %adb2s3b2eW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adob2s3b2eW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %advss3b2eW = stablehlo.multiply %adb2s3b2eW, %s3b2eWv : tensor<3072x768x1x1xf32>
    %adg2s3b2eW = stablehlo.multiply %s3b2deW, %s3b2deW : tensor<3072x768x1x1xf32>
    %advgs3b2eW = stablehlo.multiply %adob2s3b2eW, %adg2s3b2eW : tensor<3072x768x1x1xf32>
    %advns3b2eW = stablehlo.add %advss3b2eW, %advgs3b2eW : tensor<3072x768x1x1xf32>
    %adbc1s3b2eW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adbc2s3b2eW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %admhs3b2eW = stablehlo.divide %admns3b2eW, %adbc1s3b2eW : tensor<3072x768x1x1xf32>
    %advhs3b2eW = stablehlo.divide %advns3b2eW, %adbc2s3b2eW : tensor<3072x768x1x1xf32>
    %adlrs3b2eW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adepss3b2eW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adsqs3b2eW = stablehlo.sqrt %advhs3b2eW : tensor<3072x768x1x1xf32>
    %addens3b2eW = stablehlo.add %adsqs3b2eW, %adepss3b2eW : tensor<3072x768x1x1xf32>
    %adrats3b2eW = stablehlo.divide %admhs3b2eW, %addens3b2eW : tensor<3072x768x1x1xf32>
    %adsts3b2eW = stablehlo.multiply %adlrs3b2eW, %adrats3b2eW : tensor<3072x768x1x1xf32>
    %adsubs3b2eW = stablehlo.subtract %s3b2eW, %adsts3b2eW : tensor<3072x768x1x1xf32>
    %adwds3b2eW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %adwdlrs3b2eW = stablehlo.multiply %adwds3b2eW, %adlrs3b2eW : tensor<3072x768x1x1xf32>
    %adwdps3b2eW = stablehlo.multiply %adwdlrs3b2eW, %s3b2eW : tensor<3072x768x1x1xf32>
    %adnews3b2eW = stablehlo.subtract %adsubs3b2eW, %adwdps3b2eW : tensor<3072x768x1x1xf32>
    %adb1s3b2eb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adob1s3b2eb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %admss3b2eb = stablehlo.multiply %adb1s3b2eb, %s3b2ebm : tensor<3072xf32>
    %admgs3b2eb = stablehlo.multiply %adob1s3b2eb, %s3b2deb : tensor<3072xf32>
    %admns3b2eb = stablehlo.add %admss3b2eb, %admgs3b2eb : tensor<3072xf32>
    %adb2s3b2eb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adob2s3b2eb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %advss3b2eb = stablehlo.multiply %adb2s3b2eb, %s3b2ebv : tensor<3072xf32>
    %adg2s3b2eb = stablehlo.multiply %s3b2deb, %s3b2deb : tensor<3072xf32>
    %advgs3b2eb = stablehlo.multiply %adob2s3b2eb, %adg2s3b2eb : tensor<3072xf32>
    %advns3b2eb = stablehlo.add %advss3b2eb, %advgs3b2eb : tensor<3072xf32>
    %adbc1s3b2eb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adbc2s3b2eb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %admhs3b2eb = stablehlo.divide %admns3b2eb, %adbc1s3b2eb : tensor<3072xf32>
    %advhs3b2eb = stablehlo.divide %advns3b2eb, %adbc2s3b2eb : tensor<3072xf32>
    %adlrs3b2eb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adepss3b2eb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adsqs3b2eb = stablehlo.sqrt %advhs3b2eb : tensor<3072xf32>
    %addens3b2eb = stablehlo.add %adsqs3b2eb, %adepss3b2eb : tensor<3072xf32>
    %adrats3b2eb = stablehlo.divide %admhs3b2eb, %addens3b2eb : tensor<3072xf32>
    %adsts3b2eb = stablehlo.multiply %adlrs3b2eb, %adrats3b2eb : tensor<3072xf32>
    %adsubs3b2eb = stablehlo.subtract %s3b2eb, %adsts3b2eb : tensor<3072xf32>
    %adwds3b2eb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %adwdlrs3b2eb = stablehlo.multiply %adwds3b2eb, %adlrs3b2eb : tensor<3072xf32>
    %adwdps3b2eb = stablehlo.multiply %adwdlrs3b2eb, %s3b2eb : tensor<3072xf32>
    %adnews3b2eb = stablehlo.subtract %adsubs3b2eb, %adwdps3b2eb : tensor<3072xf32>
    %adb1s3b2pW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adob1s3b2pW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %admss3b2pW = stablehlo.multiply %adb1s3b2pW, %s3b2pWm : tensor<768x3072x1x1xf32>
    %admgs3b2pW = stablehlo.multiply %adob1s3b2pW, %s3b2dpW : tensor<768x3072x1x1xf32>
    %admns3b2pW = stablehlo.add %admss3b2pW, %admgs3b2pW : tensor<768x3072x1x1xf32>
    %adb2s3b2pW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adob2s3b2pW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %advss3b2pW = stablehlo.multiply %adb2s3b2pW, %s3b2pWv : tensor<768x3072x1x1xf32>
    %adg2s3b2pW = stablehlo.multiply %s3b2dpW, %s3b2dpW : tensor<768x3072x1x1xf32>
    %advgs3b2pW = stablehlo.multiply %adob2s3b2pW, %adg2s3b2pW : tensor<768x3072x1x1xf32>
    %advns3b2pW = stablehlo.add %advss3b2pW, %advgs3b2pW : tensor<768x3072x1x1xf32>
    %adbc1s3b2pW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adbc2s3b2pW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %admhs3b2pW = stablehlo.divide %admns3b2pW, %adbc1s3b2pW : tensor<768x3072x1x1xf32>
    %advhs3b2pW = stablehlo.divide %advns3b2pW, %adbc2s3b2pW : tensor<768x3072x1x1xf32>
    %adlrs3b2pW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adepss3b2pW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adsqs3b2pW = stablehlo.sqrt %advhs3b2pW : tensor<768x3072x1x1xf32>
    %addens3b2pW = stablehlo.add %adsqs3b2pW, %adepss3b2pW : tensor<768x3072x1x1xf32>
    %adrats3b2pW = stablehlo.divide %admhs3b2pW, %addens3b2pW : tensor<768x3072x1x1xf32>
    %adsts3b2pW = stablehlo.multiply %adlrs3b2pW, %adrats3b2pW : tensor<768x3072x1x1xf32>
    %adsubs3b2pW = stablehlo.subtract %s3b2pW, %adsts3b2pW : tensor<768x3072x1x1xf32>
    %adwds3b2pW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %adwdlrs3b2pW = stablehlo.multiply %adwds3b2pW, %adlrs3b2pW : tensor<768x3072x1x1xf32>
    %adwdps3b2pW = stablehlo.multiply %adwdlrs3b2pW, %s3b2pW : tensor<768x3072x1x1xf32>
    %adnews3b2pW = stablehlo.subtract %adsubs3b2pW, %adwdps3b2pW : tensor<768x3072x1x1xf32>
    %adb1s3b2pb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1s3b2pb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admss3b2pb = stablehlo.multiply %adb1s3b2pb, %s3b2pbm : tensor<768xf32>
    %admgs3b2pb = stablehlo.multiply %adob1s3b2pb, %s3b2dpb : tensor<768xf32>
    %admns3b2pb = stablehlo.add %admss3b2pb, %admgs3b2pb : tensor<768xf32>
    %adb2s3b2pb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2s3b2pb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advss3b2pb = stablehlo.multiply %adb2s3b2pb, %s3b2pbv : tensor<768xf32>
    %adg2s3b2pb = stablehlo.multiply %s3b2dpb, %s3b2dpb : tensor<768xf32>
    %advgs3b2pb = stablehlo.multiply %adob2s3b2pb, %adg2s3b2pb : tensor<768xf32>
    %advns3b2pb = stablehlo.add %advss3b2pb, %advgs3b2pb : tensor<768xf32>
    %adbc1s3b2pb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2s3b2pb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhs3b2pb = stablehlo.divide %admns3b2pb, %adbc1s3b2pb : tensor<768xf32>
    %advhs3b2pb = stablehlo.divide %advns3b2pb, %adbc2s3b2pb : tensor<768xf32>
    %adlrs3b2pb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepss3b2pb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqs3b2pb = stablehlo.sqrt %advhs3b2pb : tensor<768xf32>
    %addens3b2pb = stablehlo.add %adsqs3b2pb, %adepss3b2pb : tensor<768xf32>
    %adrats3b2pb = stablehlo.divide %admhs3b2pb, %addens3b2pb : tensor<768xf32>
    %adsts3b2pb = stablehlo.multiply %adlrs3b2pb, %adrats3b2pb : tensor<768xf32>
    %adsubs3b2pb = stablehlo.subtract %s3b2pb, %adsts3b2pb : tensor<768xf32>
    %adwds3b2pb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrs3b2pb = stablehlo.multiply %adwds3b2pb, %adlrs3b2pb : tensor<768xf32>
    %adwdps3b2pb = stablehlo.multiply %adwdlrs3b2pb, %s3b2pb : tensor<768xf32>
    %adnews3b2pb = stablehlo.subtract %adsubs3b2pb, %adwdps3b2pb : tensor<768xf32>
    %adb1s3b2lg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob1s3b2lg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admss3b2lg = stablehlo.multiply %adb1s3b2lg, %s3b2lgm : tensor<768xf32>
    %admgs3b2lg = stablehlo.multiply %adob1s3b2lg, %s3b2dlsdg : tensor<768xf32>
    %admns3b2lg = stablehlo.add %admss3b2lg, %admgs3b2lg : tensor<768xf32>
    %adb2s3b2lg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adob2s3b2lg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %advss3b2lg = stablehlo.multiply %adb2s3b2lg, %s3b2lgv : tensor<768xf32>
    %adg2s3b2lg = stablehlo.multiply %s3b2dlsdg, %s3b2dlsdg : tensor<768xf32>
    %advgs3b2lg = stablehlo.multiply %adob2s3b2lg, %adg2s3b2lg : tensor<768xf32>
    %advns3b2lg = stablehlo.add %advss3b2lg, %advgs3b2lg : tensor<768xf32>
    %adbc1s3b2lg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adbc2s3b2lg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %admhs3b2lg = stablehlo.divide %admns3b2lg, %adbc1s3b2lg : tensor<768xf32>
    %advhs3b2lg = stablehlo.divide %advns3b2lg, %adbc2s3b2lg : tensor<768xf32>
    %adlrs3b2lg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adepss3b2lg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adsqs3b2lg = stablehlo.sqrt %advhs3b2lg : tensor<768xf32>
    %addens3b2lg = stablehlo.add %adsqs3b2lg, %adepss3b2lg : tensor<768xf32>
    %adrats3b2lg = stablehlo.divide %admhs3b2lg, %addens3b2lg : tensor<768xf32>
    %adsts3b2lg = stablehlo.multiply %adlrs3b2lg, %adrats3b2lg : tensor<768xf32>
    %adsubs3b2lg = stablehlo.subtract %s3b2lg, %adsts3b2lg : tensor<768xf32>
    %adwds3b2lg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %adwdlrs3b2lg = stablehlo.multiply %adwds3b2lg, %adlrs3b2lg : tensor<768xf32>
    %adwdps3b2lg = stablehlo.multiply %adwdlrs3b2lg, %s3b2lg : tensor<768xf32>
    %adnews3b2lg = stablehlo.subtract %adsubs3b2lg, %adwdps3b2lg : tensor<768xf32>
    %adb1hng = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1hng = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admshng = stablehlo.multiply %adb1hng, %hngm : tensor<f32>
    %admghng = stablehlo.multiply %adob1hng, %hddg : tensor<f32>
    %admnhng = stablehlo.add %admshng, %admghng : tensor<f32>
    %adb2hng = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2hng = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advshng = stablehlo.multiply %adb2hng, %hngv : tensor<f32>
    %adg2hng = stablehlo.multiply %hddg, %hddg : tensor<f32>
    %advghng = stablehlo.multiply %adob2hng, %adg2hng : tensor<f32>
    %advnhng = stablehlo.add %advshng, %advghng : tensor<f32>
    %adbc1hng = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2hng = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhhng = stablehlo.divide %admnhng, %adbc1hng : tensor<f32>
    %advhhng = stablehlo.divide %advnhng, %adbc2hng : tensor<f32>
    %adlrhng = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepshng = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqhng = stablehlo.sqrt %advhhng : tensor<f32>
    %addenhng = stablehlo.add %adsqhng, %adepshng : tensor<f32>
    %adrathng = stablehlo.divide %admhhng, %addenhng : tensor<f32>
    %adsthng = stablehlo.multiply %adlrhng, %adrathng : tensor<f32>
    %adsubhng = stablehlo.subtract %hng, %adsthng : tensor<f32>
    %adwdhng = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrhng = stablehlo.multiply %adwdhng, %adlrhng : tensor<f32>
    %adwdphng = stablehlo.multiply %adwdlrhng, %hng : tensor<f32>
    %adnewhng = stablehlo.subtract %adsubhng, %adwdphng : tensor<f32>
    %adb1hnbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob1hnbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %admshnbt = stablehlo.multiply %adb1hnbt, %hnbtm : tensor<f32>
    %admghnbt = stablehlo.multiply %adob1hnbt, %hddb : tensor<f32>
    %admnhnbt = stablehlo.add %admshnbt, %admghnbt : tensor<f32>
    %adb2hnbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %adob2hnbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %advshnbt = stablehlo.multiply %adb2hnbt, %hnbtv : tensor<f32>
    %adg2hnbt = stablehlo.multiply %hddb, %hddb : tensor<f32>
    %advghnbt = stablehlo.multiply %adob2hnbt, %adg2hnbt : tensor<f32>
    %advnhnbt = stablehlo.add %advshnbt, %advghnbt : tensor<f32>
    %adbc1hnbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %adbc2hnbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %admhhnbt = stablehlo.divide %admnhnbt, %adbc1hnbt : tensor<f32>
    %advhhnbt = stablehlo.divide %advnhnbt, %adbc2hnbt : tensor<f32>
    %adlrhnbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %adepshnbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %adsqhnbt = stablehlo.sqrt %advhhnbt : tensor<f32>
    %addenhnbt = stablehlo.add %adsqhnbt, %adepshnbt : tensor<f32>
    %adrathnbt = stablehlo.divide %admhhnbt, %addenhnbt : tensor<f32>
    %adsthnbt = stablehlo.multiply %adlrhnbt, %adrathnbt : tensor<f32>
    %adsubhnbt = stablehlo.subtract %hnbt, %adsthnbt : tensor<f32>
    %adwdhnbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %adwdlrhnbt = stablehlo.multiply %adwdhnbt, %adlrhnbt : tensor<f32>
    %adwdphnbt = stablehlo.multiply %adwdlrhnbt, %hnbt : tensor<f32>
    %adnewhnbt = stablehlo.subtract %adsubhnbt, %adwdphnbt : tensor<f32>
    %adb1Wd = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %adob1Wd = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %admsWd = stablehlo.multiply %adb1Wd, %Wdm : tensor<768x10xf32>
    %admgWd = stablehlo.multiply %adob1Wd, %dWd : tensor<768x10xf32>
    %admnWd = stablehlo.add %admsWd, %admgWd : tensor<768x10xf32>
    %adb2Wd = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %adob2Wd = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %advsWd = stablehlo.multiply %adb2Wd, %Wdv : tensor<768x10xf32>
    %adg2Wd = stablehlo.multiply %dWd, %dWd : tensor<768x10xf32>
    %advgWd = stablehlo.multiply %adob2Wd, %adg2Wd : tensor<768x10xf32>
    %advnWd = stablehlo.add %advsWd, %advgWd : tensor<768x10xf32>
    %adbc1Wd = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %adbc2Wd = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %admhWd = stablehlo.divide %admnWd, %adbc1Wd : tensor<768x10xf32>
    %advhWd = stablehlo.divide %advnWd, %adbc2Wd : tensor<768x10xf32>
    %adlrWd = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %adepsWd = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %adsqWd = stablehlo.sqrt %advhWd : tensor<768x10xf32>
    %addenWd = stablehlo.add %adsqWd, %adepsWd : tensor<768x10xf32>
    %adratWd = stablehlo.divide %admhWd, %addenWd : tensor<768x10xf32>
    %adstWd = stablehlo.multiply %adlrWd, %adratWd : tensor<768x10xf32>
    %adsubWd = stablehlo.subtract %Wd, %adstWd : tensor<768x10xf32>
    %adwdWd = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %adwdlrWd = stablehlo.multiply %adwdWd, %adlrWd : tensor<768x10xf32>
    %adwdpWd = stablehlo.multiply %adwdlrWd, %Wd : tensor<768x10xf32>
    %adnewWd = stablehlo.subtract %adsubWd, %adwdpWd : tensor<768x10xf32>
    %adb1bd = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob1bd = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admsbd = stablehlo.multiply %adb1bd, %bdm : tensor<10xf32>
    %admgbd = stablehlo.multiply %adob1bd, %dbd : tensor<10xf32>
    %admnbd = stablehlo.add %admsbd, %admgbd : tensor<10xf32>
    %adb2bd = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob2bd = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %advsbd = stablehlo.multiply %adb2bd, %bdv : tensor<10xf32>
    %adg2bd = stablehlo.multiply %dbd, %dbd : tensor<10xf32>
    %advgbd = stablehlo.multiply %adob2bd, %adg2bd : tensor<10xf32>
    %advnbd = stablehlo.add %advsbd, %advgbd : tensor<10xf32>
    %adbc1bd = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adbc2bd = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admhbd = stablehlo.divide %admnbd, %adbc1bd : tensor<10xf32>
    %advhbd = stablehlo.divide %advnbd, %adbc2bd : tensor<10xf32>
    %adlrbd = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adepsbd = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adsqbd = stablehlo.sqrt %advhbd : tensor<10xf32>
    %addenbd = stablehlo.add %adsqbd, %adepsbd : tensor<10xf32>
    %adratbd = stablehlo.divide %admhbd, %addenbd : tensor<10xf32>
    %adstbd = stablehlo.multiply %adlrbd, %adratbd : tensor<10xf32>
    %adsubbd = stablehlo.subtract %bd, %adstbd : tensor<10xf32>
    %adwdbd = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adwdlrbd = stablehlo.multiply %adwdbd, %adlrbd : tensor<10xf32>
    %adwdpbd = stablehlo.multiply %adwdlrbd, %bd : tensor<10xf32>
    %adnewbd = stablehlo.subtract %adsubbd, %adwdpbd : tensor<10xf32>
    return %adnewpsW, %adnewpsb, %adnews0b0dW, %adnews0b0db, %adnews0b0ng, %adnews0b0nbt, %adnews0b0eW, %adnews0b0eb, %adnews0b0pW, %adnews0b0pb, %adnews0b0lg, %adnews0b1dW, %adnews0b1db, %adnews0b1ng, %adnews0b1nbt, %adnews0b1eW, %adnews0b1eb, %adnews0b1pW, %adnews0b1pb, %adnews0b1lg, %adnews0b2dW, %adnews0b2db, %adnews0b2ng, %adnews0b2nbt, %adnews0b2eW, %adnews0b2eb, %adnews0b2pW, %adnews0b2pb, %adnews0b2lg, %adnewd0ng, %adnewd0nbt, %adnewd0W, %adnewd0b, %adnews1b0dW, %adnews1b0db, %adnews1b0ng, %adnews1b0nbt, %adnews1b0eW, %adnews1b0eb, %adnews1b0pW, %adnews1b0pb, %adnews1b0lg, %adnews1b1dW, %adnews1b1db, %adnews1b1ng, %adnews1b1nbt, %adnews1b1eW, %adnews1b1eb, %adnews1b1pW, %adnews1b1pb, %adnews1b1lg, %adnews1b2dW, %adnews1b2db, %adnews1b2ng, %adnews1b2nbt, %adnews1b2eW, %adnews1b2eb, %adnews1b2pW, %adnews1b2pb, %adnews1b2lg, %adnewd1ng, %adnewd1nbt, %adnewd1W, %adnewd1b, %adnews2b0dW, %adnews2b0db, %adnews2b0ng, %adnews2b0nbt, %adnews2b0eW, %adnews2b0eb, %adnews2b0pW, %adnews2b0pb, %adnews2b0lg, %adnews2b1dW, %adnews2b1db, %adnews2b1ng, %adnews2b1nbt, %adnews2b1eW, %adnews2b1eb, %adnews2b1pW, %adnews2b1pb, %adnews2b1lg, %adnews2b2dW, %adnews2b2db, %adnews2b2ng, %adnews2b2nbt, %adnews2b2eW, %adnews2b2eb, %adnews2b2pW, %adnews2b2pb, %adnews2b2lg, %adnews2b3dW, %adnews2b3db, %adnews2b3ng, %adnews2b3nbt, %adnews2b3eW, %adnews2b3eb, %adnews2b3pW, %adnews2b3pb, %adnews2b3lg, %adnews2b4dW, %adnews2b4db, %adnews2b4ng, %adnews2b4nbt, %adnews2b4eW, %adnews2b4eb, %adnews2b4pW, %adnews2b4pb, %adnews2b4lg, %adnews2b5dW, %adnews2b5db, %adnews2b5ng, %adnews2b5nbt, %adnews2b5eW, %adnews2b5eb, %adnews2b5pW, %adnews2b5pb, %adnews2b5lg, %adnews2b6dW, %adnews2b6db, %adnews2b6ng, %adnews2b6nbt, %adnews2b6eW, %adnews2b6eb, %adnews2b6pW, %adnews2b6pb, %adnews2b6lg, %adnews2b7dW, %adnews2b7db, %adnews2b7ng, %adnews2b7nbt, %adnews2b7eW, %adnews2b7eb, %adnews2b7pW, %adnews2b7pb, %adnews2b7lg, %adnews2b8dW, %adnews2b8db, %adnews2b8ng, %adnews2b8nbt, %adnews2b8eW, %adnews2b8eb, %adnews2b8pW, %adnews2b8pb, %adnews2b8lg, %adnewd2ng, %adnewd2nbt, %adnewd2W, %adnewd2b, %adnews3b0dW, %adnews3b0db, %adnews3b0ng, %adnews3b0nbt, %adnews3b0eW, %adnews3b0eb, %adnews3b0pW, %adnews3b0pb, %adnews3b0lg, %adnews3b1dW, %adnews3b1db, %adnews3b1ng, %adnews3b1nbt, %adnews3b1eW, %adnews3b1eb, %adnews3b1pW, %adnews3b1pb, %adnews3b1lg, %adnews3b2dW, %adnews3b2db, %adnews3b2ng, %adnews3b2nbt, %adnews3b2eW, %adnews3b2eb, %adnews3b2pW, %adnews3b2pb, %adnews3b2lg, %adnewhng, %adnewhnbt, %adnewWd, %adnewbd, %admnpsW, %admnpsb, %admns0b0dW, %admns0b0db, %admns0b0ng, %admns0b0nbt, %admns0b0eW, %admns0b0eb, %admns0b0pW, %admns0b0pb, %admns0b0lg, %admns0b1dW, %admns0b1db, %admns0b1ng, %admns0b1nbt, %admns0b1eW, %admns0b1eb, %admns0b1pW, %admns0b1pb, %admns0b1lg, %admns0b2dW, %admns0b2db, %admns0b2ng, %admns0b2nbt, %admns0b2eW, %admns0b2eb, %admns0b2pW, %admns0b2pb, %admns0b2lg, %admnd0ng, %admnd0nbt, %admnd0W, %admnd0b, %admns1b0dW, %admns1b0db, %admns1b0ng, %admns1b0nbt, %admns1b0eW, %admns1b0eb, %admns1b0pW, %admns1b0pb, %admns1b0lg, %admns1b1dW, %admns1b1db, %admns1b1ng, %admns1b1nbt, %admns1b1eW, %admns1b1eb, %admns1b1pW, %admns1b1pb, %admns1b1lg, %admns1b2dW, %admns1b2db, %admns1b2ng, %admns1b2nbt, %admns1b2eW, %admns1b2eb, %admns1b2pW, %admns1b2pb, %admns1b2lg, %admnd1ng, %admnd1nbt, %admnd1W, %admnd1b, %admns2b0dW, %admns2b0db, %admns2b0ng, %admns2b0nbt, %admns2b0eW, %admns2b0eb, %admns2b0pW, %admns2b0pb, %admns2b0lg, %admns2b1dW, %admns2b1db, %admns2b1ng, %admns2b1nbt, %admns2b1eW, %admns2b1eb, %admns2b1pW, %admns2b1pb, %admns2b1lg, %admns2b2dW, %admns2b2db, %admns2b2ng, %admns2b2nbt, %admns2b2eW, %admns2b2eb, %admns2b2pW, %admns2b2pb, %admns2b2lg, %admns2b3dW, %admns2b3db, %admns2b3ng, %admns2b3nbt, %admns2b3eW, %admns2b3eb, %admns2b3pW, %admns2b3pb, %admns2b3lg, %admns2b4dW, %admns2b4db, %admns2b4ng, %admns2b4nbt, %admns2b4eW, %admns2b4eb, %admns2b4pW, %admns2b4pb, %admns2b4lg, %admns2b5dW, %admns2b5db, %admns2b5ng, %admns2b5nbt, %admns2b5eW, %admns2b5eb, %admns2b5pW, %admns2b5pb, %admns2b5lg, %admns2b6dW, %admns2b6db, %admns2b6ng, %admns2b6nbt, %admns2b6eW, %admns2b6eb, %admns2b6pW, %admns2b6pb, %admns2b6lg, %admns2b7dW, %admns2b7db, %admns2b7ng, %admns2b7nbt, %admns2b7eW, %admns2b7eb, %admns2b7pW, %admns2b7pb, %admns2b7lg, %admns2b8dW, %admns2b8db, %admns2b8ng, %admns2b8nbt, %admns2b8eW, %admns2b8eb, %admns2b8pW, %admns2b8pb, %admns2b8lg, %admnd2ng, %admnd2nbt, %admnd2W, %admnd2b, %admns3b0dW, %admns3b0db, %admns3b0ng, %admns3b0nbt, %admns3b0eW, %admns3b0eb, %admns3b0pW, %admns3b0pb, %admns3b0lg, %admns3b1dW, %admns3b1db, %admns3b1ng, %admns3b1nbt, %admns3b1eW, %admns3b1eb, %admns3b1pW, %admns3b1pb, %admns3b1lg, %admns3b2dW, %admns3b2db, %admns3b2ng, %admns3b2nbt, %admns3b2eW, %admns3b2eb, %admns3b2pW, %admns3b2pb, %admns3b2lg, %admnhng, %admnhnbt, %admnWd, %admnbd, %advnpsW, %advnpsb, %advns0b0dW, %advns0b0db, %advns0b0ng, %advns0b0nbt, %advns0b0eW, %advns0b0eb, %advns0b0pW, %advns0b0pb, %advns0b0lg, %advns0b1dW, %advns0b1db, %advns0b1ng, %advns0b1nbt, %advns0b1eW, %advns0b1eb, %advns0b1pW, %advns0b1pb, %advns0b1lg, %advns0b2dW, %advns0b2db, %advns0b2ng, %advns0b2nbt, %advns0b2eW, %advns0b2eb, %advns0b2pW, %advns0b2pb, %advns0b2lg, %advnd0ng, %advnd0nbt, %advnd0W, %advnd0b, %advns1b0dW, %advns1b0db, %advns1b0ng, %advns1b0nbt, %advns1b0eW, %advns1b0eb, %advns1b0pW, %advns1b0pb, %advns1b0lg, %advns1b1dW, %advns1b1db, %advns1b1ng, %advns1b1nbt, %advns1b1eW, %advns1b1eb, %advns1b1pW, %advns1b1pb, %advns1b1lg, %advns1b2dW, %advns1b2db, %advns1b2ng, %advns1b2nbt, %advns1b2eW, %advns1b2eb, %advns1b2pW, %advns1b2pb, %advns1b2lg, %advnd1ng, %advnd1nbt, %advnd1W, %advnd1b, %advns2b0dW, %advns2b0db, %advns2b0ng, %advns2b0nbt, %advns2b0eW, %advns2b0eb, %advns2b0pW, %advns2b0pb, %advns2b0lg, %advns2b1dW, %advns2b1db, %advns2b1ng, %advns2b1nbt, %advns2b1eW, %advns2b1eb, %advns2b1pW, %advns2b1pb, %advns2b1lg, %advns2b2dW, %advns2b2db, %advns2b2ng, %advns2b2nbt, %advns2b2eW, %advns2b2eb, %advns2b2pW, %advns2b2pb, %advns2b2lg, %advns2b3dW, %advns2b3db, %advns2b3ng, %advns2b3nbt, %advns2b3eW, %advns2b3eb, %advns2b3pW, %advns2b3pb, %advns2b3lg, %advns2b4dW, %advns2b4db, %advns2b4ng, %advns2b4nbt, %advns2b4eW, %advns2b4eb, %advns2b4pW, %advns2b4pb, %advns2b4lg, %advns2b5dW, %advns2b5db, %advns2b5ng, %advns2b5nbt, %advns2b5eW, %advns2b5eb, %advns2b5pW, %advns2b5pb, %advns2b5lg, %advns2b6dW, %advns2b6db, %advns2b6ng, %advns2b6nbt, %advns2b6eW, %advns2b6eb, %advns2b6pW, %advns2b6pb, %advns2b6lg, %advns2b7dW, %advns2b7db, %advns2b7ng, %advns2b7nbt, %advns2b7eW, %advns2b7eb, %advns2b7pW, %advns2b7pb, %advns2b7lg, %advns2b8dW, %advns2b8db, %advns2b8ng, %advns2b8nbt, %advns2b8eW, %advns2b8eb, %advns2b8pW, %advns2b8pb, %advns2b8lg, %advnd2ng, %advnd2nbt, %advnd2W, %advnd2b, %advns3b0dW, %advns3b0db, %advns3b0ng, %advns3b0nbt, %advns3b0eW, %advns3b0eb, %advns3b0pW, %advns3b0pb, %advns3b0lg, %advns3b1dW, %advns3b1db, %advns3b1ng, %advns3b1nbt, %advns3b1eW, %advns3b1eb, %advns3b1pW, %advns3b1pb, %advns3b1lg, %advns3b2dW, %advns3b2db, %advns3b2ng, %advns3b2nbt, %advns3b2eW, %advns3b2eb, %advns3b2pW, %advns3b2pb, %advns3b2lg, %advnhng, %advnhnbt, %advnWd, %advnbd, %loss, %bc1, %bc2 : tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
