module @m {
  func.func @convnext_adam_train_step(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<f32>, %s0b0nbt: tensor<f32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<f32>, %s0b1nbt: tensor<f32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<f32>, %s0b2nbt: tensor<f32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<f32>, %d0nbt: tensor<f32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<f32>, %s1b0nbt: tensor<f32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<f32>, %s1b1nbt: tensor<f32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<f32>, %s1b2nbt: tensor<f32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<f32>, %d1nbt: tensor<f32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<f32>, %s2b0nbt: tensor<f32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<f32>, %s2b1nbt: tensor<f32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<f32>, %s2b2nbt: tensor<f32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<f32>, %s2b3nbt: tensor<f32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<f32>, %s2b4nbt: tensor<f32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<f32>, %s2b5nbt: tensor<f32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<f32>, %s2b6nbt: tensor<f32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<f32>, %s2b7nbt: tensor<f32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<f32>, %s2b8nbt: tensor<f32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<f32>, %d2nbt: tensor<f32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<f32>, %s3b0nbt: tensor<f32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<f32>, %s3b1nbt: tensor<f32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<f32>, %s3b2nbt: tensor<f32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %hng: tensor<f32>, %hnbt: tensor<f32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>, %psWm: tensor<96x3x4x4xf32>, %psbm: tensor<96xf32>, %s0b0dWm: tensor<96x1x7x7xf32>, %s0b0dbm: tensor<96xf32>, %s0b0ngm: tensor<f32>, %s0b0nbtm: tensor<f32>, %s0b0eWm: tensor<384x96x1x1xf32>, %s0b0ebm: tensor<384xf32>, %s0b0pWm: tensor<96x384x1x1xf32>, %s0b0pbm: tensor<96xf32>, %s0b0lgm: tensor<96xf32>, %s0b1dWm: tensor<96x1x7x7xf32>, %s0b1dbm: tensor<96xf32>, %s0b1ngm: tensor<f32>, %s0b1nbtm: tensor<f32>, %s0b1eWm: tensor<384x96x1x1xf32>, %s0b1ebm: tensor<384xf32>, %s0b1pWm: tensor<96x384x1x1xf32>, %s0b1pbm: tensor<96xf32>, %s0b1lgm: tensor<96xf32>, %s0b2dWm: tensor<96x1x7x7xf32>, %s0b2dbm: tensor<96xf32>, %s0b2ngm: tensor<f32>, %s0b2nbtm: tensor<f32>, %s0b2eWm: tensor<384x96x1x1xf32>, %s0b2ebm: tensor<384xf32>, %s0b2pWm: tensor<96x384x1x1xf32>, %s0b2pbm: tensor<96xf32>, %s0b2lgm: tensor<96xf32>, %d0ngm: tensor<f32>, %d0nbtm: tensor<f32>, %d0Wm: tensor<192x96x2x2xf32>, %d0bm: tensor<192xf32>, %s1b0dWm: tensor<192x1x7x7xf32>, %s1b0dbm: tensor<192xf32>, %s1b0ngm: tensor<f32>, %s1b0nbtm: tensor<f32>, %s1b0eWm: tensor<768x192x1x1xf32>, %s1b0ebm: tensor<768xf32>, %s1b0pWm: tensor<192x768x1x1xf32>, %s1b0pbm: tensor<192xf32>, %s1b0lgm: tensor<192xf32>, %s1b1dWm: tensor<192x1x7x7xf32>, %s1b1dbm: tensor<192xf32>, %s1b1ngm: tensor<f32>, %s1b1nbtm: tensor<f32>, %s1b1eWm: tensor<768x192x1x1xf32>, %s1b1ebm: tensor<768xf32>, %s1b1pWm: tensor<192x768x1x1xf32>, %s1b1pbm: tensor<192xf32>, %s1b1lgm: tensor<192xf32>, %s1b2dWm: tensor<192x1x7x7xf32>, %s1b2dbm: tensor<192xf32>, %s1b2ngm: tensor<f32>, %s1b2nbtm: tensor<f32>, %s1b2eWm: tensor<768x192x1x1xf32>, %s1b2ebm: tensor<768xf32>, %s1b2pWm: tensor<192x768x1x1xf32>, %s1b2pbm: tensor<192xf32>, %s1b2lgm: tensor<192xf32>, %d1ngm: tensor<f32>, %d1nbtm: tensor<f32>, %d1Wm: tensor<384x192x2x2xf32>, %d1bm: tensor<384xf32>, %s2b0dWm: tensor<384x1x7x7xf32>, %s2b0dbm: tensor<384xf32>, %s2b0ngm: tensor<f32>, %s2b0nbtm: tensor<f32>, %s2b0eWm: tensor<1536x384x1x1xf32>, %s2b0ebm: tensor<1536xf32>, %s2b0pWm: tensor<384x1536x1x1xf32>, %s2b0pbm: tensor<384xf32>, %s2b0lgm: tensor<384xf32>, %s2b1dWm: tensor<384x1x7x7xf32>, %s2b1dbm: tensor<384xf32>, %s2b1ngm: tensor<f32>, %s2b1nbtm: tensor<f32>, %s2b1eWm: tensor<1536x384x1x1xf32>, %s2b1ebm: tensor<1536xf32>, %s2b1pWm: tensor<384x1536x1x1xf32>, %s2b1pbm: tensor<384xf32>, %s2b1lgm: tensor<384xf32>, %s2b2dWm: tensor<384x1x7x7xf32>, %s2b2dbm: tensor<384xf32>, %s2b2ngm: tensor<f32>, %s2b2nbtm: tensor<f32>, %s2b2eWm: tensor<1536x384x1x1xf32>, %s2b2ebm: tensor<1536xf32>, %s2b2pWm: tensor<384x1536x1x1xf32>, %s2b2pbm: tensor<384xf32>, %s2b2lgm: tensor<384xf32>, %s2b3dWm: tensor<384x1x7x7xf32>, %s2b3dbm: tensor<384xf32>, %s2b3ngm: tensor<f32>, %s2b3nbtm: tensor<f32>, %s2b3eWm: tensor<1536x384x1x1xf32>, %s2b3ebm: tensor<1536xf32>, %s2b3pWm: tensor<384x1536x1x1xf32>, %s2b3pbm: tensor<384xf32>, %s2b3lgm: tensor<384xf32>, %s2b4dWm: tensor<384x1x7x7xf32>, %s2b4dbm: tensor<384xf32>, %s2b4ngm: tensor<f32>, %s2b4nbtm: tensor<f32>, %s2b4eWm: tensor<1536x384x1x1xf32>, %s2b4ebm: tensor<1536xf32>, %s2b4pWm: tensor<384x1536x1x1xf32>, %s2b4pbm: tensor<384xf32>, %s2b4lgm: tensor<384xf32>, %s2b5dWm: tensor<384x1x7x7xf32>, %s2b5dbm: tensor<384xf32>, %s2b5ngm: tensor<f32>, %s2b5nbtm: tensor<f32>, %s2b5eWm: tensor<1536x384x1x1xf32>, %s2b5ebm: tensor<1536xf32>, %s2b5pWm: tensor<384x1536x1x1xf32>, %s2b5pbm: tensor<384xf32>, %s2b5lgm: tensor<384xf32>, %s2b6dWm: tensor<384x1x7x7xf32>, %s2b6dbm: tensor<384xf32>, %s2b6ngm: tensor<f32>, %s2b6nbtm: tensor<f32>, %s2b6eWm: tensor<1536x384x1x1xf32>, %s2b6ebm: tensor<1536xf32>, %s2b6pWm: tensor<384x1536x1x1xf32>, %s2b6pbm: tensor<384xf32>, %s2b6lgm: tensor<384xf32>, %s2b7dWm: tensor<384x1x7x7xf32>, %s2b7dbm: tensor<384xf32>, %s2b7ngm: tensor<f32>, %s2b7nbtm: tensor<f32>, %s2b7eWm: tensor<1536x384x1x1xf32>, %s2b7ebm: tensor<1536xf32>, %s2b7pWm: tensor<384x1536x1x1xf32>, %s2b7pbm: tensor<384xf32>, %s2b7lgm: tensor<384xf32>, %s2b8dWm: tensor<384x1x7x7xf32>, %s2b8dbm: tensor<384xf32>, %s2b8ngm: tensor<f32>, %s2b8nbtm: tensor<f32>, %s2b8eWm: tensor<1536x384x1x1xf32>, %s2b8ebm: tensor<1536xf32>, %s2b8pWm: tensor<384x1536x1x1xf32>, %s2b8pbm: tensor<384xf32>, %s2b8lgm: tensor<384xf32>, %d2ngm: tensor<f32>, %d2nbtm: tensor<f32>, %d2Wm: tensor<768x384x2x2xf32>, %d2bm: tensor<768xf32>, %s3b0dWm: tensor<768x1x7x7xf32>, %s3b0dbm: tensor<768xf32>, %s3b0ngm: tensor<f32>, %s3b0nbtm: tensor<f32>, %s3b0eWm: tensor<3072x768x1x1xf32>, %s3b0ebm: tensor<3072xf32>, %s3b0pWm: tensor<768x3072x1x1xf32>, %s3b0pbm: tensor<768xf32>, %s3b0lgm: tensor<768xf32>, %s3b1dWm: tensor<768x1x7x7xf32>, %s3b1dbm: tensor<768xf32>, %s3b1ngm: tensor<f32>, %s3b1nbtm: tensor<f32>, %s3b1eWm: tensor<3072x768x1x1xf32>, %s3b1ebm: tensor<3072xf32>, %s3b1pWm: tensor<768x3072x1x1xf32>, %s3b1pbm: tensor<768xf32>, %s3b1lgm: tensor<768xf32>, %s3b2dWm: tensor<768x1x7x7xf32>, %s3b2dbm: tensor<768xf32>, %s3b2ngm: tensor<f32>, %s3b2nbtm: tensor<f32>, %s3b2eWm: tensor<3072x768x1x1xf32>, %s3b2ebm: tensor<3072xf32>, %s3b2pWm: tensor<768x3072x1x1xf32>, %s3b2pbm: tensor<768xf32>, %s3b2lgm: tensor<768xf32>, %hngm: tensor<f32>, %hnbtm: tensor<f32>, %Wdm: tensor<768x10xf32>, %bdm: tensor<10xf32>, %psWv: tensor<96x3x4x4xf32>, %psbv: tensor<96xf32>, %s0b0dWv: tensor<96x1x7x7xf32>, %s0b0dbv: tensor<96xf32>, %s0b0ngv: tensor<f32>, %s0b0nbtv: tensor<f32>, %s0b0eWv: tensor<384x96x1x1xf32>, %s0b0ebv: tensor<384xf32>, %s0b0pWv: tensor<96x384x1x1xf32>, %s0b0pbv: tensor<96xf32>, %s0b0lgv: tensor<96xf32>, %s0b1dWv: tensor<96x1x7x7xf32>, %s0b1dbv: tensor<96xf32>, %s0b1ngv: tensor<f32>, %s0b1nbtv: tensor<f32>, %s0b1eWv: tensor<384x96x1x1xf32>, %s0b1ebv: tensor<384xf32>, %s0b1pWv: tensor<96x384x1x1xf32>, %s0b1pbv: tensor<96xf32>, %s0b1lgv: tensor<96xf32>, %s0b2dWv: tensor<96x1x7x7xf32>, %s0b2dbv: tensor<96xf32>, %s0b2ngv: tensor<f32>, %s0b2nbtv: tensor<f32>, %s0b2eWv: tensor<384x96x1x1xf32>, %s0b2ebv: tensor<384xf32>, %s0b2pWv: tensor<96x384x1x1xf32>, %s0b2pbv: tensor<96xf32>, %s0b2lgv: tensor<96xf32>, %d0ngv: tensor<f32>, %d0nbtv: tensor<f32>, %d0Wv: tensor<192x96x2x2xf32>, %d0bv: tensor<192xf32>, %s1b0dWv: tensor<192x1x7x7xf32>, %s1b0dbv: tensor<192xf32>, %s1b0ngv: tensor<f32>, %s1b0nbtv: tensor<f32>, %s1b0eWv: tensor<768x192x1x1xf32>, %s1b0ebv: tensor<768xf32>, %s1b0pWv: tensor<192x768x1x1xf32>, %s1b0pbv: tensor<192xf32>, %s1b0lgv: tensor<192xf32>, %s1b1dWv: tensor<192x1x7x7xf32>, %s1b1dbv: tensor<192xf32>, %s1b1ngv: tensor<f32>, %s1b1nbtv: tensor<f32>, %s1b1eWv: tensor<768x192x1x1xf32>, %s1b1ebv: tensor<768xf32>, %s1b1pWv: tensor<192x768x1x1xf32>, %s1b1pbv: tensor<192xf32>, %s1b1lgv: tensor<192xf32>, %s1b2dWv: tensor<192x1x7x7xf32>, %s1b2dbv: tensor<192xf32>, %s1b2ngv: tensor<f32>, %s1b2nbtv: tensor<f32>, %s1b2eWv: tensor<768x192x1x1xf32>, %s1b2ebv: tensor<768xf32>, %s1b2pWv: tensor<192x768x1x1xf32>, %s1b2pbv: tensor<192xf32>, %s1b2lgv: tensor<192xf32>, %d1ngv: tensor<f32>, %d1nbtv: tensor<f32>, %d1Wv: tensor<384x192x2x2xf32>, %d1bv: tensor<384xf32>, %s2b0dWv: tensor<384x1x7x7xf32>, %s2b0dbv: tensor<384xf32>, %s2b0ngv: tensor<f32>, %s2b0nbtv: tensor<f32>, %s2b0eWv: tensor<1536x384x1x1xf32>, %s2b0ebv: tensor<1536xf32>, %s2b0pWv: tensor<384x1536x1x1xf32>, %s2b0pbv: tensor<384xf32>, %s2b0lgv: tensor<384xf32>, %s2b1dWv: tensor<384x1x7x7xf32>, %s2b1dbv: tensor<384xf32>, %s2b1ngv: tensor<f32>, %s2b1nbtv: tensor<f32>, %s2b1eWv: tensor<1536x384x1x1xf32>, %s2b1ebv: tensor<1536xf32>, %s2b1pWv: tensor<384x1536x1x1xf32>, %s2b1pbv: tensor<384xf32>, %s2b1lgv: tensor<384xf32>, %s2b2dWv: tensor<384x1x7x7xf32>, %s2b2dbv: tensor<384xf32>, %s2b2ngv: tensor<f32>, %s2b2nbtv: tensor<f32>, %s2b2eWv: tensor<1536x384x1x1xf32>, %s2b2ebv: tensor<1536xf32>, %s2b2pWv: tensor<384x1536x1x1xf32>, %s2b2pbv: tensor<384xf32>, %s2b2lgv: tensor<384xf32>, %s2b3dWv: tensor<384x1x7x7xf32>, %s2b3dbv: tensor<384xf32>, %s2b3ngv: tensor<f32>, %s2b3nbtv: tensor<f32>, %s2b3eWv: tensor<1536x384x1x1xf32>, %s2b3ebv: tensor<1536xf32>, %s2b3pWv: tensor<384x1536x1x1xf32>, %s2b3pbv: tensor<384xf32>, %s2b3lgv: tensor<384xf32>, %s2b4dWv: tensor<384x1x7x7xf32>, %s2b4dbv: tensor<384xf32>, %s2b4ngv: tensor<f32>, %s2b4nbtv: tensor<f32>, %s2b4eWv: tensor<1536x384x1x1xf32>, %s2b4ebv: tensor<1536xf32>, %s2b4pWv: tensor<384x1536x1x1xf32>, %s2b4pbv: tensor<384xf32>, %s2b4lgv: tensor<384xf32>, %s2b5dWv: tensor<384x1x7x7xf32>, %s2b5dbv: tensor<384xf32>, %s2b5ngv: tensor<f32>, %s2b5nbtv: tensor<f32>, %s2b5eWv: tensor<1536x384x1x1xf32>, %s2b5ebv: tensor<1536xf32>, %s2b5pWv: tensor<384x1536x1x1xf32>, %s2b5pbv: tensor<384xf32>, %s2b5lgv: tensor<384xf32>, %s2b6dWv: tensor<384x1x7x7xf32>, %s2b6dbv: tensor<384xf32>, %s2b6ngv: tensor<f32>, %s2b6nbtv: tensor<f32>, %s2b6eWv: tensor<1536x384x1x1xf32>, %s2b6ebv: tensor<1536xf32>, %s2b6pWv: tensor<384x1536x1x1xf32>, %s2b6pbv: tensor<384xf32>, %s2b6lgv: tensor<384xf32>, %s2b7dWv: tensor<384x1x7x7xf32>, %s2b7dbv: tensor<384xf32>, %s2b7ngv: tensor<f32>, %s2b7nbtv: tensor<f32>, %s2b7eWv: tensor<1536x384x1x1xf32>, %s2b7ebv: tensor<1536xf32>, %s2b7pWv: tensor<384x1536x1x1xf32>, %s2b7pbv: tensor<384xf32>, %s2b7lgv: tensor<384xf32>, %s2b8dWv: tensor<384x1x7x7xf32>, %s2b8dbv: tensor<384xf32>, %s2b8ngv: tensor<f32>, %s2b8nbtv: tensor<f32>, %s2b8eWv: tensor<1536x384x1x1xf32>, %s2b8ebv: tensor<1536xf32>, %s2b8pWv: tensor<384x1536x1x1xf32>, %s2b8pbv: tensor<384xf32>, %s2b8lgv: tensor<384xf32>, %d2ngv: tensor<f32>, %d2nbtv: tensor<f32>, %d2Wv: tensor<768x384x2x2xf32>, %d2bv: tensor<768xf32>, %s3b0dWv: tensor<768x1x7x7xf32>, %s3b0dbv: tensor<768xf32>, %s3b0ngv: tensor<f32>, %s3b0nbtv: tensor<f32>, %s3b0eWv: tensor<3072x768x1x1xf32>, %s3b0ebv: tensor<3072xf32>, %s3b0pWv: tensor<768x3072x1x1xf32>, %s3b0pbv: tensor<768xf32>, %s3b0lgv: tensor<768xf32>, %s3b1dWv: tensor<768x1x7x7xf32>, %s3b1dbv: tensor<768xf32>, %s3b1ngv: tensor<f32>, %s3b1nbtv: tensor<f32>, %s3b1eWv: tensor<3072x768x1x1xf32>, %s3b1ebv: tensor<3072xf32>, %s3b1pWv: tensor<768x3072x1x1xf32>, %s3b1pbv: tensor<768xf32>, %s3b1lgv: tensor<768xf32>, %s3b2dWv: tensor<768x1x7x7xf32>, %s3b2dbv: tensor<768xf32>, %s3b2ngv: tensor<f32>, %s3b2nbtv: tensor<f32>, %s3b2eWv: tensor<3072x768x1x1xf32>, %s3b2ebv: tensor<3072xf32>, %s3b2pWv: tensor<768x3072x1x1xf32>, %s3b2pbv: tensor<768xf32>, %s3b2lgv: tensor<768xf32>, %hngv: tensor<f32>, %hnbtv: tensor<f32>, %Wdv: tensor<768x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<32x10xf32>) -> (tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %bsc = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    // ── ConvNeXt-T AdamW train step: gradients + optimizer are pretty(AST node) ──
    // EXCEPT the stem 4x4/s4 and the 2x2/s2 downsample WEIGHT GRADIENTS, which have no
    // VJP-cert SHlo op and stay hand-written (the two documented gaps). Their UPDATES are
    // certified here, which the SGD render's hand-written `sgd` wrap was not.
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
    %dd2Wxi = stablehlo.reshape %v833 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %dd2Wdi = stablehlo.reshape %v1386 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %dd2Wu = stablehlo.pad %dd2Wdi, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x13x13xf32>
    %dd2Wxt = stablehlo.transpose %dd2Wxi, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %dd2Wdt = stablehlo.transpose %dd2Wu, dims = [1, 0, 2, 3] : (tensor<32x768x13x13xf32>) -> tensor<768x32x13x13xf32>
    %dd2Wraw = stablehlo.convolution(%dd2Wxt, %dd2Wdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<768x32x13x13xf32>) -> tensor<384x768x2x2xf32>
    %dd2W = stablehlo.transpose %dd2Wraw, dims = [1, 0, 2, 3] : (tensor<384x768x2x2xf32>) -> tensor<768x384x2x2xf32>
    %v1492 = stablehlo.reshape %v1470 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1493 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1494 = stablehlo.multiply %v1492, %v1493 : tensor<32x384x14x14xf32>
    %v1495 = stablehlo.reshape %v1494 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1496 = stablehlo.reshape %v1495 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1497 = stablehlo.transpose %s2b8pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1498 = stablehlo.reverse %v1497, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1499 = stablehlo.convolution(%v1496, %v1498)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1500 = stablehlo.reshape %v1499 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1501 = stablehlo.multiply %v792, %v792 : tensor<32x301056xf32>
    %v1502 = stablehlo.multiply %v1501, %v792 : tensor<32x301056xf32>
    %v1503 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1504 = stablehlo.multiply %v1503, %v1502 : tensor<32x301056xf32>
    %v1505 = stablehlo.add %v792, %v1504 : tensor<32x301056xf32>
    %v1506 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1507 = stablehlo.multiply %v1506, %v1505 : tensor<32x301056xf32>
    %v1508 = stablehlo.tanh %v1507 : tensor<32x301056xf32>
    %v1509 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1510 = stablehlo.add %v1509, %v1508 : tensor<32x301056xf32>
    %v1511 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1512 = stablehlo.multiply %v1511, %v1510 : tensor<32x301056xf32>
    %v1513 = stablehlo.multiply %v1508, %v1508 : tensor<32x301056xf32>
    %v1514 = stablehlo.subtract %v1509, %v1513 : tensor<32x301056xf32>
    %v1515 = stablehlo.multiply %v1511, %v792 : tensor<32x301056xf32>
    %v1516 = stablehlo.multiply %v1515, %v1514 : tensor<32x301056xf32>
    %v1517 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1518 = stablehlo.multiply %v1517, %v1501 : tensor<32x301056xf32>
    %v1519 = stablehlo.add %v1509, %v1518 : tensor<32x301056xf32>
    %v1520 = stablehlo.multiply %v1506, %v1519 : tensor<32x301056xf32>
    %v1521 = stablehlo.multiply %v1516, %v1520 : tensor<32x301056xf32>
    %v1522 = stablehlo.add %v1512, %v1521 : tensor<32x301056xf32>
    %v1523 = stablehlo.multiply %v1500, %v1522 : tensor<32x301056xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1525 = stablehlo.transpose %s2b8eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1526 = stablehlo.reverse %v1525, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1527 = stablehlo.convolution(%v1524, %v1526)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1528 = stablehlo.reshape %v1527 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1530 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1531 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1532 = stablehlo.reduce(%v769 init: %v1529) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1533 = stablehlo.broadcast_in_dim %v1532, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1534 = stablehlo.divide %v1533, %v1530 : tensor<32x75264xf32>
    %v1535 = stablehlo.subtract %v769, %v1534 : tensor<32x75264xf32>
    %v1536 = stablehlo.multiply %v1535, %v1535 : tensor<32x75264xf32>
    %v1537 = stablehlo.reduce(%v1536 init: %v1529) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1538 = stablehlo.broadcast_in_dim %v1537, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1539 = stablehlo.divide %v1538, %v1530 : tensor<32x75264xf32>
    %v1540 = stablehlo.add %v1539, %v1531 : tensor<32x75264xf32>
    %v1541 = stablehlo.rsqrt %v1540 : tensor<32x75264xf32>
    %v1542 = stablehlo.multiply %v1535, %v1541 : tensor<32x75264xf32>
    %v1543 = stablehlo.broadcast_in_dim %s2b8ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1544 = stablehlo.multiply %v1543, %v1528 : tensor<32x75264xf32>
    %v1545 = stablehlo.reduce(%v1544 init: %v1529) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1546 = stablehlo.broadcast_in_dim %v1545, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1547 = stablehlo.multiply %v1542, %v1544 : tensor<32x75264xf32>
    %v1548 = stablehlo.reduce(%v1547 init: %v1529) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1549 = stablehlo.broadcast_in_dim %v1548, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1550 = stablehlo.multiply %v1544, %v1530 : tensor<32x75264xf32>
    %v1551 = stablehlo.subtract %v1550, %v1546 : tensor<32x75264xf32>
    %v1552 = stablehlo.multiply %v1542, %v1549 : tensor<32x75264xf32>
    %v1553 = stablehlo.subtract %v1551, %v1552 : tensor<32x75264xf32>
    %v1554 = stablehlo.divide %v1541, %v1530 : tensor<32x75264xf32>
    %v1555 = stablehlo.multiply %v1554, %v1553 : tensor<32x75264xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1557 = stablehlo.reverse %s2b8dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1558 = stablehlo.convolution(%v1556, %v1557)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1560 = stablehlo.add %v1559, %v1470 : tensor<32x75264xf32>
    %v1561 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1562 = stablehlo.reshape %v810 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1563 = stablehlo.reshape %v1470 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1564 = stablehlo.multiply %v1562, %v1563 : tensor<32x384x14x14xf32>
    %v1565 = stablehlo.reduce(%v1564 init: %v1561) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1566 = stablehlo.reshape %v805 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1567 = stablehlo.reshape %v1495 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1568 = stablehlo.transpose %v1566, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1569 = stablehlo.transpose %v1567, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1570 = stablehlo.convolution(%v1568, %v1569)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1571 = stablehlo.transpose %v1570, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1572 = stablehlo.reshape %v1495 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1573 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1574 = stablehlo.reduce(%v1572 init: %v1573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1575 = stablehlo.reshape %v787 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1576 = stablehlo.reshape %v1523 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1577 = stablehlo.transpose %v1575, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1578 = stablehlo.transpose %v1576, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1579 = stablehlo.convolution(%v1577, %v1578)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1580 = stablehlo.transpose %v1579, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1581 = stablehlo.reshape %v1523 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1583 = stablehlo.reduce(%v1581 init: %v1582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1585 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1586 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1587 = stablehlo.reduce(%v769 init: %v1584) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1588 = stablehlo.broadcast_in_dim %v1587, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1589 = stablehlo.divide %v1588, %v1585 : tensor<32x75264xf32>
    %v1590 = stablehlo.subtract %v769, %v1589 : tensor<32x75264xf32>
    %v1591 = stablehlo.multiply %v1590, %v1590 : tensor<32x75264xf32>
    %v1592 = stablehlo.reduce(%v1591 init: %v1584) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1593 = stablehlo.broadcast_in_dim %v1592, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1594 = stablehlo.divide %v1593, %v1585 : tensor<32x75264xf32>
    %v1595 = stablehlo.add %v1594, %v1586 : tensor<32x75264xf32>
    %v1596 = stablehlo.rsqrt %v1595 : tensor<32x75264xf32>
    %v1597 = stablehlo.multiply %v1590, %v1596 : tensor<32x75264xf32>
    %v1598 = stablehlo.multiply %v1528, %v1597 : tensor<32x75264xf32>
    %v1599 = stablehlo.reduce(%v1598 init: %v1584) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1600 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1601 = stablehlo.reduce(%v1528 init: %v1600) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1602 = stablehlo.reshape %v764 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1603 = stablehlo.reshape %v1555 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1604 = stablehlo.transpose %v1602, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1605 = stablehlo.transpose %v1603, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1606 = stablehlo.convolution(%v1604, %v1605)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v1608 = stablehlo.reshape %v1555 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1610 = stablehlo.reduce(%v1608 init: %v1609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1611 = stablehlo.reshape %v1560 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1612 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1613 = stablehlo.multiply %v1611, %v1612 : tensor<32x384x14x14xf32>
    %v1614 = stablehlo.reshape %v1613 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1615 = stablehlo.reshape %v1614 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1616 = stablehlo.transpose %s2b7pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1617 = stablehlo.reverse %v1616, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1618 = stablehlo.convolution(%v1615, %v1617)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1619 = stablehlo.reshape %v1618 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1620 = stablehlo.multiply %v741, %v741 : tensor<32x301056xf32>
    %v1621 = stablehlo.multiply %v1620, %v741 : tensor<32x301056xf32>
    %v1622 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1623 = stablehlo.multiply %v1622, %v1621 : tensor<32x301056xf32>
    %v1624 = stablehlo.add %v741, %v1623 : tensor<32x301056xf32>
    %v1625 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1626 = stablehlo.multiply %v1625, %v1624 : tensor<32x301056xf32>
    %v1627 = stablehlo.tanh %v1626 : tensor<32x301056xf32>
    %v1628 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1629 = stablehlo.add %v1628, %v1627 : tensor<32x301056xf32>
    %v1630 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1631 = stablehlo.multiply %v1630, %v1629 : tensor<32x301056xf32>
    %v1632 = stablehlo.multiply %v1627, %v1627 : tensor<32x301056xf32>
    %v1633 = stablehlo.subtract %v1628, %v1632 : tensor<32x301056xf32>
    %v1634 = stablehlo.multiply %v1630, %v741 : tensor<32x301056xf32>
    %v1635 = stablehlo.multiply %v1634, %v1633 : tensor<32x301056xf32>
    %v1636 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1637 = stablehlo.multiply %v1636, %v1620 : tensor<32x301056xf32>
    %v1638 = stablehlo.add %v1628, %v1637 : tensor<32x301056xf32>
    %v1639 = stablehlo.multiply %v1625, %v1638 : tensor<32x301056xf32>
    %v1640 = stablehlo.multiply %v1635, %v1639 : tensor<32x301056xf32>
    %v1641 = stablehlo.add %v1631, %v1640 : tensor<32x301056xf32>
    %v1642 = stablehlo.multiply %v1619, %v1641 : tensor<32x301056xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1644 = stablehlo.transpose %s2b7eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1645 = stablehlo.reverse %v1644, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1646 = stablehlo.convolution(%v1643, %v1645)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1647 = stablehlo.reshape %v1646 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1648 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1649 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1650 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1651 = stablehlo.reduce(%v718 init: %v1648) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1652 = stablehlo.broadcast_in_dim %v1651, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1653 = stablehlo.divide %v1652, %v1649 : tensor<32x75264xf32>
    %v1654 = stablehlo.subtract %v718, %v1653 : tensor<32x75264xf32>
    %v1655 = stablehlo.multiply %v1654, %v1654 : tensor<32x75264xf32>
    %v1656 = stablehlo.reduce(%v1655 init: %v1648) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1657 = stablehlo.broadcast_in_dim %v1656, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1658 = stablehlo.divide %v1657, %v1649 : tensor<32x75264xf32>
    %v1659 = stablehlo.add %v1658, %v1650 : tensor<32x75264xf32>
    %v1660 = stablehlo.rsqrt %v1659 : tensor<32x75264xf32>
    %v1661 = stablehlo.multiply %v1654, %v1660 : tensor<32x75264xf32>
    %v1662 = stablehlo.broadcast_in_dim %s2b7ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1663 = stablehlo.multiply %v1662, %v1647 : tensor<32x75264xf32>
    %v1664 = stablehlo.reduce(%v1663 init: %v1648) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1665 = stablehlo.broadcast_in_dim %v1664, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1666 = stablehlo.multiply %v1661, %v1663 : tensor<32x75264xf32>
    %v1667 = stablehlo.reduce(%v1666 init: %v1648) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1668 = stablehlo.broadcast_in_dim %v1667, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1669 = stablehlo.multiply %v1663, %v1649 : tensor<32x75264xf32>
    %v1670 = stablehlo.subtract %v1669, %v1665 : tensor<32x75264xf32>
    %v1671 = stablehlo.multiply %v1661, %v1668 : tensor<32x75264xf32>
    %v1672 = stablehlo.subtract %v1670, %v1671 : tensor<32x75264xf32>
    %v1673 = stablehlo.divide %v1660, %v1649 : tensor<32x75264xf32>
    %v1674 = stablehlo.multiply %v1673, %v1672 : tensor<32x75264xf32>
    %v1675 = stablehlo.reshape %v1674 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1676 = stablehlo.reverse %s2b7dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1677 = stablehlo.convolution(%v1675, %v1676)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1678 = stablehlo.reshape %v1677 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1679 = stablehlo.add %v1678, %v1560 : tensor<32x75264xf32>
    %v1680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1681 = stablehlo.reshape %v759 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1682 = stablehlo.reshape %v1560 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1683 = stablehlo.multiply %v1681, %v1682 : tensor<32x384x14x14xf32>
    %v1684 = stablehlo.reduce(%v1683 init: %v1680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1685 = stablehlo.reshape %v754 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1686 = stablehlo.reshape %v1614 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1687 = stablehlo.transpose %v1685, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1688 = stablehlo.transpose %v1686, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1689 = stablehlo.convolution(%v1687, %v1688)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1690 = stablehlo.transpose %v1689, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1691 = stablehlo.reshape %v1614 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1693 = stablehlo.reduce(%v1691 init: %v1692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1694 = stablehlo.reshape %v736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1695 = stablehlo.reshape %v1642 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1696 = stablehlo.transpose %v1694, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1697 = stablehlo.transpose %v1695, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1698 = stablehlo.convolution(%v1696, %v1697)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1699 = stablehlo.transpose %v1698, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1700 = stablehlo.reshape %v1642 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1702 = stablehlo.reduce(%v1700 init: %v1701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1703 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1704 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1705 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1706 = stablehlo.reduce(%v718 init: %v1703) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1707 = stablehlo.broadcast_in_dim %v1706, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1708 = stablehlo.divide %v1707, %v1704 : tensor<32x75264xf32>
    %v1709 = stablehlo.subtract %v718, %v1708 : tensor<32x75264xf32>
    %v1710 = stablehlo.multiply %v1709, %v1709 : tensor<32x75264xf32>
    %v1711 = stablehlo.reduce(%v1710 init: %v1703) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1712 = stablehlo.broadcast_in_dim %v1711, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1713 = stablehlo.divide %v1712, %v1704 : tensor<32x75264xf32>
    %v1714 = stablehlo.add %v1713, %v1705 : tensor<32x75264xf32>
    %v1715 = stablehlo.rsqrt %v1714 : tensor<32x75264xf32>
    %v1716 = stablehlo.multiply %v1709, %v1715 : tensor<32x75264xf32>
    %v1717 = stablehlo.multiply %v1647, %v1716 : tensor<32x75264xf32>
    %v1718 = stablehlo.reduce(%v1717 init: %v1703) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1720 = stablehlo.reduce(%v1647 init: %v1719) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1721 = stablehlo.reshape %v713 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1722 = stablehlo.reshape %v1674 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1723 = stablehlo.transpose %v1721, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1724 = stablehlo.transpose %v1722, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1725 = stablehlo.convolution(%v1723, %v1724)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v1726 = stablehlo.reshape %v1725 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v1727 = stablehlo.reshape %v1674 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1729 = stablehlo.reduce(%v1727 init: %v1728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1730 = stablehlo.reshape %v1679 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1731 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1732 = stablehlo.multiply %v1730, %v1731 : tensor<32x384x14x14xf32>
    %v1733 = stablehlo.reshape %v1732 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1734 = stablehlo.reshape %v1733 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1735 = stablehlo.transpose %s2b6pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1736 = stablehlo.reverse %v1735, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1737 = stablehlo.convolution(%v1734, %v1736)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1738 = stablehlo.reshape %v1737 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1739 = stablehlo.multiply %v690, %v690 : tensor<32x301056xf32>
    %v1740 = stablehlo.multiply %v1739, %v690 : tensor<32x301056xf32>
    %v1741 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1742 = stablehlo.multiply %v1741, %v1740 : tensor<32x301056xf32>
    %v1743 = stablehlo.add %v690, %v1742 : tensor<32x301056xf32>
    %v1744 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1745 = stablehlo.multiply %v1744, %v1743 : tensor<32x301056xf32>
    %v1746 = stablehlo.tanh %v1745 : tensor<32x301056xf32>
    %v1747 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1748 = stablehlo.add %v1747, %v1746 : tensor<32x301056xf32>
    %v1749 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1750 = stablehlo.multiply %v1749, %v1748 : tensor<32x301056xf32>
    %v1751 = stablehlo.multiply %v1746, %v1746 : tensor<32x301056xf32>
    %v1752 = stablehlo.subtract %v1747, %v1751 : tensor<32x301056xf32>
    %v1753 = stablehlo.multiply %v1749, %v690 : tensor<32x301056xf32>
    %v1754 = stablehlo.multiply %v1753, %v1752 : tensor<32x301056xf32>
    %v1755 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1756 = stablehlo.multiply %v1755, %v1739 : tensor<32x301056xf32>
    %v1757 = stablehlo.add %v1747, %v1756 : tensor<32x301056xf32>
    %v1758 = stablehlo.multiply %v1744, %v1757 : tensor<32x301056xf32>
    %v1759 = stablehlo.multiply %v1754, %v1758 : tensor<32x301056xf32>
    %v1760 = stablehlo.add %v1750, %v1759 : tensor<32x301056xf32>
    %v1761 = stablehlo.multiply %v1738, %v1760 : tensor<32x301056xf32>
    %v1762 = stablehlo.reshape %v1761 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1763 = stablehlo.transpose %s2b6eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1764 = stablehlo.reverse %v1763, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1765 = stablehlo.convolution(%v1762, %v1764)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1766 = stablehlo.reshape %v1765 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1768 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1769 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1770 = stablehlo.reduce(%v667 init: %v1767) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1771 = stablehlo.broadcast_in_dim %v1770, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1772 = stablehlo.divide %v1771, %v1768 : tensor<32x75264xf32>
    %v1773 = stablehlo.subtract %v667, %v1772 : tensor<32x75264xf32>
    %v1774 = stablehlo.multiply %v1773, %v1773 : tensor<32x75264xf32>
    %v1775 = stablehlo.reduce(%v1774 init: %v1767) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1776 = stablehlo.broadcast_in_dim %v1775, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1777 = stablehlo.divide %v1776, %v1768 : tensor<32x75264xf32>
    %v1778 = stablehlo.add %v1777, %v1769 : tensor<32x75264xf32>
    %v1779 = stablehlo.rsqrt %v1778 : tensor<32x75264xf32>
    %v1780 = stablehlo.multiply %v1773, %v1779 : tensor<32x75264xf32>
    %v1781 = stablehlo.broadcast_in_dim %s2b6ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1782 = stablehlo.multiply %v1781, %v1766 : tensor<32x75264xf32>
    %v1783 = stablehlo.reduce(%v1782 init: %v1767) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1784 = stablehlo.broadcast_in_dim %v1783, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1785 = stablehlo.multiply %v1780, %v1782 : tensor<32x75264xf32>
    %v1786 = stablehlo.reduce(%v1785 init: %v1767) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1787 = stablehlo.broadcast_in_dim %v1786, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1788 = stablehlo.multiply %v1782, %v1768 : tensor<32x75264xf32>
    %v1789 = stablehlo.subtract %v1788, %v1784 : tensor<32x75264xf32>
    %v1790 = stablehlo.multiply %v1780, %v1787 : tensor<32x75264xf32>
    %v1791 = stablehlo.subtract %v1789, %v1790 : tensor<32x75264xf32>
    %v1792 = stablehlo.divide %v1779, %v1768 : tensor<32x75264xf32>
    %v1793 = stablehlo.multiply %v1792, %v1791 : tensor<32x75264xf32>
    %v1794 = stablehlo.reshape %v1793 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1795 = stablehlo.reverse %s2b6dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1796 = stablehlo.convolution(%v1794, %v1795)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1798 = stablehlo.add %v1797, %v1679 : tensor<32x75264xf32>
    %v1799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1800 = stablehlo.reshape %v708 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1801 = stablehlo.reshape %v1679 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1802 = stablehlo.multiply %v1800, %v1801 : tensor<32x384x14x14xf32>
    %v1803 = stablehlo.reduce(%v1802 init: %v1799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1804 = stablehlo.reshape %v703 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1805 = stablehlo.reshape %v1733 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1806 = stablehlo.transpose %v1804, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1807 = stablehlo.transpose %v1805, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1808 = stablehlo.convolution(%v1806, %v1807)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1809 = stablehlo.transpose %v1808, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1810 = stablehlo.reshape %v1733 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1812 = stablehlo.reduce(%v1810 init: %v1811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1813 = stablehlo.reshape %v685 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1814 = stablehlo.reshape %v1761 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1815 = stablehlo.transpose %v1813, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1816 = stablehlo.transpose %v1814, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1817 = stablehlo.convolution(%v1815, %v1816)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1818 = stablehlo.transpose %v1817, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1819 = stablehlo.reshape %v1761 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1821 = stablehlo.reduce(%v1819 init: %v1820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1823 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1824 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1825 = stablehlo.reduce(%v667 init: %v1822) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1826 = stablehlo.broadcast_in_dim %v1825, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1827 = stablehlo.divide %v1826, %v1823 : tensor<32x75264xf32>
    %v1828 = stablehlo.subtract %v667, %v1827 : tensor<32x75264xf32>
    %v1829 = stablehlo.multiply %v1828, %v1828 : tensor<32x75264xf32>
    %v1830 = stablehlo.reduce(%v1829 init: %v1822) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1831 = stablehlo.broadcast_in_dim %v1830, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1832 = stablehlo.divide %v1831, %v1823 : tensor<32x75264xf32>
    %v1833 = stablehlo.add %v1832, %v1824 : tensor<32x75264xf32>
    %v1834 = stablehlo.rsqrt %v1833 : tensor<32x75264xf32>
    %v1835 = stablehlo.multiply %v1828, %v1834 : tensor<32x75264xf32>
    %v1836 = stablehlo.multiply %v1766, %v1835 : tensor<32x75264xf32>
    %v1837 = stablehlo.reduce(%v1836 init: %v1822) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1839 = stablehlo.reduce(%v1766 init: %v1838) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1840 = stablehlo.reshape %v662 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1841 = stablehlo.reshape %v1793 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1842 = stablehlo.transpose %v1840, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1843 = stablehlo.transpose %v1841, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1844 = stablehlo.convolution(%v1842, %v1843)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v1845 = stablehlo.reshape %v1844 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v1846 = stablehlo.reshape %v1793 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1848 = stablehlo.reduce(%v1846 init: %v1847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1849 = stablehlo.reshape %v1798 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1850 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1851 = stablehlo.multiply %v1849, %v1850 : tensor<32x384x14x14xf32>
    %v1852 = stablehlo.reshape %v1851 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1854 = stablehlo.transpose %s2b5pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1855 = stablehlo.reverse %v1854, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1856 = stablehlo.convolution(%v1853, %v1855)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1857 = stablehlo.reshape %v1856 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1858 = stablehlo.multiply %v639, %v639 : tensor<32x301056xf32>
    %v1859 = stablehlo.multiply %v1858, %v639 : tensor<32x301056xf32>
    %v1860 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1861 = stablehlo.multiply %v1860, %v1859 : tensor<32x301056xf32>
    %v1862 = stablehlo.add %v639, %v1861 : tensor<32x301056xf32>
    %v1863 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1864 = stablehlo.multiply %v1863, %v1862 : tensor<32x301056xf32>
    %v1865 = stablehlo.tanh %v1864 : tensor<32x301056xf32>
    %v1866 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1867 = stablehlo.add %v1866, %v1865 : tensor<32x301056xf32>
    %v1868 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1869 = stablehlo.multiply %v1868, %v1867 : tensor<32x301056xf32>
    %v1870 = stablehlo.multiply %v1865, %v1865 : tensor<32x301056xf32>
    %v1871 = stablehlo.subtract %v1866, %v1870 : tensor<32x301056xf32>
    %v1872 = stablehlo.multiply %v1868, %v639 : tensor<32x301056xf32>
    %v1873 = stablehlo.multiply %v1872, %v1871 : tensor<32x301056xf32>
    %v1874 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1875 = stablehlo.multiply %v1874, %v1858 : tensor<32x301056xf32>
    %v1876 = stablehlo.add %v1866, %v1875 : tensor<32x301056xf32>
    %v1877 = stablehlo.multiply %v1863, %v1876 : tensor<32x301056xf32>
    %v1878 = stablehlo.multiply %v1873, %v1877 : tensor<32x301056xf32>
    %v1879 = stablehlo.add %v1869, %v1878 : tensor<32x301056xf32>
    %v1880 = stablehlo.multiply %v1857, %v1879 : tensor<32x301056xf32>
    %v1881 = stablehlo.reshape %v1880 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1882 = stablehlo.transpose %s2b5eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1883 = stablehlo.reverse %v1882, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1884 = stablehlo.convolution(%v1881, %v1883)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1885 = stablehlo.reshape %v1884 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1887 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1888 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1889 = stablehlo.reduce(%v616 init: %v1886) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1890 = stablehlo.broadcast_in_dim %v1889, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1891 = stablehlo.divide %v1890, %v1887 : tensor<32x75264xf32>
    %v1892 = stablehlo.subtract %v616, %v1891 : tensor<32x75264xf32>
    %v1893 = stablehlo.multiply %v1892, %v1892 : tensor<32x75264xf32>
    %v1894 = stablehlo.reduce(%v1893 init: %v1886) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1895 = stablehlo.broadcast_in_dim %v1894, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1896 = stablehlo.divide %v1895, %v1887 : tensor<32x75264xf32>
    %v1897 = stablehlo.add %v1896, %v1888 : tensor<32x75264xf32>
    %v1898 = stablehlo.rsqrt %v1897 : tensor<32x75264xf32>
    %v1899 = stablehlo.multiply %v1892, %v1898 : tensor<32x75264xf32>
    %v1900 = stablehlo.broadcast_in_dim %s2b5ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v1901 = stablehlo.multiply %v1900, %v1885 : tensor<32x75264xf32>
    %v1902 = stablehlo.reduce(%v1901 init: %v1886) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1903 = stablehlo.broadcast_in_dim %v1902, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1904 = stablehlo.multiply %v1899, %v1901 : tensor<32x75264xf32>
    %v1905 = stablehlo.reduce(%v1904 init: %v1886) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1906 = stablehlo.broadcast_in_dim %v1905, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1907 = stablehlo.multiply %v1901, %v1887 : tensor<32x75264xf32>
    %v1908 = stablehlo.subtract %v1907, %v1903 : tensor<32x75264xf32>
    %v1909 = stablehlo.multiply %v1899, %v1906 : tensor<32x75264xf32>
    %v1910 = stablehlo.subtract %v1908, %v1909 : tensor<32x75264xf32>
    %v1911 = stablehlo.divide %v1898, %v1887 : tensor<32x75264xf32>
    %v1912 = stablehlo.multiply %v1911, %v1910 : tensor<32x75264xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1914 = stablehlo.reverse %s2b5dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v1915 = stablehlo.convolution(%v1913, %v1914)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1917 = stablehlo.add %v1916, %v1798 : tensor<32x75264xf32>
    %v1918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1919 = stablehlo.reshape %v657 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1920 = stablehlo.reshape %v1798 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1921 = stablehlo.multiply %v1919, %v1920 : tensor<32x384x14x14xf32>
    %v1922 = stablehlo.reduce(%v1921 init: %v1918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1923 = stablehlo.reshape %v652 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1924 = stablehlo.reshape %v1852 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1925 = stablehlo.transpose %v1923, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1926 = stablehlo.transpose %v1924, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1927 = stablehlo.convolution(%v1925, %v1926)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v1928 = stablehlo.transpose %v1927, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1929 = stablehlo.reshape %v1852 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1931 = stablehlo.reduce(%v1929 init: %v1930) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1932 = stablehlo.reshape %v634 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1933 = stablehlo.reshape %v1880 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1934 = stablehlo.transpose %v1932, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1935 = stablehlo.transpose %v1933, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v1936 = stablehlo.convolution(%v1934, %v1935)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v1937 = stablehlo.transpose %v1936, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1938 = stablehlo.reshape %v1880 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1940 = stablehlo.reduce(%v1938 init: %v1939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v1941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1942 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v1943 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v1944 = stablehlo.reduce(%v616 init: %v1941) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1945 = stablehlo.broadcast_in_dim %v1944, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1946 = stablehlo.divide %v1945, %v1942 : tensor<32x75264xf32>
    %v1947 = stablehlo.subtract %v616, %v1946 : tensor<32x75264xf32>
    %v1948 = stablehlo.multiply %v1947, %v1947 : tensor<32x75264xf32>
    %v1949 = stablehlo.reduce(%v1948 init: %v1941) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v1950 = stablehlo.broadcast_in_dim %v1949, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1951 = stablehlo.divide %v1950, %v1942 : tensor<32x75264xf32>
    %v1952 = stablehlo.add %v1951, %v1943 : tensor<32x75264xf32>
    %v1953 = stablehlo.rsqrt %v1952 : tensor<32x75264xf32>
    %v1954 = stablehlo.multiply %v1947, %v1953 : tensor<32x75264xf32>
    %v1955 = stablehlo.multiply %v1885, %v1954 : tensor<32x75264xf32>
    %v1956 = stablehlo.reduce(%v1955 init: %v1941) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1957 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1958 = stablehlo.reduce(%v1885 init: %v1957) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v1959 = stablehlo.reshape %v611 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1960 = stablehlo.reshape %v1912 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1961 = stablehlo.transpose %v1959, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1962 = stablehlo.transpose %v1960, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1963 = stablehlo.convolution(%v1961, %v1962)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v1964 = stablehlo.reshape %v1963 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v1965 = stablehlo.reshape %v1912 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1967 = stablehlo.reduce(%v1965 init: %v1966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v1968 = stablehlo.reshape %v1917 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1969 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1970 = stablehlo.multiply %v1968, %v1969 : tensor<32x384x14x14xf32>
    %v1971 = stablehlo.reshape %v1970 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1972 = stablehlo.reshape %v1971 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1973 = stablehlo.transpose %s2b4pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1974 = stablehlo.reverse %v1973, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1975 = stablehlo.convolution(%v1972, %v1974)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1976 = stablehlo.reshape %v1975 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1977 = stablehlo.multiply %v588, %v588 : tensor<32x301056xf32>
    %v1978 = stablehlo.multiply %v1977, %v588 : tensor<32x301056xf32>
    %v1979 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1980 = stablehlo.multiply %v1979, %v1978 : tensor<32x301056xf32>
    %v1981 = stablehlo.add %v588, %v1980 : tensor<32x301056xf32>
    %v1982 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1983 = stablehlo.multiply %v1982, %v1981 : tensor<32x301056xf32>
    %v1984 = stablehlo.tanh %v1983 : tensor<32x301056xf32>
    %v1985 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1986 = stablehlo.add %v1985, %v1984 : tensor<32x301056xf32>
    %v1987 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1988 = stablehlo.multiply %v1987, %v1986 : tensor<32x301056xf32>
    %v1989 = stablehlo.multiply %v1984, %v1984 : tensor<32x301056xf32>
    %v1990 = stablehlo.subtract %v1985, %v1989 : tensor<32x301056xf32>
    %v1991 = stablehlo.multiply %v1987, %v588 : tensor<32x301056xf32>
    %v1992 = stablehlo.multiply %v1991, %v1990 : tensor<32x301056xf32>
    %v1993 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1994 = stablehlo.multiply %v1993, %v1977 : tensor<32x301056xf32>
    %v1995 = stablehlo.add %v1985, %v1994 : tensor<32x301056xf32>
    %v1996 = stablehlo.multiply %v1982, %v1995 : tensor<32x301056xf32>
    %v1997 = stablehlo.multiply %v1992, %v1996 : tensor<32x301056xf32>
    %v1998 = stablehlo.add %v1988, %v1997 : tensor<32x301056xf32>
    %v1999 = stablehlo.multiply %v1976, %v1998 : tensor<32x301056xf32>
    %v2000 = stablehlo.reshape %v1999 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2001 = stablehlo.transpose %s2b4eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2002 = stablehlo.reverse %v2001, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2003 = stablehlo.convolution(%v2000, %v2002)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2004 = stablehlo.reshape %v2003 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2006 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2007 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2008 = stablehlo.reduce(%v565 init: %v2005) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2009 = stablehlo.broadcast_in_dim %v2008, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2010 = stablehlo.divide %v2009, %v2006 : tensor<32x75264xf32>
    %v2011 = stablehlo.subtract %v565, %v2010 : tensor<32x75264xf32>
    %v2012 = stablehlo.multiply %v2011, %v2011 : tensor<32x75264xf32>
    %v2013 = stablehlo.reduce(%v2012 init: %v2005) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2014 = stablehlo.broadcast_in_dim %v2013, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2015 = stablehlo.divide %v2014, %v2006 : tensor<32x75264xf32>
    %v2016 = stablehlo.add %v2015, %v2007 : tensor<32x75264xf32>
    %v2017 = stablehlo.rsqrt %v2016 : tensor<32x75264xf32>
    %v2018 = stablehlo.multiply %v2011, %v2017 : tensor<32x75264xf32>
    %v2019 = stablehlo.broadcast_in_dim %s2b4ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2020 = stablehlo.multiply %v2019, %v2004 : tensor<32x75264xf32>
    %v2021 = stablehlo.reduce(%v2020 init: %v2005) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2022 = stablehlo.broadcast_in_dim %v2021, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2023 = stablehlo.multiply %v2018, %v2020 : tensor<32x75264xf32>
    %v2024 = stablehlo.reduce(%v2023 init: %v2005) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2025 = stablehlo.broadcast_in_dim %v2024, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2026 = stablehlo.multiply %v2020, %v2006 : tensor<32x75264xf32>
    %v2027 = stablehlo.subtract %v2026, %v2022 : tensor<32x75264xf32>
    %v2028 = stablehlo.multiply %v2018, %v2025 : tensor<32x75264xf32>
    %v2029 = stablehlo.subtract %v2027, %v2028 : tensor<32x75264xf32>
    %v2030 = stablehlo.divide %v2017, %v2006 : tensor<32x75264xf32>
    %v2031 = stablehlo.multiply %v2030, %v2029 : tensor<32x75264xf32>
    %v2032 = stablehlo.reshape %v2031 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2033 = stablehlo.reverse %s2b4dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2034 = stablehlo.convolution(%v2032, %v2033)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2035 = stablehlo.reshape %v2034 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2036 = stablehlo.add %v2035, %v1917 : tensor<32x75264xf32>
    %v2037 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2038 = stablehlo.reshape %v606 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2039 = stablehlo.reshape %v1917 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2040 = stablehlo.multiply %v2038, %v2039 : tensor<32x384x14x14xf32>
    %v2041 = stablehlo.reduce(%v2040 init: %v2037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2042 = stablehlo.reshape %v601 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2043 = stablehlo.reshape %v1971 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2044 = stablehlo.transpose %v2042, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2045 = stablehlo.transpose %v2043, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2046 = stablehlo.convolution(%v2044, %v2045)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2047 = stablehlo.transpose %v2046, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2048 = stablehlo.reshape %v1971 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2049 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2050 = stablehlo.reduce(%v2048 init: %v2049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2051 = stablehlo.reshape %v583 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2052 = stablehlo.reshape %v1999 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2053 = stablehlo.transpose %v2051, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2054 = stablehlo.transpose %v2052, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2055 = stablehlo.convolution(%v2053, %v2054)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2056 = stablehlo.transpose %v2055, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2057 = stablehlo.reshape %v1999 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2058 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2059 = stablehlo.reduce(%v2057 init: %v2058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2060 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2061 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2062 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2063 = stablehlo.reduce(%v565 init: %v2060) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2064 = stablehlo.broadcast_in_dim %v2063, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2065 = stablehlo.divide %v2064, %v2061 : tensor<32x75264xf32>
    %v2066 = stablehlo.subtract %v565, %v2065 : tensor<32x75264xf32>
    %v2067 = stablehlo.multiply %v2066, %v2066 : tensor<32x75264xf32>
    %v2068 = stablehlo.reduce(%v2067 init: %v2060) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2069 = stablehlo.broadcast_in_dim %v2068, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2070 = stablehlo.divide %v2069, %v2061 : tensor<32x75264xf32>
    %v2071 = stablehlo.add %v2070, %v2062 : tensor<32x75264xf32>
    %v2072 = stablehlo.rsqrt %v2071 : tensor<32x75264xf32>
    %v2073 = stablehlo.multiply %v2066, %v2072 : tensor<32x75264xf32>
    %v2074 = stablehlo.multiply %v2004, %v2073 : tensor<32x75264xf32>
    %v2075 = stablehlo.reduce(%v2074 init: %v2060) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2076 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2077 = stablehlo.reduce(%v2004 init: %v2076) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2078 = stablehlo.reshape %v560 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2079 = stablehlo.reshape %v2031 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2080 = stablehlo.transpose %v2078, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2081 = stablehlo.transpose %v2079, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2082 = stablehlo.convolution(%v2080, %v2081)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2083 = stablehlo.reshape %v2082 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2084 = stablehlo.reshape %v2031 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2086 = stablehlo.reduce(%v2084 init: %v2085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2087 = stablehlo.reshape %v2036 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2088 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2089 = stablehlo.multiply %v2087, %v2088 : tensor<32x384x14x14xf32>
    %v2090 = stablehlo.reshape %v2089 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2091 = stablehlo.reshape %v2090 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2092 = stablehlo.transpose %s2b3pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2093 = stablehlo.reverse %v2092, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2094 = stablehlo.convolution(%v2091, %v2093)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2095 = stablehlo.reshape %v2094 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2096 = stablehlo.multiply %v537, %v537 : tensor<32x301056xf32>
    %v2097 = stablehlo.multiply %v2096, %v537 : tensor<32x301056xf32>
    %v2098 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2099 = stablehlo.multiply %v2098, %v2097 : tensor<32x301056xf32>
    %v2100 = stablehlo.add %v537, %v2099 : tensor<32x301056xf32>
    %v2101 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2102 = stablehlo.multiply %v2101, %v2100 : tensor<32x301056xf32>
    %v2103 = stablehlo.tanh %v2102 : tensor<32x301056xf32>
    %v2104 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2105 = stablehlo.add %v2104, %v2103 : tensor<32x301056xf32>
    %v2106 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2107 = stablehlo.multiply %v2106, %v2105 : tensor<32x301056xf32>
    %v2108 = stablehlo.multiply %v2103, %v2103 : tensor<32x301056xf32>
    %v2109 = stablehlo.subtract %v2104, %v2108 : tensor<32x301056xf32>
    %v2110 = stablehlo.multiply %v2106, %v537 : tensor<32x301056xf32>
    %v2111 = stablehlo.multiply %v2110, %v2109 : tensor<32x301056xf32>
    %v2112 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2113 = stablehlo.multiply %v2112, %v2096 : tensor<32x301056xf32>
    %v2114 = stablehlo.add %v2104, %v2113 : tensor<32x301056xf32>
    %v2115 = stablehlo.multiply %v2101, %v2114 : tensor<32x301056xf32>
    %v2116 = stablehlo.multiply %v2111, %v2115 : tensor<32x301056xf32>
    %v2117 = stablehlo.add %v2107, %v2116 : tensor<32x301056xf32>
    %v2118 = stablehlo.multiply %v2095, %v2117 : tensor<32x301056xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2120 = stablehlo.transpose %s2b3eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2121 = stablehlo.reverse %v2120, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2122 = stablehlo.convolution(%v2119, %v2121)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2123 = stablehlo.reshape %v2122 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2125 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2126 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2127 = stablehlo.reduce(%v514 init: %v2124) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2128 = stablehlo.broadcast_in_dim %v2127, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2129 = stablehlo.divide %v2128, %v2125 : tensor<32x75264xf32>
    %v2130 = stablehlo.subtract %v514, %v2129 : tensor<32x75264xf32>
    %v2131 = stablehlo.multiply %v2130, %v2130 : tensor<32x75264xf32>
    %v2132 = stablehlo.reduce(%v2131 init: %v2124) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2133 = stablehlo.broadcast_in_dim %v2132, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2134 = stablehlo.divide %v2133, %v2125 : tensor<32x75264xf32>
    %v2135 = stablehlo.add %v2134, %v2126 : tensor<32x75264xf32>
    %v2136 = stablehlo.rsqrt %v2135 : tensor<32x75264xf32>
    %v2137 = stablehlo.multiply %v2130, %v2136 : tensor<32x75264xf32>
    %v2138 = stablehlo.broadcast_in_dim %s2b3ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2139 = stablehlo.multiply %v2138, %v2123 : tensor<32x75264xf32>
    %v2140 = stablehlo.reduce(%v2139 init: %v2124) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2141 = stablehlo.broadcast_in_dim %v2140, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2142 = stablehlo.multiply %v2137, %v2139 : tensor<32x75264xf32>
    %v2143 = stablehlo.reduce(%v2142 init: %v2124) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2144 = stablehlo.broadcast_in_dim %v2143, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2145 = stablehlo.multiply %v2139, %v2125 : tensor<32x75264xf32>
    %v2146 = stablehlo.subtract %v2145, %v2141 : tensor<32x75264xf32>
    %v2147 = stablehlo.multiply %v2137, %v2144 : tensor<32x75264xf32>
    %v2148 = stablehlo.subtract %v2146, %v2147 : tensor<32x75264xf32>
    %v2149 = stablehlo.divide %v2136, %v2125 : tensor<32x75264xf32>
    %v2150 = stablehlo.multiply %v2149, %v2148 : tensor<32x75264xf32>
    %v2151 = stablehlo.reshape %v2150 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2152 = stablehlo.reverse %s2b3dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2153 = stablehlo.convolution(%v2151, %v2152)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2155 = stablehlo.add %v2154, %v2036 : tensor<32x75264xf32>
    %v2156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2157 = stablehlo.reshape %v555 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2158 = stablehlo.reshape %v2036 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2159 = stablehlo.multiply %v2157, %v2158 : tensor<32x384x14x14xf32>
    %v2160 = stablehlo.reduce(%v2159 init: %v2156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2161 = stablehlo.reshape %v550 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2162 = stablehlo.reshape %v2090 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2163 = stablehlo.transpose %v2161, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2164 = stablehlo.transpose %v2162, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2165 = stablehlo.convolution(%v2163, %v2164)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2166 = stablehlo.transpose %v2165, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2167 = stablehlo.reshape %v2090 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2168 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2169 = stablehlo.reduce(%v2167 init: %v2168) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2170 = stablehlo.reshape %v532 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2171 = stablehlo.reshape %v2118 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2172 = stablehlo.transpose %v2170, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2173 = stablehlo.transpose %v2171, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2174 = stablehlo.convolution(%v2172, %v2173)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2175 = stablehlo.transpose %v2174, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2176 = stablehlo.reshape %v2118 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2178 = stablehlo.reduce(%v2176 init: %v2177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2179 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2180 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2181 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2182 = stablehlo.reduce(%v514 init: %v2179) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2183 = stablehlo.broadcast_in_dim %v2182, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2184 = stablehlo.divide %v2183, %v2180 : tensor<32x75264xf32>
    %v2185 = stablehlo.subtract %v514, %v2184 : tensor<32x75264xf32>
    %v2186 = stablehlo.multiply %v2185, %v2185 : tensor<32x75264xf32>
    %v2187 = stablehlo.reduce(%v2186 init: %v2179) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2188 = stablehlo.broadcast_in_dim %v2187, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2189 = stablehlo.divide %v2188, %v2180 : tensor<32x75264xf32>
    %v2190 = stablehlo.add %v2189, %v2181 : tensor<32x75264xf32>
    %v2191 = stablehlo.rsqrt %v2190 : tensor<32x75264xf32>
    %v2192 = stablehlo.multiply %v2185, %v2191 : tensor<32x75264xf32>
    %v2193 = stablehlo.multiply %v2123, %v2192 : tensor<32x75264xf32>
    %v2194 = stablehlo.reduce(%v2193 init: %v2179) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2195 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2196 = stablehlo.reduce(%v2123 init: %v2195) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2197 = stablehlo.reshape %v509 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2198 = stablehlo.reshape %v2150 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2199 = stablehlo.transpose %v2197, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2200 = stablehlo.transpose %v2198, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2201 = stablehlo.convolution(%v2199, %v2200)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2202 = stablehlo.reshape %v2201 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2203 = stablehlo.reshape %v2150 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2205 = stablehlo.reduce(%v2203 init: %v2204) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2206 = stablehlo.reshape %v2155 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2207 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2208 = stablehlo.multiply %v2206, %v2207 : tensor<32x384x14x14xf32>
    %v2209 = stablehlo.reshape %v2208 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2210 = stablehlo.reshape %v2209 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2211 = stablehlo.transpose %s2b2pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2212 = stablehlo.reverse %v2211, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2213 = stablehlo.convolution(%v2210, %v2212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2214 = stablehlo.reshape %v2213 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2215 = stablehlo.multiply %v486, %v486 : tensor<32x301056xf32>
    %v2216 = stablehlo.multiply %v2215, %v486 : tensor<32x301056xf32>
    %v2217 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2218 = stablehlo.multiply %v2217, %v2216 : tensor<32x301056xf32>
    %v2219 = stablehlo.add %v486, %v2218 : tensor<32x301056xf32>
    %v2220 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2221 = stablehlo.multiply %v2220, %v2219 : tensor<32x301056xf32>
    %v2222 = stablehlo.tanh %v2221 : tensor<32x301056xf32>
    %v2223 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2224 = stablehlo.add %v2223, %v2222 : tensor<32x301056xf32>
    %v2225 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2226 = stablehlo.multiply %v2225, %v2224 : tensor<32x301056xf32>
    %v2227 = stablehlo.multiply %v2222, %v2222 : tensor<32x301056xf32>
    %v2228 = stablehlo.subtract %v2223, %v2227 : tensor<32x301056xf32>
    %v2229 = stablehlo.multiply %v2225, %v486 : tensor<32x301056xf32>
    %v2230 = stablehlo.multiply %v2229, %v2228 : tensor<32x301056xf32>
    %v2231 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2232 = stablehlo.multiply %v2231, %v2215 : tensor<32x301056xf32>
    %v2233 = stablehlo.add %v2223, %v2232 : tensor<32x301056xf32>
    %v2234 = stablehlo.multiply %v2220, %v2233 : tensor<32x301056xf32>
    %v2235 = stablehlo.multiply %v2230, %v2234 : tensor<32x301056xf32>
    %v2236 = stablehlo.add %v2226, %v2235 : tensor<32x301056xf32>
    %v2237 = stablehlo.multiply %v2214, %v2236 : tensor<32x301056xf32>
    %v2238 = stablehlo.reshape %v2237 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2239 = stablehlo.transpose %s2b2eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2240 = stablehlo.reverse %v2239, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2241 = stablehlo.convolution(%v2238, %v2240)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2242 = stablehlo.reshape %v2241 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2243 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2244 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2245 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2246 = stablehlo.reduce(%v463 init: %v2243) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2247 = stablehlo.broadcast_in_dim %v2246, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2248 = stablehlo.divide %v2247, %v2244 : tensor<32x75264xf32>
    %v2249 = stablehlo.subtract %v463, %v2248 : tensor<32x75264xf32>
    %v2250 = stablehlo.multiply %v2249, %v2249 : tensor<32x75264xf32>
    %v2251 = stablehlo.reduce(%v2250 init: %v2243) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2252 = stablehlo.broadcast_in_dim %v2251, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2253 = stablehlo.divide %v2252, %v2244 : tensor<32x75264xf32>
    %v2254 = stablehlo.add %v2253, %v2245 : tensor<32x75264xf32>
    %v2255 = stablehlo.rsqrt %v2254 : tensor<32x75264xf32>
    %v2256 = stablehlo.multiply %v2249, %v2255 : tensor<32x75264xf32>
    %v2257 = stablehlo.broadcast_in_dim %s2b2ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2258 = stablehlo.multiply %v2257, %v2242 : tensor<32x75264xf32>
    %v2259 = stablehlo.reduce(%v2258 init: %v2243) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2260 = stablehlo.broadcast_in_dim %v2259, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2261 = stablehlo.multiply %v2256, %v2258 : tensor<32x75264xf32>
    %v2262 = stablehlo.reduce(%v2261 init: %v2243) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2263 = stablehlo.broadcast_in_dim %v2262, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2264 = stablehlo.multiply %v2258, %v2244 : tensor<32x75264xf32>
    %v2265 = stablehlo.subtract %v2264, %v2260 : tensor<32x75264xf32>
    %v2266 = stablehlo.multiply %v2256, %v2263 : tensor<32x75264xf32>
    %v2267 = stablehlo.subtract %v2265, %v2266 : tensor<32x75264xf32>
    %v2268 = stablehlo.divide %v2255, %v2244 : tensor<32x75264xf32>
    %v2269 = stablehlo.multiply %v2268, %v2267 : tensor<32x75264xf32>
    %v2270 = stablehlo.reshape %v2269 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2271 = stablehlo.reverse %s2b2dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2272 = stablehlo.convolution(%v2270, %v2271)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2273 = stablehlo.reshape %v2272 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2274 = stablehlo.add %v2273, %v2155 : tensor<32x75264xf32>
    %v2275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2276 = stablehlo.reshape %v504 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2277 = stablehlo.reshape %v2155 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2278 = stablehlo.multiply %v2276, %v2277 : tensor<32x384x14x14xf32>
    %v2279 = stablehlo.reduce(%v2278 init: %v2275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2280 = stablehlo.reshape %v499 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2281 = stablehlo.reshape %v2209 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2282 = stablehlo.transpose %v2280, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2283 = stablehlo.transpose %v2281, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2284 = stablehlo.convolution(%v2282, %v2283)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2285 = stablehlo.transpose %v2284, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2286 = stablehlo.reshape %v2209 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2288 = stablehlo.reduce(%v2286 init: %v2287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2289 = stablehlo.reshape %v481 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2290 = stablehlo.reshape %v2237 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2291 = stablehlo.transpose %v2289, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2292 = stablehlo.transpose %v2290, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2293 = stablehlo.convolution(%v2291, %v2292)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2294 = stablehlo.transpose %v2293, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2295 = stablehlo.reshape %v2237 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2297 = stablehlo.reduce(%v2295 init: %v2296) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2299 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2300 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2301 = stablehlo.reduce(%v463 init: %v2298) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2302 = stablehlo.broadcast_in_dim %v2301, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2303 = stablehlo.divide %v2302, %v2299 : tensor<32x75264xf32>
    %v2304 = stablehlo.subtract %v463, %v2303 : tensor<32x75264xf32>
    %v2305 = stablehlo.multiply %v2304, %v2304 : tensor<32x75264xf32>
    %v2306 = stablehlo.reduce(%v2305 init: %v2298) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2307 = stablehlo.broadcast_in_dim %v2306, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2308 = stablehlo.divide %v2307, %v2299 : tensor<32x75264xf32>
    %v2309 = stablehlo.add %v2308, %v2300 : tensor<32x75264xf32>
    %v2310 = stablehlo.rsqrt %v2309 : tensor<32x75264xf32>
    %v2311 = stablehlo.multiply %v2304, %v2310 : tensor<32x75264xf32>
    %v2312 = stablehlo.multiply %v2242, %v2311 : tensor<32x75264xf32>
    %v2313 = stablehlo.reduce(%v2312 init: %v2298) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2315 = stablehlo.reduce(%v2242 init: %v2314) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2316 = stablehlo.reshape %v458 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2317 = stablehlo.reshape %v2269 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2318 = stablehlo.transpose %v2316, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2319 = stablehlo.transpose %v2317, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2320 = stablehlo.convolution(%v2318, %v2319)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2321 = stablehlo.reshape %v2320 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2322 = stablehlo.reshape %v2269 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2324 = stablehlo.reduce(%v2322 init: %v2323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2325 = stablehlo.reshape %v2274 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2326 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2327 = stablehlo.multiply %v2325, %v2326 : tensor<32x384x14x14xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2329 = stablehlo.reshape %v2328 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2330 = stablehlo.transpose %s2b1pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2331 = stablehlo.reverse %v2330, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2332 = stablehlo.convolution(%v2329, %v2331)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2333 = stablehlo.reshape %v2332 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2334 = stablehlo.multiply %v435, %v435 : tensor<32x301056xf32>
    %v2335 = stablehlo.multiply %v2334, %v435 : tensor<32x301056xf32>
    %v2336 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2337 = stablehlo.multiply %v2336, %v2335 : tensor<32x301056xf32>
    %v2338 = stablehlo.add %v435, %v2337 : tensor<32x301056xf32>
    %v2339 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2340 = stablehlo.multiply %v2339, %v2338 : tensor<32x301056xf32>
    %v2341 = stablehlo.tanh %v2340 : tensor<32x301056xf32>
    %v2342 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2343 = stablehlo.add %v2342, %v2341 : tensor<32x301056xf32>
    %v2344 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2345 = stablehlo.multiply %v2344, %v2343 : tensor<32x301056xf32>
    %v2346 = stablehlo.multiply %v2341, %v2341 : tensor<32x301056xf32>
    %v2347 = stablehlo.subtract %v2342, %v2346 : tensor<32x301056xf32>
    %v2348 = stablehlo.multiply %v2344, %v435 : tensor<32x301056xf32>
    %v2349 = stablehlo.multiply %v2348, %v2347 : tensor<32x301056xf32>
    %v2350 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2351 = stablehlo.multiply %v2350, %v2334 : tensor<32x301056xf32>
    %v2352 = stablehlo.add %v2342, %v2351 : tensor<32x301056xf32>
    %v2353 = stablehlo.multiply %v2339, %v2352 : tensor<32x301056xf32>
    %v2354 = stablehlo.multiply %v2349, %v2353 : tensor<32x301056xf32>
    %v2355 = stablehlo.add %v2345, %v2354 : tensor<32x301056xf32>
    %v2356 = stablehlo.multiply %v2333, %v2355 : tensor<32x301056xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2358 = stablehlo.transpose %s2b1eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2359 = stablehlo.reverse %v2358, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2360 = stablehlo.convolution(%v2357, %v2359)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2361 = stablehlo.reshape %v2360 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2363 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2364 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2365 = stablehlo.reduce(%v412 init: %v2362) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2366 = stablehlo.broadcast_in_dim %v2365, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2367 = stablehlo.divide %v2366, %v2363 : tensor<32x75264xf32>
    %v2368 = stablehlo.subtract %v412, %v2367 : tensor<32x75264xf32>
    %v2369 = stablehlo.multiply %v2368, %v2368 : tensor<32x75264xf32>
    %v2370 = stablehlo.reduce(%v2369 init: %v2362) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2371 = stablehlo.broadcast_in_dim %v2370, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2372 = stablehlo.divide %v2371, %v2363 : tensor<32x75264xf32>
    %v2373 = stablehlo.add %v2372, %v2364 : tensor<32x75264xf32>
    %v2374 = stablehlo.rsqrt %v2373 : tensor<32x75264xf32>
    %v2375 = stablehlo.multiply %v2368, %v2374 : tensor<32x75264xf32>
    %v2376 = stablehlo.broadcast_in_dim %s2b1ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2377 = stablehlo.multiply %v2376, %v2361 : tensor<32x75264xf32>
    %v2378 = stablehlo.reduce(%v2377 init: %v2362) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2379 = stablehlo.broadcast_in_dim %v2378, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2380 = stablehlo.multiply %v2375, %v2377 : tensor<32x75264xf32>
    %v2381 = stablehlo.reduce(%v2380 init: %v2362) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2382 = stablehlo.broadcast_in_dim %v2381, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2383 = stablehlo.multiply %v2377, %v2363 : tensor<32x75264xf32>
    %v2384 = stablehlo.subtract %v2383, %v2379 : tensor<32x75264xf32>
    %v2385 = stablehlo.multiply %v2375, %v2382 : tensor<32x75264xf32>
    %v2386 = stablehlo.subtract %v2384, %v2385 : tensor<32x75264xf32>
    %v2387 = stablehlo.divide %v2374, %v2363 : tensor<32x75264xf32>
    %v2388 = stablehlo.multiply %v2387, %v2386 : tensor<32x75264xf32>
    %v2389 = stablehlo.reshape %v2388 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2390 = stablehlo.reverse %s2b1dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2391 = stablehlo.convolution(%v2389, %v2390)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2393 = stablehlo.add %v2392, %v2274 : tensor<32x75264xf32>
    %v2394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2395 = stablehlo.reshape %v453 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2396 = stablehlo.reshape %v2274 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2397 = stablehlo.multiply %v2395, %v2396 : tensor<32x384x14x14xf32>
    %v2398 = stablehlo.reduce(%v2397 init: %v2394) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2399 = stablehlo.reshape %v448 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2400 = stablehlo.reshape %v2328 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2401 = stablehlo.transpose %v2399, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2402 = stablehlo.transpose %v2400, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2403 = stablehlo.convolution(%v2401, %v2402)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2404 = stablehlo.transpose %v2403, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2405 = stablehlo.reshape %v2328 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2407 = stablehlo.reduce(%v2405 init: %v2406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2408 = stablehlo.reshape %v430 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2409 = stablehlo.reshape %v2356 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2410 = stablehlo.transpose %v2408, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2411 = stablehlo.transpose %v2409, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2412 = stablehlo.convolution(%v2410, %v2411)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2413 = stablehlo.transpose %v2412, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2414 = stablehlo.reshape %v2356 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2416 = stablehlo.reduce(%v2414 init: %v2415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2418 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2419 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2420 = stablehlo.reduce(%v412 init: %v2417) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2421 = stablehlo.broadcast_in_dim %v2420, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2422 = stablehlo.divide %v2421, %v2418 : tensor<32x75264xf32>
    %v2423 = stablehlo.subtract %v412, %v2422 : tensor<32x75264xf32>
    %v2424 = stablehlo.multiply %v2423, %v2423 : tensor<32x75264xf32>
    %v2425 = stablehlo.reduce(%v2424 init: %v2417) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2426 = stablehlo.broadcast_in_dim %v2425, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2427 = stablehlo.divide %v2426, %v2418 : tensor<32x75264xf32>
    %v2428 = stablehlo.add %v2427, %v2419 : tensor<32x75264xf32>
    %v2429 = stablehlo.rsqrt %v2428 : tensor<32x75264xf32>
    %v2430 = stablehlo.multiply %v2423, %v2429 : tensor<32x75264xf32>
    %v2431 = stablehlo.multiply %v2361, %v2430 : tensor<32x75264xf32>
    %v2432 = stablehlo.reduce(%v2431 init: %v2417) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2433 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2434 = stablehlo.reduce(%v2361 init: %v2433) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2435 = stablehlo.reshape %v407 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2436 = stablehlo.reshape %v2388 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2437 = stablehlo.transpose %v2435, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2438 = stablehlo.transpose %v2436, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2439 = stablehlo.convolution(%v2437, %v2438)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2440 = stablehlo.reshape %v2439 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2441 = stablehlo.reshape %v2388 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2443 = stablehlo.reduce(%v2441 init: %v2442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2444 = stablehlo.reshape %v2393 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2445 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2446 = stablehlo.multiply %v2444, %v2445 : tensor<32x384x14x14xf32>
    %v2447 = stablehlo.reshape %v2446 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2448 = stablehlo.reshape %v2447 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2449 = stablehlo.transpose %s2b0pW, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2450 = stablehlo.reverse %v2449, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2451 = stablehlo.convolution(%v2448, %v2450)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2452 = stablehlo.reshape %v2451 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2453 = stablehlo.multiply %v384, %v384 : tensor<32x301056xf32>
    %v2454 = stablehlo.multiply %v2453, %v384 : tensor<32x301056xf32>
    %v2455 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2456 = stablehlo.multiply %v2455, %v2454 : tensor<32x301056xf32>
    %v2457 = stablehlo.add %v384, %v2456 : tensor<32x301056xf32>
    %v2458 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2459 = stablehlo.multiply %v2458, %v2457 : tensor<32x301056xf32>
    %v2460 = stablehlo.tanh %v2459 : tensor<32x301056xf32>
    %v2461 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2462 = stablehlo.add %v2461, %v2460 : tensor<32x301056xf32>
    %v2463 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2464 = stablehlo.multiply %v2463, %v2462 : tensor<32x301056xf32>
    %v2465 = stablehlo.multiply %v2460, %v2460 : tensor<32x301056xf32>
    %v2466 = stablehlo.subtract %v2461, %v2465 : tensor<32x301056xf32>
    %v2467 = stablehlo.multiply %v2463, %v384 : tensor<32x301056xf32>
    %v2468 = stablehlo.multiply %v2467, %v2466 : tensor<32x301056xf32>
    %v2469 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2470 = stablehlo.multiply %v2469, %v2453 : tensor<32x301056xf32>
    %v2471 = stablehlo.add %v2461, %v2470 : tensor<32x301056xf32>
    %v2472 = stablehlo.multiply %v2458, %v2471 : tensor<32x301056xf32>
    %v2473 = stablehlo.multiply %v2468, %v2472 : tensor<32x301056xf32>
    %v2474 = stablehlo.add %v2464, %v2473 : tensor<32x301056xf32>
    %v2475 = stablehlo.multiply %v2452, %v2474 : tensor<32x301056xf32>
    %v2476 = stablehlo.reshape %v2475 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2477 = stablehlo.transpose %s2b0eW, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2478 = stablehlo.reverse %v2477, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2479 = stablehlo.convolution(%v2476, %v2478)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2480 = stablehlo.reshape %v2479 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2482 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2483 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2484 = stablehlo.reduce(%v361 init: %v2481) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2485 = stablehlo.broadcast_in_dim %v2484, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2486 = stablehlo.divide %v2485, %v2482 : tensor<32x75264xf32>
    %v2487 = stablehlo.subtract %v361, %v2486 : tensor<32x75264xf32>
    %v2488 = stablehlo.multiply %v2487, %v2487 : tensor<32x75264xf32>
    %v2489 = stablehlo.reduce(%v2488 init: %v2481) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2490 = stablehlo.broadcast_in_dim %v2489, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2491 = stablehlo.divide %v2490, %v2482 : tensor<32x75264xf32>
    %v2492 = stablehlo.add %v2491, %v2483 : tensor<32x75264xf32>
    %v2493 = stablehlo.rsqrt %v2492 : tensor<32x75264xf32>
    %v2494 = stablehlo.multiply %v2487, %v2493 : tensor<32x75264xf32>
    %v2495 = stablehlo.broadcast_in_dim %s2b0ng, dims = [] : (tensor<f32>) -> tensor<32x75264xf32>
    %v2496 = stablehlo.multiply %v2495, %v2480 : tensor<32x75264xf32>
    %v2497 = stablehlo.reduce(%v2496 init: %v2481) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2498 = stablehlo.broadcast_in_dim %v2497, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2499 = stablehlo.multiply %v2494, %v2496 : tensor<32x75264xf32>
    %v2500 = stablehlo.reduce(%v2499 init: %v2481) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2501 = stablehlo.broadcast_in_dim %v2500, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2502 = stablehlo.multiply %v2496, %v2482 : tensor<32x75264xf32>
    %v2503 = stablehlo.subtract %v2502, %v2498 : tensor<32x75264xf32>
    %v2504 = stablehlo.multiply %v2494, %v2501 : tensor<32x75264xf32>
    %v2505 = stablehlo.subtract %v2503, %v2504 : tensor<32x75264xf32>
    %v2506 = stablehlo.divide %v2493, %v2482 : tensor<32x75264xf32>
    %v2507 = stablehlo.multiply %v2506, %v2505 : tensor<32x75264xf32>
    %v2508 = stablehlo.reshape %v2507 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2509 = stablehlo.reverse %s2b0dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2510 = stablehlo.convolution(%v2508, %v2509)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2511 = stablehlo.reshape %v2510 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2512 = stablehlo.add %v2511, %v2393 : tensor<32x75264xf32>
    %v2513 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2514 = stablehlo.reshape %v402 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2515 = stablehlo.reshape %v2393 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2516 = stablehlo.multiply %v2514, %v2515 : tensor<32x384x14x14xf32>
    %v2517 = stablehlo.reduce(%v2516 init: %v2513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2518 = stablehlo.reshape %v397 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2519 = stablehlo.reshape %v2447 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2520 = stablehlo.transpose %v2518, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2521 = stablehlo.transpose %v2519, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2522 = stablehlo.convolution(%v2520, %v2521)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2523 = stablehlo.transpose %v2522, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2524 = stablehlo.reshape %v2447 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2526 = stablehlo.reduce(%v2524 init: %v2525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2527 = stablehlo.reshape %v379 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2528 = stablehlo.reshape %v2475 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2529 = stablehlo.transpose %v2527, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2530 = stablehlo.transpose %v2528, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2531 = stablehlo.convolution(%v2529, %v2530)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2532 = stablehlo.transpose %v2531, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2533 = stablehlo.reshape %v2475 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2535 = stablehlo.reduce(%v2533 init: %v2534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2537 = stablehlo.constant dense<75264.0> : tensor<32x75264xf32>
    %v2538 = stablehlo.constant dense<1.0e-6> : tensor<32x75264xf32>
    %v2539 = stablehlo.reduce(%v361 init: %v2536) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2540 = stablehlo.broadcast_in_dim %v2539, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2541 = stablehlo.divide %v2540, %v2537 : tensor<32x75264xf32>
    %v2542 = stablehlo.subtract %v361, %v2541 : tensor<32x75264xf32>
    %v2543 = stablehlo.multiply %v2542, %v2542 : tensor<32x75264xf32>
    %v2544 = stablehlo.reduce(%v2543 init: %v2536) applies stablehlo.add across dimensions = [1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<32xf32>
    %v2545 = stablehlo.broadcast_in_dim %v2544, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2546 = stablehlo.divide %v2545, %v2537 : tensor<32x75264xf32>
    %v2547 = stablehlo.add %v2546, %v2538 : tensor<32x75264xf32>
    %v2548 = stablehlo.rsqrt %v2547 : tensor<32x75264xf32>
    %v2549 = stablehlo.multiply %v2542, %v2548 : tensor<32x75264xf32>
    %v2550 = stablehlo.multiply %v2480, %v2549 : tensor<32x75264xf32>
    %v2551 = stablehlo.reduce(%v2550 init: %v2536) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2553 = stablehlo.reduce(%v2480 init: %v2552) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x75264xf32>, tensor<f32>) -> tensor<f32>
    %v2554 = stablehlo.reshape %v356 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2555 = stablehlo.reshape %v2507 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2556 = stablehlo.transpose %v2554, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2557 = stablehlo.transpose %v2555, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2558 = stablehlo.convolution(%v2556, %v2557)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2559 = stablehlo.reshape %v2558 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2560 = stablehlo.reshape %v2507 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2561 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2562 = stablehlo.reduce(%v2560 init: %v2561) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2563 = stablehlo.reshape %v2512 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2565 = stablehlo.pad %v2563, %v2564, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x28x28xf32>
    %v2566 = stablehlo.transpose %d1W, dims = [1, 0, 2, 3] : (tensor<384x192x2x2xf32>) -> tensor<192x384x2x2xf32>
    %v2567 = stablehlo.reverse %v2566, dims = [2, 3] : tensor<192x384x2x2xf32>
    %v2568 = stablehlo.convolution(%v2565, %v2567)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x28x28xf32>, tensor<192x384x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v2569 = stablehlo.reshape %v2568 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2571 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2572 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2573 = stablehlo.reduce(%v333 init: %v2570) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2574 = stablehlo.broadcast_in_dim %v2573, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2575 = stablehlo.divide %v2574, %v2571 : tensor<32x150528xf32>
    %v2576 = stablehlo.subtract %v333, %v2575 : tensor<32x150528xf32>
    %v2577 = stablehlo.multiply %v2576, %v2576 : tensor<32x150528xf32>
    %v2578 = stablehlo.reduce(%v2577 init: %v2570) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2579 = stablehlo.broadcast_in_dim %v2578, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2580 = stablehlo.divide %v2579, %v2571 : tensor<32x150528xf32>
    %v2581 = stablehlo.add %v2580, %v2572 : tensor<32x150528xf32>
    %v2582 = stablehlo.rsqrt %v2581 : tensor<32x150528xf32>
    %v2583 = stablehlo.multiply %v2576, %v2582 : tensor<32x150528xf32>
    %v2584 = stablehlo.broadcast_in_dim %d1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v2585 = stablehlo.multiply %v2584, %v2569 : tensor<32x150528xf32>
    %v2586 = stablehlo.reduce(%v2585 init: %v2570) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2587 = stablehlo.broadcast_in_dim %v2586, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2588 = stablehlo.multiply %v2583, %v2585 : tensor<32x150528xf32>
    %v2589 = stablehlo.reduce(%v2588 init: %v2570) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2590 = stablehlo.broadcast_in_dim %v2589, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2591 = stablehlo.multiply %v2585, %v2571 : tensor<32x150528xf32>
    %v2592 = stablehlo.subtract %v2591, %v2587 : tensor<32x150528xf32>
    %v2593 = stablehlo.multiply %v2583, %v2590 : tensor<32x150528xf32>
    %v2594 = stablehlo.subtract %v2592, %v2593 : tensor<32x150528xf32>
    %v2595 = stablehlo.divide %v2582, %v2571 : tensor<32x150528xf32>
    %v2596 = stablehlo.multiply %v2595, %v2594 : tensor<32x150528xf32>
    %v2597 = stablehlo.reshape %v2512 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2598 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2599 = stablehlo.reduce(%v2597 init: %v2598) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2600 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2601 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2602 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2603 = stablehlo.reduce(%v333 init: %v2600) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2604 = stablehlo.broadcast_in_dim %v2603, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2605 = stablehlo.divide %v2604, %v2601 : tensor<32x150528xf32>
    %v2606 = stablehlo.subtract %v333, %v2605 : tensor<32x150528xf32>
    %v2607 = stablehlo.multiply %v2606, %v2606 : tensor<32x150528xf32>
    %v2608 = stablehlo.reduce(%v2607 init: %v2600) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2609 = stablehlo.broadcast_in_dim %v2608, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2610 = stablehlo.divide %v2609, %v2601 : tensor<32x150528xf32>
    %v2611 = stablehlo.add %v2610, %v2602 : tensor<32x150528xf32>
    %v2612 = stablehlo.rsqrt %v2611 : tensor<32x150528xf32>
    %v2613 = stablehlo.multiply %v2606, %v2612 : tensor<32x150528xf32>
    %v2614 = stablehlo.multiply %v2569, %v2613 : tensor<32x150528xf32>
    %v2615 = stablehlo.reduce(%v2614 init: %v2600) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2617 = stablehlo.reduce(%v2569 init: %v2616) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %dd1Wxi = stablehlo.reshape %v351 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %dd1Wdi = stablehlo.reshape %v2512 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %dd1Wu = stablehlo.pad %dd1Wdi, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x27x27xf32>
    %dd1Wxt = stablehlo.transpose %dd1Wxi, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %dd1Wdt = stablehlo.transpose %dd1Wu, dims = [1, 0, 2, 3] : (tensor<32x384x27x27xf32>) -> tensor<384x32x27x27xf32>
    %dd1Wraw = stablehlo.convolution(%dd1Wxt, %dd1Wdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<384x32x27x27xf32>) -> tensor<192x384x2x2xf32>
    %dd1W = stablehlo.transpose %dd1Wraw, dims = [1, 0, 2, 3] : (tensor<192x384x2x2xf32>) -> tensor<384x192x2x2xf32>
    %v2618 = stablehlo.reshape %v2596 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2619 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v2620 = stablehlo.multiply %v2618, %v2619 : tensor<32x192x28x28xf32>
    %v2621 = stablehlo.reshape %v2620 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2622 = stablehlo.reshape %v2621 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2623 = stablehlo.transpose %s1b2pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2624 = stablehlo.reverse %v2623, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v2625 = stablehlo.convolution(%v2622, %v2624)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v2626 = stablehlo.reshape %v2625 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v2627 = stablehlo.multiply %v310, %v310 : tensor<32x602112xf32>
    %v2628 = stablehlo.multiply %v2627, %v310 : tensor<32x602112xf32>
    %v2629 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v2630 = stablehlo.multiply %v2629, %v2628 : tensor<32x602112xf32>
    %v2631 = stablehlo.add %v310, %v2630 : tensor<32x602112xf32>
    %v2632 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v2633 = stablehlo.multiply %v2632, %v2631 : tensor<32x602112xf32>
    %v2634 = stablehlo.tanh %v2633 : tensor<32x602112xf32>
    %v2635 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v2636 = stablehlo.add %v2635, %v2634 : tensor<32x602112xf32>
    %v2637 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v2638 = stablehlo.multiply %v2637, %v2636 : tensor<32x602112xf32>
    %v2639 = stablehlo.multiply %v2634, %v2634 : tensor<32x602112xf32>
    %v2640 = stablehlo.subtract %v2635, %v2639 : tensor<32x602112xf32>
    %v2641 = stablehlo.multiply %v2637, %v310 : tensor<32x602112xf32>
    %v2642 = stablehlo.multiply %v2641, %v2640 : tensor<32x602112xf32>
    %v2643 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v2644 = stablehlo.multiply %v2643, %v2627 : tensor<32x602112xf32>
    %v2645 = stablehlo.add %v2635, %v2644 : tensor<32x602112xf32>
    %v2646 = stablehlo.multiply %v2632, %v2645 : tensor<32x602112xf32>
    %v2647 = stablehlo.multiply %v2642, %v2646 : tensor<32x602112xf32>
    %v2648 = stablehlo.add %v2638, %v2647 : tensor<32x602112xf32>
    %v2649 = stablehlo.multiply %v2626, %v2648 : tensor<32x602112xf32>
    %v2650 = stablehlo.reshape %v2649 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2651 = stablehlo.transpose %s1b2eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2652 = stablehlo.reverse %v2651, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v2653 = stablehlo.convolution(%v2650, %v2652)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v2654 = stablehlo.reshape %v2653 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2655 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2656 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2657 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2658 = stablehlo.reduce(%v287 init: %v2655) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2659 = stablehlo.broadcast_in_dim %v2658, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2660 = stablehlo.divide %v2659, %v2656 : tensor<32x150528xf32>
    %v2661 = stablehlo.subtract %v287, %v2660 : tensor<32x150528xf32>
    %v2662 = stablehlo.multiply %v2661, %v2661 : tensor<32x150528xf32>
    %v2663 = stablehlo.reduce(%v2662 init: %v2655) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2664 = stablehlo.broadcast_in_dim %v2663, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2665 = stablehlo.divide %v2664, %v2656 : tensor<32x150528xf32>
    %v2666 = stablehlo.add %v2665, %v2657 : tensor<32x150528xf32>
    %v2667 = stablehlo.rsqrt %v2666 : tensor<32x150528xf32>
    %v2668 = stablehlo.multiply %v2661, %v2667 : tensor<32x150528xf32>
    %v2669 = stablehlo.broadcast_in_dim %s1b2ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v2670 = stablehlo.multiply %v2669, %v2654 : tensor<32x150528xf32>
    %v2671 = stablehlo.reduce(%v2670 init: %v2655) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2672 = stablehlo.broadcast_in_dim %v2671, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2673 = stablehlo.multiply %v2668, %v2670 : tensor<32x150528xf32>
    %v2674 = stablehlo.reduce(%v2673 init: %v2655) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2675 = stablehlo.broadcast_in_dim %v2674, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2676 = stablehlo.multiply %v2670, %v2656 : tensor<32x150528xf32>
    %v2677 = stablehlo.subtract %v2676, %v2672 : tensor<32x150528xf32>
    %v2678 = stablehlo.multiply %v2668, %v2675 : tensor<32x150528xf32>
    %v2679 = stablehlo.subtract %v2677, %v2678 : tensor<32x150528xf32>
    %v2680 = stablehlo.divide %v2667, %v2656 : tensor<32x150528xf32>
    %v2681 = stablehlo.multiply %v2680, %v2679 : tensor<32x150528xf32>
    %v2682 = stablehlo.reshape %v2681 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2683 = stablehlo.reverse %s1b2dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v2684 = stablehlo.convolution(%v2682, %v2683)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v2685 = stablehlo.reshape %v2684 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2686 = stablehlo.add %v2685, %v2596 : tensor<32x150528xf32>
    %v2687 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2688 = stablehlo.reshape %v328 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2689 = stablehlo.reshape %v2596 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2690 = stablehlo.multiply %v2688, %v2689 : tensor<32x192x28x28xf32>
    %v2691 = stablehlo.reduce(%v2690 init: %v2687) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2692 = stablehlo.reshape %v323 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2693 = stablehlo.reshape %v2621 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2694 = stablehlo.transpose %v2692, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v2695 = stablehlo.transpose %v2693, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2696 = stablehlo.convolution(%v2694, %v2695)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v2697 = stablehlo.transpose %v2696, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2698 = stablehlo.reshape %v2621 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2700 = stablehlo.reduce(%v2698 init: %v2699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2701 = stablehlo.reshape %v305 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2702 = stablehlo.reshape %v2649 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2703 = stablehlo.transpose %v2701, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2704 = stablehlo.transpose %v2702, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v2705 = stablehlo.convolution(%v2703, %v2704)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v2706 = stablehlo.transpose %v2705, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2707 = stablehlo.reshape %v2649 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2709 = stablehlo.reduce(%v2707 init: %v2708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v2710 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2711 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2712 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2713 = stablehlo.reduce(%v287 init: %v2710) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2714 = stablehlo.broadcast_in_dim %v2713, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2715 = stablehlo.divide %v2714, %v2711 : tensor<32x150528xf32>
    %v2716 = stablehlo.subtract %v287, %v2715 : tensor<32x150528xf32>
    %v2717 = stablehlo.multiply %v2716, %v2716 : tensor<32x150528xf32>
    %v2718 = stablehlo.reduce(%v2717 init: %v2710) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2719 = stablehlo.broadcast_in_dim %v2718, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2720 = stablehlo.divide %v2719, %v2711 : tensor<32x150528xf32>
    %v2721 = stablehlo.add %v2720, %v2712 : tensor<32x150528xf32>
    %v2722 = stablehlo.rsqrt %v2721 : tensor<32x150528xf32>
    %v2723 = stablehlo.multiply %v2716, %v2722 : tensor<32x150528xf32>
    %v2724 = stablehlo.multiply %v2654, %v2723 : tensor<32x150528xf32>
    %v2725 = stablehlo.reduce(%v2724 init: %v2710) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2727 = stablehlo.reduce(%v2654 init: %v2726) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2728 = stablehlo.reshape %v282 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2729 = stablehlo.reshape %v2681 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2730 = stablehlo.transpose %v2728, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2731 = stablehlo.transpose %v2729, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2732 = stablehlo.convolution(%v2730, %v2731)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v2733 = stablehlo.reshape %v2732 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v2734 = stablehlo.reshape %v2681 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2736 = stablehlo.reduce(%v2734 init: %v2735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2737 = stablehlo.reshape %v2686 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2738 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v2739 = stablehlo.multiply %v2737, %v2738 : tensor<32x192x28x28xf32>
    %v2740 = stablehlo.reshape %v2739 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2741 = stablehlo.reshape %v2740 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2742 = stablehlo.transpose %s1b1pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2743 = stablehlo.reverse %v2742, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v2744 = stablehlo.convolution(%v2741, %v2743)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v2745 = stablehlo.reshape %v2744 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v2746 = stablehlo.multiply %v259, %v259 : tensor<32x602112xf32>
    %v2747 = stablehlo.multiply %v2746, %v259 : tensor<32x602112xf32>
    %v2748 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v2749 = stablehlo.multiply %v2748, %v2747 : tensor<32x602112xf32>
    %v2750 = stablehlo.add %v259, %v2749 : tensor<32x602112xf32>
    %v2751 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v2752 = stablehlo.multiply %v2751, %v2750 : tensor<32x602112xf32>
    %v2753 = stablehlo.tanh %v2752 : tensor<32x602112xf32>
    %v2754 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v2755 = stablehlo.add %v2754, %v2753 : tensor<32x602112xf32>
    %v2756 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v2757 = stablehlo.multiply %v2756, %v2755 : tensor<32x602112xf32>
    %v2758 = stablehlo.multiply %v2753, %v2753 : tensor<32x602112xf32>
    %v2759 = stablehlo.subtract %v2754, %v2758 : tensor<32x602112xf32>
    %v2760 = stablehlo.multiply %v2756, %v259 : tensor<32x602112xf32>
    %v2761 = stablehlo.multiply %v2760, %v2759 : tensor<32x602112xf32>
    %v2762 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v2763 = stablehlo.multiply %v2762, %v2746 : tensor<32x602112xf32>
    %v2764 = stablehlo.add %v2754, %v2763 : tensor<32x602112xf32>
    %v2765 = stablehlo.multiply %v2751, %v2764 : tensor<32x602112xf32>
    %v2766 = stablehlo.multiply %v2761, %v2765 : tensor<32x602112xf32>
    %v2767 = stablehlo.add %v2757, %v2766 : tensor<32x602112xf32>
    %v2768 = stablehlo.multiply %v2745, %v2767 : tensor<32x602112xf32>
    %v2769 = stablehlo.reshape %v2768 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2770 = stablehlo.transpose %s1b1eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2771 = stablehlo.reverse %v2770, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v2772 = stablehlo.convolution(%v2769, %v2771)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v2773 = stablehlo.reshape %v2772 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2775 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2776 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2777 = stablehlo.reduce(%v236 init: %v2774) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2778 = stablehlo.broadcast_in_dim %v2777, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2779 = stablehlo.divide %v2778, %v2775 : tensor<32x150528xf32>
    %v2780 = stablehlo.subtract %v236, %v2779 : tensor<32x150528xf32>
    %v2781 = stablehlo.multiply %v2780, %v2780 : tensor<32x150528xf32>
    %v2782 = stablehlo.reduce(%v2781 init: %v2774) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2783 = stablehlo.broadcast_in_dim %v2782, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2784 = stablehlo.divide %v2783, %v2775 : tensor<32x150528xf32>
    %v2785 = stablehlo.add %v2784, %v2776 : tensor<32x150528xf32>
    %v2786 = stablehlo.rsqrt %v2785 : tensor<32x150528xf32>
    %v2787 = stablehlo.multiply %v2780, %v2786 : tensor<32x150528xf32>
    %v2788 = stablehlo.broadcast_in_dim %s1b1ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v2789 = stablehlo.multiply %v2788, %v2773 : tensor<32x150528xf32>
    %v2790 = stablehlo.reduce(%v2789 init: %v2774) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2791 = stablehlo.broadcast_in_dim %v2790, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2792 = stablehlo.multiply %v2787, %v2789 : tensor<32x150528xf32>
    %v2793 = stablehlo.reduce(%v2792 init: %v2774) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2794 = stablehlo.broadcast_in_dim %v2793, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2795 = stablehlo.multiply %v2789, %v2775 : tensor<32x150528xf32>
    %v2796 = stablehlo.subtract %v2795, %v2791 : tensor<32x150528xf32>
    %v2797 = stablehlo.multiply %v2787, %v2794 : tensor<32x150528xf32>
    %v2798 = stablehlo.subtract %v2796, %v2797 : tensor<32x150528xf32>
    %v2799 = stablehlo.divide %v2786, %v2775 : tensor<32x150528xf32>
    %v2800 = stablehlo.multiply %v2799, %v2798 : tensor<32x150528xf32>
    %v2801 = stablehlo.reshape %v2800 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2802 = stablehlo.reverse %s1b1dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v2803 = stablehlo.convolution(%v2801, %v2802)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v2804 = stablehlo.reshape %v2803 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2805 = stablehlo.add %v2804, %v2686 : tensor<32x150528xf32>
    %v2806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2807 = stablehlo.reshape %v277 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2808 = stablehlo.reshape %v2686 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2809 = stablehlo.multiply %v2807, %v2808 : tensor<32x192x28x28xf32>
    %v2810 = stablehlo.reduce(%v2809 init: %v2806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2811 = stablehlo.reshape %v272 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2812 = stablehlo.reshape %v2740 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2813 = stablehlo.transpose %v2811, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v2814 = stablehlo.transpose %v2812, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2815 = stablehlo.convolution(%v2813, %v2814)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v2816 = stablehlo.transpose %v2815, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2817 = stablehlo.reshape %v2740 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2819 = stablehlo.reduce(%v2817 init: %v2818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2820 = stablehlo.reshape %v254 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2821 = stablehlo.reshape %v2768 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2822 = stablehlo.transpose %v2820, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2823 = stablehlo.transpose %v2821, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v2824 = stablehlo.convolution(%v2822, %v2823)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v2825 = stablehlo.transpose %v2824, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2826 = stablehlo.reshape %v2768 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2827 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2828 = stablehlo.reduce(%v2826 init: %v2827) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v2829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2830 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2831 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2832 = stablehlo.reduce(%v236 init: %v2829) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2833 = stablehlo.broadcast_in_dim %v2832, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2834 = stablehlo.divide %v2833, %v2830 : tensor<32x150528xf32>
    %v2835 = stablehlo.subtract %v236, %v2834 : tensor<32x150528xf32>
    %v2836 = stablehlo.multiply %v2835, %v2835 : tensor<32x150528xf32>
    %v2837 = stablehlo.reduce(%v2836 init: %v2829) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2838 = stablehlo.broadcast_in_dim %v2837, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2839 = stablehlo.divide %v2838, %v2830 : tensor<32x150528xf32>
    %v2840 = stablehlo.add %v2839, %v2831 : tensor<32x150528xf32>
    %v2841 = stablehlo.rsqrt %v2840 : tensor<32x150528xf32>
    %v2842 = stablehlo.multiply %v2835, %v2841 : tensor<32x150528xf32>
    %v2843 = stablehlo.multiply %v2773, %v2842 : tensor<32x150528xf32>
    %v2844 = stablehlo.reduce(%v2843 init: %v2829) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2845 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2846 = stablehlo.reduce(%v2773 init: %v2845) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2847 = stablehlo.reshape %v231 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2848 = stablehlo.reshape %v2800 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2849 = stablehlo.transpose %v2847, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2850 = stablehlo.transpose %v2848, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2851 = stablehlo.convolution(%v2849, %v2850)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v2852 = stablehlo.reshape %v2851 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v2853 = stablehlo.reshape %v2800 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2855 = stablehlo.reduce(%v2853 init: %v2854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2856 = stablehlo.reshape %v2805 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2857 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v2858 = stablehlo.multiply %v2856, %v2857 : tensor<32x192x28x28xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2860 = stablehlo.reshape %v2859 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2861 = stablehlo.transpose %s1b0pW, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2862 = stablehlo.reverse %v2861, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v2863 = stablehlo.convolution(%v2860, %v2862)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v2864 = stablehlo.reshape %v2863 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v2865 = stablehlo.multiply %v208, %v208 : tensor<32x602112xf32>
    %v2866 = stablehlo.multiply %v2865, %v208 : tensor<32x602112xf32>
    %v2867 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v2868 = stablehlo.multiply %v2867, %v2866 : tensor<32x602112xf32>
    %v2869 = stablehlo.add %v208, %v2868 : tensor<32x602112xf32>
    %v2870 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v2871 = stablehlo.multiply %v2870, %v2869 : tensor<32x602112xf32>
    %v2872 = stablehlo.tanh %v2871 : tensor<32x602112xf32>
    %v2873 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v2874 = stablehlo.add %v2873, %v2872 : tensor<32x602112xf32>
    %v2875 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v2876 = stablehlo.multiply %v2875, %v2874 : tensor<32x602112xf32>
    %v2877 = stablehlo.multiply %v2872, %v2872 : tensor<32x602112xf32>
    %v2878 = stablehlo.subtract %v2873, %v2877 : tensor<32x602112xf32>
    %v2879 = stablehlo.multiply %v2875, %v208 : tensor<32x602112xf32>
    %v2880 = stablehlo.multiply %v2879, %v2878 : tensor<32x602112xf32>
    %v2881 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v2882 = stablehlo.multiply %v2881, %v2865 : tensor<32x602112xf32>
    %v2883 = stablehlo.add %v2873, %v2882 : tensor<32x602112xf32>
    %v2884 = stablehlo.multiply %v2870, %v2883 : tensor<32x602112xf32>
    %v2885 = stablehlo.multiply %v2880, %v2884 : tensor<32x602112xf32>
    %v2886 = stablehlo.add %v2876, %v2885 : tensor<32x602112xf32>
    %v2887 = stablehlo.multiply %v2864, %v2886 : tensor<32x602112xf32>
    %v2888 = stablehlo.reshape %v2887 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2889 = stablehlo.transpose %s1b0eW, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2890 = stablehlo.reverse %v2889, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v2891 = stablehlo.convolution(%v2888, %v2890)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v2892 = stablehlo.reshape %v2891 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2894 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2895 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2896 = stablehlo.reduce(%v185 init: %v2893) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2897 = stablehlo.broadcast_in_dim %v2896, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2898 = stablehlo.divide %v2897, %v2894 : tensor<32x150528xf32>
    %v2899 = stablehlo.subtract %v185, %v2898 : tensor<32x150528xf32>
    %v2900 = stablehlo.multiply %v2899, %v2899 : tensor<32x150528xf32>
    %v2901 = stablehlo.reduce(%v2900 init: %v2893) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2902 = stablehlo.broadcast_in_dim %v2901, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2903 = stablehlo.divide %v2902, %v2894 : tensor<32x150528xf32>
    %v2904 = stablehlo.add %v2903, %v2895 : tensor<32x150528xf32>
    %v2905 = stablehlo.rsqrt %v2904 : tensor<32x150528xf32>
    %v2906 = stablehlo.multiply %v2899, %v2905 : tensor<32x150528xf32>
    %v2907 = stablehlo.broadcast_in_dim %s1b0ng, dims = [] : (tensor<f32>) -> tensor<32x150528xf32>
    %v2908 = stablehlo.multiply %v2907, %v2892 : tensor<32x150528xf32>
    %v2909 = stablehlo.reduce(%v2908 init: %v2893) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2910 = stablehlo.broadcast_in_dim %v2909, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2911 = stablehlo.multiply %v2906, %v2908 : tensor<32x150528xf32>
    %v2912 = stablehlo.reduce(%v2911 init: %v2893) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2913 = stablehlo.broadcast_in_dim %v2912, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2914 = stablehlo.multiply %v2908, %v2894 : tensor<32x150528xf32>
    %v2915 = stablehlo.subtract %v2914, %v2910 : tensor<32x150528xf32>
    %v2916 = stablehlo.multiply %v2906, %v2913 : tensor<32x150528xf32>
    %v2917 = stablehlo.subtract %v2915, %v2916 : tensor<32x150528xf32>
    %v2918 = stablehlo.divide %v2905, %v2894 : tensor<32x150528xf32>
    %v2919 = stablehlo.multiply %v2918, %v2917 : tensor<32x150528xf32>
    %v2920 = stablehlo.reshape %v2919 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2921 = stablehlo.reverse %s1b0dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v2922 = stablehlo.convolution(%v2920, %v2921)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v2923 = stablehlo.reshape %v2922 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v2924 = stablehlo.add %v2923, %v2805 : tensor<32x150528xf32>
    %v2925 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2926 = stablehlo.reshape %v226 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2927 = stablehlo.reshape %v2805 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2928 = stablehlo.multiply %v2926, %v2927 : tensor<32x192x28x28xf32>
    %v2929 = stablehlo.reduce(%v2928 init: %v2925) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2930 = stablehlo.reshape %v221 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2931 = stablehlo.reshape %v2859 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2932 = stablehlo.transpose %v2930, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v2933 = stablehlo.transpose %v2931, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2934 = stablehlo.convolution(%v2932, %v2933)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v2935 = stablehlo.transpose %v2934, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v2936 = stablehlo.reshape %v2859 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2938 = stablehlo.reduce(%v2936 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2939 = stablehlo.reshape %v203 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2940 = stablehlo.reshape %v2887 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2941 = stablehlo.transpose %v2939, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2942 = stablehlo.transpose %v2940, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v2943 = stablehlo.convolution(%v2941, %v2942)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v2944 = stablehlo.transpose %v2943, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v2945 = stablehlo.reshape %v2887 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v2946 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2947 = stablehlo.reduce(%v2945 init: %v2946) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v2948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2949 = stablehlo.constant dense<150528.0> : tensor<32x150528xf32>
    %v2950 = stablehlo.constant dense<1.0e-6> : tensor<32x150528xf32>
    %v2951 = stablehlo.reduce(%v185 init: %v2948) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2952 = stablehlo.broadcast_in_dim %v2951, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2953 = stablehlo.divide %v2952, %v2949 : tensor<32x150528xf32>
    %v2954 = stablehlo.subtract %v185, %v2953 : tensor<32x150528xf32>
    %v2955 = stablehlo.multiply %v2954, %v2954 : tensor<32x150528xf32>
    %v2956 = stablehlo.reduce(%v2955 init: %v2948) applies stablehlo.add across dimensions = [1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<32xf32>
    %v2957 = stablehlo.broadcast_in_dim %v2956, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v2958 = stablehlo.divide %v2957, %v2949 : tensor<32x150528xf32>
    %v2959 = stablehlo.add %v2958, %v2950 : tensor<32x150528xf32>
    %v2960 = stablehlo.rsqrt %v2959 : tensor<32x150528xf32>
    %v2961 = stablehlo.multiply %v2954, %v2960 : tensor<32x150528xf32>
    %v2962 = stablehlo.multiply %v2892, %v2961 : tensor<32x150528xf32>
    %v2963 = stablehlo.reduce(%v2962 init: %v2948) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2965 = stablehlo.reduce(%v2892 init: %v2964) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x150528xf32>, tensor<f32>) -> tensor<f32>
    %v2966 = stablehlo.reshape %v180 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2967 = stablehlo.reshape %v2919 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2968 = stablehlo.transpose %v2966, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2969 = stablehlo.transpose %v2967, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v2970 = stablehlo.convolution(%v2968, %v2969)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v2971 = stablehlo.reshape %v2970 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v2972 = stablehlo.reshape %v2919 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2973 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2974 = stablehlo.reduce(%v2972 init: %v2973) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v2975 = stablehlo.reshape %v2924 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v2976 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2977 = stablehlo.pad %v2975, %v2976, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x56x56xf32>
    %v2978 = stablehlo.transpose %d0W, dims = [1, 0, 2, 3] : (tensor<192x96x2x2xf32>) -> tensor<96x192x2x2xf32>
    %v2979 = stablehlo.reverse %v2978, dims = [2, 3] : tensor<96x192x2x2xf32>
    %v2980 = stablehlo.convolution(%v2977, %v2979)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x56x56xf32>, tensor<96x192x2x2xf32>) -> tensor<32x96x56x56xf32>
    %v2981 = stablehlo.reshape %v2980 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v2982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2983 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v2984 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v2985 = stablehlo.reduce(%v157 init: %v2982) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v2986 = stablehlo.broadcast_in_dim %v2985, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v2987 = stablehlo.divide %v2986, %v2983 : tensor<32x301056xf32>
    %v2988 = stablehlo.subtract %v157, %v2987 : tensor<32x301056xf32>
    %v2989 = stablehlo.multiply %v2988, %v2988 : tensor<32x301056xf32>
    %v2990 = stablehlo.reduce(%v2989 init: %v2982) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v2991 = stablehlo.broadcast_in_dim %v2990, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v2992 = stablehlo.divide %v2991, %v2983 : tensor<32x301056xf32>
    %v2993 = stablehlo.add %v2992, %v2984 : tensor<32x301056xf32>
    %v2994 = stablehlo.rsqrt %v2993 : tensor<32x301056xf32>
    %v2995 = stablehlo.multiply %v2988, %v2994 : tensor<32x301056xf32>
    %v2996 = stablehlo.broadcast_in_dim %d0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v2997 = stablehlo.multiply %v2996, %v2981 : tensor<32x301056xf32>
    %v2998 = stablehlo.reduce(%v2997 init: %v2982) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v2999 = stablehlo.broadcast_in_dim %v2998, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3000 = stablehlo.multiply %v2995, %v2997 : tensor<32x301056xf32>
    %v3001 = stablehlo.reduce(%v3000 init: %v2982) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3002 = stablehlo.broadcast_in_dim %v3001, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3003 = stablehlo.multiply %v2997, %v2983 : tensor<32x301056xf32>
    %v3004 = stablehlo.subtract %v3003, %v2999 : tensor<32x301056xf32>
    %v3005 = stablehlo.multiply %v2995, %v3002 : tensor<32x301056xf32>
    %v3006 = stablehlo.subtract %v3004, %v3005 : tensor<32x301056xf32>
    %v3007 = stablehlo.divide %v2994, %v2983 : tensor<32x301056xf32>
    %v3008 = stablehlo.multiply %v3007, %v3006 : tensor<32x301056xf32>
    %v3009 = stablehlo.reshape %v2924 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3010 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3011 = stablehlo.reduce(%v3009 init: %v3010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3012 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3013 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3014 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3015 = stablehlo.reduce(%v157 init: %v3012) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3016 = stablehlo.broadcast_in_dim %v3015, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3017 = stablehlo.divide %v3016, %v3013 : tensor<32x301056xf32>
    %v3018 = stablehlo.subtract %v157, %v3017 : tensor<32x301056xf32>
    %v3019 = stablehlo.multiply %v3018, %v3018 : tensor<32x301056xf32>
    %v3020 = stablehlo.reduce(%v3019 init: %v3012) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3021 = stablehlo.broadcast_in_dim %v3020, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3022 = stablehlo.divide %v3021, %v3013 : tensor<32x301056xf32>
    %v3023 = stablehlo.add %v3022, %v3014 : tensor<32x301056xf32>
    %v3024 = stablehlo.rsqrt %v3023 : tensor<32x301056xf32>
    %v3025 = stablehlo.multiply %v3018, %v3024 : tensor<32x301056xf32>
    %v3026 = stablehlo.multiply %v2981, %v3025 : tensor<32x301056xf32>
    %v3027 = stablehlo.reduce(%v3026 init: %v3012) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3028 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3029 = stablehlo.reduce(%v2981 init: %v3028) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %dd0Wxi = stablehlo.reshape %v175 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %dd0Wdi = stablehlo.reshape %v2924 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %dd0Wu = stablehlo.pad %dd0Wdi, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x55x55xf32>
    %dd0Wxt = stablehlo.transpose %dd0Wxi, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %dd0Wdt = stablehlo.transpose %dd0Wu, dims = [1, 0, 2, 3] : (tensor<32x192x55x55xf32>) -> tensor<192x32x55x55xf32>
    %dd0Wraw = stablehlo.convolution(%dd0Wxt, %dd0Wdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<192x32x55x55xf32>) -> tensor<96x192x2x2xf32>
    %dd0W = stablehlo.transpose %dd0Wraw, dims = [1, 0, 2, 3] : (tensor<96x192x2x2xf32>) -> tensor<192x96x2x2xf32>
    %v3030 = stablehlo.reshape %v3008 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3031 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3032 = stablehlo.multiply %v3030, %v3031 : tensor<32x96x56x56xf32>
    %v3033 = stablehlo.reshape %v3032 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3034 = stablehlo.reshape %v3033 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3035 = stablehlo.transpose %s0b2pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3036 = stablehlo.reverse %v3035, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3037 = stablehlo.convolution(%v3034, %v3036)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3038 = stablehlo.reshape %v3037 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3039 = stablehlo.multiply %v134, %v134 : tensor<32x1204224xf32>
    %v3040 = stablehlo.multiply %v3039, %v134 : tensor<32x1204224xf32>
    %v3041 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3042 = stablehlo.multiply %v3041, %v3040 : tensor<32x1204224xf32>
    %v3043 = stablehlo.add %v134, %v3042 : tensor<32x1204224xf32>
    %v3044 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3045 = stablehlo.multiply %v3044, %v3043 : tensor<32x1204224xf32>
    %v3046 = stablehlo.tanh %v3045 : tensor<32x1204224xf32>
    %v3047 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3048 = stablehlo.add %v3047, %v3046 : tensor<32x1204224xf32>
    %v3049 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3050 = stablehlo.multiply %v3049, %v3048 : tensor<32x1204224xf32>
    %v3051 = stablehlo.multiply %v3046, %v3046 : tensor<32x1204224xf32>
    %v3052 = stablehlo.subtract %v3047, %v3051 : tensor<32x1204224xf32>
    %v3053 = stablehlo.multiply %v3049, %v134 : tensor<32x1204224xf32>
    %v3054 = stablehlo.multiply %v3053, %v3052 : tensor<32x1204224xf32>
    %v3055 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3056 = stablehlo.multiply %v3055, %v3039 : tensor<32x1204224xf32>
    %v3057 = stablehlo.add %v3047, %v3056 : tensor<32x1204224xf32>
    %v3058 = stablehlo.multiply %v3044, %v3057 : tensor<32x1204224xf32>
    %v3059 = stablehlo.multiply %v3054, %v3058 : tensor<32x1204224xf32>
    %v3060 = stablehlo.add %v3050, %v3059 : tensor<32x1204224xf32>
    %v3061 = stablehlo.multiply %v3038, %v3060 : tensor<32x1204224xf32>
    %v3062 = stablehlo.reshape %v3061 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3063 = stablehlo.transpose %s0b2eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3064 = stablehlo.reverse %v3063, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3065 = stablehlo.convolution(%v3062, %v3064)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3066 = stablehlo.reshape %v3065 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3068 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3069 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3070 = stablehlo.reduce(%v111 init: %v3067) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3071 = stablehlo.broadcast_in_dim %v3070, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3072 = stablehlo.divide %v3071, %v3068 : tensor<32x301056xf32>
    %v3073 = stablehlo.subtract %v111, %v3072 : tensor<32x301056xf32>
    %v3074 = stablehlo.multiply %v3073, %v3073 : tensor<32x301056xf32>
    %v3075 = stablehlo.reduce(%v3074 init: %v3067) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3076 = stablehlo.broadcast_in_dim %v3075, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3077 = stablehlo.divide %v3076, %v3068 : tensor<32x301056xf32>
    %v3078 = stablehlo.add %v3077, %v3069 : tensor<32x301056xf32>
    %v3079 = stablehlo.rsqrt %v3078 : tensor<32x301056xf32>
    %v3080 = stablehlo.multiply %v3073, %v3079 : tensor<32x301056xf32>
    %v3081 = stablehlo.broadcast_in_dim %s0b2ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3082 = stablehlo.multiply %v3081, %v3066 : tensor<32x301056xf32>
    %v3083 = stablehlo.reduce(%v3082 init: %v3067) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3084 = stablehlo.broadcast_in_dim %v3083, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3085 = stablehlo.multiply %v3080, %v3082 : tensor<32x301056xf32>
    %v3086 = stablehlo.reduce(%v3085 init: %v3067) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3087 = stablehlo.broadcast_in_dim %v3086, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3088 = stablehlo.multiply %v3082, %v3068 : tensor<32x301056xf32>
    %v3089 = stablehlo.subtract %v3088, %v3084 : tensor<32x301056xf32>
    %v3090 = stablehlo.multiply %v3080, %v3087 : tensor<32x301056xf32>
    %v3091 = stablehlo.subtract %v3089, %v3090 : tensor<32x301056xf32>
    %v3092 = stablehlo.divide %v3079, %v3068 : tensor<32x301056xf32>
    %v3093 = stablehlo.multiply %v3092, %v3091 : tensor<32x301056xf32>
    %v3094 = stablehlo.reshape %v3093 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3095 = stablehlo.reverse %s0b2dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v3096 = stablehlo.convolution(%v3094, %v3095)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v3097 = stablehlo.reshape %v3096 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3098 = stablehlo.add %v3097, %v3008 : tensor<32x301056xf32>
    %v3099 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3100 = stablehlo.reshape %v152 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3101 = stablehlo.reshape %v3008 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3102 = stablehlo.multiply %v3100, %v3101 : tensor<32x96x56x56xf32>
    %v3103 = stablehlo.reduce(%v3102 init: %v3099) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3104 = stablehlo.reshape %v147 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3105 = stablehlo.reshape %v3033 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3106 = stablehlo.transpose %v3104, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3107 = stablehlo.transpose %v3105, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3108 = stablehlo.convolution(%v3106, %v3107)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v3109 = stablehlo.transpose %v3108, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3110 = stablehlo.reshape %v3033 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3111 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3112 = stablehlo.reduce(%v3110 init: %v3111) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3113 = stablehlo.reshape %v129 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3114 = stablehlo.reshape %v3061 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3115 = stablehlo.transpose %v3113, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3116 = stablehlo.transpose %v3114, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3117 = stablehlo.convolution(%v3115, %v3116)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v3118 = stablehlo.transpose %v3117, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3119 = stablehlo.reshape %v3061 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3121 = stablehlo.reduce(%v3119 init: %v3120) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v3122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3123 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3124 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3125 = stablehlo.reduce(%v111 init: %v3122) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3126 = stablehlo.broadcast_in_dim %v3125, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3127 = stablehlo.divide %v3126, %v3123 : tensor<32x301056xf32>
    %v3128 = stablehlo.subtract %v111, %v3127 : tensor<32x301056xf32>
    %v3129 = stablehlo.multiply %v3128, %v3128 : tensor<32x301056xf32>
    %v3130 = stablehlo.reduce(%v3129 init: %v3122) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3131 = stablehlo.broadcast_in_dim %v3130, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3132 = stablehlo.divide %v3131, %v3123 : tensor<32x301056xf32>
    %v3133 = stablehlo.add %v3132, %v3124 : tensor<32x301056xf32>
    %v3134 = stablehlo.rsqrt %v3133 : tensor<32x301056xf32>
    %v3135 = stablehlo.multiply %v3128, %v3134 : tensor<32x301056xf32>
    %v3136 = stablehlo.multiply %v3066, %v3135 : tensor<32x301056xf32>
    %v3137 = stablehlo.reduce(%v3136 init: %v3122) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3139 = stablehlo.reduce(%v3066 init: %v3138) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3140 = stablehlo.reshape %v106 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3141 = stablehlo.reshape %v3093 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3142 = stablehlo.transpose %v3140, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3143 = stablehlo.transpose %v3141, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3144 = stablehlo.convolution(%v3142, %v3143)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v3145 = stablehlo.reshape %v3144 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v3146 = stablehlo.reshape %v3093 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3148 = stablehlo.reduce(%v3146 init: %v3147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3149 = stablehlo.reshape %v3098 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3150 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3151 = stablehlo.multiply %v3149, %v3150 : tensor<32x96x56x56xf32>
    %v3152 = stablehlo.reshape %v3151 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3153 = stablehlo.reshape %v3152 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3154 = stablehlo.transpose %s0b1pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3155 = stablehlo.reverse %v3154, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3156 = stablehlo.convolution(%v3153, %v3155)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3157 = stablehlo.reshape %v3156 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3158 = stablehlo.multiply %v83, %v83 : tensor<32x1204224xf32>
    %v3159 = stablehlo.multiply %v3158, %v83 : tensor<32x1204224xf32>
    %v3160 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3161 = stablehlo.multiply %v3160, %v3159 : tensor<32x1204224xf32>
    %v3162 = stablehlo.add %v83, %v3161 : tensor<32x1204224xf32>
    %v3163 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3164 = stablehlo.multiply %v3163, %v3162 : tensor<32x1204224xf32>
    %v3165 = stablehlo.tanh %v3164 : tensor<32x1204224xf32>
    %v3166 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3167 = stablehlo.add %v3166, %v3165 : tensor<32x1204224xf32>
    %v3168 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3169 = stablehlo.multiply %v3168, %v3167 : tensor<32x1204224xf32>
    %v3170 = stablehlo.multiply %v3165, %v3165 : tensor<32x1204224xf32>
    %v3171 = stablehlo.subtract %v3166, %v3170 : tensor<32x1204224xf32>
    %v3172 = stablehlo.multiply %v3168, %v83 : tensor<32x1204224xf32>
    %v3173 = stablehlo.multiply %v3172, %v3171 : tensor<32x1204224xf32>
    %v3174 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3175 = stablehlo.multiply %v3174, %v3158 : tensor<32x1204224xf32>
    %v3176 = stablehlo.add %v3166, %v3175 : tensor<32x1204224xf32>
    %v3177 = stablehlo.multiply %v3163, %v3176 : tensor<32x1204224xf32>
    %v3178 = stablehlo.multiply %v3173, %v3177 : tensor<32x1204224xf32>
    %v3179 = stablehlo.add %v3169, %v3178 : tensor<32x1204224xf32>
    %v3180 = stablehlo.multiply %v3157, %v3179 : tensor<32x1204224xf32>
    %v3181 = stablehlo.reshape %v3180 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3182 = stablehlo.transpose %s0b1eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3183 = stablehlo.reverse %v3182, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3184 = stablehlo.convolution(%v3181, %v3183)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3185 = stablehlo.reshape %v3184 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3187 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3188 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3189 = stablehlo.reduce(%v60 init: %v3186) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3190 = stablehlo.broadcast_in_dim %v3189, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3191 = stablehlo.divide %v3190, %v3187 : tensor<32x301056xf32>
    %v3192 = stablehlo.subtract %v60, %v3191 : tensor<32x301056xf32>
    %v3193 = stablehlo.multiply %v3192, %v3192 : tensor<32x301056xf32>
    %v3194 = stablehlo.reduce(%v3193 init: %v3186) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3195 = stablehlo.broadcast_in_dim %v3194, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3196 = stablehlo.divide %v3195, %v3187 : tensor<32x301056xf32>
    %v3197 = stablehlo.add %v3196, %v3188 : tensor<32x301056xf32>
    %v3198 = stablehlo.rsqrt %v3197 : tensor<32x301056xf32>
    %v3199 = stablehlo.multiply %v3192, %v3198 : tensor<32x301056xf32>
    %v3200 = stablehlo.broadcast_in_dim %s0b1ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3201 = stablehlo.multiply %v3200, %v3185 : tensor<32x301056xf32>
    %v3202 = stablehlo.reduce(%v3201 init: %v3186) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3203 = stablehlo.broadcast_in_dim %v3202, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3204 = stablehlo.multiply %v3199, %v3201 : tensor<32x301056xf32>
    %v3205 = stablehlo.reduce(%v3204 init: %v3186) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3206 = stablehlo.broadcast_in_dim %v3205, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3207 = stablehlo.multiply %v3201, %v3187 : tensor<32x301056xf32>
    %v3208 = stablehlo.subtract %v3207, %v3203 : tensor<32x301056xf32>
    %v3209 = stablehlo.multiply %v3199, %v3206 : tensor<32x301056xf32>
    %v3210 = stablehlo.subtract %v3208, %v3209 : tensor<32x301056xf32>
    %v3211 = stablehlo.divide %v3198, %v3187 : tensor<32x301056xf32>
    %v3212 = stablehlo.multiply %v3211, %v3210 : tensor<32x301056xf32>
    %v3213 = stablehlo.reshape %v3212 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3214 = stablehlo.reverse %s0b1dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v3215 = stablehlo.convolution(%v3213, %v3214)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v3216 = stablehlo.reshape %v3215 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3217 = stablehlo.add %v3216, %v3098 : tensor<32x301056xf32>
    %v3218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3219 = stablehlo.reshape %v101 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3220 = stablehlo.reshape %v3098 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3221 = stablehlo.multiply %v3219, %v3220 : tensor<32x96x56x56xf32>
    %v3222 = stablehlo.reduce(%v3221 init: %v3218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3223 = stablehlo.reshape %v96 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3224 = stablehlo.reshape %v3152 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3225 = stablehlo.transpose %v3223, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3226 = stablehlo.transpose %v3224, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3227 = stablehlo.convolution(%v3225, %v3226)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v3228 = stablehlo.transpose %v3227, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3229 = stablehlo.reshape %v3152 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3231 = stablehlo.reduce(%v3229 init: %v3230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3232 = stablehlo.reshape %v78 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3233 = stablehlo.reshape %v3180 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3234 = stablehlo.transpose %v3232, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3235 = stablehlo.transpose %v3233, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3236 = stablehlo.convolution(%v3234, %v3235)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v3237 = stablehlo.transpose %v3236, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3238 = stablehlo.reshape %v3180 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3239 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3240 = stablehlo.reduce(%v3238 init: %v3239) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v3241 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3242 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3243 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3244 = stablehlo.reduce(%v60 init: %v3241) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3245 = stablehlo.broadcast_in_dim %v3244, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3246 = stablehlo.divide %v3245, %v3242 : tensor<32x301056xf32>
    %v3247 = stablehlo.subtract %v60, %v3246 : tensor<32x301056xf32>
    %v3248 = stablehlo.multiply %v3247, %v3247 : tensor<32x301056xf32>
    %v3249 = stablehlo.reduce(%v3248 init: %v3241) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3250 = stablehlo.broadcast_in_dim %v3249, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3251 = stablehlo.divide %v3250, %v3242 : tensor<32x301056xf32>
    %v3252 = stablehlo.add %v3251, %v3243 : tensor<32x301056xf32>
    %v3253 = stablehlo.rsqrt %v3252 : tensor<32x301056xf32>
    %v3254 = stablehlo.multiply %v3247, %v3253 : tensor<32x301056xf32>
    %v3255 = stablehlo.multiply %v3185, %v3254 : tensor<32x301056xf32>
    %v3256 = stablehlo.reduce(%v3255 init: %v3241) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3258 = stablehlo.reduce(%v3185 init: %v3257) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3259 = stablehlo.reshape %v55 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3260 = stablehlo.reshape %v3212 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3261 = stablehlo.transpose %v3259, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3262 = stablehlo.transpose %v3260, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3263 = stablehlo.convolution(%v3261, %v3262)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v3264 = stablehlo.reshape %v3263 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v3265 = stablehlo.reshape %v3212 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3267 = stablehlo.reduce(%v3265 init: %v3266) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3268 = stablehlo.reshape %v3217 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3269 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3270 = stablehlo.multiply %v3268, %v3269 : tensor<32x96x56x56xf32>
    %v3271 = stablehlo.reshape %v3270 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3272 = stablehlo.reshape %v3271 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3273 = stablehlo.transpose %s0b0pW, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3274 = stablehlo.reverse %v3273, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3275 = stablehlo.convolution(%v3272, %v3274)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3276 = stablehlo.reshape %v3275 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3277 = stablehlo.multiply %v32, %v32 : tensor<32x1204224xf32>
    %v3278 = stablehlo.multiply %v3277, %v32 : tensor<32x1204224xf32>
    %v3279 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3280 = stablehlo.multiply %v3279, %v3278 : tensor<32x1204224xf32>
    %v3281 = stablehlo.add %v32, %v3280 : tensor<32x1204224xf32>
    %v3282 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3283 = stablehlo.multiply %v3282, %v3281 : tensor<32x1204224xf32>
    %v3284 = stablehlo.tanh %v3283 : tensor<32x1204224xf32>
    %v3285 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3286 = stablehlo.add %v3285, %v3284 : tensor<32x1204224xf32>
    %v3287 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3288 = stablehlo.multiply %v3287, %v3286 : tensor<32x1204224xf32>
    %v3289 = stablehlo.multiply %v3284, %v3284 : tensor<32x1204224xf32>
    %v3290 = stablehlo.subtract %v3285, %v3289 : tensor<32x1204224xf32>
    %v3291 = stablehlo.multiply %v3287, %v32 : tensor<32x1204224xf32>
    %v3292 = stablehlo.multiply %v3291, %v3290 : tensor<32x1204224xf32>
    %v3293 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3294 = stablehlo.multiply %v3293, %v3277 : tensor<32x1204224xf32>
    %v3295 = stablehlo.add %v3285, %v3294 : tensor<32x1204224xf32>
    %v3296 = stablehlo.multiply %v3282, %v3295 : tensor<32x1204224xf32>
    %v3297 = stablehlo.multiply %v3292, %v3296 : tensor<32x1204224xf32>
    %v3298 = stablehlo.add %v3288, %v3297 : tensor<32x1204224xf32>
    %v3299 = stablehlo.multiply %v3276, %v3298 : tensor<32x1204224xf32>
    %v3300 = stablehlo.reshape %v3299 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3301 = stablehlo.transpose %s0b0eW, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3302 = stablehlo.reverse %v3301, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3303 = stablehlo.convolution(%v3300, %v3302)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3304 = stablehlo.reshape %v3303 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3305 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3306 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3307 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3308 = stablehlo.reduce(%v9 init: %v3305) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3309 = stablehlo.broadcast_in_dim %v3308, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3310 = stablehlo.divide %v3309, %v3306 : tensor<32x301056xf32>
    %v3311 = stablehlo.subtract %v9, %v3310 : tensor<32x301056xf32>
    %v3312 = stablehlo.multiply %v3311, %v3311 : tensor<32x301056xf32>
    %v3313 = stablehlo.reduce(%v3312 init: %v3305) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3314 = stablehlo.broadcast_in_dim %v3313, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3315 = stablehlo.divide %v3314, %v3306 : tensor<32x301056xf32>
    %v3316 = stablehlo.add %v3315, %v3307 : tensor<32x301056xf32>
    %v3317 = stablehlo.rsqrt %v3316 : tensor<32x301056xf32>
    %v3318 = stablehlo.multiply %v3311, %v3317 : tensor<32x301056xf32>
    %v3319 = stablehlo.broadcast_in_dim %s0b0ng, dims = [] : (tensor<f32>) -> tensor<32x301056xf32>
    %v3320 = stablehlo.multiply %v3319, %v3304 : tensor<32x301056xf32>
    %v3321 = stablehlo.reduce(%v3320 init: %v3305) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3322 = stablehlo.broadcast_in_dim %v3321, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3323 = stablehlo.multiply %v3318, %v3320 : tensor<32x301056xf32>
    %v3324 = stablehlo.reduce(%v3323 init: %v3305) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3325 = stablehlo.broadcast_in_dim %v3324, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3326 = stablehlo.multiply %v3320, %v3306 : tensor<32x301056xf32>
    %v3327 = stablehlo.subtract %v3326, %v3322 : tensor<32x301056xf32>
    %v3328 = stablehlo.multiply %v3318, %v3325 : tensor<32x301056xf32>
    %v3329 = stablehlo.subtract %v3327, %v3328 : tensor<32x301056xf32>
    %v3330 = stablehlo.divide %v3317, %v3306 : tensor<32x301056xf32>
    %v3331 = stablehlo.multiply %v3330, %v3329 : tensor<32x301056xf32>
    %v3332 = stablehlo.reshape %v3331 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3333 = stablehlo.reverse %s0b0dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v3334 = stablehlo.convolution(%v3332, %v3333)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v3335 = stablehlo.reshape %v3334 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3336 = stablehlo.add %v3335, %v3217 : tensor<32x301056xf32>
    %v3337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3338 = stablehlo.reshape %v50 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3339 = stablehlo.reshape %v3217 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3340 = stablehlo.multiply %v3338, %v3339 : tensor<32x96x56x56xf32>
    %v3341 = stablehlo.reduce(%v3340 init: %v3337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3342 = stablehlo.reshape %v45 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3343 = stablehlo.reshape %v3271 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3344 = stablehlo.transpose %v3342, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3345 = stablehlo.transpose %v3343, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3346 = stablehlo.convolution(%v3344, %v3345)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v3347 = stablehlo.transpose %v3346, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3348 = stablehlo.reshape %v3271 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3349 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3350 = stablehlo.reduce(%v3348 init: %v3349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v3351 = stablehlo.reshape %v27 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3352 = stablehlo.reshape %v3299 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3353 = stablehlo.transpose %v3351, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3354 = stablehlo.transpose %v3352, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v3355 = stablehlo.convolution(%v3353, %v3354)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v3356 = stablehlo.transpose %v3355, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3357 = stablehlo.reshape %v3299 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3359 = stablehlo.reduce(%v3357 init: %v3358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v3360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3361 = stablehlo.constant dense<301056.0> : tensor<32x301056xf32>
    %v3362 = stablehlo.constant dense<1.0e-6> : tensor<32x301056xf32>
    %v3363 = stablehlo.reduce(%v9 init: %v3360) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3364 = stablehlo.broadcast_in_dim %v3363, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3365 = stablehlo.divide %v3364, %v3361 : tensor<32x301056xf32>
    %v3366 = stablehlo.subtract %v9, %v3365 : tensor<32x301056xf32>
    %v3367 = stablehlo.multiply %v3366, %v3366 : tensor<32x301056xf32>
    %v3368 = stablehlo.reduce(%v3367 init: %v3360) applies stablehlo.add across dimensions = [1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<32xf32>
    %v3369 = stablehlo.broadcast_in_dim %v3368, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3370 = stablehlo.divide %v3369, %v3361 : tensor<32x301056xf32>
    %v3371 = stablehlo.add %v3370, %v3362 : tensor<32x301056xf32>
    %v3372 = stablehlo.rsqrt %v3371 : tensor<32x301056xf32>
    %v3373 = stablehlo.multiply %v3366, %v3372 : tensor<32x301056xf32>
    %v3374 = stablehlo.multiply %v3304, %v3373 : tensor<32x301056xf32>
    %v3375 = stablehlo.reduce(%v3374 init: %v3360) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3377 = stablehlo.reduce(%v3304 init: %v3376) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x301056xf32>, tensor<f32>) -> tensor<f32>
    %v3378 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3379 = stablehlo.reshape %v3331 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3380 = stablehlo.transpose %v3378, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3381 = stablehlo.transpose %v3379, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3382 = stablehlo.convolution(%v3380, %v3381)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v3383 = stablehlo.reshape %v3382 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v3384 = stablehlo.reshape %v3331 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3386 = stablehlo.reduce(%v3384 init: %v3385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %dpsWxi = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %dpsWdi = stablehlo.reshape %v3336 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %dpsWu = stablehlo.pad %dpsWdi, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x221x221xf32>
    %dpsWxt = stablehlo.transpose %dpsWxi, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %dpsWdt = stablehlo.transpose %dpsWu, dims = [1, 0, 2, 3] : (tensor<32x96x221x221xf32>) -> tensor<96x32x221x221xf32>
    %dpsWraw = stablehlo.convolution(%dpsWxt, %dpsWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<96x32x221x221xf32>) -> tensor<3x96x4x4xf32>
    %dpsW = stablehlo.transpose %dpsWraw, dims = [1, 0, 2, 3] : (tensor<3x96x4x4xf32>) -> tensor<96x3x4x4xf32>
    %v3387 = stablehlo.reshape %v3336 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3388 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3389 = stablehlo.reduce(%v3387 init: %v3388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v3390 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3391 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3392 = stablehlo.multiply %v3390, %psWm : tensor<96x3x4x4xf32>
    %v3393 = stablehlo.multiply %v3391, %dpsW : tensor<96x3x4x4xf32>
    %v3394 = stablehlo.add %v3392, %v3393 : tensor<96x3x4x4xf32>
    %v3395 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3396 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3397 = stablehlo.multiply %v3395, %psWv : tensor<96x3x4x4xf32>
    %v3398 = stablehlo.multiply %dpsW, %dpsW : tensor<96x3x4x4xf32>
    %v3399 = stablehlo.multiply %v3396, %v3398 : tensor<96x3x4x4xf32>
    %v3400 = stablehlo.add %v3397, %v3399 : tensor<96x3x4x4xf32>
    %v3401 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3402 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3403 = stablehlo.multiply %v3401, %psWm : tensor<96x3x4x4xf32>
    %v3404 = stablehlo.multiply %v3402, %dpsW : tensor<96x3x4x4xf32>
    %v3405 = stablehlo.add %v3403, %v3404 : tensor<96x3x4x4xf32>
    %v3406 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3407 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3408 = stablehlo.multiply %v3406, %psWv : tensor<96x3x4x4xf32>
    %v3409 = stablehlo.multiply %dpsW, %dpsW : tensor<96x3x4x4xf32>
    %v3410 = stablehlo.multiply %v3407, %v3409 : tensor<96x3x4x4xf32>
    %v3411 = stablehlo.add %v3408, %v3410 : tensor<96x3x4x4xf32>
    %v3412 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3413 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3414 = stablehlo.divide %v3405, %v3412 : tensor<96x3x4x4xf32>
    %v3415 = stablehlo.divide %v3411, %v3413 : tensor<96x3x4x4xf32>
    %v3416 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3417 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3418 = stablehlo.sqrt %v3415 : tensor<96x3x4x4xf32>
    %v3419 = stablehlo.add %v3418, %v3417 : tensor<96x3x4x4xf32>
    %v3420 = stablehlo.divide %v3414, %v3419 : tensor<96x3x4x4xf32>
    %v3421 = stablehlo.multiply %v3416, %v3420 : tensor<96x3x4x4xf32>
    %v3422 = stablehlo.subtract %psW, %v3421 : tensor<96x3x4x4xf32>
    %v3423 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v3424 = stablehlo.multiply %v3423, %v3416 : tensor<96x3x4x4xf32>
    %v3425 = stablehlo.multiply %v3424, %psW : tensor<96x3x4x4xf32>
    %v3426 = stablehlo.subtract %v3422, %v3425 : tensor<96x3x4x4xf32>
    %v3427 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3428 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3429 = stablehlo.multiply %v3427, %psbm : tensor<96xf32>
    %v3430 = stablehlo.multiply %v3428, %v3389 : tensor<96xf32>
    %v3431 = stablehlo.add %v3429, %v3430 : tensor<96xf32>
    %v3432 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3433 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3434 = stablehlo.multiply %v3432, %psbv : tensor<96xf32>
    %v3435 = stablehlo.multiply %v3389, %v3389 : tensor<96xf32>
    %v3436 = stablehlo.multiply %v3433, %v3435 : tensor<96xf32>
    %v3437 = stablehlo.add %v3434, %v3436 : tensor<96xf32>
    %v3438 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3439 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3440 = stablehlo.multiply %v3438, %psbm : tensor<96xf32>
    %v3441 = stablehlo.multiply %v3439, %v3389 : tensor<96xf32>
    %v3442 = stablehlo.add %v3440, %v3441 : tensor<96xf32>
    %v3443 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3444 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3445 = stablehlo.multiply %v3443, %psbv : tensor<96xf32>
    %v3446 = stablehlo.multiply %v3389, %v3389 : tensor<96xf32>
    %v3447 = stablehlo.multiply %v3444, %v3446 : tensor<96xf32>
    %v3448 = stablehlo.add %v3445, %v3447 : tensor<96xf32>
    %v3449 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3450 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3451 = stablehlo.divide %v3442, %v3449 : tensor<96xf32>
    %v3452 = stablehlo.divide %v3448, %v3450 : tensor<96xf32>
    %v3453 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3454 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3455 = stablehlo.sqrt %v3452 : tensor<96xf32>
    %v3456 = stablehlo.add %v3455, %v3454 : tensor<96xf32>
    %v3457 = stablehlo.divide %v3451, %v3456 : tensor<96xf32>
    %v3458 = stablehlo.multiply %v3453, %v3457 : tensor<96xf32>
    %v3459 = stablehlo.subtract %psb, %v3458 : tensor<96xf32>
    %v3460 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3461 = stablehlo.multiply %v3460, %v3453 : tensor<96xf32>
    %v3462 = stablehlo.multiply %v3461, %psb : tensor<96xf32>
    %v3463 = stablehlo.subtract %v3459, %v3462 : tensor<96xf32>
    %v3464 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3465 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3466 = stablehlo.multiply %v3464, %s0b0dWm : tensor<96x1x7x7xf32>
    %v3467 = stablehlo.multiply %v3465, %v3383 : tensor<96x1x7x7xf32>
    %v3468 = stablehlo.add %v3466, %v3467 : tensor<96x1x7x7xf32>
    %v3469 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3470 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3471 = stablehlo.multiply %v3469, %s0b0dWv : tensor<96x1x7x7xf32>
    %v3472 = stablehlo.multiply %v3383, %v3383 : tensor<96x1x7x7xf32>
    %v3473 = stablehlo.multiply %v3470, %v3472 : tensor<96x1x7x7xf32>
    %v3474 = stablehlo.add %v3471, %v3473 : tensor<96x1x7x7xf32>
    %v3475 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3476 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3477 = stablehlo.multiply %v3475, %s0b0dWm : tensor<96x1x7x7xf32>
    %v3478 = stablehlo.multiply %v3476, %v3383 : tensor<96x1x7x7xf32>
    %v3479 = stablehlo.add %v3477, %v3478 : tensor<96x1x7x7xf32>
    %v3480 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3481 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3482 = stablehlo.multiply %v3480, %s0b0dWv : tensor<96x1x7x7xf32>
    %v3483 = stablehlo.multiply %v3383, %v3383 : tensor<96x1x7x7xf32>
    %v3484 = stablehlo.multiply %v3481, %v3483 : tensor<96x1x7x7xf32>
    %v3485 = stablehlo.add %v3482, %v3484 : tensor<96x1x7x7xf32>
    %v3486 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3487 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3488 = stablehlo.divide %v3479, %v3486 : tensor<96x1x7x7xf32>
    %v3489 = stablehlo.divide %v3485, %v3487 : tensor<96x1x7x7xf32>
    %v3490 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3491 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3492 = stablehlo.sqrt %v3489 : tensor<96x1x7x7xf32>
    %v3493 = stablehlo.add %v3492, %v3491 : tensor<96x1x7x7xf32>
    %v3494 = stablehlo.divide %v3488, %v3493 : tensor<96x1x7x7xf32>
    %v3495 = stablehlo.multiply %v3490, %v3494 : tensor<96x1x7x7xf32>
    %v3496 = stablehlo.subtract %s0b0dW, %v3495 : tensor<96x1x7x7xf32>
    %v3497 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3498 = stablehlo.multiply %v3497, %v3490 : tensor<96x1x7x7xf32>
    %v3499 = stablehlo.multiply %v3498, %s0b0dW : tensor<96x1x7x7xf32>
    %v3500 = stablehlo.subtract %v3496, %v3499 : tensor<96x1x7x7xf32>
    %v3501 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3502 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3503 = stablehlo.multiply %v3501, %s0b0dbm : tensor<96xf32>
    %v3504 = stablehlo.multiply %v3502, %v3386 : tensor<96xf32>
    %v3505 = stablehlo.add %v3503, %v3504 : tensor<96xf32>
    %v3506 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3507 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3508 = stablehlo.multiply %v3506, %s0b0dbv : tensor<96xf32>
    %v3509 = stablehlo.multiply %v3386, %v3386 : tensor<96xf32>
    %v3510 = stablehlo.multiply %v3507, %v3509 : tensor<96xf32>
    %v3511 = stablehlo.add %v3508, %v3510 : tensor<96xf32>
    %v3512 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3513 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3514 = stablehlo.multiply %v3512, %s0b0dbm : tensor<96xf32>
    %v3515 = stablehlo.multiply %v3513, %v3386 : tensor<96xf32>
    %v3516 = stablehlo.add %v3514, %v3515 : tensor<96xf32>
    %v3517 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3518 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3519 = stablehlo.multiply %v3517, %s0b0dbv : tensor<96xf32>
    %v3520 = stablehlo.multiply %v3386, %v3386 : tensor<96xf32>
    %v3521 = stablehlo.multiply %v3518, %v3520 : tensor<96xf32>
    %v3522 = stablehlo.add %v3519, %v3521 : tensor<96xf32>
    %v3523 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3524 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3525 = stablehlo.divide %v3516, %v3523 : tensor<96xf32>
    %v3526 = stablehlo.divide %v3522, %v3524 : tensor<96xf32>
    %v3527 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3528 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3529 = stablehlo.sqrt %v3526 : tensor<96xf32>
    %v3530 = stablehlo.add %v3529, %v3528 : tensor<96xf32>
    %v3531 = stablehlo.divide %v3525, %v3530 : tensor<96xf32>
    %v3532 = stablehlo.multiply %v3527, %v3531 : tensor<96xf32>
    %v3533 = stablehlo.subtract %s0b0db, %v3532 : tensor<96xf32>
    %v3534 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3535 = stablehlo.multiply %v3534, %v3527 : tensor<96xf32>
    %v3536 = stablehlo.multiply %v3535, %s0b0db : tensor<96xf32>
    %v3537 = stablehlo.subtract %v3533, %v3536 : tensor<96xf32>
    %v3538 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3539 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3540 = stablehlo.multiply %v3538, %s0b0ngm : tensor<f32>
    %v3541 = stablehlo.multiply %v3539, %v3375 : tensor<f32>
    %v3542 = stablehlo.add %v3540, %v3541 : tensor<f32>
    %v3543 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3544 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3545 = stablehlo.multiply %v3543, %s0b0ngv : tensor<f32>
    %v3546 = stablehlo.multiply %v3375, %v3375 : tensor<f32>
    %v3547 = stablehlo.multiply %v3544, %v3546 : tensor<f32>
    %v3548 = stablehlo.add %v3545, %v3547 : tensor<f32>
    %v3549 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3550 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3551 = stablehlo.multiply %v3549, %s0b0ngm : tensor<f32>
    %v3552 = stablehlo.multiply %v3550, %v3375 : tensor<f32>
    %v3553 = stablehlo.add %v3551, %v3552 : tensor<f32>
    %v3554 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3555 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3556 = stablehlo.multiply %v3554, %s0b0ngv : tensor<f32>
    %v3557 = stablehlo.multiply %v3375, %v3375 : tensor<f32>
    %v3558 = stablehlo.multiply %v3555, %v3557 : tensor<f32>
    %v3559 = stablehlo.add %v3556, %v3558 : tensor<f32>
    %v3560 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3561 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3562 = stablehlo.divide %v3553, %v3560 : tensor<f32>
    %v3563 = stablehlo.divide %v3559, %v3561 : tensor<f32>
    %v3564 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3565 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3566 = stablehlo.sqrt %v3563 : tensor<f32>
    %v3567 = stablehlo.add %v3566, %v3565 : tensor<f32>
    %v3568 = stablehlo.divide %v3562, %v3567 : tensor<f32>
    %v3569 = stablehlo.multiply %v3564, %v3568 : tensor<f32>
    %v3570 = stablehlo.subtract %s0b0ng, %v3569 : tensor<f32>
    %v3571 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3572 = stablehlo.multiply %v3571, %v3564 : tensor<f32>
    %v3573 = stablehlo.multiply %v3572, %s0b0ng : tensor<f32>
    %v3574 = stablehlo.subtract %v3570, %v3573 : tensor<f32>
    %v3575 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3576 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3577 = stablehlo.multiply %v3575, %s0b0nbtm : tensor<f32>
    %v3578 = stablehlo.multiply %v3576, %v3377 : tensor<f32>
    %v3579 = stablehlo.add %v3577, %v3578 : tensor<f32>
    %v3580 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3581 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3582 = stablehlo.multiply %v3580, %s0b0nbtv : tensor<f32>
    %v3583 = stablehlo.multiply %v3377, %v3377 : tensor<f32>
    %v3584 = stablehlo.multiply %v3581, %v3583 : tensor<f32>
    %v3585 = stablehlo.add %v3582, %v3584 : tensor<f32>
    %v3586 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3587 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3588 = stablehlo.multiply %v3586, %s0b0nbtm : tensor<f32>
    %v3589 = stablehlo.multiply %v3587, %v3377 : tensor<f32>
    %v3590 = stablehlo.add %v3588, %v3589 : tensor<f32>
    %v3591 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3592 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3593 = stablehlo.multiply %v3591, %s0b0nbtv : tensor<f32>
    %v3594 = stablehlo.multiply %v3377, %v3377 : tensor<f32>
    %v3595 = stablehlo.multiply %v3592, %v3594 : tensor<f32>
    %v3596 = stablehlo.add %v3593, %v3595 : tensor<f32>
    %v3597 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3598 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3599 = stablehlo.divide %v3590, %v3597 : tensor<f32>
    %v3600 = stablehlo.divide %v3596, %v3598 : tensor<f32>
    %v3601 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3602 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3603 = stablehlo.sqrt %v3600 : tensor<f32>
    %v3604 = stablehlo.add %v3603, %v3602 : tensor<f32>
    %v3605 = stablehlo.divide %v3599, %v3604 : tensor<f32>
    %v3606 = stablehlo.multiply %v3601, %v3605 : tensor<f32>
    %v3607 = stablehlo.subtract %s0b0nbt, %v3606 : tensor<f32>
    %v3608 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3609 = stablehlo.multiply %v3608, %v3601 : tensor<f32>
    %v3610 = stablehlo.multiply %v3609, %s0b0nbt : tensor<f32>
    %v3611 = stablehlo.subtract %v3607, %v3610 : tensor<f32>
    %v3612 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3613 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3614 = stablehlo.multiply %v3612, %s0b0eWm : tensor<384x96x1x1xf32>
    %v3615 = stablehlo.multiply %v3613, %v3356 : tensor<384x96x1x1xf32>
    %v3616 = stablehlo.add %v3614, %v3615 : tensor<384x96x1x1xf32>
    %v3617 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3618 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3619 = stablehlo.multiply %v3617, %s0b0eWv : tensor<384x96x1x1xf32>
    %v3620 = stablehlo.multiply %v3356, %v3356 : tensor<384x96x1x1xf32>
    %v3621 = stablehlo.multiply %v3618, %v3620 : tensor<384x96x1x1xf32>
    %v3622 = stablehlo.add %v3619, %v3621 : tensor<384x96x1x1xf32>
    %v3623 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3624 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3625 = stablehlo.multiply %v3623, %s0b0eWm : tensor<384x96x1x1xf32>
    %v3626 = stablehlo.multiply %v3624, %v3356 : tensor<384x96x1x1xf32>
    %v3627 = stablehlo.add %v3625, %v3626 : tensor<384x96x1x1xf32>
    %v3628 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3629 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3630 = stablehlo.multiply %v3628, %s0b0eWv : tensor<384x96x1x1xf32>
    %v3631 = stablehlo.multiply %v3356, %v3356 : tensor<384x96x1x1xf32>
    %v3632 = stablehlo.multiply %v3629, %v3631 : tensor<384x96x1x1xf32>
    %v3633 = stablehlo.add %v3630, %v3632 : tensor<384x96x1x1xf32>
    %v3634 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3635 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3636 = stablehlo.divide %v3627, %v3634 : tensor<384x96x1x1xf32>
    %v3637 = stablehlo.divide %v3633, %v3635 : tensor<384x96x1x1xf32>
    %v3638 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3639 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3640 = stablehlo.sqrt %v3637 : tensor<384x96x1x1xf32>
    %v3641 = stablehlo.add %v3640, %v3639 : tensor<384x96x1x1xf32>
    %v3642 = stablehlo.divide %v3636, %v3641 : tensor<384x96x1x1xf32>
    %v3643 = stablehlo.multiply %v3638, %v3642 : tensor<384x96x1x1xf32>
    %v3644 = stablehlo.subtract %s0b0eW, %v3643 : tensor<384x96x1x1xf32>
    %v3645 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3646 = stablehlo.multiply %v3645, %v3638 : tensor<384x96x1x1xf32>
    %v3647 = stablehlo.multiply %v3646, %s0b0eW : tensor<384x96x1x1xf32>
    %v3648 = stablehlo.subtract %v3644, %v3647 : tensor<384x96x1x1xf32>
    %v3649 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3650 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3651 = stablehlo.multiply %v3649, %s0b0ebm : tensor<384xf32>
    %v3652 = stablehlo.multiply %v3650, %v3359 : tensor<384xf32>
    %v3653 = stablehlo.add %v3651, %v3652 : tensor<384xf32>
    %v3654 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3655 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3656 = stablehlo.multiply %v3654, %s0b0ebv : tensor<384xf32>
    %v3657 = stablehlo.multiply %v3359, %v3359 : tensor<384xf32>
    %v3658 = stablehlo.multiply %v3655, %v3657 : tensor<384xf32>
    %v3659 = stablehlo.add %v3656, %v3658 : tensor<384xf32>
    %v3660 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3661 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3662 = stablehlo.multiply %v3660, %s0b0ebm : tensor<384xf32>
    %v3663 = stablehlo.multiply %v3661, %v3359 : tensor<384xf32>
    %v3664 = stablehlo.add %v3662, %v3663 : tensor<384xf32>
    %v3665 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3666 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3667 = stablehlo.multiply %v3665, %s0b0ebv : tensor<384xf32>
    %v3668 = stablehlo.multiply %v3359, %v3359 : tensor<384xf32>
    %v3669 = stablehlo.multiply %v3666, %v3668 : tensor<384xf32>
    %v3670 = stablehlo.add %v3667, %v3669 : tensor<384xf32>
    %v3671 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3672 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3673 = stablehlo.divide %v3664, %v3671 : tensor<384xf32>
    %v3674 = stablehlo.divide %v3670, %v3672 : tensor<384xf32>
    %v3675 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3676 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3677 = stablehlo.sqrt %v3674 : tensor<384xf32>
    %v3678 = stablehlo.add %v3677, %v3676 : tensor<384xf32>
    %v3679 = stablehlo.divide %v3673, %v3678 : tensor<384xf32>
    %v3680 = stablehlo.multiply %v3675, %v3679 : tensor<384xf32>
    %v3681 = stablehlo.subtract %s0b0eb, %v3680 : tensor<384xf32>
    %v3682 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3683 = stablehlo.multiply %v3682, %v3675 : tensor<384xf32>
    %v3684 = stablehlo.multiply %v3683, %s0b0eb : tensor<384xf32>
    %v3685 = stablehlo.subtract %v3681, %v3684 : tensor<384xf32>
    %v3686 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3687 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3688 = stablehlo.multiply %v3686, %s0b0pWm : tensor<96x384x1x1xf32>
    %v3689 = stablehlo.multiply %v3687, %v3347 : tensor<96x384x1x1xf32>
    %v3690 = stablehlo.add %v3688, %v3689 : tensor<96x384x1x1xf32>
    %v3691 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3692 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3693 = stablehlo.multiply %v3691, %s0b0pWv : tensor<96x384x1x1xf32>
    %v3694 = stablehlo.multiply %v3347, %v3347 : tensor<96x384x1x1xf32>
    %v3695 = stablehlo.multiply %v3692, %v3694 : tensor<96x384x1x1xf32>
    %v3696 = stablehlo.add %v3693, %v3695 : tensor<96x384x1x1xf32>
    %v3697 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3698 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3699 = stablehlo.multiply %v3697, %s0b0pWm : tensor<96x384x1x1xf32>
    %v3700 = stablehlo.multiply %v3698, %v3347 : tensor<96x384x1x1xf32>
    %v3701 = stablehlo.add %v3699, %v3700 : tensor<96x384x1x1xf32>
    %v3702 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3703 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3704 = stablehlo.multiply %v3702, %s0b0pWv : tensor<96x384x1x1xf32>
    %v3705 = stablehlo.multiply %v3347, %v3347 : tensor<96x384x1x1xf32>
    %v3706 = stablehlo.multiply %v3703, %v3705 : tensor<96x384x1x1xf32>
    %v3707 = stablehlo.add %v3704, %v3706 : tensor<96x384x1x1xf32>
    %v3708 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3709 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3710 = stablehlo.divide %v3701, %v3708 : tensor<96x384x1x1xf32>
    %v3711 = stablehlo.divide %v3707, %v3709 : tensor<96x384x1x1xf32>
    %v3712 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3713 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3714 = stablehlo.sqrt %v3711 : tensor<96x384x1x1xf32>
    %v3715 = stablehlo.add %v3714, %v3713 : tensor<96x384x1x1xf32>
    %v3716 = stablehlo.divide %v3710, %v3715 : tensor<96x384x1x1xf32>
    %v3717 = stablehlo.multiply %v3712, %v3716 : tensor<96x384x1x1xf32>
    %v3718 = stablehlo.subtract %s0b0pW, %v3717 : tensor<96x384x1x1xf32>
    %v3719 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v3720 = stablehlo.multiply %v3719, %v3712 : tensor<96x384x1x1xf32>
    %v3721 = stablehlo.multiply %v3720, %s0b0pW : tensor<96x384x1x1xf32>
    %v3722 = stablehlo.subtract %v3718, %v3721 : tensor<96x384x1x1xf32>
    %v3723 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3724 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3725 = stablehlo.multiply %v3723, %s0b0pbm : tensor<96xf32>
    %v3726 = stablehlo.multiply %v3724, %v3350 : tensor<96xf32>
    %v3727 = stablehlo.add %v3725, %v3726 : tensor<96xf32>
    %v3728 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3729 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3730 = stablehlo.multiply %v3728, %s0b0pbv : tensor<96xf32>
    %v3731 = stablehlo.multiply %v3350, %v3350 : tensor<96xf32>
    %v3732 = stablehlo.multiply %v3729, %v3731 : tensor<96xf32>
    %v3733 = stablehlo.add %v3730, %v3732 : tensor<96xf32>
    %v3734 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3735 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3736 = stablehlo.multiply %v3734, %s0b0pbm : tensor<96xf32>
    %v3737 = stablehlo.multiply %v3735, %v3350 : tensor<96xf32>
    %v3738 = stablehlo.add %v3736, %v3737 : tensor<96xf32>
    %v3739 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3740 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3741 = stablehlo.multiply %v3739, %s0b0pbv : tensor<96xf32>
    %v3742 = stablehlo.multiply %v3350, %v3350 : tensor<96xf32>
    %v3743 = stablehlo.multiply %v3740, %v3742 : tensor<96xf32>
    %v3744 = stablehlo.add %v3741, %v3743 : tensor<96xf32>
    %v3745 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3746 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3747 = stablehlo.divide %v3738, %v3745 : tensor<96xf32>
    %v3748 = stablehlo.divide %v3744, %v3746 : tensor<96xf32>
    %v3749 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3750 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3751 = stablehlo.sqrt %v3748 : tensor<96xf32>
    %v3752 = stablehlo.add %v3751, %v3750 : tensor<96xf32>
    %v3753 = stablehlo.divide %v3747, %v3752 : tensor<96xf32>
    %v3754 = stablehlo.multiply %v3749, %v3753 : tensor<96xf32>
    %v3755 = stablehlo.subtract %s0b0pb, %v3754 : tensor<96xf32>
    %v3756 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3757 = stablehlo.multiply %v3756, %v3749 : tensor<96xf32>
    %v3758 = stablehlo.multiply %v3757, %s0b0pb : tensor<96xf32>
    %v3759 = stablehlo.subtract %v3755, %v3758 : tensor<96xf32>
    %v3760 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3761 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3762 = stablehlo.multiply %v3760, %s0b0lgm : tensor<96xf32>
    %v3763 = stablehlo.multiply %v3761, %v3341 : tensor<96xf32>
    %v3764 = stablehlo.add %v3762, %v3763 : tensor<96xf32>
    %v3765 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3766 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3767 = stablehlo.multiply %v3765, %s0b0lgv : tensor<96xf32>
    %v3768 = stablehlo.multiply %v3341, %v3341 : tensor<96xf32>
    %v3769 = stablehlo.multiply %v3766, %v3768 : tensor<96xf32>
    %v3770 = stablehlo.add %v3767, %v3769 : tensor<96xf32>
    %v3771 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3772 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3773 = stablehlo.multiply %v3771, %s0b0lgm : tensor<96xf32>
    %v3774 = stablehlo.multiply %v3772, %v3341 : tensor<96xf32>
    %v3775 = stablehlo.add %v3773, %v3774 : tensor<96xf32>
    %v3776 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3777 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3778 = stablehlo.multiply %v3776, %s0b0lgv : tensor<96xf32>
    %v3779 = stablehlo.multiply %v3341, %v3341 : tensor<96xf32>
    %v3780 = stablehlo.multiply %v3777, %v3779 : tensor<96xf32>
    %v3781 = stablehlo.add %v3778, %v3780 : tensor<96xf32>
    %v3782 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3783 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3784 = stablehlo.divide %v3775, %v3782 : tensor<96xf32>
    %v3785 = stablehlo.divide %v3781, %v3783 : tensor<96xf32>
    %v3786 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3787 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3788 = stablehlo.sqrt %v3785 : tensor<96xf32>
    %v3789 = stablehlo.add %v3788, %v3787 : tensor<96xf32>
    %v3790 = stablehlo.divide %v3784, %v3789 : tensor<96xf32>
    %v3791 = stablehlo.multiply %v3786, %v3790 : tensor<96xf32>
    %v3792 = stablehlo.subtract %s0b0lg, %v3791 : tensor<96xf32>
    %v3793 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3794 = stablehlo.multiply %v3793, %v3786 : tensor<96xf32>
    %v3795 = stablehlo.multiply %v3794, %s0b0lg : tensor<96xf32>
    %v3796 = stablehlo.subtract %v3792, %v3795 : tensor<96xf32>
    %v3797 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3798 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3799 = stablehlo.multiply %v3797, %s0b1dWm : tensor<96x1x7x7xf32>
    %v3800 = stablehlo.multiply %v3798, %v3264 : tensor<96x1x7x7xf32>
    %v3801 = stablehlo.add %v3799, %v3800 : tensor<96x1x7x7xf32>
    %v3802 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3803 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3804 = stablehlo.multiply %v3802, %s0b1dWv : tensor<96x1x7x7xf32>
    %v3805 = stablehlo.multiply %v3264, %v3264 : tensor<96x1x7x7xf32>
    %v3806 = stablehlo.multiply %v3803, %v3805 : tensor<96x1x7x7xf32>
    %v3807 = stablehlo.add %v3804, %v3806 : tensor<96x1x7x7xf32>
    %v3808 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3809 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3810 = stablehlo.multiply %v3808, %s0b1dWm : tensor<96x1x7x7xf32>
    %v3811 = stablehlo.multiply %v3809, %v3264 : tensor<96x1x7x7xf32>
    %v3812 = stablehlo.add %v3810, %v3811 : tensor<96x1x7x7xf32>
    %v3813 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3814 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3815 = stablehlo.multiply %v3813, %s0b1dWv : tensor<96x1x7x7xf32>
    %v3816 = stablehlo.multiply %v3264, %v3264 : tensor<96x1x7x7xf32>
    %v3817 = stablehlo.multiply %v3814, %v3816 : tensor<96x1x7x7xf32>
    %v3818 = stablehlo.add %v3815, %v3817 : tensor<96x1x7x7xf32>
    %v3819 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3820 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3821 = stablehlo.divide %v3812, %v3819 : tensor<96x1x7x7xf32>
    %v3822 = stablehlo.divide %v3818, %v3820 : tensor<96x1x7x7xf32>
    %v3823 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3824 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3825 = stablehlo.sqrt %v3822 : tensor<96x1x7x7xf32>
    %v3826 = stablehlo.add %v3825, %v3824 : tensor<96x1x7x7xf32>
    %v3827 = stablehlo.divide %v3821, %v3826 : tensor<96x1x7x7xf32>
    %v3828 = stablehlo.multiply %v3823, %v3827 : tensor<96x1x7x7xf32>
    %v3829 = stablehlo.subtract %s0b1dW, %v3828 : tensor<96x1x7x7xf32>
    %v3830 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v3831 = stablehlo.multiply %v3830, %v3823 : tensor<96x1x7x7xf32>
    %v3832 = stablehlo.multiply %v3831, %s0b1dW : tensor<96x1x7x7xf32>
    %v3833 = stablehlo.subtract %v3829, %v3832 : tensor<96x1x7x7xf32>
    %v3834 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3835 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3836 = stablehlo.multiply %v3834, %s0b1dbm : tensor<96xf32>
    %v3837 = stablehlo.multiply %v3835, %v3267 : tensor<96xf32>
    %v3838 = stablehlo.add %v3836, %v3837 : tensor<96xf32>
    %v3839 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3840 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3841 = stablehlo.multiply %v3839, %s0b1dbv : tensor<96xf32>
    %v3842 = stablehlo.multiply %v3267, %v3267 : tensor<96xf32>
    %v3843 = stablehlo.multiply %v3840, %v3842 : tensor<96xf32>
    %v3844 = stablehlo.add %v3841, %v3843 : tensor<96xf32>
    %v3845 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3846 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3847 = stablehlo.multiply %v3845, %s0b1dbm : tensor<96xf32>
    %v3848 = stablehlo.multiply %v3846, %v3267 : tensor<96xf32>
    %v3849 = stablehlo.add %v3847, %v3848 : tensor<96xf32>
    %v3850 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3851 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3852 = stablehlo.multiply %v3850, %s0b1dbv : tensor<96xf32>
    %v3853 = stablehlo.multiply %v3267, %v3267 : tensor<96xf32>
    %v3854 = stablehlo.multiply %v3851, %v3853 : tensor<96xf32>
    %v3855 = stablehlo.add %v3852, %v3854 : tensor<96xf32>
    %v3856 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3857 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3858 = stablehlo.divide %v3849, %v3856 : tensor<96xf32>
    %v3859 = stablehlo.divide %v3855, %v3857 : tensor<96xf32>
    %v3860 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3861 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3862 = stablehlo.sqrt %v3859 : tensor<96xf32>
    %v3863 = stablehlo.add %v3862, %v3861 : tensor<96xf32>
    %v3864 = stablehlo.divide %v3858, %v3863 : tensor<96xf32>
    %v3865 = stablehlo.multiply %v3860, %v3864 : tensor<96xf32>
    %v3866 = stablehlo.subtract %s0b1db, %v3865 : tensor<96xf32>
    %v3867 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v3868 = stablehlo.multiply %v3867, %v3860 : tensor<96xf32>
    %v3869 = stablehlo.multiply %v3868, %s0b1db : tensor<96xf32>
    %v3870 = stablehlo.subtract %v3866, %v3869 : tensor<96xf32>
    %v3871 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3872 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3873 = stablehlo.multiply %v3871, %s0b1ngm : tensor<f32>
    %v3874 = stablehlo.multiply %v3872, %v3256 : tensor<f32>
    %v3875 = stablehlo.add %v3873, %v3874 : tensor<f32>
    %v3876 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3877 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3878 = stablehlo.multiply %v3876, %s0b1ngv : tensor<f32>
    %v3879 = stablehlo.multiply %v3256, %v3256 : tensor<f32>
    %v3880 = stablehlo.multiply %v3877, %v3879 : tensor<f32>
    %v3881 = stablehlo.add %v3878, %v3880 : tensor<f32>
    %v3882 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3883 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3884 = stablehlo.multiply %v3882, %s0b1ngm : tensor<f32>
    %v3885 = stablehlo.multiply %v3883, %v3256 : tensor<f32>
    %v3886 = stablehlo.add %v3884, %v3885 : tensor<f32>
    %v3887 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3888 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3889 = stablehlo.multiply %v3887, %s0b1ngv : tensor<f32>
    %v3890 = stablehlo.multiply %v3256, %v3256 : tensor<f32>
    %v3891 = stablehlo.multiply %v3888, %v3890 : tensor<f32>
    %v3892 = stablehlo.add %v3889, %v3891 : tensor<f32>
    %v3893 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3894 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3895 = stablehlo.divide %v3886, %v3893 : tensor<f32>
    %v3896 = stablehlo.divide %v3892, %v3894 : tensor<f32>
    %v3897 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3898 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3899 = stablehlo.sqrt %v3896 : tensor<f32>
    %v3900 = stablehlo.add %v3899, %v3898 : tensor<f32>
    %v3901 = stablehlo.divide %v3895, %v3900 : tensor<f32>
    %v3902 = stablehlo.multiply %v3897, %v3901 : tensor<f32>
    %v3903 = stablehlo.subtract %s0b1ng, %v3902 : tensor<f32>
    %v3904 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3905 = stablehlo.multiply %v3904, %v3897 : tensor<f32>
    %v3906 = stablehlo.multiply %v3905, %s0b1ng : tensor<f32>
    %v3907 = stablehlo.subtract %v3903, %v3906 : tensor<f32>
    %v3908 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3909 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3910 = stablehlo.multiply %v3908, %s0b1nbtm : tensor<f32>
    %v3911 = stablehlo.multiply %v3909, %v3258 : tensor<f32>
    %v3912 = stablehlo.add %v3910, %v3911 : tensor<f32>
    %v3913 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3914 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3915 = stablehlo.multiply %v3913, %s0b1nbtv : tensor<f32>
    %v3916 = stablehlo.multiply %v3258, %v3258 : tensor<f32>
    %v3917 = stablehlo.multiply %v3914, %v3916 : tensor<f32>
    %v3918 = stablehlo.add %v3915, %v3917 : tensor<f32>
    %v3919 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3920 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3921 = stablehlo.multiply %v3919, %s0b1nbtm : tensor<f32>
    %v3922 = stablehlo.multiply %v3920, %v3258 : tensor<f32>
    %v3923 = stablehlo.add %v3921, %v3922 : tensor<f32>
    %v3924 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3925 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3926 = stablehlo.multiply %v3924, %s0b1nbtv : tensor<f32>
    %v3927 = stablehlo.multiply %v3258, %v3258 : tensor<f32>
    %v3928 = stablehlo.multiply %v3925, %v3927 : tensor<f32>
    %v3929 = stablehlo.add %v3926, %v3928 : tensor<f32>
    %v3930 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3931 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3932 = stablehlo.divide %v3923, %v3930 : tensor<f32>
    %v3933 = stablehlo.divide %v3929, %v3931 : tensor<f32>
    %v3934 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3935 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3936 = stablehlo.sqrt %v3933 : tensor<f32>
    %v3937 = stablehlo.add %v3936, %v3935 : tensor<f32>
    %v3938 = stablehlo.divide %v3932, %v3937 : tensor<f32>
    %v3939 = stablehlo.multiply %v3934, %v3938 : tensor<f32>
    %v3940 = stablehlo.subtract %s0b1nbt, %v3939 : tensor<f32>
    %v3941 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v3942 = stablehlo.multiply %v3941, %v3934 : tensor<f32>
    %v3943 = stablehlo.multiply %v3942, %s0b1nbt : tensor<f32>
    %v3944 = stablehlo.subtract %v3940, %v3943 : tensor<f32>
    %v3945 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3946 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3947 = stablehlo.multiply %v3945, %s0b1eWm : tensor<384x96x1x1xf32>
    %v3948 = stablehlo.multiply %v3946, %v3237 : tensor<384x96x1x1xf32>
    %v3949 = stablehlo.add %v3947, %v3948 : tensor<384x96x1x1xf32>
    %v3950 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3951 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3952 = stablehlo.multiply %v3950, %s0b1eWv : tensor<384x96x1x1xf32>
    %v3953 = stablehlo.multiply %v3237, %v3237 : tensor<384x96x1x1xf32>
    %v3954 = stablehlo.multiply %v3951, %v3953 : tensor<384x96x1x1xf32>
    %v3955 = stablehlo.add %v3952, %v3954 : tensor<384x96x1x1xf32>
    %v3956 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3957 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3958 = stablehlo.multiply %v3956, %s0b1eWm : tensor<384x96x1x1xf32>
    %v3959 = stablehlo.multiply %v3957, %v3237 : tensor<384x96x1x1xf32>
    %v3960 = stablehlo.add %v3958, %v3959 : tensor<384x96x1x1xf32>
    %v3961 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3962 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3963 = stablehlo.multiply %v3961, %s0b1eWv : tensor<384x96x1x1xf32>
    %v3964 = stablehlo.multiply %v3237, %v3237 : tensor<384x96x1x1xf32>
    %v3965 = stablehlo.multiply %v3962, %v3964 : tensor<384x96x1x1xf32>
    %v3966 = stablehlo.add %v3963, %v3965 : tensor<384x96x1x1xf32>
    %v3967 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3968 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3969 = stablehlo.divide %v3960, %v3967 : tensor<384x96x1x1xf32>
    %v3970 = stablehlo.divide %v3966, %v3968 : tensor<384x96x1x1xf32>
    %v3971 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3972 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3973 = stablehlo.sqrt %v3970 : tensor<384x96x1x1xf32>
    %v3974 = stablehlo.add %v3973, %v3972 : tensor<384x96x1x1xf32>
    %v3975 = stablehlo.divide %v3969, %v3974 : tensor<384x96x1x1xf32>
    %v3976 = stablehlo.multiply %v3971, %v3975 : tensor<384x96x1x1xf32>
    %v3977 = stablehlo.subtract %s0b1eW, %v3976 : tensor<384x96x1x1xf32>
    %v3978 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v3979 = stablehlo.multiply %v3978, %v3971 : tensor<384x96x1x1xf32>
    %v3980 = stablehlo.multiply %v3979, %s0b1eW : tensor<384x96x1x1xf32>
    %v3981 = stablehlo.subtract %v3977, %v3980 : tensor<384x96x1x1xf32>
    %v3982 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3983 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3984 = stablehlo.multiply %v3982, %s0b1ebm : tensor<384xf32>
    %v3985 = stablehlo.multiply %v3983, %v3240 : tensor<384xf32>
    %v3986 = stablehlo.add %v3984, %v3985 : tensor<384xf32>
    %v3987 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3988 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3989 = stablehlo.multiply %v3987, %s0b1ebv : tensor<384xf32>
    %v3990 = stablehlo.multiply %v3240, %v3240 : tensor<384xf32>
    %v3991 = stablehlo.multiply %v3988, %v3990 : tensor<384xf32>
    %v3992 = stablehlo.add %v3989, %v3991 : tensor<384xf32>
    %v3993 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3994 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3995 = stablehlo.multiply %v3993, %s0b1ebm : tensor<384xf32>
    %v3996 = stablehlo.multiply %v3994, %v3240 : tensor<384xf32>
    %v3997 = stablehlo.add %v3995, %v3996 : tensor<384xf32>
    %v3998 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v3999 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4000 = stablehlo.multiply %v3998, %s0b1ebv : tensor<384xf32>
    %v4001 = stablehlo.multiply %v3240, %v3240 : tensor<384xf32>
    %v4002 = stablehlo.multiply %v3999, %v4001 : tensor<384xf32>
    %v4003 = stablehlo.add %v4000, %v4002 : tensor<384xf32>
    %v4004 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4005 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4006 = stablehlo.divide %v3997, %v4004 : tensor<384xf32>
    %v4007 = stablehlo.divide %v4003, %v4005 : tensor<384xf32>
    %v4008 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4009 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4010 = stablehlo.sqrt %v4007 : tensor<384xf32>
    %v4011 = stablehlo.add %v4010, %v4009 : tensor<384xf32>
    %v4012 = stablehlo.divide %v4006, %v4011 : tensor<384xf32>
    %v4013 = stablehlo.multiply %v4008, %v4012 : tensor<384xf32>
    %v4014 = stablehlo.subtract %s0b1eb, %v4013 : tensor<384xf32>
    %v4015 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4016 = stablehlo.multiply %v4015, %v4008 : tensor<384xf32>
    %v4017 = stablehlo.multiply %v4016, %s0b1eb : tensor<384xf32>
    %v4018 = stablehlo.subtract %v4014, %v4017 : tensor<384xf32>
    %v4019 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4020 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4021 = stablehlo.multiply %v4019, %s0b1pWm : tensor<96x384x1x1xf32>
    %v4022 = stablehlo.multiply %v4020, %v3228 : tensor<96x384x1x1xf32>
    %v4023 = stablehlo.add %v4021, %v4022 : tensor<96x384x1x1xf32>
    %v4024 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4025 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4026 = stablehlo.multiply %v4024, %s0b1pWv : tensor<96x384x1x1xf32>
    %v4027 = stablehlo.multiply %v3228, %v3228 : tensor<96x384x1x1xf32>
    %v4028 = stablehlo.multiply %v4025, %v4027 : tensor<96x384x1x1xf32>
    %v4029 = stablehlo.add %v4026, %v4028 : tensor<96x384x1x1xf32>
    %v4030 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4031 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4032 = stablehlo.multiply %v4030, %s0b1pWm : tensor<96x384x1x1xf32>
    %v4033 = stablehlo.multiply %v4031, %v3228 : tensor<96x384x1x1xf32>
    %v4034 = stablehlo.add %v4032, %v4033 : tensor<96x384x1x1xf32>
    %v4035 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4036 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4037 = stablehlo.multiply %v4035, %s0b1pWv : tensor<96x384x1x1xf32>
    %v4038 = stablehlo.multiply %v3228, %v3228 : tensor<96x384x1x1xf32>
    %v4039 = stablehlo.multiply %v4036, %v4038 : tensor<96x384x1x1xf32>
    %v4040 = stablehlo.add %v4037, %v4039 : tensor<96x384x1x1xf32>
    %v4041 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4042 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4043 = stablehlo.divide %v4034, %v4041 : tensor<96x384x1x1xf32>
    %v4044 = stablehlo.divide %v4040, %v4042 : tensor<96x384x1x1xf32>
    %v4045 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4046 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4047 = stablehlo.sqrt %v4044 : tensor<96x384x1x1xf32>
    %v4048 = stablehlo.add %v4047, %v4046 : tensor<96x384x1x1xf32>
    %v4049 = stablehlo.divide %v4043, %v4048 : tensor<96x384x1x1xf32>
    %v4050 = stablehlo.multiply %v4045, %v4049 : tensor<96x384x1x1xf32>
    %v4051 = stablehlo.subtract %s0b1pW, %v4050 : tensor<96x384x1x1xf32>
    %v4052 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4053 = stablehlo.multiply %v4052, %v4045 : tensor<96x384x1x1xf32>
    %v4054 = stablehlo.multiply %v4053, %s0b1pW : tensor<96x384x1x1xf32>
    %v4055 = stablehlo.subtract %v4051, %v4054 : tensor<96x384x1x1xf32>
    %v4056 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4057 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4058 = stablehlo.multiply %v4056, %s0b1pbm : tensor<96xf32>
    %v4059 = stablehlo.multiply %v4057, %v3231 : tensor<96xf32>
    %v4060 = stablehlo.add %v4058, %v4059 : tensor<96xf32>
    %v4061 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4062 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4063 = stablehlo.multiply %v4061, %s0b1pbv : tensor<96xf32>
    %v4064 = stablehlo.multiply %v3231, %v3231 : tensor<96xf32>
    %v4065 = stablehlo.multiply %v4062, %v4064 : tensor<96xf32>
    %v4066 = stablehlo.add %v4063, %v4065 : tensor<96xf32>
    %v4067 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4068 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4069 = stablehlo.multiply %v4067, %s0b1pbm : tensor<96xf32>
    %v4070 = stablehlo.multiply %v4068, %v3231 : tensor<96xf32>
    %v4071 = stablehlo.add %v4069, %v4070 : tensor<96xf32>
    %v4072 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4073 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4074 = stablehlo.multiply %v4072, %s0b1pbv : tensor<96xf32>
    %v4075 = stablehlo.multiply %v3231, %v3231 : tensor<96xf32>
    %v4076 = stablehlo.multiply %v4073, %v4075 : tensor<96xf32>
    %v4077 = stablehlo.add %v4074, %v4076 : tensor<96xf32>
    %v4078 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4079 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4080 = stablehlo.divide %v4071, %v4078 : tensor<96xf32>
    %v4081 = stablehlo.divide %v4077, %v4079 : tensor<96xf32>
    %v4082 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4083 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4084 = stablehlo.sqrt %v4081 : tensor<96xf32>
    %v4085 = stablehlo.add %v4084, %v4083 : tensor<96xf32>
    %v4086 = stablehlo.divide %v4080, %v4085 : tensor<96xf32>
    %v4087 = stablehlo.multiply %v4082, %v4086 : tensor<96xf32>
    %v4088 = stablehlo.subtract %s0b1pb, %v4087 : tensor<96xf32>
    %v4089 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4090 = stablehlo.multiply %v4089, %v4082 : tensor<96xf32>
    %v4091 = stablehlo.multiply %v4090, %s0b1pb : tensor<96xf32>
    %v4092 = stablehlo.subtract %v4088, %v4091 : tensor<96xf32>
    %v4093 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4094 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4095 = stablehlo.multiply %v4093, %s0b1lgm : tensor<96xf32>
    %v4096 = stablehlo.multiply %v4094, %v3222 : tensor<96xf32>
    %v4097 = stablehlo.add %v4095, %v4096 : tensor<96xf32>
    %v4098 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4099 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4100 = stablehlo.multiply %v4098, %s0b1lgv : tensor<96xf32>
    %v4101 = stablehlo.multiply %v3222, %v3222 : tensor<96xf32>
    %v4102 = stablehlo.multiply %v4099, %v4101 : tensor<96xf32>
    %v4103 = stablehlo.add %v4100, %v4102 : tensor<96xf32>
    %v4104 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4105 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4106 = stablehlo.multiply %v4104, %s0b1lgm : tensor<96xf32>
    %v4107 = stablehlo.multiply %v4105, %v3222 : tensor<96xf32>
    %v4108 = stablehlo.add %v4106, %v4107 : tensor<96xf32>
    %v4109 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4110 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4111 = stablehlo.multiply %v4109, %s0b1lgv : tensor<96xf32>
    %v4112 = stablehlo.multiply %v3222, %v3222 : tensor<96xf32>
    %v4113 = stablehlo.multiply %v4110, %v4112 : tensor<96xf32>
    %v4114 = stablehlo.add %v4111, %v4113 : tensor<96xf32>
    %v4115 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4116 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4117 = stablehlo.divide %v4108, %v4115 : tensor<96xf32>
    %v4118 = stablehlo.divide %v4114, %v4116 : tensor<96xf32>
    %v4119 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4120 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4121 = stablehlo.sqrt %v4118 : tensor<96xf32>
    %v4122 = stablehlo.add %v4121, %v4120 : tensor<96xf32>
    %v4123 = stablehlo.divide %v4117, %v4122 : tensor<96xf32>
    %v4124 = stablehlo.multiply %v4119, %v4123 : tensor<96xf32>
    %v4125 = stablehlo.subtract %s0b1lg, %v4124 : tensor<96xf32>
    %v4126 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4127 = stablehlo.multiply %v4126, %v4119 : tensor<96xf32>
    %v4128 = stablehlo.multiply %v4127, %s0b1lg : tensor<96xf32>
    %v4129 = stablehlo.subtract %v4125, %v4128 : tensor<96xf32>
    %v4130 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4131 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4132 = stablehlo.multiply %v4130, %s0b2dWm : tensor<96x1x7x7xf32>
    %v4133 = stablehlo.multiply %v4131, %v3145 : tensor<96x1x7x7xf32>
    %v4134 = stablehlo.add %v4132, %v4133 : tensor<96x1x7x7xf32>
    %v4135 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4136 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4137 = stablehlo.multiply %v4135, %s0b2dWv : tensor<96x1x7x7xf32>
    %v4138 = stablehlo.multiply %v3145, %v3145 : tensor<96x1x7x7xf32>
    %v4139 = stablehlo.multiply %v4136, %v4138 : tensor<96x1x7x7xf32>
    %v4140 = stablehlo.add %v4137, %v4139 : tensor<96x1x7x7xf32>
    %v4141 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4142 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4143 = stablehlo.multiply %v4141, %s0b2dWm : tensor<96x1x7x7xf32>
    %v4144 = stablehlo.multiply %v4142, %v3145 : tensor<96x1x7x7xf32>
    %v4145 = stablehlo.add %v4143, %v4144 : tensor<96x1x7x7xf32>
    %v4146 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4147 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4148 = stablehlo.multiply %v4146, %s0b2dWv : tensor<96x1x7x7xf32>
    %v4149 = stablehlo.multiply %v3145, %v3145 : tensor<96x1x7x7xf32>
    %v4150 = stablehlo.multiply %v4147, %v4149 : tensor<96x1x7x7xf32>
    %v4151 = stablehlo.add %v4148, %v4150 : tensor<96x1x7x7xf32>
    %v4152 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4153 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4154 = stablehlo.divide %v4145, %v4152 : tensor<96x1x7x7xf32>
    %v4155 = stablehlo.divide %v4151, %v4153 : tensor<96x1x7x7xf32>
    %v4156 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4157 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4158 = stablehlo.sqrt %v4155 : tensor<96x1x7x7xf32>
    %v4159 = stablehlo.add %v4158, %v4157 : tensor<96x1x7x7xf32>
    %v4160 = stablehlo.divide %v4154, %v4159 : tensor<96x1x7x7xf32>
    %v4161 = stablehlo.multiply %v4156, %v4160 : tensor<96x1x7x7xf32>
    %v4162 = stablehlo.subtract %s0b2dW, %v4161 : tensor<96x1x7x7xf32>
    %v4163 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4164 = stablehlo.multiply %v4163, %v4156 : tensor<96x1x7x7xf32>
    %v4165 = stablehlo.multiply %v4164, %s0b2dW : tensor<96x1x7x7xf32>
    %v4166 = stablehlo.subtract %v4162, %v4165 : tensor<96x1x7x7xf32>
    %v4167 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4168 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4169 = stablehlo.multiply %v4167, %s0b2dbm : tensor<96xf32>
    %v4170 = stablehlo.multiply %v4168, %v3148 : tensor<96xf32>
    %v4171 = stablehlo.add %v4169, %v4170 : tensor<96xf32>
    %v4172 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4173 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4174 = stablehlo.multiply %v4172, %s0b2dbv : tensor<96xf32>
    %v4175 = stablehlo.multiply %v3148, %v3148 : tensor<96xf32>
    %v4176 = stablehlo.multiply %v4173, %v4175 : tensor<96xf32>
    %v4177 = stablehlo.add %v4174, %v4176 : tensor<96xf32>
    %v4178 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4179 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4180 = stablehlo.multiply %v4178, %s0b2dbm : tensor<96xf32>
    %v4181 = stablehlo.multiply %v4179, %v3148 : tensor<96xf32>
    %v4182 = stablehlo.add %v4180, %v4181 : tensor<96xf32>
    %v4183 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4184 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4185 = stablehlo.multiply %v4183, %s0b2dbv : tensor<96xf32>
    %v4186 = stablehlo.multiply %v3148, %v3148 : tensor<96xf32>
    %v4187 = stablehlo.multiply %v4184, %v4186 : tensor<96xf32>
    %v4188 = stablehlo.add %v4185, %v4187 : tensor<96xf32>
    %v4189 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4190 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4191 = stablehlo.divide %v4182, %v4189 : tensor<96xf32>
    %v4192 = stablehlo.divide %v4188, %v4190 : tensor<96xf32>
    %v4193 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4194 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4195 = stablehlo.sqrt %v4192 : tensor<96xf32>
    %v4196 = stablehlo.add %v4195, %v4194 : tensor<96xf32>
    %v4197 = stablehlo.divide %v4191, %v4196 : tensor<96xf32>
    %v4198 = stablehlo.multiply %v4193, %v4197 : tensor<96xf32>
    %v4199 = stablehlo.subtract %s0b2db, %v4198 : tensor<96xf32>
    %v4200 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4201 = stablehlo.multiply %v4200, %v4193 : tensor<96xf32>
    %v4202 = stablehlo.multiply %v4201, %s0b2db : tensor<96xf32>
    %v4203 = stablehlo.subtract %v4199, %v4202 : tensor<96xf32>
    %v4204 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4205 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4206 = stablehlo.multiply %v4204, %s0b2ngm : tensor<f32>
    %v4207 = stablehlo.multiply %v4205, %v3137 : tensor<f32>
    %v4208 = stablehlo.add %v4206, %v4207 : tensor<f32>
    %v4209 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4210 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4211 = stablehlo.multiply %v4209, %s0b2ngv : tensor<f32>
    %v4212 = stablehlo.multiply %v3137, %v3137 : tensor<f32>
    %v4213 = stablehlo.multiply %v4210, %v4212 : tensor<f32>
    %v4214 = stablehlo.add %v4211, %v4213 : tensor<f32>
    %v4215 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4216 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4217 = stablehlo.multiply %v4215, %s0b2ngm : tensor<f32>
    %v4218 = stablehlo.multiply %v4216, %v3137 : tensor<f32>
    %v4219 = stablehlo.add %v4217, %v4218 : tensor<f32>
    %v4220 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4221 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4222 = stablehlo.multiply %v4220, %s0b2ngv : tensor<f32>
    %v4223 = stablehlo.multiply %v3137, %v3137 : tensor<f32>
    %v4224 = stablehlo.multiply %v4221, %v4223 : tensor<f32>
    %v4225 = stablehlo.add %v4222, %v4224 : tensor<f32>
    %v4226 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4227 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4228 = stablehlo.divide %v4219, %v4226 : tensor<f32>
    %v4229 = stablehlo.divide %v4225, %v4227 : tensor<f32>
    %v4230 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4231 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4232 = stablehlo.sqrt %v4229 : tensor<f32>
    %v4233 = stablehlo.add %v4232, %v4231 : tensor<f32>
    %v4234 = stablehlo.divide %v4228, %v4233 : tensor<f32>
    %v4235 = stablehlo.multiply %v4230, %v4234 : tensor<f32>
    %v4236 = stablehlo.subtract %s0b2ng, %v4235 : tensor<f32>
    %v4237 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4238 = stablehlo.multiply %v4237, %v4230 : tensor<f32>
    %v4239 = stablehlo.multiply %v4238, %s0b2ng : tensor<f32>
    %v4240 = stablehlo.subtract %v4236, %v4239 : tensor<f32>
    %v4241 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4242 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4243 = stablehlo.multiply %v4241, %s0b2nbtm : tensor<f32>
    %v4244 = stablehlo.multiply %v4242, %v3139 : tensor<f32>
    %v4245 = stablehlo.add %v4243, %v4244 : tensor<f32>
    %v4246 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4247 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4248 = stablehlo.multiply %v4246, %s0b2nbtv : tensor<f32>
    %v4249 = stablehlo.multiply %v3139, %v3139 : tensor<f32>
    %v4250 = stablehlo.multiply %v4247, %v4249 : tensor<f32>
    %v4251 = stablehlo.add %v4248, %v4250 : tensor<f32>
    %v4252 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4253 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4254 = stablehlo.multiply %v4252, %s0b2nbtm : tensor<f32>
    %v4255 = stablehlo.multiply %v4253, %v3139 : tensor<f32>
    %v4256 = stablehlo.add %v4254, %v4255 : tensor<f32>
    %v4257 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4258 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4259 = stablehlo.multiply %v4257, %s0b2nbtv : tensor<f32>
    %v4260 = stablehlo.multiply %v3139, %v3139 : tensor<f32>
    %v4261 = stablehlo.multiply %v4258, %v4260 : tensor<f32>
    %v4262 = stablehlo.add %v4259, %v4261 : tensor<f32>
    %v4263 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4264 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4265 = stablehlo.divide %v4256, %v4263 : tensor<f32>
    %v4266 = stablehlo.divide %v4262, %v4264 : tensor<f32>
    %v4267 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4268 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4269 = stablehlo.sqrt %v4266 : tensor<f32>
    %v4270 = stablehlo.add %v4269, %v4268 : tensor<f32>
    %v4271 = stablehlo.divide %v4265, %v4270 : tensor<f32>
    %v4272 = stablehlo.multiply %v4267, %v4271 : tensor<f32>
    %v4273 = stablehlo.subtract %s0b2nbt, %v4272 : tensor<f32>
    %v4274 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4275 = stablehlo.multiply %v4274, %v4267 : tensor<f32>
    %v4276 = stablehlo.multiply %v4275, %s0b2nbt : tensor<f32>
    %v4277 = stablehlo.subtract %v4273, %v4276 : tensor<f32>
    %v4278 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4279 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4280 = stablehlo.multiply %v4278, %s0b2eWm : tensor<384x96x1x1xf32>
    %v4281 = stablehlo.multiply %v4279, %v3118 : tensor<384x96x1x1xf32>
    %v4282 = stablehlo.add %v4280, %v4281 : tensor<384x96x1x1xf32>
    %v4283 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4284 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4285 = stablehlo.multiply %v4283, %s0b2eWv : tensor<384x96x1x1xf32>
    %v4286 = stablehlo.multiply %v3118, %v3118 : tensor<384x96x1x1xf32>
    %v4287 = stablehlo.multiply %v4284, %v4286 : tensor<384x96x1x1xf32>
    %v4288 = stablehlo.add %v4285, %v4287 : tensor<384x96x1x1xf32>
    %v4289 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4290 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4291 = stablehlo.multiply %v4289, %s0b2eWm : tensor<384x96x1x1xf32>
    %v4292 = stablehlo.multiply %v4290, %v3118 : tensor<384x96x1x1xf32>
    %v4293 = stablehlo.add %v4291, %v4292 : tensor<384x96x1x1xf32>
    %v4294 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4295 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4296 = stablehlo.multiply %v4294, %s0b2eWv : tensor<384x96x1x1xf32>
    %v4297 = stablehlo.multiply %v3118, %v3118 : tensor<384x96x1x1xf32>
    %v4298 = stablehlo.multiply %v4295, %v4297 : tensor<384x96x1x1xf32>
    %v4299 = stablehlo.add %v4296, %v4298 : tensor<384x96x1x1xf32>
    %v4300 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4301 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4302 = stablehlo.divide %v4293, %v4300 : tensor<384x96x1x1xf32>
    %v4303 = stablehlo.divide %v4299, %v4301 : tensor<384x96x1x1xf32>
    %v4304 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4305 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4306 = stablehlo.sqrt %v4303 : tensor<384x96x1x1xf32>
    %v4307 = stablehlo.add %v4306, %v4305 : tensor<384x96x1x1xf32>
    %v4308 = stablehlo.divide %v4302, %v4307 : tensor<384x96x1x1xf32>
    %v4309 = stablehlo.multiply %v4304, %v4308 : tensor<384x96x1x1xf32>
    %v4310 = stablehlo.subtract %s0b2eW, %v4309 : tensor<384x96x1x1xf32>
    %v4311 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4312 = stablehlo.multiply %v4311, %v4304 : tensor<384x96x1x1xf32>
    %v4313 = stablehlo.multiply %v4312, %s0b2eW : tensor<384x96x1x1xf32>
    %v4314 = stablehlo.subtract %v4310, %v4313 : tensor<384x96x1x1xf32>
    %v4315 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4316 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4317 = stablehlo.multiply %v4315, %s0b2ebm : tensor<384xf32>
    %v4318 = stablehlo.multiply %v4316, %v3121 : tensor<384xf32>
    %v4319 = stablehlo.add %v4317, %v4318 : tensor<384xf32>
    %v4320 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4321 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4322 = stablehlo.multiply %v4320, %s0b2ebv : tensor<384xf32>
    %v4323 = stablehlo.multiply %v3121, %v3121 : tensor<384xf32>
    %v4324 = stablehlo.multiply %v4321, %v4323 : tensor<384xf32>
    %v4325 = stablehlo.add %v4322, %v4324 : tensor<384xf32>
    %v4326 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4327 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4328 = stablehlo.multiply %v4326, %s0b2ebm : tensor<384xf32>
    %v4329 = stablehlo.multiply %v4327, %v3121 : tensor<384xf32>
    %v4330 = stablehlo.add %v4328, %v4329 : tensor<384xf32>
    %v4331 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4332 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4333 = stablehlo.multiply %v4331, %s0b2ebv : tensor<384xf32>
    %v4334 = stablehlo.multiply %v3121, %v3121 : tensor<384xf32>
    %v4335 = stablehlo.multiply %v4332, %v4334 : tensor<384xf32>
    %v4336 = stablehlo.add %v4333, %v4335 : tensor<384xf32>
    %v4337 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4338 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4339 = stablehlo.divide %v4330, %v4337 : tensor<384xf32>
    %v4340 = stablehlo.divide %v4336, %v4338 : tensor<384xf32>
    %v4341 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4342 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4343 = stablehlo.sqrt %v4340 : tensor<384xf32>
    %v4344 = stablehlo.add %v4343, %v4342 : tensor<384xf32>
    %v4345 = stablehlo.divide %v4339, %v4344 : tensor<384xf32>
    %v4346 = stablehlo.multiply %v4341, %v4345 : tensor<384xf32>
    %v4347 = stablehlo.subtract %s0b2eb, %v4346 : tensor<384xf32>
    %v4348 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4349 = stablehlo.multiply %v4348, %v4341 : tensor<384xf32>
    %v4350 = stablehlo.multiply %v4349, %s0b2eb : tensor<384xf32>
    %v4351 = stablehlo.subtract %v4347, %v4350 : tensor<384xf32>
    %v4352 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4353 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4354 = stablehlo.multiply %v4352, %s0b2pWm : tensor<96x384x1x1xf32>
    %v4355 = stablehlo.multiply %v4353, %v3109 : tensor<96x384x1x1xf32>
    %v4356 = stablehlo.add %v4354, %v4355 : tensor<96x384x1x1xf32>
    %v4357 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4358 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4359 = stablehlo.multiply %v4357, %s0b2pWv : tensor<96x384x1x1xf32>
    %v4360 = stablehlo.multiply %v3109, %v3109 : tensor<96x384x1x1xf32>
    %v4361 = stablehlo.multiply %v4358, %v4360 : tensor<96x384x1x1xf32>
    %v4362 = stablehlo.add %v4359, %v4361 : tensor<96x384x1x1xf32>
    %v4363 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4364 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4365 = stablehlo.multiply %v4363, %s0b2pWm : tensor<96x384x1x1xf32>
    %v4366 = stablehlo.multiply %v4364, %v3109 : tensor<96x384x1x1xf32>
    %v4367 = stablehlo.add %v4365, %v4366 : tensor<96x384x1x1xf32>
    %v4368 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4369 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4370 = stablehlo.multiply %v4368, %s0b2pWv : tensor<96x384x1x1xf32>
    %v4371 = stablehlo.multiply %v3109, %v3109 : tensor<96x384x1x1xf32>
    %v4372 = stablehlo.multiply %v4369, %v4371 : tensor<96x384x1x1xf32>
    %v4373 = stablehlo.add %v4370, %v4372 : tensor<96x384x1x1xf32>
    %v4374 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4375 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4376 = stablehlo.divide %v4367, %v4374 : tensor<96x384x1x1xf32>
    %v4377 = stablehlo.divide %v4373, %v4375 : tensor<96x384x1x1xf32>
    %v4378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4379 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4380 = stablehlo.sqrt %v4377 : tensor<96x384x1x1xf32>
    %v4381 = stablehlo.add %v4380, %v4379 : tensor<96x384x1x1xf32>
    %v4382 = stablehlo.divide %v4376, %v4381 : tensor<96x384x1x1xf32>
    %v4383 = stablehlo.multiply %v4378, %v4382 : tensor<96x384x1x1xf32>
    %v4384 = stablehlo.subtract %s0b2pW, %v4383 : tensor<96x384x1x1xf32>
    %v4385 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4386 = stablehlo.multiply %v4385, %v4378 : tensor<96x384x1x1xf32>
    %v4387 = stablehlo.multiply %v4386, %s0b2pW : tensor<96x384x1x1xf32>
    %v4388 = stablehlo.subtract %v4384, %v4387 : tensor<96x384x1x1xf32>
    %v4389 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4390 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4391 = stablehlo.multiply %v4389, %s0b2pbm : tensor<96xf32>
    %v4392 = stablehlo.multiply %v4390, %v3112 : tensor<96xf32>
    %v4393 = stablehlo.add %v4391, %v4392 : tensor<96xf32>
    %v4394 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4395 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4396 = stablehlo.multiply %v4394, %s0b2pbv : tensor<96xf32>
    %v4397 = stablehlo.multiply %v3112, %v3112 : tensor<96xf32>
    %v4398 = stablehlo.multiply %v4395, %v4397 : tensor<96xf32>
    %v4399 = stablehlo.add %v4396, %v4398 : tensor<96xf32>
    %v4400 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4401 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4402 = stablehlo.multiply %v4400, %s0b2pbm : tensor<96xf32>
    %v4403 = stablehlo.multiply %v4401, %v3112 : tensor<96xf32>
    %v4404 = stablehlo.add %v4402, %v4403 : tensor<96xf32>
    %v4405 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4406 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4407 = stablehlo.multiply %v4405, %s0b2pbv : tensor<96xf32>
    %v4408 = stablehlo.multiply %v3112, %v3112 : tensor<96xf32>
    %v4409 = stablehlo.multiply %v4406, %v4408 : tensor<96xf32>
    %v4410 = stablehlo.add %v4407, %v4409 : tensor<96xf32>
    %v4411 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4412 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4413 = stablehlo.divide %v4404, %v4411 : tensor<96xf32>
    %v4414 = stablehlo.divide %v4410, %v4412 : tensor<96xf32>
    %v4415 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4416 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4417 = stablehlo.sqrt %v4414 : tensor<96xf32>
    %v4418 = stablehlo.add %v4417, %v4416 : tensor<96xf32>
    %v4419 = stablehlo.divide %v4413, %v4418 : tensor<96xf32>
    %v4420 = stablehlo.multiply %v4415, %v4419 : tensor<96xf32>
    %v4421 = stablehlo.subtract %s0b2pb, %v4420 : tensor<96xf32>
    %v4422 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4423 = stablehlo.multiply %v4422, %v4415 : tensor<96xf32>
    %v4424 = stablehlo.multiply %v4423, %s0b2pb : tensor<96xf32>
    %v4425 = stablehlo.subtract %v4421, %v4424 : tensor<96xf32>
    %v4426 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4427 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4428 = stablehlo.multiply %v4426, %s0b2lgm : tensor<96xf32>
    %v4429 = stablehlo.multiply %v4427, %v3103 : tensor<96xf32>
    %v4430 = stablehlo.add %v4428, %v4429 : tensor<96xf32>
    %v4431 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4432 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4433 = stablehlo.multiply %v4431, %s0b2lgv : tensor<96xf32>
    %v4434 = stablehlo.multiply %v3103, %v3103 : tensor<96xf32>
    %v4435 = stablehlo.multiply %v4432, %v4434 : tensor<96xf32>
    %v4436 = stablehlo.add %v4433, %v4435 : tensor<96xf32>
    %v4437 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4438 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4439 = stablehlo.multiply %v4437, %s0b2lgm : tensor<96xf32>
    %v4440 = stablehlo.multiply %v4438, %v3103 : tensor<96xf32>
    %v4441 = stablehlo.add %v4439, %v4440 : tensor<96xf32>
    %v4442 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4443 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4444 = stablehlo.multiply %v4442, %s0b2lgv : tensor<96xf32>
    %v4445 = stablehlo.multiply %v3103, %v3103 : tensor<96xf32>
    %v4446 = stablehlo.multiply %v4443, %v4445 : tensor<96xf32>
    %v4447 = stablehlo.add %v4444, %v4446 : tensor<96xf32>
    %v4448 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4449 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4450 = stablehlo.divide %v4441, %v4448 : tensor<96xf32>
    %v4451 = stablehlo.divide %v4447, %v4449 : tensor<96xf32>
    %v4452 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4453 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4454 = stablehlo.sqrt %v4451 : tensor<96xf32>
    %v4455 = stablehlo.add %v4454, %v4453 : tensor<96xf32>
    %v4456 = stablehlo.divide %v4450, %v4455 : tensor<96xf32>
    %v4457 = stablehlo.multiply %v4452, %v4456 : tensor<96xf32>
    %v4458 = stablehlo.subtract %s0b2lg, %v4457 : tensor<96xf32>
    %v4459 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4460 = stablehlo.multiply %v4459, %v4452 : tensor<96xf32>
    %v4461 = stablehlo.multiply %v4460, %s0b2lg : tensor<96xf32>
    %v4462 = stablehlo.subtract %v4458, %v4461 : tensor<96xf32>
    %v4463 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4464 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4465 = stablehlo.multiply %v4463, %d0ngm : tensor<f32>
    %v4466 = stablehlo.multiply %v4464, %v3027 : tensor<f32>
    %v4467 = stablehlo.add %v4465, %v4466 : tensor<f32>
    %v4468 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4469 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4470 = stablehlo.multiply %v4468, %d0ngv : tensor<f32>
    %v4471 = stablehlo.multiply %v3027, %v3027 : tensor<f32>
    %v4472 = stablehlo.multiply %v4469, %v4471 : tensor<f32>
    %v4473 = stablehlo.add %v4470, %v4472 : tensor<f32>
    %v4474 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4475 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4476 = stablehlo.multiply %v4474, %d0ngm : tensor<f32>
    %v4477 = stablehlo.multiply %v4475, %v3027 : tensor<f32>
    %v4478 = stablehlo.add %v4476, %v4477 : tensor<f32>
    %v4479 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4480 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4481 = stablehlo.multiply %v4479, %d0ngv : tensor<f32>
    %v4482 = stablehlo.multiply %v3027, %v3027 : tensor<f32>
    %v4483 = stablehlo.multiply %v4480, %v4482 : tensor<f32>
    %v4484 = stablehlo.add %v4481, %v4483 : tensor<f32>
    %v4485 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4486 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4487 = stablehlo.divide %v4478, %v4485 : tensor<f32>
    %v4488 = stablehlo.divide %v4484, %v4486 : tensor<f32>
    %v4489 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4490 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4491 = stablehlo.sqrt %v4488 : tensor<f32>
    %v4492 = stablehlo.add %v4491, %v4490 : tensor<f32>
    %v4493 = stablehlo.divide %v4487, %v4492 : tensor<f32>
    %v4494 = stablehlo.multiply %v4489, %v4493 : tensor<f32>
    %v4495 = stablehlo.subtract %d0ng, %v4494 : tensor<f32>
    %v4496 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4497 = stablehlo.multiply %v4496, %v4489 : tensor<f32>
    %v4498 = stablehlo.multiply %v4497, %d0ng : tensor<f32>
    %v4499 = stablehlo.subtract %v4495, %v4498 : tensor<f32>
    %v4500 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4501 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4502 = stablehlo.multiply %v4500, %d0nbtm : tensor<f32>
    %v4503 = stablehlo.multiply %v4501, %v3029 : tensor<f32>
    %v4504 = stablehlo.add %v4502, %v4503 : tensor<f32>
    %v4505 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4506 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4507 = stablehlo.multiply %v4505, %d0nbtv : tensor<f32>
    %v4508 = stablehlo.multiply %v3029, %v3029 : tensor<f32>
    %v4509 = stablehlo.multiply %v4506, %v4508 : tensor<f32>
    %v4510 = stablehlo.add %v4507, %v4509 : tensor<f32>
    %v4511 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4512 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4513 = stablehlo.multiply %v4511, %d0nbtm : tensor<f32>
    %v4514 = stablehlo.multiply %v4512, %v3029 : tensor<f32>
    %v4515 = stablehlo.add %v4513, %v4514 : tensor<f32>
    %v4516 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4517 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4518 = stablehlo.multiply %v4516, %d0nbtv : tensor<f32>
    %v4519 = stablehlo.multiply %v3029, %v3029 : tensor<f32>
    %v4520 = stablehlo.multiply %v4517, %v4519 : tensor<f32>
    %v4521 = stablehlo.add %v4518, %v4520 : tensor<f32>
    %v4522 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4523 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4524 = stablehlo.divide %v4515, %v4522 : tensor<f32>
    %v4525 = stablehlo.divide %v4521, %v4523 : tensor<f32>
    %v4526 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4527 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4528 = stablehlo.sqrt %v4525 : tensor<f32>
    %v4529 = stablehlo.add %v4528, %v4527 : tensor<f32>
    %v4530 = stablehlo.divide %v4524, %v4529 : tensor<f32>
    %v4531 = stablehlo.multiply %v4526, %v4530 : tensor<f32>
    %v4532 = stablehlo.subtract %d0nbt, %v4531 : tensor<f32>
    %v4533 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4534 = stablehlo.multiply %v4533, %v4526 : tensor<f32>
    %v4535 = stablehlo.multiply %v4534, %d0nbt : tensor<f32>
    %v4536 = stablehlo.subtract %v4532, %v4535 : tensor<f32>
    %v4537 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4538 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4539 = stablehlo.multiply %v4537, %d0Wm : tensor<192x96x2x2xf32>
    %v4540 = stablehlo.multiply %v4538, %dd0W : tensor<192x96x2x2xf32>
    %v4541 = stablehlo.add %v4539, %v4540 : tensor<192x96x2x2xf32>
    %v4542 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4543 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4544 = stablehlo.multiply %v4542, %d0Wv : tensor<192x96x2x2xf32>
    %v4545 = stablehlo.multiply %dd0W, %dd0W : tensor<192x96x2x2xf32>
    %v4546 = stablehlo.multiply %v4543, %v4545 : tensor<192x96x2x2xf32>
    %v4547 = stablehlo.add %v4544, %v4546 : tensor<192x96x2x2xf32>
    %v4548 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4549 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4550 = stablehlo.multiply %v4548, %d0Wm : tensor<192x96x2x2xf32>
    %v4551 = stablehlo.multiply %v4549, %dd0W : tensor<192x96x2x2xf32>
    %v4552 = stablehlo.add %v4550, %v4551 : tensor<192x96x2x2xf32>
    %v4553 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4554 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4555 = stablehlo.multiply %v4553, %d0Wv : tensor<192x96x2x2xf32>
    %v4556 = stablehlo.multiply %dd0W, %dd0W : tensor<192x96x2x2xf32>
    %v4557 = stablehlo.multiply %v4554, %v4556 : tensor<192x96x2x2xf32>
    %v4558 = stablehlo.add %v4555, %v4557 : tensor<192x96x2x2xf32>
    %v4559 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4560 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4561 = stablehlo.divide %v4552, %v4559 : tensor<192x96x2x2xf32>
    %v4562 = stablehlo.divide %v4558, %v4560 : tensor<192x96x2x2xf32>
    %v4563 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4564 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4565 = stablehlo.sqrt %v4562 : tensor<192x96x2x2xf32>
    %v4566 = stablehlo.add %v4565, %v4564 : tensor<192x96x2x2xf32>
    %v4567 = stablehlo.divide %v4561, %v4566 : tensor<192x96x2x2xf32>
    %v4568 = stablehlo.multiply %v4563, %v4567 : tensor<192x96x2x2xf32>
    %v4569 = stablehlo.subtract %d0W, %v4568 : tensor<192x96x2x2xf32>
    %v4570 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v4571 = stablehlo.multiply %v4570, %v4563 : tensor<192x96x2x2xf32>
    %v4572 = stablehlo.multiply %v4571, %d0W : tensor<192x96x2x2xf32>
    %v4573 = stablehlo.subtract %v4569, %v4572 : tensor<192x96x2x2xf32>
    %v4574 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4575 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4576 = stablehlo.multiply %v4574, %d0bm : tensor<192xf32>
    %v4577 = stablehlo.multiply %v4575, %v3011 : tensor<192xf32>
    %v4578 = stablehlo.add %v4576, %v4577 : tensor<192xf32>
    %v4579 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4580 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4581 = stablehlo.multiply %v4579, %d0bv : tensor<192xf32>
    %v4582 = stablehlo.multiply %v3011, %v3011 : tensor<192xf32>
    %v4583 = stablehlo.multiply %v4580, %v4582 : tensor<192xf32>
    %v4584 = stablehlo.add %v4581, %v4583 : tensor<192xf32>
    %v4585 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4586 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4587 = stablehlo.multiply %v4585, %d0bm : tensor<192xf32>
    %v4588 = stablehlo.multiply %v4586, %v3011 : tensor<192xf32>
    %v4589 = stablehlo.add %v4587, %v4588 : tensor<192xf32>
    %v4590 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4591 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4592 = stablehlo.multiply %v4590, %d0bv : tensor<192xf32>
    %v4593 = stablehlo.multiply %v3011, %v3011 : tensor<192xf32>
    %v4594 = stablehlo.multiply %v4591, %v4593 : tensor<192xf32>
    %v4595 = stablehlo.add %v4592, %v4594 : tensor<192xf32>
    %v4596 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4597 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4598 = stablehlo.divide %v4589, %v4596 : tensor<192xf32>
    %v4599 = stablehlo.divide %v4595, %v4597 : tensor<192xf32>
    %v4600 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4601 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4602 = stablehlo.sqrt %v4599 : tensor<192xf32>
    %v4603 = stablehlo.add %v4602, %v4601 : tensor<192xf32>
    %v4604 = stablehlo.divide %v4598, %v4603 : tensor<192xf32>
    %v4605 = stablehlo.multiply %v4600, %v4604 : tensor<192xf32>
    %v4606 = stablehlo.subtract %d0b, %v4605 : tensor<192xf32>
    %v4607 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4608 = stablehlo.multiply %v4607, %v4600 : tensor<192xf32>
    %v4609 = stablehlo.multiply %v4608, %d0b : tensor<192xf32>
    %v4610 = stablehlo.subtract %v4606, %v4609 : tensor<192xf32>
    %v4611 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4612 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4613 = stablehlo.multiply %v4611, %s1b0dWm : tensor<192x1x7x7xf32>
    %v4614 = stablehlo.multiply %v4612, %v2971 : tensor<192x1x7x7xf32>
    %v4615 = stablehlo.add %v4613, %v4614 : tensor<192x1x7x7xf32>
    %v4616 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4617 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4618 = stablehlo.multiply %v4616, %s1b0dWv : tensor<192x1x7x7xf32>
    %v4619 = stablehlo.multiply %v2971, %v2971 : tensor<192x1x7x7xf32>
    %v4620 = stablehlo.multiply %v4617, %v4619 : tensor<192x1x7x7xf32>
    %v4621 = stablehlo.add %v4618, %v4620 : tensor<192x1x7x7xf32>
    %v4622 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4623 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4624 = stablehlo.multiply %v4622, %s1b0dWm : tensor<192x1x7x7xf32>
    %v4625 = stablehlo.multiply %v4623, %v2971 : tensor<192x1x7x7xf32>
    %v4626 = stablehlo.add %v4624, %v4625 : tensor<192x1x7x7xf32>
    %v4627 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4628 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4629 = stablehlo.multiply %v4627, %s1b0dWv : tensor<192x1x7x7xf32>
    %v4630 = stablehlo.multiply %v2971, %v2971 : tensor<192x1x7x7xf32>
    %v4631 = stablehlo.multiply %v4628, %v4630 : tensor<192x1x7x7xf32>
    %v4632 = stablehlo.add %v4629, %v4631 : tensor<192x1x7x7xf32>
    %v4633 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4634 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4635 = stablehlo.divide %v4626, %v4633 : tensor<192x1x7x7xf32>
    %v4636 = stablehlo.divide %v4632, %v4634 : tensor<192x1x7x7xf32>
    %v4637 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4638 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4639 = stablehlo.sqrt %v4636 : tensor<192x1x7x7xf32>
    %v4640 = stablehlo.add %v4639, %v4638 : tensor<192x1x7x7xf32>
    %v4641 = stablehlo.divide %v4635, %v4640 : tensor<192x1x7x7xf32>
    %v4642 = stablehlo.multiply %v4637, %v4641 : tensor<192x1x7x7xf32>
    %v4643 = stablehlo.subtract %s1b0dW, %v4642 : tensor<192x1x7x7xf32>
    %v4644 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4645 = stablehlo.multiply %v4644, %v4637 : tensor<192x1x7x7xf32>
    %v4646 = stablehlo.multiply %v4645, %s1b0dW : tensor<192x1x7x7xf32>
    %v4647 = stablehlo.subtract %v4643, %v4646 : tensor<192x1x7x7xf32>
    %v4648 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4649 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4650 = stablehlo.multiply %v4648, %s1b0dbm : tensor<192xf32>
    %v4651 = stablehlo.multiply %v4649, %v2974 : tensor<192xf32>
    %v4652 = stablehlo.add %v4650, %v4651 : tensor<192xf32>
    %v4653 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4654 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4655 = stablehlo.multiply %v4653, %s1b0dbv : tensor<192xf32>
    %v4656 = stablehlo.multiply %v2974, %v2974 : tensor<192xf32>
    %v4657 = stablehlo.multiply %v4654, %v4656 : tensor<192xf32>
    %v4658 = stablehlo.add %v4655, %v4657 : tensor<192xf32>
    %v4659 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4660 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4661 = stablehlo.multiply %v4659, %s1b0dbm : tensor<192xf32>
    %v4662 = stablehlo.multiply %v4660, %v2974 : tensor<192xf32>
    %v4663 = stablehlo.add %v4661, %v4662 : tensor<192xf32>
    %v4664 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4665 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4666 = stablehlo.multiply %v4664, %s1b0dbv : tensor<192xf32>
    %v4667 = stablehlo.multiply %v2974, %v2974 : tensor<192xf32>
    %v4668 = stablehlo.multiply %v4665, %v4667 : tensor<192xf32>
    %v4669 = stablehlo.add %v4666, %v4668 : tensor<192xf32>
    %v4670 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4671 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4672 = stablehlo.divide %v4663, %v4670 : tensor<192xf32>
    %v4673 = stablehlo.divide %v4669, %v4671 : tensor<192xf32>
    %v4674 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4675 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4676 = stablehlo.sqrt %v4673 : tensor<192xf32>
    %v4677 = stablehlo.add %v4676, %v4675 : tensor<192xf32>
    %v4678 = stablehlo.divide %v4672, %v4677 : tensor<192xf32>
    %v4679 = stablehlo.multiply %v4674, %v4678 : tensor<192xf32>
    %v4680 = stablehlo.subtract %s1b0db, %v4679 : tensor<192xf32>
    %v4681 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4682 = stablehlo.multiply %v4681, %v4674 : tensor<192xf32>
    %v4683 = stablehlo.multiply %v4682, %s1b0db : tensor<192xf32>
    %v4684 = stablehlo.subtract %v4680, %v4683 : tensor<192xf32>
    %v4685 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4686 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4687 = stablehlo.multiply %v4685, %s1b0ngm : tensor<f32>
    %v4688 = stablehlo.multiply %v4686, %v2963 : tensor<f32>
    %v4689 = stablehlo.add %v4687, %v4688 : tensor<f32>
    %v4690 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4691 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4692 = stablehlo.multiply %v4690, %s1b0ngv : tensor<f32>
    %v4693 = stablehlo.multiply %v2963, %v2963 : tensor<f32>
    %v4694 = stablehlo.multiply %v4691, %v4693 : tensor<f32>
    %v4695 = stablehlo.add %v4692, %v4694 : tensor<f32>
    %v4696 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4697 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4698 = stablehlo.multiply %v4696, %s1b0ngm : tensor<f32>
    %v4699 = stablehlo.multiply %v4697, %v2963 : tensor<f32>
    %v4700 = stablehlo.add %v4698, %v4699 : tensor<f32>
    %v4701 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4702 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4703 = stablehlo.multiply %v4701, %s1b0ngv : tensor<f32>
    %v4704 = stablehlo.multiply %v2963, %v2963 : tensor<f32>
    %v4705 = stablehlo.multiply %v4702, %v4704 : tensor<f32>
    %v4706 = stablehlo.add %v4703, %v4705 : tensor<f32>
    %v4707 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4708 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4709 = stablehlo.divide %v4700, %v4707 : tensor<f32>
    %v4710 = stablehlo.divide %v4706, %v4708 : tensor<f32>
    %v4711 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4712 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4713 = stablehlo.sqrt %v4710 : tensor<f32>
    %v4714 = stablehlo.add %v4713, %v4712 : tensor<f32>
    %v4715 = stablehlo.divide %v4709, %v4714 : tensor<f32>
    %v4716 = stablehlo.multiply %v4711, %v4715 : tensor<f32>
    %v4717 = stablehlo.subtract %s1b0ng, %v4716 : tensor<f32>
    %v4718 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4719 = stablehlo.multiply %v4718, %v4711 : tensor<f32>
    %v4720 = stablehlo.multiply %v4719, %s1b0ng : tensor<f32>
    %v4721 = stablehlo.subtract %v4717, %v4720 : tensor<f32>
    %v4722 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4723 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4724 = stablehlo.multiply %v4722, %s1b0nbtm : tensor<f32>
    %v4725 = stablehlo.multiply %v4723, %v2965 : tensor<f32>
    %v4726 = stablehlo.add %v4724, %v4725 : tensor<f32>
    %v4727 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4728 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4729 = stablehlo.multiply %v4727, %s1b0nbtv : tensor<f32>
    %v4730 = stablehlo.multiply %v2965, %v2965 : tensor<f32>
    %v4731 = stablehlo.multiply %v4728, %v4730 : tensor<f32>
    %v4732 = stablehlo.add %v4729, %v4731 : tensor<f32>
    %v4733 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4734 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4735 = stablehlo.multiply %v4733, %s1b0nbtm : tensor<f32>
    %v4736 = stablehlo.multiply %v4734, %v2965 : tensor<f32>
    %v4737 = stablehlo.add %v4735, %v4736 : tensor<f32>
    %v4738 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4739 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4740 = stablehlo.multiply %v4738, %s1b0nbtv : tensor<f32>
    %v4741 = stablehlo.multiply %v2965, %v2965 : tensor<f32>
    %v4742 = stablehlo.multiply %v4739, %v4741 : tensor<f32>
    %v4743 = stablehlo.add %v4740, %v4742 : tensor<f32>
    %v4744 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4745 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4746 = stablehlo.divide %v4737, %v4744 : tensor<f32>
    %v4747 = stablehlo.divide %v4743, %v4745 : tensor<f32>
    %v4748 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4749 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4750 = stablehlo.sqrt %v4747 : tensor<f32>
    %v4751 = stablehlo.add %v4750, %v4749 : tensor<f32>
    %v4752 = stablehlo.divide %v4746, %v4751 : tensor<f32>
    %v4753 = stablehlo.multiply %v4748, %v4752 : tensor<f32>
    %v4754 = stablehlo.subtract %s1b0nbt, %v4753 : tensor<f32>
    %v4755 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v4756 = stablehlo.multiply %v4755, %v4748 : tensor<f32>
    %v4757 = stablehlo.multiply %v4756, %s1b0nbt : tensor<f32>
    %v4758 = stablehlo.subtract %v4754, %v4757 : tensor<f32>
    %v4759 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4760 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4761 = stablehlo.multiply %v4759, %s1b0eWm : tensor<768x192x1x1xf32>
    %v4762 = stablehlo.multiply %v4760, %v2944 : tensor<768x192x1x1xf32>
    %v4763 = stablehlo.add %v4761, %v4762 : tensor<768x192x1x1xf32>
    %v4764 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4765 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4766 = stablehlo.multiply %v4764, %s1b0eWv : tensor<768x192x1x1xf32>
    %v4767 = stablehlo.multiply %v2944, %v2944 : tensor<768x192x1x1xf32>
    %v4768 = stablehlo.multiply %v4765, %v4767 : tensor<768x192x1x1xf32>
    %v4769 = stablehlo.add %v4766, %v4768 : tensor<768x192x1x1xf32>
    %v4770 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4771 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4772 = stablehlo.multiply %v4770, %s1b0eWm : tensor<768x192x1x1xf32>
    %v4773 = stablehlo.multiply %v4771, %v2944 : tensor<768x192x1x1xf32>
    %v4774 = stablehlo.add %v4772, %v4773 : tensor<768x192x1x1xf32>
    %v4775 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4776 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4777 = stablehlo.multiply %v4775, %s1b0eWv : tensor<768x192x1x1xf32>
    %v4778 = stablehlo.multiply %v2944, %v2944 : tensor<768x192x1x1xf32>
    %v4779 = stablehlo.multiply %v4776, %v4778 : tensor<768x192x1x1xf32>
    %v4780 = stablehlo.add %v4777, %v4779 : tensor<768x192x1x1xf32>
    %v4781 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4782 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4783 = stablehlo.divide %v4774, %v4781 : tensor<768x192x1x1xf32>
    %v4784 = stablehlo.divide %v4780, %v4782 : tensor<768x192x1x1xf32>
    %v4785 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4786 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4787 = stablehlo.sqrt %v4784 : tensor<768x192x1x1xf32>
    %v4788 = stablehlo.add %v4787, %v4786 : tensor<768x192x1x1xf32>
    %v4789 = stablehlo.divide %v4783, %v4788 : tensor<768x192x1x1xf32>
    %v4790 = stablehlo.multiply %v4785, %v4789 : tensor<768x192x1x1xf32>
    %v4791 = stablehlo.subtract %s1b0eW, %v4790 : tensor<768x192x1x1xf32>
    %v4792 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v4793 = stablehlo.multiply %v4792, %v4785 : tensor<768x192x1x1xf32>
    %v4794 = stablehlo.multiply %v4793, %s1b0eW : tensor<768x192x1x1xf32>
    %v4795 = stablehlo.subtract %v4791, %v4794 : tensor<768x192x1x1xf32>
    %v4796 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4797 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4798 = stablehlo.multiply %v4796, %s1b0ebm : tensor<768xf32>
    %v4799 = stablehlo.multiply %v4797, %v2947 : tensor<768xf32>
    %v4800 = stablehlo.add %v4798, %v4799 : tensor<768xf32>
    %v4801 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4802 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4803 = stablehlo.multiply %v4801, %s1b0ebv : tensor<768xf32>
    %v4804 = stablehlo.multiply %v2947, %v2947 : tensor<768xf32>
    %v4805 = stablehlo.multiply %v4802, %v4804 : tensor<768xf32>
    %v4806 = stablehlo.add %v4803, %v4805 : tensor<768xf32>
    %v4807 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4808 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4809 = stablehlo.multiply %v4807, %s1b0ebm : tensor<768xf32>
    %v4810 = stablehlo.multiply %v4808, %v2947 : tensor<768xf32>
    %v4811 = stablehlo.add %v4809, %v4810 : tensor<768xf32>
    %v4812 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4813 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4814 = stablehlo.multiply %v4812, %s1b0ebv : tensor<768xf32>
    %v4815 = stablehlo.multiply %v2947, %v2947 : tensor<768xf32>
    %v4816 = stablehlo.multiply %v4813, %v4815 : tensor<768xf32>
    %v4817 = stablehlo.add %v4814, %v4816 : tensor<768xf32>
    %v4818 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4819 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4820 = stablehlo.divide %v4811, %v4818 : tensor<768xf32>
    %v4821 = stablehlo.divide %v4817, %v4819 : tensor<768xf32>
    %v4822 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4823 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4824 = stablehlo.sqrt %v4821 : tensor<768xf32>
    %v4825 = stablehlo.add %v4824, %v4823 : tensor<768xf32>
    %v4826 = stablehlo.divide %v4820, %v4825 : tensor<768xf32>
    %v4827 = stablehlo.multiply %v4822, %v4826 : tensor<768xf32>
    %v4828 = stablehlo.subtract %s1b0eb, %v4827 : tensor<768xf32>
    %v4829 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v4830 = stablehlo.multiply %v4829, %v4822 : tensor<768xf32>
    %v4831 = stablehlo.multiply %v4830, %s1b0eb : tensor<768xf32>
    %v4832 = stablehlo.subtract %v4828, %v4831 : tensor<768xf32>
    %v4833 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4834 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4835 = stablehlo.multiply %v4833, %s1b0pWm : tensor<192x768x1x1xf32>
    %v4836 = stablehlo.multiply %v4834, %v2935 : tensor<192x768x1x1xf32>
    %v4837 = stablehlo.add %v4835, %v4836 : tensor<192x768x1x1xf32>
    %v4838 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4839 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4840 = stablehlo.multiply %v4838, %s1b0pWv : tensor<192x768x1x1xf32>
    %v4841 = stablehlo.multiply %v2935, %v2935 : tensor<192x768x1x1xf32>
    %v4842 = stablehlo.multiply %v4839, %v4841 : tensor<192x768x1x1xf32>
    %v4843 = stablehlo.add %v4840, %v4842 : tensor<192x768x1x1xf32>
    %v4844 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4845 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4846 = stablehlo.multiply %v4844, %s1b0pWm : tensor<192x768x1x1xf32>
    %v4847 = stablehlo.multiply %v4845, %v2935 : tensor<192x768x1x1xf32>
    %v4848 = stablehlo.add %v4846, %v4847 : tensor<192x768x1x1xf32>
    %v4849 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4850 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4851 = stablehlo.multiply %v4849, %s1b0pWv : tensor<192x768x1x1xf32>
    %v4852 = stablehlo.multiply %v2935, %v2935 : tensor<192x768x1x1xf32>
    %v4853 = stablehlo.multiply %v4850, %v4852 : tensor<192x768x1x1xf32>
    %v4854 = stablehlo.add %v4851, %v4853 : tensor<192x768x1x1xf32>
    %v4855 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4856 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4857 = stablehlo.divide %v4848, %v4855 : tensor<192x768x1x1xf32>
    %v4858 = stablehlo.divide %v4854, %v4856 : tensor<192x768x1x1xf32>
    %v4859 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4860 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4861 = stablehlo.sqrt %v4858 : tensor<192x768x1x1xf32>
    %v4862 = stablehlo.add %v4861, %v4860 : tensor<192x768x1x1xf32>
    %v4863 = stablehlo.divide %v4857, %v4862 : tensor<192x768x1x1xf32>
    %v4864 = stablehlo.multiply %v4859, %v4863 : tensor<192x768x1x1xf32>
    %v4865 = stablehlo.subtract %s1b0pW, %v4864 : tensor<192x768x1x1xf32>
    %v4866 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v4867 = stablehlo.multiply %v4866, %v4859 : tensor<192x768x1x1xf32>
    %v4868 = stablehlo.multiply %v4867, %s1b0pW : tensor<192x768x1x1xf32>
    %v4869 = stablehlo.subtract %v4865, %v4868 : tensor<192x768x1x1xf32>
    %v4870 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4871 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4872 = stablehlo.multiply %v4870, %s1b0pbm : tensor<192xf32>
    %v4873 = stablehlo.multiply %v4871, %v2938 : tensor<192xf32>
    %v4874 = stablehlo.add %v4872, %v4873 : tensor<192xf32>
    %v4875 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4876 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4877 = stablehlo.multiply %v4875, %s1b0pbv : tensor<192xf32>
    %v4878 = stablehlo.multiply %v2938, %v2938 : tensor<192xf32>
    %v4879 = stablehlo.multiply %v4876, %v4878 : tensor<192xf32>
    %v4880 = stablehlo.add %v4877, %v4879 : tensor<192xf32>
    %v4881 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4882 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4883 = stablehlo.multiply %v4881, %s1b0pbm : tensor<192xf32>
    %v4884 = stablehlo.multiply %v4882, %v2938 : tensor<192xf32>
    %v4885 = stablehlo.add %v4883, %v4884 : tensor<192xf32>
    %v4886 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4887 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4888 = stablehlo.multiply %v4886, %s1b0pbv : tensor<192xf32>
    %v4889 = stablehlo.multiply %v2938, %v2938 : tensor<192xf32>
    %v4890 = stablehlo.multiply %v4887, %v4889 : tensor<192xf32>
    %v4891 = stablehlo.add %v4888, %v4890 : tensor<192xf32>
    %v4892 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4893 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4894 = stablehlo.divide %v4885, %v4892 : tensor<192xf32>
    %v4895 = stablehlo.divide %v4891, %v4893 : tensor<192xf32>
    %v4896 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4897 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4898 = stablehlo.sqrt %v4895 : tensor<192xf32>
    %v4899 = stablehlo.add %v4898, %v4897 : tensor<192xf32>
    %v4900 = stablehlo.divide %v4894, %v4899 : tensor<192xf32>
    %v4901 = stablehlo.multiply %v4896, %v4900 : tensor<192xf32>
    %v4902 = stablehlo.subtract %s1b0pb, %v4901 : tensor<192xf32>
    %v4903 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4904 = stablehlo.multiply %v4903, %v4896 : tensor<192xf32>
    %v4905 = stablehlo.multiply %v4904, %s1b0pb : tensor<192xf32>
    %v4906 = stablehlo.subtract %v4902, %v4905 : tensor<192xf32>
    %v4907 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4908 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4909 = stablehlo.multiply %v4907, %s1b0lgm : tensor<192xf32>
    %v4910 = stablehlo.multiply %v4908, %v2929 : tensor<192xf32>
    %v4911 = stablehlo.add %v4909, %v4910 : tensor<192xf32>
    %v4912 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4913 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4914 = stablehlo.multiply %v4912, %s1b0lgv : tensor<192xf32>
    %v4915 = stablehlo.multiply %v2929, %v2929 : tensor<192xf32>
    %v4916 = stablehlo.multiply %v4913, %v4915 : tensor<192xf32>
    %v4917 = stablehlo.add %v4914, %v4916 : tensor<192xf32>
    %v4918 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4919 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4920 = stablehlo.multiply %v4918, %s1b0lgm : tensor<192xf32>
    %v4921 = stablehlo.multiply %v4919, %v2929 : tensor<192xf32>
    %v4922 = stablehlo.add %v4920, %v4921 : tensor<192xf32>
    %v4923 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4924 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4925 = stablehlo.multiply %v4923, %s1b0lgv : tensor<192xf32>
    %v4926 = stablehlo.multiply %v2929, %v2929 : tensor<192xf32>
    %v4927 = stablehlo.multiply %v4924, %v4926 : tensor<192xf32>
    %v4928 = stablehlo.add %v4925, %v4927 : tensor<192xf32>
    %v4929 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4930 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4931 = stablehlo.divide %v4922, %v4929 : tensor<192xf32>
    %v4932 = stablehlo.divide %v4928, %v4930 : tensor<192xf32>
    %v4933 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4934 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4935 = stablehlo.sqrt %v4932 : tensor<192xf32>
    %v4936 = stablehlo.add %v4935, %v4934 : tensor<192xf32>
    %v4937 = stablehlo.divide %v4931, %v4936 : tensor<192xf32>
    %v4938 = stablehlo.multiply %v4933, %v4937 : tensor<192xf32>
    %v4939 = stablehlo.subtract %s1b0lg, %v4938 : tensor<192xf32>
    %v4940 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4941 = stablehlo.multiply %v4940, %v4933 : tensor<192xf32>
    %v4942 = stablehlo.multiply %v4941, %s1b0lg : tensor<192xf32>
    %v4943 = stablehlo.subtract %v4939, %v4942 : tensor<192xf32>
    %v4944 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4945 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4946 = stablehlo.multiply %v4944, %s1b1dWm : tensor<192x1x7x7xf32>
    %v4947 = stablehlo.multiply %v4945, %v2852 : tensor<192x1x7x7xf32>
    %v4948 = stablehlo.add %v4946, %v4947 : tensor<192x1x7x7xf32>
    %v4949 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4950 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4951 = stablehlo.multiply %v4949, %s1b1dWv : tensor<192x1x7x7xf32>
    %v4952 = stablehlo.multiply %v2852, %v2852 : tensor<192x1x7x7xf32>
    %v4953 = stablehlo.multiply %v4950, %v4952 : tensor<192x1x7x7xf32>
    %v4954 = stablehlo.add %v4951, %v4953 : tensor<192x1x7x7xf32>
    %v4955 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4956 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4957 = stablehlo.multiply %v4955, %s1b1dWm : tensor<192x1x7x7xf32>
    %v4958 = stablehlo.multiply %v4956, %v2852 : tensor<192x1x7x7xf32>
    %v4959 = stablehlo.add %v4957, %v4958 : tensor<192x1x7x7xf32>
    %v4960 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4961 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4962 = stablehlo.multiply %v4960, %s1b1dWv : tensor<192x1x7x7xf32>
    %v4963 = stablehlo.multiply %v2852, %v2852 : tensor<192x1x7x7xf32>
    %v4964 = stablehlo.multiply %v4961, %v4963 : tensor<192x1x7x7xf32>
    %v4965 = stablehlo.add %v4962, %v4964 : tensor<192x1x7x7xf32>
    %v4966 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4967 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4968 = stablehlo.divide %v4959, %v4966 : tensor<192x1x7x7xf32>
    %v4969 = stablehlo.divide %v4965, %v4967 : tensor<192x1x7x7xf32>
    %v4970 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4971 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4972 = stablehlo.sqrt %v4969 : tensor<192x1x7x7xf32>
    %v4973 = stablehlo.add %v4972, %v4971 : tensor<192x1x7x7xf32>
    %v4974 = stablehlo.divide %v4968, %v4973 : tensor<192x1x7x7xf32>
    %v4975 = stablehlo.multiply %v4970, %v4974 : tensor<192x1x7x7xf32>
    %v4976 = stablehlo.subtract %s1b1dW, %v4975 : tensor<192x1x7x7xf32>
    %v4977 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v4978 = stablehlo.multiply %v4977, %v4970 : tensor<192x1x7x7xf32>
    %v4979 = stablehlo.multiply %v4978, %s1b1dW : tensor<192x1x7x7xf32>
    %v4980 = stablehlo.subtract %v4976, %v4979 : tensor<192x1x7x7xf32>
    %v4981 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4982 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4983 = stablehlo.multiply %v4981, %s1b1dbm : tensor<192xf32>
    %v4984 = stablehlo.multiply %v4982, %v2855 : tensor<192xf32>
    %v4985 = stablehlo.add %v4983, %v4984 : tensor<192xf32>
    %v4986 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4987 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4988 = stablehlo.multiply %v4986, %s1b1dbv : tensor<192xf32>
    %v4989 = stablehlo.multiply %v2855, %v2855 : tensor<192xf32>
    %v4990 = stablehlo.multiply %v4987, %v4989 : tensor<192xf32>
    %v4991 = stablehlo.add %v4988, %v4990 : tensor<192xf32>
    %v4992 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4993 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4994 = stablehlo.multiply %v4992, %s1b1dbm : tensor<192xf32>
    %v4995 = stablehlo.multiply %v4993, %v2855 : tensor<192xf32>
    %v4996 = stablehlo.add %v4994, %v4995 : tensor<192xf32>
    %v4997 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4998 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v4999 = stablehlo.multiply %v4997, %s1b1dbv : tensor<192xf32>
    %v5000 = stablehlo.multiply %v2855, %v2855 : tensor<192xf32>
    %v5001 = stablehlo.multiply %v4998, %v5000 : tensor<192xf32>
    %v5002 = stablehlo.add %v4999, %v5001 : tensor<192xf32>
    %v5003 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5004 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5005 = stablehlo.divide %v4996, %v5003 : tensor<192xf32>
    %v5006 = stablehlo.divide %v5002, %v5004 : tensor<192xf32>
    %v5007 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5008 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5009 = stablehlo.sqrt %v5006 : tensor<192xf32>
    %v5010 = stablehlo.add %v5009, %v5008 : tensor<192xf32>
    %v5011 = stablehlo.divide %v5005, %v5010 : tensor<192xf32>
    %v5012 = stablehlo.multiply %v5007, %v5011 : tensor<192xf32>
    %v5013 = stablehlo.subtract %s1b1db, %v5012 : tensor<192xf32>
    %v5014 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5015 = stablehlo.multiply %v5014, %v5007 : tensor<192xf32>
    %v5016 = stablehlo.multiply %v5015, %s1b1db : tensor<192xf32>
    %v5017 = stablehlo.subtract %v5013, %v5016 : tensor<192xf32>
    %v5018 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5019 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5020 = stablehlo.multiply %v5018, %s1b1ngm : tensor<f32>
    %v5021 = stablehlo.multiply %v5019, %v2844 : tensor<f32>
    %v5022 = stablehlo.add %v5020, %v5021 : tensor<f32>
    %v5023 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5024 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5025 = stablehlo.multiply %v5023, %s1b1ngv : tensor<f32>
    %v5026 = stablehlo.multiply %v2844, %v2844 : tensor<f32>
    %v5027 = stablehlo.multiply %v5024, %v5026 : tensor<f32>
    %v5028 = stablehlo.add %v5025, %v5027 : tensor<f32>
    %v5029 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5030 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5031 = stablehlo.multiply %v5029, %s1b1ngm : tensor<f32>
    %v5032 = stablehlo.multiply %v5030, %v2844 : tensor<f32>
    %v5033 = stablehlo.add %v5031, %v5032 : tensor<f32>
    %v5034 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5035 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5036 = stablehlo.multiply %v5034, %s1b1ngv : tensor<f32>
    %v5037 = stablehlo.multiply %v2844, %v2844 : tensor<f32>
    %v5038 = stablehlo.multiply %v5035, %v5037 : tensor<f32>
    %v5039 = stablehlo.add %v5036, %v5038 : tensor<f32>
    %v5040 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5041 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5042 = stablehlo.divide %v5033, %v5040 : tensor<f32>
    %v5043 = stablehlo.divide %v5039, %v5041 : tensor<f32>
    %v5044 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5045 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5046 = stablehlo.sqrt %v5043 : tensor<f32>
    %v5047 = stablehlo.add %v5046, %v5045 : tensor<f32>
    %v5048 = stablehlo.divide %v5042, %v5047 : tensor<f32>
    %v5049 = stablehlo.multiply %v5044, %v5048 : tensor<f32>
    %v5050 = stablehlo.subtract %s1b1ng, %v5049 : tensor<f32>
    %v5051 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5052 = stablehlo.multiply %v5051, %v5044 : tensor<f32>
    %v5053 = stablehlo.multiply %v5052, %s1b1ng : tensor<f32>
    %v5054 = stablehlo.subtract %v5050, %v5053 : tensor<f32>
    %v5055 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5056 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5057 = stablehlo.multiply %v5055, %s1b1nbtm : tensor<f32>
    %v5058 = stablehlo.multiply %v5056, %v2846 : tensor<f32>
    %v5059 = stablehlo.add %v5057, %v5058 : tensor<f32>
    %v5060 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5061 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5062 = stablehlo.multiply %v5060, %s1b1nbtv : tensor<f32>
    %v5063 = stablehlo.multiply %v2846, %v2846 : tensor<f32>
    %v5064 = stablehlo.multiply %v5061, %v5063 : tensor<f32>
    %v5065 = stablehlo.add %v5062, %v5064 : tensor<f32>
    %v5066 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5067 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5068 = stablehlo.multiply %v5066, %s1b1nbtm : tensor<f32>
    %v5069 = stablehlo.multiply %v5067, %v2846 : tensor<f32>
    %v5070 = stablehlo.add %v5068, %v5069 : tensor<f32>
    %v5071 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5072 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5073 = stablehlo.multiply %v5071, %s1b1nbtv : tensor<f32>
    %v5074 = stablehlo.multiply %v2846, %v2846 : tensor<f32>
    %v5075 = stablehlo.multiply %v5072, %v5074 : tensor<f32>
    %v5076 = stablehlo.add %v5073, %v5075 : tensor<f32>
    %v5077 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5078 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5079 = stablehlo.divide %v5070, %v5077 : tensor<f32>
    %v5080 = stablehlo.divide %v5076, %v5078 : tensor<f32>
    %v5081 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5082 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5083 = stablehlo.sqrt %v5080 : tensor<f32>
    %v5084 = stablehlo.add %v5083, %v5082 : tensor<f32>
    %v5085 = stablehlo.divide %v5079, %v5084 : tensor<f32>
    %v5086 = stablehlo.multiply %v5081, %v5085 : tensor<f32>
    %v5087 = stablehlo.subtract %s1b1nbt, %v5086 : tensor<f32>
    %v5088 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5089 = stablehlo.multiply %v5088, %v5081 : tensor<f32>
    %v5090 = stablehlo.multiply %v5089, %s1b1nbt : tensor<f32>
    %v5091 = stablehlo.subtract %v5087, %v5090 : tensor<f32>
    %v5092 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5093 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5094 = stablehlo.multiply %v5092, %s1b1eWm : tensor<768x192x1x1xf32>
    %v5095 = stablehlo.multiply %v5093, %v2825 : tensor<768x192x1x1xf32>
    %v5096 = stablehlo.add %v5094, %v5095 : tensor<768x192x1x1xf32>
    %v5097 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5098 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5099 = stablehlo.multiply %v5097, %s1b1eWv : tensor<768x192x1x1xf32>
    %v5100 = stablehlo.multiply %v2825, %v2825 : tensor<768x192x1x1xf32>
    %v5101 = stablehlo.multiply %v5098, %v5100 : tensor<768x192x1x1xf32>
    %v5102 = stablehlo.add %v5099, %v5101 : tensor<768x192x1x1xf32>
    %v5103 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5104 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5105 = stablehlo.multiply %v5103, %s1b1eWm : tensor<768x192x1x1xf32>
    %v5106 = stablehlo.multiply %v5104, %v2825 : tensor<768x192x1x1xf32>
    %v5107 = stablehlo.add %v5105, %v5106 : tensor<768x192x1x1xf32>
    %v5108 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5109 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5110 = stablehlo.multiply %v5108, %s1b1eWv : tensor<768x192x1x1xf32>
    %v5111 = stablehlo.multiply %v2825, %v2825 : tensor<768x192x1x1xf32>
    %v5112 = stablehlo.multiply %v5109, %v5111 : tensor<768x192x1x1xf32>
    %v5113 = stablehlo.add %v5110, %v5112 : tensor<768x192x1x1xf32>
    %v5114 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5115 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5116 = stablehlo.divide %v5107, %v5114 : tensor<768x192x1x1xf32>
    %v5117 = stablehlo.divide %v5113, %v5115 : tensor<768x192x1x1xf32>
    %v5118 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5119 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5120 = stablehlo.sqrt %v5117 : tensor<768x192x1x1xf32>
    %v5121 = stablehlo.add %v5120, %v5119 : tensor<768x192x1x1xf32>
    %v5122 = stablehlo.divide %v5116, %v5121 : tensor<768x192x1x1xf32>
    %v5123 = stablehlo.multiply %v5118, %v5122 : tensor<768x192x1x1xf32>
    %v5124 = stablehlo.subtract %s1b1eW, %v5123 : tensor<768x192x1x1xf32>
    %v5125 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5126 = stablehlo.multiply %v5125, %v5118 : tensor<768x192x1x1xf32>
    %v5127 = stablehlo.multiply %v5126, %s1b1eW : tensor<768x192x1x1xf32>
    %v5128 = stablehlo.subtract %v5124, %v5127 : tensor<768x192x1x1xf32>
    %v5129 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5130 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5131 = stablehlo.multiply %v5129, %s1b1ebm : tensor<768xf32>
    %v5132 = stablehlo.multiply %v5130, %v2828 : tensor<768xf32>
    %v5133 = stablehlo.add %v5131, %v5132 : tensor<768xf32>
    %v5134 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5135 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5136 = stablehlo.multiply %v5134, %s1b1ebv : tensor<768xf32>
    %v5137 = stablehlo.multiply %v2828, %v2828 : tensor<768xf32>
    %v5138 = stablehlo.multiply %v5135, %v5137 : tensor<768xf32>
    %v5139 = stablehlo.add %v5136, %v5138 : tensor<768xf32>
    %v5140 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5141 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5142 = stablehlo.multiply %v5140, %s1b1ebm : tensor<768xf32>
    %v5143 = stablehlo.multiply %v5141, %v2828 : tensor<768xf32>
    %v5144 = stablehlo.add %v5142, %v5143 : tensor<768xf32>
    %v5145 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5146 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5147 = stablehlo.multiply %v5145, %s1b1ebv : tensor<768xf32>
    %v5148 = stablehlo.multiply %v2828, %v2828 : tensor<768xf32>
    %v5149 = stablehlo.multiply %v5146, %v5148 : tensor<768xf32>
    %v5150 = stablehlo.add %v5147, %v5149 : tensor<768xf32>
    %v5151 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5152 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5153 = stablehlo.divide %v5144, %v5151 : tensor<768xf32>
    %v5154 = stablehlo.divide %v5150, %v5152 : tensor<768xf32>
    %v5155 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5156 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5157 = stablehlo.sqrt %v5154 : tensor<768xf32>
    %v5158 = stablehlo.add %v5157, %v5156 : tensor<768xf32>
    %v5159 = stablehlo.divide %v5153, %v5158 : tensor<768xf32>
    %v5160 = stablehlo.multiply %v5155, %v5159 : tensor<768xf32>
    %v5161 = stablehlo.subtract %s1b1eb, %v5160 : tensor<768xf32>
    %v5162 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5163 = stablehlo.multiply %v5162, %v5155 : tensor<768xf32>
    %v5164 = stablehlo.multiply %v5163, %s1b1eb : tensor<768xf32>
    %v5165 = stablehlo.subtract %v5161, %v5164 : tensor<768xf32>
    %v5166 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5167 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5168 = stablehlo.multiply %v5166, %s1b1pWm : tensor<192x768x1x1xf32>
    %v5169 = stablehlo.multiply %v5167, %v2816 : tensor<192x768x1x1xf32>
    %v5170 = stablehlo.add %v5168, %v5169 : tensor<192x768x1x1xf32>
    %v5171 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5172 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5173 = stablehlo.multiply %v5171, %s1b1pWv : tensor<192x768x1x1xf32>
    %v5174 = stablehlo.multiply %v2816, %v2816 : tensor<192x768x1x1xf32>
    %v5175 = stablehlo.multiply %v5172, %v5174 : tensor<192x768x1x1xf32>
    %v5176 = stablehlo.add %v5173, %v5175 : tensor<192x768x1x1xf32>
    %v5177 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5178 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5179 = stablehlo.multiply %v5177, %s1b1pWm : tensor<192x768x1x1xf32>
    %v5180 = stablehlo.multiply %v5178, %v2816 : tensor<192x768x1x1xf32>
    %v5181 = stablehlo.add %v5179, %v5180 : tensor<192x768x1x1xf32>
    %v5182 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5183 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5184 = stablehlo.multiply %v5182, %s1b1pWv : tensor<192x768x1x1xf32>
    %v5185 = stablehlo.multiply %v2816, %v2816 : tensor<192x768x1x1xf32>
    %v5186 = stablehlo.multiply %v5183, %v5185 : tensor<192x768x1x1xf32>
    %v5187 = stablehlo.add %v5184, %v5186 : tensor<192x768x1x1xf32>
    %v5188 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5189 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5190 = stablehlo.divide %v5181, %v5188 : tensor<192x768x1x1xf32>
    %v5191 = stablehlo.divide %v5187, %v5189 : tensor<192x768x1x1xf32>
    %v5192 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5193 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5194 = stablehlo.sqrt %v5191 : tensor<192x768x1x1xf32>
    %v5195 = stablehlo.add %v5194, %v5193 : tensor<192x768x1x1xf32>
    %v5196 = stablehlo.divide %v5190, %v5195 : tensor<192x768x1x1xf32>
    %v5197 = stablehlo.multiply %v5192, %v5196 : tensor<192x768x1x1xf32>
    %v5198 = stablehlo.subtract %s1b1pW, %v5197 : tensor<192x768x1x1xf32>
    %v5199 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5200 = stablehlo.multiply %v5199, %v5192 : tensor<192x768x1x1xf32>
    %v5201 = stablehlo.multiply %v5200, %s1b1pW : tensor<192x768x1x1xf32>
    %v5202 = stablehlo.subtract %v5198, %v5201 : tensor<192x768x1x1xf32>
    %v5203 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5204 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5205 = stablehlo.multiply %v5203, %s1b1pbm : tensor<192xf32>
    %v5206 = stablehlo.multiply %v5204, %v2819 : tensor<192xf32>
    %v5207 = stablehlo.add %v5205, %v5206 : tensor<192xf32>
    %v5208 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5209 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5210 = stablehlo.multiply %v5208, %s1b1pbv : tensor<192xf32>
    %v5211 = stablehlo.multiply %v2819, %v2819 : tensor<192xf32>
    %v5212 = stablehlo.multiply %v5209, %v5211 : tensor<192xf32>
    %v5213 = stablehlo.add %v5210, %v5212 : tensor<192xf32>
    %v5214 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5215 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5216 = stablehlo.multiply %v5214, %s1b1pbm : tensor<192xf32>
    %v5217 = stablehlo.multiply %v5215, %v2819 : tensor<192xf32>
    %v5218 = stablehlo.add %v5216, %v5217 : tensor<192xf32>
    %v5219 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5220 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5221 = stablehlo.multiply %v5219, %s1b1pbv : tensor<192xf32>
    %v5222 = stablehlo.multiply %v2819, %v2819 : tensor<192xf32>
    %v5223 = stablehlo.multiply %v5220, %v5222 : tensor<192xf32>
    %v5224 = stablehlo.add %v5221, %v5223 : tensor<192xf32>
    %v5225 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5226 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5227 = stablehlo.divide %v5218, %v5225 : tensor<192xf32>
    %v5228 = stablehlo.divide %v5224, %v5226 : tensor<192xf32>
    %v5229 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5230 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5231 = stablehlo.sqrt %v5228 : tensor<192xf32>
    %v5232 = stablehlo.add %v5231, %v5230 : tensor<192xf32>
    %v5233 = stablehlo.divide %v5227, %v5232 : tensor<192xf32>
    %v5234 = stablehlo.multiply %v5229, %v5233 : tensor<192xf32>
    %v5235 = stablehlo.subtract %s1b1pb, %v5234 : tensor<192xf32>
    %v5236 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5237 = stablehlo.multiply %v5236, %v5229 : tensor<192xf32>
    %v5238 = stablehlo.multiply %v5237, %s1b1pb : tensor<192xf32>
    %v5239 = stablehlo.subtract %v5235, %v5238 : tensor<192xf32>
    %v5240 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5241 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5242 = stablehlo.multiply %v5240, %s1b1lgm : tensor<192xf32>
    %v5243 = stablehlo.multiply %v5241, %v2810 : tensor<192xf32>
    %v5244 = stablehlo.add %v5242, %v5243 : tensor<192xf32>
    %v5245 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5246 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5247 = stablehlo.multiply %v5245, %s1b1lgv : tensor<192xf32>
    %v5248 = stablehlo.multiply %v2810, %v2810 : tensor<192xf32>
    %v5249 = stablehlo.multiply %v5246, %v5248 : tensor<192xf32>
    %v5250 = stablehlo.add %v5247, %v5249 : tensor<192xf32>
    %v5251 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5252 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5253 = stablehlo.multiply %v5251, %s1b1lgm : tensor<192xf32>
    %v5254 = stablehlo.multiply %v5252, %v2810 : tensor<192xf32>
    %v5255 = stablehlo.add %v5253, %v5254 : tensor<192xf32>
    %v5256 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5257 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5258 = stablehlo.multiply %v5256, %s1b1lgv : tensor<192xf32>
    %v5259 = stablehlo.multiply %v2810, %v2810 : tensor<192xf32>
    %v5260 = stablehlo.multiply %v5257, %v5259 : tensor<192xf32>
    %v5261 = stablehlo.add %v5258, %v5260 : tensor<192xf32>
    %v5262 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5263 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5264 = stablehlo.divide %v5255, %v5262 : tensor<192xf32>
    %v5265 = stablehlo.divide %v5261, %v5263 : tensor<192xf32>
    %v5266 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5267 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5268 = stablehlo.sqrt %v5265 : tensor<192xf32>
    %v5269 = stablehlo.add %v5268, %v5267 : tensor<192xf32>
    %v5270 = stablehlo.divide %v5264, %v5269 : tensor<192xf32>
    %v5271 = stablehlo.multiply %v5266, %v5270 : tensor<192xf32>
    %v5272 = stablehlo.subtract %s1b1lg, %v5271 : tensor<192xf32>
    %v5273 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5274 = stablehlo.multiply %v5273, %v5266 : tensor<192xf32>
    %v5275 = stablehlo.multiply %v5274, %s1b1lg : tensor<192xf32>
    %v5276 = stablehlo.subtract %v5272, %v5275 : tensor<192xf32>
    %v5277 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5278 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5279 = stablehlo.multiply %v5277, %s1b2dWm : tensor<192x1x7x7xf32>
    %v5280 = stablehlo.multiply %v5278, %v2733 : tensor<192x1x7x7xf32>
    %v5281 = stablehlo.add %v5279, %v5280 : tensor<192x1x7x7xf32>
    %v5282 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5283 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5284 = stablehlo.multiply %v5282, %s1b2dWv : tensor<192x1x7x7xf32>
    %v5285 = stablehlo.multiply %v2733, %v2733 : tensor<192x1x7x7xf32>
    %v5286 = stablehlo.multiply %v5283, %v5285 : tensor<192x1x7x7xf32>
    %v5287 = stablehlo.add %v5284, %v5286 : tensor<192x1x7x7xf32>
    %v5288 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5289 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5290 = stablehlo.multiply %v5288, %s1b2dWm : tensor<192x1x7x7xf32>
    %v5291 = stablehlo.multiply %v5289, %v2733 : tensor<192x1x7x7xf32>
    %v5292 = stablehlo.add %v5290, %v5291 : tensor<192x1x7x7xf32>
    %v5293 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5294 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5295 = stablehlo.multiply %v5293, %s1b2dWv : tensor<192x1x7x7xf32>
    %v5296 = stablehlo.multiply %v2733, %v2733 : tensor<192x1x7x7xf32>
    %v5297 = stablehlo.multiply %v5294, %v5296 : tensor<192x1x7x7xf32>
    %v5298 = stablehlo.add %v5295, %v5297 : tensor<192x1x7x7xf32>
    %v5299 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5300 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5301 = stablehlo.divide %v5292, %v5299 : tensor<192x1x7x7xf32>
    %v5302 = stablehlo.divide %v5298, %v5300 : tensor<192x1x7x7xf32>
    %v5303 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5304 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5305 = stablehlo.sqrt %v5302 : tensor<192x1x7x7xf32>
    %v5306 = stablehlo.add %v5305, %v5304 : tensor<192x1x7x7xf32>
    %v5307 = stablehlo.divide %v5301, %v5306 : tensor<192x1x7x7xf32>
    %v5308 = stablehlo.multiply %v5303, %v5307 : tensor<192x1x7x7xf32>
    %v5309 = stablehlo.subtract %s1b2dW, %v5308 : tensor<192x1x7x7xf32>
    %v5310 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5311 = stablehlo.multiply %v5310, %v5303 : tensor<192x1x7x7xf32>
    %v5312 = stablehlo.multiply %v5311, %s1b2dW : tensor<192x1x7x7xf32>
    %v5313 = stablehlo.subtract %v5309, %v5312 : tensor<192x1x7x7xf32>
    %v5314 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5315 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5316 = stablehlo.multiply %v5314, %s1b2dbm : tensor<192xf32>
    %v5317 = stablehlo.multiply %v5315, %v2736 : tensor<192xf32>
    %v5318 = stablehlo.add %v5316, %v5317 : tensor<192xf32>
    %v5319 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5320 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5321 = stablehlo.multiply %v5319, %s1b2dbv : tensor<192xf32>
    %v5322 = stablehlo.multiply %v2736, %v2736 : tensor<192xf32>
    %v5323 = stablehlo.multiply %v5320, %v5322 : tensor<192xf32>
    %v5324 = stablehlo.add %v5321, %v5323 : tensor<192xf32>
    %v5325 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5326 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5327 = stablehlo.multiply %v5325, %s1b2dbm : tensor<192xf32>
    %v5328 = stablehlo.multiply %v5326, %v2736 : tensor<192xf32>
    %v5329 = stablehlo.add %v5327, %v5328 : tensor<192xf32>
    %v5330 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5331 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5332 = stablehlo.multiply %v5330, %s1b2dbv : tensor<192xf32>
    %v5333 = stablehlo.multiply %v2736, %v2736 : tensor<192xf32>
    %v5334 = stablehlo.multiply %v5331, %v5333 : tensor<192xf32>
    %v5335 = stablehlo.add %v5332, %v5334 : tensor<192xf32>
    %v5336 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5337 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5338 = stablehlo.divide %v5329, %v5336 : tensor<192xf32>
    %v5339 = stablehlo.divide %v5335, %v5337 : tensor<192xf32>
    %v5340 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5341 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5342 = stablehlo.sqrt %v5339 : tensor<192xf32>
    %v5343 = stablehlo.add %v5342, %v5341 : tensor<192xf32>
    %v5344 = stablehlo.divide %v5338, %v5343 : tensor<192xf32>
    %v5345 = stablehlo.multiply %v5340, %v5344 : tensor<192xf32>
    %v5346 = stablehlo.subtract %s1b2db, %v5345 : tensor<192xf32>
    %v5347 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5348 = stablehlo.multiply %v5347, %v5340 : tensor<192xf32>
    %v5349 = stablehlo.multiply %v5348, %s1b2db : tensor<192xf32>
    %v5350 = stablehlo.subtract %v5346, %v5349 : tensor<192xf32>
    %v5351 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5352 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5353 = stablehlo.multiply %v5351, %s1b2ngm : tensor<f32>
    %v5354 = stablehlo.multiply %v5352, %v2725 : tensor<f32>
    %v5355 = stablehlo.add %v5353, %v5354 : tensor<f32>
    %v5356 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5357 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5358 = stablehlo.multiply %v5356, %s1b2ngv : tensor<f32>
    %v5359 = stablehlo.multiply %v2725, %v2725 : tensor<f32>
    %v5360 = stablehlo.multiply %v5357, %v5359 : tensor<f32>
    %v5361 = stablehlo.add %v5358, %v5360 : tensor<f32>
    %v5362 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5363 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5364 = stablehlo.multiply %v5362, %s1b2ngm : tensor<f32>
    %v5365 = stablehlo.multiply %v5363, %v2725 : tensor<f32>
    %v5366 = stablehlo.add %v5364, %v5365 : tensor<f32>
    %v5367 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5368 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5369 = stablehlo.multiply %v5367, %s1b2ngv : tensor<f32>
    %v5370 = stablehlo.multiply %v2725, %v2725 : tensor<f32>
    %v5371 = stablehlo.multiply %v5368, %v5370 : tensor<f32>
    %v5372 = stablehlo.add %v5369, %v5371 : tensor<f32>
    %v5373 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5374 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5375 = stablehlo.divide %v5366, %v5373 : tensor<f32>
    %v5376 = stablehlo.divide %v5372, %v5374 : tensor<f32>
    %v5377 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5378 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5379 = stablehlo.sqrt %v5376 : tensor<f32>
    %v5380 = stablehlo.add %v5379, %v5378 : tensor<f32>
    %v5381 = stablehlo.divide %v5375, %v5380 : tensor<f32>
    %v5382 = stablehlo.multiply %v5377, %v5381 : tensor<f32>
    %v5383 = stablehlo.subtract %s1b2ng, %v5382 : tensor<f32>
    %v5384 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5385 = stablehlo.multiply %v5384, %v5377 : tensor<f32>
    %v5386 = stablehlo.multiply %v5385, %s1b2ng : tensor<f32>
    %v5387 = stablehlo.subtract %v5383, %v5386 : tensor<f32>
    %v5388 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5389 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5390 = stablehlo.multiply %v5388, %s1b2nbtm : tensor<f32>
    %v5391 = stablehlo.multiply %v5389, %v2727 : tensor<f32>
    %v5392 = stablehlo.add %v5390, %v5391 : tensor<f32>
    %v5393 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5394 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5395 = stablehlo.multiply %v5393, %s1b2nbtv : tensor<f32>
    %v5396 = stablehlo.multiply %v2727, %v2727 : tensor<f32>
    %v5397 = stablehlo.multiply %v5394, %v5396 : tensor<f32>
    %v5398 = stablehlo.add %v5395, %v5397 : tensor<f32>
    %v5399 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5400 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5401 = stablehlo.multiply %v5399, %s1b2nbtm : tensor<f32>
    %v5402 = stablehlo.multiply %v5400, %v2727 : tensor<f32>
    %v5403 = stablehlo.add %v5401, %v5402 : tensor<f32>
    %v5404 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5405 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5406 = stablehlo.multiply %v5404, %s1b2nbtv : tensor<f32>
    %v5407 = stablehlo.multiply %v2727, %v2727 : tensor<f32>
    %v5408 = stablehlo.multiply %v5405, %v5407 : tensor<f32>
    %v5409 = stablehlo.add %v5406, %v5408 : tensor<f32>
    %v5410 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5411 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5412 = stablehlo.divide %v5403, %v5410 : tensor<f32>
    %v5413 = stablehlo.divide %v5409, %v5411 : tensor<f32>
    %v5414 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5415 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5416 = stablehlo.sqrt %v5413 : tensor<f32>
    %v5417 = stablehlo.add %v5416, %v5415 : tensor<f32>
    %v5418 = stablehlo.divide %v5412, %v5417 : tensor<f32>
    %v5419 = stablehlo.multiply %v5414, %v5418 : tensor<f32>
    %v5420 = stablehlo.subtract %s1b2nbt, %v5419 : tensor<f32>
    %v5421 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5422 = stablehlo.multiply %v5421, %v5414 : tensor<f32>
    %v5423 = stablehlo.multiply %v5422, %s1b2nbt : tensor<f32>
    %v5424 = stablehlo.subtract %v5420, %v5423 : tensor<f32>
    %v5425 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5426 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5427 = stablehlo.multiply %v5425, %s1b2eWm : tensor<768x192x1x1xf32>
    %v5428 = stablehlo.multiply %v5426, %v2706 : tensor<768x192x1x1xf32>
    %v5429 = stablehlo.add %v5427, %v5428 : tensor<768x192x1x1xf32>
    %v5430 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5431 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5432 = stablehlo.multiply %v5430, %s1b2eWv : tensor<768x192x1x1xf32>
    %v5433 = stablehlo.multiply %v2706, %v2706 : tensor<768x192x1x1xf32>
    %v5434 = stablehlo.multiply %v5431, %v5433 : tensor<768x192x1x1xf32>
    %v5435 = stablehlo.add %v5432, %v5434 : tensor<768x192x1x1xf32>
    %v5436 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5437 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5438 = stablehlo.multiply %v5436, %s1b2eWm : tensor<768x192x1x1xf32>
    %v5439 = stablehlo.multiply %v5437, %v2706 : tensor<768x192x1x1xf32>
    %v5440 = stablehlo.add %v5438, %v5439 : tensor<768x192x1x1xf32>
    %v5441 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5442 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5443 = stablehlo.multiply %v5441, %s1b2eWv : tensor<768x192x1x1xf32>
    %v5444 = stablehlo.multiply %v2706, %v2706 : tensor<768x192x1x1xf32>
    %v5445 = stablehlo.multiply %v5442, %v5444 : tensor<768x192x1x1xf32>
    %v5446 = stablehlo.add %v5443, %v5445 : tensor<768x192x1x1xf32>
    %v5447 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5448 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5449 = stablehlo.divide %v5440, %v5447 : tensor<768x192x1x1xf32>
    %v5450 = stablehlo.divide %v5446, %v5448 : tensor<768x192x1x1xf32>
    %v5451 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5452 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5453 = stablehlo.sqrt %v5450 : tensor<768x192x1x1xf32>
    %v5454 = stablehlo.add %v5453, %v5452 : tensor<768x192x1x1xf32>
    %v5455 = stablehlo.divide %v5449, %v5454 : tensor<768x192x1x1xf32>
    %v5456 = stablehlo.multiply %v5451, %v5455 : tensor<768x192x1x1xf32>
    %v5457 = stablehlo.subtract %s1b2eW, %v5456 : tensor<768x192x1x1xf32>
    %v5458 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5459 = stablehlo.multiply %v5458, %v5451 : tensor<768x192x1x1xf32>
    %v5460 = stablehlo.multiply %v5459, %s1b2eW : tensor<768x192x1x1xf32>
    %v5461 = stablehlo.subtract %v5457, %v5460 : tensor<768x192x1x1xf32>
    %v5462 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5463 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5464 = stablehlo.multiply %v5462, %s1b2ebm : tensor<768xf32>
    %v5465 = stablehlo.multiply %v5463, %v2709 : tensor<768xf32>
    %v5466 = stablehlo.add %v5464, %v5465 : tensor<768xf32>
    %v5467 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5468 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5469 = stablehlo.multiply %v5467, %s1b2ebv : tensor<768xf32>
    %v5470 = stablehlo.multiply %v2709, %v2709 : tensor<768xf32>
    %v5471 = stablehlo.multiply %v5468, %v5470 : tensor<768xf32>
    %v5472 = stablehlo.add %v5469, %v5471 : tensor<768xf32>
    %v5473 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5474 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5475 = stablehlo.multiply %v5473, %s1b2ebm : tensor<768xf32>
    %v5476 = stablehlo.multiply %v5474, %v2709 : tensor<768xf32>
    %v5477 = stablehlo.add %v5475, %v5476 : tensor<768xf32>
    %v5478 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5479 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5480 = stablehlo.multiply %v5478, %s1b2ebv : tensor<768xf32>
    %v5481 = stablehlo.multiply %v2709, %v2709 : tensor<768xf32>
    %v5482 = stablehlo.multiply %v5479, %v5481 : tensor<768xf32>
    %v5483 = stablehlo.add %v5480, %v5482 : tensor<768xf32>
    %v5484 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5485 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5486 = stablehlo.divide %v5477, %v5484 : tensor<768xf32>
    %v5487 = stablehlo.divide %v5483, %v5485 : tensor<768xf32>
    %v5488 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5489 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5490 = stablehlo.sqrt %v5487 : tensor<768xf32>
    %v5491 = stablehlo.add %v5490, %v5489 : tensor<768xf32>
    %v5492 = stablehlo.divide %v5486, %v5491 : tensor<768xf32>
    %v5493 = stablehlo.multiply %v5488, %v5492 : tensor<768xf32>
    %v5494 = stablehlo.subtract %s1b2eb, %v5493 : tensor<768xf32>
    %v5495 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5496 = stablehlo.multiply %v5495, %v5488 : tensor<768xf32>
    %v5497 = stablehlo.multiply %v5496, %s1b2eb : tensor<768xf32>
    %v5498 = stablehlo.subtract %v5494, %v5497 : tensor<768xf32>
    %v5499 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5500 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5501 = stablehlo.multiply %v5499, %s1b2pWm : tensor<192x768x1x1xf32>
    %v5502 = stablehlo.multiply %v5500, %v2697 : tensor<192x768x1x1xf32>
    %v5503 = stablehlo.add %v5501, %v5502 : tensor<192x768x1x1xf32>
    %v5504 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5505 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5506 = stablehlo.multiply %v5504, %s1b2pWv : tensor<192x768x1x1xf32>
    %v5507 = stablehlo.multiply %v2697, %v2697 : tensor<192x768x1x1xf32>
    %v5508 = stablehlo.multiply %v5505, %v5507 : tensor<192x768x1x1xf32>
    %v5509 = stablehlo.add %v5506, %v5508 : tensor<192x768x1x1xf32>
    %v5510 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5511 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5512 = stablehlo.multiply %v5510, %s1b2pWm : tensor<192x768x1x1xf32>
    %v5513 = stablehlo.multiply %v5511, %v2697 : tensor<192x768x1x1xf32>
    %v5514 = stablehlo.add %v5512, %v5513 : tensor<192x768x1x1xf32>
    %v5515 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5516 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5517 = stablehlo.multiply %v5515, %s1b2pWv : tensor<192x768x1x1xf32>
    %v5518 = stablehlo.multiply %v2697, %v2697 : tensor<192x768x1x1xf32>
    %v5519 = stablehlo.multiply %v5516, %v5518 : tensor<192x768x1x1xf32>
    %v5520 = stablehlo.add %v5517, %v5519 : tensor<192x768x1x1xf32>
    %v5521 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5522 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5523 = stablehlo.divide %v5514, %v5521 : tensor<192x768x1x1xf32>
    %v5524 = stablehlo.divide %v5520, %v5522 : tensor<192x768x1x1xf32>
    %v5525 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5526 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5527 = stablehlo.sqrt %v5524 : tensor<192x768x1x1xf32>
    %v5528 = stablehlo.add %v5527, %v5526 : tensor<192x768x1x1xf32>
    %v5529 = stablehlo.divide %v5523, %v5528 : tensor<192x768x1x1xf32>
    %v5530 = stablehlo.multiply %v5525, %v5529 : tensor<192x768x1x1xf32>
    %v5531 = stablehlo.subtract %s1b2pW, %v5530 : tensor<192x768x1x1xf32>
    %v5532 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5533 = stablehlo.multiply %v5532, %v5525 : tensor<192x768x1x1xf32>
    %v5534 = stablehlo.multiply %v5533, %s1b2pW : tensor<192x768x1x1xf32>
    %v5535 = stablehlo.subtract %v5531, %v5534 : tensor<192x768x1x1xf32>
    %v5536 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5537 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5538 = stablehlo.multiply %v5536, %s1b2pbm : tensor<192xf32>
    %v5539 = stablehlo.multiply %v5537, %v2700 : tensor<192xf32>
    %v5540 = stablehlo.add %v5538, %v5539 : tensor<192xf32>
    %v5541 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5542 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5543 = stablehlo.multiply %v5541, %s1b2pbv : tensor<192xf32>
    %v5544 = stablehlo.multiply %v2700, %v2700 : tensor<192xf32>
    %v5545 = stablehlo.multiply %v5542, %v5544 : tensor<192xf32>
    %v5546 = stablehlo.add %v5543, %v5545 : tensor<192xf32>
    %v5547 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5548 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5549 = stablehlo.multiply %v5547, %s1b2pbm : tensor<192xf32>
    %v5550 = stablehlo.multiply %v5548, %v2700 : tensor<192xf32>
    %v5551 = stablehlo.add %v5549, %v5550 : tensor<192xf32>
    %v5552 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5553 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5554 = stablehlo.multiply %v5552, %s1b2pbv : tensor<192xf32>
    %v5555 = stablehlo.multiply %v2700, %v2700 : tensor<192xf32>
    %v5556 = stablehlo.multiply %v5553, %v5555 : tensor<192xf32>
    %v5557 = stablehlo.add %v5554, %v5556 : tensor<192xf32>
    %v5558 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5559 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5560 = stablehlo.divide %v5551, %v5558 : tensor<192xf32>
    %v5561 = stablehlo.divide %v5557, %v5559 : tensor<192xf32>
    %v5562 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5563 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5564 = stablehlo.sqrt %v5561 : tensor<192xf32>
    %v5565 = stablehlo.add %v5564, %v5563 : tensor<192xf32>
    %v5566 = stablehlo.divide %v5560, %v5565 : tensor<192xf32>
    %v5567 = stablehlo.multiply %v5562, %v5566 : tensor<192xf32>
    %v5568 = stablehlo.subtract %s1b2pb, %v5567 : tensor<192xf32>
    %v5569 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5570 = stablehlo.multiply %v5569, %v5562 : tensor<192xf32>
    %v5571 = stablehlo.multiply %v5570, %s1b2pb : tensor<192xf32>
    %v5572 = stablehlo.subtract %v5568, %v5571 : tensor<192xf32>
    %v5573 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5574 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5575 = stablehlo.multiply %v5573, %s1b2lgm : tensor<192xf32>
    %v5576 = stablehlo.multiply %v5574, %v2691 : tensor<192xf32>
    %v5577 = stablehlo.add %v5575, %v5576 : tensor<192xf32>
    %v5578 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5579 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5580 = stablehlo.multiply %v5578, %s1b2lgv : tensor<192xf32>
    %v5581 = stablehlo.multiply %v2691, %v2691 : tensor<192xf32>
    %v5582 = stablehlo.multiply %v5579, %v5581 : tensor<192xf32>
    %v5583 = stablehlo.add %v5580, %v5582 : tensor<192xf32>
    %v5584 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5585 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5586 = stablehlo.multiply %v5584, %s1b2lgm : tensor<192xf32>
    %v5587 = stablehlo.multiply %v5585, %v2691 : tensor<192xf32>
    %v5588 = stablehlo.add %v5586, %v5587 : tensor<192xf32>
    %v5589 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5590 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5591 = stablehlo.multiply %v5589, %s1b2lgv : tensor<192xf32>
    %v5592 = stablehlo.multiply %v2691, %v2691 : tensor<192xf32>
    %v5593 = stablehlo.multiply %v5590, %v5592 : tensor<192xf32>
    %v5594 = stablehlo.add %v5591, %v5593 : tensor<192xf32>
    %v5595 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5596 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5597 = stablehlo.divide %v5588, %v5595 : tensor<192xf32>
    %v5598 = stablehlo.divide %v5594, %v5596 : tensor<192xf32>
    %v5599 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5600 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5601 = stablehlo.sqrt %v5598 : tensor<192xf32>
    %v5602 = stablehlo.add %v5601, %v5600 : tensor<192xf32>
    %v5603 = stablehlo.divide %v5597, %v5602 : tensor<192xf32>
    %v5604 = stablehlo.multiply %v5599, %v5603 : tensor<192xf32>
    %v5605 = stablehlo.subtract %s1b2lg, %v5604 : tensor<192xf32>
    %v5606 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5607 = stablehlo.multiply %v5606, %v5599 : tensor<192xf32>
    %v5608 = stablehlo.multiply %v5607, %s1b2lg : tensor<192xf32>
    %v5609 = stablehlo.subtract %v5605, %v5608 : tensor<192xf32>
    %v5610 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5611 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5612 = stablehlo.multiply %v5610, %d1ngm : tensor<f32>
    %v5613 = stablehlo.multiply %v5611, %v2615 : tensor<f32>
    %v5614 = stablehlo.add %v5612, %v5613 : tensor<f32>
    %v5615 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5616 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5617 = stablehlo.multiply %v5615, %d1ngv : tensor<f32>
    %v5618 = stablehlo.multiply %v2615, %v2615 : tensor<f32>
    %v5619 = stablehlo.multiply %v5616, %v5618 : tensor<f32>
    %v5620 = stablehlo.add %v5617, %v5619 : tensor<f32>
    %v5621 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5622 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5623 = stablehlo.multiply %v5621, %d1ngm : tensor<f32>
    %v5624 = stablehlo.multiply %v5622, %v2615 : tensor<f32>
    %v5625 = stablehlo.add %v5623, %v5624 : tensor<f32>
    %v5626 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5627 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5628 = stablehlo.multiply %v5626, %d1ngv : tensor<f32>
    %v5629 = stablehlo.multiply %v2615, %v2615 : tensor<f32>
    %v5630 = stablehlo.multiply %v5627, %v5629 : tensor<f32>
    %v5631 = stablehlo.add %v5628, %v5630 : tensor<f32>
    %v5632 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5633 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5634 = stablehlo.divide %v5625, %v5632 : tensor<f32>
    %v5635 = stablehlo.divide %v5631, %v5633 : tensor<f32>
    %v5636 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5637 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5638 = stablehlo.sqrt %v5635 : tensor<f32>
    %v5639 = stablehlo.add %v5638, %v5637 : tensor<f32>
    %v5640 = stablehlo.divide %v5634, %v5639 : tensor<f32>
    %v5641 = stablehlo.multiply %v5636, %v5640 : tensor<f32>
    %v5642 = stablehlo.subtract %d1ng, %v5641 : tensor<f32>
    %v5643 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5644 = stablehlo.multiply %v5643, %v5636 : tensor<f32>
    %v5645 = stablehlo.multiply %v5644, %d1ng : tensor<f32>
    %v5646 = stablehlo.subtract %v5642, %v5645 : tensor<f32>
    %v5647 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5648 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5649 = stablehlo.multiply %v5647, %d1nbtm : tensor<f32>
    %v5650 = stablehlo.multiply %v5648, %v2617 : tensor<f32>
    %v5651 = stablehlo.add %v5649, %v5650 : tensor<f32>
    %v5652 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5653 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5654 = stablehlo.multiply %v5652, %d1nbtv : tensor<f32>
    %v5655 = stablehlo.multiply %v2617, %v2617 : tensor<f32>
    %v5656 = stablehlo.multiply %v5653, %v5655 : tensor<f32>
    %v5657 = stablehlo.add %v5654, %v5656 : tensor<f32>
    %v5658 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5659 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5660 = stablehlo.multiply %v5658, %d1nbtm : tensor<f32>
    %v5661 = stablehlo.multiply %v5659, %v2617 : tensor<f32>
    %v5662 = stablehlo.add %v5660, %v5661 : tensor<f32>
    %v5663 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5664 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5665 = stablehlo.multiply %v5663, %d1nbtv : tensor<f32>
    %v5666 = stablehlo.multiply %v2617, %v2617 : tensor<f32>
    %v5667 = stablehlo.multiply %v5664, %v5666 : tensor<f32>
    %v5668 = stablehlo.add %v5665, %v5667 : tensor<f32>
    %v5669 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5670 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5671 = stablehlo.divide %v5662, %v5669 : tensor<f32>
    %v5672 = stablehlo.divide %v5668, %v5670 : tensor<f32>
    %v5673 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5674 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5675 = stablehlo.sqrt %v5672 : tensor<f32>
    %v5676 = stablehlo.add %v5675, %v5674 : tensor<f32>
    %v5677 = stablehlo.divide %v5671, %v5676 : tensor<f32>
    %v5678 = stablehlo.multiply %v5673, %v5677 : tensor<f32>
    %v5679 = stablehlo.subtract %d1nbt, %v5678 : tensor<f32>
    %v5680 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5681 = stablehlo.multiply %v5680, %v5673 : tensor<f32>
    %v5682 = stablehlo.multiply %v5681, %d1nbt : tensor<f32>
    %v5683 = stablehlo.subtract %v5679, %v5682 : tensor<f32>
    %v5684 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5685 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5686 = stablehlo.multiply %v5684, %d1Wm : tensor<384x192x2x2xf32>
    %v5687 = stablehlo.multiply %v5685, %dd1W : tensor<384x192x2x2xf32>
    %v5688 = stablehlo.add %v5686, %v5687 : tensor<384x192x2x2xf32>
    %v5689 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5690 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5691 = stablehlo.multiply %v5689, %d1Wv : tensor<384x192x2x2xf32>
    %v5692 = stablehlo.multiply %dd1W, %dd1W : tensor<384x192x2x2xf32>
    %v5693 = stablehlo.multiply %v5690, %v5692 : tensor<384x192x2x2xf32>
    %v5694 = stablehlo.add %v5691, %v5693 : tensor<384x192x2x2xf32>
    %v5695 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5696 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5697 = stablehlo.multiply %v5695, %d1Wm : tensor<384x192x2x2xf32>
    %v5698 = stablehlo.multiply %v5696, %dd1W : tensor<384x192x2x2xf32>
    %v5699 = stablehlo.add %v5697, %v5698 : tensor<384x192x2x2xf32>
    %v5700 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5701 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5702 = stablehlo.multiply %v5700, %d1Wv : tensor<384x192x2x2xf32>
    %v5703 = stablehlo.multiply %dd1W, %dd1W : tensor<384x192x2x2xf32>
    %v5704 = stablehlo.multiply %v5701, %v5703 : tensor<384x192x2x2xf32>
    %v5705 = stablehlo.add %v5702, %v5704 : tensor<384x192x2x2xf32>
    %v5706 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5707 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5708 = stablehlo.divide %v5699, %v5706 : tensor<384x192x2x2xf32>
    %v5709 = stablehlo.divide %v5705, %v5707 : tensor<384x192x2x2xf32>
    %v5710 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5711 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5712 = stablehlo.sqrt %v5709 : tensor<384x192x2x2xf32>
    %v5713 = stablehlo.add %v5712, %v5711 : tensor<384x192x2x2xf32>
    %v5714 = stablehlo.divide %v5708, %v5713 : tensor<384x192x2x2xf32>
    %v5715 = stablehlo.multiply %v5710, %v5714 : tensor<384x192x2x2xf32>
    %v5716 = stablehlo.subtract %d1W, %v5715 : tensor<384x192x2x2xf32>
    %v5717 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v5718 = stablehlo.multiply %v5717, %v5710 : tensor<384x192x2x2xf32>
    %v5719 = stablehlo.multiply %v5718, %d1W : tensor<384x192x2x2xf32>
    %v5720 = stablehlo.subtract %v5716, %v5719 : tensor<384x192x2x2xf32>
    %v5721 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5722 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5723 = stablehlo.multiply %v5721, %d1bm : tensor<384xf32>
    %v5724 = stablehlo.multiply %v5722, %v2599 : tensor<384xf32>
    %v5725 = stablehlo.add %v5723, %v5724 : tensor<384xf32>
    %v5726 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5727 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5728 = stablehlo.multiply %v5726, %d1bv : tensor<384xf32>
    %v5729 = stablehlo.multiply %v2599, %v2599 : tensor<384xf32>
    %v5730 = stablehlo.multiply %v5727, %v5729 : tensor<384xf32>
    %v5731 = stablehlo.add %v5728, %v5730 : tensor<384xf32>
    %v5732 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5733 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5734 = stablehlo.multiply %v5732, %d1bm : tensor<384xf32>
    %v5735 = stablehlo.multiply %v5733, %v2599 : tensor<384xf32>
    %v5736 = stablehlo.add %v5734, %v5735 : tensor<384xf32>
    %v5737 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5738 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5739 = stablehlo.multiply %v5737, %d1bv : tensor<384xf32>
    %v5740 = stablehlo.multiply %v2599, %v2599 : tensor<384xf32>
    %v5741 = stablehlo.multiply %v5738, %v5740 : tensor<384xf32>
    %v5742 = stablehlo.add %v5739, %v5741 : tensor<384xf32>
    %v5743 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5744 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5745 = stablehlo.divide %v5736, %v5743 : tensor<384xf32>
    %v5746 = stablehlo.divide %v5742, %v5744 : tensor<384xf32>
    %v5747 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5748 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5749 = stablehlo.sqrt %v5746 : tensor<384xf32>
    %v5750 = stablehlo.add %v5749, %v5748 : tensor<384xf32>
    %v5751 = stablehlo.divide %v5745, %v5750 : tensor<384xf32>
    %v5752 = stablehlo.multiply %v5747, %v5751 : tensor<384xf32>
    %v5753 = stablehlo.subtract %d1b, %v5752 : tensor<384xf32>
    %v5754 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5755 = stablehlo.multiply %v5754, %v5747 : tensor<384xf32>
    %v5756 = stablehlo.multiply %v5755, %d1b : tensor<384xf32>
    %v5757 = stablehlo.subtract %v5753, %v5756 : tensor<384xf32>
    %v5758 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5759 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5760 = stablehlo.multiply %v5758, %s2b0dWm : tensor<384x1x7x7xf32>
    %v5761 = stablehlo.multiply %v5759, %v2559 : tensor<384x1x7x7xf32>
    %v5762 = stablehlo.add %v5760, %v5761 : tensor<384x1x7x7xf32>
    %v5763 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5764 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5765 = stablehlo.multiply %v5763, %s2b0dWv : tensor<384x1x7x7xf32>
    %v5766 = stablehlo.multiply %v2559, %v2559 : tensor<384x1x7x7xf32>
    %v5767 = stablehlo.multiply %v5764, %v5766 : tensor<384x1x7x7xf32>
    %v5768 = stablehlo.add %v5765, %v5767 : tensor<384x1x7x7xf32>
    %v5769 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5770 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5771 = stablehlo.multiply %v5769, %s2b0dWm : tensor<384x1x7x7xf32>
    %v5772 = stablehlo.multiply %v5770, %v2559 : tensor<384x1x7x7xf32>
    %v5773 = stablehlo.add %v5771, %v5772 : tensor<384x1x7x7xf32>
    %v5774 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5775 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5776 = stablehlo.multiply %v5774, %s2b0dWv : tensor<384x1x7x7xf32>
    %v5777 = stablehlo.multiply %v2559, %v2559 : tensor<384x1x7x7xf32>
    %v5778 = stablehlo.multiply %v5775, %v5777 : tensor<384x1x7x7xf32>
    %v5779 = stablehlo.add %v5776, %v5778 : tensor<384x1x7x7xf32>
    %v5780 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5781 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5782 = stablehlo.divide %v5773, %v5780 : tensor<384x1x7x7xf32>
    %v5783 = stablehlo.divide %v5779, %v5781 : tensor<384x1x7x7xf32>
    %v5784 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5785 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5786 = stablehlo.sqrt %v5783 : tensor<384x1x7x7xf32>
    %v5787 = stablehlo.add %v5786, %v5785 : tensor<384x1x7x7xf32>
    %v5788 = stablehlo.divide %v5782, %v5787 : tensor<384x1x7x7xf32>
    %v5789 = stablehlo.multiply %v5784, %v5788 : tensor<384x1x7x7xf32>
    %v5790 = stablehlo.subtract %s2b0dW, %v5789 : tensor<384x1x7x7xf32>
    %v5791 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v5792 = stablehlo.multiply %v5791, %v5784 : tensor<384x1x7x7xf32>
    %v5793 = stablehlo.multiply %v5792, %s2b0dW : tensor<384x1x7x7xf32>
    %v5794 = stablehlo.subtract %v5790, %v5793 : tensor<384x1x7x7xf32>
    %v5795 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5796 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5797 = stablehlo.multiply %v5795, %s2b0dbm : tensor<384xf32>
    %v5798 = stablehlo.multiply %v5796, %v2562 : tensor<384xf32>
    %v5799 = stablehlo.add %v5797, %v5798 : tensor<384xf32>
    %v5800 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5801 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5802 = stablehlo.multiply %v5800, %s2b0dbv : tensor<384xf32>
    %v5803 = stablehlo.multiply %v2562, %v2562 : tensor<384xf32>
    %v5804 = stablehlo.multiply %v5801, %v5803 : tensor<384xf32>
    %v5805 = stablehlo.add %v5802, %v5804 : tensor<384xf32>
    %v5806 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5807 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5808 = stablehlo.multiply %v5806, %s2b0dbm : tensor<384xf32>
    %v5809 = stablehlo.multiply %v5807, %v2562 : tensor<384xf32>
    %v5810 = stablehlo.add %v5808, %v5809 : tensor<384xf32>
    %v5811 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5812 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5813 = stablehlo.multiply %v5811, %s2b0dbv : tensor<384xf32>
    %v5814 = stablehlo.multiply %v2562, %v2562 : tensor<384xf32>
    %v5815 = stablehlo.multiply %v5812, %v5814 : tensor<384xf32>
    %v5816 = stablehlo.add %v5813, %v5815 : tensor<384xf32>
    %v5817 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5818 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5819 = stablehlo.divide %v5810, %v5817 : tensor<384xf32>
    %v5820 = stablehlo.divide %v5816, %v5818 : tensor<384xf32>
    %v5821 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5822 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5823 = stablehlo.sqrt %v5820 : tensor<384xf32>
    %v5824 = stablehlo.add %v5823, %v5822 : tensor<384xf32>
    %v5825 = stablehlo.divide %v5819, %v5824 : tensor<384xf32>
    %v5826 = stablehlo.multiply %v5821, %v5825 : tensor<384xf32>
    %v5827 = stablehlo.subtract %s2b0db, %v5826 : tensor<384xf32>
    %v5828 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5829 = stablehlo.multiply %v5828, %v5821 : tensor<384xf32>
    %v5830 = stablehlo.multiply %v5829, %s2b0db : tensor<384xf32>
    %v5831 = stablehlo.subtract %v5827, %v5830 : tensor<384xf32>
    %v5832 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5833 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5834 = stablehlo.multiply %v5832, %s2b0ngm : tensor<f32>
    %v5835 = stablehlo.multiply %v5833, %v2551 : tensor<f32>
    %v5836 = stablehlo.add %v5834, %v5835 : tensor<f32>
    %v5837 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5838 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5839 = stablehlo.multiply %v5837, %s2b0ngv : tensor<f32>
    %v5840 = stablehlo.multiply %v2551, %v2551 : tensor<f32>
    %v5841 = stablehlo.multiply %v5838, %v5840 : tensor<f32>
    %v5842 = stablehlo.add %v5839, %v5841 : tensor<f32>
    %v5843 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5844 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5845 = stablehlo.multiply %v5843, %s2b0ngm : tensor<f32>
    %v5846 = stablehlo.multiply %v5844, %v2551 : tensor<f32>
    %v5847 = stablehlo.add %v5845, %v5846 : tensor<f32>
    %v5848 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5849 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5850 = stablehlo.multiply %v5848, %s2b0ngv : tensor<f32>
    %v5851 = stablehlo.multiply %v2551, %v2551 : tensor<f32>
    %v5852 = stablehlo.multiply %v5849, %v5851 : tensor<f32>
    %v5853 = stablehlo.add %v5850, %v5852 : tensor<f32>
    %v5854 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5855 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5856 = stablehlo.divide %v5847, %v5854 : tensor<f32>
    %v5857 = stablehlo.divide %v5853, %v5855 : tensor<f32>
    %v5858 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5859 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5860 = stablehlo.sqrt %v5857 : tensor<f32>
    %v5861 = stablehlo.add %v5860, %v5859 : tensor<f32>
    %v5862 = stablehlo.divide %v5856, %v5861 : tensor<f32>
    %v5863 = stablehlo.multiply %v5858, %v5862 : tensor<f32>
    %v5864 = stablehlo.subtract %s2b0ng, %v5863 : tensor<f32>
    %v5865 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5866 = stablehlo.multiply %v5865, %v5858 : tensor<f32>
    %v5867 = stablehlo.multiply %v5866, %s2b0ng : tensor<f32>
    %v5868 = stablehlo.subtract %v5864, %v5867 : tensor<f32>
    %v5869 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5870 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5871 = stablehlo.multiply %v5869, %s2b0nbtm : tensor<f32>
    %v5872 = stablehlo.multiply %v5870, %v2553 : tensor<f32>
    %v5873 = stablehlo.add %v5871, %v5872 : tensor<f32>
    %v5874 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5875 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5876 = stablehlo.multiply %v5874, %s2b0nbtv : tensor<f32>
    %v5877 = stablehlo.multiply %v2553, %v2553 : tensor<f32>
    %v5878 = stablehlo.multiply %v5875, %v5877 : tensor<f32>
    %v5879 = stablehlo.add %v5876, %v5878 : tensor<f32>
    %v5880 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5881 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5882 = stablehlo.multiply %v5880, %s2b0nbtm : tensor<f32>
    %v5883 = stablehlo.multiply %v5881, %v2553 : tensor<f32>
    %v5884 = stablehlo.add %v5882, %v5883 : tensor<f32>
    %v5885 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5886 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5887 = stablehlo.multiply %v5885, %s2b0nbtv : tensor<f32>
    %v5888 = stablehlo.multiply %v2553, %v2553 : tensor<f32>
    %v5889 = stablehlo.multiply %v5886, %v5888 : tensor<f32>
    %v5890 = stablehlo.add %v5887, %v5889 : tensor<f32>
    %v5891 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5892 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5893 = stablehlo.divide %v5884, %v5891 : tensor<f32>
    %v5894 = stablehlo.divide %v5890, %v5892 : tensor<f32>
    %v5895 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5896 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5897 = stablehlo.sqrt %v5894 : tensor<f32>
    %v5898 = stablehlo.add %v5897, %v5896 : tensor<f32>
    %v5899 = stablehlo.divide %v5893, %v5898 : tensor<f32>
    %v5900 = stablehlo.multiply %v5895, %v5899 : tensor<f32>
    %v5901 = stablehlo.subtract %s2b0nbt, %v5900 : tensor<f32>
    %v5902 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v5903 = stablehlo.multiply %v5902, %v5895 : tensor<f32>
    %v5904 = stablehlo.multiply %v5903, %s2b0nbt : tensor<f32>
    %v5905 = stablehlo.subtract %v5901, %v5904 : tensor<f32>
    %v5906 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5907 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5908 = stablehlo.multiply %v5906, %s2b0eWm : tensor<1536x384x1x1xf32>
    %v5909 = stablehlo.multiply %v5907, %v2532 : tensor<1536x384x1x1xf32>
    %v5910 = stablehlo.add %v5908, %v5909 : tensor<1536x384x1x1xf32>
    %v5911 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5912 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5913 = stablehlo.multiply %v5911, %s2b0eWv : tensor<1536x384x1x1xf32>
    %v5914 = stablehlo.multiply %v2532, %v2532 : tensor<1536x384x1x1xf32>
    %v5915 = stablehlo.multiply %v5912, %v5914 : tensor<1536x384x1x1xf32>
    %v5916 = stablehlo.add %v5913, %v5915 : tensor<1536x384x1x1xf32>
    %v5917 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5918 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5919 = stablehlo.multiply %v5917, %s2b0eWm : tensor<1536x384x1x1xf32>
    %v5920 = stablehlo.multiply %v5918, %v2532 : tensor<1536x384x1x1xf32>
    %v5921 = stablehlo.add %v5919, %v5920 : tensor<1536x384x1x1xf32>
    %v5922 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5923 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5924 = stablehlo.multiply %v5922, %s2b0eWv : tensor<1536x384x1x1xf32>
    %v5925 = stablehlo.multiply %v2532, %v2532 : tensor<1536x384x1x1xf32>
    %v5926 = stablehlo.multiply %v5923, %v5925 : tensor<1536x384x1x1xf32>
    %v5927 = stablehlo.add %v5924, %v5926 : tensor<1536x384x1x1xf32>
    %v5928 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5929 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5930 = stablehlo.divide %v5921, %v5928 : tensor<1536x384x1x1xf32>
    %v5931 = stablehlo.divide %v5927, %v5929 : tensor<1536x384x1x1xf32>
    %v5932 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5933 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5934 = stablehlo.sqrt %v5931 : tensor<1536x384x1x1xf32>
    %v5935 = stablehlo.add %v5934, %v5933 : tensor<1536x384x1x1xf32>
    %v5936 = stablehlo.divide %v5930, %v5935 : tensor<1536x384x1x1xf32>
    %v5937 = stablehlo.multiply %v5932, %v5936 : tensor<1536x384x1x1xf32>
    %v5938 = stablehlo.subtract %s2b0eW, %v5937 : tensor<1536x384x1x1xf32>
    %v5939 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v5940 = stablehlo.multiply %v5939, %v5932 : tensor<1536x384x1x1xf32>
    %v5941 = stablehlo.multiply %v5940, %s2b0eW : tensor<1536x384x1x1xf32>
    %v5942 = stablehlo.subtract %v5938, %v5941 : tensor<1536x384x1x1xf32>
    %v5943 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5944 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5945 = stablehlo.multiply %v5943, %s2b0ebm : tensor<1536xf32>
    %v5946 = stablehlo.multiply %v5944, %v2535 : tensor<1536xf32>
    %v5947 = stablehlo.add %v5945, %v5946 : tensor<1536xf32>
    %v5948 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5949 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5950 = stablehlo.multiply %v5948, %s2b0ebv : tensor<1536xf32>
    %v5951 = stablehlo.multiply %v2535, %v2535 : tensor<1536xf32>
    %v5952 = stablehlo.multiply %v5949, %v5951 : tensor<1536xf32>
    %v5953 = stablehlo.add %v5950, %v5952 : tensor<1536xf32>
    %v5954 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5955 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5956 = stablehlo.multiply %v5954, %s2b0ebm : tensor<1536xf32>
    %v5957 = stablehlo.multiply %v5955, %v2535 : tensor<1536xf32>
    %v5958 = stablehlo.add %v5956, %v5957 : tensor<1536xf32>
    %v5959 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5960 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5961 = stablehlo.multiply %v5959, %s2b0ebv : tensor<1536xf32>
    %v5962 = stablehlo.multiply %v2535, %v2535 : tensor<1536xf32>
    %v5963 = stablehlo.multiply %v5960, %v5962 : tensor<1536xf32>
    %v5964 = stablehlo.add %v5961, %v5963 : tensor<1536xf32>
    %v5965 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5966 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5967 = stablehlo.divide %v5958, %v5965 : tensor<1536xf32>
    %v5968 = stablehlo.divide %v5964, %v5966 : tensor<1536xf32>
    %v5969 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5970 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5971 = stablehlo.sqrt %v5968 : tensor<1536xf32>
    %v5972 = stablehlo.add %v5971, %v5970 : tensor<1536xf32>
    %v5973 = stablehlo.divide %v5967, %v5972 : tensor<1536xf32>
    %v5974 = stablehlo.multiply %v5969, %v5973 : tensor<1536xf32>
    %v5975 = stablehlo.subtract %s2b0eb, %v5974 : tensor<1536xf32>
    %v5976 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v5977 = stablehlo.multiply %v5976, %v5969 : tensor<1536xf32>
    %v5978 = stablehlo.multiply %v5977, %s2b0eb : tensor<1536xf32>
    %v5979 = stablehlo.subtract %v5975, %v5978 : tensor<1536xf32>
    %v5980 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v5981 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v5982 = stablehlo.multiply %v5980, %s2b0pWm : tensor<384x1536x1x1xf32>
    %v5983 = stablehlo.multiply %v5981, %v2523 : tensor<384x1536x1x1xf32>
    %v5984 = stablehlo.add %v5982, %v5983 : tensor<384x1536x1x1xf32>
    %v5985 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v5986 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v5987 = stablehlo.multiply %v5985, %s2b0pWv : tensor<384x1536x1x1xf32>
    %v5988 = stablehlo.multiply %v2523, %v2523 : tensor<384x1536x1x1xf32>
    %v5989 = stablehlo.multiply %v5986, %v5988 : tensor<384x1536x1x1xf32>
    %v5990 = stablehlo.add %v5987, %v5989 : tensor<384x1536x1x1xf32>
    %v5991 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v5992 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v5993 = stablehlo.multiply %v5991, %s2b0pWm : tensor<384x1536x1x1xf32>
    %v5994 = stablehlo.multiply %v5992, %v2523 : tensor<384x1536x1x1xf32>
    %v5995 = stablehlo.add %v5993, %v5994 : tensor<384x1536x1x1xf32>
    %v5996 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v5997 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v5998 = stablehlo.multiply %v5996, %s2b0pWv : tensor<384x1536x1x1xf32>
    %v5999 = stablehlo.multiply %v2523, %v2523 : tensor<384x1536x1x1xf32>
    %v6000 = stablehlo.multiply %v5997, %v5999 : tensor<384x1536x1x1xf32>
    %v6001 = stablehlo.add %v5998, %v6000 : tensor<384x1536x1x1xf32>
    %v6002 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6003 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6004 = stablehlo.divide %v5995, %v6002 : tensor<384x1536x1x1xf32>
    %v6005 = stablehlo.divide %v6001, %v6003 : tensor<384x1536x1x1xf32>
    %v6006 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6007 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6008 = stablehlo.sqrt %v6005 : tensor<384x1536x1x1xf32>
    %v6009 = stablehlo.add %v6008, %v6007 : tensor<384x1536x1x1xf32>
    %v6010 = stablehlo.divide %v6004, %v6009 : tensor<384x1536x1x1xf32>
    %v6011 = stablehlo.multiply %v6006, %v6010 : tensor<384x1536x1x1xf32>
    %v6012 = stablehlo.subtract %s2b0pW, %v6011 : tensor<384x1536x1x1xf32>
    %v6013 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6014 = stablehlo.multiply %v6013, %v6006 : tensor<384x1536x1x1xf32>
    %v6015 = stablehlo.multiply %v6014, %s2b0pW : tensor<384x1536x1x1xf32>
    %v6016 = stablehlo.subtract %v6012, %v6015 : tensor<384x1536x1x1xf32>
    %v6017 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6018 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6019 = stablehlo.multiply %v6017, %s2b0pbm : tensor<384xf32>
    %v6020 = stablehlo.multiply %v6018, %v2526 : tensor<384xf32>
    %v6021 = stablehlo.add %v6019, %v6020 : tensor<384xf32>
    %v6022 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6023 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6024 = stablehlo.multiply %v6022, %s2b0pbv : tensor<384xf32>
    %v6025 = stablehlo.multiply %v2526, %v2526 : tensor<384xf32>
    %v6026 = stablehlo.multiply %v6023, %v6025 : tensor<384xf32>
    %v6027 = stablehlo.add %v6024, %v6026 : tensor<384xf32>
    %v6028 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6029 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6030 = stablehlo.multiply %v6028, %s2b0pbm : tensor<384xf32>
    %v6031 = stablehlo.multiply %v6029, %v2526 : tensor<384xf32>
    %v6032 = stablehlo.add %v6030, %v6031 : tensor<384xf32>
    %v6033 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6034 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6035 = stablehlo.multiply %v6033, %s2b0pbv : tensor<384xf32>
    %v6036 = stablehlo.multiply %v2526, %v2526 : tensor<384xf32>
    %v6037 = stablehlo.multiply %v6034, %v6036 : tensor<384xf32>
    %v6038 = stablehlo.add %v6035, %v6037 : tensor<384xf32>
    %v6039 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6040 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6041 = stablehlo.divide %v6032, %v6039 : tensor<384xf32>
    %v6042 = stablehlo.divide %v6038, %v6040 : tensor<384xf32>
    %v6043 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6044 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6045 = stablehlo.sqrt %v6042 : tensor<384xf32>
    %v6046 = stablehlo.add %v6045, %v6044 : tensor<384xf32>
    %v6047 = stablehlo.divide %v6041, %v6046 : tensor<384xf32>
    %v6048 = stablehlo.multiply %v6043, %v6047 : tensor<384xf32>
    %v6049 = stablehlo.subtract %s2b0pb, %v6048 : tensor<384xf32>
    %v6050 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6051 = stablehlo.multiply %v6050, %v6043 : tensor<384xf32>
    %v6052 = stablehlo.multiply %v6051, %s2b0pb : tensor<384xf32>
    %v6053 = stablehlo.subtract %v6049, %v6052 : tensor<384xf32>
    %v6054 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6055 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6056 = stablehlo.multiply %v6054, %s2b0lgm : tensor<384xf32>
    %v6057 = stablehlo.multiply %v6055, %v2517 : tensor<384xf32>
    %v6058 = stablehlo.add %v6056, %v6057 : tensor<384xf32>
    %v6059 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6060 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6061 = stablehlo.multiply %v6059, %s2b0lgv : tensor<384xf32>
    %v6062 = stablehlo.multiply %v2517, %v2517 : tensor<384xf32>
    %v6063 = stablehlo.multiply %v6060, %v6062 : tensor<384xf32>
    %v6064 = stablehlo.add %v6061, %v6063 : tensor<384xf32>
    %v6065 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6066 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6067 = stablehlo.multiply %v6065, %s2b0lgm : tensor<384xf32>
    %v6068 = stablehlo.multiply %v6066, %v2517 : tensor<384xf32>
    %v6069 = stablehlo.add %v6067, %v6068 : tensor<384xf32>
    %v6070 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6071 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6072 = stablehlo.multiply %v6070, %s2b0lgv : tensor<384xf32>
    %v6073 = stablehlo.multiply %v2517, %v2517 : tensor<384xf32>
    %v6074 = stablehlo.multiply %v6071, %v6073 : tensor<384xf32>
    %v6075 = stablehlo.add %v6072, %v6074 : tensor<384xf32>
    %v6076 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6077 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6078 = stablehlo.divide %v6069, %v6076 : tensor<384xf32>
    %v6079 = stablehlo.divide %v6075, %v6077 : tensor<384xf32>
    %v6080 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6081 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6082 = stablehlo.sqrt %v6079 : tensor<384xf32>
    %v6083 = stablehlo.add %v6082, %v6081 : tensor<384xf32>
    %v6084 = stablehlo.divide %v6078, %v6083 : tensor<384xf32>
    %v6085 = stablehlo.multiply %v6080, %v6084 : tensor<384xf32>
    %v6086 = stablehlo.subtract %s2b0lg, %v6085 : tensor<384xf32>
    %v6087 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6088 = stablehlo.multiply %v6087, %v6080 : tensor<384xf32>
    %v6089 = stablehlo.multiply %v6088, %s2b0lg : tensor<384xf32>
    %v6090 = stablehlo.subtract %v6086, %v6089 : tensor<384xf32>
    %v6091 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6092 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6093 = stablehlo.multiply %v6091, %s2b1dWm : tensor<384x1x7x7xf32>
    %v6094 = stablehlo.multiply %v6092, %v2440 : tensor<384x1x7x7xf32>
    %v6095 = stablehlo.add %v6093, %v6094 : tensor<384x1x7x7xf32>
    %v6096 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6097 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6098 = stablehlo.multiply %v6096, %s2b1dWv : tensor<384x1x7x7xf32>
    %v6099 = stablehlo.multiply %v2440, %v2440 : tensor<384x1x7x7xf32>
    %v6100 = stablehlo.multiply %v6097, %v6099 : tensor<384x1x7x7xf32>
    %v6101 = stablehlo.add %v6098, %v6100 : tensor<384x1x7x7xf32>
    %v6102 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6103 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6104 = stablehlo.multiply %v6102, %s2b1dWm : tensor<384x1x7x7xf32>
    %v6105 = stablehlo.multiply %v6103, %v2440 : tensor<384x1x7x7xf32>
    %v6106 = stablehlo.add %v6104, %v6105 : tensor<384x1x7x7xf32>
    %v6107 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6108 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6109 = stablehlo.multiply %v6107, %s2b1dWv : tensor<384x1x7x7xf32>
    %v6110 = stablehlo.multiply %v2440, %v2440 : tensor<384x1x7x7xf32>
    %v6111 = stablehlo.multiply %v6108, %v6110 : tensor<384x1x7x7xf32>
    %v6112 = stablehlo.add %v6109, %v6111 : tensor<384x1x7x7xf32>
    %v6113 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6114 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6115 = stablehlo.divide %v6106, %v6113 : tensor<384x1x7x7xf32>
    %v6116 = stablehlo.divide %v6112, %v6114 : tensor<384x1x7x7xf32>
    %v6117 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6118 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6119 = stablehlo.sqrt %v6116 : tensor<384x1x7x7xf32>
    %v6120 = stablehlo.add %v6119, %v6118 : tensor<384x1x7x7xf32>
    %v6121 = stablehlo.divide %v6115, %v6120 : tensor<384x1x7x7xf32>
    %v6122 = stablehlo.multiply %v6117, %v6121 : tensor<384x1x7x7xf32>
    %v6123 = stablehlo.subtract %s2b1dW, %v6122 : tensor<384x1x7x7xf32>
    %v6124 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6125 = stablehlo.multiply %v6124, %v6117 : tensor<384x1x7x7xf32>
    %v6126 = stablehlo.multiply %v6125, %s2b1dW : tensor<384x1x7x7xf32>
    %v6127 = stablehlo.subtract %v6123, %v6126 : tensor<384x1x7x7xf32>
    %v6128 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6129 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6130 = stablehlo.multiply %v6128, %s2b1dbm : tensor<384xf32>
    %v6131 = stablehlo.multiply %v6129, %v2443 : tensor<384xf32>
    %v6132 = stablehlo.add %v6130, %v6131 : tensor<384xf32>
    %v6133 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6134 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6135 = stablehlo.multiply %v6133, %s2b1dbv : tensor<384xf32>
    %v6136 = stablehlo.multiply %v2443, %v2443 : tensor<384xf32>
    %v6137 = stablehlo.multiply %v6134, %v6136 : tensor<384xf32>
    %v6138 = stablehlo.add %v6135, %v6137 : tensor<384xf32>
    %v6139 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6140 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6141 = stablehlo.multiply %v6139, %s2b1dbm : tensor<384xf32>
    %v6142 = stablehlo.multiply %v6140, %v2443 : tensor<384xf32>
    %v6143 = stablehlo.add %v6141, %v6142 : tensor<384xf32>
    %v6144 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6145 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6146 = stablehlo.multiply %v6144, %s2b1dbv : tensor<384xf32>
    %v6147 = stablehlo.multiply %v2443, %v2443 : tensor<384xf32>
    %v6148 = stablehlo.multiply %v6145, %v6147 : tensor<384xf32>
    %v6149 = stablehlo.add %v6146, %v6148 : tensor<384xf32>
    %v6150 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6151 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6152 = stablehlo.divide %v6143, %v6150 : tensor<384xf32>
    %v6153 = stablehlo.divide %v6149, %v6151 : tensor<384xf32>
    %v6154 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6155 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6156 = stablehlo.sqrt %v6153 : tensor<384xf32>
    %v6157 = stablehlo.add %v6156, %v6155 : tensor<384xf32>
    %v6158 = stablehlo.divide %v6152, %v6157 : tensor<384xf32>
    %v6159 = stablehlo.multiply %v6154, %v6158 : tensor<384xf32>
    %v6160 = stablehlo.subtract %s2b1db, %v6159 : tensor<384xf32>
    %v6161 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6162 = stablehlo.multiply %v6161, %v6154 : tensor<384xf32>
    %v6163 = stablehlo.multiply %v6162, %s2b1db : tensor<384xf32>
    %v6164 = stablehlo.subtract %v6160, %v6163 : tensor<384xf32>
    %v6165 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6166 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6167 = stablehlo.multiply %v6165, %s2b1ngm : tensor<f32>
    %v6168 = stablehlo.multiply %v6166, %v2432 : tensor<f32>
    %v6169 = stablehlo.add %v6167, %v6168 : tensor<f32>
    %v6170 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6171 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6172 = stablehlo.multiply %v6170, %s2b1ngv : tensor<f32>
    %v6173 = stablehlo.multiply %v2432, %v2432 : tensor<f32>
    %v6174 = stablehlo.multiply %v6171, %v6173 : tensor<f32>
    %v6175 = stablehlo.add %v6172, %v6174 : tensor<f32>
    %v6176 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6177 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6178 = stablehlo.multiply %v6176, %s2b1ngm : tensor<f32>
    %v6179 = stablehlo.multiply %v6177, %v2432 : tensor<f32>
    %v6180 = stablehlo.add %v6178, %v6179 : tensor<f32>
    %v6181 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6182 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6183 = stablehlo.multiply %v6181, %s2b1ngv : tensor<f32>
    %v6184 = stablehlo.multiply %v2432, %v2432 : tensor<f32>
    %v6185 = stablehlo.multiply %v6182, %v6184 : tensor<f32>
    %v6186 = stablehlo.add %v6183, %v6185 : tensor<f32>
    %v6187 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6188 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6189 = stablehlo.divide %v6180, %v6187 : tensor<f32>
    %v6190 = stablehlo.divide %v6186, %v6188 : tensor<f32>
    %v6191 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6192 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6193 = stablehlo.sqrt %v6190 : tensor<f32>
    %v6194 = stablehlo.add %v6193, %v6192 : tensor<f32>
    %v6195 = stablehlo.divide %v6189, %v6194 : tensor<f32>
    %v6196 = stablehlo.multiply %v6191, %v6195 : tensor<f32>
    %v6197 = stablehlo.subtract %s2b1ng, %v6196 : tensor<f32>
    %v6198 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6199 = stablehlo.multiply %v6198, %v6191 : tensor<f32>
    %v6200 = stablehlo.multiply %v6199, %s2b1ng : tensor<f32>
    %v6201 = stablehlo.subtract %v6197, %v6200 : tensor<f32>
    %v6202 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6203 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6204 = stablehlo.multiply %v6202, %s2b1nbtm : tensor<f32>
    %v6205 = stablehlo.multiply %v6203, %v2434 : tensor<f32>
    %v6206 = stablehlo.add %v6204, %v6205 : tensor<f32>
    %v6207 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6208 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6209 = stablehlo.multiply %v6207, %s2b1nbtv : tensor<f32>
    %v6210 = stablehlo.multiply %v2434, %v2434 : tensor<f32>
    %v6211 = stablehlo.multiply %v6208, %v6210 : tensor<f32>
    %v6212 = stablehlo.add %v6209, %v6211 : tensor<f32>
    %v6213 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6214 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6215 = stablehlo.multiply %v6213, %s2b1nbtm : tensor<f32>
    %v6216 = stablehlo.multiply %v6214, %v2434 : tensor<f32>
    %v6217 = stablehlo.add %v6215, %v6216 : tensor<f32>
    %v6218 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6219 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6220 = stablehlo.multiply %v6218, %s2b1nbtv : tensor<f32>
    %v6221 = stablehlo.multiply %v2434, %v2434 : tensor<f32>
    %v6222 = stablehlo.multiply %v6219, %v6221 : tensor<f32>
    %v6223 = stablehlo.add %v6220, %v6222 : tensor<f32>
    %v6224 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6225 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6226 = stablehlo.divide %v6217, %v6224 : tensor<f32>
    %v6227 = stablehlo.divide %v6223, %v6225 : tensor<f32>
    %v6228 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6229 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6230 = stablehlo.sqrt %v6227 : tensor<f32>
    %v6231 = stablehlo.add %v6230, %v6229 : tensor<f32>
    %v6232 = stablehlo.divide %v6226, %v6231 : tensor<f32>
    %v6233 = stablehlo.multiply %v6228, %v6232 : tensor<f32>
    %v6234 = stablehlo.subtract %s2b1nbt, %v6233 : tensor<f32>
    %v6235 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6236 = stablehlo.multiply %v6235, %v6228 : tensor<f32>
    %v6237 = stablehlo.multiply %v6236, %s2b1nbt : tensor<f32>
    %v6238 = stablehlo.subtract %v6234, %v6237 : tensor<f32>
    %v6239 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6240 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6241 = stablehlo.multiply %v6239, %s2b1eWm : tensor<1536x384x1x1xf32>
    %v6242 = stablehlo.multiply %v6240, %v2413 : tensor<1536x384x1x1xf32>
    %v6243 = stablehlo.add %v6241, %v6242 : tensor<1536x384x1x1xf32>
    %v6244 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6245 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6246 = stablehlo.multiply %v6244, %s2b1eWv : tensor<1536x384x1x1xf32>
    %v6247 = stablehlo.multiply %v2413, %v2413 : tensor<1536x384x1x1xf32>
    %v6248 = stablehlo.multiply %v6245, %v6247 : tensor<1536x384x1x1xf32>
    %v6249 = stablehlo.add %v6246, %v6248 : tensor<1536x384x1x1xf32>
    %v6250 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6251 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6252 = stablehlo.multiply %v6250, %s2b1eWm : tensor<1536x384x1x1xf32>
    %v6253 = stablehlo.multiply %v6251, %v2413 : tensor<1536x384x1x1xf32>
    %v6254 = stablehlo.add %v6252, %v6253 : tensor<1536x384x1x1xf32>
    %v6255 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6256 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6257 = stablehlo.multiply %v6255, %s2b1eWv : tensor<1536x384x1x1xf32>
    %v6258 = stablehlo.multiply %v2413, %v2413 : tensor<1536x384x1x1xf32>
    %v6259 = stablehlo.multiply %v6256, %v6258 : tensor<1536x384x1x1xf32>
    %v6260 = stablehlo.add %v6257, %v6259 : tensor<1536x384x1x1xf32>
    %v6261 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6262 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6263 = stablehlo.divide %v6254, %v6261 : tensor<1536x384x1x1xf32>
    %v6264 = stablehlo.divide %v6260, %v6262 : tensor<1536x384x1x1xf32>
    %v6265 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6266 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6267 = stablehlo.sqrt %v6264 : tensor<1536x384x1x1xf32>
    %v6268 = stablehlo.add %v6267, %v6266 : tensor<1536x384x1x1xf32>
    %v6269 = stablehlo.divide %v6263, %v6268 : tensor<1536x384x1x1xf32>
    %v6270 = stablehlo.multiply %v6265, %v6269 : tensor<1536x384x1x1xf32>
    %v6271 = stablehlo.subtract %s2b1eW, %v6270 : tensor<1536x384x1x1xf32>
    %v6272 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6273 = stablehlo.multiply %v6272, %v6265 : tensor<1536x384x1x1xf32>
    %v6274 = stablehlo.multiply %v6273, %s2b1eW : tensor<1536x384x1x1xf32>
    %v6275 = stablehlo.subtract %v6271, %v6274 : tensor<1536x384x1x1xf32>
    %v6276 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6277 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6278 = stablehlo.multiply %v6276, %s2b1ebm : tensor<1536xf32>
    %v6279 = stablehlo.multiply %v6277, %v2416 : tensor<1536xf32>
    %v6280 = stablehlo.add %v6278, %v6279 : tensor<1536xf32>
    %v6281 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6282 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6283 = stablehlo.multiply %v6281, %s2b1ebv : tensor<1536xf32>
    %v6284 = stablehlo.multiply %v2416, %v2416 : tensor<1536xf32>
    %v6285 = stablehlo.multiply %v6282, %v6284 : tensor<1536xf32>
    %v6286 = stablehlo.add %v6283, %v6285 : tensor<1536xf32>
    %v6287 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6288 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6289 = stablehlo.multiply %v6287, %s2b1ebm : tensor<1536xf32>
    %v6290 = stablehlo.multiply %v6288, %v2416 : tensor<1536xf32>
    %v6291 = stablehlo.add %v6289, %v6290 : tensor<1536xf32>
    %v6292 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6293 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6294 = stablehlo.multiply %v6292, %s2b1ebv : tensor<1536xf32>
    %v6295 = stablehlo.multiply %v2416, %v2416 : tensor<1536xf32>
    %v6296 = stablehlo.multiply %v6293, %v6295 : tensor<1536xf32>
    %v6297 = stablehlo.add %v6294, %v6296 : tensor<1536xf32>
    %v6298 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6299 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6300 = stablehlo.divide %v6291, %v6298 : tensor<1536xf32>
    %v6301 = stablehlo.divide %v6297, %v6299 : tensor<1536xf32>
    %v6302 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6303 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6304 = stablehlo.sqrt %v6301 : tensor<1536xf32>
    %v6305 = stablehlo.add %v6304, %v6303 : tensor<1536xf32>
    %v6306 = stablehlo.divide %v6300, %v6305 : tensor<1536xf32>
    %v6307 = stablehlo.multiply %v6302, %v6306 : tensor<1536xf32>
    %v6308 = stablehlo.subtract %s2b1eb, %v6307 : tensor<1536xf32>
    %v6309 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6310 = stablehlo.multiply %v6309, %v6302 : tensor<1536xf32>
    %v6311 = stablehlo.multiply %v6310, %s2b1eb : tensor<1536xf32>
    %v6312 = stablehlo.subtract %v6308, %v6311 : tensor<1536xf32>
    %v6313 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6314 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6315 = stablehlo.multiply %v6313, %s2b1pWm : tensor<384x1536x1x1xf32>
    %v6316 = stablehlo.multiply %v6314, %v2404 : tensor<384x1536x1x1xf32>
    %v6317 = stablehlo.add %v6315, %v6316 : tensor<384x1536x1x1xf32>
    %v6318 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6319 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6320 = stablehlo.multiply %v6318, %s2b1pWv : tensor<384x1536x1x1xf32>
    %v6321 = stablehlo.multiply %v2404, %v2404 : tensor<384x1536x1x1xf32>
    %v6322 = stablehlo.multiply %v6319, %v6321 : tensor<384x1536x1x1xf32>
    %v6323 = stablehlo.add %v6320, %v6322 : tensor<384x1536x1x1xf32>
    %v6324 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6325 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6326 = stablehlo.multiply %v6324, %s2b1pWm : tensor<384x1536x1x1xf32>
    %v6327 = stablehlo.multiply %v6325, %v2404 : tensor<384x1536x1x1xf32>
    %v6328 = stablehlo.add %v6326, %v6327 : tensor<384x1536x1x1xf32>
    %v6329 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6330 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6331 = stablehlo.multiply %v6329, %s2b1pWv : tensor<384x1536x1x1xf32>
    %v6332 = stablehlo.multiply %v2404, %v2404 : tensor<384x1536x1x1xf32>
    %v6333 = stablehlo.multiply %v6330, %v6332 : tensor<384x1536x1x1xf32>
    %v6334 = stablehlo.add %v6331, %v6333 : tensor<384x1536x1x1xf32>
    %v6335 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6336 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6337 = stablehlo.divide %v6328, %v6335 : tensor<384x1536x1x1xf32>
    %v6338 = stablehlo.divide %v6334, %v6336 : tensor<384x1536x1x1xf32>
    %v6339 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6340 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6341 = stablehlo.sqrt %v6338 : tensor<384x1536x1x1xf32>
    %v6342 = stablehlo.add %v6341, %v6340 : tensor<384x1536x1x1xf32>
    %v6343 = stablehlo.divide %v6337, %v6342 : tensor<384x1536x1x1xf32>
    %v6344 = stablehlo.multiply %v6339, %v6343 : tensor<384x1536x1x1xf32>
    %v6345 = stablehlo.subtract %s2b1pW, %v6344 : tensor<384x1536x1x1xf32>
    %v6346 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6347 = stablehlo.multiply %v6346, %v6339 : tensor<384x1536x1x1xf32>
    %v6348 = stablehlo.multiply %v6347, %s2b1pW : tensor<384x1536x1x1xf32>
    %v6349 = stablehlo.subtract %v6345, %v6348 : tensor<384x1536x1x1xf32>
    %v6350 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6351 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6352 = stablehlo.multiply %v6350, %s2b1pbm : tensor<384xf32>
    %v6353 = stablehlo.multiply %v6351, %v2407 : tensor<384xf32>
    %v6354 = stablehlo.add %v6352, %v6353 : tensor<384xf32>
    %v6355 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6356 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6357 = stablehlo.multiply %v6355, %s2b1pbv : tensor<384xf32>
    %v6358 = stablehlo.multiply %v2407, %v2407 : tensor<384xf32>
    %v6359 = stablehlo.multiply %v6356, %v6358 : tensor<384xf32>
    %v6360 = stablehlo.add %v6357, %v6359 : tensor<384xf32>
    %v6361 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6362 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6363 = stablehlo.multiply %v6361, %s2b1pbm : tensor<384xf32>
    %v6364 = stablehlo.multiply %v6362, %v2407 : tensor<384xf32>
    %v6365 = stablehlo.add %v6363, %v6364 : tensor<384xf32>
    %v6366 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6367 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6368 = stablehlo.multiply %v6366, %s2b1pbv : tensor<384xf32>
    %v6369 = stablehlo.multiply %v2407, %v2407 : tensor<384xf32>
    %v6370 = stablehlo.multiply %v6367, %v6369 : tensor<384xf32>
    %v6371 = stablehlo.add %v6368, %v6370 : tensor<384xf32>
    %v6372 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6373 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6374 = stablehlo.divide %v6365, %v6372 : tensor<384xf32>
    %v6375 = stablehlo.divide %v6371, %v6373 : tensor<384xf32>
    %v6376 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6377 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6378 = stablehlo.sqrt %v6375 : tensor<384xf32>
    %v6379 = stablehlo.add %v6378, %v6377 : tensor<384xf32>
    %v6380 = stablehlo.divide %v6374, %v6379 : tensor<384xf32>
    %v6381 = stablehlo.multiply %v6376, %v6380 : tensor<384xf32>
    %v6382 = stablehlo.subtract %s2b1pb, %v6381 : tensor<384xf32>
    %v6383 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6384 = stablehlo.multiply %v6383, %v6376 : tensor<384xf32>
    %v6385 = stablehlo.multiply %v6384, %s2b1pb : tensor<384xf32>
    %v6386 = stablehlo.subtract %v6382, %v6385 : tensor<384xf32>
    %v6387 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6388 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6389 = stablehlo.multiply %v6387, %s2b1lgm : tensor<384xf32>
    %v6390 = stablehlo.multiply %v6388, %v2398 : tensor<384xf32>
    %v6391 = stablehlo.add %v6389, %v6390 : tensor<384xf32>
    %v6392 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6393 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6394 = stablehlo.multiply %v6392, %s2b1lgv : tensor<384xf32>
    %v6395 = stablehlo.multiply %v2398, %v2398 : tensor<384xf32>
    %v6396 = stablehlo.multiply %v6393, %v6395 : tensor<384xf32>
    %v6397 = stablehlo.add %v6394, %v6396 : tensor<384xf32>
    %v6398 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6399 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6400 = stablehlo.multiply %v6398, %s2b1lgm : tensor<384xf32>
    %v6401 = stablehlo.multiply %v6399, %v2398 : tensor<384xf32>
    %v6402 = stablehlo.add %v6400, %v6401 : tensor<384xf32>
    %v6403 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6404 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6405 = stablehlo.multiply %v6403, %s2b1lgv : tensor<384xf32>
    %v6406 = stablehlo.multiply %v2398, %v2398 : tensor<384xf32>
    %v6407 = stablehlo.multiply %v6404, %v6406 : tensor<384xf32>
    %v6408 = stablehlo.add %v6405, %v6407 : tensor<384xf32>
    %v6409 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6410 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6411 = stablehlo.divide %v6402, %v6409 : tensor<384xf32>
    %v6412 = stablehlo.divide %v6408, %v6410 : tensor<384xf32>
    %v6413 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6414 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6415 = stablehlo.sqrt %v6412 : tensor<384xf32>
    %v6416 = stablehlo.add %v6415, %v6414 : tensor<384xf32>
    %v6417 = stablehlo.divide %v6411, %v6416 : tensor<384xf32>
    %v6418 = stablehlo.multiply %v6413, %v6417 : tensor<384xf32>
    %v6419 = stablehlo.subtract %s2b1lg, %v6418 : tensor<384xf32>
    %v6420 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6421 = stablehlo.multiply %v6420, %v6413 : tensor<384xf32>
    %v6422 = stablehlo.multiply %v6421, %s2b1lg : tensor<384xf32>
    %v6423 = stablehlo.subtract %v6419, %v6422 : tensor<384xf32>
    %v6424 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6425 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6426 = stablehlo.multiply %v6424, %s2b2dWm : tensor<384x1x7x7xf32>
    %v6427 = stablehlo.multiply %v6425, %v2321 : tensor<384x1x7x7xf32>
    %v6428 = stablehlo.add %v6426, %v6427 : tensor<384x1x7x7xf32>
    %v6429 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6430 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6431 = stablehlo.multiply %v6429, %s2b2dWv : tensor<384x1x7x7xf32>
    %v6432 = stablehlo.multiply %v2321, %v2321 : tensor<384x1x7x7xf32>
    %v6433 = stablehlo.multiply %v6430, %v6432 : tensor<384x1x7x7xf32>
    %v6434 = stablehlo.add %v6431, %v6433 : tensor<384x1x7x7xf32>
    %v6435 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6436 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6437 = stablehlo.multiply %v6435, %s2b2dWm : tensor<384x1x7x7xf32>
    %v6438 = stablehlo.multiply %v6436, %v2321 : tensor<384x1x7x7xf32>
    %v6439 = stablehlo.add %v6437, %v6438 : tensor<384x1x7x7xf32>
    %v6440 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6441 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6442 = stablehlo.multiply %v6440, %s2b2dWv : tensor<384x1x7x7xf32>
    %v6443 = stablehlo.multiply %v2321, %v2321 : tensor<384x1x7x7xf32>
    %v6444 = stablehlo.multiply %v6441, %v6443 : tensor<384x1x7x7xf32>
    %v6445 = stablehlo.add %v6442, %v6444 : tensor<384x1x7x7xf32>
    %v6446 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6447 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6448 = stablehlo.divide %v6439, %v6446 : tensor<384x1x7x7xf32>
    %v6449 = stablehlo.divide %v6445, %v6447 : tensor<384x1x7x7xf32>
    %v6450 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6451 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6452 = stablehlo.sqrt %v6449 : tensor<384x1x7x7xf32>
    %v6453 = stablehlo.add %v6452, %v6451 : tensor<384x1x7x7xf32>
    %v6454 = stablehlo.divide %v6448, %v6453 : tensor<384x1x7x7xf32>
    %v6455 = stablehlo.multiply %v6450, %v6454 : tensor<384x1x7x7xf32>
    %v6456 = stablehlo.subtract %s2b2dW, %v6455 : tensor<384x1x7x7xf32>
    %v6457 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6458 = stablehlo.multiply %v6457, %v6450 : tensor<384x1x7x7xf32>
    %v6459 = stablehlo.multiply %v6458, %s2b2dW : tensor<384x1x7x7xf32>
    %v6460 = stablehlo.subtract %v6456, %v6459 : tensor<384x1x7x7xf32>
    %v6461 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6462 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6463 = stablehlo.multiply %v6461, %s2b2dbm : tensor<384xf32>
    %v6464 = stablehlo.multiply %v6462, %v2324 : tensor<384xf32>
    %v6465 = stablehlo.add %v6463, %v6464 : tensor<384xf32>
    %v6466 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6467 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6468 = stablehlo.multiply %v6466, %s2b2dbv : tensor<384xf32>
    %v6469 = stablehlo.multiply %v2324, %v2324 : tensor<384xf32>
    %v6470 = stablehlo.multiply %v6467, %v6469 : tensor<384xf32>
    %v6471 = stablehlo.add %v6468, %v6470 : tensor<384xf32>
    %v6472 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6473 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6474 = stablehlo.multiply %v6472, %s2b2dbm : tensor<384xf32>
    %v6475 = stablehlo.multiply %v6473, %v2324 : tensor<384xf32>
    %v6476 = stablehlo.add %v6474, %v6475 : tensor<384xf32>
    %v6477 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6478 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6479 = stablehlo.multiply %v6477, %s2b2dbv : tensor<384xf32>
    %v6480 = stablehlo.multiply %v2324, %v2324 : tensor<384xf32>
    %v6481 = stablehlo.multiply %v6478, %v6480 : tensor<384xf32>
    %v6482 = stablehlo.add %v6479, %v6481 : tensor<384xf32>
    %v6483 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6484 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6485 = stablehlo.divide %v6476, %v6483 : tensor<384xf32>
    %v6486 = stablehlo.divide %v6482, %v6484 : tensor<384xf32>
    %v6487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6488 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6489 = stablehlo.sqrt %v6486 : tensor<384xf32>
    %v6490 = stablehlo.add %v6489, %v6488 : tensor<384xf32>
    %v6491 = stablehlo.divide %v6485, %v6490 : tensor<384xf32>
    %v6492 = stablehlo.multiply %v6487, %v6491 : tensor<384xf32>
    %v6493 = stablehlo.subtract %s2b2db, %v6492 : tensor<384xf32>
    %v6494 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6495 = stablehlo.multiply %v6494, %v6487 : tensor<384xf32>
    %v6496 = stablehlo.multiply %v6495, %s2b2db : tensor<384xf32>
    %v6497 = stablehlo.subtract %v6493, %v6496 : tensor<384xf32>
    %v6498 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6499 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6500 = stablehlo.multiply %v6498, %s2b2ngm : tensor<f32>
    %v6501 = stablehlo.multiply %v6499, %v2313 : tensor<f32>
    %v6502 = stablehlo.add %v6500, %v6501 : tensor<f32>
    %v6503 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6504 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6505 = stablehlo.multiply %v6503, %s2b2ngv : tensor<f32>
    %v6506 = stablehlo.multiply %v2313, %v2313 : tensor<f32>
    %v6507 = stablehlo.multiply %v6504, %v6506 : tensor<f32>
    %v6508 = stablehlo.add %v6505, %v6507 : tensor<f32>
    %v6509 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6510 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6511 = stablehlo.multiply %v6509, %s2b2ngm : tensor<f32>
    %v6512 = stablehlo.multiply %v6510, %v2313 : tensor<f32>
    %v6513 = stablehlo.add %v6511, %v6512 : tensor<f32>
    %v6514 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6515 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6516 = stablehlo.multiply %v6514, %s2b2ngv : tensor<f32>
    %v6517 = stablehlo.multiply %v2313, %v2313 : tensor<f32>
    %v6518 = stablehlo.multiply %v6515, %v6517 : tensor<f32>
    %v6519 = stablehlo.add %v6516, %v6518 : tensor<f32>
    %v6520 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6521 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6522 = stablehlo.divide %v6513, %v6520 : tensor<f32>
    %v6523 = stablehlo.divide %v6519, %v6521 : tensor<f32>
    %v6524 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6525 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6526 = stablehlo.sqrt %v6523 : tensor<f32>
    %v6527 = stablehlo.add %v6526, %v6525 : tensor<f32>
    %v6528 = stablehlo.divide %v6522, %v6527 : tensor<f32>
    %v6529 = stablehlo.multiply %v6524, %v6528 : tensor<f32>
    %v6530 = stablehlo.subtract %s2b2ng, %v6529 : tensor<f32>
    %v6531 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6532 = stablehlo.multiply %v6531, %v6524 : tensor<f32>
    %v6533 = stablehlo.multiply %v6532, %s2b2ng : tensor<f32>
    %v6534 = stablehlo.subtract %v6530, %v6533 : tensor<f32>
    %v6535 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6536 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6537 = stablehlo.multiply %v6535, %s2b2nbtm : tensor<f32>
    %v6538 = stablehlo.multiply %v6536, %v2315 : tensor<f32>
    %v6539 = stablehlo.add %v6537, %v6538 : tensor<f32>
    %v6540 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6541 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6542 = stablehlo.multiply %v6540, %s2b2nbtv : tensor<f32>
    %v6543 = stablehlo.multiply %v2315, %v2315 : tensor<f32>
    %v6544 = stablehlo.multiply %v6541, %v6543 : tensor<f32>
    %v6545 = stablehlo.add %v6542, %v6544 : tensor<f32>
    %v6546 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6547 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6548 = stablehlo.multiply %v6546, %s2b2nbtm : tensor<f32>
    %v6549 = stablehlo.multiply %v6547, %v2315 : tensor<f32>
    %v6550 = stablehlo.add %v6548, %v6549 : tensor<f32>
    %v6551 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6552 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6553 = stablehlo.multiply %v6551, %s2b2nbtv : tensor<f32>
    %v6554 = stablehlo.multiply %v2315, %v2315 : tensor<f32>
    %v6555 = stablehlo.multiply %v6552, %v6554 : tensor<f32>
    %v6556 = stablehlo.add %v6553, %v6555 : tensor<f32>
    %v6557 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6558 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6559 = stablehlo.divide %v6550, %v6557 : tensor<f32>
    %v6560 = stablehlo.divide %v6556, %v6558 : tensor<f32>
    %v6561 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6562 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6563 = stablehlo.sqrt %v6560 : tensor<f32>
    %v6564 = stablehlo.add %v6563, %v6562 : tensor<f32>
    %v6565 = stablehlo.divide %v6559, %v6564 : tensor<f32>
    %v6566 = stablehlo.multiply %v6561, %v6565 : tensor<f32>
    %v6567 = stablehlo.subtract %s2b2nbt, %v6566 : tensor<f32>
    %v6568 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6569 = stablehlo.multiply %v6568, %v6561 : tensor<f32>
    %v6570 = stablehlo.multiply %v6569, %s2b2nbt : tensor<f32>
    %v6571 = stablehlo.subtract %v6567, %v6570 : tensor<f32>
    %v6572 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6573 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6574 = stablehlo.multiply %v6572, %s2b2eWm : tensor<1536x384x1x1xf32>
    %v6575 = stablehlo.multiply %v6573, %v2294 : tensor<1536x384x1x1xf32>
    %v6576 = stablehlo.add %v6574, %v6575 : tensor<1536x384x1x1xf32>
    %v6577 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6578 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6579 = stablehlo.multiply %v6577, %s2b2eWv : tensor<1536x384x1x1xf32>
    %v6580 = stablehlo.multiply %v2294, %v2294 : tensor<1536x384x1x1xf32>
    %v6581 = stablehlo.multiply %v6578, %v6580 : tensor<1536x384x1x1xf32>
    %v6582 = stablehlo.add %v6579, %v6581 : tensor<1536x384x1x1xf32>
    %v6583 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6584 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6585 = stablehlo.multiply %v6583, %s2b2eWm : tensor<1536x384x1x1xf32>
    %v6586 = stablehlo.multiply %v6584, %v2294 : tensor<1536x384x1x1xf32>
    %v6587 = stablehlo.add %v6585, %v6586 : tensor<1536x384x1x1xf32>
    %v6588 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6589 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6590 = stablehlo.multiply %v6588, %s2b2eWv : tensor<1536x384x1x1xf32>
    %v6591 = stablehlo.multiply %v2294, %v2294 : tensor<1536x384x1x1xf32>
    %v6592 = stablehlo.multiply %v6589, %v6591 : tensor<1536x384x1x1xf32>
    %v6593 = stablehlo.add %v6590, %v6592 : tensor<1536x384x1x1xf32>
    %v6594 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6595 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6596 = stablehlo.divide %v6587, %v6594 : tensor<1536x384x1x1xf32>
    %v6597 = stablehlo.divide %v6593, %v6595 : tensor<1536x384x1x1xf32>
    %v6598 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6599 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6600 = stablehlo.sqrt %v6597 : tensor<1536x384x1x1xf32>
    %v6601 = stablehlo.add %v6600, %v6599 : tensor<1536x384x1x1xf32>
    %v6602 = stablehlo.divide %v6596, %v6601 : tensor<1536x384x1x1xf32>
    %v6603 = stablehlo.multiply %v6598, %v6602 : tensor<1536x384x1x1xf32>
    %v6604 = stablehlo.subtract %s2b2eW, %v6603 : tensor<1536x384x1x1xf32>
    %v6605 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6606 = stablehlo.multiply %v6605, %v6598 : tensor<1536x384x1x1xf32>
    %v6607 = stablehlo.multiply %v6606, %s2b2eW : tensor<1536x384x1x1xf32>
    %v6608 = stablehlo.subtract %v6604, %v6607 : tensor<1536x384x1x1xf32>
    %v6609 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6610 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6611 = stablehlo.multiply %v6609, %s2b2ebm : tensor<1536xf32>
    %v6612 = stablehlo.multiply %v6610, %v2297 : tensor<1536xf32>
    %v6613 = stablehlo.add %v6611, %v6612 : tensor<1536xf32>
    %v6614 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6615 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6616 = stablehlo.multiply %v6614, %s2b2ebv : tensor<1536xf32>
    %v6617 = stablehlo.multiply %v2297, %v2297 : tensor<1536xf32>
    %v6618 = stablehlo.multiply %v6615, %v6617 : tensor<1536xf32>
    %v6619 = stablehlo.add %v6616, %v6618 : tensor<1536xf32>
    %v6620 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6621 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6622 = stablehlo.multiply %v6620, %s2b2ebm : tensor<1536xf32>
    %v6623 = stablehlo.multiply %v6621, %v2297 : tensor<1536xf32>
    %v6624 = stablehlo.add %v6622, %v6623 : tensor<1536xf32>
    %v6625 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6626 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6627 = stablehlo.multiply %v6625, %s2b2ebv : tensor<1536xf32>
    %v6628 = stablehlo.multiply %v2297, %v2297 : tensor<1536xf32>
    %v6629 = stablehlo.multiply %v6626, %v6628 : tensor<1536xf32>
    %v6630 = stablehlo.add %v6627, %v6629 : tensor<1536xf32>
    %v6631 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6632 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6633 = stablehlo.divide %v6624, %v6631 : tensor<1536xf32>
    %v6634 = stablehlo.divide %v6630, %v6632 : tensor<1536xf32>
    %v6635 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6636 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6637 = stablehlo.sqrt %v6634 : tensor<1536xf32>
    %v6638 = stablehlo.add %v6637, %v6636 : tensor<1536xf32>
    %v6639 = stablehlo.divide %v6633, %v6638 : tensor<1536xf32>
    %v6640 = stablehlo.multiply %v6635, %v6639 : tensor<1536xf32>
    %v6641 = stablehlo.subtract %s2b2eb, %v6640 : tensor<1536xf32>
    %v6642 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6643 = stablehlo.multiply %v6642, %v6635 : tensor<1536xf32>
    %v6644 = stablehlo.multiply %v6643, %s2b2eb : tensor<1536xf32>
    %v6645 = stablehlo.subtract %v6641, %v6644 : tensor<1536xf32>
    %v6646 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6647 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6648 = stablehlo.multiply %v6646, %s2b2pWm : tensor<384x1536x1x1xf32>
    %v6649 = stablehlo.multiply %v6647, %v2285 : tensor<384x1536x1x1xf32>
    %v6650 = stablehlo.add %v6648, %v6649 : tensor<384x1536x1x1xf32>
    %v6651 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6652 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6653 = stablehlo.multiply %v6651, %s2b2pWv : tensor<384x1536x1x1xf32>
    %v6654 = stablehlo.multiply %v2285, %v2285 : tensor<384x1536x1x1xf32>
    %v6655 = stablehlo.multiply %v6652, %v6654 : tensor<384x1536x1x1xf32>
    %v6656 = stablehlo.add %v6653, %v6655 : tensor<384x1536x1x1xf32>
    %v6657 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6658 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6659 = stablehlo.multiply %v6657, %s2b2pWm : tensor<384x1536x1x1xf32>
    %v6660 = stablehlo.multiply %v6658, %v2285 : tensor<384x1536x1x1xf32>
    %v6661 = stablehlo.add %v6659, %v6660 : tensor<384x1536x1x1xf32>
    %v6662 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6663 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6664 = stablehlo.multiply %v6662, %s2b2pWv : tensor<384x1536x1x1xf32>
    %v6665 = stablehlo.multiply %v2285, %v2285 : tensor<384x1536x1x1xf32>
    %v6666 = stablehlo.multiply %v6663, %v6665 : tensor<384x1536x1x1xf32>
    %v6667 = stablehlo.add %v6664, %v6666 : tensor<384x1536x1x1xf32>
    %v6668 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6669 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6670 = stablehlo.divide %v6661, %v6668 : tensor<384x1536x1x1xf32>
    %v6671 = stablehlo.divide %v6667, %v6669 : tensor<384x1536x1x1xf32>
    %v6672 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6673 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6674 = stablehlo.sqrt %v6671 : tensor<384x1536x1x1xf32>
    %v6675 = stablehlo.add %v6674, %v6673 : tensor<384x1536x1x1xf32>
    %v6676 = stablehlo.divide %v6670, %v6675 : tensor<384x1536x1x1xf32>
    %v6677 = stablehlo.multiply %v6672, %v6676 : tensor<384x1536x1x1xf32>
    %v6678 = stablehlo.subtract %s2b2pW, %v6677 : tensor<384x1536x1x1xf32>
    %v6679 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6680 = stablehlo.multiply %v6679, %v6672 : tensor<384x1536x1x1xf32>
    %v6681 = stablehlo.multiply %v6680, %s2b2pW : tensor<384x1536x1x1xf32>
    %v6682 = stablehlo.subtract %v6678, %v6681 : tensor<384x1536x1x1xf32>
    %v6683 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6684 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6685 = stablehlo.multiply %v6683, %s2b2pbm : tensor<384xf32>
    %v6686 = stablehlo.multiply %v6684, %v2288 : tensor<384xf32>
    %v6687 = stablehlo.add %v6685, %v6686 : tensor<384xf32>
    %v6688 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6689 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6690 = stablehlo.multiply %v6688, %s2b2pbv : tensor<384xf32>
    %v6691 = stablehlo.multiply %v2288, %v2288 : tensor<384xf32>
    %v6692 = stablehlo.multiply %v6689, %v6691 : tensor<384xf32>
    %v6693 = stablehlo.add %v6690, %v6692 : tensor<384xf32>
    %v6694 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6695 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6696 = stablehlo.multiply %v6694, %s2b2pbm : tensor<384xf32>
    %v6697 = stablehlo.multiply %v6695, %v2288 : tensor<384xf32>
    %v6698 = stablehlo.add %v6696, %v6697 : tensor<384xf32>
    %v6699 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6700 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6701 = stablehlo.multiply %v6699, %s2b2pbv : tensor<384xf32>
    %v6702 = stablehlo.multiply %v2288, %v2288 : tensor<384xf32>
    %v6703 = stablehlo.multiply %v6700, %v6702 : tensor<384xf32>
    %v6704 = stablehlo.add %v6701, %v6703 : tensor<384xf32>
    %v6705 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6706 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6707 = stablehlo.divide %v6698, %v6705 : tensor<384xf32>
    %v6708 = stablehlo.divide %v6704, %v6706 : tensor<384xf32>
    %v6709 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6710 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6711 = stablehlo.sqrt %v6708 : tensor<384xf32>
    %v6712 = stablehlo.add %v6711, %v6710 : tensor<384xf32>
    %v6713 = stablehlo.divide %v6707, %v6712 : tensor<384xf32>
    %v6714 = stablehlo.multiply %v6709, %v6713 : tensor<384xf32>
    %v6715 = stablehlo.subtract %s2b2pb, %v6714 : tensor<384xf32>
    %v6716 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6717 = stablehlo.multiply %v6716, %v6709 : tensor<384xf32>
    %v6718 = stablehlo.multiply %v6717, %s2b2pb : tensor<384xf32>
    %v6719 = stablehlo.subtract %v6715, %v6718 : tensor<384xf32>
    %v6720 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6721 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6722 = stablehlo.multiply %v6720, %s2b2lgm : tensor<384xf32>
    %v6723 = stablehlo.multiply %v6721, %v2279 : tensor<384xf32>
    %v6724 = stablehlo.add %v6722, %v6723 : tensor<384xf32>
    %v6725 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6726 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6727 = stablehlo.multiply %v6725, %s2b2lgv : tensor<384xf32>
    %v6728 = stablehlo.multiply %v2279, %v2279 : tensor<384xf32>
    %v6729 = stablehlo.multiply %v6726, %v6728 : tensor<384xf32>
    %v6730 = stablehlo.add %v6727, %v6729 : tensor<384xf32>
    %v6731 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6732 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6733 = stablehlo.multiply %v6731, %s2b2lgm : tensor<384xf32>
    %v6734 = stablehlo.multiply %v6732, %v2279 : tensor<384xf32>
    %v6735 = stablehlo.add %v6733, %v6734 : tensor<384xf32>
    %v6736 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6737 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6738 = stablehlo.multiply %v6736, %s2b2lgv : tensor<384xf32>
    %v6739 = stablehlo.multiply %v2279, %v2279 : tensor<384xf32>
    %v6740 = stablehlo.multiply %v6737, %v6739 : tensor<384xf32>
    %v6741 = stablehlo.add %v6738, %v6740 : tensor<384xf32>
    %v6742 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6743 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6744 = stablehlo.divide %v6735, %v6742 : tensor<384xf32>
    %v6745 = stablehlo.divide %v6741, %v6743 : tensor<384xf32>
    %v6746 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6747 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6748 = stablehlo.sqrt %v6745 : tensor<384xf32>
    %v6749 = stablehlo.add %v6748, %v6747 : tensor<384xf32>
    %v6750 = stablehlo.divide %v6744, %v6749 : tensor<384xf32>
    %v6751 = stablehlo.multiply %v6746, %v6750 : tensor<384xf32>
    %v6752 = stablehlo.subtract %s2b2lg, %v6751 : tensor<384xf32>
    %v6753 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6754 = stablehlo.multiply %v6753, %v6746 : tensor<384xf32>
    %v6755 = stablehlo.multiply %v6754, %s2b2lg : tensor<384xf32>
    %v6756 = stablehlo.subtract %v6752, %v6755 : tensor<384xf32>
    %v6757 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6758 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6759 = stablehlo.multiply %v6757, %s2b3dWm : tensor<384x1x7x7xf32>
    %v6760 = stablehlo.multiply %v6758, %v2202 : tensor<384x1x7x7xf32>
    %v6761 = stablehlo.add %v6759, %v6760 : tensor<384x1x7x7xf32>
    %v6762 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6763 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6764 = stablehlo.multiply %v6762, %s2b3dWv : tensor<384x1x7x7xf32>
    %v6765 = stablehlo.multiply %v2202, %v2202 : tensor<384x1x7x7xf32>
    %v6766 = stablehlo.multiply %v6763, %v6765 : tensor<384x1x7x7xf32>
    %v6767 = stablehlo.add %v6764, %v6766 : tensor<384x1x7x7xf32>
    %v6768 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6769 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6770 = stablehlo.multiply %v6768, %s2b3dWm : tensor<384x1x7x7xf32>
    %v6771 = stablehlo.multiply %v6769, %v2202 : tensor<384x1x7x7xf32>
    %v6772 = stablehlo.add %v6770, %v6771 : tensor<384x1x7x7xf32>
    %v6773 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6774 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6775 = stablehlo.multiply %v6773, %s2b3dWv : tensor<384x1x7x7xf32>
    %v6776 = stablehlo.multiply %v2202, %v2202 : tensor<384x1x7x7xf32>
    %v6777 = stablehlo.multiply %v6774, %v6776 : tensor<384x1x7x7xf32>
    %v6778 = stablehlo.add %v6775, %v6777 : tensor<384x1x7x7xf32>
    %v6779 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6780 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6781 = stablehlo.divide %v6772, %v6779 : tensor<384x1x7x7xf32>
    %v6782 = stablehlo.divide %v6778, %v6780 : tensor<384x1x7x7xf32>
    %v6783 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6784 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6785 = stablehlo.sqrt %v6782 : tensor<384x1x7x7xf32>
    %v6786 = stablehlo.add %v6785, %v6784 : tensor<384x1x7x7xf32>
    %v6787 = stablehlo.divide %v6781, %v6786 : tensor<384x1x7x7xf32>
    %v6788 = stablehlo.multiply %v6783, %v6787 : tensor<384x1x7x7xf32>
    %v6789 = stablehlo.subtract %s2b3dW, %v6788 : tensor<384x1x7x7xf32>
    %v6790 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6791 = stablehlo.multiply %v6790, %v6783 : tensor<384x1x7x7xf32>
    %v6792 = stablehlo.multiply %v6791, %s2b3dW : tensor<384x1x7x7xf32>
    %v6793 = stablehlo.subtract %v6789, %v6792 : tensor<384x1x7x7xf32>
    %v6794 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6795 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6796 = stablehlo.multiply %v6794, %s2b3dbm : tensor<384xf32>
    %v6797 = stablehlo.multiply %v6795, %v2205 : tensor<384xf32>
    %v6798 = stablehlo.add %v6796, %v6797 : tensor<384xf32>
    %v6799 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6800 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6801 = stablehlo.multiply %v6799, %s2b3dbv : tensor<384xf32>
    %v6802 = stablehlo.multiply %v2205, %v2205 : tensor<384xf32>
    %v6803 = stablehlo.multiply %v6800, %v6802 : tensor<384xf32>
    %v6804 = stablehlo.add %v6801, %v6803 : tensor<384xf32>
    %v6805 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6806 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6807 = stablehlo.multiply %v6805, %s2b3dbm : tensor<384xf32>
    %v6808 = stablehlo.multiply %v6806, %v2205 : tensor<384xf32>
    %v6809 = stablehlo.add %v6807, %v6808 : tensor<384xf32>
    %v6810 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6811 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6812 = stablehlo.multiply %v6810, %s2b3dbv : tensor<384xf32>
    %v6813 = stablehlo.multiply %v2205, %v2205 : tensor<384xf32>
    %v6814 = stablehlo.multiply %v6811, %v6813 : tensor<384xf32>
    %v6815 = stablehlo.add %v6812, %v6814 : tensor<384xf32>
    %v6816 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6817 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6818 = stablehlo.divide %v6809, %v6816 : tensor<384xf32>
    %v6819 = stablehlo.divide %v6815, %v6817 : tensor<384xf32>
    %v6820 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6821 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6822 = stablehlo.sqrt %v6819 : tensor<384xf32>
    %v6823 = stablehlo.add %v6822, %v6821 : tensor<384xf32>
    %v6824 = stablehlo.divide %v6818, %v6823 : tensor<384xf32>
    %v6825 = stablehlo.multiply %v6820, %v6824 : tensor<384xf32>
    %v6826 = stablehlo.subtract %s2b3db, %v6825 : tensor<384xf32>
    %v6827 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6828 = stablehlo.multiply %v6827, %v6820 : tensor<384xf32>
    %v6829 = stablehlo.multiply %v6828, %s2b3db : tensor<384xf32>
    %v6830 = stablehlo.subtract %v6826, %v6829 : tensor<384xf32>
    %v6831 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6832 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6833 = stablehlo.multiply %v6831, %s2b3ngm : tensor<f32>
    %v6834 = stablehlo.multiply %v6832, %v2194 : tensor<f32>
    %v6835 = stablehlo.add %v6833, %v6834 : tensor<f32>
    %v6836 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6837 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6838 = stablehlo.multiply %v6836, %s2b3ngv : tensor<f32>
    %v6839 = stablehlo.multiply %v2194, %v2194 : tensor<f32>
    %v6840 = stablehlo.multiply %v6837, %v6839 : tensor<f32>
    %v6841 = stablehlo.add %v6838, %v6840 : tensor<f32>
    %v6842 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6843 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6844 = stablehlo.multiply %v6842, %s2b3ngm : tensor<f32>
    %v6845 = stablehlo.multiply %v6843, %v2194 : tensor<f32>
    %v6846 = stablehlo.add %v6844, %v6845 : tensor<f32>
    %v6847 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6848 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6849 = stablehlo.multiply %v6847, %s2b3ngv : tensor<f32>
    %v6850 = stablehlo.multiply %v2194, %v2194 : tensor<f32>
    %v6851 = stablehlo.multiply %v6848, %v6850 : tensor<f32>
    %v6852 = stablehlo.add %v6849, %v6851 : tensor<f32>
    %v6853 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6854 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6855 = stablehlo.divide %v6846, %v6853 : tensor<f32>
    %v6856 = stablehlo.divide %v6852, %v6854 : tensor<f32>
    %v6857 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6858 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6859 = stablehlo.sqrt %v6856 : tensor<f32>
    %v6860 = stablehlo.add %v6859, %v6858 : tensor<f32>
    %v6861 = stablehlo.divide %v6855, %v6860 : tensor<f32>
    %v6862 = stablehlo.multiply %v6857, %v6861 : tensor<f32>
    %v6863 = stablehlo.subtract %s2b3ng, %v6862 : tensor<f32>
    %v6864 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6865 = stablehlo.multiply %v6864, %v6857 : tensor<f32>
    %v6866 = stablehlo.multiply %v6865, %s2b3ng : tensor<f32>
    %v6867 = stablehlo.subtract %v6863, %v6866 : tensor<f32>
    %v6868 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6869 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6870 = stablehlo.multiply %v6868, %s2b3nbtm : tensor<f32>
    %v6871 = stablehlo.multiply %v6869, %v2196 : tensor<f32>
    %v6872 = stablehlo.add %v6870, %v6871 : tensor<f32>
    %v6873 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6874 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6875 = stablehlo.multiply %v6873, %s2b3nbtv : tensor<f32>
    %v6876 = stablehlo.multiply %v2196, %v2196 : tensor<f32>
    %v6877 = stablehlo.multiply %v6874, %v6876 : tensor<f32>
    %v6878 = stablehlo.add %v6875, %v6877 : tensor<f32>
    %v6879 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6880 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6881 = stablehlo.multiply %v6879, %s2b3nbtm : tensor<f32>
    %v6882 = stablehlo.multiply %v6880, %v2196 : tensor<f32>
    %v6883 = stablehlo.add %v6881, %v6882 : tensor<f32>
    %v6884 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6885 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6886 = stablehlo.multiply %v6884, %s2b3nbtv : tensor<f32>
    %v6887 = stablehlo.multiply %v2196, %v2196 : tensor<f32>
    %v6888 = stablehlo.multiply %v6885, %v6887 : tensor<f32>
    %v6889 = stablehlo.add %v6886, %v6888 : tensor<f32>
    %v6890 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6891 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6892 = stablehlo.divide %v6883, %v6890 : tensor<f32>
    %v6893 = stablehlo.divide %v6889, %v6891 : tensor<f32>
    %v6894 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6895 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6896 = stablehlo.sqrt %v6893 : tensor<f32>
    %v6897 = stablehlo.add %v6896, %v6895 : tensor<f32>
    %v6898 = stablehlo.divide %v6892, %v6897 : tensor<f32>
    %v6899 = stablehlo.multiply %v6894, %v6898 : tensor<f32>
    %v6900 = stablehlo.subtract %s2b3nbt, %v6899 : tensor<f32>
    %v6901 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v6902 = stablehlo.multiply %v6901, %v6894 : tensor<f32>
    %v6903 = stablehlo.multiply %v6902, %s2b3nbt : tensor<f32>
    %v6904 = stablehlo.subtract %v6900, %v6903 : tensor<f32>
    %v6905 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6906 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6907 = stablehlo.multiply %v6905, %s2b3eWm : tensor<1536x384x1x1xf32>
    %v6908 = stablehlo.multiply %v6906, %v2175 : tensor<1536x384x1x1xf32>
    %v6909 = stablehlo.add %v6907, %v6908 : tensor<1536x384x1x1xf32>
    %v6910 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6911 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6912 = stablehlo.multiply %v6910, %s2b3eWv : tensor<1536x384x1x1xf32>
    %v6913 = stablehlo.multiply %v2175, %v2175 : tensor<1536x384x1x1xf32>
    %v6914 = stablehlo.multiply %v6911, %v6913 : tensor<1536x384x1x1xf32>
    %v6915 = stablehlo.add %v6912, %v6914 : tensor<1536x384x1x1xf32>
    %v6916 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6917 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6918 = stablehlo.multiply %v6916, %s2b3eWm : tensor<1536x384x1x1xf32>
    %v6919 = stablehlo.multiply %v6917, %v2175 : tensor<1536x384x1x1xf32>
    %v6920 = stablehlo.add %v6918, %v6919 : tensor<1536x384x1x1xf32>
    %v6921 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6922 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6923 = stablehlo.multiply %v6921, %s2b3eWv : tensor<1536x384x1x1xf32>
    %v6924 = stablehlo.multiply %v2175, %v2175 : tensor<1536x384x1x1xf32>
    %v6925 = stablehlo.multiply %v6922, %v6924 : tensor<1536x384x1x1xf32>
    %v6926 = stablehlo.add %v6923, %v6925 : tensor<1536x384x1x1xf32>
    %v6927 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6928 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6929 = stablehlo.divide %v6920, %v6927 : tensor<1536x384x1x1xf32>
    %v6930 = stablehlo.divide %v6926, %v6928 : tensor<1536x384x1x1xf32>
    %v6931 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6932 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6933 = stablehlo.sqrt %v6930 : tensor<1536x384x1x1xf32>
    %v6934 = stablehlo.add %v6933, %v6932 : tensor<1536x384x1x1xf32>
    %v6935 = stablehlo.divide %v6929, %v6934 : tensor<1536x384x1x1xf32>
    %v6936 = stablehlo.multiply %v6931, %v6935 : tensor<1536x384x1x1xf32>
    %v6937 = stablehlo.subtract %s2b3eW, %v6936 : tensor<1536x384x1x1xf32>
    %v6938 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v6939 = stablehlo.multiply %v6938, %v6931 : tensor<1536x384x1x1xf32>
    %v6940 = stablehlo.multiply %v6939, %s2b3eW : tensor<1536x384x1x1xf32>
    %v6941 = stablehlo.subtract %v6937, %v6940 : tensor<1536x384x1x1xf32>
    %v6942 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6943 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6944 = stablehlo.multiply %v6942, %s2b3ebm : tensor<1536xf32>
    %v6945 = stablehlo.multiply %v6943, %v2178 : tensor<1536xf32>
    %v6946 = stablehlo.add %v6944, %v6945 : tensor<1536xf32>
    %v6947 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6948 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6949 = stablehlo.multiply %v6947, %s2b3ebv : tensor<1536xf32>
    %v6950 = stablehlo.multiply %v2178, %v2178 : tensor<1536xf32>
    %v6951 = stablehlo.multiply %v6948, %v6950 : tensor<1536xf32>
    %v6952 = stablehlo.add %v6949, %v6951 : tensor<1536xf32>
    %v6953 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6954 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6955 = stablehlo.multiply %v6953, %s2b3ebm : tensor<1536xf32>
    %v6956 = stablehlo.multiply %v6954, %v2178 : tensor<1536xf32>
    %v6957 = stablehlo.add %v6955, %v6956 : tensor<1536xf32>
    %v6958 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6959 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6960 = stablehlo.multiply %v6958, %s2b3ebv : tensor<1536xf32>
    %v6961 = stablehlo.multiply %v2178, %v2178 : tensor<1536xf32>
    %v6962 = stablehlo.multiply %v6959, %v6961 : tensor<1536xf32>
    %v6963 = stablehlo.add %v6960, %v6962 : tensor<1536xf32>
    %v6964 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6965 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6966 = stablehlo.divide %v6957, %v6964 : tensor<1536xf32>
    %v6967 = stablehlo.divide %v6963, %v6965 : tensor<1536xf32>
    %v6968 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6969 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6970 = stablehlo.sqrt %v6967 : tensor<1536xf32>
    %v6971 = stablehlo.add %v6970, %v6969 : tensor<1536xf32>
    %v6972 = stablehlo.divide %v6966, %v6971 : tensor<1536xf32>
    %v6973 = stablehlo.multiply %v6968, %v6972 : tensor<1536xf32>
    %v6974 = stablehlo.subtract %s2b3eb, %v6973 : tensor<1536xf32>
    %v6975 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v6976 = stablehlo.multiply %v6975, %v6968 : tensor<1536xf32>
    %v6977 = stablehlo.multiply %v6976, %s2b3eb : tensor<1536xf32>
    %v6978 = stablehlo.subtract %v6974, %v6977 : tensor<1536xf32>
    %v6979 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6980 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6981 = stablehlo.multiply %v6979, %s2b3pWm : tensor<384x1536x1x1xf32>
    %v6982 = stablehlo.multiply %v6980, %v2166 : tensor<384x1536x1x1xf32>
    %v6983 = stablehlo.add %v6981, %v6982 : tensor<384x1536x1x1xf32>
    %v6984 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6985 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6986 = stablehlo.multiply %v6984, %s2b3pWv : tensor<384x1536x1x1xf32>
    %v6987 = stablehlo.multiply %v2166, %v2166 : tensor<384x1536x1x1xf32>
    %v6988 = stablehlo.multiply %v6985, %v6987 : tensor<384x1536x1x1xf32>
    %v6989 = stablehlo.add %v6986, %v6988 : tensor<384x1536x1x1xf32>
    %v6990 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6991 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6992 = stablehlo.multiply %v6990, %s2b3pWm : tensor<384x1536x1x1xf32>
    %v6993 = stablehlo.multiply %v6991, %v2166 : tensor<384x1536x1x1xf32>
    %v6994 = stablehlo.add %v6992, %v6993 : tensor<384x1536x1x1xf32>
    %v6995 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6996 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v6997 = stablehlo.multiply %v6995, %s2b3pWv : tensor<384x1536x1x1xf32>
    %v6998 = stablehlo.multiply %v2166, %v2166 : tensor<384x1536x1x1xf32>
    %v6999 = stablehlo.multiply %v6996, %v6998 : tensor<384x1536x1x1xf32>
    %v7000 = stablehlo.add %v6997, %v6999 : tensor<384x1536x1x1xf32>
    %v7001 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7002 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7003 = stablehlo.divide %v6994, %v7001 : tensor<384x1536x1x1xf32>
    %v7004 = stablehlo.divide %v7000, %v7002 : tensor<384x1536x1x1xf32>
    %v7005 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7006 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7007 = stablehlo.sqrt %v7004 : tensor<384x1536x1x1xf32>
    %v7008 = stablehlo.add %v7007, %v7006 : tensor<384x1536x1x1xf32>
    %v7009 = stablehlo.divide %v7003, %v7008 : tensor<384x1536x1x1xf32>
    %v7010 = stablehlo.multiply %v7005, %v7009 : tensor<384x1536x1x1xf32>
    %v7011 = stablehlo.subtract %s2b3pW, %v7010 : tensor<384x1536x1x1xf32>
    %v7012 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7013 = stablehlo.multiply %v7012, %v7005 : tensor<384x1536x1x1xf32>
    %v7014 = stablehlo.multiply %v7013, %s2b3pW : tensor<384x1536x1x1xf32>
    %v7015 = stablehlo.subtract %v7011, %v7014 : tensor<384x1536x1x1xf32>
    %v7016 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7017 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7018 = stablehlo.multiply %v7016, %s2b3pbm : tensor<384xf32>
    %v7019 = stablehlo.multiply %v7017, %v2169 : tensor<384xf32>
    %v7020 = stablehlo.add %v7018, %v7019 : tensor<384xf32>
    %v7021 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7022 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7023 = stablehlo.multiply %v7021, %s2b3pbv : tensor<384xf32>
    %v7024 = stablehlo.multiply %v2169, %v2169 : tensor<384xf32>
    %v7025 = stablehlo.multiply %v7022, %v7024 : tensor<384xf32>
    %v7026 = stablehlo.add %v7023, %v7025 : tensor<384xf32>
    %v7027 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7028 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7029 = stablehlo.multiply %v7027, %s2b3pbm : tensor<384xf32>
    %v7030 = stablehlo.multiply %v7028, %v2169 : tensor<384xf32>
    %v7031 = stablehlo.add %v7029, %v7030 : tensor<384xf32>
    %v7032 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7033 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7034 = stablehlo.multiply %v7032, %s2b3pbv : tensor<384xf32>
    %v7035 = stablehlo.multiply %v2169, %v2169 : tensor<384xf32>
    %v7036 = stablehlo.multiply %v7033, %v7035 : tensor<384xf32>
    %v7037 = stablehlo.add %v7034, %v7036 : tensor<384xf32>
    %v7038 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7039 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7040 = stablehlo.divide %v7031, %v7038 : tensor<384xf32>
    %v7041 = stablehlo.divide %v7037, %v7039 : tensor<384xf32>
    %v7042 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7043 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7044 = stablehlo.sqrt %v7041 : tensor<384xf32>
    %v7045 = stablehlo.add %v7044, %v7043 : tensor<384xf32>
    %v7046 = stablehlo.divide %v7040, %v7045 : tensor<384xf32>
    %v7047 = stablehlo.multiply %v7042, %v7046 : tensor<384xf32>
    %v7048 = stablehlo.subtract %s2b3pb, %v7047 : tensor<384xf32>
    %v7049 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7050 = stablehlo.multiply %v7049, %v7042 : tensor<384xf32>
    %v7051 = stablehlo.multiply %v7050, %s2b3pb : tensor<384xf32>
    %v7052 = stablehlo.subtract %v7048, %v7051 : tensor<384xf32>
    %v7053 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7054 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7055 = stablehlo.multiply %v7053, %s2b3lgm : tensor<384xf32>
    %v7056 = stablehlo.multiply %v7054, %v2160 : tensor<384xf32>
    %v7057 = stablehlo.add %v7055, %v7056 : tensor<384xf32>
    %v7058 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7059 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7060 = stablehlo.multiply %v7058, %s2b3lgv : tensor<384xf32>
    %v7061 = stablehlo.multiply %v2160, %v2160 : tensor<384xf32>
    %v7062 = stablehlo.multiply %v7059, %v7061 : tensor<384xf32>
    %v7063 = stablehlo.add %v7060, %v7062 : tensor<384xf32>
    %v7064 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7065 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7066 = stablehlo.multiply %v7064, %s2b3lgm : tensor<384xf32>
    %v7067 = stablehlo.multiply %v7065, %v2160 : tensor<384xf32>
    %v7068 = stablehlo.add %v7066, %v7067 : tensor<384xf32>
    %v7069 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7070 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7071 = stablehlo.multiply %v7069, %s2b3lgv : tensor<384xf32>
    %v7072 = stablehlo.multiply %v2160, %v2160 : tensor<384xf32>
    %v7073 = stablehlo.multiply %v7070, %v7072 : tensor<384xf32>
    %v7074 = stablehlo.add %v7071, %v7073 : tensor<384xf32>
    %v7075 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7076 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7077 = stablehlo.divide %v7068, %v7075 : tensor<384xf32>
    %v7078 = stablehlo.divide %v7074, %v7076 : tensor<384xf32>
    %v7079 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7080 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7081 = stablehlo.sqrt %v7078 : tensor<384xf32>
    %v7082 = stablehlo.add %v7081, %v7080 : tensor<384xf32>
    %v7083 = stablehlo.divide %v7077, %v7082 : tensor<384xf32>
    %v7084 = stablehlo.multiply %v7079, %v7083 : tensor<384xf32>
    %v7085 = stablehlo.subtract %s2b3lg, %v7084 : tensor<384xf32>
    %v7086 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7087 = stablehlo.multiply %v7086, %v7079 : tensor<384xf32>
    %v7088 = stablehlo.multiply %v7087, %s2b3lg : tensor<384xf32>
    %v7089 = stablehlo.subtract %v7085, %v7088 : tensor<384xf32>
    %v7090 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7091 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7092 = stablehlo.multiply %v7090, %s2b4dWm : tensor<384x1x7x7xf32>
    %v7093 = stablehlo.multiply %v7091, %v2083 : tensor<384x1x7x7xf32>
    %v7094 = stablehlo.add %v7092, %v7093 : tensor<384x1x7x7xf32>
    %v7095 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7096 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7097 = stablehlo.multiply %v7095, %s2b4dWv : tensor<384x1x7x7xf32>
    %v7098 = stablehlo.multiply %v2083, %v2083 : tensor<384x1x7x7xf32>
    %v7099 = stablehlo.multiply %v7096, %v7098 : tensor<384x1x7x7xf32>
    %v7100 = stablehlo.add %v7097, %v7099 : tensor<384x1x7x7xf32>
    %v7101 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7102 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7103 = stablehlo.multiply %v7101, %s2b4dWm : tensor<384x1x7x7xf32>
    %v7104 = stablehlo.multiply %v7102, %v2083 : tensor<384x1x7x7xf32>
    %v7105 = stablehlo.add %v7103, %v7104 : tensor<384x1x7x7xf32>
    %v7106 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7107 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7108 = stablehlo.multiply %v7106, %s2b4dWv : tensor<384x1x7x7xf32>
    %v7109 = stablehlo.multiply %v2083, %v2083 : tensor<384x1x7x7xf32>
    %v7110 = stablehlo.multiply %v7107, %v7109 : tensor<384x1x7x7xf32>
    %v7111 = stablehlo.add %v7108, %v7110 : tensor<384x1x7x7xf32>
    %v7112 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7113 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7114 = stablehlo.divide %v7105, %v7112 : tensor<384x1x7x7xf32>
    %v7115 = stablehlo.divide %v7111, %v7113 : tensor<384x1x7x7xf32>
    %v7116 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7117 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7118 = stablehlo.sqrt %v7115 : tensor<384x1x7x7xf32>
    %v7119 = stablehlo.add %v7118, %v7117 : tensor<384x1x7x7xf32>
    %v7120 = stablehlo.divide %v7114, %v7119 : tensor<384x1x7x7xf32>
    %v7121 = stablehlo.multiply %v7116, %v7120 : tensor<384x1x7x7xf32>
    %v7122 = stablehlo.subtract %s2b4dW, %v7121 : tensor<384x1x7x7xf32>
    %v7123 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7124 = stablehlo.multiply %v7123, %v7116 : tensor<384x1x7x7xf32>
    %v7125 = stablehlo.multiply %v7124, %s2b4dW : tensor<384x1x7x7xf32>
    %v7126 = stablehlo.subtract %v7122, %v7125 : tensor<384x1x7x7xf32>
    %v7127 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7128 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7129 = stablehlo.multiply %v7127, %s2b4dbm : tensor<384xf32>
    %v7130 = stablehlo.multiply %v7128, %v2086 : tensor<384xf32>
    %v7131 = stablehlo.add %v7129, %v7130 : tensor<384xf32>
    %v7132 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7133 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7134 = stablehlo.multiply %v7132, %s2b4dbv : tensor<384xf32>
    %v7135 = stablehlo.multiply %v2086, %v2086 : tensor<384xf32>
    %v7136 = stablehlo.multiply %v7133, %v7135 : tensor<384xf32>
    %v7137 = stablehlo.add %v7134, %v7136 : tensor<384xf32>
    %v7138 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7139 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7140 = stablehlo.multiply %v7138, %s2b4dbm : tensor<384xf32>
    %v7141 = stablehlo.multiply %v7139, %v2086 : tensor<384xf32>
    %v7142 = stablehlo.add %v7140, %v7141 : tensor<384xf32>
    %v7143 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7144 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7145 = stablehlo.multiply %v7143, %s2b4dbv : tensor<384xf32>
    %v7146 = stablehlo.multiply %v2086, %v2086 : tensor<384xf32>
    %v7147 = stablehlo.multiply %v7144, %v7146 : tensor<384xf32>
    %v7148 = stablehlo.add %v7145, %v7147 : tensor<384xf32>
    %v7149 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7150 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7151 = stablehlo.divide %v7142, %v7149 : tensor<384xf32>
    %v7152 = stablehlo.divide %v7148, %v7150 : tensor<384xf32>
    %v7153 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7154 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7155 = stablehlo.sqrt %v7152 : tensor<384xf32>
    %v7156 = stablehlo.add %v7155, %v7154 : tensor<384xf32>
    %v7157 = stablehlo.divide %v7151, %v7156 : tensor<384xf32>
    %v7158 = stablehlo.multiply %v7153, %v7157 : tensor<384xf32>
    %v7159 = stablehlo.subtract %s2b4db, %v7158 : tensor<384xf32>
    %v7160 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7161 = stablehlo.multiply %v7160, %v7153 : tensor<384xf32>
    %v7162 = stablehlo.multiply %v7161, %s2b4db : tensor<384xf32>
    %v7163 = stablehlo.subtract %v7159, %v7162 : tensor<384xf32>
    %v7164 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7165 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7166 = stablehlo.multiply %v7164, %s2b4ngm : tensor<f32>
    %v7167 = stablehlo.multiply %v7165, %v2075 : tensor<f32>
    %v7168 = stablehlo.add %v7166, %v7167 : tensor<f32>
    %v7169 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7170 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7171 = stablehlo.multiply %v7169, %s2b4ngv : tensor<f32>
    %v7172 = stablehlo.multiply %v2075, %v2075 : tensor<f32>
    %v7173 = stablehlo.multiply %v7170, %v7172 : tensor<f32>
    %v7174 = stablehlo.add %v7171, %v7173 : tensor<f32>
    %v7175 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7176 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7177 = stablehlo.multiply %v7175, %s2b4ngm : tensor<f32>
    %v7178 = stablehlo.multiply %v7176, %v2075 : tensor<f32>
    %v7179 = stablehlo.add %v7177, %v7178 : tensor<f32>
    %v7180 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7181 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7182 = stablehlo.multiply %v7180, %s2b4ngv : tensor<f32>
    %v7183 = stablehlo.multiply %v2075, %v2075 : tensor<f32>
    %v7184 = stablehlo.multiply %v7181, %v7183 : tensor<f32>
    %v7185 = stablehlo.add %v7182, %v7184 : tensor<f32>
    %v7186 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7187 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7188 = stablehlo.divide %v7179, %v7186 : tensor<f32>
    %v7189 = stablehlo.divide %v7185, %v7187 : tensor<f32>
    %v7190 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7191 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7192 = stablehlo.sqrt %v7189 : tensor<f32>
    %v7193 = stablehlo.add %v7192, %v7191 : tensor<f32>
    %v7194 = stablehlo.divide %v7188, %v7193 : tensor<f32>
    %v7195 = stablehlo.multiply %v7190, %v7194 : tensor<f32>
    %v7196 = stablehlo.subtract %s2b4ng, %v7195 : tensor<f32>
    %v7197 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7198 = stablehlo.multiply %v7197, %v7190 : tensor<f32>
    %v7199 = stablehlo.multiply %v7198, %s2b4ng : tensor<f32>
    %v7200 = stablehlo.subtract %v7196, %v7199 : tensor<f32>
    %v7201 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7202 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7203 = stablehlo.multiply %v7201, %s2b4nbtm : tensor<f32>
    %v7204 = stablehlo.multiply %v7202, %v2077 : tensor<f32>
    %v7205 = stablehlo.add %v7203, %v7204 : tensor<f32>
    %v7206 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7207 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7208 = stablehlo.multiply %v7206, %s2b4nbtv : tensor<f32>
    %v7209 = stablehlo.multiply %v2077, %v2077 : tensor<f32>
    %v7210 = stablehlo.multiply %v7207, %v7209 : tensor<f32>
    %v7211 = stablehlo.add %v7208, %v7210 : tensor<f32>
    %v7212 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7213 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7214 = stablehlo.multiply %v7212, %s2b4nbtm : tensor<f32>
    %v7215 = stablehlo.multiply %v7213, %v2077 : tensor<f32>
    %v7216 = stablehlo.add %v7214, %v7215 : tensor<f32>
    %v7217 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7218 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7219 = stablehlo.multiply %v7217, %s2b4nbtv : tensor<f32>
    %v7220 = stablehlo.multiply %v2077, %v2077 : tensor<f32>
    %v7221 = stablehlo.multiply %v7218, %v7220 : tensor<f32>
    %v7222 = stablehlo.add %v7219, %v7221 : tensor<f32>
    %v7223 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7224 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7225 = stablehlo.divide %v7216, %v7223 : tensor<f32>
    %v7226 = stablehlo.divide %v7222, %v7224 : tensor<f32>
    %v7227 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7228 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7229 = stablehlo.sqrt %v7226 : tensor<f32>
    %v7230 = stablehlo.add %v7229, %v7228 : tensor<f32>
    %v7231 = stablehlo.divide %v7225, %v7230 : tensor<f32>
    %v7232 = stablehlo.multiply %v7227, %v7231 : tensor<f32>
    %v7233 = stablehlo.subtract %s2b4nbt, %v7232 : tensor<f32>
    %v7234 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7235 = stablehlo.multiply %v7234, %v7227 : tensor<f32>
    %v7236 = stablehlo.multiply %v7235, %s2b4nbt : tensor<f32>
    %v7237 = stablehlo.subtract %v7233, %v7236 : tensor<f32>
    %v7238 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7239 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7240 = stablehlo.multiply %v7238, %s2b4eWm : tensor<1536x384x1x1xf32>
    %v7241 = stablehlo.multiply %v7239, %v2056 : tensor<1536x384x1x1xf32>
    %v7242 = stablehlo.add %v7240, %v7241 : tensor<1536x384x1x1xf32>
    %v7243 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7244 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7245 = stablehlo.multiply %v7243, %s2b4eWv : tensor<1536x384x1x1xf32>
    %v7246 = stablehlo.multiply %v2056, %v2056 : tensor<1536x384x1x1xf32>
    %v7247 = stablehlo.multiply %v7244, %v7246 : tensor<1536x384x1x1xf32>
    %v7248 = stablehlo.add %v7245, %v7247 : tensor<1536x384x1x1xf32>
    %v7249 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7250 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7251 = stablehlo.multiply %v7249, %s2b4eWm : tensor<1536x384x1x1xf32>
    %v7252 = stablehlo.multiply %v7250, %v2056 : tensor<1536x384x1x1xf32>
    %v7253 = stablehlo.add %v7251, %v7252 : tensor<1536x384x1x1xf32>
    %v7254 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7255 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7256 = stablehlo.multiply %v7254, %s2b4eWv : tensor<1536x384x1x1xf32>
    %v7257 = stablehlo.multiply %v2056, %v2056 : tensor<1536x384x1x1xf32>
    %v7258 = stablehlo.multiply %v7255, %v7257 : tensor<1536x384x1x1xf32>
    %v7259 = stablehlo.add %v7256, %v7258 : tensor<1536x384x1x1xf32>
    %v7260 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7261 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7262 = stablehlo.divide %v7253, %v7260 : tensor<1536x384x1x1xf32>
    %v7263 = stablehlo.divide %v7259, %v7261 : tensor<1536x384x1x1xf32>
    %v7264 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7265 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7266 = stablehlo.sqrt %v7263 : tensor<1536x384x1x1xf32>
    %v7267 = stablehlo.add %v7266, %v7265 : tensor<1536x384x1x1xf32>
    %v7268 = stablehlo.divide %v7262, %v7267 : tensor<1536x384x1x1xf32>
    %v7269 = stablehlo.multiply %v7264, %v7268 : tensor<1536x384x1x1xf32>
    %v7270 = stablehlo.subtract %s2b4eW, %v7269 : tensor<1536x384x1x1xf32>
    %v7271 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7272 = stablehlo.multiply %v7271, %v7264 : tensor<1536x384x1x1xf32>
    %v7273 = stablehlo.multiply %v7272, %s2b4eW : tensor<1536x384x1x1xf32>
    %v7274 = stablehlo.subtract %v7270, %v7273 : tensor<1536x384x1x1xf32>
    %v7275 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7276 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7277 = stablehlo.multiply %v7275, %s2b4ebm : tensor<1536xf32>
    %v7278 = stablehlo.multiply %v7276, %v2059 : tensor<1536xf32>
    %v7279 = stablehlo.add %v7277, %v7278 : tensor<1536xf32>
    %v7280 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7281 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7282 = stablehlo.multiply %v7280, %s2b4ebv : tensor<1536xf32>
    %v7283 = stablehlo.multiply %v2059, %v2059 : tensor<1536xf32>
    %v7284 = stablehlo.multiply %v7281, %v7283 : tensor<1536xf32>
    %v7285 = stablehlo.add %v7282, %v7284 : tensor<1536xf32>
    %v7286 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7287 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7288 = stablehlo.multiply %v7286, %s2b4ebm : tensor<1536xf32>
    %v7289 = stablehlo.multiply %v7287, %v2059 : tensor<1536xf32>
    %v7290 = stablehlo.add %v7288, %v7289 : tensor<1536xf32>
    %v7291 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7292 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7293 = stablehlo.multiply %v7291, %s2b4ebv : tensor<1536xf32>
    %v7294 = stablehlo.multiply %v2059, %v2059 : tensor<1536xf32>
    %v7295 = stablehlo.multiply %v7292, %v7294 : tensor<1536xf32>
    %v7296 = stablehlo.add %v7293, %v7295 : tensor<1536xf32>
    %v7297 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7298 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7299 = stablehlo.divide %v7290, %v7297 : tensor<1536xf32>
    %v7300 = stablehlo.divide %v7296, %v7298 : tensor<1536xf32>
    %v7301 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7302 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7303 = stablehlo.sqrt %v7300 : tensor<1536xf32>
    %v7304 = stablehlo.add %v7303, %v7302 : tensor<1536xf32>
    %v7305 = stablehlo.divide %v7299, %v7304 : tensor<1536xf32>
    %v7306 = stablehlo.multiply %v7301, %v7305 : tensor<1536xf32>
    %v7307 = stablehlo.subtract %s2b4eb, %v7306 : tensor<1536xf32>
    %v7308 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7309 = stablehlo.multiply %v7308, %v7301 : tensor<1536xf32>
    %v7310 = stablehlo.multiply %v7309, %s2b4eb : tensor<1536xf32>
    %v7311 = stablehlo.subtract %v7307, %v7310 : tensor<1536xf32>
    %v7312 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7313 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7314 = stablehlo.multiply %v7312, %s2b4pWm : tensor<384x1536x1x1xf32>
    %v7315 = stablehlo.multiply %v7313, %v2047 : tensor<384x1536x1x1xf32>
    %v7316 = stablehlo.add %v7314, %v7315 : tensor<384x1536x1x1xf32>
    %v7317 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7318 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7319 = stablehlo.multiply %v7317, %s2b4pWv : tensor<384x1536x1x1xf32>
    %v7320 = stablehlo.multiply %v2047, %v2047 : tensor<384x1536x1x1xf32>
    %v7321 = stablehlo.multiply %v7318, %v7320 : tensor<384x1536x1x1xf32>
    %v7322 = stablehlo.add %v7319, %v7321 : tensor<384x1536x1x1xf32>
    %v7323 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7324 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7325 = stablehlo.multiply %v7323, %s2b4pWm : tensor<384x1536x1x1xf32>
    %v7326 = stablehlo.multiply %v7324, %v2047 : tensor<384x1536x1x1xf32>
    %v7327 = stablehlo.add %v7325, %v7326 : tensor<384x1536x1x1xf32>
    %v7328 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7329 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7330 = stablehlo.multiply %v7328, %s2b4pWv : tensor<384x1536x1x1xf32>
    %v7331 = stablehlo.multiply %v2047, %v2047 : tensor<384x1536x1x1xf32>
    %v7332 = stablehlo.multiply %v7329, %v7331 : tensor<384x1536x1x1xf32>
    %v7333 = stablehlo.add %v7330, %v7332 : tensor<384x1536x1x1xf32>
    %v7334 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7335 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7336 = stablehlo.divide %v7327, %v7334 : tensor<384x1536x1x1xf32>
    %v7337 = stablehlo.divide %v7333, %v7335 : tensor<384x1536x1x1xf32>
    %v7338 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7339 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7340 = stablehlo.sqrt %v7337 : tensor<384x1536x1x1xf32>
    %v7341 = stablehlo.add %v7340, %v7339 : tensor<384x1536x1x1xf32>
    %v7342 = stablehlo.divide %v7336, %v7341 : tensor<384x1536x1x1xf32>
    %v7343 = stablehlo.multiply %v7338, %v7342 : tensor<384x1536x1x1xf32>
    %v7344 = stablehlo.subtract %s2b4pW, %v7343 : tensor<384x1536x1x1xf32>
    %v7345 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7346 = stablehlo.multiply %v7345, %v7338 : tensor<384x1536x1x1xf32>
    %v7347 = stablehlo.multiply %v7346, %s2b4pW : tensor<384x1536x1x1xf32>
    %v7348 = stablehlo.subtract %v7344, %v7347 : tensor<384x1536x1x1xf32>
    %v7349 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7350 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7351 = stablehlo.multiply %v7349, %s2b4pbm : tensor<384xf32>
    %v7352 = stablehlo.multiply %v7350, %v2050 : tensor<384xf32>
    %v7353 = stablehlo.add %v7351, %v7352 : tensor<384xf32>
    %v7354 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7355 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7356 = stablehlo.multiply %v7354, %s2b4pbv : tensor<384xf32>
    %v7357 = stablehlo.multiply %v2050, %v2050 : tensor<384xf32>
    %v7358 = stablehlo.multiply %v7355, %v7357 : tensor<384xf32>
    %v7359 = stablehlo.add %v7356, %v7358 : tensor<384xf32>
    %v7360 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7361 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7362 = stablehlo.multiply %v7360, %s2b4pbm : tensor<384xf32>
    %v7363 = stablehlo.multiply %v7361, %v2050 : tensor<384xf32>
    %v7364 = stablehlo.add %v7362, %v7363 : tensor<384xf32>
    %v7365 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7366 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7367 = stablehlo.multiply %v7365, %s2b4pbv : tensor<384xf32>
    %v7368 = stablehlo.multiply %v2050, %v2050 : tensor<384xf32>
    %v7369 = stablehlo.multiply %v7366, %v7368 : tensor<384xf32>
    %v7370 = stablehlo.add %v7367, %v7369 : tensor<384xf32>
    %v7371 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7372 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7373 = stablehlo.divide %v7364, %v7371 : tensor<384xf32>
    %v7374 = stablehlo.divide %v7370, %v7372 : tensor<384xf32>
    %v7375 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7376 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7377 = stablehlo.sqrt %v7374 : tensor<384xf32>
    %v7378 = stablehlo.add %v7377, %v7376 : tensor<384xf32>
    %v7379 = stablehlo.divide %v7373, %v7378 : tensor<384xf32>
    %v7380 = stablehlo.multiply %v7375, %v7379 : tensor<384xf32>
    %v7381 = stablehlo.subtract %s2b4pb, %v7380 : tensor<384xf32>
    %v7382 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7383 = stablehlo.multiply %v7382, %v7375 : tensor<384xf32>
    %v7384 = stablehlo.multiply %v7383, %s2b4pb : tensor<384xf32>
    %v7385 = stablehlo.subtract %v7381, %v7384 : tensor<384xf32>
    %v7386 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7387 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7388 = stablehlo.multiply %v7386, %s2b4lgm : tensor<384xf32>
    %v7389 = stablehlo.multiply %v7387, %v2041 : tensor<384xf32>
    %v7390 = stablehlo.add %v7388, %v7389 : tensor<384xf32>
    %v7391 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7392 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7393 = stablehlo.multiply %v7391, %s2b4lgv : tensor<384xf32>
    %v7394 = stablehlo.multiply %v2041, %v2041 : tensor<384xf32>
    %v7395 = stablehlo.multiply %v7392, %v7394 : tensor<384xf32>
    %v7396 = stablehlo.add %v7393, %v7395 : tensor<384xf32>
    %v7397 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7398 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7399 = stablehlo.multiply %v7397, %s2b4lgm : tensor<384xf32>
    %v7400 = stablehlo.multiply %v7398, %v2041 : tensor<384xf32>
    %v7401 = stablehlo.add %v7399, %v7400 : tensor<384xf32>
    %v7402 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7403 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7404 = stablehlo.multiply %v7402, %s2b4lgv : tensor<384xf32>
    %v7405 = stablehlo.multiply %v2041, %v2041 : tensor<384xf32>
    %v7406 = stablehlo.multiply %v7403, %v7405 : tensor<384xf32>
    %v7407 = stablehlo.add %v7404, %v7406 : tensor<384xf32>
    %v7408 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7409 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7410 = stablehlo.divide %v7401, %v7408 : tensor<384xf32>
    %v7411 = stablehlo.divide %v7407, %v7409 : tensor<384xf32>
    %v7412 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7413 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7414 = stablehlo.sqrt %v7411 : tensor<384xf32>
    %v7415 = stablehlo.add %v7414, %v7413 : tensor<384xf32>
    %v7416 = stablehlo.divide %v7410, %v7415 : tensor<384xf32>
    %v7417 = stablehlo.multiply %v7412, %v7416 : tensor<384xf32>
    %v7418 = stablehlo.subtract %s2b4lg, %v7417 : tensor<384xf32>
    %v7419 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7420 = stablehlo.multiply %v7419, %v7412 : tensor<384xf32>
    %v7421 = stablehlo.multiply %v7420, %s2b4lg : tensor<384xf32>
    %v7422 = stablehlo.subtract %v7418, %v7421 : tensor<384xf32>
    %v7423 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7424 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7425 = stablehlo.multiply %v7423, %s2b5dWm : tensor<384x1x7x7xf32>
    %v7426 = stablehlo.multiply %v7424, %v1964 : tensor<384x1x7x7xf32>
    %v7427 = stablehlo.add %v7425, %v7426 : tensor<384x1x7x7xf32>
    %v7428 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7429 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7430 = stablehlo.multiply %v7428, %s2b5dWv : tensor<384x1x7x7xf32>
    %v7431 = stablehlo.multiply %v1964, %v1964 : tensor<384x1x7x7xf32>
    %v7432 = stablehlo.multiply %v7429, %v7431 : tensor<384x1x7x7xf32>
    %v7433 = stablehlo.add %v7430, %v7432 : tensor<384x1x7x7xf32>
    %v7434 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7435 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7436 = stablehlo.multiply %v7434, %s2b5dWm : tensor<384x1x7x7xf32>
    %v7437 = stablehlo.multiply %v7435, %v1964 : tensor<384x1x7x7xf32>
    %v7438 = stablehlo.add %v7436, %v7437 : tensor<384x1x7x7xf32>
    %v7439 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7440 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7441 = stablehlo.multiply %v7439, %s2b5dWv : tensor<384x1x7x7xf32>
    %v7442 = stablehlo.multiply %v1964, %v1964 : tensor<384x1x7x7xf32>
    %v7443 = stablehlo.multiply %v7440, %v7442 : tensor<384x1x7x7xf32>
    %v7444 = stablehlo.add %v7441, %v7443 : tensor<384x1x7x7xf32>
    %v7445 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7446 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7447 = stablehlo.divide %v7438, %v7445 : tensor<384x1x7x7xf32>
    %v7448 = stablehlo.divide %v7444, %v7446 : tensor<384x1x7x7xf32>
    %v7449 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7450 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7451 = stablehlo.sqrt %v7448 : tensor<384x1x7x7xf32>
    %v7452 = stablehlo.add %v7451, %v7450 : tensor<384x1x7x7xf32>
    %v7453 = stablehlo.divide %v7447, %v7452 : tensor<384x1x7x7xf32>
    %v7454 = stablehlo.multiply %v7449, %v7453 : tensor<384x1x7x7xf32>
    %v7455 = stablehlo.subtract %s2b5dW, %v7454 : tensor<384x1x7x7xf32>
    %v7456 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7457 = stablehlo.multiply %v7456, %v7449 : tensor<384x1x7x7xf32>
    %v7458 = stablehlo.multiply %v7457, %s2b5dW : tensor<384x1x7x7xf32>
    %v7459 = stablehlo.subtract %v7455, %v7458 : tensor<384x1x7x7xf32>
    %v7460 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7461 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7462 = stablehlo.multiply %v7460, %s2b5dbm : tensor<384xf32>
    %v7463 = stablehlo.multiply %v7461, %v1967 : tensor<384xf32>
    %v7464 = stablehlo.add %v7462, %v7463 : tensor<384xf32>
    %v7465 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7466 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7467 = stablehlo.multiply %v7465, %s2b5dbv : tensor<384xf32>
    %v7468 = stablehlo.multiply %v1967, %v1967 : tensor<384xf32>
    %v7469 = stablehlo.multiply %v7466, %v7468 : tensor<384xf32>
    %v7470 = stablehlo.add %v7467, %v7469 : tensor<384xf32>
    %v7471 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7472 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7473 = stablehlo.multiply %v7471, %s2b5dbm : tensor<384xf32>
    %v7474 = stablehlo.multiply %v7472, %v1967 : tensor<384xf32>
    %v7475 = stablehlo.add %v7473, %v7474 : tensor<384xf32>
    %v7476 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7477 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7478 = stablehlo.multiply %v7476, %s2b5dbv : tensor<384xf32>
    %v7479 = stablehlo.multiply %v1967, %v1967 : tensor<384xf32>
    %v7480 = stablehlo.multiply %v7477, %v7479 : tensor<384xf32>
    %v7481 = stablehlo.add %v7478, %v7480 : tensor<384xf32>
    %v7482 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7483 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7484 = stablehlo.divide %v7475, %v7482 : tensor<384xf32>
    %v7485 = stablehlo.divide %v7481, %v7483 : tensor<384xf32>
    %v7486 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7487 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7488 = stablehlo.sqrt %v7485 : tensor<384xf32>
    %v7489 = stablehlo.add %v7488, %v7487 : tensor<384xf32>
    %v7490 = stablehlo.divide %v7484, %v7489 : tensor<384xf32>
    %v7491 = stablehlo.multiply %v7486, %v7490 : tensor<384xf32>
    %v7492 = stablehlo.subtract %s2b5db, %v7491 : tensor<384xf32>
    %v7493 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7494 = stablehlo.multiply %v7493, %v7486 : tensor<384xf32>
    %v7495 = stablehlo.multiply %v7494, %s2b5db : tensor<384xf32>
    %v7496 = stablehlo.subtract %v7492, %v7495 : tensor<384xf32>
    %v7497 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7498 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7499 = stablehlo.multiply %v7497, %s2b5ngm : tensor<f32>
    %v7500 = stablehlo.multiply %v7498, %v1956 : tensor<f32>
    %v7501 = stablehlo.add %v7499, %v7500 : tensor<f32>
    %v7502 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7503 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7504 = stablehlo.multiply %v7502, %s2b5ngv : tensor<f32>
    %v7505 = stablehlo.multiply %v1956, %v1956 : tensor<f32>
    %v7506 = stablehlo.multiply %v7503, %v7505 : tensor<f32>
    %v7507 = stablehlo.add %v7504, %v7506 : tensor<f32>
    %v7508 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7509 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7510 = stablehlo.multiply %v7508, %s2b5ngm : tensor<f32>
    %v7511 = stablehlo.multiply %v7509, %v1956 : tensor<f32>
    %v7512 = stablehlo.add %v7510, %v7511 : tensor<f32>
    %v7513 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7514 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7515 = stablehlo.multiply %v7513, %s2b5ngv : tensor<f32>
    %v7516 = stablehlo.multiply %v1956, %v1956 : tensor<f32>
    %v7517 = stablehlo.multiply %v7514, %v7516 : tensor<f32>
    %v7518 = stablehlo.add %v7515, %v7517 : tensor<f32>
    %v7519 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7520 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7521 = stablehlo.divide %v7512, %v7519 : tensor<f32>
    %v7522 = stablehlo.divide %v7518, %v7520 : tensor<f32>
    %v7523 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7524 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7525 = stablehlo.sqrt %v7522 : tensor<f32>
    %v7526 = stablehlo.add %v7525, %v7524 : tensor<f32>
    %v7527 = stablehlo.divide %v7521, %v7526 : tensor<f32>
    %v7528 = stablehlo.multiply %v7523, %v7527 : tensor<f32>
    %v7529 = stablehlo.subtract %s2b5ng, %v7528 : tensor<f32>
    %v7530 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7531 = stablehlo.multiply %v7530, %v7523 : tensor<f32>
    %v7532 = stablehlo.multiply %v7531, %s2b5ng : tensor<f32>
    %v7533 = stablehlo.subtract %v7529, %v7532 : tensor<f32>
    %v7534 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7535 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7536 = stablehlo.multiply %v7534, %s2b5nbtm : tensor<f32>
    %v7537 = stablehlo.multiply %v7535, %v1958 : tensor<f32>
    %v7538 = stablehlo.add %v7536, %v7537 : tensor<f32>
    %v7539 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7540 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7541 = stablehlo.multiply %v7539, %s2b5nbtv : tensor<f32>
    %v7542 = stablehlo.multiply %v1958, %v1958 : tensor<f32>
    %v7543 = stablehlo.multiply %v7540, %v7542 : tensor<f32>
    %v7544 = stablehlo.add %v7541, %v7543 : tensor<f32>
    %v7545 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7546 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7547 = stablehlo.multiply %v7545, %s2b5nbtm : tensor<f32>
    %v7548 = stablehlo.multiply %v7546, %v1958 : tensor<f32>
    %v7549 = stablehlo.add %v7547, %v7548 : tensor<f32>
    %v7550 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7551 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7552 = stablehlo.multiply %v7550, %s2b5nbtv : tensor<f32>
    %v7553 = stablehlo.multiply %v1958, %v1958 : tensor<f32>
    %v7554 = stablehlo.multiply %v7551, %v7553 : tensor<f32>
    %v7555 = stablehlo.add %v7552, %v7554 : tensor<f32>
    %v7556 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7557 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7558 = stablehlo.divide %v7549, %v7556 : tensor<f32>
    %v7559 = stablehlo.divide %v7555, %v7557 : tensor<f32>
    %v7560 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7561 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7562 = stablehlo.sqrt %v7559 : tensor<f32>
    %v7563 = stablehlo.add %v7562, %v7561 : tensor<f32>
    %v7564 = stablehlo.divide %v7558, %v7563 : tensor<f32>
    %v7565 = stablehlo.multiply %v7560, %v7564 : tensor<f32>
    %v7566 = stablehlo.subtract %s2b5nbt, %v7565 : tensor<f32>
    %v7567 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7568 = stablehlo.multiply %v7567, %v7560 : tensor<f32>
    %v7569 = stablehlo.multiply %v7568, %s2b5nbt : tensor<f32>
    %v7570 = stablehlo.subtract %v7566, %v7569 : tensor<f32>
    %v7571 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7572 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7573 = stablehlo.multiply %v7571, %s2b5eWm : tensor<1536x384x1x1xf32>
    %v7574 = stablehlo.multiply %v7572, %v1937 : tensor<1536x384x1x1xf32>
    %v7575 = stablehlo.add %v7573, %v7574 : tensor<1536x384x1x1xf32>
    %v7576 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7577 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7578 = stablehlo.multiply %v7576, %s2b5eWv : tensor<1536x384x1x1xf32>
    %v7579 = stablehlo.multiply %v1937, %v1937 : tensor<1536x384x1x1xf32>
    %v7580 = stablehlo.multiply %v7577, %v7579 : tensor<1536x384x1x1xf32>
    %v7581 = stablehlo.add %v7578, %v7580 : tensor<1536x384x1x1xf32>
    %v7582 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7583 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7584 = stablehlo.multiply %v7582, %s2b5eWm : tensor<1536x384x1x1xf32>
    %v7585 = stablehlo.multiply %v7583, %v1937 : tensor<1536x384x1x1xf32>
    %v7586 = stablehlo.add %v7584, %v7585 : tensor<1536x384x1x1xf32>
    %v7587 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7588 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7589 = stablehlo.multiply %v7587, %s2b5eWv : tensor<1536x384x1x1xf32>
    %v7590 = stablehlo.multiply %v1937, %v1937 : tensor<1536x384x1x1xf32>
    %v7591 = stablehlo.multiply %v7588, %v7590 : tensor<1536x384x1x1xf32>
    %v7592 = stablehlo.add %v7589, %v7591 : tensor<1536x384x1x1xf32>
    %v7593 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7594 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7595 = stablehlo.divide %v7586, %v7593 : tensor<1536x384x1x1xf32>
    %v7596 = stablehlo.divide %v7592, %v7594 : tensor<1536x384x1x1xf32>
    %v7597 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7598 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7599 = stablehlo.sqrt %v7596 : tensor<1536x384x1x1xf32>
    %v7600 = stablehlo.add %v7599, %v7598 : tensor<1536x384x1x1xf32>
    %v7601 = stablehlo.divide %v7595, %v7600 : tensor<1536x384x1x1xf32>
    %v7602 = stablehlo.multiply %v7597, %v7601 : tensor<1536x384x1x1xf32>
    %v7603 = stablehlo.subtract %s2b5eW, %v7602 : tensor<1536x384x1x1xf32>
    %v7604 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7605 = stablehlo.multiply %v7604, %v7597 : tensor<1536x384x1x1xf32>
    %v7606 = stablehlo.multiply %v7605, %s2b5eW : tensor<1536x384x1x1xf32>
    %v7607 = stablehlo.subtract %v7603, %v7606 : tensor<1536x384x1x1xf32>
    %v7608 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7609 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7610 = stablehlo.multiply %v7608, %s2b5ebm : tensor<1536xf32>
    %v7611 = stablehlo.multiply %v7609, %v1940 : tensor<1536xf32>
    %v7612 = stablehlo.add %v7610, %v7611 : tensor<1536xf32>
    %v7613 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7614 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7615 = stablehlo.multiply %v7613, %s2b5ebv : tensor<1536xf32>
    %v7616 = stablehlo.multiply %v1940, %v1940 : tensor<1536xf32>
    %v7617 = stablehlo.multiply %v7614, %v7616 : tensor<1536xf32>
    %v7618 = stablehlo.add %v7615, %v7617 : tensor<1536xf32>
    %v7619 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7620 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7621 = stablehlo.multiply %v7619, %s2b5ebm : tensor<1536xf32>
    %v7622 = stablehlo.multiply %v7620, %v1940 : tensor<1536xf32>
    %v7623 = stablehlo.add %v7621, %v7622 : tensor<1536xf32>
    %v7624 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7625 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7626 = stablehlo.multiply %v7624, %s2b5ebv : tensor<1536xf32>
    %v7627 = stablehlo.multiply %v1940, %v1940 : tensor<1536xf32>
    %v7628 = stablehlo.multiply %v7625, %v7627 : tensor<1536xf32>
    %v7629 = stablehlo.add %v7626, %v7628 : tensor<1536xf32>
    %v7630 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7631 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7632 = stablehlo.divide %v7623, %v7630 : tensor<1536xf32>
    %v7633 = stablehlo.divide %v7629, %v7631 : tensor<1536xf32>
    %v7634 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7635 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7636 = stablehlo.sqrt %v7633 : tensor<1536xf32>
    %v7637 = stablehlo.add %v7636, %v7635 : tensor<1536xf32>
    %v7638 = stablehlo.divide %v7632, %v7637 : tensor<1536xf32>
    %v7639 = stablehlo.multiply %v7634, %v7638 : tensor<1536xf32>
    %v7640 = stablehlo.subtract %s2b5eb, %v7639 : tensor<1536xf32>
    %v7641 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7642 = stablehlo.multiply %v7641, %v7634 : tensor<1536xf32>
    %v7643 = stablehlo.multiply %v7642, %s2b5eb : tensor<1536xf32>
    %v7644 = stablehlo.subtract %v7640, %v7643 : tensor<1536xf32>
    %v7645 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7646 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7647 = stablehlo.multiply %v7645, %s2b5pWm : tensor<384x1536x1x1xf32>
    %v7648 = stablehlo.multiply %v7646, %v1928 : tensor<384x1536x1x1xf32>
    %v7649 = stablehlo.add %v7647, %v7648 : tensor<384x1536x1x1xf32>
    %v7650 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7651 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7652 = stablehlo.multiply %v7650, %s2b5pWv : tensor<384x1536x1x1xf32>
    %v7653 = stablehlo.multiply %v1928, %v1928 : tensor<384x1536x1x1xf32>
    %v7654 = stablehlo.multiply %v7651, %v7653 : tensor<384x1536x1x1xf32>
    %v7655 = stablehlo.add %v7652, %v7654 : tensor<384x1536x1x1xf32>
    %v7656 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7657 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7658 = stablehlo.multiply %v7656, %s2b5pWm : tensor<384x1536x1x1xf32>
    %v7659 = stablehlo.multiply %v7657, %v1928 : tensor<384x1536x1x1xf32>
    %v7660 = stablehlo.add %v7658, %v7659 : tensor<384x1536x1x1xf32>
    %v7661 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7662 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7663 = stablehlo.multiply %v7661, %s2b5pWv : tensor<384x1536x1x1xf32>
    %v7664 = stablehlo.multiply %v1928, %v1928 : tensor<384x1536x1x1xf32>
    %v7665 = stablehlo.multiply %v7662, %v7664 : tensor<384x1536x1x1xf32>
    %v7666 = stablehlo.add %v7663, %v7665 : tensor<384x1536x1x1xf32>
    %v7667 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7668 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7669 = stablehlo.divide %v7660, %v7667 : tensor<384x1536x1x1xf32>
    %v7670 = stablehlo.divide %v7666, %v7668 : tensor<384x1536x1x1xf32>
    %v7671 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7672 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7673 = stablehlo.sqrt %v7670 : tensor<384x1536x1x1xf32>
    %v7674 = stablehlo.add %v7673, %v7672 : tensor<384x1536x1x1xf32>
    %v7675 = stablehlo.divide %v7669, %v7674 : tensor<384x1536x1x1xf32>
    %v7676 = stablehlo.multiply %v7671, %v7675 : tensor<384x1536x1x1xf32>
    %v7677 = stablehlo.subtract %s2b5pW, %v7676 : tensor<384x1536x1x1xf32>
    %v7678 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7679 = stablehlo.multiply %v7678, %v7671 : tensor<384x1536x1x1xf32>
    %v7680 = stablehlo.multiply %v7679, %s2b5pW : tensor<384x1536x1x1xf32>
    %v7681 = stablehlo.subtract %v7677, %v7680 : tensor<384x1536x1x1xf32>
    %v7682 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7683 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7684 = stablehlo.multiply %v7682, %s2b5pbm : tensor<384xf32>
    %v7685 = stablehlo.multiply %v7683, %v1931 : tensor<384xf32>
    %v7686 = stablehlo.add %v7684, %v7685 : tensor<384xf32>
    %v7687 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7688 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7689 = stablehlo.multiply %v7687, %s2b5pbv : tensor<384xf32>
    %v7690 = stablehlo.multiply %v1931, %v1931 : tensor<384xf32>
    %v7691 = stablehlo.multiply %v7688, %v7690 : tensor<384xf32>
    %v7692 = stablehlo.add %v7689, %v7691 : tensor<384xf32>
    %v7693 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7694 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7695 = stablehlo.multiply %v7693, %s2b5pbm : tensor<384xf32>
    %v7696 = stablehlo.multiply %v7694, %v1931 : tensor<384xf32>
    %v7697 = stablehlo.add %v7695, %v7696 : tensor<384xf32>
    %v7698 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7699 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7700 = stablehlo.multiply %v7698, %s2b5pbv : tensor<384xf32>
    %v7701 = stablehlo.multiply %v1931, %v1931 : tensor<384xf32>
    %v7702 = stablehlo.multiply %v7699, %v7701 : tensor<384xf32>
    %v7703 = stablehlo.add %v7700, %v7702 : tensor<384xf32>
    %v7704 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7705 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7706 = stablehlo.divide %v7697, %v7704 : tensor<384xf32>
    %v7707 = stablehlo.divide %v7703, %v7705 : tensor<384xf32>
    %v7708 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7709 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7710 = stablehlo.sqrt %v7707 : tensor<384xf32>
    %v7711 = stablehlo.add %v7710, %v7709 : tensor<384xf32>
    %v7712 = stablehlo.divide %v7706, %v7711 : tensor<384xf32>
    %v7713 = stablehlo.multiply %v7708, %v7712 : tensor<384xf32>
    %v7714 = stablehlo.subtract %s2b5pb, %v7713 : tensor<384xf32>
    %v7715 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7716 = stablehlo.multiply %v7715, %v7708 : tensor<384xf32>
    %v7717 = stablehlo.multiply %v7716, %s2b5pb : tensor<384xf32>
    %v7718 = stablehlo.subtract %v7714, %v7717 : tensor<384xf32>
    %v7719 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7720 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7721 = stablehlo.multiply %v7719, %s2b5lgm : tensor<384xf32>
    %v7722 = stablehlo.multiply %v7720, %v1922 : tensor<384xf32>
    %v7723 = stablehlo.add %v7721, %v7722 : tensor<384xf32>
    %v7724 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7725 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7726 = stablehlo.multiply %v7724, %s2b5lgv : tensor<384xf32>
    %v7727 = stablehlo.multiply %v1922, %v1922 : tensor<384xf32>
    %v7728 = stablehlo.multiply %v7725, %v7727 : tensor<384xf32>
    %v7729 = stablehlo.add %v7726, %v7728 : tensor<384xf32>
    %v7730 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7731 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7732 = stablehlo.multiply %v7730, %s2b5lgm : tensor<384xf32>
    %v7733 = stablehlo.multiply %v7731, %v1922 : tensor<384xf32>
    %v7734 = stablehlo.add %v7732, %v7733 : tensor<384xf32>
    %v7735 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7736 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7737 = stablehlo.multiply %v7735, %s2b5lgv : tensor<384xf32>
    %v7738 = stablehlo.multiply %v1922, %v1922 : tensor<384xf32>
    %v7739 = stablehlo.multiply %v7736, %v7738 : tensor<384xf32>
    %v7740 = stablehlo.add %v7737, %v7739 : tensor<384xf32>
    %v7741 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7742 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7743 = stablehlo.divide %v7734, %v7741 : tensor<384xf32>
    %v7744 = stablehlo.divide %v7740, %v7742 : tensor<384xf32>
    %v7745 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7746 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7747 = stablehlo.sqrt %v7744 : tensor<384xf32>
    %v7748 = stablehlo.add %v7747, %v7746 : tensor<384xf32>
    %v7749 = stablehlo.divide %v7743, %v7748 : tensor<384xf32>
    %v7750 = stablehlo.multiply %v7745, %v7749 : tensor<384xf32>
    %v7751 = stablehlo.subtract %s2b5lg, %v7750 : tensor<384xf32>
    %v7752 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7753 = stablehlo.multiply %v7752, %v7745 : tensor<384xf32>
    %v7754 = stablehlo.multiply %v7753, %s2b5lg : tensor<384xf32>
    %v7755 = stablehlo.subtract %v7751, %v7754 : tensor<384xf32>
    %v7756 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7757 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7758 = stablehlo.multiply %v7756, %s2b6dWm : tensor<384x1x7x7xf32>
    %v7759 = stablehlo.multiply %v7757, %v1845 : tensor<384x1x7x7xf32>
    %v7760 = stablehlo.add %v7758, %v7759 : tensor<384x1x7x7xf32>
    %v7761 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7762 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7763 = stablehlo.multiply %v7761, %s2b6dWv : tensor<384x1x7x7xf32>
    %v7764 = stablehlo.multiply %v1845, %v1845 : tensor<384x1x7x7xf32>
    %v7765 = stablehlo.multiply %v7762, %v7764 : tensor<384x1x7x7xf32>
    %v7766 = stablehlo.add %v7763, %v7765 : tensor<384x1x7x7xf32>
    %v7767 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7768 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7769 = stablehlo.multiply %v7767, %s2b6dWm : tensor<384x1x7x7xf32>
    %v7770 = stablehlo.multiply %v7768, %v1845 : tensor<384x1x7x7xf32>
    %v7771 = stablehlo.add %v7769, %v7770 : tensor<384x1x7x7xf32>
    %v7772 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7773 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7774 = stablehlo.multiply %v7772, %s2b6dWv : tensor<384x1x7x7xf32>
    %v7775 = stablehlo.multiply %v1845, %v1845 : tensor<384x1x7x7xf32>
    %v7776 = stablehlo.multiply %v7773, %v7775 : tensor<384x1x7x7xf32>
    %v7777 = stablehlo.add %v7774, %v7776 : tensor<384x1x7x7xf32>
    %v7778 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7779 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7780 = stablehlo.divide %v7771, %v7778 : tensor<384x1x7x7xf32>
    %v7781 = stablehlo.divide %v7777, %v7779 : tensor<384x1x7x7xf32>
    %v7782 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7783 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7784 = stablehlo.sqrt %v7781 : tensor<384x1x7x7xf32>
    %v7785 = stablehlo.add %v7784, %v7783 : tensor<384x1x7x7xf32>
    %v7786 = stablehlo.divide %v7780, %v7785 : tensor<384x1x7x7xf32>
    %v7787 = stablehlo.multiply %v7782, %v7786 : tensor<384x1x7x7xf32>
    %v7788 = stablehlo.subtract %s2b6dW, %v7787 : tensor<384x1x7x7xf32>
    %v7789 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7790 = stablehlo.multiply %v7789, %v7782 : tensor<384x1x7x7xf32>
    %v7791 = stablehlo.multiply %v7790, %s2b6dW : tensor<384x1x7x7xf32>
    %v7792 = stablehlo.subtract %v7788, %v7791 : tensor<384x1x7x7xf32>
    %v7793 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7794 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7795 = stablehlo.multiply %v7793, %s2b6dbm : tensor<384xf32>
    %v7796 = stablehlo.multiply %v7794, %v1848 : tensor<384xf32>
    %v7797 = stablehlo.add %v7795, %v7796 : tensor<384xf32>
    %v7798 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7799 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7800 = stablehlo.multiply %v7798, %s2b6dbv : tensor<384xf32>
    %v7801 = stablehlo.multiply %v1848, %v1848 : tensor<384xf32>
    %v7802 = stablehlo.multiply %v7799, %v7801 : tensor<384xf32>
    %v7803 = stablehlo.add %v7800, %v7802 : tensor<384xf32>
    %v7804 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7805 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7806 = stablehlo.multiply %v7804, %s2b6dbm : tensor<384xf32>
    %v7807 = stablehlo.multiply %v7805, %v1848 : tensor<384xf32>
    %v7808 = stablehlo.add %v7806, %v7807 : tensor<384xf32>
    %v7809 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7810 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7811 = stablehlo.multiply %v7809, %s2b6dbv : tensor<384xf32>
    %v7812 = stablehlo.multiply %v1848, %v1848 : tensor<384xf32>
    %v7813 = stablehlo.multiply %v7810, %v7812 : tensor<384xf32>
    %v7814 = stablehlo.add %v7811, %v7813 : tensor<384xf32>
    %v7815 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7816 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7817 = stablehlo.divide %v7808, %v7815 : tensor<384xf32>
    %v7818 = stablehlo.divide %v7814, %v7816 : tensor<384xf32>
    %v7819 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7820 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7821 = stablehlo.sqrt %v7818 : tensor<384xf32>
    %v7822 = stablehlo.add %v7821, %v7820 : tensor<384xf32>
    %v7823 = stablehlo.divide %v7817, %v7822 : tensor<384xf32>
    %v7824 = stablehlo.multiply %v7819, %v7823 : tensor<384xf32>
    %v7825 = stablehlo.subtract %s2b6db, %v7824 : tensor<384xf32>
    %v7826 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7827 = stablehlo.multiply %v7826, %v7819 : tensor<384xf32>
    %v7828 = stablehlo.multiply %v7827, %s2b6db : tensor<384xf32>
    %v7829 = stablehlo.subtract %v7825, %v7828 : tensor<384xf32>
    %v7830 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7831 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7832 = stablehlo.multiply %v7830, %s2b6ngm : tensor<f32>
    %v7833 = stablehlo.multiply %v7831, %v1837 : tensor<f32>
    %v7834 = stablehlo.add %v7832, %v7833 : tensor<f32>
    %v7835 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7836 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7837 = stablehlo.multiply %v7835, %s2b6ngv : tensor<f32>
    %v7838 = stablehlo.multiply %v1837, %v1837 : tensor<f32>
    %v7839 = stablehlo.multiply %v7836, %v7838 : tensor<f32>
    %v7840 = stablehlo.add %v7837, %v7839 : tensor<f32>
    %v7841 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7842 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7843 = stablehlo.multiply %v7841, %s2b6ngm : tensor<f32>
    %v7844 = stablehlo.multiply %v7842, %v1837 : tensor<f32>
    %v7845 = stablehlo.add %v7843, %v7844 : tensor<f32>
    %v7846 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7847 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7848 = stablehlo.multiply %v7846, %s2b6ngv : tensor<f32>
    %v7849 = stablehlo.multiply %v1837, %v1837 : tensor<f32>
    %v7850 = stablehlo.multiply %v7847, %v7849 : tensor<f32>
    %v7851 = stablehlo.add %v7848, %v7850 : tensor<f32>
    %v7852 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7853 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7854 = stablehlo.divide %v7845, %v7852 : tensor<f32>
    %v7855 = stablehlo.divide %v7851, %v7853 : tensor<f32>
    %v7856 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7857 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7858 = stablehlo.sqrt %v7855 : tensor<f32>
    %v7859 = stablehlo.add %v7858, %v7857 : tensor<f32>
    %v7860 = stablehlo.divide %v7854, %v7859 : tensor<f32>
    %v7861 = stablehlo.multiply %v7856, %v7860 : tensor<f32>
    %v7862 = stablehlo.subtract %s2b6ng, %v7861 : tensor<f32>
    %v7863 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7864 = stablehlo.multiply %v7863, %v7856 : tensor<f32>
    %v7865 = stablehlo.multiply %v7864, %s2b6ng : tensor<f32>
    %v7866 = stablehlo.subtract %v7862, %v7865 : tensor<f32>
    %v7867 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7868 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7869 = stablehlo.multiply %v7867, %s2b6nbtm : tensor<f32>
    %v7870 = stablehlo.multiply %v7868, %v1839 : tensor<f32>
    %v7871 = stablehlo.add %v7869, %v7870 : tensor<f32>
    %v7872 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7873 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7874 = stablehlo.multiply %v7872, %s2b6nbtv : tensor<f32>
    %v7875 = stablehlo.multiply %v1839, %v1839 : tensor<f32>
    %v7876 = stablehlo.multiply %v7873, %v7875 : tensor<f32>
    %v7877 = stablehlo.add %v7874, %v7876 : tensor<f32>
    %v7878 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7879 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7880 = stablehlo.multiply %v7878, %s2b6nbtm : tensor<f32>
    %v7881 = stablehlo.multiply %v7879, %v1839 : tensor<f32>
    %v7882 = stablehlo.add %v7880, %v7881 : tensor<f32>
    %v7883 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7884 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7885 = stablehlo.multiply %v7883, %s2b6nbtv : tensor<f32>
    %v7886 = stablehlo.multiply %v1839, %v1839 : tensor<f32>
    %v7887 = stablehlo.multiply %v7884, %v7886 : tensor<f32>
    %v7888 = stablehlo.add %v7885, %v7887 : tensor<f32>
    %v7889 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7890 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7891 = stablehlo.divide %v7882, %v7889 : tensor<f32>
    %v7892 = stablehlo.divide %v7888, %v7890 : tensor<f32>
    %v7893 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7894 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7895 = stablehlo.sqrt %v7892 : tensor<f32>
    %v7896 = stablehlo.add %v7895, %v7894 : tensor<f32>
    %v7897 = stablehlo.divide %v7891, %v7896 : tensor<f32>
    %v7898 = stablehlo.multiply %v7893, %v7897 : tensor<f32>
    %v7899 = stablehlo.subtract %s2b6nbt, %v7898 : tensor<f32>
    %v7900 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v7901 = stablehlo.multiply %v7900, %v7893 : tensor<f32>
    %v7902 = stablehlo.multiply %v7901, %s2b6nbt : tensor<f32>
    %v7903 = stablehlo.subtract %v7899, %v7902 : tensor<f32>
    %v7904 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7905 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7906 = stablehlo.multiply %v7904, %s2b6eWm : tensor<1536x384x1x1xf32>
    %v7907 = stablehlo.multiply %v7905, %v1818 : tensor<1536x384x1x1xf32>
    %v7908 = stablehlo.add %v7906, %v7907 : tensor<1536x384x1x1xf32>
    %v7909 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7910 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7911 = stablehlo.multiply %v7909, %s2b6eWv : tensor<1536x384x1x1xf32>
    %v7912 = stablehlo.multiply %v1818, %v1818 : tensor<1536x384x1x1xf32>
    %v7913 = stablehlo.multiply %v7910, %v7912 : tensor<1536x384x1x1xf32>
    %v7914 = stablehlo.add %v7911, %v7913 : tensor<1536x384x1x1xf32>
    %v7915 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7916 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7917 = stablehlo.multiply %v7915, %s2b6eWm : tensor<1536x384x1x1xf32>
    %v7918 = stablehlo.multiply %v7916, %v1818 : tensor<1536x384x1x1xf32>
    %v7919 = stablehlo.add %v7917, %v7918 : tensor<1536x384x1x1xf32>
    %v7920 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7921 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7922 = stablehlo.multiply %v7920, %s2b6eWv : tensor<1536x384x1x1xf32>
    %v7923 = stablehlo.multiply %v1818, %v1818 : tensor<1536x384x1x1xf32>
    %v7924 = stablehlo.multiply %v7921, %v7923 : tensor<1536x384x1x1xf32>
    %v7925 = stablehlo.add %v7922, %v7924 : tensor<1536x384x1x1xf32>
    %v7926 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7927 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7928 = stablehlo.divide %v7919, %v7926 : tensor<1536x384x1x1xf32>
    %v7929 = stablehlo.divide %v7925, %v7927 : tensor<1536x384x1x1xf32>
    %v7930 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7931 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7932 = stablehlo.sqrt %v7929 : tensor<1536x384x1x1xf32>
    %v7933 = stablehlo.add %v7932, %v7931 : tensor<1536x384x1x1xf32>
    %v7934 = stablehlo.divide %v7928, %v7933 : tensor<1536x384x1x1xf32>
    %v7935 = stablehlo.multiply %v7930, %v7934 : tensor<1536x384x1x1xf32>
    %v7936 = stablehlo.subtract %s2b6eW, %v7935 : tensor<1536x384x1x1xf32>
    %v7937 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7938 = stablehlo.multiply %v7937, %v7930 : tensor<1536x384x1x1xf32>
    %v7939 = stablehlo.multiply %v7938, %s2b6eW : tensor<1536x384x1x1xf32>
    %v7940 = stablehlo.subtract %v7936, %v7939 : tensor<1536x384x1x1xf32>
    %v7941 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7942 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7943 = stablehlo.multiply %v7941, %s2b6ebm : tensor<1536xf32>
    %v7944 = stablehlo.multiply %v7942, %v1821 : tensor<1536xf32>
    %v7945 = stablehlo.add %v7943, %v7944 : tensor<1536xf32>
    %v7946 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7947 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7948 = stablehlo.multiply %v7946, %s2b6ebv : tensor<1536xf32>
    %v7949 = stablehlo.multiply %v1821, %v1821 : tensor<1536xf32>
    %v7950 = stablehlo.multiply %v7947, %v7949 : tensor<1536xf32>
    %v7951 = stablehlo.add %v7948, %v7950 : tensor<1536xf32>
    %v7952 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7953 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7954 = stablehlo.multiply %v7952, %s2b6ebm : tensor<1536xf32>
    %v7955 = stablehlo.multiply %v7953, %v1821 : tensor<1536xf32>
    %v7956 = stablehlo.add %v7954, %v7955 : tensor<1536xf32>
    %v7957 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7958 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7959 = stablehlo.multiply %v7957, %s2b6ebv : tensor<1536xf32>
    %v7960 = stablehlo.multiply %v1821, %v1821 : tensor<1536xf32>
    %v7961 = stablehlo.multiply %v7958, %v7960 : tensor<1536xf32>
    %v7962 = stablehlo.add %v7959, %v7961 : tensor<1536xf32>
    %v7963 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7964 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7965 = stablehlo.divide %v7956, %v7963 : tensor<1536xf32>
    %v7966 = stablehlo.divide %v7962, %v7964 : tensor<1536xf32>
    %v7967 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7968 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7969 = stablehlo.sqrt %v7966 : tensor<1536xf32>
    %v7970 = stablehlo.add %v7969, %v7968 : tensor<1536xf32>
    %v7971 = stablehlo.divide %v7965, %v7970 : tensor<1536xf32>
    %v7972 = stablehlo.multiply %v7967, %v7971 : tensor<1536xf32>
    %v7973 = stablehlo.subtract %s2b6eb, %v7972 : tensor<1536xf32>
    %v7974 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7975 = stablehlo.multiply %v7974, %v7967 : tensor<1536xf32>
    %v7976 = stablehlo.multiply %v7975, %s2b6eb : tensor<1536xf32>
    %v7977 = stablehlo.subtract %v7973, %v7976 : tensor<1536xf32>
    %v7978 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7979 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7980 = stablehlo.multiply %v7978, %s2b6pWm : tensor<384x1536x1x1xf32>
    %v7981 = stablehlo.multiply %v7979, %v1809 : tensor<384x1536x1x1xf32>
    %v7982 = stablehlo.add %v7980, %v7981 : tensor<384x1536x1x1xf32>
    %v7983 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7984 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7985 = stablehlo.multiply %v7983, %s2b6pWv : tensor<384x1536x1x1xf32>
    %v7986 = stablehlo.multiply %v1809, %v1809 : tensor<384x1536x1x1xf32>
    %v7987 = stablehlo.multiply %v7984, %v7986 : tensor<384x1536x1x1xf32>
    %v7988 = stablehlo.add %v7985, %v7987 : tensor<384x1536x1x1xf32>
    %v7989 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7990 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7991 = stablehlo.multiply %v7989, %s2b6pWm : tensor<384x1536x1x1xf32>
    %v7992 = stablehlo.multiply %v7990, %v1809 : tensor<384x1536x1x1xf32>
    %v7993 = stablehlo.add %v7991, %v7992 : tensor<384x1536x1x1xf32>
    %v7994 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7995 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7996 = stablehlo.multiply %v7994, %s2b6pWv : tensor<384x1536x1x1xf32>
    %v7997 = stablehlo.multiply %v1809, %v1809 : tensor<384x1536x1x1xf32>
    %v7998 = stablehlo.multiply %v7995, %v7997 : tensor<384x1536x1x1xf32>
    %v7999 = stablehlo.add %v7996, %v7998 : tensor<384x1536x1x1xf32>
    %v8000 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8001 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8002 = stablehlo.divide %v7993, %v8000 : tensor<384x1536x1x1xf32>
    %v8003 = stablehlo.divide %v7999, %v8001 : tensor<384x1536x1x1xf32>
    %v8004 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8005 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8006 = stablehlo.sqrt %v8003 : tensor<384x1536x1x1xf32>
    %v8007 = stablehlo.add %v8006, %v8005 : tensor<384x1536x1x1xf32>
    %v8008 = stablehlo.divide %v8002, %v8007 : tensor<384x1536x1x1xf32>
    %v8009 = stablehlo.multiply %v8004, %v8008 : tensor<384x1536x1x1xf32>
    %v8010 = stablehlo.subtract %s2b6pW, %v8009 : tensor<384x1536x1x1xf32>
    %v8011 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8012 = stablehlo.multiply %v8011, %v8004 : tensor<384x1536x1x1xf32>
    %v8013 = stablehlo.multiply %v8012, %s2b6pW : tensor<384x1536x1x1xf32>
    %v8014 = stablehlo.subtract %v8010, %v8013 : tensor<384x1536x1x1xf32>
    %v8015 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8016 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8017 = stablehlo.multiply %v8015, %s2b6pbm : tensor<384xf32>
    %v8018 = stablehlo.multiply %v8016, %v1812 : tensor<384xf32>
    %v8019 = stablehlo.add %v8017, %v8018 : tensor<384xf32>
    %v8020 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8021 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8022 = stablehlo.multiply %v8020, %s2b6pbv : tensor<384xf32>
    %v8023 = stablehlo.multiply %v1812, %v1812 : tensor<384xf32>
    %v8024 = stablehlo.multiply %v8021, %v8023 : tensor<384xf32>
    %v8025 = stablehlo.add %v8022, %v8024 : tensor<384xf32>
    %v8026 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8027 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8028 = stablehlo.multiply %v8026, %s2b6pbm : tensor<384xf32>
    %v8029 = stablehlo.multiply %v8027, %v1812 : tensor<384xf32>
    %v8030 = stablehlo.add %v8028, %v8029 : tensor<384xf32>
    %v8031 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8032 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8033 = stablehlo.multiply %v8031, %s2b6pbv : tensor<384xf32>
    %v8034 = stablehlo.multiply %v1812, %v1812 : tensor<384xf32>
    %v8035 = stablehlo.multiply %v8032, %v8034 : tensor<384xf32>
    %v8036 = stablehlo.add %v8033, %v8035 : tensor<384xf32>
    %v8037 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8038 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8039 = stablehlo.divide %v8030, %v8037 : tensor<384xf32>
    %v8040 = stablehlo.divide %v8036, %v8038 : tensor<384xf32>
    %v8041 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8042 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8043 = stablehlo.sqrt %v8040 : tensor<384xf32>
    %v8044 = stablehlo.add %v8043, %v8042 : tensor<384xf32>
    %v8045 = stablehlo.divide %v8039, %v8044 : tensor<384xf32>
    %v8046 = stablehlo.multiply %v8041, %v8045 : tensor<384xf32>
    %v8047 = stablehlo.subtract %s2b6pb, %v8046 : tensor<384xf32>
    %v8048 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8049 = stablehlo.multiply %v8048, %v8041 : tensor<384xf32>
    %v8050 = stablehlo.multiply %v8049, %s2b6pb : tensor<384xf32>
    %v8051 = stablehlo.subtract %v8047, %v8050 : tensor<384xf32>
    %v8052 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8053 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8054 = stablehlo.multiply %v8052, %s2b6lgm : tensor<384xf32>
    %v8055 = stablehlo.multiply %v8053, %v1803 : tensor<384xf32>
    %v8056 = stablehlo.add %v8054, %v8055 : tensor<384xf32>
    %v8057 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8058 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8059 = stablehlo.multiply %v8057, %s2b6lgv : tensor<384xf32>
    %v8060 = stablehlo.multiply %v1803, %v1803 : tensor<384xf32>
    %v8061 = stablehlo.multiply %v8058, %v8060 : tensor<384xf32>
    %v8062 = stablehlo.add %v8059, %v8061 : tensor<384xf32>
    %v8063 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8064 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8065 = stablehlo.multiply %v8063, %s2b6lgm : tensor<384xf32>
    %v8066 = stablehlo.multiply %v8064, %v1803 : tensor<384xf32>
    %v8067 = stablehlo.add %v8065, %v8066 : tensor<384xf32>
    %v8068 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8069 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8070 = stablehlo.multiply %v8068, %s2b6lgv : tensor<384xf32>
    %v8071 = stablehlo.multiply %v1803, %v1803 : tensor<384xf32>
    %v8072 = stablehlo.multiply %v8069, %v8071 : tensor<384xf32>
    %v8073 = stablehlo.add %v8070, %v8072 : tensor<384xf32>
    %v8074 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8075 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8076 = stablehlo.divide %v8067, %v8074 : tensor<384xf32>
    %v8077 = stablehlo.divide %v8073, %v8075 : tensor<384xf32>
    %v8078 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8079 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8080 = stablehlo.sqrt %v8077 : tensor<384xf32>
    %v8081 = stablehlo.add %v8080, %v8079 : tensor<384xf32>
    %v8082 = stablehlo.divide %v8076, %v8081 : tensor<384xf32>
    %v8083 = stablehlo.multiply %v8078, %v8082 : tensor<384xf32>
    %v8084 = stablehlo.subtract %s2b6lg, %v8083 : tensor<384xf32>
    %v8085 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8086 = stablehlo.multiply %v8085, %v8078 : tensor<384xf32>
    %v8087 = stablehlo.multiply %v8086, %s2b6lg : tensor<384xf32>
    %v8088 = stablehlo.subtract %v8084, %v8087 : tensor<384xf32>
    %v8089 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8090 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8091 = stablehlo.multiply %v8089, %s2b7dWm : tensor<384x1x7x7xf32>
    %v8092 = stablehlo.multiply %v8090, %v1726 : tensor<384x1x7x7xf32>
    %v8093 = stablehlo.add %v8091, %v8092 : tensor<384x1x7x7xf32>
    %v8094 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8095 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8096 = stablehlo.multiply %v8094, %s2b7dWv : tensor<384x1x7x7xf32>
    %v8097 = stablehlo.multiply %v1726, %v1726 : tensor<384x1x7x7xf32>
    %v8098 = stablehlo.multiply %v8095, %v8097 : tensor<384x1x7x7xf32>
    %v8099 = stablehlo.add %v8096, %v8098 : tensor<384x1x7x7xf32>
    %v8100 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8101 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8102 = stablehlo.multiply %v8100, %s2b7dWm : tensor<384x1x7x7xf32>
    %v8103 = stablehlo.multiply %v8101, %v1726 : tensor<384x1x7x7xf32>
    %v8104 = stablehlo.add %v8102, %v8103 : tensor<384x1x7x7xf32>
    %v8105 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8106 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8107 = stablehlo.multiply %v8105, %s2b7dWv : tensor<384x1x7x7xf32>
    %v8108 = stablehlo.multiply %v1726, %v1726 : tensor<384x1x7x7xf32>
    %v8109 = stablehlo.multiply %v8106, %v8108 : tensor<384x1x7x7xf32>
    %v8110 = stablehlo.add %v8107, %v8109 : tensor<384x1x7x7xf32>
    %v8111 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8112 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8113 = stablehlo.divide %v8104, %v8111 : tensor<384x1x7x7xf32>
    %v8114 = stablehlo.divide %v8110, %v8112 : tensor<384x1x7x7xf32>
    %v8115 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8116 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8117 = stablehlo.sqrt %v8114 : tensor<384x1x7x7xf32>
    %v8118 = stablehlo.add %v8117, %v8116 : tensor<384x1x7x7xf32>
    %v8119 = stablehlo.divide %v8113, %v8118 : tensor<384x1x7x7xf32>
    %v8120 = stablehlo.multiply %v8115, %v8119 : tensor<384x1x7x7xf32>
    %v8121 = stablehlo.subtract %s2b7dW, %v8120 : tensor<384x1x7x7xf32>
    %v8122 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8123 = stablehlo.multiply %v8122, %v8115 : tensor<384x1x7x7xf32>
    %v8124 = stablehlo.multiply %v8123, %s2b7dW : tensor<384x1x7x7xf32>
    %v8125 = stablehlo.subtract %v8121, %v8124 : tensor<384x1x7x7xf32>
    %v8126 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8127 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8128 = stablehlo.multiply %v8126, %s2b7dbm : tensor<384xf32>
    %v8129 = stablehlo.multiply %v8127, %v1729 : tensor<384xf32>
    %v8130 = stablehlo.add %v8128, %v8129 : tensor<384xf32>
    %v8131 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8132 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8133 = stablehlo.multiply %v8131, %s2b7dbv : tensor<384xf32>
    %v8134 = stablehlo.multiply %v1729, %v1729 : tensor<384xf32>
    %v8135 = stablehlo.multiply %v8132, %v8134 : tensor<384xf32>
    %v8136 = stablehlo.add %v8133, %v8135 : tensor<384xf32>
    %v8137 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8138 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8139 = stablehlo.multiply %v8137, %s2b7dbm : tensor<384xf32>
    %v8140 = stablehlo.multiply %v8138, %v1729 : tensor<384xf32>
    %v8141 = stablehlo.add %v8139, %v8140 : tensor<384xf32>
    %v8142 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8143 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8144 = stablehlo.multiply %v8142, %s2b7dbv : tensor<384xf32>
    %v8145 = stablehlo.multiply %v1729, %v1729 : tensor<384xf32>
    %v8146 = stablehlo.multiply %v8143, %v8145 : tensor<384xf32>
    %v8147 = stablehlo.add %v8144, %v8146 : tensor<384xf32>
    %v8148 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8149 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8150 = stablehlo.divide %v8141, %v8148 : tensor<384xf32>
    %v8151 = stablehlo.divide %v8147, %v8149 : tensor<384xf32>
    %v8152 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8153 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8154 = stablehlo.sqrt %v8151 : tensor<384xf32>
    %v8155 = stablehlo.add %v8154, %v8153 : tensor<384xf32>
    %v8156 = stablehlo.divide %v8150, %v8155 : tensor<384xf32>
    %v8157 = stablehlo.multiply %v8152, %v8156 : tensor<384xf32>
    %v8158 = stablehlo.subtract %s2b7db, %v8157 : tensor<384xf32>
    %v8159 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8160 = stablehlo.multiply %v8159, %v8152 : tensor<384xf32>
    %v8161 = stablehlo.multiply %v8160, %s2b7db : tensor<384xf32>
    %v8162 = stablehlo.subtract %v8158, %v8161 : tensor<384xf32>
    %v8163 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8164 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8165 = stablehlo.multiply %v8163, %s2b7ngm : tensor<f32>
    %v8166 = stablehlo.multiply %v8164, %v1718 : tensor<f32>
    %v8167 = stablehlo.add %v8165, %v8166 : tensor<f32>
    %v8168 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8169 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8170 = stablehlo.multiply %v8168, %s2b7ngv : tensor<f32>
    %v8171 = stablehlo.multiply %v1718, %v1718 : tensor<f32>
    %v8172 = stablehlo.multiply %v8169, %v8171 : tensor<f32>
    %v8173 = stablehlo.add %v8170, %v8172 : tensor<f32>
    %v8174 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8175 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8176 = stablehlo.multiply %v8174, %s2b7ngm : tensor<f32>
    %v8177 = stablehlo.multiply %v8175, %v1718 : tensor<f32>
    %v8178 = stablehlo.add %v8176, %v8177 : tensor<f32>
    %v8179 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8180 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8181 = stablehlo.multiply %v8179, %s2b7ngv : tensor<f32>
    %v8182 = stablehlo.multiply %v1718, %v1718 : tensor<f32>
    %v8183 = stablehlo.multiply %v8180, %v8182 : tensor<f32>
    %v8184 = stablehlo.add %v8181, %v8183 : tensor<f32>
    %v8185 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8186 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8187 = stablehlo.divide %v8178, %v8185 : tensor<f32>
    %v8188 = stablehlo.divide %v8184, %v8186 : tensor<f32>
    %v8189 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8190 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8191 = stablehlo.sqrt %v8188 : tensor<f32>
    %v8192 = stablehlo.add %v8191, %v8190 : tensor<f32>
    %v8193 = stablehlo.divide %v8187, %v8192 : tensor<f32>
    %v8194 = stablehlo.multiply %v8189, %v8193 : tensor<f32>
    %v8195 = stablehlo.subtract %s2b7ng, %v8194 : tensor<f32>
    %v8196 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8197 = stablehlo.multiply %v8196, %v8189 : tensor<f32>
    %v8198 = stablehlo.multiply %v8197, %s2b7ng : tensor<f32>
    %v8199 = stablehlo.subtract %v8195, %v8198 : tensor<f32>
    %v8200 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8201 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8202 = stablehlo.multiply %v8200, %s2b7nbtm : tensor<f32>
    %v8203 = stablehlo.multiply %v8201, %v1720 : tensor<f32>
    %v8204 = stablehlo.add %v8202, %v8203 : tensor<f32>
    %v8205 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8206 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8207 = stablehlo.multiply %v8205, %s2b7nbtv : tensor<f32>
    %v8208 = stablehlo.multiply %v1720, %v1720 : tensor<f32>
    %v8209 = stablehlo.multiply %v8206, %v8208 : tensor<f32>
    %v8210 = stablehlo.add %v8207, %v8209 : tensor<f32>
    %v8211 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8212 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8213 = stablehlo.multiply %v8211, %s2b7nbtm : tensor<f32>
    %v8214 = stablehlo.multiply %v8212, %v1720 : tensor<f32>
    %v8215 = stablehlo.add %v8213, %v8214 : tensor<f32>
    %v8216 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8217 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8218 = stablehlo.multiply %v8216, %s2b7nbtv : tensor<f32>
    %v8219 = stablehlo.multiply %v1720, %v1720 : tensor<f32>
    %v8220 = stablehlo.multiply %v8217, %v8219 : tensor<f32>
    %v8221 = stablehlo.add %v8218, %v8220 : tensor<f32>
    %v8222 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8223 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8224 = stablehlo.divide %v8215, %v8222 : tensor<f32>
    %v8225 = stablehlo.divide %v8221, %v8223 : tensor<f32>
    %v8226 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8227 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8228 = stablehlo.sqrt %v8225 : tensor<f32>
    %v8229 = stablehlo.add %v8228, %v8227 : tensor<f32>
    %v8230 = stablehlo.divide %v8224, %v8229 : tensor<f32>
    %v8231 = stablehlo.multiply %v8226, %v8230 : tensor<f32>
    %v8232 = stablehlo.subtract %s2b7nbt, %v8231 : tensor<f32>
    %v8233 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8234 = stablehlo.multiply %v8233, %v8226 : tensor<f32>
    %v8235 = stablehlo.multiply %v8234, %s2b7nbt : tensor<f32>
    %v8236 = stablehlo.subtract %v8232, %v8235 : tensor<f32>
    %v8237 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8238 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8239 = stablehlo.multiply %v8237, %s2b7eWm : tensor<1536x384x1x1xf32>
    %v8240 = stablehlo.multiply %v8238, %v1699 : tensor<1536x384x1x1xf32>
    %v8241 = stablehlo.add %v8239, %v8240 : tensor<1536x384x1x1xf32>
    %v8242 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8243 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8244 = stablehlo.multiply %v8242, %s2b7eWv : tensor<1536x384x1x1xf32>
    %v8245 = stablehlo.multiply %v1699, %v1699 : tensor<1536x384x1x1xf32>
    %v8246 = stablehlo.multiply %v8243, %v8245 : tensor<1536x384x1x1xf32>
    %v8247 = stablehlo.add %v8244, %v8246 : tensor<1536x384x1x1xf32>
    %v8248 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8249 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8250 = stablehlo.multiply %v8248, %s2b7eWm : tensor<1536x384x1x1xf32>
    %v8251 = stablehlo.multiply %v8249, %v1699 : tensor<1536x384x1x1xf32>
    %v8252 = stablehlo.add %v8250, %v8251 : tensor<1536x384x1x1xf32>
    %v8253 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8254 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8255 = stablehlo.multiply %v8253, %s2b7eWv : tensor<1536x384x1x1xf32>
    %v8256 = stablehlo.multiply %v1699, %v1699 : tensor<1536x384x1x1xf32>
    %v8257 = stablehlo.multiply %v8254, %v8256 : tensor<1536x384x1x1xf32>
    %v8258 = stablehlo.add %v8255, %v8257 : tensor<1536x384x1x1xf32>
    %v8259 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8260 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8261 = stablehlo.divide %v8252, %v8259 : tensor<1536x384x1x1xf32>
    %v8262 = stablehlo.divide %v8258, %v8260 : tensor<1536x384x1x1xf32>
    %v8263 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8264 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8265 = stablehlo.sqrt %v8262 : tensor<1536x384x1x1xf32>
    %v8266 = stablehlo.add %v8265, %v8264 : tensor<1536x384x1x1xf32>
    %v8267 = stablehlo.divide %v8261, %v8266 : tensor<1536x384x1x1xf32>
    %v8268 = stablehlo.multiply %v8263, %v8267 : tensor<1536x384x1x1xf32>
    %v8269 = stablehlo.subtract %s2b7eW, %v8268 : tensor<1536x384x1x1xf32>
    %v8270 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8271 = stablehlo.multiply %v8270, %v8263 : tensor<1536x384x1x1xf32>
    %v8272 = stablehlo.multiply %v8271, %s2b7eW : tensor<1536x384x1x1xf32>
    %v8273 = stablehlo.subtract %v8269, %v8272 : tensor<1536x384x1x1xf32>
    %v8274 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8275 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8276 = stablehlo.multiply %v8274, %s2b7ebm : tensor<1536xf32>
    %v8277 = stablehlo.multiply %v8275, %v1702 : tensor<1536xf32>
    %v8278 = stablehlo.add %v8276, %v8277 : tensor<1536xf32>
    %v8279 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8280 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8281 = stablehlo.multiply %v8279, %s2b7ebv : tensor<1536xf32>
    %v8282 = stablehlo.multiply %v1702, %v1702 : tensor<1536xf32>
    %v8283 = stablehlo.multiply %v8280, %v8282 : tensor<1536xf32>
    %v8284 = stablehlo.add %v8281, %v8283 : tensor<1536xf32>
    %v8285 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8286 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8287 = stablehlo.multiply %v8285, %s2b7ebm : tensor<1536xf32>
    %v8288 = stablehlo.multiply %v8286, %v1702 : tensor<1536xf32>
    %v8289 = stablehlo.add %v8287, %v8288 : tensor<1536xf32>
    %v8290 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8291 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8292 = stablehlo.multiply %v8290, %s2b7ebv : tensor<1536xf32>
    %v8293 = stablehlo.multiply %v1702, %v1702 : tensor<1536xf32>
    %v8294 = stablehlo.multiply %v8291, %v8293 : tensor<1536xf32>
    %v8295 = stablehlo.add %v8292, %v8294 : tensor<1536xf32>
    %v8296 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8297 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8298 = stablehlo.divide %v8289, %v8296 : tensor<1536xf32>
    %v8299 = stablehlo.divide %v8295, %v8297 : tensor<1536xf32>
    %v8300 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8301 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8302 = stablehlo.sqrt %v8299 : tensor<1536xf32>
    %v8303 = stablehlo.add %v8302, %v8301 : tensor<1536xf32>
    %v8304 = stablehlo.divide %v8298, %v8303 : tensor<1536xf32>
    %v8305 = stablehlo.multiply %v8300, %v8304 : tensor<1536xf32>
    %v8306 = stablehlo.subtract %s2b7eb, %v8305 : tensor<1536xf32>
    %v8307 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8308 = stablehlo.multiply %v8307, %v8300 : tensor<1536xf32>
    %v8309 = stablehlo.multiply %v8308, %s2b7eb : tensor<1536xf32>
    %v8310 = stablehlo.subtract %v8306, %v8309 : tensor<1536xf32>
    %v8311 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8312 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8313 = stablehlo.multiply %v8311, %s2b7pWm : tensor<384x1536x1x1xf32>
    %v8314 = stablehlo.multiply %v8312, %v1690 : tensor<384x1536x1x1xf32>
    %v8315 = stablehlo.add %v8313, %v8314 : tensor<384x1536x1x1xf32>
    %v8316 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8317 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8318 = stablehlo.multiply %v8316, %s2b7pWv : tensor<384x1536x1x1xf32>
    %v8319 = stablehlo.multiply %v1690, %v1690 : tensor<384x1536x1x1xf32>
    %v8320 = stablehlo.multiply %v8317, %v8319 : tensor<384x1536x1x1xf32>
    %v8321 = stablehlo.add %v8318, %v8320 : tensor<384x1536x1x1xf32>
    %v8322 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8323 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8324 = stablehlo.multiply %v8322, %s2b7pWm : tensor<384x1536x1x1xf32>
    %v8325 = stablehlo.multiply %v8323, %v1690 : tensor<384x1536x1x1xf32>
    %v8326 = stablehlo.add %v8324, %v8325 : tensor<384x1536x1x1xf32>
    %v8327 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8328 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8329 = stablehlo.multiply %v8327, %s2b7pWv : tensor<384x1536x1x1xf32>
    %v8330 = stablehlo.multiply %v1690, %v1690 : tensor<384x1536x1x1xf32>
    %v8331 = stablehlo.multiply %v8328, %v8330 : tensor<384x1536x1x1xf32>
    %v8332 = stablehlo.add %v8329, %v8331 : tensor<384x1536x1x1xf32>
    %v8333 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8334 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8335 = stablehlo.divide %v8326, %v8333 : tensor<384x1536x1x1xf32>
    %v8336 = stablehlo.divide %v8332, %v8334 : tensor<384x1536x1x1xf32>
    %v8337 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8338 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8339 = stablehlo.sqrt %v8336 : tensor<384x1536x1x1xf32>
    %v8340 = stablehlo.add %v8339, %v8338 : tensor<384x1536x1x1xf32>
    %v8341 = stablehlo.divide %v8335, %v8340 : tensor<384x1536x1x1xf32>
    %v8342 = stablehlo.multiply %v8337, %v8341 : tensor<384x1536x1x1xf32>
    %v8343 = stablehlo.subtract %s2b7pW, %v8342 : tensor<384x1536x1x1xf32>
    %v8344 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8345 = stablehlo.multiply %v8344, %v8337 : tensor<384x1536x1x1xf32>
    %v8346 = stablehlo.multiply %v8345, %s2b7pW : tensor<384x1536x1x1xf32>
    %v8347 = stablehlo.subtract %v8343, %v8346 : tensor<384x1536x1x1xf32>
    %v8348 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8349 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8350 = stablehlo.multiply %v8348, %s2b7pbm : tensor<384xf32>
    %v8351 = stablehlo.multiply %v8349, %v1693 : tensor<384xf32>
    %v8352 = stablehlo.add %v8350, %v8351 : tensor<384xf32>
    %v8353 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8354 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8355 = stablehlo.multiply %v8353, %s2b7pbv : tensor<384xf32>
    %v8356 = stablehlo.multiply %v1693, %v1693 : tensor<384xf32>
    %v8357 = stablehlo.multiply %v8354, %v8356 : tensor<384xf32>
    %v8358 = stablehlo.add %v8355, %v8357 : tensor<384xf32>
    %v8359 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8360 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8361 = stablehlo.multiply %v8359, %s2b7pbm : tensor<384xf32>
    %v8362 = stablehlo.multiply %v8360, %v1693 : tensor<384xf32>
    %v8363 = stablehlo.add %v8361, %v8362 : tensor<384xf32>
    %v8364 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8365 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8366 = stablehlo.multiply %v8364, %s2b7pbv : tensor<384xf32>
    %v8367 = stablehlo.multiply %v1693, %v1693 : tensor<384xf32>
    %v8368 = stablehlo.multiply %v8365, %v8367 : tensor<384xf32>
    %v8369 = stablehlo.add %v8366, %v8368 : tensor<384xf32>
    %v8370 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8371 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8372 = stablehlo.divide %v8363, %v8370 : tensor<384xf32>
    %v8373 = stablehlo.divide %v8369, %v8371 : tensor<384xf32>
    %v8374 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8375 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8376 = stablehlo.sqrt %v8373 : tensor<384xf32>
    %v8377 = stablehlo.add %v8376, %v8375 : tensor<384xf32>
    %v8378 = stablehlo.divide %v8372, %v8377 : tensor<384xf32>
    %v8379 = stablehlo.multiply %v8374, %v8378 : tensor<384xf32>
    %v8380 = stablehlo.subtract %s2b7pb, %v8379 : tensor<384xf32>
    %v8381 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8382 = stablehlo.multiply %v8381, %v8374 : tensor<384xf32>
    %v8383 = stablehlo.multiply %v8382, %s2b7pb : tensor<384xf32>
    %v8384 = stablehlo.subtract %v8380, %v8383 : tensor<384xf32>
    %v8385 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8386 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8387 = stablehlo.multiply %v8385, %s2b7lgm : tensor<384xf32>
    %v8388 = stablehlo.multiply %v8386, %v1684 : tensor<384xf32>
    %v8389 = stablehlo.add %v8387, %v8388 : tensor<384xf32>
    %v8390 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8391 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8392 = stablehlo.multiply %v8390, %s2b7lgv : tensor<384xf32>
    %v8393 = stablehlo.multiply %v1684, %v1684 : tensor<384xf32>
    %v8394 = stablehlo.multiply %v8391, %v8393 : tensor<384xf32>
    %v8395 = stablehlo.add %v8392, %v8394 : tensor<384xf32>
    %v8396 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8397 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8398 = stablehlo.multiply %v8396, %s2b7lgm : tensor<384xf32>
    %v8399 = stablehlo.multiply %v8397, %v1684 : tensor<384xf32>
    %v8400 = stablehlo.add %v8398, %v8399 : tensor<384xf32>
    %v8401 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8402 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8403 = stablehlo.multiply %v8401, %s2b7lgv : tensor<384xf32>
    %v8404 = stablehlo.multiply %v1684, %v1684 : tensor<384xf32>
    %v8405 = stablehlo.multiply %v8402, %v8404 : tensor<384xf32>
    %v8406 = stablehlo.add %v8403, %v8405 : tensor<384xf32>
    %v8407 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8408 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8409 = stablehlo.divide %v8400, %v8407 : tensor<384xf32>
    %v8410 = stablehlo.divide %v8406, %v8408 : tensor<384xf32>
    %v8411 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8412 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8413 = stablehlo.sqrt %v8410 : tensor<384xf32>
    %v8414 = stablehlo.add %v8413, %v8412 : tensor<384xf32>
    %v8415 = stablehlo.divide %v8409, %v8414 : tensor<384xf32>
    %v8416 = stablehlo.multiply %v8411, %v8415 : tensor<384xf32>
    %v8417 = stablehlo.subtract %s2b7lg, %v8416 : tensor<384xf32>
    %v8418 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8419 = stablehlo.multiply %v8418, %v8411 : tensor<384xf32>
    %v8420 = stablehlo.multiply %v8419, %s2b7lg : tensor<384xf32>
    %v8421 = stablehlo.subtract %v8417, %v8420 : tensor<384xf32>
    %v8422 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8423 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8424 = stablehlo.multiply %v8422, %s2b8dWm : tensor<384x1x7x7xf32>
    %v8425 = stablehlo.multiply %v8423, %v1607 : tensor<384x1x7x7xf32>
    %v8426 = stablehlo.add %v8424, %v8425 : tensor<384x1x7x7xf32>
    %v8427 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8428 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8429 = stablehlo.multiply %v8427, %s2b8dWv : tensor<384x1x7x7xf32>
    %v8430 = stablehlo.multiply %v1607, %v1607 : tensor<384x1x7x7xf32>
    %v8431 = stablehlo.multiply %v8428, %v8430 : tensor<384x1x7x7xf32>
    %v8432 = stablehlo.add %v8429, %v8431 : tensor<384x1x7x7xf32>
    %v8433 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8434 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8435 = stablehlo.multiply %v8433, %s2b8dWm : tensor<384x1x7x7xf32>
    %v8436 = stablehlo.multiply %v8434, %v1607 : tensor<384x1x7x7xf32>
    %v8437 = stablehlo.add %v8435, %v8436 : tensor<384x1x7x7xf32>
    %v8438 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8439 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8440 = stablehlo.multiply %v8438, %s2b8dWv : tensor<384x1x7x7xf32>
    %v8441 = stablehlo.multiply %v1607, %v1607 : tensor<384x1x7x7xf32>
    %v8442 = stablehlo.multiply %v8439, %v8441 : tensor<384x1x7x7xf32>
    %v8443 = stablehlo.add %v8440, %v8442 : tensor<384x1x7x7xf32>
    %v8444 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8445 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8446 = stablehlo.divide %v8437, %v8444 : tensor<384x1x7x7xf32>
    %v8447 = stablehlo.divide %v8443, %v8445 : tensor<384x1x7x7xf32>
    %v8448 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8449 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8450 = stablehlo.sqrt %v8447 : tensor<384x1x7x7xf32>
    %v8451 = stablehlo.add %v8450, %v8449 : tensor<384x1x7x7xf32>
    %v8452 = stablehlo.divide %v8446, %v8451 : tensor<384x1x7x7xf32>
    %v8453 = stablehlo.multiply %v8448, %v8452 : tensor<384x1x7x7xf32>
    %v8454 = stablehlo.subtract %s2b8dW, %v8453 : tensor<384x1x7x7xf32>
    %v8455 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8456 = stablehlo.multiply %v8455, %v8448 : tensor<384x1x7x7xf32>
    %v8457 = stablehlo.multiply %v8456, %s2b8dW : tensor<384x1x7x7xf32>
    %v8458 = stablehlo.subtract %v8454, %v8457 : tensor<384x1x7x7xf32>
    %v8459 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8460 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8461 = stablehlo.multiply %v8459, %s2b8dbm : tensor<384xf32>
    %v8462 = stablehlo.multiply %v8460, %v1610 : tensor<384xf32>
    %v8463 = stablehlo.add %v8461, %v8462 : tensor<384xf32>
    %v8464 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8465 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8466 = stablehlo.multiply %v8464, %s2b8dbv : tensor<384xf32>
    %v8467 = stablehlo.multiply %v1610, %v1610 : tensor<384xf32>
    %v8468 = stablehlo.multiply %v8465, %v8467 : tensor<384xf32>
    %v8469 = stablehlo.add %v8466, %v8468 : tensor<384xf32>
    %v8470 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8471 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8472 = stablehlo.multiply %v8470, %s2b8dbm : tensor<384xf32>
    %v8473 = stablehlo.multiply %v8471, %v1610 : tensor<384xf32>
    %v8474 = stablehlo.add %v8472, %v8473 : tensor<384xf32>
    %v8475 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8476 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8477 = stablehlo.multiply %v8475, %s2b8dbv : tensor<384xf32>
    %v8478 = stablehlo.multiply %v1610, %v1610 : tensor<384xf32>
    %v8479 = stablehlo.multiply %v8476, %v8478 : tensor<384xf32>
    %v8480 = stablehlo.add %v8477, %v8479 : tensor<384xf32>
    %v8481 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8482 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8483 = stablehlo.divide %v8474, %v8481 : tensor<384xf32>
    %v8484 = stablehlo.divide %v8480, %v8482 : tensor<384xf32>
    %v8485 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8486 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8487 = stablehlo.sqrt %v8484 : tensor<384xf32>
    %v8488 = stablehlo.add %v8487, %v8486 : tensor<384xf32>
    %v8489 = stablehlo.divide %v8483, %v8488 : tensor<384xf32>
    %v8490 = stablehlo.multiply %v8485, %v8489 : tensor<384xf32>
    %v8491 = stablehlo.subtract %s2b8db, %v8490 : tensor<384xf32>
    %v8492 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8493 = stablehlo.multiply %v8492, %v8485 : tensor<384xf32>
    %v8494 = stablehlo.multiply %v8493, %s2b8db : tensor<384xf32>
    %v8495 = stablehlo.subtract %v8491, %v8494 : tensor<384xf32>
    %v8496 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8497 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8498 = stablehlo.multiply %v8496, %s2b8ngm : tensor<f32>
    %v8499 = stablehlo.multiply %v8497, %v1599 : tensor<f32>
    %v8500 = stablehlo.add %v8498, %v8499 : tensor<f32>
    %v8501 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8502 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8503 = stablehlo.multiply %v8501, %s2b8ngv : tensor<f32>
    %v8504 = stablehlo.multiply %v1599, %v1599 : tensor<f32>
    %v8505 = stablehlo.multiply %v8502, %v8504 : tensor<f32>
    %v8506 = stablehlo.add %v8503, %v8505 : tensor<f32>
    %v8507 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8508 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8509 = stablehlo.multiply %v8507, %s2b8ngm : tensor<f32>
    %v8510 = stablehlo.multiply %v8508, %v1599 : tensor<f32>
    %v8511 = stablehlo.add %v8509, %v8510 : tensor<f32>
    %v8512 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8513 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8514 = stablehlo.multiply %v8512, %s2b8ngv : tensor<f32>
    %v8515 = stablehlo.multiply %v1599, %v1599 : tensor<f32>
    %v8516 = stablehlo.multiply %v8513, %v8515 : tensor<f32>
    %v8517 = stablehlo.add %v8514, %v8516 : tensor<f32>
    %v8518 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8519 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8520 = stablehlo.divide %v8511, %v8518 : tensor<f32>
    %v8521 = stablehlo.divide %v8517, %v8519 : tensor<f32>
    %v8522 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8523 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8524 = stablehlo.sqrt %v8521 : tensor<f32>
    %v8525 = stablehlo.add %v8524, %v8523 : tensor<f32>
    %v8526 = stablehlo.divide %v8520, %v8525 : tensor<f32>
    %v8527 = stablehlo.multiply %v8522, %v8526 : tensor<f32>
    %v8528 = stablehlo.subtract %s2b8ng, %v8527 : tensor<f32>
    %v8529 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8530 = stablehlo.multiply %v8529, %v8522 : tensor<f32>
    %v8531 = stablehlo.multiply %v8530, %s2b8ng : tensor<f32>
    %v8532 = stablehlo.subtract %v8528, %v8531 : tensor<f32>
    %v8533 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8534 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8535 = stablehlo.multiply %v8533, %s2b8nbtm : tensor<f32>
    %v8536 = stablehlo.multiply %v8534, %v1601 : tensor<f32>
    %v8537 = stablehlo.add %v8535, %v8536 : tensor<f32>
    %v8538 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8539 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8540 = stablehlo.multiply %v8538, %s2b8nbtv : tensor<f32>
    %v8541 = stablehlo.multiply %v1601, %v1601 : tensor<f32>
    %v8542 = stablehlo.multiply %v8539, %v8541 : tensor<f32>
    %v8543 = stablehlo.add %v8540, %v8542 : tensor<f32>
    %v8544 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8545 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8546 = stablehlo.multiply %v8544, %s2b8nbtm : tensor<f32>
    %v8547 = stablehlo.multiply %v8545, %v1601 : tensor<f32>
    %v8548 = stablehlo.add %v8546, %v8547 : tensor<f32>
    %v8549 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8550 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8551 = stablehlo.multiply %v8549, %s2b8nbtv : tensor<f32>
    %v8552 = stablehlo.multiply %v1601, %v1601 : tensor<f32>
    %v8553 = stablehlo.multiply %v8550, %v8552 : tensor<f32>
    %v8554 = stablehlo.add %v8551, %v8553 : tensor<f32>
    %v8555 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8556 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8557 = stablehlo.divide %v8548, %v8555 : tensor<f32>
    %v8558 = stablehlo.divide %v8554, %v8556 : tensor<f32>
    %v8559 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8560 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8561 = stablehlo.sqrt %v8558 : tensor<f32>
    %v8562 = stablehlo.add %v8561, %v8560 : tensor<f32>
    %v8563 = stablehlo.divide %v8557, %v8562 : tensor<f32>
    %v8564 = stablehlo.multiply %v8559, %v8563 : tensor<f32>
    %v8565 = stablehlo.subtract %s2b8nbt, %v8564 : tensor<f32>
    %v8566 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8567 = stablehlo.multiply %v8566, %v8559 : tensor<f32>
    %v8568 = stablehlo.multiply %v8567, %s2b8nbt : tensor<f32>
    %v8569 = stablehlo.subtract %v8565, %v8568 : tensor<f32>
    %v8570 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8571 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8572 = stablehlo.multiply %v8570, %s2b8eWm : tensor<1536x384x1x1xf32>
    %v8573 = stablehlo.multiply %v8571, %v1580 : tensor<1536x384x1x1xf32>
    %v8574 = stablehlo.add %v8572, %v8573 : tensor<1536x384x1x1xf32>
    %v8575 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8576 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8577 = stablehlo.multiply %v8575, %s2b8eWv : tensor<1536x384x1x1xf32>
    %v8578 = stablehlo.multiply %v1580, %v1580 : tensor<1536x384x1x1xf32>
    %v8579 = stablehlo.multiply %v8576, %v8578 : tensor<1536x384x1x1xf32>
    %v8580 = stablehlo.add %v8577, %v8579 : tensor<1536x384x1x1xf32>
    %v8581 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8582 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8583 = stablehlo.multiply %v8581, %s2b8eWm : tensor<1536x384x1x1xf32>
    %v8584 = stablehlo.multiply %v8582, %v1580 : tensor<1536x384x1x1xf32>
    %v8585 = stablehlo.add %v8583, %v8584 : tensor<1536x384x1x1xf32>
    %v8586 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8587 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8588 = stablehlo.multiply %v8586, %s2b8eWv : tensor<1536x384x1x1xf32>
    %v8589 = stablehlo.multiply %v1580, %v1580 : tensor<1536x384x1x1xf32>
    %v8590 = stablehlo.multiply %v8587, %v8589 : tensor<1536x384x1x1xf32>
    %v8591 = stablehlo.add %v8588, %v8590 : tensor<1536x384x1x1xf32>
    %v8592 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8593 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8594 = stablehlo.divide %v8585, %v8592 : tensor<1536x384x1x1xf32>
    %v8595 = stablehlo.divide %v8591, %v8593 : tensor<1536x384x1x1xf32>
    %v8596 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8597 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8598 = stablehlo.sqrt %v8595 : tensor<1536x384x1x1xf32>
    %v8599 = stablehlo.add %v8598, %v8597 : tensor<1536x384x1x1xf32>
    %v8600 = stablehlo.divide %v8594, %v8599 : tensor<1536x384x1x1xf32>
    %v8601 = stablehlo.multiply %v8596, %v8600 : tensor<1536x384x1x1xf32>
    %v8602 = stablehlo.subtract %s2b8eW, %v8601 : tensor<1536x384x1x1xf32>
    %v8603 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8604 = stablehlo.multiply %v8603, %v8596 : tensor<1536x384x1x1xf32>
    %v8605 = stablehlo.multiply %v8604, %s2b8eW : tensor<1536x384x1x1xf32>
    %v8606 = stablehlo.subtract %v8602, %v8605 : tensor<1536x384x1x1xf32>
    %v8607 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8608 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8609 = stablehlo.multiply %v8607, %s2b8ebm : tensor<1536xf32>
    %v8610 = stablehlo.multiply %v8608, %v1583 : tensor<1536xf32>
    %v8611 = stablehlo.add %v8609, %v8610 : tensor<1536xf32>
    %v8612 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8613 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8614 = stablehlo.multiply %v8612, %s2b8ebv : tensor<1536xf32>
    %v8615 = stablehlo.multiply %v1583, %v1583 : tensor<1536xf32>
    %v8616 = stablehlo.multiply %v8613, %v8615 : tensor<1536xf32>
    %v8617 = stablehlo.add %v8614, %v8616 : tensor<1536xf32>
    %v8618 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8619 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8620 = stablehlo.multiply %v8618, %s2b8ebm : tensor<1536xf32>
    %v8621 = stablehlo.multiply %v8619, %v1583 : tensor<1536xf32>
    %v8622 = stablehlo.add %v8620, %v8621 : tensor<1536xf32>
    %v8623 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8624 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8625 = stablehlo.multiply %v8623, %s2b8ebv : tensor<1536xf32>
    %v8626 = stablehlo.multiply %v1583, %v1583 : tensor<1536xf32>
    %v8627 = stablehlo.multiply %v8624, %v8626 : tensor<1536xf32>
    %v8628 = stablehlo.add %v8625, %v8627 : tensor<1536xf32>
    %v8629 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8630 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8631 = stablehlo.divide %v8622, %v8629 : tensor<1536xf32>
    %v8632 = stablehlo.divide %v8628, %v8630 : tensor<1536xf32>
    %v8633 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8634 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8635 = stablehlo.sqrt %v8632 : tensor<1536xf32>
    %v8636 = stablehlo.add %v8635, %v8634 : tensor<1536xf32>
    %v8637 = stablehlo.divide %v8631, %v8636 : tensor<1536xf32>
    %v8638 = stablehlo.multiply %v8633, %v8637 : tensor<1536xf32>
    %v8639 = stablehlo.subtract %s2b8eb, %v8638 : tensor<1536xf32>
    %v8640 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8641 = stablehlo.multiply %v8640, %v8633 : tensor<1536xf32>
    %v8642 = stablehlo.multiply %v8641, %s2b8eb : tensor<1536xf32>
    %v8643 = stablehlo.subtract %v8639, %v8642 : tensor<1536xf32>
    %v8644 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8645 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8646 = stablehlo.multiply %v8644, %s2b8pWm : tensor<384x1536x1x1xf32>
    %v8647 = stablehlo.multiply %v8645, %v1571 : tensor<384x1536x1x1xf32>
    %v8648 = stablehlo.add %v8646, %v8647 : tensor<384x1536x1x1xf32>
    %v8649 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8650 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8651 = stablehlo.multiply %v8649, %s2b8pWv : tensor<384x1536x1x1xf32>
    %v8652 = stablehlo.multiply %v1571, %v1571 : tensor<384x1536x1x1xf32>
    %v8653 = stablehlo.multiply %v8650, %v8652 : tensor<384x1536x1x1xf32>
    %v8654 = stablehlo.add %v8651, %v8653 : tensor<384x1536x1x1xf32>
    %v8655 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8656 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8657 = stablehlo.multiply %v8655, %s2b8pWm : tensor<384x1536x1x1xf32>
    %v8658 = stablehlo.multiply %v8656, %v1571 : tensor<384x1536x1x1xf32>
    %v8659 = stablehlo.add %v8657, %v8658 : tensor<384x1536x1x1xf32>
    %v8660 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8661 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8662 = stablehlo.multiply %v8660, %s2b8pWv : tensor<384x1536x1x1xf32>
    %v8663 = stablehlo.multiply %v1571, %v1571 : tensor<384x1536x1x1xf32>
    %v8664 = stablehlo.multiply %v8661, %v8663 : tensor<384x1536x1x1xf32>
    %v8665 = stablehlo.add %v8662, %v8664 : tensor<384x1536x1x1xf32>
    %v8666 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8667 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8668 = stablehlo.divide %v8659, %v8666 : tensor<384x1536x1x1xf32>
    %v8669 = stablehlo.divide %v8665, %v8667 : tensor<384x1536x1x1xf32>
    %v8670 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8671 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8672 = stablehlo.sqrt %v8669 : tensor<384x1536x1x1xf32>
    %v8673 = stablehlo.add %v8672, %v8671 : tensor<384x1536x1x1xf32>
    %v8674 = stablehlo.divide %v8668, %v8673 : tensor<384x1536x1x1xf32>
    %v8675 = stablehlo.multiply %v8670, %v8674 : tensor<384x1536x1x1xf32>
    %v8676 = stablehlo.subtract %s2b8pW, %v8675 : tensor<384x1536x1x1xf32>
    %v8677 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8678 = stablehlo.multiply %v8677, %v8670 : tensor<384x1536x1x1xf32>
    %v8679 = stablehlo.multiply %v8678, %s2b8pW : tensor<384x1536x1x1xf32>
    %v8680 = stablehlo.subtract %v8676, %v8679 : tensor<384x1536x1x1xf32>
    %v8681 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8682 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8683 = stablehlo.multiply %v8681, %s2b8pbm : tensor<384xf32>
    %v8684 = stablehlo.multiply %v8682, %v1574 : tensor<384xf32>
    %v8685 = stablehlo.add %v8683, %v8684 : tensor<384xf32>
    %v8686 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8687 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8688 = stablehlo.multiply %v8686, %s2b8pbv : tensor<384xf32>
    %v8689 = stablehlo.multiply %v1574, %v1574 : tensor<384xf32>
    %v8690 = stablehlo.multiply %v8687, %v8689 : tensor<384xf32>
    %v8691 = stablehlo.add %v8688, %v8690 : tensor<384xf32>
    %v8692 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8693 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8694 = stablehlo.multiply %v8692, %s2b8pbm : tensor<384xf32>
    %v8695 = stablehlo.multiply %v8693, %v1574 : tensor<384xf32>
    %v8696 = stablehlo.add %v8694, %v8695 : tensor<384xf32>
    %v8697 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8698 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8699 = stablehlo.multiply %v8697, %s2b8pbv : tensor<384xf32>
    %v8700 = stablehlo.multiply %v1574, %v1574 : tensor<384xf32>
    %v8701 = stablehlo.multiply %v8698, %v8700 : tensor<384xf32>
    %v8702 = stablehlo.add %v8699, %v8701 : tensor<384xf32>
    %v8703 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8704 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8705 = stablehlo.divide %v8696, %v8703 : tensor<384xf32>
    %v8706 = stablehlo.divide %v8702, %v8704 : tensor<384xf32>
    %v8707 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8708 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8709 = stablehlo.sqrt %v8706 : tensor<384xf32>
    %v8710 = stablehlo.add %v8709, %v8708 : tensor<384xf32>
    %v8711 = stablehlo.divide %v8705, %v8710 : tensor<384xf32>
    %v8712 = stablehlo.multiply %v8707, %v8711 : tensor<384xf32>
    %v8713 = stablehlo.subtract %s2b8pb, %v8712 : tensor<384xf32>
    %v8714 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8715 = stablehlo.multiply %v8714, %v8707 : tensor<384xf32>
    %v8716 = stablehlo.multiply %v8715, %s2b8pb : tensor<384xf32>
    %v8717 = stablehlo.subtract %v8713, %v8716 : tensor<384xf32>
    %v8718 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8719 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8720 = stablehlo.multiply %v8718, %s2b8lgm : tensor<384xf32>
    %v8721 = stablehlo.multiply %v8719, %v1565 : tensor<384xf32>
    %v8722 = stablehlo.add %v8720, %v8721 : tensor<384xf32>
    %v8723 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8724 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8725 = stablehlo.multiply %v8723, %s2b8lgv : tensor<384xf32>
    %v8726 = stablehlo.multiply %v1565, %v1565 : tensor<384xf32>
    %v8727 = stablehlo.multiply %v8724, %v8726 : tensor<384xf32>
    %v8728 = stablehlo.add %v8725, %v8727 : tensor<384xf32>
    %v8729 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8730 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8731 = stablehlo.multiply %v8729, %s2b8lgm : tensor<384xf32>
    %v8732 = stablehlo.multiply %v8730, %v1565 : tensor<384xf32>
    %v8733 = stablehlo.add %v8731, %v8732 : tensor<384xf32>
    %v8734 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8735 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8736 = stablehlo.multiply %v8734, %s2b8lgv : tensor<384xf32>
    %v8737 = stablehlo.multiply %v1565, %v1565 : tensor<384xf32>
    %v8738 = stablehlo.multiply %v8735, %v8737 : tensor<384xf32>
    %v8739 = stablehlo.add %v8736, %v8738 : tensor<384xf32>
    %v8740 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8741 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8742 = stablehlo.divide %v8733, %v8740 : tensor<384xf32>
    %v8743 = stablehlo.divide %v8739, %v8741 : tensor<384xf32>
    %v8744 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8745 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8746 = stablehlo.sqrt %v8743 : tensor<384xf32>
    %v8747 = stablehlo.add %v8746, %v8745 : tensor<384xf32>
    %v8748 = stablehlo.divide %v8742, %v8747 : tensor<384xf32>
    %v8749 = stablehlo.multiply %v8744, %v8748 : tensor<384xf32>
    %v8750 = stablehlo.subtract %s2b8lg, %v8749 : tensor<384xf32>
    %v8751 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8752 = stablehlo.multiply %v8751, %v8744 : tensor<384xf32>
    %v8753 = stablehlo.multiply %v8752, %s2b8lg : tensor<384xf32>
    %v8754 = stablehlo.subtract %v8750, %v8753 : tensor<384xf32>
    %v8755 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8756 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8757 = stablehlo.multiply %v8755, %d2ngm : tensor<f32>
    %v8758 = stablehlo.multiply %v8756, %v1489 : tensor<f32>
    %v8759 = stablehlo.add %v8757, %v8758 : tensor<f32>
    %v8760 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8761 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8762 = stablehlo.multiply %v8760, %d2ngv : tensor<f32>
    %v8763 = stablehlo.multiply %v1489, %v1489 : tensor<f32>
    %v8764 = stablehlo.multiply %v8761, %v8763 : tensor<f32>
    %v8765 = stablehlo.add %v8762, %v8764 : tensor<f32>
    %v8766 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8767 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8768 = stablehlo.multiply %v8766, %d2ngm : tensor<f32>
    %v8769 = stablehlo.multiply %v8767, %v1489 : tensor<f32>
    %v8770 = stablehlo.add %v8768, %v8769 : tensor<f32>
    %v8771 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8772 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8773 = stablehlo.multiply %v8771, %d2ngv : tensor<f32>
    %v8774 = stablehlo.multiply %v1489, %v1489 : tensor<f32>
    %v8775 = stablehlo.multiply %v8772, %v8774 : tensor<f32>
    %v8776 = stablehlo.add %v8773, %v8775 : tensor<f32>
    %v8777 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8778 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8779 = stablehlo.divide %v8770, %v8777 : tensor<f32>
    %v8780 = stablehlo.divide %v8776, %v8778 : tensor<f32>
    %v8781 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8782 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8783 = stablehlo.sqrt %v8780 : tensor<f32>
    %v8784 = stablehlo.add %v8783, %v8782 : tensor<f32>
    %v8785 = stablehlo.divide %v8779, %v8784 : tensor<f32>
    %v8786 = stablehlo.multiply %v8781, %v8785 : tensor<f32>
    %v8787 = stablehlo.subtract %d2ng, %v8786 : tensor<f32>
    %v8788 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8789 = stablehlo.multiply %v8788, %v8781 : tensor<f32>
    %v8790 = stablehlo.multiply %v8789, %d2ng : tensor<f32>
    %v8791 = stablehlo.subtract %v8787, %v8790 : tensor<f32>
    %v8792 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8793 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8794 = stablehlo.multiply %v8792, %d2nbtm : tensor<f32>
    %v8795 = stablehlo.multiply %v8793, %v1491 : tensor<f32>
    %v8796 = stablehlo.add %v8794, %v8795 : tensor<f32>
    %v8797 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8798 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8799 = stablehlo.multiply %v8797, %d2nbtv : tensor<f32>
    %v8800 = stablehlo.multiply %v1491, %v1491 : tensor<f32>
    %v8801 = stablehlo.multiply %v8798, %v8800 : tensor<f32>
    %v8802 = stablehlo.add %v8799, %v8801 : tensor<f32>
    %v8803 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8804 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8805 = stablehlo.multiply %v8803, %d2nbtm : tensor<f32>
    %v8806 = stablehlo.multiply %v8804, %v1491 : tensor<f32>
    %v8807 = stablehlo.add %v8805, %v8806 : tensor<f32>
    %v8808 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8809 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8810 = stablehlo.multiply %v8808, %d2nbtv : tensor<f32>
    %v8811 = stablehlo.multiply %v1491, %v1491 : tensor<f32>
    %v8812 = stablehlo.multiply %v8809, %v8811 : tensor<f32>
    %v8813 = stablehlo.add %v8810, %v8812 : tensor<f32>
    %v8814 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8815 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8816 = stablehlo.divide %v8807, %v8814 : tensor<f32>
    %v8817 = stablehlo.divide %v8813, %v8815 : tensor<f32>
    %v8818 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8819 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8820 = stablehlo.sqrt %v8817 : tensor<f32>
    %v8821 = stablehlo.add %v8820, %v8819 : tensor<f32>
    %v8822 = stablehlo.divide %v8816, %v8821 : tensor<f32>
    %v8823 = stablehlo.multiply %v8818, %v8822 : tensor<f32>
    %v8824 = stablehlo.subtract %d2nbt, %v8823 : tensor<f32>
    %v8825 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8826 = stablehlo.multiply %v8825, %v8818 : tensor<f32>
    %v8827 = stablehlo.multiply %v8826, %d2nbt : tensor<f32>
    %v8828 = stablehlo.subtract %v8824, %v8827 : tensor<f32>
    %v8829 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8830 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8831 = stablehlo.multiply %v8829, %d2Wm : tensor<768x384x2x2xf32>
    %v8832 = stablehlo.multiply %v8830, %dd2W : tensor<768x384x2x2xf32>
    %v8833 = stablehlo.add %v8831, %v8832 : tensor<768x384x2x2xf32>
    %v8834 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8835 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8836 = stablehlo.multiply %v8834, %d2Wv : tensor<768x384x2x2xf32>
    %v8837 = stablehlo.multiply %dd2W, %dd2W : tensor<768x384x2x2xf32>
    %v8838 = stablehlo.multiply %v8835, %v8837 : tensor<768x384x2x2xf32>
    %v8839 = stablehlo.add %v8836, %v8838 : tensor<768x384x2x2xf32>
    %v8840 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8841 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8842 = stablehlo.multiply %v8840, %d2Wm : tensor<768x384x2x2xf32>
    %v8843 = stablehlo.multiply %v8841, %dd2W : tensor<768x384x2x2xf32>
    %v8844 = stablehlo.add %v8842, %v8843 : tensor<768x384x2x2xf32>
    %v8845 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8846 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8847 = stablehlo.multiply %v8845, %d2Wv : tensor<768x384x2x2xf32>
    %v8848 = stablehlo.multiply %dd2W, %dd2W : tensor<768x384x2x2xf32>
    %v8849 = stablehlo.multiply %v8846, %v8848 : tensor<768x384x2x2xf32>
    %v8850 = stablehlo.add %v8847, %v8849 : tensor<768x384x2x2xf32>
    %v8851 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8852 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8853 = stablehlo.divide %v8844, %v8851 : tensor<768x384x2x2xf32>
    %v8854 = stablehlo.divide %v8850, %v8852 : tensor<768x384x2x2xf32>
    %v8855 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8856 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8857 = stablehlo.sqrt %v8854 : tensor<768x384x2x2xf32>
    %v8858 = stablehlo.add %v8857, %v8856 : tensor<768x384x2x2xf32>
    %v8859 = stablehlo.divide %v8853, %v8858 : tensor<768x384x2x2xf32>
    %v8860 = stablehlo.multiply %v8855, %v8859 : tensor<768x384x2x2xf32>
    %v8861 = stablehlo.subtract %d2W, %v8860 : tensor<768x384x2x2xf32>
    %v8862 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v8863 = stablehlo.multiply %v8862, %v8855 : tensor<768x384x2x2xf32>
    %v8864 = stablehlo.multiply %v8863, %d2W : tensor<768x384x2x2xf32>
    %v8865 = stablehlo.subtract %v8861, %v8864 : tensor<768x384x2x2xf32>
    %v8866 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8867 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8868 = stablehlo.multiply %v8866, %d2bm : tensor<768xf32>
    %v8869 = stablehlo.multiply %v8867, %v1473 : tensor<768xf32>
    %v8870 = stablehlo.add %v8868, %v8869 : tensor<768xf32>
    %v8871 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8872 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8873 = stablehlo.multiply %v8871, %d2bv : tensor<768xf32>
    %v8874 = stablehlo.multiply %v1473, %v1473 : tensor<768xf32>
    %v8875 = stablehlo.multiply %v8872, %v8874 : tensor<768xf32>
    %v8876 = stablehlo.add %v8873, %v8875 : tensor<768xf32>
    %v8877 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8878 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8879 = stablehlo.multiply %v8877, %d2bm : tensor<768xf32>
    %v8880 = stablehlo.multiply %v8878, %v1473 : tensor<768xf32>
    %v8881 = stablehlo.add %v8879, %v8880 : tensor<768xf32>
    %v8882 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8883 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8884 = stablehlo.multiply %v8882, %d2bv : tensor<768xf32>
    %v8885 = stablehlo.multiply %v1473, %v1473 : tensor<768xf32>
    %v8886 = stablehlo.multiply %v8883, %v8885 : tensor<768xf32>
    %v8887 = stablehlo.add %v8884, %v8886 : tensor<768xf32>
    %v8888 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8889 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8890 = stablehlo.divide %v8881, %v8888 : tensor<768xf32>
    %v8891 = stablehlo.divide %v8887, %v8889 : tensor<768xf32>
    %v8892 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8893 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8894 = stablehlo.sqrt %v8891 : tensor<768xf32>
    %v8895 = stablehlo.add %v8894, %v8893 : tensor<768xf32>
    %v8896 = stablehlo.divide %v8890, %v8895 : tensor<768xf32>
    %v8897 = stablehlo.multiply %v8892, %v8896 : tensor<768xf32>
    %v8898 = stablehlo.subtract %d2b, %v8897 : tensor<768xf32>
    %v8899 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8900 = stablehlo.multiply %v8899, %v8892 : tensor<768xf32>
    %v8901 = stablehlo.multiply %v8900, %d2b : tensor<768xf32>
    %v8902 = stablehlo.subtract %v8898, %v8901 : tensor<768xf32>
    %v8903 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8904 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8905 = stablehlo.multiply %v8903, %s3b0dWm : tensor<768x1x7x7xf32>
    %v8906 = stablehlo.multiply %v8904, %v1433 : tensor<768x1x7x7xf32>
    %v8907 = stablehlo.add %v8905, %v8906 : tensor<768x1x7x7xf32>
    %v8908 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8909 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8910 = stablehlo.multiply %v8908, %s3b0dWv : tensor<768x1x7x7xf32>
    %v8911 = stablehlo.multiply %v1433, %v1433 : tensor<768x1x7x7xf32>
    %v8912 = stablehlo.multiply %v8909, %v8911 : tensor<768x1x7x7xf32>
    %v8913 = stablehlo.add %v8910, %v8912 : tensor<768x1x7x7xf32>
    %v8914 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8915 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8916 = stablehlo.multiply %v8914, %s3b0dWm : tensor<768x1x7x7xf32>
    %v8917 = stablehlo.multiply %v8915, %v1433 : tensor<768x1x7x7xf32>
    %v8918 = stablehlo.add %v8916, %v8917 : tensor<768x1x7x7xf32>
    %v8919 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8920 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8921 = stablehlo.multiply %v8919, %s3b0dWv : tensor<768x1x7x7xf32>
    %v8922 = stablehlo.multiply %v1433, %v1433 : tensor<768x1x7x7xf32>
    %v8923 = stablehlo.multiply %v8920, %v8922 : tensor<768x1x7x7xf32>
    %v8924 = stablehlo.add %v8921, %v8923 : tensor<768x1x7x7xf32>
    %v8925 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8926 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8927 = stablehlo.divide %v8918, %v8925 : tensor<768x1x7x7xf32>
    %v8928 = stablehlo.divide %v8924, %v8926 : tensor<768x1x7x7xf32>
    %v8929 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8930 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8931 = stablehlo.sqrt %v8928 : tensor<768x1x7x7xf32>
    %v8932 = stablehlo.add %v8931, %v8930 : tensor<768x1x7x7xf32>
    %v8933 = stablehlo.divide %v8927, %v8932 : tensor<768x1x7x7xf32>
    %v8934 = stablehlo.multiply %v8929, %v8933 : tensor<768x1x7x7xf32>
    %v8935 = stablehlo.subtract %s3b0dW, %v8934 : tensor<768x1x7x7xf32>
    %v8936 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v8937 = stablehlo.multiply %v8936, %v8929 : tensor<768x1x7x7xf32>
    %v8938 = stablehlo.multiply %v8937, %s3b0dW : tensor<768x1x7x7xf32>
    %v8939 = stablehlo.subtract %v8935, %v8938 : tensor<768x1x7x7xf32>
    %v8940 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8941 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8942 = stablehlo.multiply %v8940, %s3b0dbm : tensor<768xf32>
    %v8943 = stablehlo.multiply %v8941, %v1436 : tensor<768xf32>
    %v8944 = stablehlo.add %v8942, %v8943 : tensor<768xf32>
    %v8945 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8946 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8947 = stablehlo.multiply %v8945, %s3b0dbv : tensor<768xf32>
    %v8948 = stablehlo.multiply %v1436, %v1436 : tensor<768xf32>
    %v8949 = stablehlo.multiply %v8946, %v8948 : tensor<768xf32>
    %v8950 = stablehlo.add %v8947, %v8949 : tensor<768xf32>
    %v8951 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8952 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8953 = stablehlo.multiply %v8951, %s3b0dbm : tensor<768xf32>
    %v8954 = stablehlo.multiply %v8952, %v1436 : tensor<768xf32>
    %v8955 = stablehlo.add %v8953, %v8954 : tensor<768xf32>
    %v8956 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8957 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8958 = stablehlo.multiply %v8956, %s3b0dbv : tensor<768xf32>
    %v8959 = stablehlo.multiply %v1436, %v1436 : tensor<768xf32>
    %v8960 = stablehlo.multiply %v8957, %v8959 : tensor<768xf32>
    %v8961 = stablehlo.add %v8958, %v8960 : tensor<768xf32>
    %v8962 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8963 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8964 = stablehlo.divide %v8955, %v8962 : tensor<768xf32>
    %v8965 = stablehlo.divide %v8961, %v8963 : tensor<768xf32>
    %v8966 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8967 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8968 = stablehlo.sqrt %v8965 : tensor<768xf32>
    %v8969 = stablehlo.add %v8968, %v8967 : tensor<768xf32>
    %v8970 = stablehlo.divide %v8964, %v8969 : tensor<768xf32>
    %v8971 = stablehlo.multiply %v8966, %v8970 : tensor<768xf32>
    %v8972 = stablehlo.subtract %s3b0db, %v8971 : tensor<768xf32>
    %v8973 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v8974 = stablehlo.multiply %v8973, %v8966 : tensor<768xf32>
    %v8975 = stablehlo.multiply %v8974, %s3b0db : tensor<768xf32>
    %v8976 = stablehlo.subtract %v8972, %v8975 : tensor<768xf32>
    %v8977 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8978 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8979 = stablehlo.multiply %v8977, %s3b0ngm : tensor<f32>
    %v8980 = stablehlo.multiply %v8978, %v1425 : tensor<f32>
    %v8981 = stablehlo.add %v8979, %v8980 : tensor<f32>
    %v8982 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8983 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8984 = stablehlo.multiply %v8982, %s3b0ngv : tensor<f32>
    %v8985 = stablehlo.multiply %v1425, %v1425 : tensor<f32>
    %v8986 = stablehlo.multiply %v8983, %v8985 : tensor<f32>
    %v8987 = stablehlo.add %v8984, %v8986 : tensor<f32>
    %v8988 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8989 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8990 = stablehlo.multiply %v8988, %s3b0ngm : tensor<f32>
    %v8991 = stablehlo.multiply %v8989, %v1425 : tensor<f32>
    %v8992 = stablehlo.add %v8990, %v8991 : tensor<f32>
    %v8993 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8994 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v8995 = stablehlo.multiply %v8993, %s3b0ngv : tensor<f32>
    %v8996 = stablehlo.multiply %v1425, %v1425 : tensor<f32>
    %v8997 = stablehlo.multiply %v8994, %v8996 : tensor<f32>
    %v8998 = stablehlo.add %v8995, %v8997 : tensor<f32>
    %v8999 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9000 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9001 = stablehlo.divide %v8992, %v8999 : tensor<f32>
    %v9002 = stablehlo.divide %v8998, %v9000 : tensor<f32>
    %v9003 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9004 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9005 = stablehlo.sqrt %v9002 : tensor<f32>
    %v9006 = stablehlo.add %v9005, %v9004 : tensor<f32>
    %v9007 = stablehlo.divide %v9001, %v9006 : tensor<f32>
    %v9008 = stablehlo.multiply %v9003, %v9007 : tensor<f32>
    %v9009 = stablehlo.subtract %s3b0ng, %v9008 : tensor<f32>
    %v9010 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9011 = stablehlo.multiply %v9010, %v9003 : tensor<f32>
    %v9012 = stablehlo.multiply %v9011, %s3b0ng : tensor<f32>
    %v9013 = stablehlo.subtract %v9009, %v9012 : tensor<f32>
    %v9014 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9015 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9016 = stablehlo.multiply %v9014, %s3b0nbtm : tensor<f32>
    %v9017 = stablehlo.multiply %v9015, %v1427 : tensor<f32>
    %v9018 = stablehlo.add %v9016, %v9017 : tensor<f32>
    %v9019 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9020 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9021 = stablehlo.multiply %v9019, %s3b0nbtv : tensor<f32>
    %v9022 = stablehlo.multiply %v1427, %v1427 : tensor<f32>
    %v9023 = stablehlo.multiply %v9020, %v9022 : tensor<f32>
    %v9024 = stablehlo.add %v9021, %v9023 : tensor<f32>
    %v9025 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9026 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9027 = stablehlo.multiply %v9025, %s3b0nbtm : tensor<f32>
    %v9028 = stablehlo.multiply %v9026, %v1427 : tensor<f32>
    %v9029 = stablehlo.add %v9027, %v9028 : tensor<f32>
    %v9030 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9031 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9032 = stablehlo.multiply %v9030, %s3b0nbtv : tensor<f32>
    %v9033 = stablehlo.multiply %v1427, %v1427 : tensor<f32>
    %v9034 = stablehlo.multiply %v9031, %v9033 : tensor<f32>
    %v9035 = stablehlo.add %v9032, %v9034 : tensor<f32>
    %v9036 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9037 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9038 = stablehlo.divide %v9029, %v9036 : tensor<f32>
    %v9039 = stablehlo.divide %v9035, %v9037 : tensor<f32>
    %v9040 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9041 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9042 = stablehlo.sqrt %v9039 : tensor<f32>
    %v9043 = stablehlo.add %v9042, %v9041 : tensor<f32>
    %v9044 = stablehlo.divide %v9038, %v9043 : tensor<f32>
    %v9045 = stablehlo.multiply %v9040, %v9044 : tensor<f32>
    %v9046 = stablehlo.subtract %s3b0nbt, %v9045 : tensor<f32>
    %v9047 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9048 = stablehlo.multiply %v9047, %v9040 : tensor<f32>
    %v9049 = stablehlo.multiply %v9048, %s3b0nbt : tensor<f32>
    %v9050 = stablehlo.subtract %v9046, %v9049 : tensor<f32>
    %v9051 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9052 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9053 = stablehlo.multiply %v9051, %s3b0eWm : tensor<3072x768x1x1xf32>
    %v9054 = stablehlo.multiply %v9052, %v1406 : tensor<3072x768x1x1xf32>
    %v9055 = stablehlo.add %v9053, %v9054 : tensor<3072x768x1x1xf32>
    %v9056 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9057 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9058 = stablehlo.multiply %v9056, %s3b0eWv : tensor<3072x768x1x1xf32>
    %v9059 = stablehlo.multiply %v1406, %v1406 : tensor<3072x768x1x1xf32>
    %v9060 = stablehlo.multiply %v9057, %v9059 : tensor<3072x768x1x1xf32>
    %v9061 = stablehlo.add %v9058, %v9060 : tensor<3072x768x1x1xf32>
    %v9062 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9063 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9064 = stablehlo.multiply %v9062, %s3b0eWm : tensor<3072x768x1x1xf32>
    %v9065 = stablehlo.multiply %v9063, %v1406 : tensor<3072x768x1x1xf32>
    %v9066 = stablehlo.add %v9064, %v9065 : tensor<3072x768x1x1xf32>
    %v9067 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9068 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9069 = stablehlo.multiply %v9067, %s3b0eWv : tensor<3072x768x1x1xf32>
    %v9070 = stablehlo.multiply %v1406, %v1406 : tensor<3072x768x1x1xf32>
    %v9071 = stablehlo.multiply %v9068, %v9070 : tensor<3072x768x1x1xf32>
    %v9072 = stablehlo.add %v9069, %v9071 : tensor<3072x768x1x1xf32>
    %v9073 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9074 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9075 = stablehlo.divide %v9066, %v9073 : tensor<3072x768x1x1xf32>
    %v9076 = stablehlo.divide %v9072, %v9074 : tensor<3072x768x1x1xf32>
    %v9077 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9078 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9079 = stablehlo.sqrt %v9076 : tensor<3072x768x1x1xf32>
    %v9080 = stablehlo.add %v9079, %v9078 : tensor<3072x768x1x1xf32>
    %v9081 = stablehlo.divide %v9075, %v9080 : tensor<3072x768x1x1xf32>
    %v9082 = stablehlo.multiply %v9077, %v9081 : tensor<3072x768x1x1xf32>
    %v9083 = stablehlo.subtract %s3b0eW, %v9082 : tensor<3072x768x1x1xf32>
    %v9084 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9085 = stablehlo.multiply %v9084, %v9077 : tensor<3072x768x1x1xf32>
    %v9086 = stablehlo.multiply %v9085, %s3b0eW : tensor<3072x768x1x1xf32>
    %v9087 = stablehlo.subtract %v9083, %v9086 : tensor<3072x768x1x1xf32>
    %v9088 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9089 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9090 = stablehlo.multiply %v9088, %s3b0ebm : tensor<3072xf32>
    %v9091 = stablehlo.multiply %v9089, %v1409 : tensor<3072xf32>
    %v9092 = stablehlo.add %v9090, %v9091 : tensor<3072xf32>
    %v9093 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9094 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9095 = stablehlo.multiply %v9093, %s3b0ebv : tensor<3072xf32>
    %v9096 = stablehlo.multiply %v1409, %v1409 : tensor<3072xf32>
    %v9097 = stablehlo.multiply %v9094, %v9096 : tensor<3072xf32>
    %v9098 = stablehlo.add %v9095, %v9097 : tensor<3072xf32>
    %v9099 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9100 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9101 = stablehlo.multiply %v9099, %s3b0ebm : tensor<3072xf32>
    %v9102 = stablehlo.multiply %v9100, %v1409 : tensor<3072xf32>
    %v9103 = stablehlo.add %v9101, %v9102 : tensor<3072xf32>
    %v9104 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9105 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9106 = stablehlo.multiply %v9104, %s3b0ebv : tensor<3072xf32>
    %v9107 = stablehlo.multiply %v1409, %v1409 : tensor<3072xf32>
    %v9108 = stablehlo.multiply %v9105, %v9107 : tensor<3072xf32>
    %v9109 = stablehlo.add %v9106, %v9108 : tensor<3072xf32>
    %v9110 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9111 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9112 = stablehlo.divide %v9103, %v9110 : tensor<3072xf32>
    %v9113 = stablehlo.divide %v9109, %v9111 : tensor<3072xf32>
    %v9114 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9115 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9116 = stablehlo.sqrt %v9113 : tensor<3072xf32>
    %v9117 = stablehlo.add %v9116, %v9115 : tensor<3072xf32>
    %v9118 = stablehlo.divide %v9112, %v9117 : tensor<3072xf32>
    %v9119 = stablehlo.multiply %v9114, %v9118 : tensor<3072xf32>
    %v9120 = stablehlo.subtract %s3b0eb, %v9119 : tensor<3072xf32>
    %v9121 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9122 = stablehlo.multiply %v9121, %v9114 : tensor<3072xf32>
    %v9123 = stablehlo.multiply %v9122, %s3b0eb : tensor<3072xf32>
    %v9124 = stablehlo.subtract %v9120, %v9123 : tensor<3072xf32>
    %v9125 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9126 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9127 = stablehlo.multiply %v9125, %s3b0pWm : tensor<768x3072x1x1xf32>
    %v9128 = stablehlo.multiply %v9126, %v1397 : tensor<768x3072x1x1xf32>
    %v9129 = stablehlo.add %v9127, %v9128 : tensor<768x3072x1x1xf32>
    %v9130 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9131 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9132 = stablehlo.multiply %v9130, %s3b0pWv : tensor<768x3072x1x1xf32>
    %v9133 = stablehlo.multiply %v1397, %v1397 : tensor<768x3072x1x1xf32>
    %v9134 = stablehlo.multiply %v9131, %v9133 : tensor<768x3072x1x1xf32>
    %v9135 = stablehlo.add %v9132, %v9134 : tensor<768x3072x1x1xf32>
    %v9136 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9137 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9138 = stablehlo.multiply %v9136, %s3b0pWm : tensor<768x3072x1x1xf32>
    %v9139 = stablehlo.multiply %v9137, %v1397 : tensor<768x3072x1x1xf32>
    %v9140 = stablehlo.add %v9138, %v9139 : tensor<768x3072x1x1xf32>
    %v9141 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9142 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9143 = stablehlo.multiply %v9141, %s3b0pWv : tensor<768x3072x1x1xf32>
    %v9144 = stablehlo.multiply %v1397, %v1397 : tensor<768x3072x1x1xf32>
    %v9145 = stablehlo.multiply %v9142, %v9144 : tensor<768x3072x1x1xf32>
    %v9146 = stablehlo.add %v9143, %v9145 : tensor<768x3072x1x1xf32>
    %v9147 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9148 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9149 = stablehlo.divide %v9140, %v9147 : tensor<768x3072x1x1xf32>
    %v9150 = stablehlo.divide %v9146, %v9148 : tensor<768x3072x1x1xf32>
    %v9151 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9152 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9153 = stablehlo.sqrt %v9150 : tensor<768x3072x1x1xf32>
    %v9154 = stablehlo.add %v9153, %v9152 : tensor<768x3072x1x1xf32>
    %v9155 = stablehlo.divide %v9149, %v9154 : tensor<768x3072x1x1xf32>
    %v9156 = stablehlo.multiply %v9151, %v9155 : tensor<768x3072x1x1xf32>
    %v9157 = stablehlo.subtract %s3b0pW, %v9156 : tensor<768x3072x1x1xf32>
    %v9158 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9159 = stablehlo.multiply %v9158, %v9151 : tensor<768x3072x1x1xf32>
    %v9160 = stablehlo.multiply %v9159, %s3b0pW : tensor<768x3072x1x1xf32>
    %v9161 = stablehlo.subtract %v9157, %v9160 : tensor<768x3072x1x1xf32>
    %v9162 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9163 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9164 = stablehlo.multiply %v9162, %s3b0pbm : tensor<768xf32>
    %v9165 = stablehlo.multiply %v9163, %v1400 : tensor<768xf32>
    %v9166 = stablehlo.add %v9164, %v9165 : tensor<768xf32>
    %v9167 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9168 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9169 = stablehlo.multiply %v9167, %s3b0pbv : tensor<768xf32>
    %v9170 = stablehlo.multiply %v1400, %v1400 : tensor<768xf32>
    %v9171 = stablehlo.multiply %v9168, %v9170 : tensor<768xf32>
    %v9172 = stablehlo.add %v9169, %v9171 : tensor<768xf32>
    %v9173 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9174 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9175 = stablehlo.multiply %v9173, %s3b0pbm : tensor<768xf32>
    %v9176 = stablehlo.multiply %v9174, %v1400 : tensor<768xf32>
    %v9177 = stablehlo.add %v9175, %v9176 : tensor<768xf32>
    %v9178 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9179 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9180 = stablehlo.multiply %v9178, %s3b0pbv : tensor<768xf32>
    %v9181 = stablehlo.multiply %v1400, %v1400 : tensor<768xf32>
    %v9182 = stablehlo.multiply %v9179, %v9181 : tensor<768xf32>
    %v9183 = stablehlo.add %v9180, %v9182 : tensor<768xf32>
    %v9184 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9185 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9186 = stablehlo.divide %v9177, %v9184 : tensor<768xf32>
    %v9187 = stablehlo.divide %v9183, %v9185 : tensor<768xf32>
    %v9188 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9189 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9190 = stablehlo.sqrt %v9187 : tensor<768xf32>
    %v9191 = stablehlo.add %v9190, %v9189 : tensor<768xf32>
    %v9192 = stablehlo.divide %v9186, %v9191 : tensor<768xf32>
    %v9193 = stablehlo.multiply %v9188, %v9192 : tensor<768xf32>
    %v9194 = stablehlo.subtract %s3b0pb, %v9193 : tensor<768xf32>
    %v9195 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9196 = stablehlo.multiply %v9195, %v9188 : tensor<768xf32>
    %v9197 = stablehlo.multiply %v9196, %s3b0pb : tensor<768xf32>
    %v9198 = stablehlo.subtract %v9194, %v9197 : tensor<768xf32>
    %v9199 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9200 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9201 = stablehlo.multiply %v9199, %s3b0lgm : tensor<768xf32>
    %v9202 = stablehlo.multiply %v9200, %v1391 : tensor<768xf32>
    %v9203 = stablehlo.add %v9201, %v9202 : tensor<768xf32>
    %v9204 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9205 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9206 = stablehlo.multiply %v9204, %s3b0lgv : tensor<768xf32>
    %v9207 = stablehlo.multiply %v1391, %v1391 : tensor<768xf32>
    %v9208 = stablehlo.multiply %v9205, %v9207 : tensor<768xf32>
    %v9209 = stablehlo.add %v9206, %v9208 : tensor<768xf32>
    %v9210 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9211 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9212 = stablehlo.multiply %v9210, %s3b0lgm : tensor<768xf32>
    %v9213 = stablehlo.multiply %v9211, %v1391 : tensor<768xf32>
    %v9214 = stablehlo.add %v9212, %v9213 : tensor<768xf32>
    %v9215 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9216 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9217 = stablehlo.multiply %v9215, %s3b0lgv : tensor<768xf32>
    %v9218 = stablehlo.multiply %v1391, %v1391 : tensor<768xf32>
    %v9219 = stablehlo.multiply %v9216, %v9218 : tensor<768xf32>
    %v9220 = stablehlo.add %v9217, %v9219 : tensor<768xf32>
    %v9221 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9222 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9223 = stablehlo.divide %v9214, %v9221 : tensor<768xf32>
    %v9224 = stablehlo.divide %v9220, %v9222 : tensor<768xf32>
    %v9225 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9226 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9227 = stablehlo.sqrt %v9224 : tensor<768xf32>
    %v9228 = stablehlo.add %v9227, %v9226 : tensor<768xf32>
    %v9229 = stablehlo.divide %v9223, %v9228 : tensor<768xf32>
    %v9230 = stablehlo.multiply %v9225, %v9229 : tensor<768xf32>
    %v9231 = stablehlo.subtract %s3b0lg, %v9230 : tensor<768xf32>
    %v9232 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9233 = stablehlo.multiply %v9232, %v9225 : tensor<768xf32>
    %v9234 = stablehlo.multiply %v9233, %s3b0lg : tensor<768xf32>
    %v9235 = stablehlo.subtract %v9231, %v9234 : tensor<768xf32>
    %v9236 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9237 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9238 = stablehlo.multiply %v9236, %s3b1dWm : tensor<768x1x7x7xf32>
    %v9239 = stablehlo.multiply %v9237, %v1314 : tensor<768x1x7x7xf32>
    %v9240 = stablehlo.add %v9238, %v9239 : tensor<768x1x7x7xf32>
    %v9241 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9242 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9243 = stablehlo.multiply %v9241, %s3b1dWv : tensor<768x1x7x7xf32>
    %v9244 = stablehlo.multiply %v1314, %v1314 : tensor<768x1x7x7xf32>
    %v9245 = stablehlo.multiply %v9242, %v9244 : tensor<768x1x7x7xf32>
    %v9246 = stablehlo.add %v9243, %v9245 : tensor<768x1x7x7xf32>
    %v9247 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9248 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9249 = stablehlo.multiply %v9247, %s3b1dWm : tensor<768x1x7x7xf32>
    %v9250 = stablehlo.multiply %v9248, %v1314 : tensor<768x1x7x7xf32>
    %v9251 = stablehlo.add %v9249, %v9250 : tensor<768x1x7x7xf32>
    %v9252 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9253 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9254 = stablehlo.multiply %v9252, %s3b1dWv : tensor<768x1x7x7xf32>
    %v9255 = stablehlo.multiply %v1314, %v1314 : tensor<768x1x7x7xf32>
    %v9256 = stablehlo.multiply %v9253, %v9255 : tensor<768x1x7x7xf32>
    %v9257 = stablehlo.add %v9254, %v9256 : tensor<768x1x7x7xf32>
    %v9258 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9259 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9260 = stablehlo.divide %v9251, %v9258 : tensor<768x1x7x7xf32>
    %v9261 = stablehlo.divide %v9257, %v9259 : tensor<768x1x7x7xf32>
    %v9262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9263 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9264 = stablehlo.sqrt %v9261 : tensor<768x1x7x7xf32>
    %v9265 = stablehlo.add %v9264, %v9263 : tensor<768x1x7x7xf32>
    %v9266 = stablehlo.divide %v9260, %v9265 : tensor<768x1x7x7xf32>
    %v9267 = stablehlo.multiply %v9262, %v9266 : tensor<768x1x7x7xf32>
    %v9268 = stablehlo.subtract %s3b1dW, %v9267 : tensor<768x1x7x7xf32>
    %v9269 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9270 = stablehlo.multiply %v9269, %v9262 : tensor<768x1x7x7xf32>
    %v9271 = stablehlo.multiply %v9270, %s3b1dW : tensor<768x1x7x7xf32>
    %v9272 = stablehlo.subtract %v9268, %v9271 : tensor<768x1x7x7xf32>
    %v9273 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9274 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9275 = stablehlo.multiply %v9273, %s3b1dbm : tensor<768xf32>
    %v9276 = stablehlo.multiply %v9274, %v1317 : tensor<768xf32>
    %v9277 = stablehlo.add %v9275, %v9276 : tensor<768xf32>
    %v9278 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9279 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9280 = stablehlo.multiply %v9278, %s3b1dbv : tensor<768xf32>
    %v9281 = stablehlo.multiply %v1317, %v1317 : tensor<768xf32>
    %v9282 = stablehlo.multiply %v9279, %v9281 : tensor<768xf32>
    %v9283 = stablehlo.add %v9280, %v9282 : tensor<768xf32>
    %v9284 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9285 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9286 = stablehlo.multiply %v9284, %s3b1dbm : tensor<768xf32>
    %v9287 = stablehlo.multiply %v9285, %v1317 : tensor<768xf32>
    %v9288 = stablehlo.add %v9286, %v9287 : tensor<768xf32>
    %v9289 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9290 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9291 = stablehlo.multiply %v9289, %s3b1dbv : tensor<768xf32>
    %v9292 = stablehlo.multiply %v1317, %v1317 : tensor<768xf32>
    %v9293 = stablehlo.multiply %v9290, %v9292 : tensor<768xf32>
    %v9294 = stablehlo.add %v9291, %v9293 : tensor<768xf32>
    %v9295 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9296 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9297 = stablehlo.divide %v9288, %v9295 : tensor<768xf32>
    %v9298 = stablehlo.divide %v9294, %v9296 : tensor<768xf32>
    %v9299 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9300 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9301 = stablehlo.sqrt %v9298 : tensor<768xf32>
    %v9302 = stablehlo.add %v9301, %v9300 : tensor<768xf32>
    %v9303 = stablehlo.divide %v9297, %v9302 : tensor<768xf32>
    %v9304 = stablehlo.multiply %v9299, %v9303 : tensor<768xf32>
    %v9305 = stablehlo.subtract %s3b1db, %v9304 : tensor<768xf32>
    %v9306 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9307 = stablehlo.multiply %v9306, %v9299 : tensor<768xf32>
    %v9308 = stablehlo.multiply %v9307, %s3b1db : tensor<768xf32>
    %v9309 = stablehlo.subtract %v9305, %v9308 : tensor<768xf32>
    %v9310 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9311 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9312 = stablehlo.multiply %v9310, %s3b1ngm : tensor<f32>
    %v9313 = stablehlo.multiply %v9311, %v1306 : tensor<f32>
    %v9314 = stablehlo.add %v9312, %v9313 : tensor<f32>
    %v9315 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9316 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9317 = stablehlo.multiply %v9315, %s3b1ngv : tensor<f32>
    %v9318 = stablehlo.multiply %v1306, %v1306 : tensor<f32>
    %v9319 = stablehlo.multiply %v9316, %v9318 : tensor<f32>
    %v9320 = stablehlo.add %v9317, %v9319 : tensor<f32>
    %v9321 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9322 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9323 = stablehlo.multiply %v9321, %s3b1ngm : tensor<f32>
    %v9324 = stablehlo.multiply %v9322, %v1306 : tensor<f32>
    %v9325 = stablehlo.add %v9323, %v9324 : tensor<f32>
    %v9326 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9327 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9328 = stablehlo.multiply %v9326, %s3b1ngv : tensor<f32>
    %v9329 = stablehlo.multiply %v1306, %v1306 : tensor<f32>
    %v9330 = stablehlo.multiply %v9327, %v9329 : tensor<f32>
    %v9331 = stablehlo.add %v9328, %v9330 : tensor<f32>
    %v9332 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9333 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9334 = stablehlo.divide %v9325, %v9332 : tensor<f32>
    %v9335 = stablehlo.divide %v9331, %v9333 : tensor<f32>
    %v9336 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9337 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9338 = stablehlo.sqrt %v9335 : tensor<f32>
    %v9339 = stablehlo.add %v9338, %v9337 : tensor<f32>
    %v9340 = stablehlo.divide %v9334, %v9339 : tensor<f32>
    %v9341 = stablehlo.multiply %v9336, %v9340 : tensor<f32>
    %v9342 = stablehlo.subtract %s3b1ng, %v9341 : tensor<f32>
    %v9343 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9344 = stablehlo.multiply %v9343, %v9336 : tensor<f32>
    %v9345 = stablehlo.multiply %v9344, %s3b1ng : tensor<f32>
    %v9346 = stablehlo.subtract %v9342, %v9345 : tensor<f32>
    %v9347 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9348 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9349 = stablehlo.multiply %v9347, %s3b1nbtm : tensor<f32>
    %v9350 = stablehlo.multiply %v9348, %v1308 : tensor<f32>
    %v9351 = stablehlo.add %v9349, %v9350 : tensor<f32>
    %v9352 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9353 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9354 = stablehlo.multiply %v9352, %s3b1nbtv : tensor<f32>
    %v9355 = stablehlo.multiply %v1308, %v1308 : tensor<f32>
    %v9356 = stablehlo.multiply %v9353, %v9355 : tensor<f32>
    %v9357 = stablehlo.add %v9354, %v9356 : tensor<f32>
    %v9358 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9359 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9360 = stablehlo.multiply %v9358, %s3b1nbtm : tensor<f32>
    %v9361 = stablehlo.multiply %v9359, %v1308 : tensor<f32>
    %v9362 = stablehlo.add %v9360, %v9361 : tensor<f32>
    %v9363 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9364 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9365 = stablehlo.multiply %v9363, %s3b1nbtv : tensor<f32>
    %v9366 = stablehlo.multiply %v1308, %v1308 : tensor<f32>
    %v9367 = stablehlo.multiply %v9364, %v9366 : tensor<f32>
    %v9368 = stablehlo.add %v9365, %v9367 : tensor<f32>
    %v9369 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9370 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9371 = stablehlo.divide %v9362, %v9369 : tensor<f32>
    %v9372 = stablehlo.divide %v9368, %v9370 : tensor<f32>
    %v9373 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9374 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9375 = stablehlo.sqrt %v9372 : tensor<f32>
    %v9376 = stablehlo.add %v9375, %v9374 : tensor<f32>
    %v9377 = stablehlo.divide %v9371, %v9376 : tensor<f32>
    %v9378 = stablehlo.multiply %v9373, %v9377 : tensor<f32>
    %v9379 = stablehlo.subtract %s3b1nbt, %v9378 : tensor<f32>
    %v9380 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9381 = stablehlo.multiply %v9380, %v9373 : tensor<f32>
    %v9382 = stablehlo.multiply %v9381, %s3b1nbt : tensor<f32>
    %v9383 = stablehlo.subtract %v9379, %v9382 : tensor<f32>
    %v9384 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9385 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9386 = stablehlo.multiply %v9384, %s3b1eWm : tensor<3072x768x1x1xf32>
    %v9387 = stablehlo.multiply %v9385, %v1287 : tensor<3072x768x1x1xf32>
    %v9388 = stablehlo.add %v9386, %v9387 : tensor<3072x768x1x1xf32>
    %v9389 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9390 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9391 = stablehlo.multiply %v9389, %s3b1eWv : tensor<3072x768x1x1xf32>
    %v9392 = stablehlo.multiply %v1287, %v1287 : tensor<3072x768x1x1xf32>
    %v9393 = stablehlo.multiply %v9390, %v9392 : tensor<3072x768x1x1xf32>
    %v9394 = stablehlo.add %v9391, %v9393 : tensor<3072x768x1x1xf32>
    %v9395 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9396 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9397 = stablehlo.multiply %v9395, %s3b1eWm : tensor<3072x768x1x1xf32>
    %v9398 = stablehlo.multiply %v9396, %v1287 : tensor<3072x768x1x1xf32>
    %v9399 = stablehlo.add %v9397, %v9398 : tensor<3072x768x1x1xf32>
    %v9400 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9401 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9402 = stablehlo.multiply %v9400, %s3b1eWv : tensor<3072x768x1x1xf32>
    %v9403 = stablehlo.multiply %v1287, %v1287 : tensor<3072x768x1x1xf32>
    %v9404 = stablehlo.multiply %v9401, %v9403 : tensor<3072x768x1x1xf32>
    %v9405 = stablehlo.add %v9402, %v9404 : tensor<3072x768x1x1xf32>
    %v9406 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9407 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9408 = stablehlo.divide %v9399, %v9406 : tensor<3072x768x1x1xf32>
    %v9409 = stablehlo.divide %v9405, %v9407 : tensor<3072x768x1x1xf32>
    %v9410 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9411 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9412 = stablehlo.sqrt %v9409 : tensor<3072x768x1x1xf32>
    %v9413 = stablehlo.add %v9412, %v9411 : tensor<3072x768x1x1xf32>
    %v9414 = stablehlo.divide %v9408, %v9413 : tensor<3072x768x1x1xf32>
    %v9415 = stablehlo.multiply %v9410, %v9414 : tensor<3072x768x1x1xf32>
    %v9416 = stablehlo.subtract %s3b1eW, %v9415 : tensor<3072x768x1x1xf32>
    %v9417 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9418 = stablehlo.multiply %v9417, %v9410 : tensor<3072x768x1x1xf32>
    %v9419 = stablehlo.multiply %v9418, %s3b1eW : tensor<3072x768x1x1xf32>
    %v9420 = stablehlo.subtract %v9416, %v9419 : tensor<3072x768x1x1xf32>
    %v9421 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9422 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9423 = stablehlo.multiply %v9421, %s3b1ebm : tensor<3072xf32>
    %v9424 = stablehlo.multiply %v9422, %v1290 : tensor<3072xf32>
    %v9425 = stablehlo.add %v9423, %v9424 : tensor<3072xf32>
    %v9426 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9427 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9428 = stablehlo.multiply %v9426, %s3b1ebv : tensor<3072xf32>
    %v9429 = stablehlo.multiply %v1290, %v1290 : tensor<3072xf32>
    %v9430 = stablehlo.multiply %v9427, %v9429 : tensor<3072xf32>
    %v9431 = stablehlo.add %v9428, %v9430 : tensor<3072xf32>
    %v9432 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9433 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9434 = stablehlo.multiply %v9432, %s3b1ebm : tensor<3072xf32>
    %v9435 = stablehlo.multiply %v9433, %v1290 : tensor<3072xf32>
    %v9436 = stablehlo.add %v9434, %v9435 : tensor<3072xf32>
    %v9437 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9438 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9439 = stablehlo.multiply %v9437, %s3b1ebv : tensor<3072xf32>
    %v9440 = stablehlo.multiply %v1290, %v1290 : tensor<3072xf32>
    %v9441 = stablehlo.multiply %v9438, %v9440 : tensor<3072xf32>
    %v9442 = stablehlo.add %v9439, %v9441 : tensor<3072xf32>
    %v9443 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9444 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9445 = stablehlo.divide %v9436, %v9443 : tensor<3072xf32>
    %v9446 = stablehlo.divide %v9442, %v9444 : tensor<3072xf32>
    %v9447 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9448 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9449 = stablehlo.sqrt %v9446 : tensor<3072xf32>
    %v9450 = stablehlo.add %v9449, %v9448 : tensor<3072xf32>
    %v9451 = stablehlo.divide %v9445, %v9450 : tensor<3072xf32>
    %v9452 = stablehlo.multiply %v9447, %v9451 : tensor<3072xf32>
    %v9453 = stablehlo.subtract %s3b1eb, %v9452 : tensor<3072xf32>
    %v9454 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9455 = stablehlo.multiply %v9454, %v9447 : tensor<3072xf32>
    %v9456 = stablehlo.multiply %v9455, %s3b1eb : tensor<3072xf32>
    %v9457 = stablehlo.subtract %v9453, %v9456 : tensor<3072xf32>
    %v9458 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9459 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9460 = stablehlo.multiply %v9458, %s3b1pWm : tensor<768x3072x1x1xf32>
    %v9461 = stablehlo.multiply %v9459, %v1278 : tensor<768x3072x1x1xf32>
    %v9462 = stablehlo.add %v9460, %v9461 : tensor<768x3072x1x1xf32>
    %v9463 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9464 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9465 = stablehlo.multiply %v9463, %s3b1pWv : tensor<768x3072x1x1xf32>
    %v9466 = stablehlo.multiply %v1278, %v1278 : tensor<768x3072x1x1xf32>
    %v9467 = stablehlo.multiply %v9464, %v9466 : tensor<768x3072x1x1xf32>
    %v9468 = stablehlo.add %v9465, %v9467 : tensor<768x3072x1x1xf32>
    %v9469 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9470 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9471 = stablehlo.multiply %v9469, %s3b1pWm : tensor<768x3072x1x1xf32>
    %v9472 = stablehlo.multiply %v9470, %v1278 : tensor<768x3072x1x1xf32>
    %v9473 = stablehlo.add %v9471, %v9472 : tensor<768x3072x1x1xf32>
    %v9474 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9475 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9476 = stablehlo.multiply %v9474, %s3b1pWv : tensor<768x3072x1x1xf32>
    %v9477 = stablehlo.multiply %v1278, %v1278 : tensor<768x3072x1x1xf32>
    %v9478 = stablehlo.multiply %v9475, %v9477 : tensor<768x3072x1x1xf32>
    %v9479 = stablehlo.add %v9476, %v9478 : tensor<768x3072x1x1xf32>
    %v9480 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9481 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9482 = stablehlo.divide %v9473, %v9480 : tensor<768x3072x1x1xf32>
    %v9483 = stablehlo.divide %v9479, %v9481 : tensor<768x3072x1x1xf32>
    %v9484 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9485 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9486 = stablehlo.sqrt %v9483 : tensor<768x3072x1x1xf32>
    %v9487 = stablehlo.add %v9486, %v9485 : tensor<768x3072x1x1xf32>
    %v9488 = stablehlo.divide %v9482, %v9487 : tensor<768x3072x1x1xf32>
    %v9489 = stablehlo.multiply %v9484, %v9488 : tensor<768x3072x1x1xf32>
    %v9490 = stablehlo.subtract %s3b1pW, %v9489 : tensor<768x3072x1x1xf32>
    %v9491 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9492 = stablehlo.multiply %v9491, %v9484 : tensor<768x3072x1x1xf32>
    %v9493 = stablehlo.multiply %v9492, %s3b1pW : tensor<768x3072x1x1xf32>
    %v9494 = stablehlo.subtract %v9490, %v9493 : tensor<768x3072x1x1xf32>
    %v9495 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9496 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9497 = stablehlo.multiply %v9495, %s3b1pbm : tensor<768xf32>
    %v9498 = stablehlo.multiply %v9496, %v1281 : tensor<768xf32>
    %v9499 = stablehlo.add %v9497, %v9498 : tensor<768xf32>
    %v9500 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9501 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9502 = stablehlo.multiply %v9500, %s3b1pbv : tensor<768xf32>
    %v9503 = stablehlo.multiply %v1281, %v1281 : tensor<768xf32>
    %v9504 = stablehlo.multiply %v9501, %v9503 : tensor<768xf32>
    %v9505 = stablehlo.add %v9502, %v9504 : tensor<768xf32>
    %v9506 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9507 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9508 = stablehlo.multiply %v9506, %s3b1pbm : tensor<768xf32>
    %v9509 = stablehlo.multiply %v9507, %v1281 : tensor<768xf32>
    %v9510 = stablehlo.add %v9508, %v9509 : tensor<768xf32>
    %v9511 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9512 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9513 = stablehlo.multiply %v9511, %s3b1pbv : tensor<768xf32>
    %v9514 = stablehlo.multiply %v1281, %v1281 : tensor<768xf32>
    %v9515 = stablehlo.multiply %v9512, %v9514 : tensor<768xf32>
    %v9516 = stablehlo.add %v9513, %v9515 : tensor<768xf32>
    %v9517 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9518 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9519 = stablehlo.divide %v9510, %v9517 : tensor<768xf32>
    %v9520 = stablehlo.divide %v9516, %v9518 : tensor<768xf32>
    %v9521 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9522 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9523 = stablehlo.sqrt %v9520 : tensor<768xf32>
    %v9524 = stablehlo.add %v9523, %v9522 : tensor<768xf32>
    %v9525 = stablehlo.divide %v9519, %v9524 : tensor<768xf32>
    %v9526 = stablehlo.multiply %v9521, %v9525 : tensor<768xf32>
    %v9527 = stablehlo.subtract %s3b1pb, %v9526 : tensor<768xf32>
    %v9528 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9529 = stablehlo.multiply %v9528, %v9521 : tensor<768xf32>
    %v9530 = stablehlo.multiply %v9529, %s3b1pb : tensor<768xf32>
    %v9531 = stablehlo.subtract %v9527, %v9530 : tensor<768xf32>
    %v9532 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9533 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9534 = stablehlo.multiply %v9532, %s3b1lgm : tensor<768xf32>
    %v9535 = stablehlo.multiply %v9533, %v1272 : tensor<768xf32>
    %v9536 = stablehlo.add %v9534, %v9535 : tensor<768xf32>
    %v9537 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9538 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9539 = stablehlo.multiply %v9537, %s3b1lgv : tensor<768xf32>
    %v9540 = stablehlo.multiply %v1272, %v1272 : tensor<768xf32>
    %v9541 = stablehlo.multiply %v9538, %v9540 : tensor<768xf32>
    %v9542 = stablehlo.add %v9539, %v9541 : tensor<768xf32>
    %v9543 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9544 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9545 = stablehlo.multiply %v9543, %s3b1lgm : tensor<768xf32>
    %v9546 = stablehlo.multiply %v9544, %v1272 : tensor<768xf32>
    %v9547 = stablehlo.add %v9545, %v9546 : tensor<768xf32>
    %v9548 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9549 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9550 = stablehlo.multiply %v9548, %s3b1lgv : tensor<768xf32>
    %v9551 = stablehlo.multiply %v1272, %v1272 : tensor<768xf32>
    %v9552 = stablehlo.multiply %v9549, %v9551 : tensor<768xf32>
    %v9553 = stablehlo.add %v9550, %v9552 : tensor<768xf32>
    %v9554 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9555 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9556 = stablehlo.divide %v9547, %v9554 : tensor<768xf32>
    %v9557 = stablehlo.divide %v9553, %v9555 : tensor<768xf32>
    %v9558 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9559 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9560 = stablehlo.sqrt %v9557 : tensor<768xf32>
    %v9561 = stablehlo.add %v9560, %v9559 : tensor<768xf32>
    %v9562 = stablehlo.divide %v9556, %v9561 : tensor<768xf32>
    %v9563 = stablehlo.multiply %v9558, %v9562 : tensor<768xf32>
    %v9564 = stablehlo.subtract %s3b1lg, %v9563 : tensor<768xf32>
    %v9565 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9566 = stablehlo.multiply %v9565, %v9558 : tensor<768xf32>
    %v9567 = stablehlo.multiply %v9566, %s3b1lg : tensor<768xf32>
    %v9568 = stablehlo.subtract %v9564, %v9567 : tensor<768xf32>
    %v9569 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9570 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9571 = stablehlo.multiply %v9569, %s3b2dWm : tensor<768x1x7x7xf32>
    %v9572 = stablehlo.multiply %v9570, %v1195 : tensor<768x1x7x7xf32>
    %v9573 = stablehlo.add %v9571, %v9572 : tensor<768x1x7x7xf32>
    %v9574 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9575 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9576 = stablehlo.multiply %v9574, %s3b2dWv : tensor<768x1x7x7xf32>
    %v9577 = stablehlo.multiply %v1195, %v1195 : tensor<768x1x7x7xf32>
    %v9578 = stablehlo.multiply %v9575, %v9577 : tensor<768x1x7x7xf32>
    %v9579 = stablehlo.add %v9576, %v9578 : tensor<768x1x7x7xf32>
    %v9580 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9581 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9582 = stablehlo.multiply %v9580, %s3b2dWm : tensor<768x1x7x7xf32>
    %v9583 = stablehlo.multiply %v9581, %v1195 : tensor<768x1x7x7xf32>
    %v9584 = stablehlo.add %v9582, %v9583 : tensor<768x1x7x7xf32>
    %v9585 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9586 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9587 = stablehlo.multiply %v9585, %s3b2dWv : tensor<768x1x7x7xf32>
    %v9588 = stablehlo.multiply %v1195, %v1195 : tensor<768x1x7x7xf32>
    %v9589 = stablehlo.multiply %v9586, %v9588 : tensor<768x1x7x7xf32>
    %v9590 = stablehlo.add %v9587, %v9589 : tensor<768x1x7x7xf32>
    %v9591 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9592 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9593 = stablehlo.divide %v9584, %v9591 : tensor<768x1x7x7xf32>
    %v9594 = stablehlo.divide %v9590, %v9592 : tensor<768x1x7x7xf32>
    %v9595 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9596 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9597 = stablehlo.sqrt %v9594 : tensor<768x1x7x7xf32>
    %v9598 = stablehlo.add %v9597, %v9596 : tensor<768x1x7x7xf32>
    %v9599 = stablehlo.divide %v9593, %v9598 : tensor<768x1x7x7xf32>
    %v9600 = stablehlo.multiply %v9595, %v9599 : tensor<768x1x7x7xf32>
    %v9601 = stablehlo.subtract %s3b2dW, %v9600 : tensor<768x1x7x7xf32>
    %v9602 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v9603 = stablehlo.multiply %v9602, %v9595 : tensor<768x1x7x7xf32>
    %v9604 = stablehlo.multiply %v9603, %s3b2dW : tensor<768x1x7x7xf32>
    %v9605 = stablehlo.subtract %v9601, %v9604 : tensor<768x1x7x7xf32>
    %v9606 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9607 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9608 = stablehlo.multiply %v9606, %s3b2dbm : tensor<768xf32>
    %v9609 = stablehlo.multiply %v9607, %v1198 : tensor<768xf32>
    %v9610 = stablehlo.add %v9608, %v9609 : tensor<768xf32>
    %v9611 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9612 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9613 = stablehlo.multiply %v9611, %s3b2dbv : tensor<768xf32>
    %v9614 = stablehlo.multiply %v1198, %v1198 : tensor<768xf32>
    %v9615 = stablehlo.multiply %v9612, %v9614 : tensor<768xf32>
    %v9616 = stablehlo.add %v9613, %v9615 : tensor<768xf32>
    %v9617 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9618 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9619 = stablehlo.multiply %v9617, %s3b2dbm : tensor<768xf32>
    %v9620 = stablehlo.multiply %v9618, %v1198 : tensor<768xf32>
    %v9621 = stablehlo.add %v9619, %v9620 : tensor<768xf32>
    %v9622 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9623 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9624 = stablehlo.multiply %v9622, %s3b2dbv : tensor<768xf32>
    %v9625 = stablehlo.multiply %v1198, %v1198 : tensor<768xf32>
    %v9626 = stablehlo.multiply %v9623, %v9625 : tensor<768xf32>
    %v9627 = stablehlo.add %v9624, %v9626 : tensor<768xf32>
    %v9628 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9629 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9630 = stablehlo.divide %v9621, %v9628 : tensor<768xf32>
    %v9631 = stablehlo.divide %v9627, %v9629 : tensor<768xf32>
    %v9632 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9633 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9634 = stablehlo.sqrt %v9631 : tensor<768xf32>
    %v9635 = stablehlo.add %v9634, %v9633 : tensor<768xf32>
    %v9636 = stablehlo.divide %v9630, %v9635 : tensor<768xf32>
    %v9637 = stablehlo.multiply %v9632, %v9636 : tensor<768xf32>
    %v9638 = stablehlo.subtract %s3b2db, %v9637 : tensor<768xf32>
    %v9639 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9640 = stablehlo.multiply %v9639, %v9632 : tensor<768xf32>
    %v9641 = stablehlo.multiply %v9640, %s3b2db : tensor<768xf32>
    %v9642 = stablehlo.subtract %v9638, %v9641 : tensor<768xf32>
    %v9643 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9644 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9645 = stablehlo.multiply %v9643, %s3b2ngm : tensor<f32>
    %v9646 = stablehlo.multiply %v9644, %v1187 : tensor<f32>
    %v9647 = stablehlo.add %v9645, %v9646 : tensor<f32>
    %v9648 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9649 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9650 = stablehlo.multiply %v9648, %s3b2ngv : tensor<f32>
    %v9651 = stablehlo.multiply %v1187, %v1187 : tensor<f32>
    %v9652 = stablehlo.multiply %v9649, %v9651 : tensor<f32>
    %v9653 = stablehlo.add %v9650, %v9652 : tensor<f32>
    %v9654 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9655 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9656 = stablehlo.multiply %v9654, %s3b2ngm : tensor<f32>
    %v9657 = stablehlo.multiply %v9655, %v1187 : tensor<f32>
    %v9658 = stablehlo.add %v9656, %v9657 : tensor<f32>
    %v9659 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9660 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9661 = stablehlo.multiply %v9659, %s3b2ngv : tensor<f32>
    %v9662 = stablehlo.multiply %v1187, %v1187 : tensor<f32>
    %v9663 = stablehlo.multiply %v9660, %v9662 : tensor<f32>
    %v9664 = stablehlo.add %v9661, %v9663 : tensor<f32>
    %v9665 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9666 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9667 = stablehlo.divide %v9658, %v9665 : tensor<f32>
    %v9668 = stablehlo.divide %v9664, %v9666 : tensor<f32>
    %v9669 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9670 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9671 = stablehlo.sqrt %v9668 : tensor<f32>
    %v9672 = stablehlo.add %v9671, %v9670 : tensor<f32>
    %v9673 = stablehlo.divide %v9667, %v9672 : tensor<f32>
    %v9674 = stablehlo.multiply %v9669, %v9673 : tensor<f32>
    %v9675 = stablehlo.subtract %s3b2ng, %v9674 : tensor<f32>
    %v9676 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9677 = stablehlo.multiply %v9676, %v9669 : tensor<f32>
    %v9678 = stablehlo.multiply %v9677, %s3b2ng : tensor<f32>
    %v9679 = stablehlo.subtract %v9675, %v9678 : tensor<f32>
    %v9680 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9681 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9682 = stablehlo.multiply %v9680, %s3b2nbtm : tensor<f32>
    %v9683 = stablehlo.multiply %v9681, %v1189 : tensor<f32>
    %v9684 = stablehlo.add %v9682, %v9683 : tensor<f32>
    %v9685 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9686 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9687 = stablehlo.multiply %v9685, %s3b2nbtv : tensor<f32>
    %v9688 = stablehlo.multiply %v1189, %v1189 : tensor<f32>
    %v9689 = stablehlo.multiply %v9686, %v9688 : tensor<f32>
    %v9690 = stablehlo.add %v9687, %v9689 : tensor<f32>
    %v9691 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9692 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9693 = stablehlo.multiply %v9691, %s3b2nbtm : tensor<f32>
    %v9694 = stablehlo.multiply %v9692, %v1189 : tensor<f32>
    %v9695 = stablehlo.add %v9693, %v9694 : tensor<f32>
    %v9696 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9697 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9698 = stablehlo.multiply %v9696, %s3b2nbtv : tensor<f32>
    %v9699 = stablehlo.multiply %v1189, %v1189 : tensor<f32>
    %v9700 = stablehlo.multiply %v9697, %v9699 : tensor<f32>
    %v9701 = stablehlo.add %v9698, %v9700 : tensor<f32>
    %v9702 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9703 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9704 = stablehlo.divide %v9695, %v9702 : tensor<f32>
    %v9705 = stablehlo.divide %v9701, %v9703 : tensor<f32>
    %v9706 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9707 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9708 = stablehlo.sqrt %v9705 : tensor<f32>
    %v9709 = stablehlo.add %v9708, %v9707 : tensor<f32>
    %v9710 = stablehlo.divide %v9704, %v9709 : tensor<f32>
    %v9711 = stablehlo.multiply %v9706, %v9710 : tensor<f32>
    %v9712 = stablehlo.subtract %s3b2nbt, %v9711 : tensor<f32>
    %v9713 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9714 = stablehlo.multiply %v9713, %v9706 : tensor<f32>
    %v9715 = stablehlo.multiply %v9714, %s3b2nbt : tensor<f32>
    %v9716 = stablehlo.subtract %v9712, %v9715 : tensor<f32>
    %v9717 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9718 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9719 = stablehlo.multiply %v9717, %s3b2eWm : tensor<3072x768x1x1xf32>
    %v9720 = stablehlo.multiply %v9718, %v1168 : tensor<3072x768x1x1xf32>
    %v9721 = stablehlo.add %v9719, %v9720 : tensor<3072x768x1x1xf32>
    %v9722 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9723 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9724 = stablehlo.multiply %v9722, %s3b2eWv : tensor<3072x768x1x1xf32>
    %v9725 = stablehlo.multiply %v1168, %v1168 : tensor<3072x768x1x1xf32>
    %v9726 = stablehlo.multiply %v9723, %v9725 : tensor<3072x768x1x1xf32>
    %v9727 = stablehlo.add %v9724, %v9726 : tensor<3072x768x1x1xf32>
    %v9728 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9729 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9730 = stablehlo.multiply %v9728, %s3b2eWm : tensor<3072x768x1x1xf32>
    %v9731 = stablehlo.multiply %v9729, %v1168 : tensor<3072x768x1x1xf32>
    %v9732 = stablehlo.add %v9730, %v9731 : tensor<3072x768x1x1xf32>
    %v9733 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9734 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9735 = stablehlo.multiply %v9733, %s3b2eWv : tensor<3072x768x1x1xf32>
    %v9736 = stablehlo.multiply %v1168, %v1168 : tensor<3072x768x1x1xf32>
    %v9737 = stablehlo.multiply %v9734, %v9736 : tensor<3072x768x1x1xf32>
    %v9738 = stablehlo.add %v9735, %v9737 : tensor<3072x768x1x1xf32>
    %v9739 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9740 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9741 = stablehlo.divide %v9732, %v9739 : tensor<3072x768x1x1xf32>
    %v9742 = stablehlo.divide %v9738, %v9740 : tensor<3072x768x1x1xf32>
    %v9743 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9744 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9745 = stablehlo.sqrt %v9742 : tensor<3072x768x1x1xf32>
    %v9746 = stablehlo.add %v9745, %v9744 : tensor<3072x768x1x1xf32>
    %v9747 = stablehlo.divide %v9741, %v9746 : tensor<3072x768x1x1xf32>
    %v9748 = stablehlo.multiply %v9743, %v9747 : tensor<3072x768x1x1xf32>
    %v9749 = stablehlo.subtract %s3b2eW, %v9748 : tensor<3072x768x1x1xf32>
    %v9750 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v9751 = stablehlo.multiply %v9750, %v9743 : tensor<3072x768x1x1xf32>
    %v9752 = stablehlo.multiply %v9751, %s3b2eW : tensor<3072x768x1x1xf32>
    %v9753 = stablehlo.subtract %v9749, %v9752 : tensor<3072x768x1x1xf32>
    %v9754 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9755 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9756 = stablehlo.multiply %v9754, %s3b2ebm : tensor<3072xf32>
    %v9757 = stablehlo.multiply %v9755, %v1171 : tensor<3072xf32>
    %v9758 = stablehlo.add %v9756, %v9757 : tensor<3072xf32>
    %v9759 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9760 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9761 = stablehlo.multiply %v9759, %s3b2ebv : tensor<3072xf32>
    %v9762 = stablehlo.multiply %v1171, %v1171 : tensor<3072xf32>
    %v9763 = stablehlo.multiply %v9760, %v9762 : tensor<3072xf32>
    %v9764 = stablehlo.add %v9761, %v9763 : tensor<3072xf32>
    %v9765 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9766 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9767 = stablehlo.multiply %v9765, %s3b2ebm : tensor<3072xf32>
    %v9768 = stablehlo.multiply %v9766, %v1171 : tensor<3072xf32>
    %v9769 = stablehlo.add %v9767, %v9768 : tensor<3072xf32>
    %v9770 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9771 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9772 = stablehlo.multiply %v9770, %s3b2ebv : tensor<3072xf32>
    %v9773 = stablehlo.multiply %v1171, %v1171 : tensor<3072xf32>
    %v9774 = stablehlo.multiply %v9771, %v9773 : tensor<3072xf32>
    %v9775 = stablehlo.add %v9772, %v9774 : tensor<3072xf32>
    %v9776 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9777 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9778 = stablehlo.divide %v9769, %v9776 : tensor<3072xf32>
    %v9779 = stablehlo.divide %v9775, %v9777 : tensor<3072xf32>
    %v9780 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9781 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9782 = stablehlo.sqrt %v9779 : tensor<3072xf32>
    %v9783 = stablehlo.add %v9782, %v9781 : tensor<3072xf32>
    %v9784 = stablehlo.divide %v9778, %v9783 : tensor<3072xf32>
    %v9785 = stablehlo.multiply %v9780, %v9784 : tensor<3072xf32>
    %v9786 = stablehlo.subtract %s3b2eb, %v9785 : tensor<3072xf32>
    %v9787 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v9788 = stablehlo.multiply %v9787, %v9780 : tensor<3072xf32>
    %v9789 = stablehlo.multiply %v9788, %s3b2eb : tensor<3072xf32>
    %v9790 = stablehlo.subtract %v9786, %v9789 : tensor<3072xf32>
    %v9791 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9792 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9793 = stablehlo.multiply %v9791, %s3b2pWm : tensor<768x3072x1x1xf32>
    %v9794 = stablehlo.multiply %v9792, %v1159 : tensor<768x3072x1x1xf32>
    %v9795 = stablehlo.add %v9793, %v9794 : tensor<768x3072x1x1xf32>
    %v9796 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9797 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9798 = stablehlo.multiply %v9796, %s3b2pWv : tensor<768x3072x1x1xf32>
    %v9799 = stablehlo.multiply %v1159, %v1159 : tensor<768x3072x1x1xf32>
    %v9800 = stablehlo.multiply %v9797, %v9799 : tensor<768x3072x1x1xf32>
    %v9801 = stablehlo.add %v9798, %v9800 : tensor<768x3072x1x1xf32>
    %v9802 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9803 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9804 = stablehlo.multiply %v9802, %s3b2pWm : tensor<768x3072x1x1xf32>
    %v9805 = stablehlo.multiply %v9803, %v1159 : tensor<768x3072x1x1xf32>
    %v9806 = stablehlo.add %v9804, %v9805 : tensor<768x3072x1x1xf32>
    %v9807 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9808 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9809 = stablehlo.multiply %v9807, %s3b2pWv : tensor<768x3072x1x1xf32>
    %v9810 = stablehlo.multiply %v1159, %v1159 : tensor<768x3072x1x1xf32>
    %v9811 = stablehlo.multiply %v9808, %v9810 : tensor<768x3072x1x1xf32>
    %v9812 = stablehlo.add %v9809, %v9811 : tensor<768x3072x1x1xf32>
    %v9813 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9814 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9815 = stablehlo.divide %v9806, %v9813 : tensor<768x3072x1x1xf32>
    %v9816 = stablehlo.divide %v9812, %v9814 : tensor<768x3072x1x1xf32>
    %v9817 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9818 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9819 = stablehlo.sqrt %v9816 : tensor<768x3072x1x1xf32>
    %v9820 = stablehlo.add %v9819, %v9818 : tensor<768x3072x1x1xf32>
    %v9821 = stablehlo.divide %v9815, %v9820 : tensor<768x3072x1x1xf32>
    %v9822 = stablehlo.multiply %v9817, %v9821 : tensor<768x3072x1x1xf32>
    %v9823 = stablehlo.subtract %s3b2pW, %v9822 : tensor<768x3072x1x1xf32>
    %v9824 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v9825 = stablehlo.multiply %v9824, %v9817 : tensor<768x3072x1x1xf32>
    %v9826 = stablehlo.multiply %v9825, %s3b2pW : tensor<768x3072x1x1xf32>
    %v9827 = stablehlo.subtract %v9823, %v9826 : tensor<768x3072x1x1xf32>
    %v9828 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9829 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9830 = stablehlo.multiply %v9828, %s3b2pbm : tensor<768xf32>
    %v9831 = stablehlo.multiply %v9829, %v1162 : tensor<768xf32>
    %v9832 = stablehlo.add %v9830, %v9831 : tensor<768xf32>
    %v9833 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9834 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9835 = stablehlo.multiply %v9833, %s3b2pbv : tensor<768xf32>
    %v9836 = stablehlo.multiply %v1162, %v1162 : tensor<768xf32>
    %v9837 = stablehlo.multiply %v9834, %v9836 : tensor<768xf32>
    %v9838 = stablehlo.add %v9835, %v9837 : tensor<768xf32>
    %v9839 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9840 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9841 = stablehlo.multiply %v9839, %s3b2pbm : tensor<768xf32>
    %v9842 = stablehlo.multiply %v9840, %v1162 : tensor<768xf32>
    %v9843 = stablehlo.add %v9841, %v9842 : tensor<768xf32>
    %v9844 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9845 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9846 = stablehlo.multiply %v9844, %s3b2pbv : tensor<768xf32>
    %v9847 = stablehlo.multiply %v1162, %v1162 : tensor<768xf32>
    %v9848 = stablehlo.multiply %v9845, %v9847 : tensor<768xf32>
    %v9849 = stablehlo.add %v9846, %v9848 : tensor<768xf32>
    %v9850 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9851 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9852 = stablehlo.divide %v9843, %v9850 : tensor<768xf32>
    %v9853 = stablehlo.divide %v9849, %v9851 : tensor<768xf32>
    %v9854 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9855 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9856 = stablehlo.sqrt %v9853 : tensor<768xf32>
    %v9857 = stablehlo.add %v9856, %v9855 : tensor<768xf32>
    %v9858 = stablehlo.divide %v9852, %v9857 : tensor<768xf32>
    %v9859 = stablehlo.multiply %v9854, %v9858 : tensor<768xf32>
    %v9860 = stablehlo.subtract %s3b2pb, %v9859 : tensor<768xf32>
    %v9861 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9862 = stablehlo.multiply %v9861, %v9854 : tensor<768xf32>
    %v9863 = stablehlo.multiply %v9862, %s3b2pb : tensor<768xf32>
    %v9864 = stablehlo.subtract %v9860, %v9863 : tensor<768xf32>
    %v9865 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9866 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9867 = stablehlo.multiply %v9865, %s3b2lgm : tensor<768xf32>
    %v9868 = stablehlo.multiply %v9866, %v1153 : tensor<768xf32>
    %v9869 = stablehlo.add %v9867, %v9868 : tensor<768xf32>
    %v9870 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9871 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9872 = stablehlo.multiply %v9870, %s3b2lgv : tensor<768xf32>
    %v9873 = stablehlo.multiply %v1153, %v1153 : tensor<768xf32>
    %v9874 = stablehlo.multiply %v9871, %v9873 : tensor<768xf32>
    %v9875 = stablehlo.add %v9872, %v9874 : tensor<768xf32>
    %v9876 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9877 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9878 = stablehlo.multiply %v9876, %s3b2lgm : tensor<768xf32>
    %v9879 = stablehlo.multiply %v9877, %v1153 : tensor<768xf32>
    %v9880 = stablehlo.add %v9878, %v9879 : tensor<768xf32>
    %v9881 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9882 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9883 = stablehlo.multiply %v9881, %s3b2lgv : tensor<768xf32>
    %v9884 = stablehlo.multiply %v1153, %v1153 : tensor<768xf32>
    %v9885 = stablehlo.multiply %v9882, %v9884 : tensor<768xf32>
    %v9886 = stablehlo.add %v9883, %v9885 : tensor<768xf32>
    %v9887 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9888 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9889 = stablehlo.divide %v9880, %v9887 : tensor<768xf32>
    %v9890 = stablehlo.divide %v9886, %v9888 : tensor<768xf32>
    %v9891 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9892 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9893 = stablehlo.sqrt %v9890 : tensor<768xf32>
    %v9894 = stablehlo.add %v9893, %v9892 : tensor<768xf32>
    %v9895 = stablehlo.divide %v9889, %v9894 : tensor<768xf32>
    %v9896 = stablehlo.multiply %v9891, %v9895 : tensor<768xf32>
    %v9897 = stablehlo.subtract %s3b2lg, %v9896 : tensor<768xf32>
    %v9898 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v9899 = stablehlo.multiply %v9898, %v9891 : tensor<768xf32>
    %v9900 = stablehlo.multiply %v9899, %s3b2lg : tensor<768xf32>
    %v9901 = stablehlo.subtract %v9897, %v9900 : tensor<768xf32>
    %v9902 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9903 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9904 = stablehlo.multiply %v9902, %hngm : tensor<f32>
    %v9905 = stablehlo.multiply %v9903, %v1077 : tensor<f32>
    %v9906 = stablehlo.add %v9904, %v9905 : tensor<f32>
    %v9907 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9908 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9909 = stablehlo.multiply %v9907, %hngv : tensor<f32>
    %v9910 = stablehlo.multiply %v1077, %v1077 : tensor<f32>
    %v9911 = stablehlo.multiply %v9908, %v9910 : tensor<f32>
    %v9912 = stablehlo.add %v9909, %v9911 : tensor<f32>
    %v9913 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9914 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9915 = stablehlo.multiply %v9913, %hngm : tensor<f32>
    %v9916 = stablehlo.multiply %v9914, %v1077 : tensor<f32>
    %v9917 = stablehlo.add %v9915, %v9916 : tensor<f32>
    %v9918 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9919 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9920 = stablehlo.multiply %v9918, %hngv : tensor<f32>
    %v9921 = stablehlo.multiply %v1077, %v1077 : tensor<f32>
    %v9922 = stablehlo.multiply %v9919, %v9921 : tensor<f32>
    %v9923 = stablehlo.add %v9920, %v9922 : tensor<f32>
    %v9924 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9925 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9926 = stablehlo.divide %v9917, %v9924 : tensor<f32>
    %v9927 = stablehlo.divide %v9923, %v9925 : tensor<f32>
    %v9928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9929 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9930 = stablehlo.sqrt %v9927 : tensor<f32>
    %v9931 = stablehlo.add %v9930, %v9929 : tensor<f32>
    %v9932 = stablehlo.divide %v9926, %v9931 : tensor<f32>
    %v9933 = stablehlo.multiply %v9928, %v9932 : tensor<f32>
    %v9934 = stablehlo.subtract %hng, %v9933 : tensor<f32>
    %v9935 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9936 = stablehlo.multiply %v9935, %v9928 : tensor<f32>
    %v9937 = stablehlo.multiply %v9936, %hng : tensor<f32>
    %v9938 = stablehlo.subtract %v9934, %v9937 : tensor<f32>
    %v9939 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9940 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9941 = stablehlo.multiply %v9939, %hnbtm : tensor<f32>
    %v9942 = stablehlo.multiply %v9940, %v1079 : tensor<f32>
    %v9943 = stablehlo.add %v9941, %v9942 : tensor<f32>
    %v9944 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9945 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9946 = stablehlo.multiply %v9944, %hnbtv : tensor<f32>
    %v9947 = stablehlo.multiply %v1079, %v1079 : tensor<f32>
    %v9948 = stablehlo.multiply %v9945, %v9947 : tensor<f32>
    %v9949 = stablehlo.add %v9946, %v9948 : tensor<f32>
    %v9950 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9951 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9952 = stablehlo.multiply %v9950, %hnbtm : tensor<f32>
    %v9953 = stablehlo.multiply %v9951, %v1079 : tensor<f32>
    %v9954 = stablehlo.add %v9952, %v9953 : tensor<f32>
    %v9955 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9956 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9957 = stablehlo.multiply %v9955, %hnbtv : tensor<f32>
    %v9958 = stablehlo.multiply %v1079, %v1079 : tensor<f32>
    %v9959 = stablehlo.multiply %v9956, %v9958 : tensor<f32>
    %v9960 = stablehlo.add %v9957, %v9959 : tensor<f32>
    %v9961 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9962 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9963 = stablehlo.divide %v9954, %v9961 : tensor<f32>
    %v9964 = stablehlo.divide %v9960, %v9962 : tensor<f32>
    %v9965 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9966 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9967 = stablehlo.sqrt %v9964 : tensor<f32>
    %v9968 = stablehlo.add %v9967, %v9966 : tensor<f32>
    %v9969 = stablehlo.divide %v9963, %v9968 : tensor<f32>
    %v9970 = stablehlo.multiply %v9965, %v9969 : tensor<f32>
    %v9971 = stablehlo.subtract %hnbt, %v9970 : tensor<f32>
    %v9972 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<f32>
    %v9973 = stablehlo.multiply %v9972, %v9965 : tensor<f32>
    %v9974 = stablehlo.multiply %v9973, %hnbt : tensor<f32>
    %v9975 = stablehlo.subtract %v9971, %v9974 : tensor<f32>
    %v9976 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v9977 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v9978 = stablehlo.multiply %v9976, %Wdm : tensor<768x10xf32>
    %v9979 = stablehlo.multiply %v9977, %v1059 : tensor<768x10xf32>
    %v9980 = stablehlo.add %v9978, %v9979 : tensor<768x10xf32>
    %v9981 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v9982 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v9983 = stablehlo.multiply %v9981, %Wdv : tensor<768x10xf32>
    %v9984 = stablehlo.multiply %v1059, %v1059 : tensor<768x10xf32>
    %v9985 = stablehlo.multiply %v9982, %v9984 : tensor<768x10xf32>
    %v9986 = stablehlo.add %v9983, %v9985 : tensor<768x10xf32>
    %v9987 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v9988 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v9989 = stablehlo.multiply %v9987, %Wdm : tensor<768x10xf32>
    %v9990 = stablehlo.multiply %v9988, %v1059 : tensor<768x10xf32>
    %v9991 = stablehlo.add %v9989, %v9990 : tensor<768x10xf32>
    %v9992 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v9993 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v9994 = stablehlo.multiply %v9992, %Wdv : tensor<768x10xf32>
    %v9995 = stablehlo.multiply %v1059, %v1059 : tensor<768x10xf32>
    %v9996 = stablehlo.multiply %v9993, %v9995 : tensor<768x10xf32>
    %v9997 = stablehlo.add %v9994, %v9996 : tensor<768x10xf32>
    %v9998 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v9999 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10000 = stablehlo.divide %v9991, %v9998 : tensor<768x10xf32>
    %v10001 = stablehlo.divide %v9997, %v9999 : tensor<768x10xf32>
    %v10002 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10003 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10004 = stablehlo.sqrt %v10001 : tensor<768x10xf32>
    %v10005 = stablehlo.add %v10004, %v10003 : tensor<768x10xf32>
    %v10006 = stablehlo.divide %v10000, %v10005 : tensor<768x10xf32>
    %v10007 = stablehlo.multiply %v10002, %v10006 : tensor<768x10xf32>
    %v10008 = stablehlo.subtract %Wd, %v10007 : tensor<768x10xf32>
    %v10009 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v10010 = stablehlo.multiply %v10009, %v10002 : tensor<768x10xf32>
    %v10011 = stablehlo.multiply %v10010, %Wd : tensor<768x10xf32>
    %v10012 = stablehlo.subtract %v10008, %v10011 : tensor<768x10xf32>
    %v10013 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10014 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10015 = stablehlo.multiply %v10013, %bdm : tensor<10xf32>
    %v10016 = stablehlo.multiply %v10014, %v1061 : tensor<10xf32>
    %v10017 = stablehlo.add %v10015, %v10016 : tensor<10xf32>
    %v10018 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10019 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10020 = stablehlo.multiply %v10018, %bdv : tensor<10xf32>
    %v10021 = stablehlo.multiply %v1061, %v1061 : tensor<10xf32>
    %v10022 = stablehlo.multiply %v10019, %v10021 : tensor<10xf32>
    %v10023 = stablehlo.add %v10020, %v10022 : tensor<10xf32>
    %v10024 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10025 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10026 = stablehlo.multiply %v10024, %bdm : tensor<10xf32>
    %v10027 = stablehlo.multiply %v10025, %v1061 : tensor<10xf32>
    %v10028 = stablehlo.add %v10026, %v10027 : tensor<10xf32>
    %v10029 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10030 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10031 = stablehlo.multiply %v10029, %bdv : tensor<10xf32>
    %v10032 = stablehlo.multiply %v1061, %v1061 : tensor<10xf32>
    %v10033 = stablehlo.multiply %v10030, %v10032 : tensor<10xf32>
    %v10034 = stablehlo.add %v10031, %v10033 : tensor<10xf32>
    %v10035 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10036 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10037 = stablehlo.divide %v10028, %v10035 : tensor<10xf32>
    %v10038 = stablehlo.divide %v10034, %v10036 : tensor<10xf32>
    %v10039 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10040 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10041 = stablehlo.sqrt %v10038 : tensor<10xf32>
    %v10042 = stablehlo.add %v10041, %v10040 : tensor<10xf32>
    %v10043 = stablehlo.divide %v10037, %v10042 : tensor<10xf32>
    %v10044 = stablehlo.multiply %v10039, %v10043 : tensor<10xf32>
    %v10045 = stablehlo.subtract %bd, %v10044 : tensor<10xf32>
    %v10046 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v10047 = stablehlo.multiply %v10046, %v10039 : tensor<10xf32>
    %v10048 = stablehlo.multiply %v10047, %bd : tensor<10xf32>
    %v10049 = stablehlo.subtract %v10045, %v10048 : tensor<10xf32>
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
    return %v3426, %v3463, %v3500, %v3537, %v3574, %v3611, %v3648, %v3685, %v3722, %v3759, %v3796, %v3833, %v3870, %v3907, %v3944, %v3981, %v4018, %v4055, %v4092, %v4129, %v4166, %v4203, %v4240, %v4277, %v4314, %v4351, %v4388, %v4425, %v4462, %v4499, %v4536, %v4573, %v4610, %v4647, %v4684, %v4721, %v4758, %v4795, %v4832, %v4869, %v4906, %v4943, %v4980, %v5017, %v5054, %v5091, %v5128, %v5165, %v5202, %v5239, %v5276, %v5313, %v5350, %v5387, %v5424, %v5461, %v5498, %v5535, %v5572, %v5609, %v5646, %v5683, %v5720, %v5757, %v5794, %v5831, %v5868, %v5905, %v5942, %v5979, %v6016, %v6053, %v6090, %v6127, %v6164, %v6201, %v6238, %v6275, %v6312, %v6349, %v6386, %v6423, %v6460, %v6497, %v6534, %v6571, %v6608, %v6645, %v6682, %v6719, %v6756, %v6793, %v6830, %v6867, %v6904, %v6941, %v6978, %v7015, %v7052, %v7089, %v7126, %v7163, %v7200, %v7237, %v7274, %v7311, %v7348, %v7385, %v7422, %v7459, %v7496, %v7533, %v7570, %v7607, %v7644, %v7681, %v7718, %v7755, %v7792, %v7829, %v7866, %v7903, %v7940, %v7977, %v8014, %v8051, %v8088, %v8125, %v8162, %v8199, %v8236, %v8273, %v8310, %v8347, %v8384, %v8421, %v8458, %v8495, %v8532, %v8569, %v8606, %v8643, %v8680, %v8717, %v8754, %v8791, %v8828, %v8865, %v8902, %v8939, %v8976, %v9013, %v9050, %v9087, %v9124, %v9161, %v9198, %v9235, %v9272, %v9309, %v9346, %v9383, %v9420, %v9457, %v9494, %v9531, %v9568, %v9605, %v9642, %v9679, %v9716, %v9753, %v9790, %v9827, %v9864, %v9901, %v9938, %v9975, %v10012, %v10049, %v3394, %v3431, %v3468, %v3505, %v3542, %v3579, %v3616, %v3653, %v3690, %v3727, %v3764, %v3801, %v3838, %v3875, %v3912, %v3949, %v3986, %v4023, %v4060, %v4097, %v4134, %v4171, %v4208, %v4245, %v4282, %v4319, %v4356, %v4393, %v4430, %v4467, %v4504, %v4541, %v4578, %v4615, %v4652, %v4689, %v4726, %v4763, %v4800, %v4837, %v4874, %v4911, %v4948, %v4985, %v5022, %v5059, %v5096, %v5133, %v5170, %v5207, %v5244, %v5281, %v5318, %v5355, %v5392, %v5429, %v5466, %v5503, %v5540, %v5577, %v5614, %v5651, %v5688, %v5725, %v5762, %v5799, %v5836, %v5873, %v5910, %v5947, %v5984, %v6021, %v6058, %v6095, %v6132, %v6169, %v6206, %v6243, %v6280, %v6317, %v6354, %v6391, %v6428, %v6465, %v6502, %v6539, %v6576, %v6613, %v6650, %v6687, %v6724, %v6761, %v6798, %v6835, %v6872, %v6909, %v6946, %v6983, %v7020, %v7057, %v7094, %v7131, %v7168, %v7205, %v7242, %v7279, %v7316, %v7353, %v7390, %v7427, %v7464, %v7501, %v7538, %v7575, %v7612, %v7649, %v7686, %v7723, %v7760, %v7797, %v7834, %v7871, %v7908, %v7945, %v7982, %v8019, %v8056, %v8093, %v8130, %v8167, %v8204, %v8241, %v8278, %v8315, %v8352, %v8389, %v8426, %v8463, %v8500, %v8537, %v8574, %v8611, %v8648, %v8685, %v8722, %v8759, %v8796, %v8833, %v8870, %v8907, %v8944, %v8981, %v9018, %v9055, %v9092, %v9129, %v9166, %v9203, %v9240, %v9277, %v9314, %v9351, %v9388, %v9425, %v9462, %v9499, %v9536, %v9573, %v9610, %v9647, %v9684, %v9721, %v9758, %v9795, %v9832, %v9869, %v9906, %v9943, %v9980, %v10017, %v3400, %v3437, %v3474, %v3511, %v3548, %v3585, %v3622, %v3659, %v3696, %v3733, %v3770, %v3807, %v3844, %v3881, %v3918, %v3955, %v3992, %v4029, %v4066, %v4103, %v4140, %v4177, %v4214, %v4251, %v4288, %v4325, %v4362, %v4399, %v4436, %v4473, %v4510, %v4547, %v4584, %v4621, %v4658, %v4695, %v4732, %v4769, %v4806, %v4843, %v4880, %v4917, %v4954, %v4991, %v5028, %v5065, %v5102, %v5139, %v5176, %v5213, %v5250, %v5287, %v5324, %v5361, %v5398, %v5435, %v5472, %v5509, %v5546, %v5583, %v5620, %v5657, %v5694, %v5731, %v5768, %v5805, %v5842, %v5879, %v5916, %v5953, %v5990, %v6027, %v6064, %v6101, %v6138, %v6175, %v6212, %v6249, %v6286, %v6323, %v6360, %v6397, %v6434, %v6471, %v6508, %v6545, %v6582, %v6619, %v6656, %v6693, %v6730, %v6767, %v6804, %v6841, %v6878, %v6915, %v6952, %v6989, %v7026, %v7063, %v7100, %v7137, %v7174, %v7211, %v7248, %v7285, %v7322, %v7359, %v7396, %v7433, %v7470, %v7507, %v7544, %v7581, %v7618, %v7655, %v7692, %v7729, %v7766, %v7803, %v7840, %v7877, %v7914, %v7951, %v7988, %v8025, %v8062, %v8099, %v8136, %v8173, %v8210, %v8247, %v8284, %v8321, %v8358, %v8395, %v8432, %v8469, %v8506, %v8543, %v8580, %v8617, %v8654, %v8691, %v8728, %v8765, %v8802, %v8839, %v8876, %v8913, %v8950, %v8987, %v9024, %v9061, %v9098, %v9135, %v9172, %v9209, %v9246, %v9283, %v9320, %v9357, %v9394, %v9431, %v9468, %v9505, %v9542, %v9579, %v9616, %v9653, %v9690, %v9727, %v9764, %v9801, %v9838, %v9875, %v9912, %v9949, %v9986, %v10023, %loss, %bc1, %bc2 : tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<f32>, tensor<f32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<f32>, tensor<f32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<f32>, tensor<f32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
